#!/bin/bash
# Stage the packed `multiagent` env (scripts/pack_env.sh) and the model weights
# onto the node's local disk, and point every cache there. Source this from a
# job script, then call `stage_env && stage_weights && import_check`. It sets
# LOCAL, PY and MODEL_PATH, exports the cache variables, and defines cleanup().
#
# Why: the env is ~94k small files; importing it from NFS on a cold compute
# node took hours (job 18358466). A sequential tarball read takes ~1-3 min.
#
# Knobs:
#   ENV_TAR        tarball to stage (default: /iliad2/u/ifdita/envs/multiagent-latest.tar.zst)
#   LOCAL_BASE     default /tmp, which Slurm makes private to the job and deletes
#                  when the job ends; /scr/$USER keeps the directory visible to
#                  other shells on the node (then cleanup() removes it on exit)
#   COPY_WEIGHTS   0 reads the safetensors from NFS instead of copying them (default 1)
#   PILOT_MODEL    HF model id whose snapshot is staged (default openai/gpt-oss-20b)

HF_NFS=/iliad/u/ifdita/hf_cache
ENV_TAR=${ENV_TAR:-$(readlink -f /iliad2/u/ifdita/envs/multiagent-latest.tar.zst)}
PILOT_MODEL=${PILOT_MODEL:-openai/gpt-oss-20b}

_t0() { _T0=$(date +%s); }
_timing() { echo "TIMING $1 $(( $(date +%s) - _T0 ))s"; }

stage_env() {
    local base=${LOCAL_BASE:-/tmp}
    mkdir -p "$base" 2>/dev/null && [ -w "$base" ] || base=/tmp
    LOCAL="$base/maa-${SLURM_JOB_ID:-$$}"
    # leftovers of finished jobs under the same base (only possible under /scr/$USER)
    local d j
    for d in "$base"/maa-*; do
        [ -d "$d" ] || continue
        j=${d##*-}
        [ "$j" = "${SLURM_JOB_ID:-$$}" ] && continue
        squeue -h -j "$j" > /dev/null 2>&1 && continue
        rm -rf "$d"
    done
    mkdir -p "$LOCAL"/{env,cache,tmp,hf} || return 1
    echo "STAGE local=$LOCAL  $(df -h "$LOCAL" | tail -1)"
    echo "STAGE tarball=$ENV_TAR ($(du -h "$ENV_TAR" 2>/dev/null | cut -f1))"
    [ -s "$ENV_TAR" ] || { echo "STAGE tarball missing; run scripts/pack_env.sh"; return 1; }

    local zstd
    zstd=$(command -v zstd || echo /iliad/u/ifdita/miniconda3/bin/zstd)
    _t0
    tar -I "$zstd" -xf "$ENV_TAR" -C "$LOCAL/env" || { echo "STAGE extract failed"; return 1; }
    _timing extract
    # entry points in bin/ carry the NFS python's shebang; jobs use $PY -m anyway
    local f
    for f in "$LOCAL/env"/bin/*; do
        [ -f "$f" ] && [ "$(head -c2 "$f" 2>/dev/null)" = "#!" ] || continue
        sed -i "1s|^#!/iliad/u/ifdita/miniconda3/envs/multiagent/bin/|#!$LOCAL/env/bin/|" "$f"
    done

    PY="$LOCAL/env/bin/python"
    export PATH="$LOCAL/env/bin:$PATH" PYTHONNOUSERSITE=1
    export XDG_CACHE_HOME="$LOCAL/cache" TRITON_CACHE_DIR="$LOCAL/cache/triton" \
        VLLM_CACHE_ROOT="$LOCAL/cache/vllm" TORCHINDUCTOR_CACHE_DIR="$LOCAL/cache/inductor" \
        CUDA_CACHE_PATH="$LOCAL/cache/nv" FLASHINFER_WORKSPACE_BASE="$LOCAL/cache" \
        TORCH_EXTENSIONS_DIR="$LOCAL/cache/torch_ext" TMPDIR="$LOCAL/tmp" \
        HF_HOME="$LOCAL/hf" HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
    # gpt-oss's harmony tokenizer downloads its encoding on first use; serve it from a seeded copy
    if [ -d "$HF_NFS/tiktoken" ]; then
        cp -r "$HF_NFS/tiktoken" "$LOCAL/tiktoken"
        export TIKTOKEN_ENCODINGS_BASE="$LOCAL/tiktoken" TIKTOKEN_CACHE_DIR="$LOCAL/tiktoken"
    fi
}

stage_weights() {
    local snap
    snap=$(ls -d "$HF_NFS/hub/models--${PILOT_MODEL//\//--}/snapshots"/*/ 2>/dev/null | head -1)
    [ -n "$snap" ] || { echo "STAGE no snapshot for $PILOT_MODEL under $HF_NFS"; return 1; }
    snap=${snap%/}
    if [ "${COPY_WEIGHTS:-1}" = 1 ]; then
        _t0
        cp -rL "$snap" "$LOCAL/model" || { echo "STAGE weights copy failed"; return 1; }
        MODEL_PATH="$LOCAL/model"
        _timing weights
    else
        MODEL_PATH="$snap"
    fi
    echo "STAGE model_path=$MODEL_PATH"
}

import_check() {
    _t0
    timeout 900 "$PY" -c "
import sys, torch, vllm
from transformers import AutoConfig
AutoConfig.from_pretrained('$MODEL_PATH')
print('import ok', sys.prefix, torch.__version__, vllm.__version__)" || { echo "STAGE import check failed"; return 1; }
    _timing import+config
    [ $(( $(date +%s) - _T0 )) -lt 120 ] || echo "WARN imports are slow; is $LOCAL on local disk? $(df -h "$LOCAL" | tail -1)"
}

cleanup() {
    [ -n "${LOCAL:-}" ] && rm -rf "$LOCAL"
}
