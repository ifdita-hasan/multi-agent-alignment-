#!/bin/bash
# Serve gpt-oss-20b with vLLM on the iliad partition and run the pilot against
# it, all inside one job. The env and the weights are staged to node-local disk
# first (scripts/stage_env.sh): importing the env from NFS takes hours.
#
#   mkdir -p logs/slurm
#   sbatch scripts/serve_vllm.sbatch                      # stage, serve, smoke test, dry run (logs/try), full run (logs/run1), exit
#   sbatch --gres=gpu:a40:1 --constraint=ampere scripts/serve_vllm.sbatch   # a specific card type (CLI flags override #SBATCH lines)
#   SERVE_ONLY=1 sbatch scripts/serve_vllm.sbatch         # stage + serve, stay up; drive the client yourself (see README)
#   sbatch scripts/stage_test.sbatch                      # CPU-only staging test (runs this script with STAGE_ONLY=1)
#
# Env knobs: PORT, PILOT_MODEL, MAX_MODEL_LEN, GPU_UTIL, MAX_SEQS, EXTRA_VLLM_ARGS,
# TRY (dry-run dir), OUT (full-run dir), KEEP_SERVER=1, READY_TIMEOUT (s), and
# stage_env.sh's ENV_TAR / LOCAL_BASE / COPY_WEIGHTS. Extra arguments go to the
# full run's run_pilot (for example --episodes 5).
#
#SBATCH --job-name=maa-vllm
#SBATCH --partition=iliad
#SBATCH --account=iliad
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G|141G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail   # deliberately no -e: a failed smoke test must not take the server down
cd /iliad2/u/ifdita/multi-agent-alignment- || exit 1
mkdir -p logs/slurm
export PILOT_MODEL=${PILOT_MODEL:-openai/gpt-oss-20b}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_HOME=/usr/local/cuda-12.8
JOB_T0=$(date +%s)
echo "$(date)  job ${SLURM_JOB_ID:-?} on $(hostname)"

# --- stage the env and the weights to local disk -----------------------------
source scripts/stage_env.sh
if ! { stage_env && stage_weights && import_check; }; then
    echo "staging failed"
    cleanup
    exit 1
fi
trap 'cleanup' EXIT
trap 'exit 143' TERM INT
echo "TIMING staging-total $(( $(date +%s) - JOB_T0 ))s"

if [ "${STAGE_ONLY:-0}" = 1 ]; then
    "$PY" -m pytest tests/ -q -p no:cacheprovider
    exit $?
fi

# --- serve ---------------------------------------------------------------------
nvidia-smi --query-gpu=name,memory.total --format=csv
PORT=${PORT:-$((10000 + SLURM_JOB_ID % 20000))}   # shared nodes: never assume 8000 is free
export PILOT_BASE_URL="http://localhost:$PORT/v1"
LOG="logs/slurm/vllm-$SLURM_JOB_ID.log"
t0=$(date +%s)
# gpt-oss returns its reasoning in the `reasoning` field without a reasoning
# parser; the tool-calling flags are the ones the vLLM gpt-oss recipe documents.
"$PY" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" --served-model-name "$PILOT_MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
    --max-num-seqs "${MAX_SEQS:-32}" \
    --enable-auto-tool-choice --tool-call-parser openai \
    ${EXTRA_VLLM_ARGS:-} > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null; cleanup' EXIT

READY_TIMEOUT=${READY_TIMEOUT:-1800}
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    if ! kill -0 $SERVER 2>/dev/null; then
        echo "vLLM exited before serving; last lines of $LOG:"; tail -20 "$LOG"; exit 1
    fi
    if [ $(( $(date +%s) - t0 )) -ge "$READY_TIMEOUT" ]; then
        echo "server not ready after ${READY_TIMEOUT}s; last lines of $LOG:"; tail -20 "$LOG"; exit 1
    fi
    sleep 10
done
echo "TIMING server-ready $(( $(date +%s) - t0 ))s"
URL="http://$(hostname):$PORT/v1"
echo "$URL" > "logs/slurm/server-$SLURM_JOB_ID.url"
echo "$(date)  READY  export PILOT_BASE_URL=$URL"
if [ "${SERVE_ONLY:-0}" = 1 ]; then
    wait $SERVER
    exit 0
fi

# --- pipeline: smoke test -> dry run -> gate -> full run --------------------------
stage() { local name=$1 t=$(date +%s); shift; "$@"; local rc=$?; echo "TIMING $name $(( $(date +%s) - t ))s rc=$rc"; return $rc; }
TRY=${TRY:-logs/try}
OUT=${OUT:-logs/run1}

if ! stage smoke "$PY" -m pilot.smoke_test; then
    echo "smoke test failed; the server stays up for debugging:"
    echo "  srun --jobid=$SLURM_JOB_ID --overlap --cpu-bind=none --pty bash   # then curl \$PILOT_BASE_URL/models"
    wait $SERVER
    exit 1
fi
stage dryrun "$PY" -m pilot.run_pilot --out "$TRY" --episodes 2
nfail=$(ls "$TRY"/*.failed 2>/dev/null | wc -l)
if [ "$nfail" -gt "${MAX_FAILED_DRY:-2}" ]; then
    echo "dry run has $nfail failed episodes (limit ${MAX_FAILED_DRY:-2}); not starting the full run"
    exit 1
fi
[ "$nfail" -gt 0 ] && echo "dry run had $nfail failed episode(s); re-running the job retries them"
stage full "$PY" -m pilot.run_pilot --out "$OUT" "$@"
rc=$?
echo "TIMING job-total $(( $(date +%s) - JOB_T0 ))s"
[ "${KEEP_SERVER:-0}" = 1 ] && wait $SERVER
exit $rc
