#!/bin/bash
# Pack the `multiagent` conda env into one tarball on /iliad2 so that compute
# jobs can stage it to node-local disk (scripts/stage_env.sh).
#
# Why: the env is ~94k small files. Importing it from NFS on a cold compute node
# took hours (torch alone 11 min, job 18358466). One sequential read of a
# tarball takes a minute or two. The env relocates as a plain tarball: its
# shared libraries use $ORIGIN-relative RPATHs, and jobs run everything as
# `<local>/bin/python -m ...`, never through the shebangs in bin/.
#
#   scripts/pack_env.sh        # idempotent: re-run after any pip install in the env
#
# Output: /iliad2/u/ifdita/envs/multiagent-<date>-<hash>.tar.zst and the symlink
# multiagent-latest.tar.zst. The hash is of `pip list --format=freeze`.
set -euo pipefail

E=${ENV_PREFIX:-/iliad/u/ifdita/miniconda3/envs/multiagent}
OUT_DIR=${OUT_DIR:-/iliad2/u/ifdita/envs}
ZSTD=$(command -v zstd || echo /iliad/u/ifdita/miniconda3/bin/zstd)
mkdir -p "$OUT_DIR"

HASH=$("$E/bin/python" -m pip list --format=freeze 2>/dev/null | sha1sum | cut -c1-8)
OUT="$OUT_DIR/multiagent-$(date +%Y%m%d)-$HASH.tar.zst"
if [ -s "$OUT" ]; then
    echo "up to date: $OUT"
else
    echo "$(date)  packing $E -> $OUT"
    tar -C "$E" -I "$ZSTD -3 -T0" -cf "$OUT.partial" . && mv "$OUT.partial" "$OUT"
fi
ln -sfn "$(basename "$OUT")" "$OUT_DIR/multiagent-latest.tar.zst"
ls -lh "$OUT" "$OUT_DIR/multiagent-latest.tar.zst"
echo "$(date)  done"
