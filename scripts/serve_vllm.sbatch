#!/bin/bash
# Template: start vLLM with gpt-oss-20b, check the endpoint, run the pilot.
# Submit from the repo root:  sbatch scripts/serve_vllm.sbatch
#
# Edit before use: the partition (and account, if your cluster needs one), and
# the environment setup below. This script has not been run on a cluster yet.

#SBATCH --job-name=maa-pilot
#SBATCH --partition=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# --- environment setup: adapt to your cluster ---------------------------------
# module load python/3.12 cuda
# source ~/venvs/maa/bin/activate      # needs: pip install vllm -r requirements.txt
# ------------------------------------------------------------------------------

PORT=${PORT:-8000}
OUT=${OUT:-logs/run_$SLURM_JOB_ID}
export PILOT_BASE_URL="http://localhost:$PORT/v1"
export PILOT_MODEL="openai/gpt-oss-20b"

# The tool-calling flags are the ones vLLM documents for gpt-oss; the smoke test
# below is what confirms they work with the installed vLLM version.
vllm serve "$PILOT_MODEL" --port "$PORT" \
    --enable-auto-tool-choice --tool-call-parser openai &
SERVER=$!
trap 'kill $SERVER' EXIT

until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    kill -0 $SERVER  # stop waiting if the server died
    sleep 10
done

python -m pilot.smoke_test
python -m pilot.run_pilot --out "$OUT" "$@"
