#!/bin/bash
# CPU-only check that staging the env to local disk works and imports take seconds.
#
#   sbatch scripts/stage_test.sbatch                 # any iliad node
#   sbatch -w iliad5 scripts/stage_test.sbatch       # a specific node
#   grep -E 'STAGE|TIMING|WARN|import ok|passed|failed' logs/slurm/maa-stage-<jobid>.out
#
#SBATCH --job-name=maa-stage
#SBATCH --partition=iliad
#SBATCH --account=iliad
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out

cd /iliad2/u/ifdita/multi-agent-alignment- || exit 1
export STAGE_ONLY=1
bash scripts/serve_vllm.sbatch
rc=$?
echo "stage test exit code $rc"
exit $rc
