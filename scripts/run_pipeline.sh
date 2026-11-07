#!/bin/bash
# Run the pilot end to end against a running serve job, detached from any
# terminal, so the session that launched it can be closed:
#
#   setsid nohup scripts/run_pipeline.sh <jobid> > logs/pipeline-<jobid>.out 2>&1 < /dev/null &
#
# Steps: wait for the server URL file, smoke test, 2-episode dry run (logs/try),
# then the full run (logs/run1) only if the dry run had no failed episodes.
# Both runs are resumable: re-run the same command and finished episodes are skipped.
set -uo pipefail

JOBID=${1:?usage: run_pipeline.sh <slurm job id> [dry-run dir] [full-run dir]}
TRY=${2:-logs/try}
RUN=${3:-logs/run1}

cd "$(dirname "$0")/.."
source /iliad/u/ifdita/miniconda3/etc/profile.d/conda.sh
conda activate multiagent

URLFILE="logs/slurm/server-$JOBID.url"
echo "$(date)  waiting for $URLFILE"
until [ -s "$URLFILE" ]; do
    if ! squeue -h -j "$JOBID" 2>/dev/null | grep -q .; then
        echo "$(date)  job $JOBID is no longer in the queue; see logs/slurm/maa-vllm-$JOBID.out"
        exit 1
    fi
    sleep 20
done
export PILOT_BASE_URL
PILOT_BASE_URL=$(cat "$URLFILE")
echo "$(date)  server at $PILOT_BASE_URL"

echo "$(date)  smoke test"
if ! python -m pilot.smoke_test; then
    echo "$(date)  smoke test FAILED; stopping before any episode"
    exit 1
fi

echo "$(date)  dry run -> $TRY"
python -m pilot.run_pilot --out "$TRY" --episodes 2
if ls "$TRY"/*.failed > /dev/null 2>&1; then
    echo "$(date)  dry run has failed episodes; not starting the full run"
    exit 1
fi

echo "$(date)  full run -> $RUN"
python -m pilot.run_pilot --out "$RUN"
status=$?
echo "$(date)  finished with exit code $status (0 = every episode completed)"
exit $status
