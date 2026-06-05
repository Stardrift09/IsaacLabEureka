#!/usr/bin/env bash
# BSC cluster helper — status, logs, staging checks, all over one SSH config.
# Reads connection + paths from IsaacLab/docker/cluster/.env.cluster.bsc so nothing
# is hardcoded. Usage:
#
#   scripts/bsc.sh jobs              # squeue for our user
#   scripts/bsc.sh log [JOBID]       # tail latest (or given) eureka-<id>.out
#   scripts/bsc.sh err [JOBID]       # tail latest (or given) eureka-<id>.err
#   scripts/bsc.sh watch [JOBID]     # live tail -f of the .out
#   scripts/bsc.sh fetch             # rsync all cluster logs -> logs/cluster/logs
#   scripts/bsc.sh cancel JOBID      # scancel
#   scripts/bsc.sh quota             # GPFS usage of key dirs
#   scripts/bsc.sh llm               # pending LLM requests (no response yet)
#   scripts/bsc.sh check             # verify staged deps (HF model, container, assets)
#   scripts/bsc.sh ssh               # interactive shell on the login node
#   scripts/bsc.sh run -- <args>     # cd repo && ssh-run any command on the cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ENV_FILE="$SCRIPT_DIR/../IsaacLab/docker/cluster/.env.cluster.bsc"
[ -f "$ENV_FILE" ] || { echo "env file not found: $ENV_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

LOGS_DIR="$CLUSTER_EUREKA_DIR/logs"
HF_MODEL_DIR="$CLUSTER_HF_CACHE_DIR/hub/models--prithivMLmods--DeepCaption-VLA-7B"
USER_NAME="${CLUSTER_LOGIN##*@}"; USER_NAME="${CLUSTER_LOGIN%@*}"

rsh() { ssh "$CLUSTER_LOGIN" "$@"; }

# Resolve a job id: explicit arg, else the newest eureka-*.out on the cluster.
latest_jobid() {
    rsh "ls -t $LOGS_DIR/eureka-*.out 2>/dev/null | head -1 | sed -E 's#.*/eureka-([0-9]+)\.out#\1#'"
}

cmd="${1:-help}"; shift || true

case "$cmd" in
    jobs)
        rsh "squeue -u \$USER -o '%.10i %.9P %.20j %.8T %.10M %.6D %R'"
        ;;
    log|err)
        ext=$([ "$cmd" = log ] && echo out || echo err)
        jid="${1:-$(latest_jobid)}"
        [ -n "$jid" ] || { echo "no job id and no logs found" >&2; exit 1; }
        echo "== $LOGS_DIR/eureka-$jid.$ext =="
        rsh "tail -n 120 $LOGS_DIR/eureka-$jid.$ext"
        ;;
    watch)
        jid="${1:-$(latest_jobid)}"
        [ -n "$jid" ] || { echo "no job id and no logs found" >&2; exit 1; }
        echo "== live tail eureka-$jid.out (Ctrl-C to stop) =="
        rsh "tail -f $LOGS_DIR/eureka-$jid.out"
        ;;
    fetch)
        # Mirror the entire cluster logs dir (slurm .out/.err, rl_runs checkpoints,
        # eureka feedback) down to logs/cluster/logs. Trailing slash on the source =
        # copy its contents into dest. This can be large (checkpoints) — that's intended.
        dest="$SCRIPT_DIR/../logs/cluster/logs"
        mkdir -p "$dest"
        rsync -aP "$CLUSTER_LOGIN:$LOGS_DIR/" "$dest/"
        echo "fetched all of $LOGS_DIR -> $dest"
        du -sh "$dest" 2>/dev/null
        ;;
    cancel)
        [ -n "${1:-}" ] || { echo "usage: bsc.sh cancel JOBID" >&2; exit 1; }
        rsh "scancel $1" && echo "cancelled $1"
        ;;
    quota)
        rsh "echo '== quota =='; (lfs quota -h /gpfs/scratch 2>/dev/null || quota -s 2>/dev/null || echo 'no quota tool'); \
             echo; echo '== key dir sizes =='; \
             du -sh $CLUSTER_EUREKA_DIR $CLUSTER_HF_CACHE_DIR $CLUSTER_SIF_PATH $CLUSTER_ISAAC_SIM_CACHE_DIR 2>/dev/null"
        ;;
    llm)
        rsh "cd $CLUSTER_EUREKA_DIR/llm_requests 2>/dev/null && \
             for r in *_request.json; do [ -e \"\$r\" ] || continue; \
               resp=\${r/_request.json/_response.json}; done2=\${r/_request.json/_request.done.json}; \
               if [ ! -e \"\$resp\" ] && [ ! -e \"\$done2\" ]; then echo \"PENDING \$r\"; fi; \
             done; echo 'done'"
        ;;
    check)
        echo "== container tar =="
        rsh "ls -lh $CLUSTER_SIF_PATH/isaac-lab-eureka.tar 2>/dev/null || echo 'MISSING container tar'"
        echo "== HF VLM model (offline) =="
        rsh "if [ -d $HF_MODEL_DIR ]; then du -sh $HF_MODEL_DIR; \
                 echo 'safetensors:'; find $HF_MODEL_DIR/snapshots -name '*.safetensors' 2>/dev/null | wc -l; \
                 find $HF_MODEL_DIR/snapshots -name '*.safetensors' 2>/dev/null | head; \
             else echo 'MISSING $HF_MODEL_DIR'; fi"
        echo "== latest run: staged offline_assets + libero =="
        rsh "ls -t $CLUSTER_EUREKA_DIR/runs 2>/dev/null | head -1 | while read ts; do \
               echo \"run \$ts\"; \
               ls -d $CLUSTER_EUREKA_DIR/runs/\$ts/offline_assets 2>/dev/null || echo '  no offline_assets'; \
               ls -d $CLUSTER_EUREKA_DIR/runs/\$ts/libero 2>/dev/null || echo '  no libero'; \
             done"
        ;;
    ssh)
        exec ssh "$CLUSTER_LOGIN"
        ;;
    run)
        [ "${1:-}" = "--" ] && shift
        rsh "$*"
        ;;
    *)
        sed -n '2,20p' "${BASH_SOURCE[0]}"
        ;;
esac
