#!/bin/sh
# Railway entrypoint for MinerU's CPU pipeline API.
#
# The one thing this has to get right is thread count. MinerU's pipeline backend
# is torch end to end (layout, OCR, table structure, formula recognition), and
# torch sizes its intra-op thread pool from the number of cores it can SEE --
# which inside a container is the HOST's core count, not the cgroup's quota. On
# Railway that means a pool several times wider than the CPU the container is
# actually allowed to use, and the oversubscription costs far more than the
# parallelism gains. The same trap is recorded in this repo for torch (moss-tts,
# ~5x) and for llama.cpp (ollama, ~78x).
#
# torch reads OMP_NUM_THREADS when its ATen thread pool is first initialised, so
# it has to be in the environment BEFORE python imports torch -- i.e. here, not
# in the launcher module.
set -e

MOUNT="${MINERU_DATA_DIR:-/data}"

# --- CPU quota -> thread count ---------------------------------------------
cpus=""
if [ -r /sys/fs/cgroup/cpu.max ]; then
    # cgroup v2: "<quota> <period>", or "max <period>" when unlimited.
    read -r quota period < /sys/fs/cgroup/cpu.max || true
    case "$quota" in
        ''|max|*[!0-9]*) : ;;
        *) [ -n "$period" ] && cpus=$(( (quota + period - 1) / period )) ;;
    esac
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
    quota="$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
    period="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
    case "$quota" in
        ''|-1|*[!0-9]*) : ;;
        *) [ "$period" -gt 0 ] && cpus=$(( (quota + period - 1) / period )) ;;
    esac
fi
if [ -z "$cpus" ]; then
    cpus="$(nproc 2>/dev/null || echo 1)"
fi
[ "$cpus" -lt 1 ] && cpus=1

# A deployer who sets OMP_NUM_THREADS explicitly keeps it.
if [ -z "$OMP_NUM_THREADS" ]; then
    OMP_NUM_THREADS="$cpus"
fi
export OMP_NUM_THREADS
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$OMP_NUM_THREADS}"

# There is no GPU on a Railway container. Pin it rather than letting MinerU's
# auto-detection decide, so a torch build that thinks it sees CUDA cannot pick a
# device that does not exist.
export MINERU_DEVICE_MODE="${MINERU_DEVICE_MODE:-cpu}"

# --- Volume -----------------------------------------------------------------
#
# Async task results and their output trees live under the mount. The image runs
# as root, so Railway's uid-0 volume needs no ownership repair -- but the repo's
# own omniroute finding was that an unwritable mount can look perfectly healthy,
# so the owner is printed and a write is proven rather than assumed.
if [ -d "$MOUNT" ]; then
    rm -rf "$MOUNT/lost+found" 2>/dev/null || true
    mkdir -p "$MOUNT/output"
    if : > "$MOUNT/.write-test" 2>/dev/null; then
        rm -f "$MOUNT/.write-test"
        writable=yes
    else
        writable=NO
    fi
    echo "railway-entrypoint: mount=$MOUNT owner=$(stat -c %u:%g "$MOUNT" 2>/dev/null || echo none) writable=$writable"
fi
export MINERU_API_OUTPUT_ROOT="${MINERU_API_OUTPUT_ROOT:-$MOUNT/output}"

echo "railway-entrypoint: cpu_quota=$cpus OMP_NUM_THREADS=$OMP_NUM_THREADS device=$MINERU_DEVICE_MODE port=${PORT:-8000} model_source=${MINERU_MODEL_SOURCE:-unset}"

exec python /usr/local/bin/railway_mineru_api.py \
    --host 0.0.0.0 --port "${PORT:-8000}" "$@"
