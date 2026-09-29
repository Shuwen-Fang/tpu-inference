#!/bin/bash
# Runs the host-parity phases listed in PHASES and prints every RESULT line at
# the end, so the inferact and cicd jobs can be diffed line by line. A failing
# phase does not stop the later ones.
set -uo pipefail

IMAGE="${DEBUG_IMAGE:?DEBUG_IMAGE must be set}"
PHASES="${PHASES:-facts,numerics,kernels}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT="$PWD/inferact_debug_out/$(hostname)"
mkdir -p "$OUT"
failed=""

# Same docker flags as run_in_docker.sh, minus the caches.
in_image() {
  docker run --rm --privileged --net host --shm-size=16G \
    -v "$HERE:/debug:ro" -v "$OUT:/out" \
    -e TPU_VERSION=tpu7x -e JAX_COMPILATION_CACHE_DIR= \
    "$IMAGE" "$@"
}

run_phase() {
  local name=$1
  shift
  echo "--- :mag: $name"
  "$@" 2>&1 | tee "$OUT/$name.log"
  local rc=${PIPESTATUS[0]}
  echo "RESULT phase.$name.exit=$rc" | tee -a "$OUT/$name.log"
  [ "$rc" -eq 0 ] || failed="$failed $name"
}

echo "--- :docker: pull $IMAGE"
docker rm -f vllm-tpu >/dev/null 2>&1 || true
docker pull -q "$IMAGE" || exit 1

for phase in ${PHASES//,/ }; do
  case $phase in
    facts)
      run_phase host_facts bash "$HERE/host_facts.sh"
      run_phase dmesg in_image bash -c \
        "dmesg -T | grep -iE 'tpu|vfio|iommu|accel|dma|error|fail' | tail -80"
      run_phase device_facts in_image python3 /debug/device_facts.py
      ;;
    numerics)
      run_phase numerics in_image python3 /debug/collectives_check.py
      ;;
    kernels)
      run_phase kernels in_image bash /debug/kernel_tests.sh
      ;;
    *)
      echo "unknown phase: $phase"
      failed="$failed $phase"
      ;;
  esac
done

echo "+++ :clipboard: RESULT lines for $(hostname)"
cat "$OUT"/*.log | grep -h '^RESULT ' | tee "$OUT/results.txt"
if [ -n "$failed" ]; then
  echo "Failed phases:$failed"
  exit 1
fi
