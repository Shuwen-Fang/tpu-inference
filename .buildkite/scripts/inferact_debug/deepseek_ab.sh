#!/bin/bash
# Runs the CI DeepSeek-R1 MMLU check (mmlu.sh, same env as the tpu7x Accuracy
# step) once per variant in DS_VARIANTS, with fewer prompts, and prints the
# accuracy, the unparsed rate and the raw completions of the first prompts.
# A variant only changes the libtpu ICI-resiliency setting:
#   baseline      what the host's tpu-env gives
#   noresilient   ICI resiliency forced off
#   resilient     ICI resiliency forced on
set -u

IMAGE="${DEBUG_IMAGE:?}"
OUT="${OUT:?}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NUM_PROMPTS="${DS_NUM_PROMPTS:-500}"
JAX_CACHE="/mnt/disks/persist/tpu_jax_cache/jax0.11.0_tpu7x"
mkdir -p "$JAX_CACHE" /mnt/disks/persist/models
# setup_docker_env.sh keeps HF_TOKEN in /etc/environment.
[ -n "${HF_TOKEN:-}" ] || { set -a; . /etc/environment; set +a; }
rc=0

variant_env() {
  case $1 in
    baseline) ;;
    noresilient) echo "-e LIBTPU_INIT_ARGS=--deepsea_ici_resilient=false -e ENABLE_ICI_RESILIENCY=false" ;;
    resilient) echo "-e LIBTPU_INIT_ARGS=--deepsea_ici_resilient=true -e ENABLE_ICI_RESILIENCY=true" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}

for v in ${DS_VARIANTS:-baseline}; do
  echo "--- :whale: DeepSeek-R1 MMLU, variant $v"
  extra=$(variant_env "$v") || { rc=1; continue; }
  docker rm -f vllm-tpu >/dev/null 2>&1
  # shellcheck disable=SC2086
  docker run --name vllm-tpu --rm --privileged --net host --shm-size=16G \
    -v /mnt/disks/persist/models:/tmp/hf_home -v "$JAX_CACHE:$JAX_CACHE" \
    -v "$HERE:/debug:ro" -v "$OUT:/out" \
    -e HF_HOME=/tmp/hf_home -e HF_TOKEN="${HF_TOKEN:-}" \
    -e VLLM_XLA_CACHE_PATH="$JAX_CACHE" -e JAX_COMPILATION_CACHE_DIR="$JAX_CACHE" \
    -e TPU_VERSION=tpu7x -e NEW_MODEL_DESIGN=1 -e USE_V6E8_QUEUE=False \
    -e SKIP_ACCURACY_TESTS=False -e VLLM_MLA_DISABLE=0 \
    -e MOE_REQUANTIZE_BLOCK_SIZE=512 -e MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 \
    -e MODEL_IMPL_TYPE=vllm -e MINIMUM_ACCURACY_THRESHOLD=0.84 \
    -e VARIANT="$v" -e NUM_PROMPTS="$NUM_PROMPTS" $extra \
    "$IMAGE" bash /debug/deepseek_in_container.sh 2>&1 | tee "$OUT/deepseek_$v.log"
  code=${PIPESTATUS[0]}
  echo "RESULT ds.$v.exit=$code"
  [ "$code" -eq 0 ] || rc=1
done
exit $rc
