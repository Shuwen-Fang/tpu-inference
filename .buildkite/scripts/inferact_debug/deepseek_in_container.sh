#!/bin/bash
# Inside the CI image: confirm which ICI-resiliency setting libtpu picked up,
# then run mmlu.sh with the benchmark client patched to log every prompt and
# completion. The container is thrown away afterwards.
set -u
cd /workspace/tpu_inference || exit 1

echo "=== libtpu flags for variant $VARIANT (LIBTPU_INIT_ARGS='${LIBTPU_INIT_ARGS:-}')"
TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 python3 -c "import jax; jax.devices()" \
  > /tmp/libtpu_init.log 2>&1
grep -oE "(deepsea_ici_resilient|resilient_collective_emitter)[^' ]*" /tmp/libtpu_init.log \
  | sort | uniq -c
flag=$(grep -oE "deepsea_ici_resilient=[a-z]+" /tmp/libtpu_init.log | tail -1)
echo "RESULT ds.$VARIANT.libtpu_flag=${flag:-none}"

client=scripts/vllm/benchmarking/benchmark_serving.py
sed -i 's|^logger = logging.getLogger(__name__)$|&\nlogger.addHandler(logging.StreamHandler())|' "$client"
sed -i 's|--run-eval|--run-eval --debug|' tests/e2e/benchmarking/mmlu.sh

bash tests/e2e/benchmarking/mmlu.sh -m "gs://tpu-commons-ci/deepseek/r1" -n "$NUM_PROMPTS" -l 4 \
  > /tmp/mmlu.log 2>&1
code=$?
grep -vE '^(Prompt|Output): ' /tmp/mmlu.log | tail -60
grep -E "'accuracy'|Total token throughput|Request throughput" /tmp/mmlu.log | tail -3

echo "=== first 40 completions (prompt tail | output)"
python3 - /tmp/mmlu.log <<'EOF'
import re, sys
text = open(sys.argv[1], errors="replace").read()
pairs = re.findall(r"Prompt: (.*?)\nOutput: (.*?)(?=\n(?:Prompt: |\S+ \S+ |\[|INFO|WARNING|$))", text, re.S)
for prompt, out in pairs[:40]:
    print(f"...{prompt[-80:]!r} | {out!r}")
print(f"completions logged: {len(pairs)}")
EOF

acc=$(grep -oP "'accuracy': \K[0-9.]+" /tmp/mmlu.log | tail -1)
unp=$(grep -oP "'unparsed_rate': \K[0-9.]+" /tmp/mmlu.log | tail -1)
tput=$(awk '/Total token throughput \(tok\/s\):/ {print $NF}' /tmp/mmlu.log | tail -1)
echo "RESULT ds.$VARIANT.accuracy=${acc:-none}"
echo "RESULT ds.$VARIANT.unparsed_rate=${unp:-none}"
echo "RESULT ds.$VARIANT.total_tok_per_s=${tput:-none}"
cp /tmp/mmlu.log "/out/mmlu_$VARIANT.log"
exit $code
