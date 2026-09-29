#!/bin/bash
# Kernel and layer tests for what DeepSeek-R1 uses and CI never runs on a
# tpu7x-8 host: MLA, the MoE kernels, gather-reduce, quantized matmul and MoE
# requantization. Runs inside the CI image against its own tests.
set -u
cd /workspace/tpu_inference || exit 1

TESTS=(
  tests/kernels/mla_v2_test.py
  tests/kernels/fused_moe_v1_test.py
  tests/kernels/ragged_gather_reduce_v2_test.py
  tests/kernels/gather_reduce_test.py
  tests/kernels/quantized_matmul_kernel_test.py
  tests/layers/common/test_process_weights.py
)
rc=0
for t in "${TESTS[@]}"; do
  name=$(basename "$t" .py)
  echo "=== $t"
  timeout 900 python3 -m pytest -q -rfE -p no:cacheprovider \
    --junitxml="/out/junit_$name.xml" "$t"
  code=$?
  summary=$(python3 - "/out/junit_$name.xml" <<'EOF'
import sys, xml.etree.ElementTree as ET
try:
    s = ET.parse(sys.argv[1]).getroot()
    s = s if s.tag == "testsuite" else s.find("testsuite")
    print(f"tests={s.get('tests')}_failures={s.get('failures')}_errors={s.get('errors')}_skipped={s.get('skipped')}")
except Exception as e:
    print(f"no_junit({type(e).__name__})")
EOF
)
  echo "RESULT kernels.$name=exit$code""_$summary"
  [ "$code" -eq 0 ] || rc=1
done
exit $rc
