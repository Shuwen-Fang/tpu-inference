"""What the CI container sees on this host, plus host and device speed checks.

Prints RESULT key=value lines for the side-by-side diff. The libtpu init log
goes to /out/libtpu_init.log.
"""
import os
import re
import subprocess
import sys
import time

import numpy as np


def result(key, value):
    print(f"RESULT {key}={value}", flush=True)


def section(name):
    print(f"\n=== {name}", flush=True)


def timed(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


section("environment")
for k, v in sorted(os.environ.items()):
    if re.match(r"(TPU|JAX|XLA|LIBTPU|PJRT|MEGASCALE)", k):
        print(f"{k}={v}")
        result(f"env.{k}", v)

# libtpu logs its topology and flag decisions at init. It must run before this
# process opens the TPU, since only one process can hold it.
section("libtpu init log")
env = dict(os.environ, TPU_STDERR_LOG_LEVEL="0", TPU_MIN_LOG_LEVEL="0")
proc = subprocess.run(
    [sys.executable, "-c", "import jax; print(len(jax.devices()))"],
    env=env, capture_output=True, text=True, timeout=600)
with open("/out/libtpu_init.log", "w") as f:
    f.write(proc.stderr)
seen = set()
for line in proc.stderr.splitlines():
    body = re.sub(r"^\S+ \S+ +\d+ \S+\] ", "", line)
    if re.search(r"resilien|topology|wrap|twist|bounds|ici|dcn|version|"
                 r"firmware|driver|chips?_per|megascale|fallback|degrad|"
                 r"disabl|override", body, re.I) and body not in seen:
        seen.add(body)
        print(body[:300])
    if len(seen) >= 120:
        break
result("libtpu.init_log_lines", len(proc.stderr.splitlines()))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_compilation_cache", False)

section("versions")
from importlib import metadata  # noqa: E402

for pkg in ("jax", "jaxlib", "libtpu", "libtpu-nightly", "vllm", "tpu-inference",
            "torchax", "tokamax", "numpy"):
    try:
        result(f"pkg.{pkg}", metadata.version(pkg))
    except metadata.PackageNotFoundError:
        pass
jax.print_environment_info()

section("devices")
devs = jax.devices()
result("jax.device_count", len(devs))
result("jax.process_count", jax.process_count())
for d in devs:
    stats = d.memory_stats() or {}
    line = (f"id={d.id} kind={d.device_kind} coords={getattr(d, 'coords', None)} "
            f"core={getattr(d, 'core_on_chip', None)} "
            f"hbm_limit={stats.get('bytes_limit', 0) / 2**30:.2f}GiB")
    print(line)
    result(f"device.{d.id}", line.replace(" ", "_"))

section("host cpu")
t = timed(lambda: sum(range(20_000_000)), reps=3)
result("cpu.python_loop_s", f"{t:.3f}")
a = np.random.default_rng(0).standard_normal((4096, 4096), dtype=np.float32)
t = timed(lambda: a @ a, reps=5)
result("cpu.sgemm_4k_all_threads_gflops", f"{2 * 4096**3 / t / 1e9:.1f}")
single = subprocess.run(
    [sys.executable, "-c",
     "import numpy as np, time; a=np.ones((2048,2048),np.float32); a@a; "
     "t=time.perf_counter(); [a@a for _ in range(5)]; "
     "print(2*2048**3*5/(time.perf_counter()-t)/1e9)"],
    env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
             MKL_NUM_THREADS="1"),
    capture_output=True, text=True)
result("cpu.sgemm_2k_one_thread_gflops", f"{float(single.stdout or 0):.1f}")
src = np.ones(2**29, np.float64)  # 4 GiB
dst = np.empty_like(src)
t = timed(lambda: np.copyto(dst, src), reps=3)
result("cpu.memcpy_4gib_gbps", f"{2 * src.nbytes / t / 1e9:.1f}")
del src, dst

section("xla compile time (cache off)")


def mlp(x, ws):
    for w in ws:
        x = jax.nn.gelu(x @ w)
    return x


mesh = jax.sharding.Mesh(np.array(devs), ("x",))
shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "x"))
total = 0.0
for d in (1024, 1536, 2048):  # distinct shapes so nothing is reused
    x = jax.ShapeDtypeStruct((256, d), jnp.bfloat16)
    ws = [jax.ShapeDtypeStruct((d, d), jnp.bfloat16, sharding=shard)] * 48
    t0 = time.perf_counter()
    jax.jit(mlp).lower(x, ws).compile()
    dt = time.perf_counter() - t0
    total += dt
    result(f"compile.mlp48_d{d}_s", f"{dt:.2f}")
result("compile.total_s", f"{total:.2f}")

section("host <-> device transfer")
host = np.ones((2**29,), np.float16)  # 1 GiB
for d in devs[:2]:
    t = timed(lambda: jax.device_put(host, d).block_until_ready())
    result(f"xfer.h2d_1gib_dev{d.id}_gbps", f"{host.nbytes / t / 1e9:.2f}")
all_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
big = np.ones((len(devs) * 2**29,), np.float16)  # 1 GiB per device
t = timed(lambda: jax.device_put(big, all_shard).block_until_ready())
result("xfer.h2d_all_devices_gbps", f"{big.nbytes / t / 1e9:.2f}")
on_dev = jax.device_put(host, devs[0])
t = timed(lambda: np.asarray(on_dev))
result("xfer.d2h_1gib_gbps", f"{host.nbytes / t / 1e9:.2f}")
del big, on_dev

section("per-device matmul and hbm speed")
for d in devs:
    x = jax.device_put(jnp.ones((8192, 8192), jnp.bfloat16), d)
    mm = jax.jit(lambda a: a @ a)
    mm(x).block_until_ready()
    t = timed(lambda: mm(x).block_until_ready(), reps=5)
    result(f"tpu.bf16_matmul_8k_dev{d.id}_tflops", f"{2 * 8192**3 / t / 1e12:.1f}")
    y = jax.device_put(jnp.ones((2**30,), jnp.bfloat16), d)  # 2 GiB
    cp = jax.jit(lambda a: a + 1)
    cp(y).block_until_ready()
    t = timed(lambda: cp(y).block_until_ready(), reps=5)
    result(f"tpu.hbm_rw_dev{d.id}_gbps", f"{2 * y.nbytes / t / 1e9:.0f}")
    del x, y
