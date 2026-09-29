"""Checks TPU collectives and per-device matmuls against CPU references.

all_gather, all_to_all and ppermute only move data, so their results must
match the input bit for bit. psum and psum_scatter are compared against a
float64 reference with a dtype-dependent tolerance. Each op is repeated and
every repeat must be bitwise identical to the first, which catches
intermittent corruption. The same matmul runs on every device, and the
outputs must be bitwise identical across devices.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

REPEATS = 20
devs = jax.devices()
n = len(devs)
mesh = Mesh(np.array(devs), ("x",))
rows = NamedSharding(mesh, P("x"))
rng = np.random.default_rng(0)
failures = 0


def result(key, value):
    print(f"RESULT {key}={value}", flush=True)


def smap(fn):
    return jax.jit(shard_map(fn, mesh=mesh, in_specs=P("x"), out_specs=P("x")))


# Each op takes one shard of shape (1, n*c, d) and returns (1, ...).
def op_all_gather(x):
    return jax.lax.all_gather(x[0], "x", tiled=True)[None]


def op_all_to_all(x):
    return jax.lax.all_to_all(x[0], "x", 0, 0, tiled=True)[None]


def op_psum(x):
    return jax.lax.psum(x, "x")


def op_psum_scatter(x):
    return jax.lax.psum_scatter(x[0], "x", scatter_dimension=0, tiled=True)[None]


def op_ppermute(x, shift):
    return jax.lax.ppermute(x, "x", [(j, (j + shift) % n) for j in range(n)])


def ref_all_gather(x):
    return np.broadcast_to(x.reshape(1, -1, x.shape[-1]), (n, n * x.shape[1], x.shape[2]))


def ref_all_to_all(x):
    c = x.shape[1] // n
    blocks = x.reshape(n, n, c, x.shape[2])  # [src, dst_block, c, d]
    return blocks.transpose(1, 0, 2, 3).reshape(n, n * c, x.shape[2])


def ref_psum(x):
    return np.broadcast_to(x.astype(np.float64).sum(0, keepdims=True), x.shape)


def ref_psum_scatter(x):
    c = x.shape[1] // n
    return x.astype(np.float64).sum(0).reshape(n, c, x.shape[2])


def ref_ppermute(x, shift):
    return np.roll(x, shift, axis=0)


OPS = [("all_gather", op_all_gather, ref_all_gather, True),
       ("all_to_all", op_all_to_all, ref_all_to_all, True),
       ("psum", op_psum, ref_psum, False),
       ("psum_scatter", op_psum_scatter, ref_psum_scatter, False)]
OPS += [(f"ppermute_{s}", functools.partial(op_ppermute, shift=s),
         functools.partial(ref_ppermute, shift=s), True) for s in range(1, n)]

# Per-shard (n*c, d): tiny, 4 MiB-ish, and DeepSeek's hidden size at 56 MiB bf16.
SIZES = [(16, 128), (128, 2048), (512, 7168)]
DTYPES = [(jnp.float32, 1e-5), (jnp.bfloat16, 2e-2), (jnp.int32, 0)]


def make_input(c, d, dtype):
    shape = (n, n * c, d)
    if dtype == jnp.int32:
        return rng.integers(-1000, 1000, shape, dtype=np.int32)
    return rng.standard_normal(shape, dtype=np.float32).astype(dtype)


for c, d in SIZES:
    for dtype, tol in DTYPES:
        host = make_input(c, d, dtype)
        x = jax.device_put(host, rows)
        name_dt = jnp.dtype(dtype).name
        for name, op, ref, exact in OPS:
            fn = smap(op)
            first = fn(x)
            same = all(bool(jnp.array_equal(fn(x), first)) for _ in range(REPEATS - 1))
            got = np.asarray(first).astype(np.float64)
            want = np.asarray(ref(host)).astype(np.float64)
            if exact:
                ok = got.shape == want.shape and np.array_equal(got, want)
                err = 0.0 if ok else float(np.abs(got - want).max()) if got.shape == want.shape else float("inf")
            else:
                scale = max(float(np.abs(want).max()), 1.0)
                err = float(np.abs(got - want).max()) / scale
                ok = err <= tol
            ok = ok and same
            failures += not ok
            key = f"coll.{name}.{name_dt}.{n * c}x{d}"
            result(key, f"{'ok' if ok else 'MISMATCH'}_err={err:.3g}_repeatable={same}")

# Same program and inputs on every device must give bitwise-identical output.
for dtype in (jnp.bfloat16, jnp.float8_e4m3fn):
    a = rng.standard_normal((4096, 4096), dtype=np.float32)
    b = rng.standard_normal((4096, 4096), dtype=np.float32)
    mm = jax.jit(lambda p, q: jnp.dot(p, q, preferred_element_type=jnp.float32))
    outs = []
    for dev in devs:
        pa = jax.device_put(jnp.asarray(a, dtype), dev)
        pb = jax.device_put(jnp.asarray(b, dtype), dev)
        outs.append(np.asarray(mm(pa, pb)))
    ref = (np.asarray(jnp.asarray(a, dtype), np.float64)
           @ np.asarray(jnp.asarray(b, dtype), np.float64))
    rel = [float(np.abs(o - ref).max() / np.abs(ref).max()) for o in outs]
    identical = all(np.array_equal(o, outs[0]) for o in outs)
    ok = identical and max(rel) < 1e-2
    failures += not ok
    result(f"matmul.{jnp.dtype(dtype).name}.cross_device",
           f"{'ok' if ok else 'MISMATCH'}_identical={identical}_max_rel_err={max(rel):.3g}")

# A long all_to_all loop on DeepSeek-sized expert dispatch buffers.
host = make_input(512, 7168, jnp.bfloat16)
x = jax.device_put(host, rows)
fn = smap(op_all_to_all)
want = fn(x)
bad = sum(not bool(jnp.array_equal(fn(x), want)) for _ in range(300))
failures += bad > 0
result("coll.all_to_all_soak_300", f"{'ok' if bad == 0 else 'MISMATCH'}_bad_iters={bad}")

result("numerics.failures", failures)
raise SystemExit(1 if failures else 0)
