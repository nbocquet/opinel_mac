"""Mamba-3 rollback replay of several layers in one Metal launch (issue #7).

After a partly refused verify, each Mamba-3 layer replays its recurrence over the
`keep` accepted tokens from the state before the pass (m3_prefill without output).
One m3_prefill per layer costs about 80 µs for 30 µs of useful memory traffic: 29
launches of 48 threadgroups, and MLX zero-fills the state output first
(init_value=0). This kernel is OVH's m3_prefill source (vendor/ovh_olala/metal.py),
taken as is and changed only at its edges:

- one batch row per layer, each with its own parameter pack (stacked) and its own
  state buffer, passed as separate inputs and outputs: no copy of the 1.7 MB
  states into a stacked array;
- the number of tokens is a compile-time constant, the range [0, keep) is fixed,
  and neither y nor log(alpha) is written;
- no zero fill: the kernel writes every element of the new state.

Metal allows 31 buffers per kernel, so the layers go in groups of GROUP. Same
arithmetic as m3_prefill, in the same order: the new states are bit-identical.
"""

import math
from pathlib import Path
import re

import mlx.core as mx

GROUP = 10  # layers per launch (2 buffers each, plus proj and the parameter packs)

_kernels = {}
_packs = {}  # stacked parameter packs per group of mixers, built once
_source = None


def _m3_prefill_source():
    """The f-string body of OVH's m3_prefill kernel source, read from the vendored file."""
    global _source
    if _source is None:
        text = (Path(__file__).resolve().parent / "vendor" / "ovh_olala" / "metal.py").read_text()
        body = text[text.index("def m3_prefill("):]
        start = body.index('src = f"""') + len('src = f"""')
        _source = body[start:body.index('"""', start)]
    return _source


def replay_source(count, keep, H, P, N, R, A, n_static, prm_size, PB=64):
    """Metal source of the grouped replay: m3_prefill's, with the edits listed in the module docstring."""
    NSG_ = 16
    SW = P * N + R * N + P + A
    RPSG = (PB + NSG_ - 1) // NSG_
    seg, write_y = 1, False
    names = {"H": H, "P": P, "N": N, "R": R, "A": A, "PB": PB, "NSG_": NSG_, "RPSG": RPSG, "n_static": n_static,
             "SW": SW, "seg": seg, "write_y": write_y, "math": math}
    src = eval('f"""' + _m3_prefill_source() + '"""', {}, names)  # noqa: S307 - the vendored, pinned source

    def replace(old, new):
        nonlocal src
        if src.count(old) != 1:
            raise RuntimeError(f"Source m3_prefill inattendue : {old!r}")
        src = src.replace(old, new)

    replace("const int L = proj_shape[1];", f"const int L = {keep};")
    replace("const int s0 = rng[2 * b], s1 = rng[2 * b + 1];", "const int s0 = 0, s1 = L;")
    select = "\n".join(f"            case {i}: sin_ = st{i} + (size_t)h * SW; sout = so{i} + (size_t)h * SW; break;"
                       for i in range(count))
    replace("const device float* sin_ = st + ((size_t)b * H + h) * SW;\n"
            "        device float* sout = st_out + ((size_t)b * H + h) * SW;",
            "const device float* sin_ = st0;\n        device float* sout = so0;\n"
            f"        switch (b) {{\n{select}\n        }}\n"
            f"        const device float* prm_l = prm + (size_t)b * {prm_size};")
    replace("if (pb == 0 && tid == 0) logA[b * H + h] = lsum;", "(void)lsum;")
    # every parameter read goes to this layer's pack
    definition = f"const device float* prm_l = prm + (size_t)b * {prm_size};"
    head, tail = src.split(definition, 1)
    return head + definition + re.sub(r"\bprm\b", "prm_l", tail)


def _kernel(count, keep, H, P, N, R, A, n_static, prm_size):
    key = (count, keep, H, P, N, R, A, n_static, prm_size)
    if key in _kernels:
        return _kernels[key]
    from vendor.ovh_olala import metal

    src = replay_source(count, keep, H, P, N, R, A, n_static, prm_size, PB=metal.M3P_PB)
    _kernels[key] = mx.fast.metal_kernel(
        name=f"opinel_m3_replay_{count}_{keep}",
        input_names=["proj", "prm"] + [f"st{i}" for i in range(count)],
        output_names=[f"so{i}" for i in range(count)],
        source=src,
        header=metal._HDR,
    )
    return _kernels[key]


def replay(mixers, states, projections, keep):
    """New states (1, H, SW) of Mamba-3 layers after their first `keep` recorded tokens.

    mixers: the layers' mixers (for their parameter packs and sizes); states: the states
    before the verify (1, H, SW) f32; projections: their in_proj outputs (1, L, PW) f32."""
    from vendor.ovh_olala import metal

    out = []
    for begin in range(0, len(mixers), GROUP):
        group = list(range(begin, min(begin + GROUP, len(mixers))))
        m = mixers[group[0]]
        ids = tuple(id(mixers[i]) for i in group)
        if ids not in _packs:
            _packs[ids] = mx.stack([mixers[i]._prm for i in group])
            mx.eval(_packs[ids])
        prm = _packs[ids]
        proj = mx.concatenate([projections[i][:, :keep].astype(mx.float32) for i in group])
        kernel = _kernel(len(group), keep, m.H, m.P, m.N, m.R, m.A, m.n_static, prm.shape[1])
        SW = m.P * m.N + m.R * m.N + m.P + m.A
        npb = m.P // metal.M3P_PB
        out += kernel(
            inputs=[proj, prm] + [states[i].astype(mx.float32) for i in group],
            grid=(len(group) * m.H * npb * 16 * 32, 1, 1),
            threadgroup=(16 * 32, 1, 1),
            output_shapes=[(1, m.H, SW)] * len(group),
            output_dtypes=[mx.float32] * len(group),
        )
    return out
