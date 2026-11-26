"""Verify of a small block (2 to 8 tokens of one sequence) with the decode kernels of the OVH Olala module.

``metal.decode`` handles one token for each of T rows; the verify takes the L
tokens of the block as rows. Row-wise parts are unchanged: input projections
with their geodesic prologue, the fused MoE (kept up to 8 rows, where the
generic path switches to the sorted prefill experts), mixer projection, head.
Three changes:

- dense projections: from 3 rows on, MLX's quantized matmul, which reads each
  weight once for all rows (flat up to 4 rows), instead of the decode GEMV,
  whose cost grows with every row (scripts/profile_verify.py --rows);
- Mamba-3: the decode step applied to the L tokens one after the other, from the
  cached state (no chunked scan for so few tokens). Each intermediate state is
  produced anyway, so the rollback just picks the one after the kept tokens;
- attention (7 layers of 36): the generic ``DiffTPAttention`` mixer, whose
  multi-token kernel sees the cache and the earlier tokens of the block.

Also returns the residual stream after the drafter's layers (Télégraphe context).
"""

from contextlib import contextmanager

import mlx.core as mx

from opinel_telegraphe import TARGET_LAYERS

MAX_ROWS = 8
MAMBA_BY_TOKEN = True  # Mamba-3 decode step per token (False: one sequential m3_prefill for the block)
ATTENTION_BY_TOKEN = True  # attention layers through the decode kernels, one token after the other
DENSE_MATMUL_FROM = 3  # rows from which the dense projections go through MLX's quantized matmul


@contextmanager
def _small_block_kernels(metal):
    """Swap the GEMV dispatch of the OVH module for the duration of one verify."""
    gemv, limit = metal.gemv, metal.GEMV_MAX_TQ

    def dispatch(x_inputs, ws, prologue="plain", LI=0, Ka=0, sa=0, sb=0, T=1, K=None, RPS=None, deps=(),
                 stats_x=None):
        if T < DENSE_MATMUL_FROM:
            return gemv(x_inputs, ws, prologue, LI, Ka, sa, sb, T, K, RPS, deps, stats_x)
        if prologue == "geos":
            prologue, x_inputs = "geo", x_inputs[:3]
        out = metal._gemv_mlx(x_inputs, ws, prologue, LI, Ka, sa, sb, T, K or ws[0].cols)
        return out + [None] if stats_x is not None else out

    metal.gemv, metal.GEMV_MAX_TQ = dispatch, MAX_ROWS  # GEMV_MAX_TQ also keeps the fused MoE up to 8 rows
    try:
        yield
    finally:
        metal.gemv, metal.GEMV_MAX_TQ = gemv, limit


def _attention_by_token(metal, step, layer, cache, big, small, start, recorder=None):
    """Attention of one layer for the T block tokens, one decode-kernel pass per token (causal by
    construction, the cache written in place like metal.decode). Returns ymix (T, ...) f32."""
    att = layer.mixer
    Hq, Hk, Dh, Rk = att.n_heads, att.n_kv, att.head_dim, att.rank
    kc, sc = cache[0], cache[1]
    factored = "Factored" in type(kc).__name__
    T = big.shape[0]
    outs, shifts = [], []
    for t in range(T):
        b, s = big[t:t + 1], small[t:t + 1]
        pos = mx.array([start + t], dtype=mx.int32)
        if factored:
            if sc[0] is None:
                kprev = vprev = mx.zeros((1, 1, Hk, Dh), mx.float32)
                akprev = avprev = mx.zeros((1, 1, Hk, Rk), mx.float32)
            else:
                kprev, vprev = sc[0].astype(mx.float32), sc[1].astype(mx.float32)
                akprev, avprev = sc[2].astype(mx.float32), sc[3].astype(mx.float32)
            kc.ensure_inplace(1)
            wpos = mx.full((1,), kc.write_index, dtype=mx.int32)
            q, kraw, vraw, akraw, avraw = metal.v_prep_factored(
                b, s, kprev, vprev, akprev, avprev, step["prm"], pos, Hq, Hk, Dh, Rk, att.n_noise, att.window, 1,
                kc.rows_k, kc.rows_v, kc.coef_k, kc.coef_v, wpos)
            kc.commit_inplace(1)
            sc[0], sc[1], sc[2], sc[3] = kraw, vraw, akraw, avraw
            S = kc.write_index
            kstart = mx.full((1,), max(S - 1 - att.window, 0), dtype=mx.int32)
            o, m, l = metal.tpa_decode(q, kc.rows_k, kc.rows_v, kc.coef_k, kc.coef_v, kstart,
                                       mx.array([S], dtype=mx.int32), S, att.scale, att.softcap, 1, Hq, Hk, Dh, Rk)
            shifts.append((kraw, vraw, akraw, avraw))
        else:
            if sc[0] is None:
                kprev = vprev = mx.zeros((1, 1, Hk, Dh), mx.float32)
            else:
                kprev, vprev = sc[0].astype(mx.float32), sc[1].astype(mx.float32)
            q, k_new, v_new, kraw, vraw = metal.v_prep(b, s, kprev, vprev, step["prm"], pos, Hq, Hk, Dh, Rk,
                                                       att.n_noise, att.window, 1, kc.keys.dtype if kc.keys is not None
                                                       else mx.bfloat16)
            keys, values = kc.update_and_fetch(k_new, v_new)
            sc[0], sc[1] = kraw, vraw
            S = keys.shape[2]
            kstart = mx.full((1,), max(S - 1 - att.window, 0), dtype=mx.int32)
            o, m, l = metal.attn_decode(q, keys, values, kstart, att.scale, att.softcap, 1, Hq, Hk, Dh, S=S,
                                        slen=mx.array([S], dtype=mx.int32))
            shifts.append((kraw, vraw))
        outs.append(metal.attn_combine(o, m, l, s, b, Hq, Hk, Dh, att.snr, att.n_noise, att.gate_bias, 1,
                                       small_off=2 * Hk * Rk + 2 * Hk, big_off=Hq * Dh + 2 * Rk * Dh))
    if recorder is not None:  # same layout as the DiffTPAttention.project patch: one (T, ...) array per slot
        recorder.attention[id(att)] = tuple(mx.concatenate([sh[i].reshape(1, 1, *sh[i].shape[-2:])
                                                            for sh in shifts], axis=1) for i in range(len(shifts[0])))
    return mx.concatenate(outs, axis=0)


def verify_block(model, tokens, cache, recorder=None, aux_layers=TARGET_LAYERS):
    """Logits (L, vocab) and drafter aux rows (L, 5 * D) of the L tokens; updates cache like a forward.

    With a recorder (opinel_speculative.VerifyRecorder), records the Mamba-3 states after
    each token and, through the attention patch, the raw k / v of each position."""
    from vendor.ovh_olala import metal
    from vendor.ovh_olala.model import _attention_mask, _positions

    inner = model.model
    plan = metal._plan(inner)
    T = len(tokens)
    if not 2 <= T <= MAX_ROWS:
        raise ValueError(f"La vérification par les noyaux de décodage traite de 2 à {MAX_ROWS} tokens.")
    x = inner.embedding(mx.array(tokens)[None]).astype(mx.float32)
    D = x.shape[-1]
    kv0 = cache[inner.attn_idx[0]][0]
    positions = _positions(kv0, 1, T)
    start = kv0.offset
    mask = _attention_mask(kv0, T, inner.args.slw_wsize)
    dtype = inner.compute_dtype()
    resid, g, gst = x.reshape(T, D), None, None
    prev_geo, prev_li = None, 0
    aux = []
    with _small_block_kernels(metal):
        for step in plan.layers:
            li = step["li"]
            layer = inner.layers[li]
            if step["type"] == "M":
                H, P, N, R, A, n_static = step["dims"]
                outs = metal._in_gemv([step["in"]], resid, g, prev_geo, prev_li, gst, T, D)
                resid, proj = outs[-1], outs[0]
                state = cache[li][0]
                if state is None:
                    state = mx.zeros((1, H, P * N + R * N + P + A), mx.float32)
                if MAMBA_BY_TOKEN:
                    ys, states = [], []
                    for t in range(T):
                        y, state = metal.m3_step(proj[t:t + 1], step["prm"]["m3"], state, H, P, N, R, A, n_static)
                        ys.append(y)
                        states.append(state)
                    y = mx.concatenate(ys, axis=0)
                    if recorder is not None:
                        recorder.mamba[id(layer.mixer)] = ("states", states)
                else:
                    if recorder is not None:
                        recorder.mamba[id(layer.mixer)] = (state, proj[None])
                    y, state = metal.m3_prefill(proj[None], step["prm"]["m3"], state, H, P, N, R, A, n_static)[:2]
                    y = y.reshape(T, -1)
                cache[li][0] = state
                gmix = metal.gemv([y], [step["mixer_proj"]], T=T)[0]
            elif ATTENTION_BY_TOKEN:
                outs = metal._in_gemv([step["in"], step["in_small"]], resid, g, prev_geo, prev_li, gst, T, D)
                resid = outs[-1]
                ymix = _attention_by_token(metal, step, layer, cache[li], outs[0], outs[1], start, recorder)
                gmix = metal.gemv([ymix], [step["mixer_proj"]], T=T)[0]
            else:
                if g is not None:
                    resid = metal.geodesic(resid, g, prev_geo, prev_li, T, D)
                else:  # first layer: the decode path normalizes the embedding in its first prologue
                    resid = resid / mx.maximum(mx.sqrt(mx.sum(resid * resid, axis=-1, keepdims=True)), 1e-12)
                c = cache[li]
                y = layer.mixer(resid.reshape(1, T, D).astype(dtype), c[0], c[1], mask, positions)
                gmix = metal.gemv([y.reshape(T, -1).astype(mx.float32)], [step["mixer_proj"]], T=T)[0]
            resid, g, gst = metal._moe_block(step, resid, gmix, T, D)
            prev_geo, prev_li = step["geo_mlp"], li
            if li in aux_layers:
                aux.append(metal.geodesic(resid, g, prev_geo, prev_li, T, D))
        final = metal.geodesic(resid, g, prev_geo, prev_li, T, D)
        logits = metal.head(model.lm_head, final.reshape(T, 1, D)).reshape(T, -1)
    return logits, mx.concatenate(aux, axis=-1)
