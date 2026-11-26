"""Speculative decoding of the MLX target with Télégraphe: verify K + 1 tokens, keep the accepted prefix.

The target processes the anchor (the last token it chose) and K drafts in one
forward. When only the first ``keep`` of these L = K + 1 positions are accepted,
the caches are brought back to the state after those ``keep`` tokens:

- Mamba-3 layers: the state before the forward and the outputs of ``in_proj``
  are recorded during the forward (``Mamba3Mixer._fused_prefill``); the recurrence
  is replayed on the first ``keep`` rows only (``metal.m3_prefill``, no y written).
  No weight is read again.
- attention layers: the KV cache is trimmed by L - keep positions; the token-shift
  cache gets back the raw k / v (and, in the factored layout, the coefficients
  a_k / a_v) of the last kept position, recorded from ``DiffTPAttention.project``.

The residual stream after layers 1, 9, 17, 25 and 33 of the kept positions
becomes Télégraphe's context.
"""

import os
import time

import mlx.core as mx
import numpy as np

from opinel_telegraphe import DraftContext, capture_aux
from opinel_verify import verify_block

_RECORDER = {"active": None}


def _install_patches():
    from vendor.ovh_olala import model as olala

    if getattr(olala, "_speculative_patched", False):
        return
    fused_prefill = olala.Mamba3Mixer._fused_prefill
    project = olala.DiffTPAttention.project

    def recorded_fused_prefill(self, out, cache):
        recorder = _RECORDER["active"]
        if recorder is not None:
            recorder.mamba[id(self)] = (None if cache is None else cache[0], out)
        return fused_prefill(self, out, cache)

    def recorded_project(self, h, factors=False):
        result = project(self, h, factors)
        recorder = _RECORDER["active"]
        if recorder is not None:
            # k_raw, v_raw, then a_k, a_v in the factored layout
            recorder.attention[id(self)] = (result[1], result[2]) + ((result[9], result[10]) if factors else ())
        return result

    olala.Mamba3Mixer._fused_prefill = recorded_fused_prefill
    olala.DiffTPAttention.project = recorded_project
    olala._speculative_patched = True


class VerifyRecorder:
    """Records, during one multi-token forward, what a rollback to a shorter prefix needs."""

    def __init__(self, model):
        _install_patches()
        if os.environ.get("OLALA_FUSED_PREFILL", "1") != "1":
            raise ValueError("Le retour arrière suppose le préremplissage fusionné (OLALA_FUSED_PREFILL=1).")
        self.layers = model.model.layers
        self.mamba, self.attention = {}, {}

    def __enter__(self):
        self.mamba.clear()
        self.attention.clear()
        _RECORDER["active"] = self
        return self

    def __exit__(self, *exc):
        _RECORDER["active"] = None
        return False

    def rollback(self, cache, keep, total):
        """Bring cache back to the state after the first `keep` of the `total` tokens of the recorded forward."""
        from vendor.ovh_olala import metal

        if keep == total:
            return
        if not 1 <= keep < total:
            raise ValueError(f"Retour arrière impossible : {keep} sur {total}.")
        for index, layer in enumerate(self.layers):
            mixer = layer.mixer
            if layer.layer_type == "M":
                entry = self.mamba[id(mixer)]
                if entry[0] == "states":  # opinel_verify: the state after each token is already there
                    cache[index][0] = entry[1][keep - 1]
                    continue
                state, out = entry
                if state is None:
                    state = mx.zeros((out.shape[0], mixer.H, mixer.P * mixer.N + mixer.R * mixer.N + mixer.P
                                      + mixer.A), mx.float32)
                cache[index][0] = metal.m3_prefill(
                    out[:, :keep].astype(mx.float32), mixer._prm, state.astype(mx.float32), mixer.H, mixer.P,
                    mixer.N, mixer.R, mixer.A, mixer.n_static, write_y=False)[1]
            else:
                kv_cache, shift_cache = cache[index][0], cache[index][1]
                kv_cache.trim(total - keep)
                recorded = self.attention[id(mixer)]
                for slot, values in enumerate(recorded):
                    shift_cache[slot] = values[:, keep - 1:keep]


ASYNC_TAIL = True  # rollback and drafter context append evaluated asynchronously (issue #4)


def rejection_step(drafts, candidate_ids, log_q, law_ids, law_probs, rng):
    """Speculative sampling of one verified block (Leviathan et al., Chen et al.), exact for the target law.

    drafts (K,): the drafted tokens; candidate_ids / log_q (K, k): the draft law q_i over the
    candidates of step i (the row of the walk, given the chosen predecessor); law_ids / law_probs
    (K + 1, m): the target law p_j at each verified position (vllm_law, zero outside law_ids).
    Draft i is kept with probability min(1, p_i(x) / q_i(x)); at the first rejection the next
    token is drawn from norm(max(p_i - q_i, 0)); when all K are kept, from p_K.
    Returns (accepted, next token).
    """
    for i, token in enumerate(drafts):
        p_row = dict(zip(law_ids[i].tolist(), law_probs[i].tolist()))
        q_row = dict(zip(candidate_ids[i].tolist(), np.exp(log_q[i]).tolist()))
        p_x, q_x = p_row.get(int(token), 0.0), q_row.get(int(token), 0.0)
        if q_x > 0 and rng.random() * q_x < p_x:
            continue
        ids = law_ids[i]
        residual = np.maximum(law_probs[i] - np.array([q_row.get(int(t), 0.0) for t in ids]), 0.0)
        if residual.sum() <= 0:  # p == q on the support: any draw from p is exact
            residual = law_probs[i]
        return i, int(ids[rng.choice(len(ids), p=residual / residual.sum())])
    last = len(drafts)
    return last, int(law_ids[last][rng.choice(law_ids.shape[1], p=law_probs[last] / law_probs[last].sum())])


class Speculator:
    """Greedy speculative generation of the MLX target with a Télégraphe drafter."""

    def __init__(self, model, drafter, steps=3, decode_kernels="auto", layers_per_submit=None):
        """decode_kernels: verify through the decode kernels of the OVH module (opinel_verify, True) or its
        layer-by-layer prefill path (False); "auto" times both on this machine and keeps the faster
        (the decode kernels win on M5, the prefill path on M4)."""
        self.model, self.drafter, self.steps = model, drafter, steps
        self.layers_per_submit = layers_per_submit
        self.timings = None
        if steps + 1 > 8:
            decode_kernels = False
        self.records = capture_aux(model)
        self.recorder = VerifyRecorder(model)
        self.decode_kernels = self.calibrate() if decode_kernels == "auto" else bool(decode_kernels)

    def calibrate(self, context=1024, repeats=12):
        """Time one verify of K + 1 tokens both ways after a context-token prompt; True if the decode
        kernels are faster. Each verify restarts from a copy of the same cache (about 2 s in all)."""
        import copy
        import statistics

        from mlx_lm.models.cache import make_prompt_cache

        rng = np.random.default_rng(0)
        vocab = self.model.model.embedding.weight.shape[0]
        prompt = rng.integers(1000, min(vocab, 100000), size=context + self.steps + 1).tolist()
        base = make_prompt_cache(self.model)
        for start in range(0, context, 512):
            self.model(mx.array(prompt[start:min(start + 512, context)])[None], cache=base)
            mx.eval([c.state for c in base])
        block = prompt[context:]
        paths = {"noyaux de décodage": lambda c: verify_block(self.model, block, c,
                                                               layers_per_submit=self.layers_per_submit),
                 "chemin de préremplissage": lambda c: self._forward(block, c)}
        times = {name: [] for name in paths}
        for repeat in range(repeats + 2):
            for name, run in paths.items():
                work = copy.deepcopy(base)
                mx.eval([c.state for c in work])
                started = time.perf_counter()
                mx.eval(run(work))
                if repeat >= 2:
                    times[name].append(time.perf_counter() - started)
        self.timings = {name: 1000 * statistics.median(values) for name, values in times.items()}
        return self.timings["noyaux de décodage"] <= self.timings["chemin de préremplissage"]

    def _forward(self, tokens, cache):
        self.records.clear()
        logits = self.model(mx.array(tokens)[None], cache=cache)
        aux = mx.concatenate([r[0] for r in self.records], axis=-1)
        self.records.clear()
        return logits[0], aux

    def _extend(self, context, tokens, cache, start):
        """Forward tokens (positions start..) and append their drafter K / V to context; returns the logits."""
        logits, aux = self._forward(tokens, cache)
        k, v = self.drafter.context_kv(aux, start=start)
        if context is None:
            d = self.drafter
            context = DraftContext(d.layers, d.kv_heads, d.head_dim)
        context.append(k, v)
        mx.eval(logits, context.arrays())
        return logits, context

    def prefill(self, prompt, cache, context=None, start=0, on_prefix=None, chunk=2048):
        """Prefill prompt[start:] on a cache (and drafter context) that already hold prompt[:start].

        Returns (greedy first token, drafter context, logits of the last position). on_prefix(cache,
        context) is called once every token but the last is in: the prompt cache snapshot of the chat.
        The last token goes alone through the layer-by-layer path (OLALA_FUSED=0): the fused decode
        would skip the layers whose output feeds the drafter.
        """
        end = len(prompt) - 1
        if not 0 <= start <= end:
            raise ValueError("Préremplissage spéculatif : reprise au-delà du prompt.")
        for offset in range(start, end, chunk):
            _, context = self._extend(context, prompt[offset:min(offset + chunk, end)], cache, offset)
        if on_prefix is not None:
            on_prefix(cache, context)
        fused = os.environ.get("OLALA_FUSED")
        os.environ["OLALA_FUSED"] = "0"
        try:
            logits, context = self._extend(context, prompt[end:], cache, end)
        finally:
            if fused is None:
                del os.environ["OLALA_FUSED"]
            else:
                os.environ["OLALA_FUSED"] = fused
        return int(mx.argmax(logits[-1])), context, logits[-1]

    def generate(self, prompt, max_tokens, eos=(), cache=None, stats=None, temperature=0.0, top_p=1.0, top_k=0,
                 seed=None, context=None, start=0, on_prefix=None):
        """Yield the target's tokens, several per verify forward: greedy at temperature 0, otherwise
        sampled from the vLLM-order law (temperature, top-k, top-p) by rejection, which keeps that
        law exact whatever the drafter proposes. cache / context / start: resume a prefill that
        already holds prompt[:start] (prompt cache of the chat); on_prefix: see prefill."""
        from opinel_mlx import vllm_law

        sampling = temperature > 0
        rng = np.random.default_rng(seed)
        if sampling and seed is not None:
            np.random.seed(seed)  # the drafter's walk draws with numpy's global generator

        def law(logits):
            ids, values = vllm_law(logits.astype(mx.float32), temperature, top_p, top_k)
            probs = mx.softmax(values, axis=-1)
            mx.eval(ids, probs)
            return np.array(ids), np.array(probs.astype(mx.float32), dtype=np.float64)

        from mlx_lm.models.cache import make_prompt_cache

        cache = cache if cache is not None else make_prompt_cache(self.model)
        stats = stats if stats is not None else {}
        stats.update(steps=0, accepted=[], kept=[], draft_seconds=0.0, verify_seconds=0.0)
        began = time.perf_counter()
        anchor, context, last = self.prefill(prompt, cache, context, start, on_prefix)
        stats["prefill_seconds"] = time.perf_counter() - began
        if sampling:
            ids, probs = law(last[None])
            anchor = int(ids[0][rng.choice(ids.shape[1], p=probs[0] / probs[0].sum())])
        position = len(prompt)  # anchor position = number of context rows
        produced = 1
        yield anchor
        if anchor in eos:
            return
        while produced < max_tokens:
            started = time.perf_counter()
            drafts, log_q, candidates = self.drafter.draft([anchor], [position], context, self.steps,
                                                           temperature=temperature, with_candidates=True)
            drafted = time.perf_counter()
            block = [anchor] + [int(t) for t in drafts[0]]
            with self.recorder:
                if self.decode_kernels:
                    logits, aux = verify_block(self.model, block, cache, recorder=self.recorder,
                                               layers_per_submit=self.layers_per_submit)
                else:
                    logits, aux = self._forward(block, cache)
                if sampling:
                    mx.async_eval(aux)
                    ids, probs = law(logits)
                    accepted, next_token = rejection_step(drafts[0], candidates[0], log_q[0], ids, probs, rng)
                else:
                    choice = mx.argmax(logits, axis=-1)
                    mx.eval(choice, aux)
                    choice = choice.tolist()
                    accepted = 0
                    while accepted < self.steps and block[accepted + 1] == choice[accepted]:
                        accepted += 1
                    next_token = choice[accepted]
            keep = accepted + 1
            self.recorder.rollback(cache, keep, len(block))
            context.append(*self.drafter.context_kv(aux[:keep], start=position))
            # without waiting: the GPU rolls back while the CPU builds the next draft graph
            (mx.async_eval if ASYNC_TAIL else mx.eval)(context.arrays(), [c.state for c in cache])
            position += keep
            anchor = next_token
            stats["steps"] += 1
            stats["accepted"].append(accepted)
            stats["kept"].append(keep)
            stats["draft_seconds"] += drafted - started
            stats["verify_seconds"] += time.perf_counter() - drafted
            for token in block[1:keep] + [anchor]:
                produced += 1
                yield token
                if token in eos or produced >= max_tokens:
                    return
