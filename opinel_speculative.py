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

from opinel_telegraphe import capture_aux

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
                state, out = self.mamba[id(mixer)]
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


class Speculator:
    """Greedy speculative generation of the MLX target with a Télégraphe drafter."""

    def __init__(self, model, drafter, steps=3):
        self.model, self.drafter, self.steps = model, drafter, steps
        self.records = capture_aux(model)
        self.recorder = VerifyRecorder(model)

    def _forward(self, tokens, cache):
        self.records.clear()
        logits = self.model(mx.array(tokens)[None], cache=cache)
        aux = mx.concatenate([r[0] for r in self.records], axis=-1)
        self.records.clear()
        return logits[0], aux

    def prefill(self, prompt, cache, chunk=2048):
        """Prefill the whole prompt; return the target's greedy first token and the drafter's context."""
        keys = values = None
        logits = None
        for start in range(0, len(prompt), chunk):
            logits, aux = self._forward(prompt[start:start + chunk], cache)
            k, v = self.drafter.context_kv(aux, start=start)
            keys = k if keys is None else [mx.concatenate([a, b], axis=1) for a, b in zip(keys, k)]
            values = v if values is None else [mx.concatenate([a, b], axis=1) for a, b in zip(values, v)]
            mx.eval(logits, keys, values)
        return int(mx.argmax(logits[-1])), (keys, values)

    def generate(self, prompt, max_tokens, eos=(), cache=None, stats=None):
        """Yield the target's greedy tokens, several per verify forward."""
        from mlx_lm.models.cache import make_prompt_cache

        cache = cache if cache is not None else make_prompt_cache(self.model)
        stats = stats if stats is not None else {}
        stats.update(steps=0, accepted=[], kept=[], draft_seconds=0.0, verify_seconds=0.0)
        anchor, (keys, values) = self.prefill(prompt, cache)
        position = len(prompt)  # anchor position = number of context rows
        produced = 1
        yield anchor
        if anchor in eos:
            return
        while produced < max_tokens:
            started = time.perf_counter()
            drafts, _ = self.drafter.draft([anchor], [position], (keys, values), self.steps)
            drafted = time.perf_counter()
            block = [anchor] + [int(t) for t in drafts[0]]
            with self.recorder:
                logits, aux = self._forward(block, cache)
                choice = mx.argmax(logits, axis=-1)
                mx.eval(choice, aux)
            choice = choice.tolist()
            accepted = 0
            while accepted < self.steps and block[accepted + 1] == choice[accepted]:
                accepted += 1
            keep = accepted + 1
            self.recorder.rollback(cache, keep, len(block))
            k, v = self.drafter.context_kv(aux[:keep], start=position)
            keys = [mx.concatenate([a, b], axis=1) for a, b in zip(keys, k)]
            values = [mx.concatenate([a, b], axis=1) for a, b in zip(values, v)]
            mx.eval(keys, values, [c.state for c in cache])
            position += keep
            anchor = choice[accepted]
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
