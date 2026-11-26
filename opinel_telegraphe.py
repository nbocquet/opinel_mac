"""Télégraphe (DSpine drafter of Opinel N°8) in MLX.

Port of the eager path of the vLLM fork, branch ``opinel``
(``vllm/model_executor/models/qwen3_dspine.py`` and ``qwen3_dflash.py``,
``v1/worker/gpu/spec_decode/dflash2/speculator.py``):

- context: the residual stream of the target after layers 1, 9, 17, 25 and 33,
  concatenated, through ``fc`` and ``hidden_norm``, gives every draft layer its
  context K/V (k_norm, RoPE at the context position);
- block: the anchor (last token known to the target) then K mask tokens, at the
  positions that follow the context, through 5 Qwen3 layers (causal block,
  sliding window of 4096 over the context), each followed by the adjacent causal
  injection (position t receives a gated message from t - 1, position 1 from the
  anchor's code);
- candidates: the top 16 of the drafter's head at each mask position; their
  transitions are rescored by an exact fp32 rewrite of the last injection with the
  candidate predecessor's code, and the walk picks one candidate per position.
"""

import hashlib
import json
import math
from pathlib import Path
import zipfile

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent
REPOSITORY = "maurienne-ai/Olala-v1.0-mini-7A1B-Telegraphe"
REVISION = "25003c080aa6f0f520368aa68f8d203d5519e255"
FILES = {
    "config.json": None,
    "model.safetensors.index.json": None,
    "mask_embedding.pt": "737d4d9336a4a0faa99db8c1bd0738a46aabb0889d723a1d30f621086b14821b",
    "model-00001-of-00002.safetensors": "c395d2bb5e18ec892a8fff88e18726690d0764a0b8338c4b7d990256639c0b30",
    "model-00002-of-00002.safetensors": "48b156474b2ec0e35cf938219e947654859a24dcf13368e1bbc74a8ecbfe1ffb",
    "LICENSE": None,
    "NOTICE": None,
    "README.md": None,
}
DEFAULT_FOLDER = ROOT / "models" / "telegraphe"
TARGET_LAYERS = (1, 9, 17, 25, 33)


def mask_embedding_from_archive(path):
    """The (1536,) bf16 mask embedding of mask_embedding.pt, read from the archive's raw storage.

    The pickle is never executed: the archive holds a single 3072-byte storage
    (data/0, little-endian bf16), whose SHA256 of the whole file is checked first.
    """
    with zipfile.ZipFile(path) as archive:
        storages = [info for info in archive.infolist() if info.filename.endswith("/data/0")]
        if len(storages) != 1 or storages[0].file_size != 2 * 1536:
            raise ValueError("mask_embedding.pt inattendu.")
        raw = np.frombuffer(archive.read(storages[0]), dtype="<u2").astype(np.uint32) << 16
    return raw.view(np.float32)


def prepare(folder=DEFAULT_FOLDER, offline=False):
    """Download Télégraphe at the pinned revision into folder, check its hashes, extract the mask embedding."""
    from huggingface_hub import snapshot_download

    folder = Path(folder)
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=folder, local_files_only=offline,
                      allow_patterns=list(FILES))
    for name, expected in FILES.items():
        if expected is not None:
            digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"Empreinte inattendue pour {name} : {digest}")
    np.save(folder / "mask_embedding.npy", mask_embedding_from_archive(folder / "mask_embedding.pt"))
    return folder


def rms_norm(x, weight, eps):
    return mx.fast.rms_norm(x, weight, eps)


def rms(x):
    """Root mean square in fp32, floored like the reference (finfo.tiny)."""
    x = x.astype(mx.float32)
    return mx.sqrt(mx.maximum(mx.mean(x * x, axis=-1, keepdims=True), np.finfo(np.float32).tiny))


def linear(x, weight):
    return x @ weight.T


class Telegraphe:
    def __init__(self, folder=DEFAULT_FOLDER, embedding=None, normalize_embeddings=True, target_head=None,
                 quantize_bits=None, group_size=64):
        """embedding: the target's table; target_head: the target's (quantized) lm_head, used instead of the
        drafter's own bf16 head (OLALA_DSPINE_SHARED_HEAD); quantize_bits: affine quantization of the decoder
        linears and fc (OLALA_DSPINE_INT8 quantizes them per channel), the injection staying in bf16."""
        folder = Path(folder)
        self.config = json.loads((folder / "config.json").read_text())
        draft = self.config["dflash_config"]
        if tuple(draft["target_layer_ids"]) != TARGET_LAYERS or not draft.get("causal", False):
            raise ValueError("Checkpoint Télégraphe inattendu.")
        self.block_size = int(draft["block_size"])
        self.mask_token_id = int(draft["mask_token_id"])
        self.transfer_dim = int(draft["transfer_dim"])
        self.top_k = int(draft["transition_top_k"])
        self.transfer_eps = float(draft.get("transfer_space_eps", 1e-6))
        self.eps = float(self.config["rms_norm_eps"])
        self.layers = int(self.config["num_hidden_layers"])
        self.heads = int(self.config["num_attention_heads"])
        self.kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config["head_dim"])
        self.rope_theta = float(self.config["rope_theta"])
        self.window = int(self.config["sliding_window"]) if self.config.get("use_sliding_window") else None
        weights = {}
        for name in sorted(set(json.loads((folder / "model.safetensors.index.json").read_text())
                               ["weight_map"].values())):
            weights.update(mx.load(str(folder / name)))
        self.w = weights
        self.mask_embedding = mx.array(np.load(folder / "mask_embedding.npy")).astype(mx.bfloat16)
        if embedding is None:
            raise ValueError("Télégraphe lit la table d'embedding de la cible.")
        if normalize_embeddings:
            # vLLM folds the target's nGPT normalization into the shared table at load time.
            table = embedding.astype(mx.float32)
            table = table / mx.maximum(mx.sqrt(mx.sum(table * table, axis=-1, keepdims=True)), 1e-12)
            embedding = table.astype(mx.bfloat16)
        self.embedding = embedding
        self.target_head = target_head
        if target_head is not None:
            del self.w["lm_head.weight"]
        self.fuse_projections()
        self.quantized = {}
        if quantize_bits:
            for name in list(self.w):
                if name == "fc.weight" or (name.startswith("layers.") and name.endswith("proj.weight")):
                    self.quantized[name] = mx.quantize(self.w.pop(name), group_size=group_size, bits=quantize_bits)
            self.quantize_args = {"group_size": group_size, "bits": quantize_bits}
        mx.eval(self.w, self.quantized, self.mask_embedding, self.embedding)

    def fuse_projections(self):
        """One matmul for q / k / v and one for gate / up (fewer kernel launches). Rows are quantized
        independently, so the fused weights quantize to exactly the separate ones. k_proj and v_proj
        stay for context_kv."""
        for i in range(self.layers):
            attn, mlp = f"layers.{i}.self_attn.", f"layers.{i}.mlp."
            self.w[attn + "qkv_proj.weight"] = mx.concatenate(
                [self.w.pop(attn + "q_proj.weight"), self.w[attn + "k_proj.weight"], self.w[attn + "v_proj.weight"]])
            self.w[mlp + "gate_up_proj.weight"] = mx.concatenate(
                [self.w.pop(mlp + "gate_proj.weight"), self.w.pop(mlp + "up_proj.weight")])

    def p(self, name):
        return self.w[name]

    def mm(self, x, name):
        if name in self.quantized:
            weight, scales, biases = self.quantized[name]
            return mx.quantized_matmul(x, weight, scales, biases, transpose=True, **self.quantize_args)
        return linear(x, self.w[name])

    def head_logits(self, hidden):
        if self.target_head is not None:
            return self.target_head(hidden)
        return linear(hidden, self.p("lm_head.weight"))

    def head_rows(self, ids):
        """fp32 rows of the head for token ids: what its matmul multiplies by (dequantized in fp32)."""
        head = self.target_head
        if head is None:
            return self.p("lm_head.weight")[ids].astype(mx.float32)
        return mx.dequantize(head.weight[ids], head.scales[ids].astype(mx.float32),
                             head.biases[ids].astype(mx.float32), group_size=head.group_size, bits=head.bits)

    # --- context -----------------------------------------------------------------------
    def context_kv(self, aux, start=0):
        """K/V of every draft layer for context rows `aux` (T, 5 * 1536) at positions start.."""
        states = self.mm(aux.astype(mx.bfloat16), "fc.weight")
        states = rms_norm(states, self.p("hidden_norm.weight"), self.eps)
        keys, values = [], []
        for i in range(self.layers):
            prefix = f"layers.{i}.self_attn."
            k = self.mm(states, prefix + "k_proj.weight").reshape(-1, self.kv_heads, self.head_dim)
            k = rms_norm(k, self.p(prefix + "k_norm.weight"), self.eps)
            k = mx.fast.rope(k.transpose(1, 0, 2)[None], self.head_dim, traditional=False,
                             base=self.rope_theta, scale=1.0, offset=start)[0]
            v = self.mm(states, prefix + "v_proj.weight").reshape(-1, self.kv_heads, self.head_dim)
            keys.append(k)  # (kv_heads, T, head_dim)
            values.append(v.transpose(1, 0, 2))
        return keys, values

    # --- injection ---------------------------------------------------------------------
    def project(self, layer, states):
        normalized = states.astype(mx.float32) / (rms(states) + self.transfer_eps)
        projected = linear(normalized.astype(states.dtype), self.p(f"injection.readouts.{layer}.weight"))
        projected = projected.astype(mx.float32)
        norm = mx.sqrt(mx.sum(projected * projected, axis=-1, keepdims=True))
        return (math.sqrt(self.transfer_dim) * projected / (norm + self.transfer_eps)).astype(states.dtype)

    def write(self, layer, receiver_states, receiver_features, predecessor_features, refine=False):
        def lin(name, features):
            weight = self.p(f"injection.{name}.weight")
            if refine:
                return linear(features.astype(mx.float32), weight.astype(mx.float32))
            return linear(features, weight).astype(mx.float32)

        gate = mx.sigmoid(linear(receiver_features, self.p("injection.receiver_gate.weight")).astype(mx.float32)
                          + lin("predecessor_gate", predecessor_features)
                          + self.p("injection.gate_bias").astype(mx.float32))
        gated = gate * lin("message", predecessor_features)
        if not refine:
            gated = gated.astype(receiver_states.dtype)
        update = lin(f"writes.{layer}", gated)
        result = receiver_states.astype(mx.float32) + rms(receiver_states) * update
        return result if refine else result.astype(receiver_states.dtype)

    def inject(self, layer, states, anchor_features):
        """states (B, n, H): position 0 kept, 1 gets the anchor's code, t > 1 the features of t - 1."""
        features = self.project(layer, states)
        predecessor = mx.concatenate([anchor_features[:, None], features[:, 1:-1]], axis=1)
        written = self.write(layer, states[:, 1:], features[:, 1:], predecessor)
        return mx.concatenate([states[:, :1], written], axis=1)

    def token_features(self, ids):
        return (self.p("transfer_codebook")[ids].astype(mx.float32)
                * math.sqrt(self.transfer_dim)).astype(mx.bfloat16)

    # --- block forward -----------------------------------------------------------------
    def attention_mask(self, context_lengths, context_size, n):
        """Boolean (B, 1, n, context_size + n): context rows p < length and inside the window, causal block."""
        lengths = mx.array(context_lengths)[:, None, None]
        query = lengths + mx.arange(n)[None, :, None]
        key = mx.arange(context_size)[None, None, :]
        visible = mx.broadcast_to(key < lengths, (len(context_lengths), n, context_size))
        if self.window is not None:
            visible = visible & (query - key < self.window)
        block = mx.broadcast_to(mx.tril(mx.ones((n, n), dtype=mx.bool_))[None], (len(context_lengths), n, n))
        return mx.concatenate([visible, block], axis=-1)[:, None]

    def forward(self, anchors, context_lengths, context, steps):
        """Draft hidden states for B blocks.

        anchors: B anchor token ids, at positions context_lengths (the number of context rows
        each block sees); context: (keys, values) of context_kv, shared by the blocks.
        Returns (final hidden (B, n, H), pre-injection states of the last layer (B, n, H)).
        """
        n = steps + 1
        if n > self.block_size:
            raise ValueError(f"Télégraphe propose au plus {self.block_size - 1} tokens.")
        B = len(anchors)
        anchor_ids = mx.array(anchors)
        x = mx.concatenate([self.embedding[anchor_ids][:, None],
                            mx.broadcast_to(self.mask_embedding, (B, steps, self.mask_embedding.shape[-1]))], axis=1)
        anchor_features = self.token_features(anchor_ids)
        keys_ctx, values_ctx = context
        context_size = keys_ctx[0].shape[1]
        mask = self.attention_mask(context_lengths, context_size, n)
        offsets = mx.array(context_lengths)
        pre = x
        for i in range(self.layers):
            q, k, v = self._compiled("pre", i)(x, offsets)
            keys = mx.concatenate([mx.broadcast_to(keys_ctx[i][None], (B, *keys_ctx[i].shape)), k], axis=2)
            values = mx.concatenate([mx.broadcast_to(values_ctx[i][None], (B, *values_ctx[i].shape)), v], axis=2)
            out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.head_dim ** -0.5, mask=mask)
            x, pre = self._compiled("post", i)(x, out, anchor_features)
        return rms_norm(x, self.p("norm.weight"), self.eps), pre

    def _layer_pre(self, i, x, offsets):
        """Fixed-shape part of layer i before the attention: q (B, heads, n, d), k and v (B, kv, n, d)."""
        B, n = x.shape[:2]
        prefix = f"layers.{i}."
        h = rms_norm(x, self.p(prefix + "input_layernorm.weight"), self.eps)
        q_size, kv_size = self.heads * self.head_dim, self.kv_heads * self.head_dim
        qkv = self.mm(h, prefix + "self_attn.qkv_proj.weight")
        q = qkv[..., :q_size].reshape(B, n, self.heads, self.head_dim)
        k = qkv[..., q_size:q_size + kv_size].reshape(B, n, self.kv_heads, self.head_dim)
        v = qkv[..., q_size + kv_size:].reshape(B, n, self.kv_heads, self.head_dim)
        q = rms_norm(q, self.p(prefix + "self_attn.q_norm.weight"), self.eps).transpose(0, 2, 1, 3)
        k = rms_norm(k, self.p(prefix + "self_attn.k_norm.weight"), self.eps).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offsets)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offsets)
        return q, k, v.transpose(0, 2, 1, 3)

    def _layer_post(self, i, x, out, anchor_features):
        """Fixed-shape part of layer i after the attention: output projection, MLP, injection."""
        B, n = x.shape[:2]
        prefix = f"layers.{i}."
        x = x + self.mm(out.transpose(0, 2, 1, 3).reshape(B, n, -1), prefix + "self_attn.o_proj.weight")
        h = rms_norm(x, self.p(prefix + "post_attention_layernorm.weight"), self.eps)
        gate, up = mx.split(self.mm(h, prefix + "mlp.gate_up_proj.weight"), 2, axis=-1)
        pre = x + self.mm(gate * mx.sigmoid(gate) * up, prefix + "mlp.down_proj.weight")
        return self.inject(i, pre, anchor_features), pre

    def _compiled(self, part, i):
        """mx.compile of a fixed-shape layer part (fuses its element-wise operations); plain when COMPILE is off."""
        function = getattr(self, f"_layer_{part}")
        if not COMPILE:
            return lambda *args: function(i, *args)
        cache = self.__dict__.setdefault("_compiled_parts", {})
        if (part, i) not in cache:
            cache[(part, i)] = mx.compile(lambda *args, _f=function, _i=i: _f(_i, *args))
        return cache[(part, i)]

    # --- candidates and walk -----------------------------------------------------------
    def candidates(self, hidden):
        """Top-k candidate ids (B, steps, k) of the drafter's head at the mask positions."""
        logits = self.head_logits(hidden[:, 1:]).astype(mx.float32)
        # the last top_k of an ascending partition: no negated copy of the vocabulary-wide logits
        return mx.argpartition(logits, kth=logits.shape[-1] - self.top_k, axis=-1)[..., -self.top_k:]

    def transition_scores(self, candidate_ids, final_pre, anchors):
        """scores (B, steps, k_prev, k): exact fp32 last write with each candidate predecessor."""
        B, steps, k = candidate_ids.shape
        anchor_ids = mx.broadcast_to(mx.array(anchors)[:, None, None], (B, 1, k))
        predecessor_ids = mx.concatenate([anchor_ids, candidate_ids[:, :-1]], axis=1)
        receiver = final_pre[:, 1:]  # (B, steps, H)
        layer = self.layers - 1
        receiver_features = self.project(layer, receiver)[:, :, None]
        exact = self.p("transfer_codebook")[predecessor_ids].astype(mx.float32) * math.sqrt(self.transfer_dim)
        refined = self.write(layer, receiver[:, :, None], receiver_features, exact, refine=True)  # (B, s, k, H)
        norm = self.p("norm.weight").astype(mx.float32)
        normalized = refined * mx.rsqrt(mx.mean(refined * refined, axis=-1, keepdims=True) + self.eps) * norm
        successor = self.head_rows(candidate_ids)  # (B, s, k, H)
        return normalized @ successor.swapaxes(-1, -2)  # (B, s, k_prev, k)

    @staticmethod
    def walk(candidate_ids, scores, temperature=0.0):
        """One candidate per position, following the chosen predecessor; returns (tokens, draft log-probs).

        Greedy at temperature 0; otherwise a draw from softmax(scores / T) over the
        candidates (the draft distribution q that the rejection sampling checks).
        """
        candidate_ids = np.array(candidate_ids)
        scores = np.array(scores.astype(mx.float32), dtype=np.float64)
        B, steps, k = candidate_ids.shape
        tokens = np.zeros((B, steps), dtype=np.int64)
        log_q = np.zeros((B, steps, k))
        previous = np.zeros(B, dtype=np.int64)
        rows = np.arange(B)
        for step in range(steps):
            row = scores[rows, step, previous]  # (B, k)
            if temperature > 0:
                z = row / temperature
                z = z - z.max(-1, keepdims=True)
                log_q[:, step] = z - np.log(np.exp(z).sum(-1, keepdims=True))
                index = np.array([np.random.choice(k, p=np.exp(lq)) for lq in log_q[:, step]])
            else:
                index = row.argmax(-1)
            tokens[:, step] = candidate_ids[rows, step, index]
            previous = index
        return tokens, log_q

    def draft(self, anchors, context_lengths, context, steps, temperature=0.0):
        hidden, final_pre = self.forward(anchors, context_lengths, context, steps)
        candidate_ids = self.candidates(hidden)
        scores = self.transition_scores(candidate_ids, final_pre, anchors)
        mx.eval(candidate_ids, scores)
        return self.walk(candidate_ids, scores, temperature)


COMPILE = True  # mx.compile the fixed-shape parts of the draft layers

_SINKS = {}  # id(layer) -> list filled by the patched __call__ (MLX modules hold no plain attributes)


def capture_aux(model):
    """Patch the Olala layers so a forward records the residual stream after TARGET_LAYERS.

    Returns a list that each forward fills with (B, L, 1536) arrays, in layer order;
    the caller clears it. The fused single-token decode (metal.decode) bypasses the
    layers: only multi-token forwards are captured. Calling it again for the same model
    returns the same list (one list per model, shared by its users).
    """
    existing = _SINKS.get(id(model.model.layers[TARGET_LAYERS[0]]))
    if existing is not None:
        return existing
    records = []
    for index in TARGET_LAYERS:
        layer = model.model.layers[index]
        cls = type(layer)
        if not getattr(cls, "_telegraphe_patched", False):
            original = cls.__call__

            def __call__(self, *args, _original=original, **kwargs):
                out = _original(self, *args, **kwargs)
                sink = _SINKS.get(id(self))
                if sink is not None:
                    sink.append(out)
                return out

            cls.__call__ = __call__
            cls._telegraphe_patched = True
        _SINKS[id(layer)] = records
    return records
