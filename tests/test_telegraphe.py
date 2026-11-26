import io
import math
import tempfile
import unittest
from pathlib import Path
import zipfile

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None


def tiny_drafter(seed=0, layers=2, hidden=32, heads=4, kv_heads=2, head_dim=8, vocab=50, transfer=8, message=4,
                 top_k=4, window=None):
    """A Telegraphe with random weights and the checkpoint's structure, scaled down."""
    from opinel_telegraphe import Telegraphe

    rng = np.random.default_rng(seed)

    def w(*shape, scale=0.2):
        return mx.array(rng.normal(0, scale, shape).astype(np.float32)).astype(mx.bfloat16)

    d = Telegraphe.__new__(Telegraphe)
    d.block_size, d.mask_token_id, d.transfer_dim, d.top_k = 8, vocab - 1, transfer, top_k
    d.transfer_eps, d.eps, d.layers, d.heads, d.kv_heads, d.head_dim = 1e-6, 1e-6, layers, heads, kv_heads, head_dim
    d.rope_theta, d.window = 1e6, window
    d.quantized, d.target_head = {}, None
    weights = {"fc.weight": w(hidden, 5 * hidden), "hidden_norm.weight": w(hidden) + 1, "norm.weight": w(hidden) + 1,
               "lm_head.weight": w(vocab, hidden), "transfer_codebook": w(vocab, transfer),
               "injection.message.weight": w(message, transfer), "injection.receiver_gate.weight": w(message, transfer),
               "injection.predecessor_gate.weight": w(message, transfer), "injection.gate_bias": w(message)}
    for i in range(layers):
        p = f"layers.{i}."
        weights.update({
            p + "input_layernorm.weight": w(hidden) + 1, p + "post_attention_layernorm.weight": w(hidden) + 1,
            p + "self_attn.q_proj.weight": w(heads * head_dim, hidden),
            p + "self_attn.k_proj.weight": w(kv_heads * head_dim, hidden),
            p + "self_attn.v_proj.weight": w(kv_heads * head_dim, hidden),
            p + "self_attn.o_proj.weight": w(hidden, heads * head_dim),
            p + "self_attn.q_norm.weight": w(head_dim) + 1, p + "self_attn.k_norm.weight": w(head_dim) + 1,
            p + "mlp.gate_proj.weight": w(2 * hidden, hidden), p + "mlp.up_proj.weight": w(2 * hidden, hidden),
            p + "mlp.down_proj.weight": w(hidden, 2 * hidden),
            f"injection.readouts.{i}.weight": w(transfer, hidden), f"injection.writes.{i}.weight": w(hidden, message),
        })
    d.w = weights
    d.fuse_projections()
    d.embedding = w(vocab, hidden, scale=1.0)
    d.mask_embedding = w(hidden, scale=1.0)
    return d


@unittest.skipIf(mx is None, "MLX absent de cet environnement")
class TelegrapheTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)

    def test_walk_follows_the_chosen_predecessor(self):
        from opinel_telegraphe import Telegraphe
        candidates = mx.array([[[10, 11], [20, 21]]])
        scores = mx.array([[[[0.0, 1.0], [0.0, 1.0]],       # step 0: candidate 1 (11)
                            [[5.0, 0.0], [0.0, 5.0]]]])     # step 1 depends on the predecessor: row 1 -> 21
        tokens, _ = Telegraphe.walk(candidates, scores)
        self.assertEqual(tokens.tolist(), [[11, 21]])

    def test_sampled_walk_returns_the_draft_distribution(self):
        from opinel_telegraphe import Telegraphe
        np.random.seed(0)
        candidates = mx.array([[[1, 2, 3]]])
        scores = mx.array([[[[0.0, math.log(2.0), math.log(5.0)]] * 3]])
        _, log_q = Telegraphe.walk(candidates, scores, temperature=1.0)
        np.testing.assert_allclose(np.exp(log_q[0, 0]), [0.125, 0.25, 0.625], rtol=1e-6)

    def test_attention_sees_earlier_context_inside_the_window_and_a_causal_block(self):
        d = tiny_drafter(window=3)
        mask = np.array(d.attention_mask([2, 4], context_size=4, n=2))[:, 0]
        # Block 0: context rows 0-1; the query at position 3 loses row 0 (window 3 keeps position - key < 3).
        np.testing.assert_array_equal(mask[0], [[1, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1]])
        # Block 1: queries at 4 and 5 see rows 2-3, then row 3 only.
        np.testing.assert_array_equal(mask[1], [[0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1]])

    def test_without_a_window_every_earlier_context_row_is_visible(self):
        d = tiny_drafter()
        mask = np.array(d.attention_mask([3], context_size=4, n=2))[0, 0]
        np.testing.assert_array_equal(mask, [[1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1]])

    def test_injection_keeps_the_anchor_and_links_each_position_to_the_previous_one(self):
        d = tiny_drafter()
        states = mx.random.normal((2, 4, 32), key=mx.random.key(1)).astype(mx.bfloat16)
        anchor = d.token_features(mx.array([3, 7]))
        out = d.inject(0, states, anchor)
        np.testing.assert_array_equal(np.array(out[:, 0].astype(mx.float32)), np.array(states[:, 0].astype(mx.float32)))
        features = d.project(0, states)
        want = d.write(0, states[:, 2:3], features[:, 2:3], features[:, 1:2])
        np.testing.assert_allclose(np.array(out[:, 2:3].astype(mx.float32)), np.array(want.astype(mx.float32)))
        first = d.write(0, states[:, 1:2], features[:, 1:2], anchor[:, None])
        np.testing.assert_allclose(np.array(out[:, 1:2].astype(mx.float32)), np.array(first.astype(mx.float32)))

    def test_a_shorter_block_is_an_exact_prefix_of_a_longer_one(self):
        d = tiny_drafter(window=6)
        aux = mx.random.normal((9, 5 * 32), key=mx.random.key(2))
        context = d.context_kv(aux)
        long, _ = d.draft([5, 8], [9, 6], context, steps=7)
        short, _ = d.draft([5, 8], [9, 6], context, steps=3)
        np.testing.assert_array_equal(long[:, :3], short)

    def test_a_block_ignores_context_rows_past_its_anchor(self):
        d = tiny_drafter()
        aux = mx.random.normal((8, 5 * 32), key=mx.random.key(3))
        changed = mx.concatenate([aux[:5], aux[5:] + 3.0])
        a = d.forward([4], [5], d.context_kv(aux), steps=3)[0]
        b = d.forward([4], [5], d.context_kv(changed), steps=3)[0]
        np.testing.assert_array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))

    def test_mask_embedding_is_read_from_the_raw_storage(self):
        from opinel_telegraphe import mask_embedding_from_archive
        values = np.linspace(-1, 1, 1536).astype(np.float32)
        bf16 = (values.view(np.uint32) >> 16).astype("<u2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask_embedding.pt"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("mask_embedding/data.pkl", b"never unpickled")
                archive.writestr("mask_embedding/data/0", bf16.tobytes())
            got = mask_embedding_from_archive(path)
        np.testing.assert_allclose(got, values, atol=1e-2)
        self.assertEqual(got.dtype, np.float32)


@unittest.skipIf(mx is None, "MLX absent de cet environnement")
class DraftContextTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)

    def test_in_place_context_drafts_like_the_concatenated_one(self):
        from opinel_telegraphe import DraftContext
        d = tiny_drafter()
        aux = mx.array(np.random.default_rng(3).normal(0, 1, (11, 5 * 32)).astype(np.float32)).astype(mx.bfloat16)
        keys, values = d.context_kv(aux[:7])
        k2, v2 = d.context_kv(aux[7:], start=7)
        concatenated = ([mx.concatenate([a, b], axis=1) for a, b in zip(keys, k2)],
                        [mx.concatenate([a, b], axis=1) for a, b in zip(values, v2)])
        context = DraftContext(d.layers, d.kv_heads, d.head_dim, step=8).append(keys, values).append(k2, v2)
        self.assertEqual((context.length, context.capacity), (11, 16))
        expected, _ = d.forward([3], [11], concatenated, 3)
        got, _ = d.forward([3], [11], context, 3)
        np.testing.assert_allclose(np.array(got.astype(mx.float32)), np.array(expected.astype(mx.float32)),
                                   atol=1e-2)

    def test_copy_is_not_changed_by_later_writes(self):
        from opinel_telegraphe import DraftContext
        context = DraftContext(1, 1, 2, step=4)
        context.append([mx.ones((1, 2, 2))], [mx.ones((1, 2, 2))])
        twin = context.copy()
        context.append([mx.full((1, 2, 2), 5.0)], [mx.full((1, 2, 2), 5.0)])
        self.assertEqual(twin.length, 2)
        self.assertEqual(float(twin.keys[0][:, 2:4].sum()), 0.0)
