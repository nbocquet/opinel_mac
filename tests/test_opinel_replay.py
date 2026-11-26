import re
import unittest

try:
    import mlx.core as mx
except ImportError:  # PyTorch environment
    mx = None


@unittest.skipIf(mx is None, "MLX absent de cet environnement")
class ReplaySourceTests(unittest.TestCase):
    def test_grouped_replay_source_keeps_m3_prefill_and_only_changes_its_edges(self):
        from opinel_replay import replay_source
        src = replay_source(count=3, keep=2, H=48, P=64, N=128, R=4, A=32, n_static=6288, prm_size=1000)
        self.assertIn("const int L = 2;", src)
        self.assertIn("const int s0 = 0, s1 = L;", src)
        self.assertNotIn("rng[", src)
        self.assertNotIn("logA[", src)
        self.assertNotIn("proj_shape", src)
        self.assertEqual([int(c) for c in re.findall(r"case (\d+): sin_ = st\1 ", src)], [0, 1, 2])
        definition, rest = src.split("const device float* prm_l = prm + (size_t)b * 1000;")
        self.assertIsNone(re.search(r"\bprm\b", rest))  # every parameter read goes to the layer's pack
        self.assertIn("prm_l[O_DT + h]", rest)
        self.assertIn("srow[i] = alpha * srow[i] + beta * kvprev[i] + gamma * kv;", src)  # the recurrence, as is


if __name__ == "__main__":
    unittest.main()
