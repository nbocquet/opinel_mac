import unittest

import numpy as np

try:
    import mlx.core as mx
except ImportError:  # PyTorch environment
    mx = None


@unittest.skipIf(mx is None, "MLX absent de cet environnement")
class RejectionStepTests(unittest.TestCase):
    """The first emitted token must follow the target law p whatever the draft law q."""

    def histogram(self, q_ids, q, law_ids, p, draws=60000, seed=0):
        from opinel_speculative import rejection_step
        rng = np.random.default_rng(seed)
        log_q = np.log(np.array([q]))
        candidates = np.array([q_ids])
        law_ids = np.array([law_ids, law_ids])
        law_probs = np.array([p, p])
        counts = {}
        for _ in range(draws):
            draft = candidates[0][rng.choice(len(q), p=q)]
            accepted, token = rejection_step(np.array([draft]), candidates, log_q, law_ids, law_probs, rng)
            first = int(draft) if accepted == 1 else token
            counts[first] = counts.get(first, 0) + 1
        return {k: v / draws for k, v in counts.items()}

    def test_output_follows_the_target_law_when_the_draft_differs(self):
        p = [0.5, 0.3, 0.2, 0.0]
        got = self.histogram([10, 11, 12, 13], [0.1, 0.2, 0.3, 0.4], [10, 11, 12, 13], p)
        for token, expected in zip([10, 11, 12, 13], p):
            self.assertAlmostEqual(got.get(token, 0.0), expected, delta=0.01)

    def test_draft_candidates_outside_the_target_support_are_never_kept(self):
        # q proposes 20 and 21, which p excludes (top-k / top-p cut): only p's tokens come out.
        got = self.histogram([20, 21, 10], [0.4, 0.4, 0.2], [10, 11], [0.25, 0.75])
        self.assertEqual(set(got), {10, 11})
        self.assertAlmostEqual(got[10], 0.25, delta=0.01)

    def test_identical_laws_keep_every_draft(self):
        from opinel_speculative import rejection_step
        rng = np.random.default_rng(1)
        ids = np.array([[5, 6], [5, 6], [5, 6]])
        probs = np.array([[0.6, 0.4]] * 3)
        accepted, _ = rejection_step(np.array([5, 6]), ids[:2], np.log(probs[:2]), ids, probs, rng)
        self.assertEqual(accepted, 2)


if __name__ == "__main__":
    unittest.main()
