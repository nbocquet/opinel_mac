import unittest

from scripts.check_logit_parity import compare, log_softmax, parity_metrics, top_entries

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy non installé")
class LogitParityTests(unittest.TestCase):
    def tops(self, logits, count):
        return top_entries(log_softmax(np.array(logits, dtype=float)), count)

    def test_identical_distributions_agree_with_zero_delta(self):
        index, values = self.tops([[5., 3., 2., 1., 0., -1., -2.]], 6)
        reference = self.tops([[105., 103., 102., 101., 100., 99., 98.]], 5)  # same law, shifted logits
        measured = parity_metrics(index, values, *reference, np.array([True]))
        self.assertEqual(measured["top1_agreement"], 1.)
        self.assertAlmostEqual(measured["top5_mean_abs_log_probability_delta"], 0., places=12)
        self.assertEqual((measured["clear_errors"], measured["reference_top5_outside_candidate_top_k"]), (0, 0))

    def test_delta_uses_only_the_reference_top5_whatever_the_tail(self):
        logits = np.array([[4., 3., 2., 1., 0., -30., -40.]])
        tail_changed = logits.copy()
        tail_changed[0, 5:] = [-10., -12.]  # visible on the full vocabulary, not on the top-5
        index, values = self.tops(tail_changed, 7)
        measured = parity_metrics(index, values, *self.tops(logits, 5), np.array([True]))
        self.assertLess(measured["top5_mean_abs_log_probability_delta"], 1e-4)

    def test_disagreement_on_a_confident_reference_is_a_clear_error(self):
        reference = self.tops([[6., 0., 0., 0., 0., 0.]], 5)   # p(top-1) > 0.5
        candidate = self.tops([[0., 6., 0., 0., 0., 0.]], 6)
        measured = parity_metrics(*candidate, *reference, np.array([True]))
        self.assertEqual((measured["top1_agreement"], measured["clear_errors"]), (0., 1))
        uncertain = self.tops([[0.2, 0., 0.1, 0., 0., 0.]], 5)
        self.assertEqual(parity_metrics(*candidate, *uncertain, np.array([True]))["clear_errors"], 0)

    def test_missing_tokens_and_padding_are_handled(self):
        reference = self.tops([[5., 4., 3., 2., 1., 0.]], 5)
        candidate = self.tops([[5., 4., 3., 2., 1., 0.]], 3)  # top-3 only: two tokens missing
        measured = parity_metrics(*candidate, *reference, np.array([True]))
        self.assertEqual(measured["reference_top5_outside_candidate_top_k"], 2)
        padded = (np.concatenate([candidate[0], candidate[0]]), np.concatenate([candidate[1], np.full((1, 3), -np.inf)]))
        both = (np.concatenate([reference[0], reference[0]]), np.concatenate([reference[1], reference[1]]))
        self.assertEqual(parity_metrics(*padded, *both, np.array([True, False]))["positions"], 1)
        with self.assertRaises(ValueError):
            parity_metrics(*padded, *both, np.array([True, True]))
        with self.assertRaises(ValueError):
            parity_metrics(*candidate, *reference, np.array([False]))

    def test_comparison_separates_long_prompts(self):
        index, values = self.tops([[3., 2., 1., 0., -1., -2.]] * 4, 6)
        reference = self.tops([[3., 2., 1., 0., -1., -2.]] * 4, 5)
        candidate = {"continuation": np.array([[1, 1], [1, -1]]), "candidate_index": index.reshape(2, 2, 6),
                     "candidate_lp": values.reshape(2, 2, 6)}
        generation = {"steps": 2, "backend": "mlx", "prompts": [{"id": "p01"}, {"id": "long_readme"}]}
        result = compare(generation, candidate, {"backend": "torch"},
                         {"reference_index": reference[0].reshape(2, 2, 5),
                          "reference_lp": reference[1].reshape(2, 2, 5)})
        self.assertEqual((result["tous"]["positions"], result["courts"]["positions"], result["longs"]["positions"]),
                         (3, 2, 1))
