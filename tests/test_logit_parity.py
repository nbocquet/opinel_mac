import unittest

from scripts.check_logit_parity import metrics

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy non installé")
class LogitParityTests(unittest.TestCase):
    def test_constant_logit_shift_does_not_change_distribution(self):
        reference = np.array([[1., 2., 3.], [3., 0., -1.]])
        measured = metrics(reference, reference + 100.)
        self.assertEqual(measured["top1_agreement"], 1.)
        self.assertAlmostEqual(measured["mean_kl_nats"], 0., places=12)

    def test_changed_ranking_produces_measurable_divergence(self):
        measured = metrics(np.array([[5., 0.]]), np.array([[0., 5.]]))
        self.assertEqual(measured["top1_agreement"], 0.)
        self.assertGreater(measured["mean_kl_nats"], 1.)

    def test_nonfinite_values_and_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics(np.array([[float("nan")]]), np.array([[1.]]))
        with self.assertRaises(ValueError):
            metrics(np.zeros((2, 3)), np.zeros((3, 2)))
