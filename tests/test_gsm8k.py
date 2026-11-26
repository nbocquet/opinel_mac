import unittest

from scripts.evaluate_gsm8k import gold_answer, is_correct, last_number, load_questions, summarize


class Gsm8kScoringTests(unittest.TestCase):
    def test_gold_answer_follows_the_final_marker(self):
        self.assertEqual(gold_answer("9 * 2 = <<9*2=18>>18\n#### 18"), 18)
        self.assertEqual(gold_answer("...\n#### 1,250"), 1250)
        self.assertEqual(gold_answer("...\n#### -3"), -3)

    def test_prediction_is_the_last_number_of_the_final_answer(self):
        self.assertEqual(last_number("16 - 3 - 4 = 9 eggs, 9 × 2 = 18.\n**18**"), 18)
        self.assertEqual(last_number("She pays $1,250.00 in total"), 1250)
        self.assertEqual(last_number("The answer is 3.5."), 3.5)
        self.assertIsNone(last_number("No number here"))
        self.assertIsNone(last_number(None))

    def test_numeric_comparison_ignores_formatting_only(self):
        self.assertTrue(is_correct(18.0, 18))
        self.assertFalse(is_correct(18.5, 18))
        self.assertFalse(is_correct(None, 18))

    def test_summary_flags_a_gap_beyond_three_points_at_full_size(self):
        results = [{"correct": index >= 30, "error": None, "seconds": 1.0} for index in range(300)]
        summary = summarize({"configuration": {"reasoning_effort": "low"}, "results": results})
        self.assertEqual((summary["correct"], summary["accuracy"]), (270, 0.9))
        self.assertAlmostEqual(summary["standard_error"], 0.0173, places=4)
        self.assertEqual(summary["gap_points"], -4.3)
        self.assertTrue(summary["investigate"])
        small = summarize({"configuration": {"reasoning_effort": "low"}, "results": results[:10]})
        self.assertFalse(small["investigate"])

    def test_pinned_dataset_is_verified_when_present(self):
        try:
            questions = load_questions()
        except OSError:
            self.skipTest("Jeu GSM8K absent et réseau indisponible")
        self.assertEqual(len(questions), 1319)
        self.assertEqual(gold_answer(questions[0]["answer"]), 18)
