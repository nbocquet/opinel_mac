from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from opinel_mac import ROOT
from reliability import (annotate_report, check_answer, compare_reports, load_suite,
                         run_evaluation, summarize, verify_assets)


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "report.json"
        self.suite = {"name": "fixture", "sha256": "fixture-hash", "cases": [
            {"id": "exact", "category": "reasoning", "messages": [{"role": "user", "content": "2 + 2 ?"}],
             "rubric": "Répondre 4.", "check": {"type": "exact", "expected": "4", "scope": "full"}},
            {"id": "review", "category": "uncertainty", "messages": [{"role": "user", "content": "Mon prénom ?"}],
             "rubric": "Ne pas inventer."},
        ]}
        self.config = {"seed": 7, "backend": "torch", "preset": "baseline"}

    def run_fixture(self, generate=None, **kwargs):
        return run_evaluation(self.suite, self.config, self.output, generate or (lambda messages, seed: "4"),
                              progress=lambda line: None, **kwargs)

    def test_suite_has_thirty_balanced_cases_and_valid_conversations(self):
        suite = load_suite(ROOT / "evaluation" / "reliability_fr.json")
        self.assertEqual(len(suite["cases"]), 30)
        self.assertEqual(set(Counter(case["category"] for case in suite["cases"]).values()), {6})
        self.assertEqual(len({case["id"] for case in suite["cases"]}), 30)

    def test_reference_assets_are_checked_instead_of_trusting_a_boolean(self):
        folder = Path(self.temporary.name)
        data = b"verified weights"
        (folder / "model").write_bytes(data)
        manifest = {"files": {"model": {"size": len(data), "algorithm": "sha256",
                                        "digest": hashlib.sha256(data).hexdigest()}}}
        verify_assets(folder, manifest)
        (folder / "model").write_bytes(b"different bytes!")
        with self.assertRaises(ValueError):
            verify_assets(folder, manifest)

    def test_json_check_rejects_boolean_in_place_of_number_and_extra_text(self):
        case = {"check": {"type": "json", "scope": "full", "expected": {"n": 1}}}
        self.assertEqual(check_answer(case, '{"n": true}')["verdict"], "fail")
        self.assertEqual(check_answer(case, '```json\n{"n": 1}\n```')["verdict"], "fail")
        self.assertEqual(check_answer(case, '{"n": 1}')["verdict"], "pass")

    def test_keyword_mention_is_not_used_to_grade_freeform_facts(self):
        self.assertEqual(check_answer({"rubric": "Beaufort"}, "Le Beaufort est interdit.")["verdict"], "unreviewed")

    def test_format_pass_does_not_become_content_pass(self):
        case = {"check": {"type": "max_words", "scope": "format", "expected": 10}}
        self.assertEqual(check_answer(case, "Un texte court mais faux.")["verdict"], "unreviewed")

    def test_rubric_is_not_sent_to_generator_and_mutation_cannot_change_record(self):
        def generate(messages, seed):
            self.assertEqual(set(messages[0]), {"role", "content"})
            messages.clear()
            return "4"
        report = self.run_fixture(generate)
        self.assertEqual(report["results"][0]["messages"], self.suite["cases"][0]["messages"])
        self.assertEqual(report["summary"]["pass"], 1)
        self.assertEqual(report["summary"]["unreviewed"], 1)
        self.assertIsNone(report["summary"]["score"])

    def test_interrupt_and_resume_keep_completed_cases_without_duplication(self):
        calls = []

        def interrupted(messages, seed):
            calls.append(seed)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return "4"
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(interrupted)
        stored = json.loads(self.output.read_text())
        self.assertEqual(len(stored["results"]), 1)
        self.assertEqual(stored["summary"]["not_run"], 1)
        report = self.run_fixture(lambda messages, seed: calls.append(seed) or "Je ne sais pas.", resume=True)
        self.assertEqual(calls, [7, 8, 8])
        self.assertEqual(len(report["results"]), 2)

    def test_selection_uses_the_same_case_seed_as_a_full_run(self):
        seeds = []
        self.run_fixture(lambda messages, seed: seeds.append(seed) or "unknown", case_ids=["review"])
        self.assertEqual(seeds, [8])

    def test_resume_refuses_changed_configuration(self):
        self.run_fixture()
        self.config["seed"] = 8
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.run_fixture(resume=True)

    def test_existing_report_is_not_overwritten_without_resume(self):
        self.run_fixture()
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "existant"):
            self.run_fixture()
        self.assertEqual(self.output.read_bytes(), before)

    def test_annotations_are_attributed_and_score_waits_for_all_reviews(self):
        self.run_fixture()
        report = annotate_report(self.output, "review", "uncertain", "À examiner", "Codex", "assistant")
        self.assertIsNone(report["summary"]["score"])
        report = annotate_report(self.output, "review", "fail", "Le modèle invente un prénom.", "Codex", "assistant")
        self.assertEqual(report["summary"]["score"], 0.5)
        self.assertEqual(report["summary"]["reviewers"], {"assistant": 1})
        self.assertEqual(report["results"][1]["review_history"][0]["verdict"], "uncertain")

    def test_generation_error_is_distinct_from_factually_wrong_answer(self):
        def failure(messages, seed):
            raise ValueError("Réponse tronquée")
        report = self.run_fixture(failure)
        self.assertEqual(report["summary"]["error"], 2)
        self.assertEqual(report["summary"]["fail"], 0)
        with self.assertRaises(ValueError):
            annotate_report(self.output, "review", "pass", "ok", "user", "human")

    def test_comparison_refuses_different_questions_and_exposes_parameter_changes(self):
        first = self.run_fixture()
        other = copy.deepcopy(first)
        other["configuration"]["preset"] = "ovh"
        compared = compare_reports([first, other])
        self.assertEqual(compared["configuration_differences"]["preset"], ["baseline", "ovh"])
        self.assertEqual(len(compared["paired_cases"]), 2)
        other["case_ids"] = ["exact"]
        with self.assertRaisesRegex(ValueError, "non comparables"):
            compare_reports([first, other])

    def test_duplicate_results_cannot_inflate_the_score(self):
        report = self.run_fixture()
        report["results"].append(report["results"][0])
        with self.assertRaisesRegex(ValueError, "dupliqués"):
            summarize(report)
