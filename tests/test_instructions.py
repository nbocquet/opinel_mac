from pathlib import Path
import unittest

from instructions import PROFILES, describe, resolve, with_instructions
from opinel_mac import parse_args, prepare_turn
from opinel_mlx import prepare_mlx_turn
from reliability import check_answer, load_suite

ROOT = Path(__file__).resolve().parents[1]


class RecordingTokenizer:
    """Records the messages handed to the chat template; one token per character."""

    def __init__(self):
        self.rendered = []

    def apply_chat_template(self, messages, **kwargs):
        self.rendered.append(messages)
        return "".join(message["content"] for message in messages)

    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def __call__(self, text, **kwargs):
        import torch

        class Batch(dict):
            input_ids = torch.zeros((1, len(text)), dtype=torch.long)

            def to(self, device):
                return self
        return Batch()


class InstructionProfileTests(unittest.TestCase):
    def test_default_sends_nothing_and_keeps_report_identity(self):
        self.assertEqual(parse_args(["chat"]).instructions, "aucune")
        self.assertEqual(with_instructions([{"role": "user", "content": "x"}], resolve("aucune")),
                         [{"role": "user", "content": "x"}])
        self.assertIsNone(describe("aucune"))

    def test_profile_is_identified_by_content_hash(self):
        identity = describe("opinel-fr-v1")
        self.assertEqual(identity["profile"], "opinel-fr-v1")
        self.assertEqual(len(identity["sha256"]), 64)

    def test_unknown_profile_is_refused(self):
        with self.assertRaises(ValueError):
            resolve("inconnu")

    def test_existing_system_message_is_not_silently_replaced(self):
        with self.assertRaises(ValueError):
            with_instructions([{"role": "system", "content": "a"}, {"role": "user", "content": "b"}], "c")

    def test_profiles_are_french_and_do_not_quote_evaluation_answers(self):
        text = PROFILES["opinel-fr-v1"]
        for leaked in ("Reblochon", "Beaufort", "Gruyère", "kelvin", "Chambéry", "Canberra"):
            self.assertNotIn(leaked, text)


class InjectionTests(unittest.TestCase):
    messages = [{"role": "user", "content": "a" * 600},
                {"role": "assistant", "content": "b" * 600},
                {"role": "user", "content": "nouvelle"}]

    def test_mlx_developer_message_survives_history_trimming_and_stays_out_of_history(self):
        tokenizer = RecordingTokenizer()
        text = resolve("opinel-fr-v1")
        _, retained = prepare_mlx_turn(tokenizer, self.messages, "low", trim_history=True,
                                       context_length=1024 + len(text) + 900, max_new_tokens=1024,
                                       instructions=text)
        self.assertEqual(retained, self.messages[-1:])
        self.assertTrue(all(rendered[0] == {"role": "developer", "content": text}
                            for rendered in tokenizer.rendered))
        self.assertNotIn("developer", [message["role"] for message in retained])

    def test_torch_path_renders_developer_message_first(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch absent")
        tokenizer = RecordingTokenizer()
        _, retained = prepare_turn(tokenizer, self.messages[-1:], "cpu", "low", instructions="consigne")
        self.assertEqual(tokenizer.rendered[-1][0], {"role": "developer", "content": "consigne"})
        self.assertEqual(retained, self.messages[-1:])


class HoldoutSuiteTests(unittest.TestCase):
    def test_holdout_is_valid_distinct_and_covers_register_requests(self):
        holdout = load_suite(ROOT / "evaluation" / "reliability_fr_holdout.json")
        main = load_suite(ROOT / "evaluation" / "reliability_fr.json")
        self.assertNotEqual(holdout["name"], main["name"])
        self.assertFalse({case["id"] for case in holdout["cases"]} & {case["id"] for case in main["cases"]})
        # A short request may recur on purpose, but never within the same conversation.
        conversation = lambda case: tuple(message["content"] for message in case["messages"])
        self.assertFalse({conversation(case) for case in holdout["cases"]}
                         & {conversation(case) for case in main["cases"]})
        register = [case for case in holdout["cases"] if case["id"].startswith("hi0") and "check" not in case]
        self.assertGreaterEqual(len(register), 4)

    def test_holdout_exact_checks_accept_their_expected_answer(self):
        holdout = load_suite(ROOT / "evaluation" / "reliability_fr_holdout.json")
        for case in holdout["cases"]:
            check = case.get("check")
            if check and check["type"] == "exact":
                with self.subTest(case=case["id"]):
                    self.assertTrue(check_answer(case, check["expected"] + ".")["check_passed"])


if __name__ == "__main__":
    unittest.main()
