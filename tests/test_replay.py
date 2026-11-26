import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from scripts.replay_conversation import DEFAULT_CONVERSATION, compare, replay


class ReplayTests(unittest.TestCase):
    def test_answers_enter_history_and_failed_turns_do_not(self):
        seen = []

        def generate(messages):
            seen.append([m["content"] for m in messages])
            if messages[-1]["content"] == "boom":
                raise ValueError("Limite atteinte")
            return f"réponse à {messages[-1]['content']}", messages

        turns = replay(generate, ["a", "boom", "b"])
        self.assertEqual([t["answer"] for t in turns], ["réponse à a", None, "réponse à b"])
        self.assertEqual(turns[1]["error"], "Limite atteinte")
        self.assertEqual(seen[2], ["a", "réponse à a", "b"])

    def test_reference_conversation_is_well_formed(self):
        conversation = json.loads(DEFAULT_CONVERSATION.read_text())
        self.assertEqual(len(conversation["messages"]), 9)
        self.assertIn("Haute-Savoie", conversation["attendu"])

    def test_comparison_prints_each_variant_per_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for label, answer in (("bf16-greedy", "Oui"), ("n8-greedy", None)):
                path = Path(directory) / f"{label}.json"
                path.write_text(json.dumps({"label": label, "format": "f", "temperature": 0.0, "expected": "X",
                                            "turns": [{"user": "q", "answer": answer, "error": "e"}]}))
                paths.append(path)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                compare(SimpleNamespace(reports=paths))
        text = output.getvalue()
        self.assertIn("Attendu : X", text)
        self.assertIn("--- bf16-greedy", text)
        self.assertIn("(échec : e)", text)
