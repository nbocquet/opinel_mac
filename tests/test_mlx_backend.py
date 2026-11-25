import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from opinel_mac import MODEL, parse_args
from opinel_mlx import collect_response, prepare_mlx_turn, validate_provenance


class CountingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "".join(message["content"] for message in messages)

    def encode(self, text, **kwargs):
        return list(range(len(text)))


class MlxContractTests(unittest.TestCase):
    def test_backend_is_explicit_and_does_not_change_torch_default(self):
        self.assertEqual(parse_args(["chat"]).backend, "torch")
        self.assertEqual(parse_args(["chat", "--backend", "mlx"]).backend, "mlx")

    def test_default_context_and_explicit_override(self):
        self.assertEqual(parse_args(["chat", "--backend", "mlx"]).context_length, 8192)
        self.assertEqual(parse_args(["chat"]).context_length, 2048)
        self.assertEqual(parse_args(["chat", "--backend", "mlx", "--context-length", "4096"]).context_length, 4096)

    def test_context_must_leave_room_for_the_prompt(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["chat", "--backend", "mlx", "--context-length", "1024"])

    def test_default_context_retains_history_longer_than_the_old_limit(self):
        messages = [{"role": "user", "content": "x" * 600},
                    {"role": "assistant", "content": "y" * 600},
                    {"role": "user", "content": "new"}]
        tokens, retained = prepare_mlx_turn(CountingTokenizer(), messages, "low", trim_history=True)
        self.assertEqual(len(tokens), 1203)
        self.assertEqual(retained, messages)

    def test_context_reserves_generation_tokens_and_checks_exact_boundary(self):
        tokenizer = CountingTokenizer()
        tokens, _ = prepare_mlx_turn(tokenizer, [{"role": "user", "content": "x" * 7168}], "low")
        self.assertEqual(len(tokens) + 1024, 8192)
        with self.assertRaisesRegex(ValueError, "7168"):
            prepare_mlx_turn(tokenizer, [{"role": "user", "content": "x" * 7169}], "low")
        tokens, _ = prepare_mlx_turn(tokenizer, [{"role": "user", "content": "x" * 7936}],
                                     "low", max_new_tokens=256)
        self.assertEqual(len(tokens) + 256, 8192)

    def test_original_ovh_checkpoint_cannot_be_mistaken_for_opinel(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "absents"):
                validate_provenance(directory)
            manifest = {"source_repository": "OVHaiLLM/Olala-v1.0-mini-7A1B", "verified_tensors": 945,
                        "format": "mlx-q8"}
            path = Path(directory) / "OPINEL_PROVENANCE.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Opinel vérifié"):
                validate_provenance(directory)
            manifest["source_repository"] = MODEL
            path.write_text(json.dumps(manifest))
            self.assertEqual(validate_provenance(directory), manifest)

    def test_streaming_mlx_deltas_expose_only_final_and_keep_complete_answer(self):
        segments = ["private reasoning<|channel_end|><|chan", "nel_start|>final<|content|>Bon",
                    "jour été !<|channel_end|>"]
        responses = [SimpleNamespace(text=text, finish_reason=None, generation_tokens=number)
                     for number, text in enumerate(segments, 1)]
        responses[-1].finish_reason = "stop"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            answer, last = collect_response(responses, stream=True)
        self.assertEqual(answer, "Bonjour été !")
        self.assertEqual(output.getvalue(), "Olala > Bonjour été !\n")
        self.assertEqual(last.finish_reason, "stop")

    def test_truncated_reasoning_is_not_silently_accepted(self):
        responses = [SimpleNamespace(text="unfinished analysis", finish_reason="length", generation_tokens=1024)]
        with self.assertRaisesRegex(ValueError, "1024 tokens"):
            collect_response(responses)

    def test_complete_final_can_end_exactly_at_budget_limit(self):
        response = SimpleNamespace(text="<|channel_start|>final<|content|>4<|channel_end|>",
                                   finish_reason="length", generation_tokens=256)
        answer, _ = collect_response([response])
        self.assertEqual(answer, "4")

    def test_context_trimming_removes_only_old_complete_pairs(self):
        messages = [{"role": "user", "content": "x" * 600},
                    {"role": "assistant", "content": "y" * 600},
                    {"role": "user", "content": "new"}]
        tokens, retained = prepare_mlx_turn(CountingTokenizer(), messages, "low", trim_history=True,
                                            context_length=2048)
        self.assertEqual(tokens, [0, 1, 2])
        self.assertEqual(retained, messages[-1:])
        self.assertEqual(len(messages), 3)
