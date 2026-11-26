import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from opinel_mac import MLX_MAX_NEW_TOKENS, MODEL, parse_args
from opinel_mlx import ThinkingCounter, collect_response, prepare_mlx_turn, validate_provenance

try:
    import mlx.core as mx
except ImportError:
    mx = None


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
        # The prompt keeps 7168 tokens next to the default 4096-token reasoning budget.
        self.assertEqual(parse_args(["chat", "--backend", "mlx"]).context_length, 4096 + 7168)
        self.assertEqual(parse_args(["chat", "--backend", "mlx", "--max-new-tokens", "512"]).context_length, 8192)
        self.assertEqual(parse_args(["chat"]).context_length, 2048)
        small = parse_args(["chat", "--backend", "mlx", "--context-length", "4096"])
        self.assertEqual((small.context_length, small.max_new_tokens), (4096, 2048))

    def test_mlx_budget_follows_reasoning_effort_and_has_its_own_ceiling(self):
        self.assertEqual(parse_args(["chat", "--backend", "mlx"]).max_new_tokens, 4096)
        self.assertEqual(parse_args(["generate", "--backend", "mlx", "--reasoning-effort", "medium"]).max_new_tokens, 4096)
        high = parse_args(["chat", "--backend", "mlx", "--reasoning-effort", "high"])
        self.assertEqual((high.max_new_tokens, high.context_length), (8192, 8192 + 7168))
        self.assertEqual(parse_args(["chat", "--backend", "mlx", "--max-new-tokens", "16384"]).context_length,
                         16384 + 7168)
        for argv in (["chat", "--backend", "mlx", "--max-new-tokens", "16385"],
                     ["chat", "--max-new-tokens", "2048"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                parse_args(argv)

    def test_chat_samples_like_the_model_card_while_generate_stays_greedy(self):
        chat = parse_args(["chat", "--backend", "mlx"])
        self.assertEqual((chat.temperature, chat.top_p, chat.top_k), (0.8, 0.9, 40))
        single = parse_args(["generate"])
        self.assertEqual((single.temperature, single.top_p, single.top_k), (0.0, 1.0, 0))
        explicit = parse_args(["chat", "--temperature", "0"])
        self.assertEqual((explicit.temperature, explicit.top_p), (0.0, 0.9))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["chat", "--top-p", "0"])

    def test_context_must_leave_room_for_the_prompt(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["chat", "--backend", "mlx", "--context-length", "1024", "--max-new-tokens", "1024"])

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
        responses = [SimpleNamespace(text="unfinished analysis", finish_reason="length",
                                     generation_tokens=MLX_MAX_NEW_TOKENS)]
        with self.assertRaisesRegex(ValueError, f"{MLX_MAX_NEW_TOKENS} tokens") as failure:
            collect_response(responses)
        self.assertNotIn("plus élevé", str(failure.exception))
        self.assertIn("/reset", str(failure.exception))

    def test_smaller_exhausted_budget_can_be_increased(self):
        responses = [SimpleNamespace(text="unfinished analysis", finish_reason="length", generation_tokens=4096)]
        with self.assertRaisesRegex(ValueError, "max-new-tokens 8192"):
            collect_response(responses)
        responses[0].generation_tokens = 12000
        with self.assertRaisesRegex(ValueError, f"max-new-tokens {MLX_MAX_NEW_TOKENS}"):
            collect_response(responses)

    def test_thinking_counter_runs_during_analysis_and_is_erased_before_the_answer(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True

        terminal = Terminal()
        segments = ["analyse" * 3, "<|channel_end|><|channel_start|>final<|content|>Oui", "<|channel_end|>"]
        responses = [SimpleNamespace(text=text, finish_reason=None, generation_tokens=16 * number)
                     for number, text in enumerate(segments, 1)]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            answer, _ = collect_response(responses, stream=True, counter=ThinkingCounter(terminal))
        self.assertEqual(answer, "Oui")
        self.assertEqual(output.getvalue(), "Olala > Oui\n")
        self.assertEqual(terminal.getvalue(), "\rréflexion… 16 tokens\r\033[K")

    def test_thinking_counter_stays_silent_outside_a_terminal(self):
        quiet = io.StringIO()
        counter = ThinkingCounter(quiet)
        counter.update(16)
        counter.clear()
        self.assertEqual(quiet.getvalue(), "")

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


@unittest.skipIf(mx is None, "MLX absent de cet environnement")
class VllmSamplerTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)
        mx.random.seed(0)

    def draws(self, logits, count=2000, **settings):
        from opinel_mlx import make_vllm_sampler
        sampler = make_vllm_sampler(**settings)
        batch = mx.broadcast_to(mx.array(logits, dtype=mx.float32), (count, len(logits)))
        return set(sampler(batch).tolist())

    def test_zero_temperature_is_greedy(self):
        self.assertEqual(self.draws([0.0, 3.0, 1.0], temperature=0.0, top_p=0.9, top_k=40), {1})

    def test_top_k_keeps_only_the_k_most_likely_tokens(self):
        self.assertEqual(self.draws([0.0, 1.0, 2.0, 3.0], temperature=1.0, top_p=1.0, top_k=2), {2, 3})

    def test_top_p_applies_after_the_temperature(self):
        # p = 0.5 / 0.3 / 0.2 at T = 1: top-p 0.6 keeps two tokens.
        logits = [float(v) for v in mx.log(mx.array([0.5, 0.3, 0.2])).tolist()]
        self.assertEqual(self.draws(logits, temperature=1.0, top_p=0.6, top_k=0), {0, 1})
        # At T = 0.5 the law sharpens to 0.66 / 0.24 / 0.10: the first token alone covers 0.6.
        self.assertEqual(self.draws(logits, temperature=0.5, top_p=0.6, top_k=0), {0})
        # mlx-lm order (filter at T = 1, temperature at the draw) would keep tokens 0 and 1 here.

    def test_without_filters_every_token_can_be_drawn(self):
        self.assertEqual(self.draws([0.0, 0.1, 0.2], temperature=1.0, top_p=1.0, top_k=0), {0, 1, 2})
