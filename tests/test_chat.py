import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opinel_mac import FinalChannelSink, generate_turn, make_final_streamer, parse_args, prepare_turn, run_chat

try:
    import torch
except ImportError:
    torch = None


class StreamingTests(unittest.TestCase):
    def test_fragmented_protocol_never_exposes_private_or_later_channels(self):
        raw = ("private reasoning<|channel_end|><|channel_start|>final<|content|>"
               "Bonjour, été !<|channel_end|><|channel_start|>tools<|content|>hidden")
        for width in (1, 2, 7, 43, len(raw)):
            with self.subTest(width=width):
                chunks = []
                sink = FinalChannelSink(chunks.append)
                for offset in range(0, len(raw), width):
                    sink.feed(raw[offset:offset + width])
                self.assertEqual("".join(chunks), "Bonjour, été !")

    def test_no_final_channel_produces_no_output(self):
        output = []
        sink = FinalChannelSink(output.append)
        sink.feed("private text " * 1000)
        self.assertEqual(output, [])
        self.assertLess(len(sink.pending), 50)

    def test_text_is_emitted_before_channel_end_arrives(self):
        output = []
        sink = FinalChannelSink(output.append)
        sink.feed("<|channel_start|>final<|content|>Voici une réponse ")
        self.assertEqual("".join(output), "Voici une réponse ")
        sink.feed("avec un symbole <")
        sink.feed(" 3.<|channel_end|>")
        self.assertEqual("".join(output), "Voici une réponse avec un symbole < 3.")

    @unittest.skipIf(torch is None, "PyTorch non installé")
    def test_transformers_streamer_skips_prompt_and_streams_final_words(self):
        tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "".join(map(chr, tokens)))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            streamer = make_final_streamer(tokenizer)
            streamer.put(torch.tensor([[ord(c) for c in "prompt secret"]]))
            prefix = ("analysis secret<|channel_end|>"
                      "<|channel_start|>final<|content|>Bonjour été ")
            for char in prefix:
                streamer.put(torch.tensor([ord(char)]))
            self.assertEqual(output.getvalue(), "Bonjour été ")
            for char in "!<|channel_end|><|im_end|>":
                streamer.put(torch.tensor([ord(char)]))
            streamer.end()
        self.assertEqual(output.getvalue(), "Bonjour été !")

    @unittest.skipIf(torch is None, "PyTorch non installé")
    def test_speaker_prefix_waits_for_final_text(self):
        tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "".join(map(chr, tokens)))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            streamer = make_final_streamer(tokenizer, prefix="Olala > ")
            streamer.on_finalized_text("private thoughts ")
            self.assertEqual(output.getvalue(), "")
            streamer.on_finalized_text("<|channel_start|>final<|content|>Bonjour ")
            streamer.on_finalized_text("Camille<|channel_end|>")
        self.assertEqual(output.getvalue(), "Olala > Bonjour Camille")


class BudgetTests(unittest.TestCase):
    def test_chat_budget_leaves_room_for_hidden_reasoning(self):
        self.assertEqual(parse_args(["chat"]).max_new_tokens, 1024)
        self.assertEqual(parse_args(["generate"]).max_new_tokens, 256)

    def test_explicit_budget_is_respected(self):
        self.assertEqual(parse_args(["chat", "--max-new-tokens", "128"]).max_new_tokens, 128)

    def test_speculative_block_defaults_to_the_measured_best(self):
        self.assertIsNone(parse_args(["chat", "--backend", "mlx"]).speculatif)
        self.assertEqual(parse_args(["chat", "--backend", "mlx", "--speculatif"]).speculatif, 2)
        self.assertEqual(parse_args(["generate", "--backend", "mlx", "--speculatif"]).speculatif, 3)
        self.assertEqual(parse_args(["chat", "--backend", "mlx", "--speculatif", "4"]).speculatif, 4)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["chat", "--speculatif"])
            with self.assertRaises(SystemExit):
                parse_args(["chat", "--backend", "mlx", "--speculatif", "8"])

    @unittest.skipIf(torch is None, "PyTorch non installé")
    def test_exhausted_reasoning_budget_explains_how_to_retry(self):
        args = SimpleNamespace(device="cpu", reasoning_effort="low", max_new_tokens=256)
        model = SimpleNamespace(generate=lambda **kwargs: torch.zeros(1, 257, dtype=torch.long))
        tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda *a, **kw: "unfinished private output")

        class Batch(dict):
            @property
            def input_ids(self):
                return self["input_ids"]

        inputs = Batch(input_ids=torch.zeros(1, 1, dtype=torch.long))
        with patch("opinel_mac.prepare_turn", return_value=(inputs, [])), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "256 tokens.*max-new-tokens 1024"):
                generate_turn(model, tokenizer, [], args)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return sum(len(message["content"]) for message in messages)

    def __call__(self, length, **kwargs):
        batch = SimpleNamespace(input_ids=SimpleNamespace(shape=(1, length)))
        batch.to = lambda device: batch
        return batch


class ContextTests(unittest.TestCase):
    def test_oldest_complete_pair_is_dropped_without_mutating_history(self):
        messages = [{"role": "user", "content": "x" * 600},
                    {"role": "assistant", "content": "y" * 600},
                    {"role": "user", "content": "nouvelle question"}]
        _, retained = prepare_turn(FakeTokenizer(), messages, "cpu", "low", trim_history=True)
        self.assertEqual(retained, messages[-1:])
        self.assertEqual(len(messages), 3)

    def test_oversized_current_message_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "trop long"):
            prepare_turn(FakeTokenizer(), [{"role": "user", "content": "x" * 1025}],
                         "cpu", "low", trim_history=True)

    def test_single_generation_does_not_silently_trim(self):
        with self.assertRaises(ValueError):
            prepare_turn(FakeTokenizer(), [{"role": "user", "content": "x" * 1025}],
                         "cpu", "low")


class ChatTests(unittest.TestCase):
    def run_session(self, commands, generate):
        with patch("builtins.input", side_effect=commands), \
                patch("opinel_mac.generate_turn", side_effect=generate) as mocked, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            status = run_chat(object(), object(), SimpleNamespace())
        self.assertEqual(status, 0)
        return mocked

    def test_history_is_carried_then_reset_without_reloading(self):
        histories = []

        def reply(model, tokenizer, messages, args, **kwargs):
            histories.append(list(messages))
            return "réponse", messages

        generated = self.run_session(["Bonjour", "Et ensuite ?", "/reset", "Nouveau", "/quit"], reply)
        self.assertEqual(generated.call_count, 3)
        self.assertEqual([m["role"] for m in histories[1]], ["user", "assistant", "user"])
        self.assertEqual(histories[2], [{"role": "user", "content": "Nouveau"}])
        self.assertIs(generated.call_args_list[0].args[0], generated.call_args_list[2].args[0])

    def test_failed_or_interrupted_turn_is_not_kept(self):
        histories = []

        def reply(model, tokenizer, messages, args, **kwargs):
            histories.append(list(messages))
            if len(histories) == 1:
                raise ValueError("Réponse finale tronquée")
            if len(histories) == 2:
                raise KeyboardInterrupt()
            return "ok", messages

        self.run_session(["tronqué", "interrompu", "suite", "/quit"], reply)
        self.assertEqual(histories[-1], [{"role": "user", "content": "suite"}])

    def test_empty_lines_commands_and_eof_do_not_generate(self):
        generated = self.run_session([" ", "/unknown", "/reset", EOFError()], None)
        generated.assert_not_called()


if __name__ == "__main__":
    unittest.main()
