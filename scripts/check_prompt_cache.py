"""Does resuming a chat turn from the previous prompt's snapshot change the logits?

For each turn k >= 2 of a recorded conversation, the logits of the new positions
(previous answer, new message, then 32 decoding steps) are computed three ways:

    reference   full prefill in 2048-token chunks, as mlx-lm does without a cache
    chunking    full prefill in 512-token chunks: the noise floor of a different split
    snapshot    the real chat path: snapshot after prompt k-1 (minus its last token),
                64 decoding steps on the live cache (the hidden reasoning), then the
                snapshot is restored and only the new tokens are computed

The snapshot is also compared with a full prefill cut at the same place, without
any snapshot or hidden decoding ("même coupure"): restoring is exact if the two
agree, and the snapshot path is then just another chunking of the same prompt.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DECODE_STEPS = 32
HIDDEN_STEPS = 64


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path, help="Rejeu JSON (scripts/replay_conversation.py run)")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "opinel-n8-mlx-n8")
    parser.add_argument("--reasoning-effort", default="low")
    args = parser.parse_args()

    from opinel_mac import configure_environment
    configure_environment()
    os.environ["OLALA_DETERMINISTIC"] = "1"
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from opinel_mlx import load_mlx_model, prepare_mlx_turn

    model, tokenizer = load_mlx_model(args.model_dir, "mps")
    turns = [t for t in json.loads(args.replay.read_text())["turns"] if t["answer"] is not None]
    messages, prompts = [], []
    for turn in turns:
        messages.append({"role": "user", "content": turn["user"]})
        prompts.append(prepare_mlx_turn(tokenizer, messages, args.reasoning_effort,
                                        context_length=10 ** 6, max_new_tokens=1)[0])
        messages.append({"role": "assistant", "content": turn["answer"]})

    def log_probs(logits):
        logits = logits[0].astype(mx.float32)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def prefill(cache, tokens, step, keep_from=None):
        """Feed tokens in chunks; return log-probs of positions >= keep_from (relative)."""
        kept = []
        for start in range(0, len(tokens), step):
            chunk = tokens[start:start + step]
            out = model(mx.array(chunk)[None], cache=cache)
            if keep_from is not None and start + len(chunk) > keep_from:
                kept.append(log_probs(out[:, max(0, keep_from - start):]))
            mx.eval([c.state for c in cache] + kept)
        return mx.concatenate(kept) if kept else None

    def decode(cache, tokens):
        rows = []
        for token in tokens:
            rows.append(log_probs(model(mx.array([[token]]), cache=cache)))
            mx.eval([c.state for c in cache] + rows[-1:])
        return mx.concatenate(rows)

    def full(prompt, reused, step, continuation):
        cache = make_prompt_cache(model)
        head = prefill(cache, prompt, step, keep_from=reused)
        return mx.concatenate([head, decode(cache, continuation)])

    def compare(candidate, reference):
        top1 = (mx.argmax(candidate, -1) == mx.argmax(reference, -1)).astype(mx.float32)
        top5 = mx.argpartition(-reference, kth=4, axis=-1)[:, :5]
        delta = mx.abs(mx.take_along_axis(candidate, top5, -1) - mx.take_along_axis(reference, top5, -1))
        return float(top1.mean()), float(delta.mean()), float(delta.max())

    print(f"{'tour':>4} {'nouveaux':>8} {'repris':>7}   {'découpage 512':^26}   {'instantané':^26}   "
          f"{'instantané / même coupure':^26}")
    print(f"{'':>4} {'':>8} {'':>7}" + f"   {'top-1':>7} {'top-5 moy':>9} {'max':>7}" * 3)
    worst = {"chunking": 0.0, "snapshot": 0.0, "same_cut": 0.0}
    for k in range(1, len(prompts)):
        previous, prompt = prompts[k - 1], prompts[k]
        if prompt[:len(previous)] != previous:
            print(f"{k + 1:>4} préfixe rompu : le chat recalculerait tout")
            continue
        reused = len(previous) - 1
        # Reference continuation: the greedy decode of the reference itself.
        cache = make_prompt_cache(model)
        last = prefill(cache, prompt, 2048, keep_from=len(prompt) - 1)
        continuation = [int(mx.argmax(last[-1]))]
        for _ in range(DECODE_STEPS - 1):
            continuation.append(int(mx.argmax(decode(cache, continuation[-1:])[-1])))
        reference = full(prompt, reused, 2048, continuation)
        chunking = full(prompt, reused, 512, continuation)

        live = make_prompt_cache(model)
        prefill(live, previous[:-1], 2048)
        snapshot = copy.deepcopy(live)
        decode(live, previous[-1:] + [continuation[0]] * (HIDDEN_STEPS - 1))  # stands for the hidden reasoning
        restored = copy.deepcopy(snapshot)
        head = prefill(restored, prompt[reused:], 2048, keep_from=0)
        cached = mx.concatenate([head, decode(restored, continuation)])

        cut = make_prompt_cache(model)
        prefill(cut, prompt[:reused], 2048)
        same_cut = mx.concatenate([prefill(cut, prompt[reused:], 2048, keep_from=0), decode(cut, continuation)])

        rows = compare(chunking, reference), compare(cached, reference), compare(cached, same_cut)
        for name, row in zip(worst, rows):
            worst[name] = max(worst[name], row[2])
        print(f"{k + 1:>4} {len(prompt) - reused:>8} {reused:>7}"
              + "".join(f"   {r[0]:7.2%} {r[1]:9.4f} {r[2]:7.3f}" for r in rows), flush=True)
    print(f"\nÉcart maximal : découpage {worst['chunking']:.3f}, instantané {worst['snapshot']:.3f}, "
          f"instantané contre même coupure {worst['same_cut']:.3f}.")
    print("Instantané identique à la même coupure => restauration exacte ; l'écart restant à la référence")
    print("est celui de tout autre découpage du préremplissage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
