"""Rank the target's tokens by how often Opinel writes them, for a restricted Télégraphe head (issue #5).

The target generates answers, hidden reasoning included (the drafter proposes those
tokens too), with the chat's sampling (0.8 / 0.9 / 40, reasoning low) on prompts that
the acceptance and loop measurements do not use: the reliability suites, the two
conversations, and GSM8K test questions 300 onwards (the evaluation takes the first
300). Tokens are ranked by count; unseen tokens follow in id order (the byte-pair
merge order of the tokenizer, roughly a frequency order), special tokens first.

    .venv-mlx/bin/python scripts/telegraphe_vocab.py   # about 15 min, writes evaluation/telegraphe_vocab.json
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEEP = 65536  # ranks written to the file


def corpus_prompts(gsm8k_count):
    """Message lists: reliability suites, conversations (each message alone), GSM8K questions 300..."""
    prompts = []
    for name in ("reliability_fr.json", "reliability_fr_holdout.json"):
        prompts += [case["messages"] for case in json.loads((ROOT / "evaluation" / name).read_text())["cases"]]
    for name in ("conversation_saint_julien.json", "conversation_variee.json"):
        prompts += [[{"role": "user", "content": m}] for m in json.loads((ROOT / "evaluation" / name).read_text())["messages"]]
    lines = (ROOT / ".cache" / "gsm8k" / "test.jsonl").read_text().splitlines()[300:300 + gsm8k_count]
    prompts += [[{"role": "user", "content": json.loads(line)["question"]
                  + "\n\nGive the final answer as a number on the last line."}] for line in lines]
    return prompts


def ranking(counts, vocabulary, special):
    """Special ids, then seen ids by decreasing count (ties by id), then unseen ids by id."""
    seen = sorted((i for i in counts if i not in special), key=lambda i: (-counts[i], i))
    chosen = set(special) | set(seen)
    return sorted(special) + seen + [i for i in range(vocabulary) if i not in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "opinel-n8-mlx-n8")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--gsm8k", type=int, default=40)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "telegraphe_vocab.json")
    args = parser.parse_args()

    from opinel_mac import configure_environment
    configure_environment()
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from opinel_mlx import load_mlx_model, make_vllm_sampler

    model, tokenizer = load_mlx_model(args.model_dir, "mps")
    mx.random.seed(0)
    sampler = make_vllm_sampler(0.8, 0.9, 40)
    counts, total = Counter(), 0
    prompts = corpus_prompts(args.gsm8k)
    started = time.perf_counter()
    for number, messages in enumerate(prompts, 1):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             reasoning_effort="low")
        prompt = tokenizer.encode(text, add_special_tokens=False)
        for token, _ in generate_step(mx.array(prompt), model, max_tokens=args.tokens, sampler=sampler):
            counts[int(token)] += 1
            total += 1
            if int(token) in tokenizer.eos_token_ids:
                break
        print(f"{number:>3}/{len(prompts)} : {total} tokens, {len(counts)} distincts", flush=True)
    vocabulary = model.lm_head.weight.shape[0]
    special = [i for i in set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values()) if i < vocabulary]
    ranked = ranking(counts, vocabulary, special)
    args.output.write_text(json.dumps({
        "description": "Ids de la cible par fréquence dans ses propres réponses (réflexion comprise), "
                       "pour la tête restreinte de Télégraphe ; puis les ids jamais vus, dans l'ordre des ids.",
        "corpus": {"prompts": len(prompts), "tokens": total, "distinct": len(counts),
                   "sampling": "0.8 / 0.9 / 40, réflexion low", "max_tokens": args.tokens,
                   "sources": "reliability_fr, reliability_fr_holdout, conversations, GSM8K test 300+"},
        "vocabulary": vocabulary, "ranked": ranked[:KEEP],
    }) + "\n")
    print(f"{args.output.relative_to(ROOT)} : {total} tokens, {len(counts)} distincts, "
          f"{time.perf_counter() - started:.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
