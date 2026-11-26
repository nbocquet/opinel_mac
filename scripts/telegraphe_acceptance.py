"""Offline acceptance of Télégraphe on the MLX target, without any speculative loop.

For each prompt the target writes a greedy continuation; the whole sequence is
then read back in one forward that records the residual stream after layers 1,
9, 17, 25 and 33. At every position t of the continuation, Télégraphe drafts 7
tokens from the context rows < t and the anchor token t; a draft token counts
while it equals the target's greedy choice. Tokens per step = accepted drafts + 1
(the token the target adds itself), the quantity of the model card (greedy, 2,000
held-out answers: 2.96 at K = 3, 3.70 at K = 5, 4.20 at K = 7). The block is causal,
so the 7-token draft gives K = 3 and 5 by truncation.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CARD = {3: 2.96, 5: 3.70, 7: 4.20}
STEPS = 7


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "opinel-n8-mlx-n8")
    parser.add_argument("--drafter-dir", type=Path, default=ROOT / "models" / "telegraphe")
    parser.add_argument("--prompts", type=Path, default=ROOT / "evaluation" / "parity_prompts.json")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=256, help="Continuation greedy de la cible")
    parser.add_argument("--raw-embeddings", action="store_true",
                        help="Lignes brutes de la table d'embedding au lieu des lignes normalisées (vLLM)")
    parser.add_argument("--shared-head", action="store_true", help="Tête INT8 de la cible au lieu de la tête bf16")
    parser.add_argument("--quantize-bits", type=int, choices=[4, 8], help="Couches du drafter quantifiées")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from opinel_mac import configure_environment
    configure_environment()
    os.environ.setdefault("OLALA_DETERMINISTIC", "1")
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from opinel_mlx import load_mlx_model
    from opinel_telegraphe import Telegraphe, capture_aux, prepare

    if not (args.drafter_dir / "mask_embedding.npy").exists():
        prepare(args.drafter_dir)  # pinned revision, hashes checked (≈ 0,9 Go)

    model, tokenizer = load_mlx_model(args.model_dir, "mps")
    drafter = Telegraphe(args.drafter_dir, embedding=model.model.embedding.weight,
                         normalize_embeddings=not args.raw_embeddings,
                         target_head=model.lm_head if args.shared_head else None, quantize_bits=args.quantize_bits)
    records = capture_aux(model)
    prompts = json.loads(args.prompts.read_text())["prompts"][:args.limit]

    totals = {k: [] for k in CARD}
    per_prompt = []
    for number, item in enumerate(prompts, 1):
        text = tokenizer.apply_chat_template([{"role": "user", "content": item}], tokenize=False,
                                             add_generation_prompt=True, reasoning_effort="low")
        prompt = tokenizer.encode(text, add_special_tokens=False)
        continuation = []
        for token, _ in generate_step(mx.array(prompt), model, max_tokens=args.tokens,
                                      sampler=lambda logprobs: mx.argmax(logprobs, axis=-1)):
            continuation.append(int(token))
            if int(token) in tokenizer.eos_token_ids:
                break
        sequence = prompt + continuation
        cache = make_prompt_cache(model)
        records.clear()
        targets, aux_rows = [], []
        for start in range(0, len(sequence), 2048):
            logits = model(mx.array(sequence[start:start + 2048])[None], cache=cache)
            targets.append(mx.argmax(logits[0], axis=-1))
            aux_rows.append(mx.concatenate([r[0] for r in records], axis=-1))
            records.clear()
            mx.eval(targets[-1], aux_rows[-1])
        target = np.array(mx.concatenate(targets))  # target[t] = greedy choice for position t + 1
        aux = mx.concatenate(aux_rows)
        context = drafter.context_kv(aux)
        mx.eval(context)

        anchors = list(range(len(prompt), len(sequence) - 1))
        accepted = {k: [] for k in CARD}
        started = time.perf_counter()
        for begin in range(0, len(anchors), args.batch):
            chunk = anchors[begin:begin + args.batch]
            drafts, _ = drafter.draft([sequence[t] for t in chunk], chunk, context, STEPS)
            for t, draft in zip(chunk, drafts):
                count = 0
                for j in range(STEPS):
                    position = t + j
                    if position >= len(target) or draft[j] != target[position]:
                        break
                    # The next draft token is only checked if the target's choice is the token
                    # actually in the sequence (the context the drafter's next step assumes).
                    if position + 1 >= len(sequence) or sequence[position + 1] != target[position]:
                        count += 1
                        break
                    count += 1
                for k in CARD:
                    accepted[k].append(min(count, k) + 1)
        seconds = time.perf_counter() - started
        row = {"id": item[:28].replace("\n", " "), "prompt_tokens": len(prompt), "anchors": len(anchors),
               **{f"tau_{k}": float(np.mean(accepted[k])) for k in CARD},
               "draft_ms_per_block": 1000 * seconds / max(1, len(anchors))}
        per_prompt.append(row)
        for k in CARD:
            totals[k].extend(accepted[k])
        print(f"{number:>2}/{len(prompts)} {row['id']:<28} {len(anchors):>4} positions   "
              + "   ".join(f"K={k} : {row[f'tau_{k}']:.2f}" for k in CARD), flush=True)

    summary = {f"tau_{k}": float(np.mean(totals[k])) for k in CARD}
    print("\nTokens par passe (acceptés + 1), toutes positions :")
    for k in CARD:
        print(f"  K = {k} : {summary[f'tau_{k}']:.2f}   (fiche, RTX 5090, 2 000 réponses : {CARD[k]:.2f})")
    print(f"Embeddings : {'bruts' if args.raw_embeddings else 'normalisés (comme vLLM)'} ; tête : "
          f"{'INT8 de la cible' if args.shared_head else 'bf16 du drafter'} ; couches : "
          f"{f'{args.quantize_bits} bits' if args.quantize_bits else 'bf16'}.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"raw_embeddings": args.raw_embeddings, "shared_head": args.shared_head,
                                           "quantize_bits": args.quantize_bits, "summary": summary,
                                           "prompts": per_prompt}, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
