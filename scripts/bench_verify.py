"""How much does one target forward over L tokens cost, against one decoding step?

Speculative decoding (Télégraphe) replaces L - 1 single-token steps by one
forward over L = K + 1 tokens (the bonus token and K drafts). On a Mac the
decoding step is bound by memory bandwidth; a forward over L tokens reads the
shared weights once but up to 6 L distinct routed experts. The ratio
t(L) / t(1) bounds the speed-up:

    speed-up ≈ tokens accepted per step × t(1) / (t(L) + t(draft))

with the model card's acceptance (greedy: 2.96 at K = 3, 3.70 at K = 5, 4.20 at
K = 7). Every measurement restarts from the same snapshot after a real prompt.
"""

import argparse
import copy
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ACCEPTED = {3: (2.96, 2.93), 5: (3.70, 3.63), 7: (4.20, 4.09)}  # greedy, sampling (card, RTX 5090)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "opinel-n8-mlx-n8")
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--prompt-file", type=Path, default=ROOT / "README.md")
    parser.add_argument("--prompt-tokens", type=int, default=1000)
    args = parser.parse_args()

    from opinel_mac import configure_environment
    configure_environment()
    os.environ.setdefault("OLALA_DETERMINISTIC", "1")
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from opinel_mlx import load_mlx_model

    model, tokenizer = load_mlx_model(args.model_dir, "mps")
    tokens = tokenizer.encode(args.prompt_file.read_text(), add_special_tokens=False)[:args.prompt_tokens]
    cache = make_prompt_cache(model)
    for start in range(0, len(tokens), 2048):
        model(mx.array(tokens[start:start + 2048])[None], cache=cache)
        mx.eval([c.state for c in cache])
    # Real continuation tokens, so the routed experts are those of a real text.
    following = tokenizer.encode(args.prompt_file.read_text(), add_special_tokens=False)
    following = following[args.prompt_tokens:args.prompt_tokens + 64]

    def timed(length):
        times = []
        for repeat in range(args.repeats + 3):
            work = copy.deepcopy(cache)
            chunk = following[(repeat * length) % 48:(repeat * length) % 48 + length]
            mx.synchronize()
            started = time.perf_counter()
            mx.eval(model(mx.array(chunk)[None], cache=work))
            elapsed = time.perf_counter() - started
            if repeat >= 3:  # warm-up: kernel compilation for this length
                times.append(elapsed)
        return statistics.median(times)

    def decode_steps(steps):
        """Consecutive single-token steps, as mlx-lm decodes (fused decode path)."""
        times = []
        for repeat in range(3 + args.repeats // 4):
            work = copy.deepcopy(cache)
            mx.synchronize()
            started = time.perf_counter()
            for token in following[:steps]:
                mx.eval(model(mx.array([[token]]), cache=work))
            if repeat >= 3:
                times.append((time.perf_counter() - started) / steps)
        return statistics.median(times)

    one = decode_steps(16)
    print(f"Contexte : {len(tokens)} tokens. Décodage token par token : {one * 1000:.2f} ms "
          f"({1 / one:.0f} tokens/s, synchronisé à chaque pas).")
    print(f"\n{'L':>3} {'t(L) ms':>8} {'t(L)/t(1)':>9}   plafond du gain (vérification seule, drafter gratuit)")
    for length in (1, 2, 3, 4, 5, 6, 8):
        t = timed(length)
        k = length - 1
        bound = ""
        if k in ACCEPTED:
            greedy, sampled = ACCEPTED[k]
            bound = f"K = {k} : ×{greedy * one / t:.2f} greedy, ×{sampled * one / t:.2f} échantillonné"
        print(f"{length:>3} {t * 1000:8.2f} {t / one:9.2f}   {bound}", flush=True)
    print("\nAcceptation : moyennes de la fiche Télégraphe (RTX 5090), à remesurer sur Mac.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
