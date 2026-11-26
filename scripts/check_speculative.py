"""Speculative decoding with Télégraphe against plain decoding, on the MLX target (greedy, or sampled
with --temperature / --top-p / --top-k: rejection sampling, exact for the target law).

For each prompt, the plain greedy decode (mlx-lm, fused single-token path) and
the speculative one (verify K + 1 tokens, roll back to the accepted prefix) each
write up to N tokens. Reported: decoding speed (first token excluded, prefill
excluded), tokens per verify forward, and how many leading tokens agree. The
two paths use different kernels (multi-token verify against fused decode): a
late divergence is expected, an early nonsense one is not.

Rollback check (--rollback-check): one verify of 6 tokens rolled back to keep
tokens, against a forward of those keep tokens alone. The KV and token-shift
caches must be identical, the Mamba-3 state equal up to fp32 rounding (the
replay kernel is the sequential one, the forward uses the chunked one).
With keep = 1 the direct forward is a single token: the fused decode path.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "opinel-n8-mlx-n8")
    parser.add_argument("--drafter-dir", type=Path, default=ROOT / "models" / "telegraphe")
    parser.add_argument("--prompts", type=Path, default=ROOT / "evaluation" / "parity_prompts.json")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, nargs="+", default=[3, 5, 7], help="Valeurs de K")
    parser.add_argument("--own-head", action="store_true", help="Tête bf16 du drafter au lieu de la tête INT8 de la cible")
    parser.add_argument("--bf16-layers", action="store_true", help="Couches du drafter en bf16 au lieu de 8 bits")
    parser.add_argument("--verify", choices=["auto", "decode", "prefill"], default="auto",
                        help="Chemin de vérification : noyaux de décodage (opinel_verify), chemin de préremplissage "
                             "d'OVH, ou auto (le plus rapide sur cette machine, chronométré au chargement)")
    parser.add_argument("--mamba-by-token", action="store_true",
                        help="Mamba-3 par un pas de décodage par token au lieu d'un seul m3_prefill par bloc")
    parser.add_argument("--layers-per-submit", type=int, nargs="+",
                        help="Envoi au GPU toutes les N couches pendant la construction (0 : à la fin) ; "
                             "plusieurs valeurs sont mesurées en alternance")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 : greedy ; sinon échantillonnage (réglages du chat : 0.8, --top-p 0.9, --top-k 40)")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollback-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from opinel_mac import configure_environment
    configure_environment()
    os.environ.setdefault("OLALA_DETERMINISTIC", "1")
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from opinel_mlx import load_mlx_model
    from opinel_speculative import Speculator
    from opinel_telegraphe import Telegraphe, prepare
    import opinel_verify
    opinel_verify.MAMBA_BY_TOKEN = args.mamba_by_token

    if not (args.drafter_dir / "mask_embedding.npy").exists():
        prepare(args.drafter_dir)
    model, tokenizer = load_mlx_model(args.model_dir, "mps")
    drafter = Telegraphe(args.drafter_dir, embedding=model.model.embedding.weight,
                         target_head=None if args.own_head else model.lm_head,
                         quantize_bits=None if args.bf16_layers else 8)
    eos = set(tokenizer.eos_token_ids)
    prompts = json.loads(args.prompts.read_text())["prompts"][:args.limit]
    from opinel_mlx import make_vllm_sampler
    sampler = make_vllm_sampler(args.temperature, args.top_p, args.top_k)
    mx.random.seed(args.seed)
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}

    def encode(text):
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                 add_generation_prompt=True, reasoning_effort="low")
        return tokenizer.encode(rendered, add_special_tokens=False)

    def plain(prompt):
        tokens, first = [], None
        for token, _ in generate_step(mx.array(prompt), model, max_tokens=args.tokens, sampler=sampler):
            if first is None:
                first = time.perf_counter()
            tokens.append(int(token))
            if int(token) in eos:
                break
        return tokens, time.perf_counter() - first

    def speculative(speculator, prompt):
        tokens, first, stats = [], None, {}
        for token in speculator.generate(prompt, args.tokens, eos=eos, stats=stats, seed=args.seed, **sampling):
            if first is None:
                first = time.perf_counter()
            tokens.append(token)
        return tokens, time.perf_counter() - first, stats

    def agree(a, b):
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    encoded = [encode(text) for text in prompts]
    # Plain and speculative runs alternate prompt by prompt: a Mac that heats up (or changes power
    # source) during the run then slows every variant alike instead of the last ones measured.
    submits = args.layers_per_submit or [None]
    variants = [(steps, submit) for steps in args.steps for submit in submits]
    verify = {"auto": "auto", "decode": True, "prefill": False}[args.verify]
    speculators = {variant: Speculator(model, drafter, steps=variant[0], decode_kernels=verify,
                                       layers_per_submit=variant[1]) for variant in variants}
    for (steps, _), speculator in speculators.items():
        path = "noyaux de décodage" if speculator.decode_kernels else "chemin de préremplissage"
        timing = ("" if speculator.timings is None else " (" + ", ".join(
            f"{name} {value:.1f} ms" for name, value in speculator.timings.items()) + ")")
        print(f"K = {steps} : vérification par {path}{timing}", flush=True)
    plain(encoded[0])  # warm-up
    for speculator in speculators.values():
        speculative(speculator, encoded[0])  # warm-up of each block length
    baseline = []
    runs = {variant: [] for variant in variants}
    for prompt in encoded:
        baseline.append(plain(prompt))
        for steps, speculator in speculators.items():
            runs[steps].append(speculative(speculator, prompt))
    plain_rate = sum(len(t) - 1 for t, _ in baseline) / sum(s for _, s in baseline)
    print(f"Décodage normal : {plain_rate:.1f} tokens/s ({len(encoded)} prompts, jusqu'à {args.tokens} tokens, "
          f"mesures alternées).", flush=True)

    report = {"plain_tokens_per_second": plain_rate, "interleaved": True, "sampling": sampling, "speculative": {}}
    for variant in variants:
        steps, submit = variant
        label = f"K = {steps}" + ("" if submit is None else f", envoi toutes les {submit or '—'} couches")
        produced, seconds, kept, draft_s, verify_s, agreements = 0, 0.0, [], 0.0, 0.0, []
        for (tokens, elapsed, stats), (reference, _) in zip(runs[variant], baseline):
            produced += len(tokens) - 1
            seconds += elapsed
            kept += stats["kept"]
            draft_s += stats["draft_seconds"]
            verify_s += stats["verify_seconds"]
            agreements.append(agree(tokens, reference) / max(1, min(len(tokens), len(reference))))
        rate = produced / seconds
        row = {"verify_path": "decode" if speculators[variant].decode_kernels else "prefill",
               "verify_calibration_ms": speculators[variant].timings, "tokens_per_second": rate, "speedup": rate / plain_rate, "tokens_per_step": float(np.mean(kept)),
               "draft_ms": 1000 * draft_s / len(kept), "verify_ms": 1000 * verify_s / len(kept),
               "identical_prefix_median": statistics.median(agreements)}
        report["speculative"][str(steps) if submit is None else f"{steps}/{submit}"] = row
        print(f"{label} : {rate:.1f} tokens/s (×{row['speedup']:.2f}), {row['tokens_per_step']:.2f} tokens par "
              f"passe, drafter {row['draft_ms']:.1f} ms + vérification {row['verify_ms']:.1f} ms par passe"
              + ("" if args.temperature > 0 else
                 f", début identique au greedy normal : {100 * row['identical_prefix_median']:.0f} % (médiane)"),
              flush=True)

    if args.rollback_check:
        # One verify of 6 tokens after a 500-token prompt, rolled back to `keep`, against a forward of the
        # first `keep` tokens alone: same KV and token-shift caches, Mamba-3 state up to fp32 rounding
        # (sequential replay kernel against the chunked one).
        from opinel_speculative import VerifyRecorder
        recorder = VerifyRecorder(model)
        ids = tokenizer.encode((ROOT / "evaluation" / "texte_contexte.md").read_text(), add_special_tokens=False)[:506]
        base = make_prompt_cache(model)
        model(mx.array(ids[:500])[None], cache=base)
        mx.eval([c.state for c in base])
        block = ids[500:506]
        report["rollback"] = {}
        for keep in (1, 2, 3, 5):
            rolled = copy.deepcopy(base)
            with recorder:
                mx.eval(model(mx.array(block)[None], cache=rolled))
            recorder.rollback(rolled, keep, len(block))
            mx.eval([c.state for c in rolled])
            direct = copy.deepcopy(base)
            # Evaluate the logits: the fused single-token decode (keep = 1) writes the caches in place as a
            # side effect of the logits, which evaluating the cache arrays alone would not run.
            mx.eval(model(mx.array(block[:keep])[None], cache=direct))
            mx.eval([c.state for c in rolled], [c.state for c in direct])
            mamba, kv, shift = 0.0, 0.0, 0.0
            for index, layer in enumerate(model.model.layers):
                if layer.layer_type == "M":
                    x, y = np.array(rolled[index][0]), np.array(direct[index][0])
                    delta = x - y
                    # The last A entries are rotary angles: the kernels wrap them differently around ±π.
                    angles = delta[..., -layer.mixer.A:]
                    delta[..., -layer.mixer.A:] = (angles + np.pi) % (2 * np.pi) - np.pi
                    mamba = max(mamba, float(np.abs(delta).max() / (np.abs(y).max() + 1e-12)))
                    continue
                if rolled[index][0].offset != direct[index][0].offset:
                    raise SystemExit(f"Longueur de cache KV différente à la couche {index}.")
                offset = rolled[index][0].offset
                for x, y in zip(rolled[index][0].state, direct[index][0].state):
                    x, y = np.array(x.astype(mx.float32)), np.array(y.astype(mx.float32))
                    kv = max(kv, float(np.abs(x[:, :offset] - y[:, :offset]).max()))
                for slot in range(len(rolled[index][1].state)):
                    x = np.array(rolled[index][1][slot].astype(mx.float32))
                    y = np.array(direct[index][1][slot].astype(mx.float32)).reshape(x.shape)
                    shift = max(shift, float(np.abs(x - y).max()))
            report["rollback"][str(keep)] = {"mamba_relative": mamba, "kv": kv, "shift": shift}
            note = " (contre le décodage fusionné : autre noyau)" if keep == 1 else ""
            print(f"Retour arrière à {keep}/6 : état Mamba-3 écart relatif {mamba:.1e}, cache KV écart {kv:.1e}, "
                  f"décalage écart {shift:.1e}{note}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
