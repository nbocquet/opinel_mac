"""Where does the multi-token forward of the OVH module lose its time against the fused decode?

Three measurements, after a real 1,000-token prompt, every forward restarting
from the same snapshot (median of the repeats, synchronous):

1. t(L) for L = 1 (fused decode), L = 1 through the generic layer path
   (OLALA_FUSED=0) and L = 2, 4, 6: tells a path overhead from a per-token cost.
2. --breakdown: the generic path layer by layer, with an evaluation barrier
   after each part (Mamba-3 mixer, attention mixer, mixer projection and
   geodesic, MoE). Barriers add their own cost; the shares are what matters.
3. --variants: the same t(L) table in a fresh process for each OVH switch read
   at import (sequential Mamba-3 prefill kernel, MoE kernel version, full TPA
   cache, non-deterministic MoE sums).
4. --rows: the fused decode itself over T independent sequences (its batching
   use): the best a fused verify of T tokens could do with these kernels. Each
   extra row reads its own 6 routed experts per layer.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LENGTHS = (1, 2, 4, 6)
VARIANTS = {
    "référence": {},
    "OLALA_M3_CHUNKED=0 (Mamba-3 séquentiel)": {"OLALA_M3_CHUNKED": "0"},
    "OLALA_MOE_V3=0": {"OLALA_MOE_V3": "0"},
    "OLALA_TPA_CACHE=full": {"OLALA_TPA_CACHE": "full"},
    "sans OLALA_DETERMINISTIC": {"OLALA_DETERMINISTIC": "0"},
}


def setup(args):
    from opinel_mac import configure_environment
    configure_environment()
    os.environ.setdefault("OLALA_DETERMINISTIC", "1")
    if os.environ["OLALA_DETERMINISTIC"] == "0":
        del os.environ["OLALA_DETERMINISTIC"]
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from opinel_mlx import load_mlx_model

    model, tokenizer = load_mlx_model(args.model_dir, "mps")
    tokens = tokenizer.encode((ROOT / "README.md").read_text(), add_special_tokens=False)
    cache = make_prompt_cache(model)
    for start in range(0, 1000, 500):
        model(mx.array(tokens[start:start + 500])[None], cache=cache)
        mx.eval([c.state for c in cache])
    return mx, model, cache, tokens[1000:1064]


def timed(mx, model, cache, following, length, repeats, fused=True):
    os.environ["OLALA_FUSED"] = "1" if fused else "0"
    times = []
    for repeat in range(repeats + 3):
        work = copy.deepcopy(cache)
        start = (repeat * length) % 48
        mx.synchronize()
        began = time.perf_counter()
        mx.eval(model(mx.array(following[start:start + length])[None], cache=work))
        if repeat >= 3:
            times.append(time.perf_counter() - began)
    os.environ["OLALA_FUSED"] = "1"
    return 1000 * statistics.median(times)


def table(args):
    mx, model, cache, following = setup(args)
    rows = {"1 (décodage fusionné)": timed(mx, model, cache, following, 1, args.repeats),
            "1 (chemin générique, OLALA_FUSED=0)": timed(mx, model, cache, following, 1, args.repeats, fused=False)}
    for length in LENGTHS[1:]:
        rows[str(length)] = timed(mx, model, cache, following, length, args.repeats)
    return rows


def breakdown(args):
    mx, model, cache, following = setup(args)
    from vendor.ovh_olala import model as olala

    parts = {}

    def clock(name, fn):
        def wrapped(*a, **k):
            mx.synchronize()
            began = time.perf_counter()
            out = fn(*a, **k)
            mx.eval(out)
            parts[name] = parts.get(name, 0.0) + time.perf_counter() - began
            return out
        return wrapped

    m3, attn, moe = olala.Mamba3Mixer.__call__, olala.DiffTPAttention.__call__, olala.OlalaMoE.__call__
    olala.Mamba3Mixer.__call__ = clock("Mamba-3 (in_proj + récurrence)", m3)
    olala.DiffTPAttention.__call__ = clock("attention (projections + noyau)", attn)
    olala.OlalaMoE.__call__ = clock("MoE (routeur + experts)", moe)
    results = {}
    os.environ["OLALA_FUSED"] = "0"
    for length in (1, 4):
        totals = []
        sums = {}
        for repeat in range(args.repeats // 2 + 3):
            parts.clear()
            work = copy.deepcopy(cache)
            mx.synchronize()
            began = time.perf_counter()
            mx.eval(model(mx.array(following[:length])[None], cache=work))
            total = time.perf_counter() - began
            if repeat >= 3:
                totals.append(total)
                for name, value in parts.items():
                    sums.setdefault(name, []).append(value)
        row = {name: 1000 * statistics.median(v) for name, v in sums.items()}
        row["total (avec barrières)"] = 1000 * statistics.median(totals)
        row["reste (projection, géodésique, tête, barrières)"] = row["total (avec barrières)"] - sum(
            v for k, v in row.items() if k != "total (avec barrières)")
        results[f"L = {length}"] = row
    os.environ["OLALA_FUSED"] = "1"
    olala.Mamba3Mixer.__call__, olala.DiffTPAttention.__call__, olala.OlalaMoE.__call__ = m3, attn, moe
    return results


def rows(args):
    mx, model, cache, following = setup(args)
    from vendor.ovh_olala.model import make_cache

    text = following * 64
    results = {}
    for count in (1, 2, 4, 5, 6, 8):
        batch = make_cache(model.model.layers)
        mx.eval(model(mx.array([text[i * 8:i * 8 + 64] for i in range(count)]), cache=batch))
        times = []
        for repeat in range(args.repeats + 3):
            work = copy.deepcopy(batch)
            step = mx.array([[text[500 + repeat + i]] for i in range(count)])
            mx.synchronize()
            began = time.perf_counter()
            mx.eval(model(step, cache=work))
            if repeat >= 3:
                times.append(time.perf_counter() - began)
        results[count] = 1000 * statistics.median(times)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "opinel-n8-mlx-n8")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--rows", action="store_true")
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.json:
        print(json.dumps(table(args)))
        return 0
    if args.variants:
        results = {}
        for label, env in VARIANTS.items():
            out = subprocess.run([sys.executable, __file__, "--json", "--repeats", str(args.repeats),
                                  "--model-dir", str(args.model_dir)], env={**os.environ, **env},
                                 capture_output=True, text=True, check=True).stdout
            results[label] = json.loads(out.strip().splitlines()[-1])
            print(f"{label:42}" + "  ".join(f"L={k.split()[0]}{'g' if 'générique' in k else ''}: {v:5.1f}"
                                            for k, v in results[label].items()), flush=True)
        return 0
    if args.rows:
        for count, value in rows(args).items():
            print(f"décodage fusionné, {count} séquence(s) indépendante(s) : {value:6.2f} ms")
        return 0
    if args.breakdown:
        for length, row in breakdown(args).items():
            print(f"\n{length}, chemin générique, barrière après chaque partie (ms, somme sur les 36 couches) :")
            for name, value in row.items():
                print(f"  {name:48} {value:6.2f}")
        return 0
    for label, value in table(args).items():
        print(f"L = {label:40} {value:6.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
