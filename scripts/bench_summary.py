"""Regenerate benchmarks/summary.md from benchmarks/results/<date>/<machine>/*.json.

One line per machine and date: plain decode (the warm-up witness of the greedy
run), verify of 4 tokens by the faster path on that machine (decode kernels of
opinel_verify, or the OVH prefill path) with its name, greedy speedup at
K = 3, sampled speedup at K = 2 (chat settings), chat speedup on the varied
conversation. A missing bench leaves "—".
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
HEADER = ("| Machine | Date | Puce | Décodage normal | Vérif 4 tokens | Greedy K = 3 | Échantillonné K = 2 "
          "| Chat varié |\n|---|---|---|---|---|---|---|---|")


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def number(value, fmt):
    return "—" if value is None else fmt.format(value).replace(".", ",")


def row(folder, machines):
    machine, date = folder.name, folder.parent.name
    sheet = machines.get(machine, {})
    profile = load(folder / "profile_verify.json")
    greedy = load(folder / "speculative_greedy.json")
    sampled = load(folder / "speculative_sampled.json")
    chat = load(folder / "chat_speed_variee.json")
    timings = (profile or {}).get("milliseconds", {})
    paths = {"décodage": timings.get("4 (noyaux de décodage, opinel_verify)"), "préremplissage": timings.get("4")}
    paths = {name: value for name, value in paths.items() if value is not None}
    best = min(paths, key=paths.get) if paths else None
    plain = (greedy or {}).get("plain_tokens_per_second")
    k3 = (greedy or {}).get("speculative", {}).get("3", {}).get("speedup")
    k2 = (sampled or {}).get("speculative", {}).get("2", {}).get("speedup")
    chat_gain = None
    if chat:
        rates = chat.get("tokens_per_second", {})
        if rates.get("normal") and rates.get("spéculatif"):
            chat_gain = rates["spéculatif"] / rates["normal"]
    quick = any((r or {}).get("meta", {}).get("quick") for r in (profile, greedy, sampled))
    label = machine + (" (rapide)" if quick else "")
    verify = number(paths.get(best), "{:.1f} ms") + (f" ({best})" if best else "")
    return (f"| {label} | {date} | {sheet.get('chip', '—')} | {number(plain, '{:.1f} tokens/s')} | {verify} "
            f"| {number(k3, '×{:.2f}')} | {number(k2, '×{:.2f}')} | {number(chat_gain, '×{:.2f}')} |")


def summary(benchmarks=BENCHMARKS):
    machines = {path.stem: json.loads(path.read_text()) for path in sorted((benchmarks / "machines").glob("*.json"))}
    folders = sorted(p for p in (benchmarks / "results").glob("*/*") if p.is_dir())
    lines = ["# Synthèse des benchmarks", "",
             "Généré par `scripts/bench_summary.py` ; protocole dans `benchmarks/README.md`.", "", HEADER]
    lines += [row(folder, machines) for folder in folders]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", type=Path, default=BENCHMARKS)
    args = parser.parse_args()
    text = summary(args.benchmarks)
    (args.benchmarks / "summary.md").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
