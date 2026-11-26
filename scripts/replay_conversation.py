"""Replay a fixed list of user messages on one MLX export, like the interactive chat.

Each answer enters the history as in ``./opinel chat`` (failed turns are not kept),
so a variant is judged on the conversation it actually produces.

    run      replay one variant and save its answers as JSON
    compare  print several replays side by side, turn by turn
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONVERSATION = ROOT / "evaluation" / "conversation_saint_julien.json"


def replay(generate, user_messages):
    """Run the turns through `generate(messages) -> (answer, retained)`; return the transcript."""
    history, turns = [], []
    for text in user_messages:
        candidate = history + [{"role": "user", "content": text}]
        started = time.perf_counter()
        try:
            answer, retained = generate(candidate)
            error = None
            history = retained + [{"role": "assistant", "content": answer}]
        except (ValueError, RuntimeError) as failure:
            answer, error = None, str(failure)
        turns.append({"user": text, "answer": answer, "error": error,
                      "seconds": round(time.perf_counter() - started, 1)})
    return turns


def run(args):
    import mlx.core as mx
    from opinel_mac import configure_environment, generate_turn, parse_args
    from opinel_mlx import PromptCache, load_mlx_model, validate_provenance

    if args.output.exists():
        print(f"Déjà fait : {args.output}")
        return 0
    configure_environment()
    os.environ["OLALA_DETERMINISTIC"] = "1"
    conversation = json.loads(args.conversation.read_text())
    settings = parse_args(["chat", "--backend", "mlx", "--temperature", str(args.temperature),
                           "--reasoning-effort", args.reasoning_effort])
    provenance = validate_provenance(args.model_dir)
    model, tokenizer = load_mlx_model(args.model_dir, args.device)
    mx.random.seed(args.seed)
    prompt_cache = None if args.no_prompt_cache else PromptCache(model)

    def generate(messages):
        return generate_turn(model, tokenizer, messages, settings, trim_history=True, prompt_cache=prompt_cache)

    turns = []
    for number, turn in enumerate(replay(generate, conversation["messages"]), 1):
        turns.append(turn)
        print(f"\n[{args.label}] tour {number} — Vous > {turn['user']}\nOlala > "
              f"{turn['answer'] if turn['answer'] is not None else '(échec : ' + turn['error'] + ')'}", flush=True)
    report = {"label": args.label, "format": provenance["format"], "model_dir": str(Path(args.model_dir).resolve()),
              "temperature": settings.temperature, "top_p": settings.top_p, "top_k": settings.top_k,
              "reasoning_effort": settings.reasoning_effort, "seed": args.seed, "device": args.device,
              "prompt_cache": prompt_cache is not None,
              "conversation": str(args.conversation), "expected": conversation.get("attendu"), "turns": turns}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\nEnregistré : {args.output}")
    return 0


def speed(args):
    """Plain chat against --speculatif on the same conversation, turn by turn in alternation: a Mac
    that heats up slows both alike. Each variant keeps its own history and prompt cache."""
    import mlx.core as mx
    from opinel_mac import configure_environment, generate_turn, load_speculator, parse_args
    from opinel_mlx import PromptCache, load_mlx_model

    configure_environment()
    conversation = json.loads(args.conversation.read_text())
    model, tokenizer = load_mlx_model(args.model_dir, args.device)
    mx.random.seed(args.seed)
    base = ["chat", "--backend", "mlx", "--temperature", str(args.temperature),
            "--reasoning-effort", args.reasoning_effort]
    variants = {"normal": parse_args(base), "spéculatif": parse_args(base + ["--speculatif"])}
    variants["spéculatif"].speculator = load_speculator(model, variants["spéculatif"].speculatif)
    caches = {name: PromptCache(model) for name in variants}
    histories = {name: [] for name in variants}
    totals = {name: {"tokens": 0, "seconds": 0.0, "decode": []} for name in variants}
    for number, text in enumerate(conversation["messages"], 1):
        for name, settings in variants.items():
            candidate = histories[name] + [{"role": "user", "content": text}]
            try:
                answer, retained = generate_turn(model, tokenizer, candidate, settings, trim_history=True,
                                                 prompt_cache=caches[name])
                histories[name] = retained + [{"role": "assistant", "content": answer}]
            except (ValueError, RuntimeError) as failure:
                print(f"[{name}] tour {number} : échec ({failure})", flush=True)
                continue
            turn = settings.last_turn
            totals[name]["tokens"] += turn["tokens"]
            totals[name]["seconds"] += turn["seconds"]
            totals[name]["decode"].append(turn["decode_tokens_per_second"])
            per_step = f", {turn['tokens_per_step']:.2f} tokens par passe" if turn.get("tokens_per_step") else ""
            print(f"[{name}] tour {number} : prompt {turn['context_tokens']} tokens, {turn['tokens']} générés en "
                  f"{turn['seconds']:.1f} s, décodage {turn['decode_tokens_per_second']:.1f} tokens/s{per_step}",
                  flush=True)
    rates = {name: total["tokens"] / total["seconds"] for name, total in totals.items()}
    for name, total in totals.items():
        print(f"{name:11} : {total['tokens']} tokens en {total['seconds']:.1f} s, {rates[name]:.1f} tokens/s "
              f"(préremplissage compris), décodage médian {sorted(total['decode'])[len(total['decode']) // 2]:.1f} "
              f"tokens/s")
    print(f"Gain du chat spéculatif : ×{rates['spéculatif'] / rates['normal']:.2f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"temperature": args.temperature, "totals": totals,
                                           "tokens_per_second": rates}, ensure_ascii=False, indent=2) + "\n")
    return 0


def compare(args):
    reports = [json.loads(path.read_text()) for path in args.reports]
    expected = reports[0].get("expected")
    if expected:
        print(f"Attendu : {expected}\n")
    for number in range(max(len(report["turns"]) for report in reports)):
        print("=" * 100)
        print(f"Tour {number + 1} — Vous > {reports[0]['turns'][number]['user']}")
        for report in reports:
            turn = report["turns"][number] if number < len(report["turns"]) else None
            text = "(absent)" if turn is None else turn["answer"] or f"(échec : {turn['error']})"
            seconds = "" if turn is None or "seconds" not in turn else f", {turn['seconds']} s"
            print(f"\n--- {report['label']} ({report['format']}, T={report['temperature']}{seconds}) ---"
                  f"\n{text.strip()}")
        print()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    replay_parser = commands.add_parser("run")
    replay_parser.add_argument("--model-dir", type=Path, required=True)
    replay_parser.add_argument("--label", required=True)
    replay_parser.add_argument("--temperature", type=float, default=0.0)
    replay_parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    replay_parser.add_argument("--seed", type=int, default=0)
    replay_parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    replay_parser.add_argument("--conversation", type=Path, default=DEFAULT_CONVERSATION)
    replay_parser.add_argument("--no-prompt-cache", action="store_true",
                               help="Recalculer tout le prompt à chaque tour")
    replay_parser.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("reports", type=Path, nargs="+")
    timing = commands.add_parser("speed", help="Chat normal contre --speculatif, tours alternés")
    timing.add_argument("--model-dir", type=Path, default=ROOT / "models" / "opinel-n8-mlx-n8")
    timing.add_argument("--temperature", type=float, default=0.8)
    timing.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    timing.add_argument("--seed", type=int, default=0)
    timing.add_argument("--device", choices=["mps", "cpu"], default="mps")
    timing.add_argument("--conversation", type=Path, default=DEFAULT_CONVERSATION)
    timing.add_argument("--output", type=Path)
    args = parser.parse_args()
    return {"run": run, "compare": compare, "speed": speed}[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, ImportError, FileNotFoundError, RuntimeError) as error:
        print(f"Rejeu interrompu : {error}", file=sys.stderr)
        raise SystemExit(2)
