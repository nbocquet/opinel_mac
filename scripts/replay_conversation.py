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
    from opinel_mlx import load_mlx_model, validate_provenance

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

    def generate(messages):
        return generate_turn(model, tokenizer, messages, settings, trim_history=True)

    turns = []
    for number, turn in enumerate(replay(generate, conversation["messages"]), 1):
        turns.append(turn)
        print(f"\n[{args.label}] tour {number} — Vous > {turn['user']}\nOlala > "
              f"{turn['answer'] if turn['answer'] is not None else '(échec : ' + turn['error'] + ')'}", flush=True)
    report = {"label": args.label, "format": provenance["format"], "model_dir": str(Path(args.model_dir).resolve()),
              "temperature": settings.temperature, "top_p": settings.top_p, "top_k": settings.top_k,
              "reasoning_effort": settings.reasoning_effort, "seed": args.seed, "device": args.device,
              "conversation": str(args.conversation), "expected": conversation.get("attendu"), "turns": turns}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\nEnregistré : {args.output}")
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
            print(f"\n--- {report['label']} ({report['format']}, T={report['temperature']}) ---\n{text.strip()}")
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
    replay_parser.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, ImportError, FileNotFoundError, RuntimeError) as error:
        print(f"Rejeu interrompu : {error}", file=sys.stderr)
        raise SystemExit(2)
