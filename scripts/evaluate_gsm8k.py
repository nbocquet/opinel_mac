"""End-to-end GSM8K check, comparable to the Opinel model card (issue #1, §13).

The first 300 test questions, each followed by the card's instruction, sampled
at temperature 0.8 / top-p 0.9 / top-k 40 with a 6144-token budget; the answer
is the last number of the final channel. Card scores for N°8: 94.3 % (low) and
95.0 % (medium); at n = 300 the standard error is about 1.4 points, so a gap of
more than 3 points deserves an investigation.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
            "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/grade_school_math/data/test.jsonl")
DATA_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
DATA_PATH = ROOT / ".cache" / "gsm8k" / "test.jsonl"
INSTRUCTION = "\n\nGive the final answer as a number on the last line."
CARD_SCORES = {"low": 0.943, "medium": 0.950}
SETTINGS = {"temperature": 0.8, "top_p": 0.9, "top_k": 40, "max_new_tokens": 6144}
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def load_questions(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != DATA_SHA256:
            raise ValueError("Empreinte GSM8K inattendue : téléchargement refusé.")
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != DATA_SHA256:
        raise ValueError(f"Empreinte GSM8K inattendue : {path}")
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def parse_number(text):
    """Numeric value of a GSM8K-style number ('1,250', '-3', '4.50'), or None."""
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def gold_answer(answer):
    return parse_number(answer.rsplit("####", 1)[-1].strip())


def last_number(text):
    matches = NUMBER.findall(text or "")
    return parse_number(matches[-1].rstrip(",")) if matches else None


def is_correct(prediction, gold):
    return prediction is not None and gold is not None and abs(prediction - gold) < 1e-6


def summarize(report):
    results = report["results"]
    count = len(results)
    correct = sum(result["correct"] for result in results)
    accuracy = correct / count if count else 0.0
    summary = {"questions": count, "correct": correct, "accuracy": accuracy,
               "standard_error": math.sqrt(accuracy * (1 - accuracy) / count) if count else None,
               "no_final_answer": sum(result["error"] is not None for result in results),
               "median_generation_seconds": sorted(r["seconds"] for r in results)[count // 2] if count else None}
    card = CARD_SCORES.get(report["configuration"]["reasoning_effort"])
    if card is not None and count:
        summary["card_accuracy"] = card
        summary["gap_points"] = round(100 * (accuracy - card), 2)
        summary["investigate"] = count >= 300 and abs(accuracy - card) > 0.03
    return summary


def write_atomic(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    os.replace(handle.name, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", type=Path, help="Afficher le résumé d'un rapport existant")
    args = parser.parse_args()
    if args.summary:
        print(json.dumps(summarize(json.loads(args.summary.read_text())), ensure_ascii=False, indent=2))
        return 0
    questions = load_questions()[:args.limit]

    from opinel_mac import MLX_PROMPT_TOKENS, configure_environment, generate_turn
    from opinel_mlx import default_folder, load_mlx_model, validate_provenance
    configure_environment()
    os.environ["OLALA_DETERMINISTIC"] = "1"
    folder = (args.model_dir or default_folder()).resolve()
    configuration = {"dataset": DATA_URL, "dataset_sha256": DATA_SHA256, "questions": len(questions),
                     "instruction": INSTRUCTION, "reasoning_effort": args.reasoning_effort, "seed": args.seed,
                     "sampler": "vllm-order", "device": args.device, "model_dir": str(folder),
                     "provenance": validate_provenance(folder), **SETTINGS}
    if args.output is None:
        if args.resume:
            parser.error("--resume nécessite --output.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / ".cache" / "evaluations" / f"gsm8k-{args.reasoning_effort}-{stamp}.json"
    report = {"configuration": configuration, "results": []}
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if not args.resume or report["configuration"] != configuration:
            parser.error("Rapport existant incompatible ; utiliser un nouveau chemin ou une reprise identique.")
    elif args.resume:
        parser.error("Rapport à reprendre absent.")
    done = {result["index"] for result in report["results"]}
    print(f"Rapport : {args.output}", flush=True)

    import mlx.core as mx
    model, tokenizer = load_mlx_model(folder, args.device)
    settings = argparse.Namespace(backend="mlx", reasoning_effort=args.reasoning_effort,
                                  context_length=SETTINGS["max_new_tokens"] + MLX_PROMPT_TOKENS, **SETTINGS)
    for index, item in enumerate(questions):
        if index in done:
            continue
        mx.random.seed(args.seed + index)
        started = time.perf_counter()
        answer, error = None, None
        try:
            answer, _ = generate_turn(model, tokenizer, [{"role": "user", "content": item["question"] + INSTRUCTION}],
                                      settings)
        except ValueError as failure:
            error = str(failure)
        prediction, gold = last_number(answer), gold_answer(item["answer"])
        report["results"].append({"index": index, "gold": gold, "prediction": prediction,
                                  "correct": is_correct(prediction, gold), "error": error,
                                  "answer": answer, "seconds": round(time.perf_counter() - started, 2)})
        write_atomic(args.output, report)
        summary = summarize(report)
        print(f"{index + 1}/{len(questions)} : {'juste' if report['results'][-1]['correct'] else 'faux'} "
              f"(précision {summary['accuracy']:.3f} sur {summary['questions']})", flush=True)
    print(json.dumps(summarize(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, ImportError, FileNotFoundError, RuntimeError) as error:
        print(f"Évaluation interrompue : {error}", file=sys.stderr)
        raise SystemExit(2)
