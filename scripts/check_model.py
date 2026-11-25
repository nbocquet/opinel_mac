"""Explicit integration check against the downloaded model (loads ~14 GB)."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opinel_mac import configure_environment, generate_turn, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", choices=["disabled", "low", "medium", "high"], default="low")
    args = parser.parse_args()
    configure_environment()
    if not 1 <= args.max_new_tokens <= 1024:
        parser.error("--max-new-tokens doit être compris entre 1 et 1024.")
    if args.backend == "mlx":
        from opinel_mlx import default_folder, load_mlx_model
        folder = args.model_dir or default_folder()
        model, tokenizer = load_mlx_model(folder, args.device)
    else:
        from transformers import AutoTokenizer
        folder = args.model_dir or ROOT / "models" / "opinel-n8"
        tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True, local_files_only=True)
        model = load_model(folder, args.device)
    results = []
    for prompt, expected in [
        ("Combien font 2 + 2 ? Réponds uniquement par le nombre.", "4"),
        ("Calcule 7 × 8. Réponds uniquement par le nombre.", "56"),
        ("What is 2 + 2? Answer with the number only.", "4"),
        ("Quelle est la capitale de la France ? Réponds en un seul mot.", "Paris"),
    ]:
        started = time.perf_counter()
        try:
            answer, _ = generate_turn(model, tokenizer, [{"role": "user", "content": prompt}], args)
        except (ValueError, RuntimeError) as error:
            answer = f"ERROR: {error}"
        result = {
            "prompt": prompt, "expected": expected, "answer": answer,
            "passed": answer.strip().rstrip(".!") == expected,
            "seconds": round(time.perf_counter() - started, 2),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report = {"backend": args.backend, "model_dir": str(folder), "device": args.device,
              "reasoning_effort": args.reasoning_effort, "results": results}
    destination = ROOT / ".cache" / f"{args.backend}-model-check.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
