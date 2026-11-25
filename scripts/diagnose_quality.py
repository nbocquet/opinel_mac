"""Capture factual/correction/style regressions; answers require human review."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opinel_mac import configure_environment, generate_turn, load_model

CASES = [
    {
        "id": "fondue_comparison_exact",
        "messages": [{"role": "user", "content": "donnes moi la différence entre fondue suisse et savoyarde"}],
        "review": "Rejeu exact du premier message signalé ; vérifier les fromages et la distinction entre traditions.",
        "sources": ["https://www.fromage-beaufort.com/fr/news/on-fond-pour-la-fondue",
                    "https://fribourg.ch/fr/terroir-fribourg/recettes/fondue-moitie-moitie/"],
    },
    {
        "id": "fondue_fresh",
        "messages": [{"role": "user", "content":
                      "Quels fromages composent habituellement une fondue savoyarde ? Réponds en une phrase."}],
        "review": "Vérifier les fromages cités ; le reblochon ne doit pas être présenté comme obligatoire.",
        "source": "https://www.fromage-beaufort.com/fr/news/on-fond-pour-la-fondue",
    },
    {
        "id": "correct_prior_error",
        "messages": [
            {"role": "user", "content": "Quel est le fromage principal de la fondue savoyarde ?"},
            {"role": "assistant", "content": "Le fromage principal de la fondue savoyarde est le reblochon."},
            {"role": "user", "content": "Non, le reblochon c'est la tartiflette. On utilise notamment du Beaufort pour la fondue. Corrige ta réponse en une phrase."},
        ],
        "review": "La réponse doit corriger l'erreur, sans réaffirmer que le reblochon est essentiel.",
        "source": "https://www.fromage-beaufort.com/fr/news/on-fond-pour-la-fondue",
    },
    {
        "id": "tutoiement_in_context",
        "messages": [
            {"role": "user", "content": "Tu connais la fondue savoyarde ?"},
            {"role": "assistant", "content": "Oui, je peux vous expliquer sa recette. Que souhaitez-vous savoir ?"},
            {"role": "user", "content": "tutoie moi"},
        ],
        "review": "La réponse doit comprendre la demande de tutoiement, sans produire un tutoriel ou une recette.",
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--reasoning-effort", choices=["disabled", "low", "medium", "high"], default="low")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--case", choices=[case["id"] for case in CASES], action="append",
                        help="Limiter le diagnostic à certains cas (répétable)")
    parser.add_argument("--reference", action="store_true", help="MLX sans les kernels Metal fusionnés OVH")
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 1024:
        parser.error("--max-new-tokens doit être compris entre 1 et 1024.")
    args.context_length = 8192 if args.backend == "mlx" else 2048
    configure_environment()
    if args.reference and args.backend != "mlx":
        parser.error("--reference s'utilise avec --backend mlx.")
    if args.reference:
        os.environ["OLALA_MLX_REF"] = "1"
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
    mode = "mlx-reference" if args.reference else args.backend
    report = {"backend": args.backend, "reference_kernels": args.reference,
              "device": args.device, "model_dir": str(folder),
              "reasoning_effort": args.reasoning_effort, "max_new_tokens": args.max_new_tokens,
              "evaluation": "Human review required; these captures are not a quality pass.", "results": results}
    destination = ROOT / ".cache" / f"quality-{mode}-{args.reasoning_effort}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        if args.case and case["id"] not in args.case:
            continue
        started = time.perf_counter()
        answer, error = None, None
        try:
            answer, _ = generate_turn(model, tokenizer, case["messages"], args)
        except (ValueError, RuntimeError) as failure:
            error = str(failure)
        result = {**case, "answer": answer, "error": error,
                  "seconds": round(time.perf_counter() - started, 2)}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Captures à examiner : {destination}", file=sys.stderr)
    return 0 if all(result["error"] is None for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
