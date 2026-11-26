"""Run, review and compare the fixed 30-case French reliability suite."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opinel_mac import configure_environment, generate_turn, load_model
from instructions import PROFILES, describe
from reliability import annotate_report, compare_reports, load_suite, run_evaluation, summarize, verify_assets

SUITES = {"principale": "reliability_fr.json", "temoin": "reliability_fr_holdout.json"}

PRESETS = {
    "baseline": {"reasoning_effort": "low", "temperature": 0.0, "top_p": 1.0, "top_k": 0,
                 "max_new_tokens": 1024, "context_length": 8192},
    "ovh": {"reasoning_effort": "medium", "temperature": 0.8, "top_p": 0.9, "top_k": 40,
            "max_new_tokens": 4096, "context_length": 8192},
}


def print_summary(report):
    print(json.dumps(summarize(report), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    run.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    run.add_argument("--model-kind", choices=["opinel", "olala-original"], default="opinel")
    run.add_argument("--model-dir", type=Path)
    run.add_argument("--preset", choices=PRESETS, default="baseline")
    run.add_argument("--reference", action="store_true", help="MLX sans les kernels Metal fusionnés OVH")
    run.add_argument("--instructions", choices=PROFILES, default="aucune",
                     help="Consignes développeur envoyées avant chaque conversation")
    run.add_argument("--suite", choices=SUITES, default="principale",
                     help="principale : 30 cas de réglage ; temoin : cas réservés à la validation")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-new-tokens", type=int)
    run.add_argument("--case", action="append")
    run.add_argument("--output", type=Path)
    run.add_argument("--resume", action="store_true")
    review = commands.add_parser("review")
    review.add_argument("report", type=Path)
    review.add_argument("--case")
    review.add_argument("--verdict", choices=["pass", "fail", "uncertain"])
    review.add_argument("--note")
    review.add_argument("--reviewer", default="user")
    review.add_argument("--kind", choices=["human", "assistant"], default="human")
    compare = commands.add_parser("compare")
    compare.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "compare":
        comparison = compare_reports([json.loads(path.read_text()) for path in args.reports])
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return 0
    if args.command == "review":
        if args.verdict:
            if not args.case or not args.note:
                parser.error("Une annotation nécessite --case et --note.")
            report = annotate_report(args.report, args.case, args.verdict, args.note, args.reviewer, args.kind)
        else:
            report = json.loads(args.report.read_text())
            for result in report["results"]:
                if args.case and result["id"] != args.case:
                    continue
                print(json.dumps(result, ensure_ascii=False, indent=2))
        print_summary(report)
        return 0

    suite = load_suite(ROOT / "evaluation" / SUITES[args.suite])
    if args.case and set(args.case) - {case["id"] for case in suite["cases"]}:
        parser.error("Identifiant de cas inconnu.")
    if args.reference and args.backend != "mlx":
        parser.error("--reference nécessite --backend mlx.")
    if args.model_kind == "olala-original" and args.backend != "torch":
        parser.error("La référence originale utilise ici le même moteur PyTorch que la référence Opinel.")
    settings = dict(PRESETS[args.preset])
    if args.max_new_tokens is not None:
        settings["max_new_tokens"] = args.max_new_tokens
    if not 1 <= settings["max_new_tokens"] < settings["context_length"]:
        parser.error("Le budget de génération doit être positif et inférieur au contexte.")
    for key, value in settings.items():
        setattr(args, key, value)
    configure_environment()
    # Floating-point atomics in the fused MoE vary between runs (issue #1, §6).
    os.environ["OLALA_DETERMINISTIC"] = "1"
    if args.reference:
        os.environ["OLALA_MLX_REF"] = "1"
    if args.backend == "mlx":
        from opinel_mlx import default_folder
        folder = args.model_dir or default_folder()
    else:
        folder = args.model_dir or ROOT / "models" / ("olala-original" if args.model_kind == "olala-original" else "opinel-n8")
    config = {"backend": args.backend, "device": args.device, "model_kind": args.model_kind,
              "model_dir": str(folder.resolve()), "preset": args.preset, "seed": args.seed,
              "reference_kernels": args.backend == "mlx" and (args.reference or os.environ.get("OLALA_MLX_REF") == "1" or args.device == "cpu"),
              **settings}
    if args.backend == "mlx":
        config["deterministic_kernels"] = True
        config["tpa_cache"] = os.environ.get("OLALA_TPA_CACHE", "auto")
    # Absent without instructions, so earlier reports keep the same identity.
    if describe(args.instructions):
        config["instructions"] = describe(args.instructions)
    provenance = folder / ("EVALUATION_REFERENCE.json" if args.model_kind == "olala-original" else "OPINEL_PROVENANCE.json")
    if provenance.exists():
        config["provenance"] = json.loads(provenance.read_text())
    if args.output is None:
        if args.resume:
            parser.error("--resume nécessite --output.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = ROOT / ".cache" / "evaluations" / f"{args.model_kind}-{args.backend}-{args.preset}-{args.instructions}-{stamp}.json"
    # Validate resume/overwrite before loading the model, using the same identity contract.
    selected = [case["id"] for case in suite["cases"] if args.case is None or case["id"] in args.case]
    if args.output.exists():
        old = json.loads(args.output.read_text())
        if not args.resume or old.get("suite_sha256") != suite["sha256"] or old.get("configuration") != config or old.get("case_ids") != selected:
            parser.error("Rapport existant incompatible ; utiliser un nouveau chemin ou une reprise identique.")
    elif args.resume:
        parser.error("Rapport à reprendre absent.")
    print(f"Rapport : {args.output}", flush=True)
    if args.backend == "mlx":
        from opinel_mlx import load_mlx_model
        model, tokenizer = load_mlx_model(folder, args.device)
    else:
        from transformers import AutoTokenizer
        if args.model_kind == "olala-original":
            expected = json.loads((ROOT / "evaluation/original_reference.json").read_text())
            proof = config.get("provenance", {})
            if (not proof.get("verified") or proof.get("repository") != expected["repository"]
                    or proof.get("revision") != expected["revision"]):
                parser.error("Référence originale non vérifiée. Préparer EVALUATION_REFERENCE.json.")
            verify_assets(folder, expected)
            tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True, local_files_only=True)
            model = load_model(folder, args.device, context_length=args.context_length,
                               unquantized_reference=True, architecture_folder=ROOT / "models" / "opinel-n8")
        else:
            tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True, local_files_only=True)
            model = load_model(folder, args.device, context_length=args.context_length)

    def generate(messages, seed):
        if args.backend == "mlx":
            import mlx.core as mx
            mx.random.seed(seed)
        else:
            import torch
            torch.manual_seed(seed)
        answer, _ = generate_turn(model, tokenizer, messages, args)
        return answer

    report = run_evaluation(suite, config, args.output, generate, args.case, args.resume,
                            progress=lambda line: print(line, flush=True))
    print_summary(report)
    print("Le score reste incomplet tant que des réponses attendent une relecture.")
    return 1 if report["summary"]["error"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, ImportError, FileNotFoundError) as error:
        print(f"Évaluation interrompue : {error}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("Évaluation interrompue ; les cas terminés sont conservés pour --resume.", file=sys.stderr)
        raise SystemExit(130)
