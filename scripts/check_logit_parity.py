"""Compare next-token distributions on fixed inputs, without sampling answers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opinel_mac import configure_environment, load_model
from reliability import load_suite


def metrics(reference, candidate):
    import numpy as np

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Dimensions de logits incompatibles.")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Logits non finis.")

    def log_probs(logits):
        logits = logits.astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))

    left, right = log_probs(reference), log_probs(candidate)
    kl = (np.exp(left) * (left - right)).sum(axis=-1)
    return {"positions": reference.shape[0],
            "top1_agreement": float((reference.argmax(-1) == candidate.argmax(-1)).mean()),
            "mean_kl_nats": float(kl.mean()), "max_kl_nats": float(kl.max()),
            "mean_abs_log_probability_delta": float(np.abs(left - right).mean()),
            "note": "Mesures numériques sur entrées fixes ; aucun seuil de qualité factuelle n'est déduit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--backend", choices=["torch", "mlx"], required=True)
    capture.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    capture.add_argument("--model-dir", type=Path)
    capture.add_argument("--reference", action="store_true", help="MLX sans kernels fusionnés")
    capture.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    args = parser.parse_args()
    import numpy as np

    if args.command == "compare":
        with np.load(args.reference, allow_pickle=False) as left, np.load(args.candidate, allow_pickle=False) as right:
            a, b = json.loads(str(left["metadata"])), json.loads(str(right["metadata"]))
            if a["inputs"] != b["inputs"] or a["source_sha256"] != b["source_sha256"]:
                parser.error("Entrées ou poids sources différents : ce ne serait pas un contrôle de parité.")
            result = {"reference": a, "candidate": b, "metrics": metrics(left["logits"], right["logits"])}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.reference and args.backend != "mlx":
        parser.error("--reference nécessite --backend mlx.")
    if args.output.exists():
        parser.error("La capture existe déjà ; choisir un autre chemin.")
    configure_environment()
    if args.reference:
        os.environ["OLALA_MLX_REF"] = "1"
    if args.backend == "mlx":
        import mlx.core as mx
        from opinel_mlx import load_mlx_model, validate_provenance
        folder = args.model_dir or ROOT / "models/opinel-n8-mlx-bf16"
        provenance = validate_provenance(folder)
        if provenance["format"] != "bf16-reference":
            parser.error("Utiliser la référence BF16 pour isoler les kernels de la quantification.")
        digest = provenance["source_sha256"]
        model, tokenizer = load_mlx_model(folder, args.device)
    else:
        import torch
        from transformers import AutoTokenizer
        from opinel_conversion import file_sha256
        folder = args.model_dir or ROOT / "models/opinel-n8"
        digest = file_sha256(folder / "model.safetensors")
        model = load_model(folder, args.device, context_length=8192)
        tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True, local_files_only=True)
    suite = load_suite(ROOT / "evaluation/reliability_fr.json")
    rows, inputs = [], []
    for case_id, answer in [("f01_moitie_moitie", "Gruyère et Vacherin Fribourgeois."),
                            ("r01_multiplication", "56"), ("i01_tutoie_moi", "D'accord, je te tutoie.")]:
        case = next(case for case in suite["cases"] if case["id"] == case_id)
        text = tokenizer.apply_chat_template(case["messages"], tokenize=False,
                                            add_generation_prompt=True, reasoning_effort="low")
        prompt = tokenizer.encode(text, add_special_tokens=False)
        forced = tokenizer.encode("<|channel_end|><|channel_start|>final<|content|>" + answer,
                                   add_special_tokens=False)[:8]
        fingerprint = hashlib.sha256(json.dumps([prompt, forced]).encode()).hexdigest()
        inputs.append({"case": case_id, "tokens_sha256": fingerprint, "positions": len(forced) + 1})
        print(f"Capture : {case_id}", flush=True)
        if args.backend == "torch":
            with torch.inference_mode():
                output = model(input_ids=torch.tensor([prompt], device=args.device), use_cache=True, logits_to_keep=1)
                rows.append(output.logits[0, -1].float().cpu().numpy())
                for index, token in enumerate(forced):
                    output = model(input_ids=torch.tensor([[token]], device=args.device),
                                   past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1,
                                   cache_position=torch.tensor([len(prompt) + index], device=args.device))
                    rows.append(output.logits[0, -1].float().cpu().numpy())
        else:
            cache = model.make_cache()
            logits = model(mx.array([prompt]), cache=cache)[0, -1].astype(mx.float32)
            mx.eval(logits, [entry.state for entry in cache])
            rows.append(np.array(logits))
            for token in forced:
                logits = model(mx.array([[token]]), cache=cache)[0, -1].astype(mx.float32)
                mx.eval(logits, [entry.state for entry in cache])
                rows.append(np.array(logits))
    metadata = {"backend": args.backend, "device": args.device, "model_dir": str(folder.resolve()),
                "reference_kernels": args.reference, "source_sha256": digest, "inputs": inputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, metadata=json.dumps(metadata), logits=np.stack(rows))
    print(f"Capture enregistrée : {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, ImportError, FileNotFoundError) as error:
        print(f"Parité non vérifiée : {error}", file=sys.stderr)
        raise SystemExit(2)
