"""Numerical parity, following the Opinel team's method (issue #1, §3 and §6).

1. ``generate``: the backend under test continues about sixty varied prompts
   greedily (256 tokens) and records its own top-64 log-probabilities at each
   step, on its real decoding path.
2. ``capture``: a reference reads the same continuations back in teacher
   forcing (one prefill) and records its top-5 at each position. The reference
   is PyTorch on CPU in FP32 with INT8 experts (``--precision fp32``), or in
   BF16 for the "BF16 floor" line.
3. ``compare``: top-1 agreement, mean |Δ log p| on the reference's top-5, and
   clear errors (reference top-1 with p > 0.5 but a different generated token).

Team's reference points on N°8 (GPU, 64 prompts × 256 tokens): top-1 agreement
97.7-97.9 % and top-5 delta 0.098-0.104; BF16 floor on CPU 97.9-98.2 % and
0.089-0.091.
"""

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

CANDIDATE_TOP = 64
REFERENCE_TOP = 5
CLEAR_ERROR_PROBABILITY = 0.5
SUITES = ("reliability_fr.json", "reliability_fr_holdout.json")


def log_softmax(logits):
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def top_entries(log_probs, count):
    """Indices and values of the `count` largest entries per row, sorted."""
    import numpy as np

    index = np.argpartition(-log_probs, count - 1, axis=-1)[..., :count]
    values = np.take_along_axis(log_probs, index, axis=-1)
    order = np.argsort(-values, axis=-1, kind="stable")
    return np.take_along_axis(index, order, axis=-1), np.take_along_axis(values, order, axis=-1)


def parity_metrics(candidate_index, candidate_lp, reference_index, reference_lp, valid):
    """Compare a candidate's recorded top-K with a reference's top-5 on the same positions.

    candidate_index/lp: (..., K), the candidate's own distribution (index 0 is its
    greedy token). reference_index/lp: (..., 5). valid: boolean mask of positions.
    """
    import numpy as np

    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        raise ValueError("Aucune position à comparer.")
    for array in (candidate_lp, reference_lp):
        if not np.isfinite(array[valid]).all():
            raise ValueError("Log-probabilités non finies.")
    agreement = candidate_index[..., 0] == reference_index[..., 0]
    # Candidate log-probability at each reference top-5 token, from its recorded top-K.
    match = candidate_index[..., None, :] == reference_index[..., :, None]   # (..., 5, K)
    found = match.any(axis=-1)
    position = match.argmax(axis=-1)                                         # (..., 5)
    recorded = np.take_along_axis(candidate_lp, position, axis=-1)
    # Outside the candidate's top-K, its K-th value bounds the true log-probability
    # from above: the delta is then underestimated, and the count is reported.
    looked_up = np.where(found, recorded, candidate_lp[..., -1:])
    delta = np.abs(reference_lp - looked_up)
    clear = (np.exp(reference_lp[..., 0]) > CLEAR_ERROR_PROBABILITY) & ~agreement
    return {
        "positions": int(valid.sum()),
        "top1_agreement": float(agreement[valid].mean()),
        "top5_mean_abs_log_probability_delta": float(delta[valid].mean()),
        "clear_errors": int(clear[valid].sum()),
        "clear_error_rate": float(clear[valid].mean()),
        "reference_top5_outside_candidate_top_k": int((~found)[valid].sum()),
    }


def build_prompts(tokenizer, reasoning_effort="low"):
    """Reliability cases, varied short prompts and two long documents, as token lists."""
    prompts = []

    def add(prompt_id, messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             reasoning_effort=reasoning_effort)
        prompts.append({"id": prompt_id, "tokens": tokenizer.encode(text, add_special_tokens=False)})

    for suite in SUITES:
        for case in load_suite(ROOT / "evaluation" / suite)["cases"]:
            add(case["id"], case["messages"])
    spec = json.loads((ROOT / "evaluation" / "parity_prompts.json").read_text())
    for number, text in enumerate(spec["prompts"], 1):
        add(f"p{number:02d}", [{"role": "user", "content": text}])
    for document in spec["long_documents"]:
        body = tokenizer.encode((ROOT / document["path"]).read_text(), add_special_tokens=False)
        if len(body) < document["tokens"]:
            raise ValueError(f"{document['path']} compte moins de {document['tokens']} tokens.")
        excerpt = tokenizer.decode(body[:document["tokens"]])
        add(document["id"], [{"role": "user", "content": f"{excerpt}\n\n{document['question']}"}])
    return prompts


def fingerprint(prompts):
    return hashlib.sha256(json.dumps([[p["id"], p["tokens"]] for p in prompts]).encode()).hexdigest()


def environment():
    return {name: os.environ.get(name) for name in
            ("OLALA_DETERMINISTIC", "OLALA_TPA_CACHE", "OLALA_MLX_REF", "OLALA_TPA_TGS")}


def load_backend(args):
    """Return (model, tokenizer, metadata) for the requested backend and precision."""
    if args.backend == "mlx":
        from opinel_mlx import default_folder, load_mlx_model, validate_provenance
        folder = args.model_dir or default_folder()
        provenance = validate_provenance(folder)
        model, tokenizer = load_mlx_model(folder, args.device)
        return model, tokenizer, {"backend": "mlx", "format": provenance["format"],
                                  "source_sha256": provenance["source_sha256"],
                                  "model_dir": str(Path(folder).resolve())}
    from transformers import AutoTokenizer
    from opinel_conversion import file_sha256
    folder = args.model_dir or ROOT / "models" / "opinel-n8"
    model = load_model(folder, args.device, context_length=8192, reference_fp32=args.precision == "fp32")
    tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True, local_files_only=True)
    return model, tokenizer, {"backend": "torch", "format": f"torch-{args.precision}",
                              "source_sha256": file_sha256(folder / "model.safetensors"),
                              "model_dir": str(Path(folder).resolve())}


def generate_mlx(model, tokenizer, prompt, steps):
    import mlx.core as mx
    import numpy as np
    from mlx_lm.generate import generate_step

    tokens, top_index, top_lp = [], [], []
    for token, logprobs in generate_step(mx.array(prompt), model, max_tokens=steps,
                                         sampler=lambda lp: mx.argmax(lp, axis=-1)):
        row = np.array(logprobs.astype(mx.float32)).reshape(-1)
        index, values = top_entries(log_softmax(row), CANDIDATE_TOP)
        tokens.append(int(token))
        top_index.append(index)
        top_lp.append(values)
        if int(token) in tokenizer.eos_token_ids:
            break
    return tokens, top_index, top_lp


def generate_torch(model, prompt, steps, device):
    import torch

    with torch.inference_mode():
        output = model.generate(torch.tensor([prompt], device=device), max_new_tokens=steps, do_sample=False,
                                use_cache=True, output_logits=True, return_dict_in_generate=True)
    tokens = output.sequences[0, len(prompt):].tolist()
    top_index, top_lp = [], []
    for logits in output.logits:
        index, values = top_entries(log_softmax(logits[0].float().cpu().numpy()), CANDIDATE_TOP)
        top_index.append(index)
        top_lp.append(values)
    return tokens, top_index, top_lp


def run_generate(args):
    import numpy as np

    model, tokenizer, metadata = load_backend(args)
    prompts = build_prompts(tokenizer)
    if args.limit:
        prompts = prompts[:args.limit]
    count, steps = len(prompts), args.max_new_tokens
    continuation = np.full((count, steps), -1, dtype=np.int64)
    candidate_index = np.zeros((count, steps, CANDIDATE_TOP), dtype=np.int64)
    candidate_lp = np.full((count, steps, CANDIDATE_TOP), -np.inf, dtype=np.float32)
    for row, prompt in enumerate(prompts):
        print(f"Génération {row + 1}/{count} : {prompt['id']} ({len(prompt['tokens'])} tokens)", flush=True)
        if args.backend == "mlx":
            tokens, index, values = generate_mlx(model, tokenizer, prompt["tokens"], steps)
        else:
            tokens, index, values = generate_torch(model, prompt["tokens"], steps, args.device)
        length = len(tokens)
        continuation[row, :length] = tokens
        candidate_index[row, :length] = np.stack(index)
        candidate_lp[row, :length] = np.stack(values)
    metadata.update({"kind": "generation", "device": args.device, "steps": steps, "environment": environment(),
                     "prompts_sha256": fingerprint(prompts),
                     "prompts": [{"id": p["id"], "length": len(p["tokens"])} for p in prompts]})
    flat = np.array([token for p in prompts for token in p["tokens"]], dtype=np.int64)
    save(args.output, metadata, prompt_tokens=flat, continuation=continuation,
         candidate_index=candidate_index, candidate_lp=candidate_lp)


def prompt_lists(metadata, flat):
    prompts, start = [], 0
    for entry in metadata["prompts"]:
        prompts.append({"id": entry["id"], "tokens": flat[start:start + entry["length"]].tolist()})
        start += entry["length"]
    if start != len(flat):
        raise ValueError("Tokens de prompt incohérents avec les métadonnées.")
    return prompts


def run_capture(args):
    import numpy as np
    import torch

    generation, arrays = load(args.generation)
    if generation.get("kind") != "generation":
        raise ValueError("Le fichier fourni n'est pas une génération.")
    prompts = prompt_lists(generation, arrays["prompt_tokens"])
    if fingerprint(prompts) != generation["prompts_sha256"]:
        raise ValueError("Empreinte des prompts invalide.")
    model, _, metadata = load_backend(args)
    if metadata["source_sha256"] != generation["source_sha256"]:
        raise ValueError("Poids sources différents : ce ne serait pas un contrôle de parité.")
    continuation = arrays["continuation"]
    count, steps = continuation.shape
    reference_index = np.zeros((count, steps, REFERENCE_TOP), dtype=np.int64)
    reference_lp = np.full((count, steps, REFERENCE_TOP), -np.inf, dtype=np.float32)
    for row, prompt in enumerate(prompts):
        length = int((continuation[row] >= 0).sum())
        if not length:
            continue
        print(f"Relecture {row + 1}/{count} : {prompt['id']}", flush=True)
        sequence = prompt["tokens"] + continuation[row, :length].tolist()
        with torch.inference_mode():
            # Logits at position i predict token i + 1: keep the last `length + 1`, drop the final one.
            logits = model(input_ids=torch.tensor([sequence], device=args.device), use_cache=False,
                           logits_to_keep=length + 1).logits[0, :-1]
        index, values = top_entries(log_softmax(logits.float().cpu().numpy()), REFERENCE_TOP)
        reference_index[row, :length] = index
        reference_lp[row, :length] = values
    metadata.update({"kind": "reference", "device": args.device,
                     "generation_sha256": file_digest(args.generation), "environment": environment()})
    save(args.output, metadata, reference_index=reference_index, reference_lp=reference_lp)


def compare(generation, candidate, reference_metadata, reference):
    import numpy as np

    valid = candidate["continuation"] >= 0
    longs = np.array([entry["id"].startswith("long_") for entry in generation["prompts"]])
    result = {"candidate": {key: generation.get(key) for key in ("backend", "format", "device", "environment")},
              "reference": {key: reference_metadata.get(key) for key in ("backend", "format", "device")},
              "prompts": int(len(valid)), "steps": generation["steps"]}
    for name, rows in (("tous", np.ones(len(valid), dtype=bool)), ("courts", ~longs), ("longs", longs)):
        if rows.any():
            result[name] = parity_metrics(candidate["candidate_index"][rows], candidate["candidate_lp"][rows],
                                          reference["reference_index"][rows], reference["reference_lp"][rows],
                                          valid[rows])
    result["repères_opinel_n8"] = {
        "chemins_gpu": {"top1_agreement": "0.977-0.979", "top5_delta": "0.098-0.104"},
        "plancher_bf16_cpu": {"top1_agreement": "0.979-0.982", "top5_delta": "0.089-0.091"}}
    return result


def run_compare(args):
    generation, candidate = load(args.generation)
    reference_metadata, reference = load(args.reference)
    if (reference_metadata.get("kind") != "reference"
            or reference_metadata["generation_sha256"] != file_digest(args.generation)):
        raise ValueError("La référence n'a pas été capturée sur cette génération.")
    print(json.dumps(compare(generation, candidate, reference_metadata, reference), ensure_ascii=False, indent=2))


def file_digest(path):
    from opinel_conversion import file_sha256
    return file_sha256(path)


def save(path, metadata, **arrays):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, metadata=json.dumps(metadata), **arrays)
    print(f"Enregistré : {path}")


def load(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata"])), {key: data[key] for key in data.files if key != "metadata"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Continuations greedy du backend testé")
    generate.add_argument("--backend", choices=["torch", "mlx"], required=True)
    generate.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    generate.add_argument("--model-dir", type=Path)
    generate.add_argument("--precision", choices=["bf16", "fp32"], default="bf16", help="PyTorch seulement")
    generate.add_argument("--reference", action="store_true", help="MLX sans kernels fusionnés")
    generate.add_argument("--max-new-tokens", type=int, default=256)
    generate.add_argument("--limit", type=int, help="Premiers prompts seulement (essai rapide)")
    generate.add_argument("--output", type=Path, required=True)
    capture = commands.add_parser("capture", help="Relecture en teacher forcing par la référence PyTorch")
    capture.add_argument("generation", type=Path)
    capture.add_argument("--precision", choices=["bf16", "fp32"], default="fp32")
    capture.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    capture.add_argument("--model-dir", type=Path)
    capture.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("generation", type=Path)
    comparison.add_argument("reference", type=Path)
    args = parser.parse_args()
    if getattr(args, "output", None) is not None and args.output.exists():
        parser.error("Le fichier de sortie existe déjà ; choisir un autre chemin.")
    if args.command == "compare":
        run_compare(args)
        return 0
    configure_environment()
    # Floating-point atomics in the fused MoE otherwise vary between runs (§6).
    os.environ["OLALA_DETERMINISTIC"] = "1"
    if args.command == "generate":
        if args.reference:
            if args.backend != "mlx":
                parser.error("--reference nécessite --backend mlx.")
            os.environ["OLALA_MLX_REF"] = "1"
        if args.backend == "mlx" and args.precision != "bf16":
            parser.error("--precision ne s'applique qu'à PyTorch.")
        run_generate(args)
    else:
        args.backend = "torch"
        run_capture(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, ImportError, FileNotFoundError) as error:
        print(f"Parité non vérifiée : {error}", file=sys.stderr)
        raise SystemExit(2)
