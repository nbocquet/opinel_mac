"""Export Opinel's N8 weights, then use OVH's MLX conversion recipe."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from opinel_mac import MODEL, REVISION, ROOT, dequantize, hf_parameter_name, validate_recipe

PROVENANCE = "OPINEL_PROVENANCE.json"
PINNED_SHA256 = "fa5de48d7fd4bc422ecb409c52883561da665a68411927f9bfbd416e0b50c775"


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def reference_tensors(reader):
    import torch

    keys = set(reader.keys())
    used_scales = set()
    seen = set()
    for name in sorted(keys):
        if name.endswith(".weight_scale"):
            continue
        value = reader.get_tensor(name)
        if value.dtype == torch.int8:
            if not name.endswith(".weight"):
                raise ValueError(f"Tenseur INT8 inattendu : {name}")
            scale = name.removesuffix(".weight") + ".weight_scale"
            if scale not in keys:
                raise ValueError(f"Échelle manquante : {scale}")
            value = dequantize(value, reader.get_tensor(scale), torch.bfloat16)
            used_scales.add(scale)
        target = hf_parameter_name(name)
        if target in seen:
            raise ValueError(f"Paramètre dupliqué : {target}")
        seen.add(target)
        yield target, value.contiguous()
    if used_scales != {key for key in keys if key.endswith(".weight_scale")}:
        raise ValueError("Échelles inutilisées dans le checkpoint source.")


def verify_reference(source, exported):
    """Compare every exported tensor with the PyTorch runner's exact BF16 input."""
    import torch
    from safetensors import safe_open

    index = json.loads((exported / "model.safetensors.index.json").read_text())["weight_map"]
    observed = set()
    with ExitStack() as stack:
        readers = {name: stack.enter_context(safe_open(exported / name, framework="pt"))
                   for name in set(index.values())}
        stored = set()
        for filename, reader in readers.items():
            for name in reader.keys():
                if name in stored or index.get(name) != filename:
                    raise ValueError(f"Index de poids incohérent : {name}")
                stored.add(name)
        original = stack.enter_context(safe_open(source / "model.safetensors", framework="pt"))
        for name, expected in reference_tensors(original):
            if name not in index:
                raise ValueError(f"Tenseur exporté manquant : {name}")
            actual = readers[index[name]].get_tensor(name)
            if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                raise ValueError(f"Tenseur différent après export : {name}")
            observed.add(name)
    if observed != set(index) or observed != stored:
        raise ValueError("Tenseurs supplémentaires ou absents après export.")
    return len(observed)


def export_reference(source, destination, shard_bytes=512 * 1024 ** 2):
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, destination = Path(source).resolve(), Path(destination).resolve()
    config = json.loads((source / "config.json").read_text())
    validate_recipe(config)
    digest = file_sha256(source / "model.safetensors")
    if destination.exists():
        manifest = json.loads((destination / PROVENANCE).read_text())
        if manifest["source_sha256"] != digest:
            raise ValueError("Le dossier BF16 existant provient d'autres poids. Choisir un autre dossier.")
        count = verify_reference(source, destination)
        print(f"Export BF16 existant vérifié : {count} tenseurs.", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    try:
        weight_map, pending, size, total, part = {}, {}, 0, 0, 0

        def flush():
            nonlocal pending, size, part
            if not pending:
                return
            part += 1
            name = f"model-{part:05d}.safetensors"
            save_file(pending, staging / name, metadata={"format": "pt"})
            weight_map.update({key: name for key in pending})
            pending, size = {}, 0
            print(f"Export BF16 : fragment {part} écrit.", flush=True)

        with safe_open(source / "model.safetensors", framework="pt") as reader:
            for name, value in reference_tensors(reader):
                nbytes = value.numel() * value.element_size()
                if pending and size + nbytes > shard_bytes:
                    flush()
                pending[name] = value
                size += nbytes
                total += nbytes
            flush()
        (staging / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": total}, "weight_map": weight_map,
        }, indent=2) + "\n")
        config.pop("quantization_config", None)
        config.pop("auto_map", None)
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                     "generation_config.json", "LICENSE", "NOTICE"):
            if (source / name).exists():
                shutil.copyfile(source / name, staging / name)
        count = verify_reference(source, staging)
        provenance = {
            "source_repository": MODEL,
            "source_revision": REVISION if digest == PINNED_SHA256 else None,
            "source_sha256": digest,
            "format": "bf16-reference",
            "verified_tensors": count,
            "verification": "All tensors equal the PyTorch BF16 reference, including original FP32 tensors.",
        }
        (staging / PROVENANCE).write_text(json.dumps(provenance, indent=2) + "\n")
        # Publish the folder only once the complete, verified export is available.
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
        print(f"Export vérifié : {count} tenseurs, {total / 1e9:.2f} Go.", flush=True)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_conversion(args):
    source = args.model_dir or ROOT / "models" / "opinel-n8"
    python = ROOT / ".venv-mlx" / "bin" / "python"
    if args.recipe == "n8" and not args.export_only:
        # Direct transcription: no BF16 export, no requantization (issue #1, §1).
        if not python.exists():
            raise ValueError("Installer l'environnement MLX selon README.md.")
        output = args.output_dir or ROOT / "models" / "opinel-n8-mlx-n8"
        return subprocess.run([
            str(python), str(ROOT / "scripts" / "convert_mlx_n8.py"),
            "--source", str(source), "--output", str(output),
        ], check=False).returncode
    reference = ROOT / "models" / "opinel-n8-mlx-bf16"
    if args.recipe == "bf16" and args.output_dir:
        reference = args.output_dir
    reference = export_reference(source, reference)
    if args.export_only or args.recipe == "bf16":
        return 0
    if not python.exists():
        raise ValueError("Installer l'environnement MLX selon README.md ; l'export BF16 est prêt.")
    output = args.output_dir or ROOT / "models" / "opinel-n8-mlx-q8"
    return subprocess.run([
        str(python), str(ROOT / "scripts" / "convert_mlx.py"),
        "--source", str(reference), "--output", str(output),
    ], check=False).returncode
