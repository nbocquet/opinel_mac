"""Transcribe Opinel N8 directly into MLX's affine 8-bit format, without requantizing.

N8 stores each linear layer as INT8 codes q in [-127, 127] with one FP32 scale per
output row: w = s * q. MLX's affine format is w = scale * u + bias with u in
[0, 255], so u = q + 128, scale = s and bias = -128 * s (issue #1, §1 and §7).
The only rounding is the scale's, stored in bf16 (at most 2^-8 relative, the same
for a whole row). Unlike the Q8 recipe, no weight moves to a new grid.

Kept in floating point:
  - the embedding (BF16 in N8: one row read per token);
  - in_proj_small (W_A_k, W_A_v, shift_proj_k, shift_proj_v, lambda_proj): OVH's
    module keeps it unquantized and casts its input to the weight dtype, so it is
    dequantized exactly to FP32;
  - every other non-INT8 tensor (router, norms, geodesic scalars), unchanged.

Only mlx is needed: this runs in .venv-mlx, on CPU, without Metal.
"""

import json
from pathlib import Path
import shutil
import tempfile

from opinel_mac import MODEL, REVISION, hf_parameter_name, validate_recipe

GROUP_SIZE = 128
FORMAT = "mlx-n8"
IN_PROJ_SMALL = ("W_A_k", "W_A_v", "shift_proj_k", "shift_proj_v", "lambda_proj")
SCALE_TOLERANCE = 2.0 ** -8  # bf16 round-to-nearest, relative
COPIED_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "generation_config.json", "LICENSE", "NOTICE")


def tensor_plan(name, is_int8):
    """'quantize', 'float32' (exact dequantization) or 'keep' for a source tensor."""
    if name.endswith(".weight_scale"):
        raise ValueError(f"Échelle traitée avec son poids : {name}")
    if not is_int8:
        return "keep"
    if not name.endswith(".weight"):
        raise ValueError(f"Tenseur INT8 inattendu : {name}")
    module = name.removesuffix(".weight").rsplit(".", 1)[-1]
    if module in IN_PROJ_SMALL:
        return "float32"
    if module == "moe_gate" or module == "embedding":
        raise ValueError(f"{name} devrait être en flottant dans N8.")
    return "quantize"


def n8_affine(q, s, group_size=GROUP_SIZE):
    """q: int8 [..., N, K], s: float32 [..., N] -> (packed weight, scales, biases)."""
    import mlx.core as mx

    if q.dtype != mx.int8 or s.shape != q.shape[:-1]:
        raise ValueError("Poids INT8 ou dimensions des échelles N8 invalides.")
    if q.shape[-1] % group_size:
        raise ValueError(f"Dimension d'entrée {q.shape[-1]} non divisible par {group_size}.")
    if mx.any(q == -128).item():
        raise ValueError("Code INT8 -128 hors du format N8 symétrique.")
    codes = (q.astype(mx.int16) + 128).astype(mx.uint8)
    weight = mx.view(codes, mx.uint32)  # four codes per word, little-endian: MLX's layout
    scales = mx.repeat(s.astype(mx.bfloat16)[..., None], q.shape[-1] // group_size, axis=-1)
    return weight, scales, -128 * scales  # -128·s is exact in bf16


def check_affine(q, s, weight, scales, biases, group_size=GROUP_SIZE):
    """Dequantize in FP32 and require exactly q * bf16(s), with bf16(s) close to s."""
    import mlx.core as mx

    rounded = s.astype(mx.bfloat16).astype(mx.float32)
    error = mx.abs(rounded - s) / mx.maximum(mx.abs(s), mx.array(1e-30, mx.float32))
    worst = mx.max(error).item() if error.size else 0.0
    if worst > SCALE_TOLERANCE:
        raise ValueError(f"Échelle bf16 trop éloignée : écart relatif {worst:.3g}.")
    # In FP32, u·s_bf16 and -128·s_bf16 are exact, so dequantization must equal q·s_bf16.
    restored = mx.dequantize(weight, scales.astype(mx.float32), biases.astype(mx.float32),
                             group_size=group_size, bits=8)
    if not mx.array_equal(restored, q.astype(mx.float32) * rounded[..., None]).item():
        raise ValueError("La déquantification MLX ne reproduit pas q × échelle.")
    return worst


def transcribe(source_tensors, group_size=GROUP_SIZE):
    """Yield (name, array) for the MLX checkpoint, plus the worst scale rounding seen."""
    import mlx.core as mx

    names = set(source_tensors)
    used_scales, seen = set(), set()
    report = {"quantized": 0, "float32": 0, "kept": 0, "max_scale_rounding": 0.0}
    for name in sorted(names):
        if name.endswith(".weight_scale"):
            continue
        value = source_tensors[name]
        plan = tensor_plan(name, value.dtype == mx.int8)
        target = hf_parameter_name(name)
        if plan == "keep":
            outputs = {target: value}
            report["kept"] += 1
        else:
            scale_name = name.removesuffix(".weight") + ".weight_scale"
            if scale_name not in names:
                raise ValueError(f"Échelle manquante : {scale_name}")
            scale = source_tensors[scale_name].astype(mx.float32)
            used_scales.add(scale_name)
            if plan == "float32":
                outputs = {target: value.astype(mx.float32) * scale[..., None]}
                report["float32"] += 1
            else:
                weight, scales, biases = n8_affine(value, scale, group_size)
                worst = check_affine(value, scale, weight, scales, biases, group_size)
                report["max_scale_rounding"] = max(report["max_scale_rounding"], worst)
                prefix = target.removesuffix(".weight")
                outputs = {target: weight, f"{prefix}.scales": scales, f"{prefix}.biases": biases}
                report["quantized"] += 1
        for key, array in outputs.items():
            if key in seen:
                raise ValueError(f"Paramètre dupliqué : {key}")
            seen.add(key)
            yield key, array
    if used_scales != {name for name in names if name.endswith(".weight_scale")}:
        raise ValueError("Échelles inutilisées dans le checkpoint source.")
    yield None, report


def mlx_config(config, group_size=GROUP_SIZE):
    validate_recipe(config)
    result = {key: value for key, value in config.items() if key not in ("quantization_config", "auto_map")}
    # mlx-lm only quantizes the modules whose .scales are present in the weights.
    result["quantization"] = {"group_size": group_size, "bits": 8, "mode": "affine"}
    return result


def convert_n8(source, destination, group_size=GROUP_SIZE, shard_bytes=512 * 1024 ** 2, load_check=True):
    import mlx.core as mx
    from opinel_conversion import PINNED_SHA256, PROVENANCE, file_sha256

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"{destination} existe déjà ; choisir un nouveau dossier.")
    config = mlx_config(json.loads((source / "config.json").read_text()), group_size)
    digest = file_sha256(source / "model.safetensors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    try:
        tensors = mx.load(str(source / "model.safetensors"))
        weight_map, pending, size, total, part, report = {}, {}, 0, 0, 0, None

        def flush():
            nonlocal pending, size, part
            if not pending:
                return
            part += 1
            name = f"model-{part:05d}.safetensors"
            mx.save_safetensors(str(staging / name), pending, metadata={"format": "mlx"})
            weight_map.update({key: name for key in pending})
            pending, size = {}, 0
            print(f"Conversion N8 : fragment {part} écrit.", flush=True)

        for name, array in transcribe(tensors, group_size):
            if name is None:
                report = array
                continue
            mx.eval(array)
            if pending and size + array.nbytes > shard_bytes:
                flush()
            pending[name] = array
            size += array.nbytes
            total += array.nbytes
        flush()
        del tensors
        (staging / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": total}, "weight_map": weight_map,
        }, indent=2) + "\n")
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for name in COPIED_FILES:
            if (source / name).exists():
                shutil.copyfile(source / name, staging / name)
        verified = verify_written(source, staging, group_size)
        if load_check:
            # Strict native loading catches a wrong name, shape or merged projection.
            from vendor.ovh_olala import register
            register()
            from mlx_lm import load
            model, _ = load(str(staging), lazy=True)
            mx.eval(model.parameters())
            del model
        provenance = {
            "source_repository": MODEL,
            "source_revision": REVISION if digest == PINNED_SHA256 else None,
            "source_sha256": digest,
            "format": FORMAT, "bits": 8, "group_size": group_size,
            "verified_tensors": verified,
            "quantized_tensors": report["quantized"],
            "dequantized_float32_tensors": report["float32"],
            "max_scale_rounding": report["max_scale_rounding"],
            "quantization": "N8 INT8 codes transcribed as MLX affine 8-bit (u = q + 128, bias = -128·s); "
                            "only the per-row scale is rounded to bf16.",
            "verification": "Each quantized tensor dequantizes in FP32 to exactly q × bf16(scale); "
                            "floating tensors equal the source; strict mlx-lm load.",
        }
        if (Path(__file__).parent / "vendor/ovh_olala/UPSTREAM.json").exists():
            provenance["engine_revision"] = json.loads(
                (Path(__file__).parent / "vendor/ovh_olala/UPSTREAM.json").read_text())["revision"]
        (staging / PROVENANCE).write_text(json.dumps(provenance, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
        print(f"Opinel MLX N8 prêt : {destination} ({verified} tenseurs vérifiés, {total / 1e9:.2f} Go).",
              flush=True)
        return provenance
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_written(source, exported, group_size=GROUP_SIZE):
    """Reread the written shards and compare them with a fresh transcription of the source."""
    import mlx.core as mx

    index = json.loads((exported / "model.safetensors.index.json").read_text())["weight_map"]
    written = {}
    for filename in sorted(set(index.values())):
        for name, array in mx.load(str(exported / filename)).items():
            if name in written or index.get(name) != filename:
                raise ValueError(f"Index de poids incohérent : {name}")
            written[name] = array
    expected = set()
    for name, array in transcribe(mx.load(str(source / "model.safetensors")), group_size):
        if name is None:
            continue
        if name not in written:
            raise ValueError(f"Tenseur écrit manquant : {name}")
        if written[name].dtype != array.dtype or not mx.array_equal(written[name], array).item():
            raise ValueError(f"Tenseur différent après écriture : {name}")
        expected.add(name)
    if expected != set(written) or expected != set(index):
        raise ValueError("Tenseurs supplémentaires ou absents après écriture.")
    return len(expected)
