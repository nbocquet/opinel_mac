"""Apply OVH's Q8 recipe to the verified BF16 export of Opinel N8."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    manifest = json.loads((source / "OPINEL_PROVENANCE.json").read_text())
    if manifest.get("format") != "bf16-reference" or not manifest.get("verified_tensors"):
        parser.error("La source doit être un export BF16 Opinel vérifié.")
    if output.exists():
        parser.error("Le dossier de sortie existe déjà ; choisir un nouveau dossier.")
    try:
        import mlx.core as mx
    except ImportError as error:
        print(f"MLX indisponible : {error}\nRelancer la conversion depuis le terminal local du Mac.", file=sys.stderr)
        return 2
    from vendor.ovh_olala import register
    register()
    from vendor.ovh_olala.model import quant_predicate
    from mlx_lm.convert import convert
    from mlx_lm import load

    def predicate(path, module):
        if not hasattr(module, "to_quantized") or not quant_predicate(path, module):
            return False
        return {"bits": 8, "group_size": 64, "mode": "affine"}

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".partial-", dir=output.parent) as temporary:
        staging = Path(temporary) / "model"
        convert(str(source), str(staging), quantize=True, q_bits=8, q_group_size=64,
                dtype="bfloat16", quant_predicate=predicate)
        gc.collect()
        mx.clear_cache()
        # Mirror OVH's converter and keep the tokenizer independent of remote code.
        config = json.loads((staging / "config.json").read_text())
        config.pop("auto_map", None)
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        # Strict native loading catches incompatible fused projections/quantization.
        model, _ = load(str(staging), lazy=True)
        mx.eval(model.parameters())
        del model
        for name in ("LICENSE", "NOTICE"):
            if (source / name).exists():
                shutil.copyfile(source / name, staging / name)
        manifest.update({
            "format": "mlx-q8", "bits": 8, "group_size": 64,
            "quantization": "OVH affine Q8 recipe; additional rounding relative to N8 BF16.",
            "engine_revision": json.loads((ROOT / "vendor/ovh_olala/UPSTREAM.json").read_text())["revision"],
        })
        (staging / "OPINEL_PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    print(f"Opinel MLX Q8 prêt : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
