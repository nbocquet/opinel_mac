"""Transcribe Opinel N8 directly into MLX affine 8-bit weights (no BF16 export, no requantization)."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Checkpoint Opinel N8 d'origine")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Le dossier de sortie existe déjà ; choisir un nouveau dossier.")
    try:
        import mlx.core as mx
    except ImportError as error:
        print(f"MLX indisponible : {error}", file=sys.stderr)
        return 2
    # The transcription and its checks need no GPU; CPU avoids depending on Metal.
    mx.set_default_device(mx.cpu)
    from opinel_n8_mlx import convert_n8
    try:
        convert_n8(args.source, args.output)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"Conversion interrompue : {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
