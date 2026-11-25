"""Download and verify the pinned unmodified OVH checkpoint for evaluation."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from opinel_mac import configure_environment
    from reliability import verify_assets
    configure_environment()
    manifest = json.loads((ROOT / "evaluation/original_reference.json").read_text())
    folder = ROOT / "models/olala-original"
    if any(not (folder / name).exists() for name in manifest["files"]):
        from huggingface_hub import snapshot_download
        snapshot_download(manifest["repository"], revision=manifest["revision"],
                          local_dir=folder, allow_patterns=list(manifest["files"]))
    verify_assets(folder, manifest, progress=lambda name: print(f"Vérifié : {name}", flush=True))
    proof = {"repository": manifest["repository"], "revision": manifest["revision"],
             "weights_sha256": manifest["files"]["model.safetensors"]["digest"],
             "verified": True, "implementation": "Shared Opinel PyTorch fallback; original weights and config."}
    (folder / "EVALUATION_REFERENCE.json").write_text(json.dumps(proof, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
