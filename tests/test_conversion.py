import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from opinel_conversion import PROVENANCE, export_reference, verify_reference
from opinel_mac import ROOT

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None


class VendorTests(unittest.TestCase):
    def test_ovh_sources_match_the_pinned_manifest(self):
        folder = ROOT / "vendor" / "ovh_olala"
        upstream = json.loads((folder / "UPSTREAM.json").read_text())
        self.assertEqual(upstream["revision"], "58083ca0ac4ca0654b508f74e1bdf7b6e0e173f4")
        for name, expected in upstream["files"].items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), expected, name)


@unittest.skipIf(torch is None, "PyTorch non installé")
class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        config = {"model_type": "olala", "auto_map": {"AutoConfig": "unused"},
                  "quantization_config": {"quant_algo": "MIXED_PRECISION", "quantized_layers": {
                      "model.layers.0.mlp.fc1_latent_proj": {"quant_algo": "W8A16"}}}}
        (self.source / "config.json").write_text(json.dumps(config))
        (self.source / "NOTICE").write_text("model notice")
        (self.source / "LICENSE").write_text("model license")
        self.weights = {
            "model.layers.0.mlp.fc1_latent_proj.weight": torch.tensor([[2, -4], [3, 1]], dtype=torch.int8),
            "model.layers.0.mlp.fc1_latent_proj.weight_scale": torch.tensor([0.5, 2.]),
            "model.layers.0.geodesic_mixer.scale": torch.tensor(0.123456789, dtype=torch.float32),
            "model.embedding.weight": torch.tensor([[0.5, -1.]], dtype=torch.bfloat16),
        }
        self.save_source()
        self.destination = self.root / "export"

    def save_source(self):
        save_file(self.weights, self.source / "model.safetensors")

    def test_sharded_export_preserves_dequantized_weights_f32_and_licenses(self):
        export_reference(self.source, self.destination, shard_bytes=8)
        self.assertEqual(verify_reference(self.source, self.destination), 3)
        index = json.loads((self.destination / "model.safetensors.index.json").read_text())
        self.assertGreater(len(set(index["weight_map"].values())), 1)
        stored = {}
        for filename in set(index["weight_map"].values()):
            stored.update(load_file(self.destination / filename))
        torch.testing.assert_close(stored["model.layers.0.mlp.down_proj.weight"],
                                   torch.tensor([[1., -2.], [6., 2.]], dtype=torch.bfloat16))
        scale = stored["model.layers.0.geodesic_mixer.scale"]
        self.assertEqual(scale.dtype, torch.float32)
        self.assertTrue(torch.equal(scale, self.weights["model.layers.0.geodesic_mixer.scale"]))
        self.assertEqual((self.destination / "NOTICE").read_text(), "model notice")
        config = json.loads((self.destination / "config.json").read_text())
        self.assertNotIn("quantization_config", config)
        self.assertNotIn("auto_map", config)
        self.assertEqual(json.loads((self.destination / PROVENANCE).read_text())["verified_tensors"], 3)

    def test_corrupt_export_is_rejected_instead_of_reused(self):
        export_reference(self.source, self.destination)
        filename = next(self.destination.glob("*.safetensors"))
        weights = load_file(filename)
        weights["model.embedding.weight"].zero_()
        save_file(weights, filename)
        with self.assertRaisesRegex(ValueError, "différent"):
            export_reference(self.source, self.destination)

    def test_existing_export_from_other_source_is_not_overwritten(self):
        export_reference(self.source, self.destination)
        before = (self.destination / PROVENANCE).read_bytes()
        self.weights["model.embedding.weight"].zero_()
        self.save_source()
        with self.assertRaisesRegex(ValueError, "d'autres poids"):
            export_reference(self.source, self.destination)
        self.assertEqual((self.destination / PROVENANCE).read_bytes(), before)

    def test_orphan_scale_fails_without_publishing_partial_folder(self):
        self.weights["orphan.weight_scale"] = torch.tensor([1.])
        self.save_source()
        with self.assertRaisesRegex(ValueError, "inutilisées"):
            export_reference(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob("export.partial-*")), [])
