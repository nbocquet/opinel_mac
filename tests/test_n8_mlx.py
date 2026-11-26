import json
from pathlib import Path
import tempfile
import unittest

from opinel_n8_mlx import GROUP_SIZE, mlx_config, tensor_plan

try:
    import mlx.core as mx
except ImportError:
    mx = None

RECIPE = {"quant_algo": "MIXED_PRECISION", "quantized_layers": {"lm_head": {"quant_algo": "W8A16"}}}


class PlanTests(unittest.TestCase):
    def test_linear_layers_are_quantized_and_small_projections_dequantized(self):
        self.assertEqual(tensor_plan("model.layers.0.mixer.in_proj.weight", True), "quantize")
        self.assertEqual(tensor_plan("model.layers.3.mlp.experts.experts.weight", True), "quantize")
        self.assertEqual(tensor_plan("lm_head.weight", True), "quantize")
        for module in ("W_A_k", "W_A_v", "shift_proj_k", "shift_proj_v", "lambda_proj"):
            self.assertEqual(tensor_plan(f"model.layers.4.mixer.{module}.weight", True), "float32")

    def test_floating_tensors_are_kept_and_inconsistent_ones_refused(self):
        self.assertEqual(tensor_plan("model.embedding.weight", False), "keep")
        self.assertEqual(tensor_plan("model.layers.0.mlp.moe_gate.weight", False), "keep")
        for name in ("model.embedding.weight", "model.layers.0.mlp.moe_gate.weight",
                     "model.layers.0.mixer.softmax_scaler", "lm_head.weight_scale"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                tensor_plan(name, True)

    def test_config_declares_affine_8_bits_and_drops_modelopt(self):
        config = mlx_config({"model_type": "olala", "auto_map": {}, "quantization_config": RECIPE})
        self.assertEqual(config, {"model_type": "olala",
                                  "quantization": {"group_size": 128, "bits": 8, "mode": "affine"}})
        with self.assertRaises(ValueError):
            mlx_config({"model_type": "olala"})


@unittest.skipIf(mx is None, "MLX absent de cet environnement")
class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)
        mx.random.seed(0)

    @staticmethod
    def codes(*shape):
        return mx.random.randint(-127, 128, shape).astype(mx.int8)

    def source(self):
        scale = lambda *shape: mx.random.uniform(1e-4, 2e-2, shape)  # noqa: E731
        return {
            "model.embedding.weight": mx.random.normal((16, 256)).astype(mx.bfloat16),
            "model.layers.0.mixer.in_proj.weight": self.codes(8, 256),
            "model.layers.0.mixer.in_proj.weight_scale": scale(8),
            "model.layers.0.mixer.W_A_k.weight": self.codes(3, 256),
            "model.layers.0.mixer.W_A_k.weight_scale": scale(3),
            "model.layers.0.mlp.fc1_latent_proj.weight": self.codes(4, 256),
            "model.layers.0.mlp.fc1_latent_proj.weight_scale": scale(4),
            "model.layers.0.mlp.experts.experts.weight": self.codes(2, 4, 128),
            "model.layers.0.mlp.experts.experts.weight_scale": scale(2, 4),
            "model.layers.0.mlp.moe_gate.weight": mx.random.normal((2, 256)),
        }

    def transcribe(self, tensors):
        from opinel_n8_mlx import transcribe
        output = dict(transcribe(tensors))
        return output, output.pop(None)

    def test_affine_codes_reproduce_n8_with_only_the_scale_rounded(self):
        from opinel_n8_mlx import n8_affine
        q, s = self.codes(5, 256), mx.random.uniform(1e-4, 2e-2, (5,))
        weight, scales, biases = n8_affine(q, s)
        self.assertEqual((weight.dtype, weight.shape, scales.shape), (mx.uint32, (5, 64), (5, 2)))
        # Little-endian packing: the first code sits in the low byte.
        self.assertEqual(weight[0, 0].item() & 0xFF, q[0, 0].item() + 128)
        restored = mx.dequantize(weight, scales.astype(mx.float32), biases.astype(mx.float32),
                                 group_size=GROUP_SIZE, bits=8)
        reference = q.astype(mx.float32) * s[:, None]
        self.assertLessEqual(mx.max(mx.abs(restored - reference) / mx.maximum(mx.abs(reference), 1e-30)).item(),
                             2.0 ** -8)

    def test_symmetric_format_and_group_divisibility_are_enforced(self):
        from opinel_n8_mlx import n8_affine
        with self.assertRaisesRegex(ValueError, "-128"):
            n8_affine(mx.full((1, 128), -128, mx.int8), mx.ones((1,)))
        with self.assertRaisesRegex(ValueError, "divisible"):
            n8_affine(self.codes(1, 192), mx.ones((1,)))

    def test_transcription_names_dtypes_and_exact_floating_tensors(self):
        tensors = self.source()
        output, report = self.transcribe(tensors)
        self.assertEqual((report["quantized"], report["float32"], report["kept"]), (3, 1, 2))
        self.assertLessEqual(report["max_scale_rounding"], 2.0 ** -8)
        self.assertNotIn("model.layers.0.mlp.fc1_latent_proj.weight", output)
        self.assertEqual(output["model.layers.0.mlp.down_proj.weight"].dtype, mx.uint32)
        self.assertIn("model.layers.0.mlp.down_proj.scales", output)
        self.assertEqual(output["model.layers.0.mlp.experts.experts.scales"].shape, (2, 4, 1))
        self.assertTrue(mx.array_equal(output["model.embedding.weight"], tensors["model.embedding.weight"]))
        self.assertEqual(output["model.embedding.weight"].dtype, mx.bfloat16)
        self.assertNotIn("model.embedding.scales", output)
        self.assertNotIn("model.layers.0.mlp.moe_gate.scales", output)
        small = output["model.layers.0.mixer.W_A_k.weight"]
        self.assertEqual(small.dtype, mx.float32)
        self.assertTrue(mx.array_equal(small, tensors["model.layers.0.mixer.W_A_k.weight"].astype(mx.float32)
                                       * tensors["model.layers.0.mixer.W_A_k.weight_scale"][:, None]))

    def test_missing_or_unused_scales_are_refused(self):
        tensors = self.source()
        del tensors["model.layers.0.mixer.in_proj.weight_scale"]
        with self.assertRaisesRegex(ValueError, "manquante"):
            self.transcribe(tensors)
        tensors = self.source()
        tensors["model.layers.0.extra.weight_scale"] = mx.ones((1,))
        with self.assertRaisesRegex(ValueError, "inutilisées"):
            self.transcribe(tensors)

    def test_conversion_writes_verified_shards_and_provenance(self):
        from opinel_n8_mlx import convert_n8
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "out"
            source.mkdir()
            mx.save_safetensors(str(source / "model.safetensors"), self.source())
            (source / "config.json").write_text(json.dumps({"model_type": "olala", "quantization_config": RECIPE}))
            (source / "NOTICE").write_text("notice")
            provenance = convert_n8(source, output, shard_bytes=4096, load_check=False)
            self.assertEqual(provenance["format"], "mlx-n8")
            self.assertEqual(provenance["verified_tensors"], 2 + 1 + 3 * 3)
            self.assertIsNone(provenance["source_revision"])
            self.assertGreater(len(list(output.glob("model-*.safetensors"))), 1)
            self.assertEqual(json.loads((output / "config.json").read_text())["quantization"]["group_size"], 128)
            self.assertEqual(json.loads((output / "OPINEL_PROVENANCE.json").read_text()), provenance)
            self.assertEqual((output / "NOTICE").read_text(), "notice")
            self.assertEqual([path.name for path in Path(directory).iterdir() if "partial" in path.name], [])
            with self.assertRaises(FileExistsError):
                convert_n8(source, output, load_check=False)
