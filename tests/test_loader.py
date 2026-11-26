import unittest
from types import SimpleNamespace

from opinel_mac import dequantize, final_response, hf_parameter_name, precise_geodesic, validate_recipe

try:
    import torch
except ImportError:
    torch = None


class RecipeTests(unittest.TestCase):
    def test_latent_projection_mapping_leaves_shared_experts_unchanged(self):
        self.assertEqual(hf_parameter_name("model.layers.3.mlp.fc1_latent_proj.weight"),
                         "model.layers.3.mlp.down_proj.weight")
        self.assertEqual(hf_parameter_name("model.layers.3.mlp.fc2_latent_proj.weight"),
                         "model.layers.3.mlp.up_proj.weight")
        name = "model.layers.3.mlp.shared_experts.fc_1.weight"
        self.assertEqual(hf_parameter_name(name), name)

    def test_rejects_n4_and_grouped_quantization(self):
        for recipe in ({"quant_algo": "W4A16_AFFINE"}, {"quant_algo": "W8A16", "group_size": 32}):
            with self.assertRaises(ValueError):
                validate_recipe({"quantization_config": {
                    "quant_algo": "MIXED_PRECISION", "quantized_layers": {"lm_head": recipe}}})

    def test_accepts_n8_per_channel(self):
        validate_recipe({"quantization_config": {
            "quant_algo": "MIXED_PRECISION", "quantized_layers": {"lm_head": {"quant_algo": "W8A16"}}}})


class ResponseTests(unittest.TestCase):
    def test_only_complete_final_channel_is_returned(self):
        generated = ("private text<|channel_end|><|channel_start|>final<|content|>"
                     "4<|channel_end|><|im_end|>")
        self.assertEqual(final_response(generated), "4")

    def test_truncated_final_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tronquée"):
            final_response("<|channel_start|>final<|content|>2")

    def test_generation_without_final_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Aucun canal final"):
            final_response("unfinished output")


@unittest.skipIf(torch is None, "PyTorch non installé")
class DequantizationTests(unittest.TestCase):
    def test_geodesic_orthogonal_update_rotates_by_clamped_angle(self):
        module = SimpleNamespace(scale=torch.tensor([1.]), bias=torch.tensor([0.]),
                                 layer_idx=0, clamp=torch.pi / 4)
        x = torch.tensor([[1., 0.]], dtype=torch.bfloat16)
        g = torch.tensor([[0., 1.]], dtype=torch.bfloat16)
        expected = torch.tensor([[2 ** -0.5, 2 ** -0.5]], dtype=torch.bfloat16)
        torch.testing.assert_close(precise_geodesic(module, x, g), expected)

    def test_geodesic_zero_update_preserves_residual(self):
        module = SimpleNamespace(scale=torch.tensor([1.]), bias=torch.tensor([0.]),
                                 layer_idx=3, clamp=torch.pi / 4)
        x = torch.tensor([[0.25, -0.5]], dtype=torch.bfloat16)
        torch.testing.assert_close(precise_geodesic(module, x, torch.zeros_like(x)), x)

    def test_linear_scales_each_output_row(self):
        weights = torch.tensor([[2, -4], [3, 1]], dtype=torch.int8)
        scales = torch.tensor([0.5, 2.0])
        torch.testing.assert_close(dequantize(weights, scales, torch.float32),
                                   torch.tensor([[1., -2.], [6., 2.]]))

    def test_expert_scales_have_independent_expert_and_row_axes(self):
        weights = torch.ones(2, 3, 4, dtype=torch.int8)
        scales = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        expected = scales[:, :, None].expand(2, 3, 4)
        torch.testing.assert_close(dequantize(weights, scales, torch.float32), expected)

    def test_invalid_scale_shape_fails(self):
        with self.assertRaises(ValueError):
            dequantize(torch.ones(2, 3, dtype=torch.int8), torch.ones(3), torch.float32)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(torch is None, "PyTorch non installé")
class Int8ExpertTests(unittest.TestCase):
    def test_int8_experts_match_dequantized_fp32_mixture_and_leave_no_parameters(self):
        from opinel_mac import install_int8_experts

        class TorchMoEMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts, self.input_size = 3, 4
                self.activation = lambda x: torch.relu(x).square()
                self.experts = torch.nn.Module()
                self.experts.weight = torch.nn.Parameter(torch.zeros(3, 5, 4))
                self.output_experts = torch.nn.Module()
                self.output_experts.weight = torch.nn.Parameter(torch.zeros(3, 4, 5))

        model = torch.nn.Module()
        model.mlp = torch.nn.Module()
        model.mlp.experts = TorchMoEMLP()
        torch.manual_seed(0)
        q_in, q_out = torch.randint(-127, 128, (3, 5, 4), dtype=torch.int8), torch.randint(-127, 128, (3, 4, 5), dtype=torch.int8)
        s_in, s_out = torch.rand(3, 5) / 50, torch.rand(3, 4) / 50
        install_int8_experts(model, {"mlp.experts.experts.weight": (q_in, s_in),
                                     "mlp.experts.output_experts.weight": (q_out, s_out)})
        self.assertEqual(list(model.parameters()), [])
        self.assertEqual(model.state_dict(), {})
        x = torch.randn(2, 4)
        idx, p = torch.tensor([[0, 2], [2, 1]]), torch.tensor([[0.7, 0.3], [0.6, 0.4]])
        w_in, w_out = q_in.float() * s_in[..., None], q_out.float() * s_out[..., None]
        expected = torch.stack([
            sum(p[t, k] * (w_out[idx[t, k]] @ torch.relu(w_in[idx[t, k]] @ x[t]).square()) for k in range(2))
            for t in range(2)])
        torch.testing.assert_close(model.mlp.experts(x, p, idx), expected)
        with self.assertRaisesRegex(ValueError, "TorchMoEMLP"):
            install_int8_experts(torch.nn.Module(), {"other.weight": (q_in, s_in)})
