"""Experimental Opinel N8 reference runner; no CUDA or vLLM dependency."""

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parent
MODEL = "maurienne-ai/Olala-v1.0-mini-7A1B-Opinel-N8"
REVISION = "184f9b2d4e4bd02cec585ad20b441b44d3e07b55"
FINAL_MARKER = "<|channel_start|>final<|content|>"


def hf_parameter_name(name):
    """Map the fork's latent projection names to the supplied HF model."""
    return name.replace(".mlp.fc1_latent_proj.", ".mlp.down_proj.").replace(
        ".mlp.fc2_latent_proj.", ".mlp.up_proj."
    )


def validate_recipe(config):
    quant = config.get("quantization_config", {})
    layers = quant.get("quantized_layers", {})
    if quant.get("quant_algo") != "MIXED_PRECISION" or not layers:
        raise ValueError("Le checkpoint doit être Opinel N8 MIXED_PRECISION.")
    for name, recipe in layers.items():
        if recipe.get("quant_algo") != "W8A16" or recipe.get("group_size"):
            raise ValueError(f"Recette non prise en charge pour {name}: {recipe}")


def dequantize(weight, scale, dtype):
    """N8 per-output-channel quantization, including [expert, output, input]."""
    import torch

    if weight.dtype != torch.int8 or scale.shape != weight.shape[:-1]:
        raise ValueError("Poids INT8 ou dimensions des échelles N8 invalides.")
    return (weight.float() * scale.float().unsqueeze(-1)).to(dtype)


def configure_environment():
    # The supplied model has explicit portable implementations for these paths.
    os.environ["OLALA_TORCH_MAMBA3"] = "1"
    os.environ["OLALA_TORCH_MOE"] = "1"
    os.environ["OLALA_ATTN_IMPL"] = "eager"
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".cache" / "modules"))


def final_response(text):
    """Return only a complete final channel; refuse truncated generations."""
    marker = FINAL_MARKER
    if marker not in text:
        raise ValueError("Aucun canal final produit. Augmenter la limite de tokens.")
    content = text.split(marker, 1)[1]
    if "<|channel_end|>" not in content:
        raise ValueError("Réponse finale tronquée. Augmenter la limite de tokens.")
    return content.split("<|channel_end|>", 1)[0]


class FinalChannelSink:
    """Filter decoded chunks, including protocol markers split across chunks."""

    def __init__(self, write):
        self.write = write
        self.pending = ""
        self.started = False
        self.closed = False

    def feed(self, text):
        if self.closed:
            return
        self.pending += text
        if not self.started:
            position = self.pending.find(FINAL_MARKER)
            if position < 0:
                self.pending = self.pending[-(len(FINAL_MARKER) - 1):]
                return
            self.pending = self.pending[position + len(FINAL_MARKER):]
            self.started = True
        # Stop at any new protocol token, never streaming a later channel.
        position = self.pending.find("<|")
        if position >= 0:
            self.write(self.pending[:position])
            self.pending = ""
            self.closed = True
        else:
            # A final '<' could be the beginning of a split control token.
            hold = 1 if self.pending.endswith("<") else 0
            ready = self.pending[:-1] if hold else self.pending
            if ready:
                self.write(ready)
            self.pending = "<" if hold else ""


def make_final_streamer(tokenizer, prefix=""):
    from transformers import TextStreamer

    began = False

    def write(text):
        nonlocal began
        if not text:
            return
        if not began:
            sys.stdout.write(prefix)
            began = True
        sys.stdout.write(text)
        sys.stdout.flush()

    sink = FinalChannelSink(write)

    class FinalStreamer(TextStreamer):
        def on_finalized_text(self, text, stream_end=False):
            sink.feed(text)

    return FinalStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False,
                         clean_up_tokenization_spaces=False)


def prepare_turn(tokenizer, messages, device, reasoning_effort, trim_history=False, max_input_tokens=1024,
                 instructions=""):
    from instructions import with_instructions

    retained = list(messages)
    while True:
        prompt = tokenizer.apply_chat_template(
            with_instructions(retained, instructions), tokenize=False, add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if inputs.input_ids.shape[-1] <= max_input_tokens:
            return inputs.to(device), retained
        if not trim_history or len(retained) <= 1:
            raise ValueError(f"Message trop long : limite de {max_input_tokens} tokens, template compris.")
        # The conversation consists of complete user/assistant pairs + this user.
        retained = retained[2:]


def instructions_text(args):
    from instructions import resolve

    return resolve(getattr(args, "instructions", "aucune"))


def generate_turn(model, tokenizer, messages, args, stream=False, trim_history=False):
    if getattr(args, "backend", "torch") == "mlx":
        from opinel_mlx import generate_turn_mlx
        return generate_turn_mlx(model, tokenizer, messages, args, stream, trim_history)
    import torch

    inputs, retained = prepare_turn(
        tokenizer, messages, args.device, args.reasoning_effort, trim_history,
        max_input_tokens=min(1024, getattr(args, "context_length", 2048) - args.max_new_tokens),
        instructions=instructions_text(args),
    )
    removed = (len(messages) - len(retained)) // 2
    if removed:
        print(f"Contexte réduit : {removed} échange(s) ancien(s) retiré(s).", file=sys.stderr)
    streamer = make_final_streamer(tokenizer, prefix="Olala > ") if stream else None
    if stream:
        print("Olala prépare sa réponse…", file=sys.stderr, flush=True)
    started = time.perf_counter()
    temperature = getattr(args, "temperature", 0.0)
    sampling = {"do_sample": temperature > 0}
    if temperature > 0:
        sampling.update(temperature=temperature, top_p=args.top_p, top_k=args.top_k)
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                **sampling, use_cache=True, logits_to_keep=1,
                pad_token_id=tokenizer.pad_token_id, streamer=streamer,
            )
        if args.device == "mps":
            torch.mps.synchronize()
    finally:
        if stream:
            print(flush=True)
    elapsed = time.perf_counter() - started
    count = output.shape[-1] - inputs.input_ids.shape[-1]
    print(f"Génération : {count} tokens en {elapsed:.1f} s ({count / elapsed:.2f} tokens/s)", file=sys.stderr)
    result = tokenizer.decode(output[0, inputs.input_ids.shape[-1]:], skip_special_tokens=False)
    try:
        answer = final_response(result)
    except ValueError as error:
        if count >= args.max_new_tokens:
            suggestion = ("Réessayer avec --max-new-tokens 1024."
                          if args.max_new_tokens < 1024 else
                          "Réessayer avec une demande plus courte.")
            raise ValueError(
                f"Limite de {args.max_new_tokens} tokens atteinte avant une réponse finale complète. "
                f"{suggestion}"
            ) from error
        raise
    return answer, retained


def run_chat(model, tokenizer, args):
    messages = []
    context = getattr(args, "context_length", None)
    suffix = f" Contexte : {context} tokens." if context is not None else ""
    print(f"Opinel est prêt.{suffix} /reset : effacer l'historique · /quit : quitter")
    while True:
        try:
            prompt = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt == "/quit":
            return 0
        if prompt == "/reset":
            messages.clear()
            print("Historique effacé. Le modèle reste chargé.")
            continue
        if prompt.startswith("/"):
            print("Commandes disponibles : /reset, /quit")
            continue
        candidate = messages + [{"role": "user", "content": prompt}]
        try:
            answer, retained = generate_turn(
                model, tokenizer, candidate, args, stream=True, trim_history=True,
            )
        except KeyboardInterrupt:
            print("Réponse interrompue ; échange non conservé.", file=sys.stderr)
            continue
        except (ValueError, RuntimeError) as error:
            print(f"Échange non conservé : {error}", file=sys.stderr)
            continue
        # Only complete final replies enter subsequent prompts; never analysis.
        messages = retained + [{"role": "assistant", "content": answer}]


def precise_geodesic(module, residual, update):
    """Use FP32 reductions, as the fork's CUDA geodesic kernel does."""
    import torch

    dtype = residual.dtype
    x, g = residual.float(), update.float()
    radius_squared = x.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
    projection = (x * g).sum(dim=-1, keepdim=True) / radius_squared
    tangent = g - projection * x
    tangent_norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    radius = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    theta = (tangent_norm / radius).clamp(max=module.clamp)
    theta = ((theta * module.scale.float() + module.bias.float()) /
             (module.layer_idx + 1)).clamp(max=module.clamp)
    return (x * torch.cos(theta) + tangent * (radius * torch.sin(theta) / tangent_norm)).to(dtype)


def snapshot(offline=False):
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        MODEL, revision=REVISION, local_files_only=offline,
        allow_patterns=["*.json", "*.py", "*.jinja", "*.safetensors", "LICENSE", "NOTICE"],
    ))


def load_model(folder, device, *, context_length=2048, unquantized_reference=False, architecture_folder=None):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    raw = json.loads((folder / "config.json").read_text())
    if unquantized_reference:
        if raw.get("quantization_config") or architecture_folder is None:
            raise ValueError("Référence BF16 attendue avec une architecture de comparaison explicite.")
        template = AutoConfig.from_pretrained(architecture_folder, trust_remote_code=True, local_files_only=True)
        config = type(template).from_dict(raw)
        config.auto_map = template.auto_map
        config._name_or_path = str(Path(architecture_folder).resolve())
    else:
        validate_recipe(raw)
        config = AutoConfig.from_pretrained(folder, trust_remote_code=True, local_files_only=True)
    # Bypass NVIDIA ModelOpt; load and dequantize every weight ourselves.
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    # The upstream RoPE cache allocates twice this limit in each attention layer.
    # Our CLI already caps prompt + continuation at 2048 tokens.
    config.max_position_embeddings = min(config.max_position_embeddings, context_length)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, attn_implementation="eager"
        )
    state = {}
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as source:
        keys = set(source.keys())
        for number, name in enumerate(sorted(keys), 1):
            if name.endswith(".weight_scale"):
                continue
            tensor = source.get_tensor(name)
            if tensor.dtype == torch.int8:
                if unquantized_reference:
                    raise ValueError("La référence annoncée BF16 contient des poids INT8.")
                scale_name = name.removesuffix(".weight") + ".weight_scale"
                if scale_name not in keys:
                    raise ValueError(f"Échelle manquante : {scale_name}")
                tensor = dequantize(tensor, source.get_tensor(scale_name), torch.bfloat16)
            target = hf_parameter_name(name)
            if target in state:
                raise ValueError(f"Paramètre dupliqué : {target}")
            state[target] = tensor.to(device)
            if number % 100 == 0:
                print(f"Chargement : {number}/{len(keys)} tenseurs", file=sys.stderr)
    # Refuse missing/unexpected keys rather than running with random parameters.
    model.load_state_dict(state, strict=True, assign=True)
    # Mirror the supplied model's from_pretrained post-load hook. In particular,
    # its geodesic scale/bias scalars must have the same shape and promotion rules.
    model.widen_geodesic_scalars()
    for module in model.modules():
        if type(module).__name__ == "OlalaGeodesicNorm":
            module.forward = MethodType(precise_geodesic, module)
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.is_meta:
                if name == "tokens_per_expert":
                    setattr(module, name, torch.zeros(buffer.shape, device=device, dtype=buffer.dtype))
                elif name == "inv_freq" and type(module).__name__ == "OlalaRotaryEmbedding":
                    dim = config.head_dim
                    frequencies = 1.0 / (config.rope_theta ** (
                        torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
                    ))
                    module.inv_freq = frequencies
                else:
                    raise RuntimeError(f"Buffer non initialisé : {type(module).__name__}.{name}")
    model.eval()
    return model


def doctor():
    print(f"Système : {platform.system()} {platform.machine()}")
    print(f"Python : {platform.python_version()}")
    if sys.platform == "darwin":
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            hardware = json.loads(result.stdout)["SPHardwareDataType"][0]
            print(f"Puce : {hardware.get('chip_type', '?')}")
            print(f"Mémoire : {hardware.get('physical_memory', '?')}")
    try:
        import torch
    except ImportError:
        print("PyTorch absent. Installer : python -m pip install -r requirements.txt")
        return 1
    print(f"PyTorch : {torch.__version__}")
    print(f"MPS compilé : {torch.backends.mps.is_built()}")
    print(f"MPS disponible : {torch.backends.mps.is_available()}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "download", "generate", "chat", "convert-mlx"])
    parser.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    parser.add_argument("--context-length", type=int,
                        help="Fenêtre totale, entrée et génération : 8192 pour MLX, 2048 pour PyTorch")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--prompt", default="Présente-toi en une phrase en français.")
    parser.add_argument("--max-new-tokens", type=int,
                        help="Budget incluant le raisonnement : 1024 en chat, 256 en génération simple")
    parser.add_argument("--reasoning-effort", choices=["disabled", "low", "medium", "high"], default="low")
    parser.add_argument("--instructions", choices=["aucune", "opinel-fr-v1"], default="aucune",
                        help="Consignes développeur ajoutées au template (expérimental)")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--model-dir", type=Path, help="Checkpoint local déjà téléchargé")
    parser.add_argument("--recipe", choices=["bf16", "q8"], default="q8", help="Conversion MLX")
    parser.add_argument("--output-dir", type=Path, help="Dossier de sortie de la conversion")
    parser.add_argument("--export-only", action="store_true", help="Exporter et vérifier le BF16 sans initialiser MLX")
    args = parser.parse_args(argv)
    if args.command == "convert-mlx" and args.backend != "torch":
        parser.error("convert-mlx utilise l'environnement PyTorch pour lire N8 ; retirer --backend mlx.")
    if args.max_new_tokens is None:
        args.max_new_tokens = 1024 if args.command == "chat" else 256
    if args.max_new_tokens < 1 or args.max_new_tokens > 1024:
        parser.error("--max-new-tokens doit être compris entre 1 et 1024.")
    if args.context_length is None:
        args.context_length = 8192 if args.backend == "mlx" else 2048
    maximum_context = 32768 if args.backend == "mlx" else 2048
    if not 1 <= args.context_length <= maximum_context:
        parser.error(f"--context-length doit être compris entre 1 et {maximum_context} pour {args.backend}.")
    if args.context_length <= args.max_new_tokens:
        parser.error("--context-length doit dépasser --max-new-tokens pour laisser de la place au prompt.")
    return args


def main():
    args = parse_args()
    configure_environment()
    if args.command == "convert-mlx":
        from opinel_conversion import run_conversion
        try:
            return run_conversion(args)
        except (ValueError, FileNotFoundError) as error:
            print(f"Conversion interrompue : {error}", file=sys.stderr)
            return 2
    if args.backend == "mlx":
        from opinel_mlx import default_folder, doctor_mlx, load_mlx_model
        if args.command == "doctor":
            return doctor_mlx()
        if args.command == "download":
            print("Pour MLX, exécuter ./opinel convert-mlx avec les poids Opinel locaux.", file=sys.stderr)
            return 2
        folder = args.model_dir or default_folder()
        try:
            started = time.perf_counter()
            model, tokenizer = load_mlx_model(folder, args.device)
            print(f"Opinel chargé avec le moteur OVH MLX en {time.perf_counter() - started:.1f} s.", file=sys.stderr)
            if args.command == "chat":
                return run_chat(model, tokenizer, args)
            answer, _ = generate_turn(model, tokenizer, [{"role": "user", "content": args.prompt}], args)
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"Backend MLX indisponible : {error}", file=sys.stderr)
            return 2
        print(answer)
        return 0
    if args.command == "doctor":
        return doctor()
    import torch
    from transformers import AutoTokenizer

    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    if args.command in ("generate", "chat") and args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS indisponible. Utiliser --device cpu pour le chemin de référence.", file=sys.stderr)
        return 2
    local = ROOT / "models" / "opinel-n8"
    folder = args.model_dir.resolve() if args.model_dir else (local if local.exists() else snapshot(args.offline))
    if args.command == "download":
        print(folder)
        return 0
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=True)
    print(f"Prototype expérimental sur {args.device} : poids déquantifiés (~14 Go), contexte court.", file=sys.stderr)
    started = time.perf_counter()
    model = load_model(folder, args.device)
    print(f"Chargement terminé en {time.perf_counter() - started:.1f} s", file=sys.stderr)
    if args.command == "chat":
        return run_chat(model, tokenizer, args)
    try:
        answer, _ = generate_turn(
            model, tokenizer, [{"role": "user", "content": args.prompt}], args,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
