"""Opinel inference through the pinned OVH Olala MLX/Metal implementation."""

import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from opinel_mac import FinalChannelSink, MODEL, ROOT, final_response


def validate_provenance(folder):
    path = Path(folder) / "OPINEL_PROVENANCE.json"
    if not path.exists():
        raise ValueError("Poids Opinel MLX absents : exécuter ./opinel convert-mlx depuis le terminal local.")
    manifest = json.loads(path.read_text())
    if (manifest.get("source_repository") != MODEL or not manifest.get("verified_tensors")
            or manifest.get("format") not in ("bf16-reference", "mlx-q8")):
        raise ValueError("Le checkpoint n'est pas un export Opinel vérifié.")
    return manifest


def default_folder():
    quantized = ROOT / "models" / "opinel-n8-mlx-q8"
    return quantized if quantized.exists() else ROOT / "models" / "opinel-n8-mlx-bf16"


def load_mlx_model(folder, device):
    manifest = validate_provenance(folder)
    import mlx.core as mx

    if device == "cpu":
        os.environ["OLALA_MLX_REF"] = "1"
        mx.set_default_device(mx.cpu)
    else:
        if not mx.metal.is_available():
            raise RuntimeError("Metal indisponible : lancer cette commande depuis le terminal local du Mac.")
        mx.set_default_device(mx.gpu)
    from vendor.ovh_olala import register
    register()
    from mlx_lm import load

    print(f"Poids Opinel : {manifest['format']} ({Path(folder).resolve()}).", file=sys.stderr)
    model, tokenizer = load(str(Path(folder).resolve()))
    mx.eval(model.parameters())
    return model, tokenizer


def doctor_mlx():
    for package in ("mlx", "mlx-metal", "mlx-lm"):
        try:
            print(f"{package} : {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package} : absent")
            return 1
    try:
        import mlx.core as mx
        print(f"Metal disponible : {mx.metal.is_available()}")
    except ImportError as error:
        print(f"MLX ne peut pas s'initialiser : {error}", file=sys.stderr)
        return 1
    return 0


def prepare_mlx_turn(tokenizer, messages, reasoning_effort, trim_history=False,
                     context_length=8192, max_new_tokens=1024):
    max_input_tokens = context_length - max_new_tokens
    if max_input_tokens < 1:
        raise ValueError("Le budget de génération ne laisse aucune place au prompt.")
    retained = list(messages)
    while True:
        prompt = tokenizer.apply_chat_template(
            retained, tokenize=False, add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) <= max_input_tokens:
            return tokens, retained
        if not trim_history or len(retained) <= 1:
            raise ValueError(f"Message trop long : limite de {max_input_tokens} tokens pour le prompt, "
                             f"dont le template ({max_new_tokens} réservés à la génération).")
        retained = retained[2:]


def collect_response(responses, stream=False):
    """Consume MLX text deltas without revealing analysis or protocol tokens."""
    began = False

    def write(text):
        nonlocal began
        if not text:
            return
        if not began:
            print("Olala > ", end="", flush=True)
            began = True
        print(text, end="", flush=True)

    sink = FinalChannelSink(write)
    chunks, last = [], None
    try:
        for response in responses:
            chunks.append(response.text)
            if stream:
                sink.feed(response.text)
            last = response
    finally:
        if began:
            print(flush=True)
    if last is None:
        raise ValueError("Le moteur MLX n'a généré aucun token.")
    try:
        answer = final_response("".join(chunks))
    except ValueError as error:
        if last.finish_reason == "length":
            suggestion = ("Réessayer avec --max-new-tokens 1024."
                          if last.generation_tokens < 1024 else
                          "Réessayer avec une demande plus courte, ou /reset pour repartir sans historique.")
            raise ValueError(
                f"Limite de {last.generation_tokens} tokens atteinte avant une réponse finale complète. "
                f"{suggestion}"
            ) from error
        raise
    return answer, last


def generate_turn_mlx(model, tokenizer, messages, args, stream=False, trim_history=False):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    tokens, retained = prepare_mlx_turn(
        tokenizer, messages, args.reasoning_effort, trim_history,
        context_length=getattr(args, "context_length", 8192), max_new_tokens=args.max_new_tokens,
    )
    removed = (len(messages) - len(retained)) // 2
    if removed:
        print(f"Contexte réduit : {removed} échange(s) ancien(s) retiré(s).", file=sys.stderr)
    if stream:
        print("Olala prépare sa réponse…", file=sys.stderr, flush=True)
    started = time.perf_counter()
    responses = stream_generate(
        model, tokenizer, prompt=tokens, max_tokens=args.max_new_tokens,
        sampler=make_sampler(temp=0.0),
    )
    answer, last = collect_response(responses, stream)
    elapsed = time.perf_counter() - started
    print(f"Génération MLX : {last.generation_tokens} tokens en {elapsed:.1f} s "
          f"({last.generation_tokens / elapsed:.2f} tokens/s, préremplissage compris). "
          f"Décodage : {last.generation_tps:.2f} tokens/s.", file=sys.stderr)
    return answer, retained
