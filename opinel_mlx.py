"""Opinel inference through the pinned OVH Olala MLX/Metal implementation."""

import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from opinel_mac import FinalChannelSink, MLX_MAX_NEW_TOKENS, MODEL, ROOT, final_response


def validate_provenance(folder):
    path = Path(folder) / "OPINEL_PROVENANCE.json"
    if not path.exists():
        raise ValueError("Poids Opinel MLX absents : exécuter ./opinel convert-mlx depuis le terminal local.")
    manifest = json.loads(path.read_text())
    if (manifest.get("source_repository") != MODEL or not manifest.get("verified_tensors")
            or manifest.get("format") not in ("bf16-reference", "mlx-q8", "mlx-n8")):
        raise ValueError("Le checkpoint n'est pas un export Opinel vérifié.")
    return manifest


def default_folder():
    """Direct N8 first, then the Q8 requantization, then the BF16 reference."""
    for name in ("opinel-n8-mlx-n8", "opinel-n8-mlx-q8"):
        if (ROOT / "models" / name).exists():
            return ROOT / "models" / name
    return ROOT / "models" / "opinel-n8-mlx-bf16"


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
                     context_length=8192, max_new_tokens=1024, instructions=""):
    from instructions import with_instructions

    max_input_tokens = context_length - max_new_tokens
    if max_input_tokens < 1:
        raise ValueError("Le budget de génération ne laisse aucune place au prompt.")
    retained = list(messages)
    while True:
        prompt = tokenizer.apply_chat_template(
            with_instructions(retained, instructions), tokenize=False, add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) <= max_input_tokens:
            return tokens, retained
        if not trim_history or len(retained) <= 1:
            raise ValueError(f"Message trop long : limite de {max_input_tokens} tokens pour le prompt, "
                             f"dont le template ({max_new_tokens} réservés à la génération).")
        retained = retained[2:]


def make_vllm_sampler(temperature, top_p, top_k):
    """Sample as vLLM does: temperature, then top-k, then top-p on the renormalized law.

    mlx-lm's make_sampler filters at T = 1 and only applies the temperature when
    drawing (issue #1, §11); the model card's scores use vLLM's order.
    """
    import mlx.core as mx

    if temperature <= 0:
        return lambda logprobs: mx.argmax(logprobs, axis=-1)

    def sampler(logprobs):
        x = logprobs / temperature
        if 0 < top_k < x.shape[-1]:
            index = mx.argpartition(-x, kth=top_k - 1, axis=-1)[..., :top_k]
        else:
            index = mx.broadcast_to(mx.arange(x.shape[-1]), x.shape)
        values = mx.take_along_axis(x, index, axis=-1)
        order = mx.argsort(-values, axis=-1)
        values = mx.take_along_axis(values, order, axis=-1)
        index = mx.take_along_axis(index, order, axis=-1)
        if top_p < 1:
            probabilities = mx.softmax(values, axis=-1)
            # Keep a token while the mass before it is below top_p: the first always stays.
            keep = mx.cumsum(probabilities, axis=-1) - probabilities < top_p
            values = mx.where(keep, values, -mx.inf)
        choice = mx.random.categorical(values)
        return mx.take_along_axis(index, choice[..., None], axis=-1).squeeze(-1)

    return sampler


class ThinkingCounter:
    """Show hidden reasoning progress on an interactive stderr, erased before the answer."""

    def __init__(self, stream=None, every=16):
        self.stream = stream or sys.stderr
        self.enabled = self.stream.isatty()
        self.every = every
        self.shown = False

    def update(self, tokens):
        if self.enabled and tokens % self.every == 0:
            self.stream.write(f"\rréflexion… {tokens} tokens")
            self.stream.flush()
            self.shown = True

    def clear(self):
        if self.shown:
            self.stream.write("\r\033[K")
            self.stream.flush()
            self.shown = False


def collect_response(responses, stream=False, max_budget=MLX_MAX_NEW_TOKENS, counter=None):
    """Consume MLX text deltas without revealing analysis or protocol tokens."""
    began = False
    counter = counter or (ThinkingCounter() if stream else None)

    def write(text):
        nonlocal began
        if not text:
            return
        if not began:
            if counter:
                counter.clear()
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
                if not began and counter:
                    counter.update(response.generation_tokens)
            last = response
    finally:
        if counter:
            counter.clear()
        if began:
            print(flush=True)
    if last is None:
        raise ValueError("Le moteur MLX n'a généré aucun token.")
    try:
        answer = final_response("".join(chunks))
    except ValueError as error:
        if last.finish_reason == "length":
            suggestion = (f"Réessayer avec --max-new-tokens {min(2 * last.generation_tokens, max_budget)}."
                          if last.generation_tokens < max_budget else
                          "Réessayer avec une demande plus courte, ou /reset pour repartir sans historique.")
            raise ValueError(
                f"Limite de {last.generation_tokens} tokens atteinte avant une réponse finale complète. "
                f"{suggestion}"
            ) from error
        raise
    return answer, last


class PromptCache:
    """Reuse the previous turn's prefill in a chat.

    The prompt of turn n is a prefix of the prompt of turn n+1 as long as no
    exchange was removed (the template renders past answers after the generation
    prompt, without their reasoning). The Mamba states cannot be rewound, so the
    snapshot is taken at the end of the prefill (every prompt token but the last,
    which the first decoding step consumes), before any reasoning is generated.
    Snapshots share the MLX buffers; the Olala caches copy on write (vendor).
    """

    def __init__(self, model):
        self.model = model
        self.tokens = []
        self.snapshot = None

    def clear(self):
        self.tokens, self.snapshot = [], None

    def start(self, tokens):
        """Return (cache, tokens still to process, reused tokens)."""
        from mlx_lm.models.cache import make_prompt_cache

        reused = len(self.tokens)
        if self.snapshot is not None and 0 < reused < len(tokens) and tokens[:reused] == self.tokens:
            return copy.deepcopy(self.snapshot), tokens[reused:], reused
        return make_prompt_cache(self.model), tokens, 0

    def progress(self, tokens, cache):
        """prompt_progress_callback that snapshots the cache once only the last token is left."""
        def callback(processed, total):
            if processed == total - 1:
                self.tokens, self.snapshot = list(tokens[:-1]), copy.deepcopy(cache)
        return callback


def generate_turn_mlx(model, tokenizer, messages, args, stream=False, trim_history=False, prompt_cache=None):
    from mlx_lm import stream_generate
    from instructions import resolve

    tokens, retained = prepare_mlx_turn(
        tokenizer, messages, args.reasoning_effort, trim_history,
        context_length=getattr(args, "context_length", 8192), max_new_tokens=args.max_new_tokens,
        instructions=resolve(getattr(args, "instructions", "aucune")),
    )
    removed = (len(messages) - len(retained)) // 2
    if removed:
        print(f"Contexte réduit : {removed} échange(s) ancien(s) retiré(s).", file=sys.stderr)
    if stream:
        print("Olala prépare sa réponse…", file=sys.stderr, flush=True)
    started = time.perf_counter()
    extra, reused, remaining = {}, 0, tokens
    if prompt_cache is not None:
        cache, remaining, reused = prompt_cache.start(tokens)
        # Snapshot as soon as the prefill is done: a failed turn still leaves its prompt reusable.
        prompt_cache.snapshot = None
        extra = {"prompt_cache": cache, "prompt_progress_callback": prompt_cache.progress(tokens, cache)}
    responses = stream_generate(
        model, tokenizer, prompt=remaining, max_tokens=args.max_new_tokens,
        sampler=make_vllm_sampler(getattr(args, "temperature", 0.0),
                                  getattr(args, "top_p", 1.0), getattr(args, "top_k", 0)),
        **extra,
    )
    answer, last = collect_response(responses, stream)
    elapsed = time.perf_counter() - started
    reuse = f", dont {reused} repris du tour précédent" if reused else ""
    print(f"Génération MLX : {last.generation_tokens} tokens en {elapsed:.1f} s "
          f"({last.generation_tokens / elapsed:.2f} tokens/s, préremplissage compris). "
          f"Préremplissage : {len(tokens)} tokens{reuse}, "
          f"{len(remaining)} calculés à {last.prompt_tps:.0f} tokens/s. "
          f"Décodage : {last.generation_tps:.2f} tokens/s. "
          f"Pic mémoire : {last.peak_memory:.1f} Go.", file=sys.stderr)
    return answer, retained
