# Opinel on Apple Silicon Macs

*[Version française](README.md)*

Run **[Olala v1.0 mini 7A1B Opinel N°8](https://huggingface.co/maurienne-ai/Olala-v1.0-mini-7A1B-Opinel-N8)**,
a French-language model, locally on an Apple Silicon Mac, on the GPU, with no
server and no network access once installed.

Olala v1.0 mini 7A1B is a hybrid model (Mamba-3 layers plus attention, with a
mixture of experts): 7 billion parameters in total, about 1 billion active per
token. Opinel N°8 is the 8-bit (INT8) release published by the Opinel team.
Its official runtime is a vLLM fork that needs CUDA, so it does not run on a
Mac as is.

**What this project adds:** the N°8 INT8 weights are loaded into MLX
**exactly as published, without requantizing them**, and the team's
speculative decoding drafter, Télégraphe, is ported to MLX with exact
rollback. The engine reuses OVH's Olala module for MLX/Metal.

## Status (26 November 2026)

**What works**

- An **interactive chat** in French on the Mac's GPU, with history, hidden
  reasoning and a streamed answer.
- **About 77 tokens/s** decoding on a MacBook Air M5 when cool, and about 60
  on a Mac mini M4, with about 9 GB of peak memory. The MacBook Air has no
  fan: under sustained load it drops to 55–65 tokens/s (see
  [benchmarks](benchmarks/summary.md)).
- **Speculative decoding** (`--speculatif`) makes a real chat about 10 %
  faster, and short answers up to 25 % faster. It picks the fastest
  verification path for your chip automatically.
- **Verified fidelity.** The N°8 weights are transcribed losslessly, and the
  outputs match the reference implementation 98.7 % of the time (same top-1
  token). On the GSM8K math benchmark, it scores 95.0 % with `low` reasoning,
  against 94.3 % reported by the Opinel team.

**Limitations of the model itself**

These come from the model, not from the Mac port: the same errors appear with
the reference implementation, and the Opinel team has confirmed them.

- Explicit requests to use **tu or vous** (informal/formal "you") are not
  reliably followed.
- **Factual and common-sense mistakes** (for example, walking to the gas
  station to fill up the car), even with long reasoning.
- **User corrections** are often ignored.
- **Slang and regional expressions** are sometimes misunderstood.

It is a small model (about 1 billion active parameters per token): check what
it says.

## Requirements

- An Apple Silicon Mac. Tested on M5 and M4 with 24 GB. The chat peaks at
  about 9 GB; 16 GB should be enough, but this is untested.
- Python 3.12 arm64, for example `/opt/homebrew/bin/python3.12`.
- About 7 GB to download (N°8 weights), then 7.5 GB for the MLX version and
  1 GB for Télégraphe.
- The Mac plugged in: on battery, macOS throttles the GPU.

## Installation

Two Python environments: `.venv` (PyTorch) downloads and converts the weights,
and `.venv-mlx` runs the chat.

```sh
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
/opt/homebrew/bin/python3.12 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install -r requirements-mlx.lock.txt

./opinel download          # Opinel N°8 weights, about 7 GB
./opinel convert-mlx       # transcription to MLX format, in models/opinel-n8-mlx-n8
./opinel doctor --backend mlx
```

The model is pinned to a specific revision and its files are checked.
Télégraphe's weights are downloaded the first time you use `--speculatif`.
Once everything is downloaded, `--offline` prevents any network access.

## Usage

```sh
./opinel chat --backend mlx --speculatif
```

Type a message and press Enter. "Olala prépare sa réponse…" is shown while the
model reasons, then the answer streams in.

| In the chat | Effect |
|---|---|
| `/lire <file> [question]` | send a whole text file, followed by a question (default: summarize it) |
| `/reset` | clear the history |
| `/quit` or Ctrl-D | quit |
| Ctrl-C during an answer | interrupt it |

Useful options:

| Option | Effect |
|---|---|
| `--speculatif` | Télégraphe speculative decoding: faster, with the same answers as normal mode |
| `--reasoning-effort low\|medium\|high` | length of the hidden reasoning (`low` by default) |
| `--temperature 0` | deterministic answers (default 0.8, top-p 0.9, top-k 40, as in the model card) |
| `--context-length N` | context window size, up to 32,768 tokens |
| `--offline` | no network access |

For a long document, enlarge the window: `--context-length 32768`. With the
default window (11,264 tokens), about 7,000 tokens remain for history and
documents, roughly twenty pages.

For a single question, without the chat:

```sh
./opinel generate --backend mlx --prompt "Combien font 7 × 8 ?"
```

Below each answer, a line shows the speed, the number of tokens and the
memory used. The history stays in memory and is never written to disk; the
oldest exchanges are dropped when the window is full.

Without `--backend mlx`, the launcher uses a PyTorch version that is much
slower (about 7 tokens/s on GPU, 14 GB of memory). It serves as the reference
for the checks.

## Benchmark your Mac

```sh
caffeinate -dimsu .venv-mlx/bin/python scripts/bench_suite.py --machine <chip>-<nickname>
```

The full series takes about 45 minutes. Results for the machines measured so
far are in [`benchmarks/summary.md`](benchmarks/summary.md), and the protocol
in [`benchmarks/README.md`](benchmarks/README.md). Results from other Macs are
welcome as pull requests.

## Going further

These documents are in French:

- [`SUIVI.md`](SUIVI.md): the technical log, covering the N°8 conversion,
  numerical parity, evaluations, speculative decoding, detailed measurements
  and open leads.
- [`evaluation/README.md`](evaluation/README.md): the evaluation suites and
  their review rules.
- [Issue #1](../../issues/1): discussion with the Opinel team.

Unit tests:

```sh
.venv/bin/python -m unittest discover -s tests
.venv-mlx/bin/python -m unittest discover -s tests
```

## Licenses

The Opinel N°8 model and Télégraphe are under the Apache 2.0 license; their
`LICENSE` and `NOTICE` files are kept with the weights. OVH's Olala module,
under the MIT license, is in `vendor/ovh_olala/`.
