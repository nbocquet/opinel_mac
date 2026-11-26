# Contributing

Interested in running Opinel on Apple Silicon? Contributions are welcome:
open an issue, share benchmarks from your Mac, or submit a pull request.
Bug reports, documentation improvements, and results from additional chips
are all useful. You do not need a code change to contribute.

## Open an issue

Use [GitHub Issues](../../issues) to report a bug, suggest an improvement,
ask a question, or discuss an experiment. Check existing issues first.

For a bug report, include:

- Your Mac model, Apple Silicon chip, RAM, and macOS version.
- The repository commit (`git rev-parse HEAD`), Python version, and backend
  used (MLX or PyTorch).
- The exact command and options, steps to reproduce, and expected versus
  actual behavior.
- Relevant error messages or logs. For generation problems, include the
  prompt and relevant conversation history, reasoning effort, sampling
  settings, and whether speculative decoding was enabled.

Remove personal information, credentials, and private documents before
sharing prompts or logs. If you cannot share the original input, a small
reproduction with public or synthetic text helps.

## Benchmark your Mac

Results from other Apple Silicon Macs help us understand performance across
chips, memory configurations, and cooling systems. Please use the existing
suite so results can be compared.

### Prepare the environment

Follow the [English installation instructions](README.en.md#installation)
to install both Python environments, download the N°8 weights, and convert
them to MLX. Run commands below from the repository root. The benchmark
suite uses `.venv-mlx` and `models/opinel-n8-mlx-n8`; Télégraphe weights are
downloaded on first use if missing.

Plug your Mac into power, let it cool before starting, and close heavy
applications and virtual machines. The suite checks power and some busy
background processes. Avoid `--force` for comparable submissions; if you
need it, explain why and describe the conditions.

### Run the suite

Choose a lowercase identifier combining your chip and a nickname, such as
`m4-alice` or `m3max-bob`. Do not use your hostname or serial number. Replace
`m4-alice` below with your identifier.

```sh
# Full suite: about 45 minutes, depending on your Mac.
caffeinate -dimsu .venv-mlx/bin/python scripts/bench_suite.py --machine m4-alice

# Short suite: about 10 minutes; omits the conversation replay.
caffeinate -dimsu .venv-mlx/bin/python scripts/bench_suite.py --machine m4-alice --quick
```

Choose either the full or short suite for a run. `caffeinate` prevents sleep.
If a run is interrupted, rerun the same command: completed result files are
preserved and skipped. Do not reuse the same date and machine output folder
to compare code changes or switch between short and full runs.

The suite measures:

- Target-model verification for different token counts and execution paths.
- Decoding across multiple independent sequences.
- Standard versus speculative decoding with greedy sampling.
- Standard versus speculative decoding with the chat sampling settings
  (temperature 0.8, top-p 0.9, top-k 40).
- Conversation speed with and without speculative decoding (full suite only).

Results are saved as JSON under `benchmarks/results/<date>/<machine>/`.
A machine description is saved under `benchmarks/machines/<machine>.json`.
The files record hardware and software details, commits, and run metadata.
Review them before sharing.

### Share the results

Regenerate the summary after the run:

```sh
.venv-mlx/bin/python scripts/bench_summary.py
```

Open a pull request containing:

- `benchmarks/machines/<machine>.json`.
- Your new `benchmarks/results/<date>/<machine>/` files.
- The regenerated `benchmarks/summary.md`.

Mention whether the run was full or short, the tested commit, power mode,
cooling conditions, and any background load or deviations from the protocol.
Include raw results rather than only a tokens-per-second headline. If you
prefer, open an issue and attach the JSON results instead.

For performance changes, provide a baseline and a changed-code run on the
same Mac with the same settings, and identify both commits. Keep the runs
in separate output folders. Compare standard and speculative decoding
within the same run: thermal throttling can change absolute throughput,
especially on fanless MacBook Air models. Small differences can be noise;
repeat measurements when a claimed gain is only a few percent.

See the [benchmark protocol](benchmarks/README.md) (in French) and
[existing results](benchmarks/summary.md) for more detail.

## Submit a pull request

1. Fork the repository and create a branch for your change.
2. Keep the change focused and follow the surrounding code style. For a
   substantial change, opening an issue first helps discuss the approach.
3. Add or update tests for meaningful behavior changes and bug fixes, and
   update documentation when commands or behavior change.
4. Run the relevant tests. The unit test commands are:

   ```sh
   .venv/bin/python -m unittest discover -s tests
   .venv-mlx/bin/python -m unittest discover -s tests
   ```

5. Open a PR explaining the problem, what changed, how you validated it,
   and any limitations. Link related issues and report tests you could not
   run. For performance work, include reproducible benchmarks as described
   above; for inference changes, include relevant correctness or parity
   checks as well.

Do not include model weights, virtual environments, caches, credentials, or
private data. Keep upstream license and attribution notices intact.

Documentation-only and benchmark-result PRs do not require running inference
tests unless they also change code. Clear, reproducible reports are welcome
even if you do not yet have a fix.
