#!/bin/sh
# Is the mlx-n8 export at fault, or the model itself? Run from the Mac's terminal (Metal).
#
#   scripts/diagnose_n8.sh            rejeu de la conversation + parité numérique (≈ 30-60 min)
#   scripts/diagnose_n8.sh rejeu      rejeu seulement (≈ 5 min)
#   scripts/diagnose_n8.sh parite     parité seulement
#   scripts/diagnose_n8.sh cache      reprise du cache de prompt : rejeu avec et sans, puis logits
#
# Already finished steps are skipped: rerun the script after an interruption.
# Results go to .cache/diagnostic-n8/; delete the folder to start over.
set -eu
cd "$(dirname "$0")/.."
step=${1:-tout}
out=.cache/diagnostic-n8
mlx=.venv-mlx/bin/python
torch=.venv/bin/python
n8=models/opinel-n8-mlx-n8
bf16=models/opinel-n8-mlx-bf16
mkdir -p "$out"

for folder in "$n8" "$bf16"; do
    if [ ! -f "$folder/OPINEL_PROVENANCE.json" ]; then
        echo "Export absent : $folder (voir README, ./opinel convert-mlx)." >&2
        exit 1
    fi
done

if [ "$step" = tout ] || [ "$step" = rejeu ]; then
    echo "== 1/2 Rejeu de la conversation Saint-Julien (greedy, puis N°8 échantillonné) =="
    $mlx scripts/replay_conversation.py run --model-dir "$bf16" --label bf16-greedy --output "$out/rejeu-bf16-greedy.json"
    $mlx scripts/replay_conversation.py run --model-dir "$n8" --label n8-greedy --output "$out/rejeu-n8-greedy.json"
    $mlx scripts/replay_conversation.py run --model-dir "$n8" --label n8-T0.8 --temperature 0.8 \
        --output "$out/rejeu-n8-sample.json"
    $mlx scripts/replay_conversation.py compare "$out/rejeu-bf16-greedy.json" "$out/rejeu-n8-greedy.json" \
        "$out/rejeu-n8-sample.json" | tee "$out/rejeu-comparaison.txt"
fi

if [ "$step" = tout ] || [ "$step" = parite ]; then
    echo "== 2/2 Parité numérique : N°8 et BF16 MLX contre la référence FP32 PyTorch =="
    for name in n8 bf16; do
        eval folder=\$$name
        if [ ! -f "$out/parite-$name.npz" ]; then
            $mlx scripts/check_logit_parity.py generate --backend mlx --device mps \
                --model-dir "$folder" --output "$out/parite-$name.npz"
        fi
        if [ ! -f "$out/parite-$name-ref-fp32.npz" ]; then
            $torch scripts/check_logit_parity.py capture "$out/parite-$name.npz" \
                --output "$out/parite-$name-ref-fp32.npz"
        fi
        $torch scripts/check_logit_parity.py compare "$out/parite-$name.npz" "$out/parite-$name-ref-fp32.npz" \
            > "$out/parite-$name.json"
    done
    $torch - "$out" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print(f"{'export':8} {'groupe':7} {'top-1':>7} {'écart top-5':>12} {'erreurs franches':>17}")
for name in ("n8", "bf16"):
    result = json.loads((out / f"parite-{name}.json").read_text())
    for group in ("tous", "courts", "longs"):
        m = result[group]
        print(f"{name:8} {group:7} {m['top1_agreement']:7.2%} {m['top5_mean_abs_log_probability_delta']:12.3f} "
              f"{m['clear_errors']:>9} / {m['positions']}")
print("\nRepères Opinel (N°8, GPU) : top-1 97,7-97,9 %, écart top-5 0,098-0,104.")
print("N°8 nettement moins bon que BF16 => la conversion est en cause ; proches => limite du modèle.")
EOF
fi
if [ "$step" = cache ]; then
    echo "== Reprise du cache de prompt : mêmes réponses, préremplissage plus court =="
    $mlx scripts/replay_conversation.py run --model-dir "$n8" --label n8-sans-cache --no-prompt-cache \
        --output "$out/cache-n8-sans.json"
    $mlx scripts/replay_conversation.py run --model-dir "$n8" --label n8-avec-cache \
        --output "$out/cache-n8-avec.json"
    $mlx scripts/replay_conversation.py compare "$out/cache-n8-sans.json" "$out/cache-n8-avec.json" \
        > "$out/cache-comparaison.txt"
    $mlx - "$out" <<'PY' | tee -a "$out/cache-comparaison.txt"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
off, on = (json.loads((out / f"cache-n8-{name}.json").read_text())["turns"] for name in ("sans", "avec"))
same = sum(a["answer"] == b["answer"] for a, b in zip(off, on))
print(f"\nRéponses identiques : {same} / {len(off)} tours")
print(f"{'tour':>5} {'sans cache':>11} {'avec cache':>11}")
for number, (a, b) in enumerate(zip(off, on), 1):
    print(f"{number:>5} {a['seconds']:>9} s {b['seconds']:>9} s")
print(f"{'total':>5} {sum(t['seconds'] for t in off):>9.1f} s {sum(t['seconds'] for t in on):>9.1f} s")
print("Les réponses greedy divergent vite : le préremplissage est découpé autrement. Le contrôle suivant")
print("compare les logits eux-mêmes.")
PY
    $mlx scripts/check_prompt_cache.py "$out/cache-n8-sans.json" --model-dir "$n8" | tee "$out/cache-logits.txt"
fi
echo "Résultats : $out/"
