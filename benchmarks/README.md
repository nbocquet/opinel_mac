# Benchmarks Opinel MLX

Mesures comparables d'une machine à l'autre : décodage normal, coût de la
vérification Télégraphe, gain du décodage spéculatif (greedy, échantillonné,
chat). La synthèse est dans [`summary.md`](summary.md).

## Lancer la série

Prérequis : le dépôt cloné, l'environnement `.venv-mlx` installé, les poids
Opinel convertis (`models/opinel-n8-mlx-n8`) et ceux de Télégraphe
(`models/telegraphe`, téléchargés au premier lancement s'ils manquent).

```
.venv-mlx/bin/python scripts/bench_suite.py --machine m5-<pseudo>          # série complète, environ 45 min
.venv-mlx/bin/python scripts/bench_suite.py --machine m5-<pseudo> --quick  # environ 10 min
.venv-mlx/bin/python scripts/bench_summary.py                              # régénère summary.md
```

L'identifiant est la puce puis un pseudo, en minuscules (`m4-bob`, `m3max-alice`).
Ni nom d'hôte ni numéro de série : la fiche machine ne contient que la puce, le
nombre de cœurs, la mémoire, les versions logicielles et les commits.

La série enchaîne, dans `results/<date>/<machine>/` :

| Fichier | Script | Ce qu'il mesure |
|---|---|---|
| `profile_verify.json` | `profile_verify.py` | passe de la cible pour L = 1 à 6 tokens : décodage fusionné, chemin générique, noyaux de décodage (`opinel_verify`) |
| `profile_verify_rows.json` | `profile_verify.py --rows` | décodage fusionné sur 1 à 8 séquences indépendantes |
| `speculative_greedy.json` | `check_speculative.py` | boucle greedy, K = 2, 3, 4 |
| `speculative_sampled.json` | `check_speculative.py --temperature 0.8 --top-p 0.9 --top-k 40` | boucle échantillonnée (réglages du chat) |
| `chat_speed_variee.json` | `replay_conversation.py speed` | chat normal contre `--speculatif` sur `evaluation/conversation_variee.json` (série complète seulement) |

Un résultat existant n'est jamais réécrit : relancer la même commande reprend
là où la série s'est arrêtée. Chaque fichier porte un bloc `meta` (machine,
date, commit, variables `OLALA_*`, alimentation). Le décodage normal mesuré en
tête des runs spéculatifs sert de témoin : s'il est bien en dessous des autres
runs de la même machine, celle-ci était chaude ou chargée.

## Pièges déjà rencontrés

- **Batterie** : macOS bride le GPU ; le décodage normal tombe de 77 à 62-70
  tokens/s sur M5. La série refuse de démarrer hors secteur (`--force` pour
  passer outre).
- **Inventaire logiciel et antivirus** (`ndtrack`, `flxfsscan`, `xprotectd`) et
  **machines virtuelles** : un cœur occupé à 100 % ralentit la boucle
  spéculative, dont une part dépend du CPU (construction du graphe, loi
  d'échantillonnage). La série vérifie qu'aucun ne dépasse 20 % de CPU.
- **Chauffe** : sur un run de 25 minutes, le Mac ralentit progressivement. Les
  scripts alternent donc les variantes (décodage normal, chaque K, chat normal
  et spéculatif) prompt par prompt ou tour par tour : mesurées l'une après
  l'autre, les dernières paraîtraient plus lentes. Ne comparer que des rapports
  mesurés dans le même run.
- **Bruit** : d'un run à l'autre, les rapports varient de quelques pourcents.

## Contribuer

Ouvrir une pull request qui ajoute seulement `benchmarks/machines/<id>.json` et
`benchmarks/results/<date>/<id>/`, puis régénérer `summary.md`.
