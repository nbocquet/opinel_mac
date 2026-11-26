# Opinel sur Mac Apple Silicon

*[English version](README.en.md)*

Faire tourner **[Olala v1.0 mini 7A1B Opinel N°8](https://huggingface.co/maurienne-ai/Olala-v1.0-mini-7A1B-Opinel-N8)**,
un modèle de langue français, en local sur un Mac Apple Silicon, sur le GPU,
sans serveur ni connexion après l'installation.

Olala v1.0 mini 7A1B est un modèle hybride (couches Mamba-3 et attention, avec
un mélange d'experts) : 7 milliards de paramètres au total, environ 1 milliard
actifs par token. Opinel N°8 en est la version 8 bits (INT8) publiée par
l'équipe Opinel. Son moteur officiel, un fork de vLLM, demande CUDA : il ne
tourne pas tel quel sur Mac.

**Ce que ce projet apporte :** les poids INT8 de N°8 sont chargés dans MLX
**tels quels, sans les requantifier**, et Télégraphe, le drafter de décodage
spéculatif de l'équipe, est porté en MLX avec un retour arrière exact. Le
moteur réutilise le module Olala d'OVH pour MLX/Metal.

## Où on en est (26 novembre 2026)

**Ce qui marche**

- Un **chat interactif** en français sur le GPU du Mac, avec un historique,
  une réflexion masquée et une réponse affichée au fil de l'eau.
- **Environ 77 tokens/s** en décodage sur un MacBook Air M5 à froid, et 60
  sur un Mac mini M4, pour environ 9 Go de mémoire au plus fort. Le MacBook
  Air n'a pas de ventilateur : sous charge prolongée, il descend à 55-65
  tokens/s (voir les [benchmarks](benchmarks/summary.md)).
- **Le décodage spéculatif** (`--speculatif`) accélère d'environ 10 % un chat
  réel, et jusqu'à 25 % les réponses courtes. Il s'adapte tout seul à la puce.
- **Une fidélité vérifiée.** Les poids N°8 sont transcrits sans perte, et les
  sorties concordent à 98,7 % avec la référence (premier choix identique). Sur
  le test de calcul GSM8K, on obtient 95,0 % en réflexion `low`, contre 94,3 %
  annoncés par l'équipe.

**Les limites du modèle lui-même**

Ces défauts viennent du modèle, pas de son portage sur Mac. Les mêmes erreurs
apparaissent avec la référence, et l'équipe Opinel les a confirmées.

- **Le tutoiement et le vouvoiement** demandés explicitement ne sont pas
  respectés de façon fiable.
- **Les erreurs factuelles et de bon sens** (exemple : aller à pied faire le
  plein de la voiture), y compris avec une réflexion longue.
- **Les corrections de l'utilisateur** ne sont souvent pas prises en compte.
- **L'argot et les régionalismes** sont parfois mal compris (« ça farte ? »).

Il s'agit d'un petit modèle (environ 1 milliard de paramètres actifs par
token) : vérifier ce qu'il affirme.

## Prérequis

- Un Mac Apple Silicon. Testé sur M5 et M4 avec 24 Go. Le chat utilise
  environ 9 Go au plus fort ; un Mac de 16 Go devrait suffire, mais ce n'est
  pas testé.
- Python 3.12 arm64, par exemple `/opt/homebrew/bin/python3.12`.
- Environ 7 Go à télécharger (poids N°8), puis 7,5 Go pour la version MLX, et
  1 Go pour Télégraphe.
- Le Mac branché sur secteur : sur batterie, macOS bride le GPU.

## Installation

Deux environnements Python : `.venv` (PyTorch) télécharge et convertit les
poids, et `.venv-mlx` fait tourner le chat.

```sh
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
/opt/homebrew/bin/python3.12 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install -r requirements-mlx.lock.txt

./opinel download          # poids Opinel N°8, environ 7 Go
./opinel convert-mlx       # transcription au format MLX, dans models/opinel-n8-mlx-n8
./opinel doctor --backend mlx
```

Le modèle est épinglé à une révision précise et ses fichiers sont vérifiés.
Les poids de Télégraphe sont téléchargés au premier lancement de
`--speculatif`. Une fois tout téléchargé, `--offline` évite tout accès au
réseau.

## Utilisation

```sh
./opinel chat --backend mlx --speculatif
```

Taper un message puis Entrée. « Olala prépare sa réponse… » s'affiche pendant
la réflexion, puis la réponse vient au fil de l'eau.

| Dans le chat | Effet |
|---|---|
| `/lire <fichier> [question]` | envoyer tout un fichier texte, suivi d'une question (par défaut : le résumer) |
| `/reset` | effacer l'historique |
| `/quit` ou Ctrl-D | quitter |
| Ctrl-C pendant une réponse | l'interrompre |

Options utiles :

| Option | Effet |
|---|---|
| `--speculatif` | décodage spéculatif Télégraphe, plus rapide, mêmes réponses qu'en mode normal |
| `--reasoning-effort low\|medium\|high` | durée de la réflexion masquée (`low` par défaut) |
| `--temperature 0` | réponses déterministes (par défaut 0,8, top-p 0,9, top-k 40, comme la fiche du modèle) |
| `--context-length N` | taille de la fenêtre de contexte, jusqu'à 32 768 tokens |
| `--offline` | aucun accès réseau |

Pour un long document, agrandir la fenêtre : `--context-length 32768`. Avec la
fenêtre par défaut (11 264 tokens), il reste environ 7 000 tokens pour
l'historique et les documents, soit une vingtaine de pages.

Pour une seule question, sans chat :

```sh
./opinel generate --backend mlx --prompt "Combien font 7 × 8 ?"
```

Sous chaque réponse, une ligne indique la vitesse, le nombre de tokens et la
mémoire utilisée. L'historique reste en mémoire, sans être enregistré sur
disque ; les échanges les plus anciens sont retirés quand la fenêtre est
pleine.

Sans `--backend mlx`, le lanceur utilise une version PyTorch, bien plus lente
(environ 7 tokens/s sur GPU, 14 Go de mémoire). Elle sert de référence pour
les vérifications.

## Mesurer sa machine

```sh
caffeinate -dimsu .venv-mlx/bin/python scripts/bench_suite.py --machine <puce>-<pseudo>
```

La série dure environ 45 minutes. Les résultats des machines déjà mesurées sont
dans [`benchmarks/summary.md`](benchmarks/summary.md), et le protocole dans
[`benchmarks/README.md`](benchmarks/README.md). Les contributions d'autres Mac
sont les bienvenues par pull request.

## Pour aller plus loin

- [`SUIVI.md`](SUIVI.md) : le journal technique, avec la conversion N°8, la
  parité numérique, les évaluations, le décodage spéculatif, les mesures
  détaillées et les pistes restantes.
- [`evaluation/README.md`](evaluation/README.md) : les suites d'évaluation et
  leurs règles de relecture.
- [Issue #1](../../issues/1) : les échanges avec l'équipe Opinel.

Tests unitaires :

```sh
.venv/bin/python -m unittest discover -s tests
.venv-mlx/bin/python -m unittest discover -s tests
```

## Licences

Le code de ce dépôt est sous licence [Apache 2.0](LICENSE) ; voir aussi
[`NOTICE`](NOTICE). Le module Olala d'OVH, sous licence MIT, est dans
`vendor/ovh_olala/` avec sa licence. Les poids d'Opinel N°8 et de Télégraphe
(Apache 2.0) ne sont pas dans le dépôt : téléchargés à part, ils gardent leurs
fichiers `LICENSE` et `NOTICE`.
