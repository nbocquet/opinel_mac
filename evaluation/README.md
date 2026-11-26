# Évaluation de la fiabilité

`reliability_fr.json` contient 30 cas fixes en français, six par catégorie :
faits, raisonnement, corrections, consignes, incertitude. Chaque cas possède
une conversation et une grille de correction. Les sources et réponses attendues
servent à la relecture ; elles ne sont jamais envoyées au modèle.

Ce jeu ciblé complète les replays de `scripts/diagnose_quality.py`, notamment
le comparatif long des fondues. Il ne mesure pas une fiabilité générale. Le
score évalue la demande entière, y compris les contraintes de format.

## Exécuter les mêmes cas

Depuis la racine du projet, la référence Opinel fonctionne sur CPU :

```sh
.venv/bin/python scripts/evaluate_reliability.py run --backend torch --device cpu --preset baseline --output .cache/evaluations/opinel-torch-baseline.json
```

Dans le terminal du Mac ayant accès au GPU, comparer MLX fusionné et MLX de
référence, avec les mêmes poids BF16 :

```sh
.venv-mlx/bin/python scripts/evaluate_reliability.py run --backend mlx --device mps --model-dir models/opinel-n8-mlx-bf16 --output .cache/evaluations/opinel-mlx-baseline.json
.venv-mlx/bin/python scripts/evaluate_reliability.py run --backend mlx --device mps --model-dir models/opinel-n8-mlx-bf16 --reference --output .cache/evaluations/opinel-mlx-reference.json
```

Le preset `baseline` reproduit `low`, génération déterministe et budget de
1024 tokens. Le preset `ovh` teste `medium`, température 0,8, top-p 0,9,
top-k 40 et budget de 4096 tokens. Ces réglages sont ceux présentés dans la
[fiche OVH MLX](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B-MLX).
Avec MLX, l'échantillonnage suit l'ordre de vLLM (température, puis top-k,
puis top-p sur la loi renormalisée), comme les mesures de la fiche Opinel ;
le `make_sampler` de mlx-lm filtrait la loi à T = 1. Les rapports `ovh`
antérieurs ne sont donc pas strictement comparables. Toutes les évaluations
imposent `OLALA_DETERMINISTIC=1`, qui fixe l'ordre de sommation des experts
dans les kernels fusionnés (environ 3 % plus lent selon OVH) ; les rapports
MLX enregistrent `deterministic_kernels` et `tpa_cache`.
Ce second preset modifie plusieurs facteurs ; il ne permet pas d'attribuer
un écart à la seule température ou au seul raisonnement. Une seule graine
ne suffit pas à caractériser un réglage aléatoire : répéter avec `--seed 1`,
`--seed 2`, etc. pour une étude de sa variabilité.

Chaque résultat est sauvegardé immédiatement. Après interruption, relancer
la commande identique avec `--resume`. Une configuration ou une sélection
différente est refusée. Les rapports existants ne sont pas écrasés sans reprise.
`--case ID` permet de cibler un cas ; la graine de ce cas reste la même que
dans l'exécution complète.

## Comparer Olala original à Opinel

```sh
.venv/bin/python scripts/prepare_evaluation_reference.py
.venv/bin/python scripts/evaluate_reliability.py run --backend torch --device cpu --model-kind olala-original --output .cache/evaluations/olala-original-torch-baseline.json
```

La préparation télécharge environ 13,6 Go de poids originaux, fixe la révision
`94836be2ae9bfd5c7c7e7f68f481c9f14f2ee6ea` et vérifie les empreintes de tous
les fichiers. Les données restent dans `models/`, hors Git.

Les deux versions utilisent ici le **même fallback PyTorch Olala**, avec leurs
poids et configurations respectifs, et un chargement strict. Ce n'est pas un
second moteur indépendant : cela contrôle la variation des poids. L'écart
inclut à la fois les retouches Opinel et sa quantification N8 ; ce test ne
sépare pas leurs effets. Le chat habituel conserve ses poids Opinel.

## Relire sans confondre score et exécution

```sh
.venv/bin/python scripts/evaluate_reliability.py review .cache/evaluations/opinel-torch-baseline.json --case i01_tutoie_moi
.venv/bin/python scripts/evaluate_reliability.py review .cache/evaluations/opinel-torch-baseline.json --case i01_tutoie_moi --verdict fail --note "Produit une recette au lieu de passer au tutoiement." --reviewer user
```

Les vérifications exactes concernent uniquement des critères déterministes
(calcul, JSON, liste imposée). Un respect de longueur ne valide pas le contenu.
Les réponses libres restent `unreviewed` jusqu'à une annotation justifiée :
`pass`, `fail` ou `uncertain`. Une erreur de génération est comptée séparément.
Les annotations indiquent le relecteur, la date et son type (`human` ou
`assistant`) ; les révisions de jugement sont conservées. Une annotation
d'assistant ne constitue pas une validation humaine indépendante.

Le score global reste absent tant que des cas sont non exécutés ou non relus.
Une fois tous les cas examinés, il vaut cas réussis / cas prévus ; les erreurs
de génération restent des demandes non satisfaites, pas des preuves d'erreur
factuelle. Le code de sortie de `run` signale les erreurs d'exécution, pas la
véracité des réponses. Les prompts, réponses et grilles restent inspectables.

```sh
.venv/bin/python scripts/evaluate_reliability.py compare .cache/evaluations/opinel-torch-baseline.json .cache/evaluations/olala-original-torch-baseline.json
```

La comparaison exige le même jeu (empreinte SHA256) et les mêmes cas. Elle
affiche les différences de configuration, les comptes par catégorie, la
latence médiane et les verdicts appariés. Un score partiel n'est pas annoncé
comme une amélioration.

## Consignes développeur et suite témoin

Le chat n'envoie par défaut aucun message `developer` : Olala ne reçoit que
le prompt système générique d'OVH, en anglais. `instructions.py` définit un
profil expérimental `opinel-fr-v1` (français, registre tu/vous, aveu
d'incertitude, corrections, format). Il est inséré au rendu du template, ne
disparaît pas quand l'historique est réduit et n'entre jamais dans
l'historique. Le rapport enregistre son nom et son empreinte SHA256 ;
sans consignes, la configuration reste identique aux rapports antérieurs.

`reliability_fr_holdout.json` contient 14 cas témoins, écrits avant toute
mesure du profil : autres formulations du tutoiement et du vouvoiement
(dont « tutoie moi » dans un contexte Python), persistance du registre, faits,
corrections et incertitude. **Ne pas modifier les consignes d'après ces
réponses** ; sinon ils cessent d'être témoins. Un nouveau profil reçoit un
nouveau nom (`opinel-fr-v2`, etc.).

Mesure avant/après, depuis le terminal du Mac (MLX, même preset et graine) :

```sh
for suite in principale temoin; do
  for profil in aucune opinel-fr-v1; do
    .venv-mlx/bin/python scripts/evaluate_reliability.py run --backend mlx --device mps \
      --model-dir models/opinel-n8-mlx-bf16 --suite $suite --instructions $profil \
      --output .cache/evaluations/opinel-mlx-$suite-$profil.json
  done
done
.venv/bin/python scripts/evaluate_reliability.py compare .cache/evaluations/opinel-mlx-principale-aucune.json .cache/evaluations/opinel-mlx-principale-opinel-fr-v1.json
.venv/bin/python scripts/evaluate_reliability.py compare .cache/evaluations/opinel-mlx-temoin-aucune.json .cache/evaluations/opinel-mlx-temoin-opinel-fr-v1.json
```

Une amélioration sur la suite principale seule ne suffit pas ; elle doit se
retrouver sur les cas témoins. Ces consignes peuvent corriger un comportement
(registre, prudence) mais n'ajoutent aucune connaissance au modèle.

## Vérifier les distributions de tokens

Des réponses libres différentes ne suffisent pas à prouver un bug numérique.
`scripts/check_logit_parity.py` applique la méthode de l'équipe Opinel
(issue #1, §3 et §6) :

1. `generate` : le backend testé continue en greedy, sur 256 tokens, 62 prompts
   variés. Ce sont les 44 cas des deux suites de fiabilité, les 16 prompts de
   `parity_prompts.json` et deux documents longs de 3000 et 4500 tokens, qui
   couvrent le préremplissage MLX par blocs de 2048 tokens et le noyau
   d'attention au-delà de 1024 clés. Chaque pas enregistre le top-64 des
   log-probabilités du vrai chemin de décodage.
2. `capture` : PyTorch sur CPU relit ces continuations en teacher forcing,
   en un seul préremplissage. Par défaut, la référence est en FP32, avec les
   experts routés gardés en INT8 et déquantifiés expert par expert (environ
   11 Go au lieu de 27). `--precision bf16` donne la ligne « plancher bf16 ».
3. `compare` : accord top-1, écart moyen |Δ log p| sur le top-5 de la
   référence, et erreurs franches (top-1 de la référence avec p > 0,5 mais
   token généré différent). Les résultats sont ventilés entre prompts courts
   et longs.

Repères de l'équipe sur N°8 (GPU, 64 prompts × 256 tokens) : accord top-1 de
97,7 à 97,9 %, écart top-5 de 0,098 à 0,104 ; plancher bf16 sur CPU : 97,9 à
98,2 % et 0,089 à 0,091. Les commandes forcent `OLALA_DETERMINISTIC=1` et
enregistrent `OLALA_TPA_CACHE`.

```sh
G=.cache/parity
# Backend testé : N°8 direct, puis Q8, et les deux dispositions du cache d'attention.
.venv-mlx/bin/python scripts/check_logit_parity.py generate --backend mlx --device mps \
  --model-dir models/opinel-n8-mlx-n8 --output $G/mlx-n8.npz
OLALA_TPA_CACHE=full .venv-mlx/bin/python scripts/check_logit_parity.py generate --backend mlx --device mps \
  --model-dir models/opinel-n8-mlx-n8 --output $G/mlx-n8-full.npz
OLALA_TPA_CACHE=factored .venv-mlx/bin/python scripts/check_logit_parity.py generate --backend mlx --device mps \
  --model-dir models/opinel-n8-mlx-n8 --output $G/mlx-n8-factored.npz
.venv-mlx/bin/python scripts/check_logit_parity.py generate --backend mlx --device mps \
  --model-dir models/opinel-n8-mlx-q8 --output $G/mlx-q8.npz
# Référence FP32 (CPU, lente : un préremplissage par prompt), puis comparaison.
.venv/bin/python scripts/check_logit_parity.py capture $G/mlx-n8.npz --output $G/mlx-n8-ref-fp32.npz
.venv/bin/python scripts/check_logit_parity.py compare $G/mlx-n8.npz $G/mlx-n8-ref-fp32.npz
```

Chaque `capture` est liée à l'empreinte de sa génération. Pour le plancher
bf16, générer avec `--backend torch --device cpu`, puis capturer cette
génération avec la référence FP32. `--limit 4` permet un essai rapide. Aucun
verdict de fiabilité factuelle n'est déduit de ces métriques.

## Texte de contexte des mesures de vitesse

`texte_contexte.md` est une copie figée de l'ancien README (26 novembre
2026). `scripts/profile_verify.py`, `check_speculative.py --rollback-check` et
`bench_verify.py` en tirent leur contexte de prompt : le figer garde les
mesures comparables d'une série à l'autre, quand le README change.
