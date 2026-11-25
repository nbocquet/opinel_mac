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
Le contrôle suivant fixe les prompts et les tokens suivants, sans échantillonnage :

```sh
.venv/bin/python scripts/check_logit_parity.py capture --backend torch --device cpu --output .cache/parity-torch.npz
.venv-mlx/bin/python scripts/check_logit_parity.py capture --backend mlx --device mps --output .cache/parity-mlx.npz
.venv-mlx/bin/python scripts/check_logit_parity.py capture --backend mlx --device mps --reference --output .cache/parity-mlx-reference.npz
.venv/bin/python scripts/check_logit_parity.py compare .cache/parity-mlx-reference.npz .cache/parity-mlx.npz
.venv/bin/python scripts/check_logit_parity.py compare .cache/parity-torch.npz .cache/parity-mlx-reference.npz
```

Les captures vérifient les entrées et la provenance commune, puis mesurent
l'accord top-1, la divergence KL et l'écart de log-probabilité. Elles incluent
le préremplissage et plusieurs pas avec cache. Le Q8 MLX est refusé pour isoler
les kernels de la requantification. Aucun seuil universel ni verdict de
fiabilité factuelle n'est déduit de ces métriques.
