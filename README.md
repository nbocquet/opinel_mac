# Opinel sur Mac Apple Silicon

Prototype expérimental pour **Olala v1.0 mini 7A1B Opinel N°8**, préparé sur un
MacBook Air **M5, 24 Go**, macOS 27.0.1. **La génération fonctionne sur CPU et
sur GPU MPS.** Le mode conversation est vérifié sur CPU.

## Pourquoi le fork vLLM ne suffit pas

Le créateur demande [`nbald/vllm`, branche `opinel`](https://github.com/nbald/vllm/tree/opinel).
La [fiche du modèle](https://huggingface.co/maurienne-ai/Olala-v1.0-mini-7A1B-Opinel-N8)
exige CUDA 13 pour ses kernels NVIDIA. Ces commandes ne constituent pas une
installation pour le GPU Apple. [vLLM Metal](https://github.com/vllm-project/vllm-metal)
est une autre implémentation : Olala n'apparaît pas dans sa
[matrice de compatibilité consultée le 25 novembre 2026](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md).

## Approche du prototype

`opinel_mac.py` utilise les implémentations PyTorch de référence déjà présentes
dans `modeling_olala.py` (Mamba-3, experts MoE et attention eager). Il charge les
poids N8 selon le [format documenté par le fork](https://github.com/nbald/vllm/blob/opinel/docs/features/quantization/olala.md),
avec `weight * weight_scale` par canal, et traduit les noms des projections latentes.
Le chargement strict refuse tout paramètre manquant ou inattendu. Les calculs
géodésiques utilisent des réductions FP32, comme le kernel du fork.

Les poids INT8 sont déquantifiés en BF16 : prévoir **environ 14 Go pour les poids**,
plus les allocations temporaires, les états et macOS. Les 7,1 Go du téléchargement
ne représentent donc pas la consommation de ce prototype. Commencer avec une
seule requête et un contexte court. `--device cpu` permet une exécution portable.
Un essai MPS réalisé depuis le terminal du Mac a donné 158 tokens en 22,8 secondes
(6,92 tokens/s, préremplissage et raisonnement compris), après 9,4 secondes de chargement.

## Installation et exécution

Python 3.12 et les dépendances sont installés dans `.venv`. Les wheels ont été
téléchargés directement et leurs empreintes SHA256 vérifiées avant installation.
`requirements.lock.txt` conserve les versions installées. Pour recréer cet environnement :

```sh
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python opinel_mac.py doctor
.venv/bin/python opinel_mac.py download
.venv/bin/python opinel_mac.py generate --device mps --prompt "Bonjour !"
# Référence CPU si MPS échoue :
.venv/bin/python opinel_mac.py generate --device cpu --max-new-tokens 256
```

Le téléchargement représente environ 7,1 Go. Modèle et code distant sont épinglés
à la révision `184f9b2d4e4bd02cec585ad20b441b44d3e07b55`.
L'exécution charge le code Python de cette révision via `trust_remote_code=True`.
Les caches restent dans `.cache/`. Après téléchargement, ajouter `--offline`.
La génération simple est déterministe (greedy). Le chat échantillonne avec les
réglages de la fiche du modèle, `temperature=0.8, top_p=0.9, top_k=40` ; tous
ses scores sont mesurés ainsi. `--temperature 0` rend le chat greedy, et
`--temperature`, `--top-p` et `--top-k` règlent chaque paramètre. Le
raisonnement est `low` par défaut. Avec PyTorch, la limite est de 256 tokens en
génération simple et de 1024 en chat (1024 au plus). Ce budget inclut le
raisonnement masqué et la réponse visible. Le mode `disabled` a échoué sur des calculs simples
pendant les essais ; conserver `low`. Les réponses tronquées sont signalées
et ne sont pas affichées comme des réponses complètes.

Le lanceur `./opinel` utilise automatiquement le checkpoint de `models/opinel-n8/`
lorsqu'il est présent. Depuis le dossier du projet :

```sh
./opinel doctor
./opinel generate --prompt "Combien font 2 + 2 ?"
# À exécuter dans un terminal local pour essayer le GPU :
./opinel generate --device mps --prompt "Bonjour !"
```

Dans la session de développement, PyTorch 2.15.0 inclut MPS mais le déclare
indisponible ; les essais GPU se font depuis le terminal local. Le code ne calcule les logits
que pour le dernier token. Olala n'a pas de RoPE (`rope_theta = 0`) : la vraie
limite du chemin PyTorch est le préremplissage `eager`, qui matérialise des
scores L × L en FP32 (environ 0,2 Go à 1k tokens, 12,9 Go à 8k, par couche
d'attention).
Le périphérique par défaut est `auto` : MPS lorsqu'il est accessible, sinon CPU.

## Conversation interactive

```sh
./opinel chat --device mps
# Ou laisser le choix automatique MPS/CPU :
./opinel chat
```

Le modèle est chargé une seule fois. Saisir un message puis Entrée ; la réponse
finale s'affiche progressivement, mot par mot. Le message « Olala prépare sa
réponse… » indique l'attente ; le préfixe « Olala > » apparaît au début de la
réponse visible. Le raisonnement reste masqué, donc un délai peut précéder les premiers mots.

- `/reset` efface l'historique sans recharger le modèle.
- Avec `--backend mlx`, chaque tour reprend le préremplissage du précédent : le
  prompt du tour n est un préfixe de celui du tour n+1, seuls la réponse
  précédente et le nouveau message sont calculés. L'instantané est pris juste
  avant la réflexion, que l'historique ne garde pas (les états Mamba ne se
  rembobinent pas). Quand un ancien échange est retiré du contexte, le prompt
  est recalculé en entier. La ligne de statistiques indique les tokens repris ;
  `--no-prompt-cache` désactive la reprise. Sur la conversation Saint-Julien,
  l'instantané restauré donne les mêmes logits qu'un préremplissage complet
  coupé au même endroit (top-1 identique à 100 %, écart top-5 ≤ 0,005) ; tout
  changement de découpage du préremplissage, avec ou sans cache, déplace déjà
  les logits d'environ 0,1, ce qui suffit à faire diverger un rejeu greedy
  (`scripts/check_prompt_cache.py`). Les 9 tours prennent 67 s au lieu de 85 s.
- `/quit` ou Ctrl-D quitte la session.
- Ctrl-C pendant une réponse l'interrompt et permet de saisir un nouveau message.
  À l'invite, Ctrl-C quitte la session.

L'historique reste en mémoire pendant la session, sans enregistrement sur disque.
Seules les réponses finales complètes sont conservées. Si une génération est
tronquée ou échoue, l'échange n'entre pas dans l'historique ; un avertissement
précise que le texte éventuellement déjà affiché est incomplet. Les échanges les
plus anciens sont retirés, par paires question/réponse, lorsque le prompt dépasse
le budget disponible ; ce retrait est signalé. Avec MLX, le budget de génération
est de **4096 tokens** en `disabled`, `low` et `medium`, et de **8192** en `high`
(16384 au plus). La fenêtre totale ajoute 7168 tokens pour les instructions,
l'historique et le nouveau message : 11264 tokens par défaut. Pendant
l'analyse masquée, un compteur « réflexion… N tokens » s'affiche sur stderr ;
la ligne de statistiques indique ensuite la vitesse de préremplissage, celle
du décodage et le pic mémoire. Avec PyTorch,
la fenêtre reste de 2048 tokens et l'entrée limitée à 1024. Un message seul trop
long est refusé.
Les options `--reasoning-effort low` et `--max-new-tokens 512` fonctionnent aussi en chat.
`--instructions opinel-fr-v1` ajoute des consignes développeur en français (expérimental,
désactivé par défaut tant que leur effet n'est pas mesuré ; voir `evaluation/README.md`).
`--context-length` règle la fenêtre totale (jusqu'à 32768 pour MLX, 2048 pour
PyTorch). Cette fenêtre doit dépasser le budget `--max-new-tokens`. Avec MLX,
une fenêtre explicite sans budget explicite réserve au plus la moitié de la
fenêtre à la génération.
Une limite explicite plus basse reste respectée ; si elle est atteinte, le message
d'erreur indique le budget épuisé et comment réessayer.

## Backend OVH MLX/Metal avec les poids Opinel

Le backend `mlx` réutilise le module Olala du
[fork OVH mlx-engine](https://github.com/OVHai-LLM/mlx-engine), avec les poids
**Opinel N8**. Il ne télécharge pas les poids du modèle Olala original.
Les sources inchangées et leur licence MIT sont dans `vendor/ovh_olala/` ;
`UPSTREAM.json` fixe le commit `58083ca0ac4ca0654b508f74e1bdf7b6e0e173f4`
et les empreintes des fichiers. Le serveur complet OVH n'est pas nécessaire.

**État de validation (26 novembre 2026, M5, 24 Go) :**

| Export | Décodage | Pic mémoire en chat | Accord top-1 | Écart top-5 | Erreurs franches |
|---|---|---|---|---|---|
| N°8 direct (`mlx-n8`) | 76–80 tokens/s | 7,9–9,1 Go | 98,7 % | 0,074 | 15 / 12 395 |
| BF16 (`bf16-reference`) | 45–46 tokens/s | ≈ 14 Go de poids | 99,0 % | 0,060 | 10 / 12 650 |

La parité est mesurée contre la référence PyTorch FP32 (experts INT8) sur
62 prompts × 256 tokens, dont deux documents de 3000 et 4500 tokens, avec
`OLALA_DETERMINISTIC=1`. Les deux exports font mieux que les repères de
l'équipe Opinel (GPU : 97,7–97,9 % et 0,098–0,104 ; plancher bf16 CPU :
97,9–98,2 % et 0,089–0,091), aussi sur les documents longs (N°8 : 98,0 % et
0,065). N°8 reste légèrement derrière BF16, alors qu'il n'arrondit qu'une
échelle par ligne. La précision du produit matriciel quantifié de MLX est la
cause la plus probable ; l'écart reste faible. Sur GSM8K (300 questions,
protocole de la fiche), N°8 direct obtient 95,0 % en `low` et 94,3 % en
`medium`, pour 94,3 % et 95,0 % annoncés : les deux écarts (±0,7 point) sont
dans l'erreur-type de 1,3 point. Les erreurs relues sont des fautes de
raisonnement, presque toujours sur les mêmes questions dans les deux modes,
pas des échecs d'extraction ; les 600 réponses donnent un nombre final. Le rejeu de la conversation
Saint-Julien (`scripts/diagnose_n8.sh`) montre les mêmes erreurs factuelles
avec les poids BF16 en greedy qu'avec N°8 : elles viennent du modèle, pas de
la conversion. La conversion Q8 n'a pas été mesurée. Le contexte de 8192 tokens
a été testé avec le vrai tokenizer ; une génération utilisant toute la
fenêtre reste à mesurer.

Pour essayer les poids BF16 déjà exportés avec le moteur MLX :

```sh
./opinel chat --backend mlx
# Greedy, avec un budget de raisonnement plus large :
./opinel chat --backend mlx --temperature 0 --reasoning-effort high
```

Pour transcrire N°8 directement au format MLX, puis l'utiliser :

```sh
./opinel convert-mlx
./opinel chat --backend mlx
```

Le lanceur choisit l'environnement `.venv-mlx` et charge, dans cet ordre, le
premier dossier présent : `models/opinel-n8-mlx-n8/`, `models/opinel-n8-mlx-q8/`,
`models/opinel-n8-mlx-bf16/`. Le format et le chemin chargés sont affichés.
`--model-dir` permet de choisir explicitement un export ; son fichier
`OPINEL_PROVENANCE.json` est obligatoire. Sans `--backend mlx`, le lanceur
conserve le backend PyTorch.

**Recette `n8` (par défaut).** Suivant l'issue #1 (§1, §7), les codes INT8 de
N°8 se transcrivent exactement au format affine 8 bits de MLX : `u = q + 128`,
`scale = weight_scale` (une valeur par ligne, répétée sur des groupes de 128),
`bias = -128 · scale`. Seule l'échelle est arrondie en bf16, au plus 2⁻⁸ en
relatif et de façon identique sur toute la ligne ; aucun poids ne change de
grille. Restent en flottant : l'embedding (BF16 dans N°8), `in_proj_small`
(déquantifié exactement en FP32, comme l'attend le module OVH), le routeur
`moe_gate` (FP32) et les normes. La conversion, faite par `opinel_n8_mlx.py`,
tourne sur CPU sans Metal et sans export BF16 intermédiaire. Elle vérifie
chaque tenseur quantifié : sa déquantification FP32 doit valoir exactement
`q × bf16(scale)`. Elle relit ensuite les fragments écrits et termine par un
chargement strict avec mlx-lm. Le manifeste enregistre le format `mlx-n8` et
l'arrondi d'échelle maximal observé.

**Recettes de comparaison.** `--recipe q8` passe par l'ancienne voie, en deux
étapes :

1. Déquantification N8 en BF16, avec conservation des tenseurs FP32 d'origine,
   traduction des noms des projections et vérification intégrale. L'export de
   13,58 Go est écrit par fragments et n'est publié qu'après vérification.
2. Recette OVH Q8 affine, 8 bits par groupe de 64, avec fusion des projections
   et chargement natif strict de contrôle. Cette seconde quantification ajoute
   un arrondi par poids ; elle n'est pas une copie numérique exacte de N8.
   L'embedding reste désormais en BF16.

`./opinel convert-mlx --export-only` exécute uniquement la première étape.
`--recipe bf16` conserve cette référence sans nouvelle quantification.
`--output-dir` choisit une autre destination. Aucun export existant n'est écrasé ;
une référence BF16 réutilisée est vérifiée de nouveau. Les licences et notices
du modèle sont copiées avec les poids.

Pour recréer l'environnement MLX (Python arm64 3.12 sur ce Mac) :

```sh
/opt/homebrew/bin/python3.12 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install -r requirements-mlx.lock.txt
./opinel doctor --backend mlx
```

Les versions MLX/Metal sont fixées à 0.32.3 et mlx-lm au commit
`87b7b583a697537aa68f47130b40884700b5f55f`. Le chat conserve les commandes,
l'historique borné et le filtrage des canaux.

Pour comparer les mêmes questions sur les deux backends, depuis le terminal :

```sh
.venv/bin/python scripts/check_model.py --backend torch --device mps
.venv-mlx/bin/python scripts/check_model.py --backend mlx --device mps
```

Les résultats et durées sont écrits dans `.cache/torch-model-check.json` et
`.cache/mlx-model-check.json`. Les quatre questions sont des contrôles simples,
pas une validation complète de la parité numérique ou de la qualité.

### Diagnostic de la qualité des réponses

**N°8 direct ou modèle ?** `scripts/diagnose_n8.sh` sépare les causes. Il rejoue
la conversation réelle de `evaluation/conversation_saint_julien.json` en greedy
avec les poids BF16, puis avec N°8 direct, puis avec N°8 échantillonné à
T = 0,8, et affiche les réponses tour par tour. Il mesure ensuite la parité
des deux exports contre la référence FP32. Les étapes déjà faites sont
sautées ; les résultats sont dans `.cache/diagnostic-n8/`.

```sh
scripts/diagnose_n8.sh          # tout (≈ 30-60 min)
scripts/diagnose_n8.sh rejeu    # rejeu seulement (≈ 5 min)
scripts/diagnose_n8.sh parite   # parité seulement
scripts/diagnose_n8.sh cache    # reprise du cache de prompt : rejeu avec et sans, puis logits
```

Des conversations réelles ont montré des erreurs factuelles répétées, un refus
de prendre en compte des corrections et une confusion entre « tutoie-moi » et
une demande de tutoriel. Les tests logiciels et la vitesse de décodage ne
mesurent pas ces défauts. Aucun correctif général des connaissances n'est revendiqué.

`scripts/diagnose_quality.py` rejoue des cas identiques, dont le premier message
exact de la conversation signalée, sur chaque backend :

```sh
.venv/bin/python scripts/diagnose_quality.py --backend torch --device cpu
.venv-mlx/bin/python scripts/diagnose_quality.py --backend mlx --device mps
# Référence MLX sans les kernels Metal fusionnés OVH :
.venv-mlx/bin/python scripts/diagnose_quality.py --backend mlx --device mps --reference
```

Les captures JSON sont dans `.cache/quality-<backend>-<effort>.json` (mode
`mlx-reference` pour la troisième commande). Elles nécessitent une lecture
humaine : une génération terminée n'est pas un verdict de réponse correcte.
Les liens de référence servent à évaluer les réponses et ne sont pas injectés
dans les prompts. `--case fondue_comparison_exact` cible le premier message ;
`--reasoning-effort medium` permet une comparaison séparée du réglage `low`.

Le diagnostic PyTorch CPU en `low` a reproduit la demande « tutoie moi »
interprétée comme un tutoriel culinaire, et le comparatif exact des deux fondues
a aussi produit des confusions factuelles. Sur un cas court distinct, le modèle
a cependant accepté une correction vers le Beaufort. Ces observations confirment
des défauts dans la version de référence ; elles ne remplacent pas une comparaison
numérique entre les kernels MLX fusionnés et leur chemin de référence.

### Évaluation de fiabilité (25 novembre 2026)

`evaluation/reliability_fr.json` fixe 30 cas en français (faits, raisonnement,
corrections, consignes, incertitude). Commandes, grilles et règles de relecture :
voir [`evaluation/README.md`](evaluation/README.md). Les rapports sont écrits
dans `.cache/evaluations/`, hors Git.

**Opinel N8 et Olala original**, avec le même fallback PyTorch CPU, le preset
`baseline` (`low`, déterministe, 1024 tokens) et la graine 0 :

| | Opinel N8 | Olala original |
|---|---|---|
| Réussis | 27 | 26 |
| Échecs | 2 (`tutoie moi`, `vouvoie moi`) | 1 (« metro » au lieu de « mètre ») |
| Budget de 1024 tokens épuisé | 0 | 3 (fondue moitié-moitié, `tutoie moi`, `vouvoie moi`) |
| Incertain | 1 (x indéterminé, avec une contrainte inventée « entre 0 et 10 ») | 0 |
| Latence médiane par cas | 25,7 s | 28,7 s |

Les verdicts concordent sur 25 cas. Aucune des deux versions ne satisfait
« tutoie moi » ni « vouvoie moi » : Opinel produit une recette au vouvoiement,
puis continue à tutoyer ; l'original n'arrive à aucune réponse finale dans le
budget. Ce défaut ne semble donc pas introduit par les retouches Opinel ou la
quantification N8. Les écarts restants (trois cas) sont trop peu nombreux pour
classer les deux versions.

**Parité PyTorch/MLX** (poids Opinel BF16, 3 prompts, 25 positions avec
préremplissage et pas en cache ; captures MLX faites depuis le terminal du Mac) :

| Comparaison | Accord top-1 | KL moyenne | KL max | Écart moyen de log-prob. |
|---|---|---|---|---|
| MLX référence → MLX fusionné | 25/25 | 0,0031 | 0,041 | 0,109 |
| PyTorch CPU → MLX référence | 24/25 | 0,0096 | 0,089 | 0,161 |
| PyTorch CPU → MLX fusionné | 24/25 | 0,0067 | 0,079 | 0,099 |

Le seul désaccord top-1 est une quasi-égalité, dans le calcul `7 × 8`.
PyTorch donne « 1 » à −0,90 et « 7 » à −1,02 (écart de 0,13 nat), alors que
les deux chemins MLX préfèrent « 7 ». Les kernels Metal fusionnés OVH restent
proches de leur chemin de référence. Sur ces entrées, rien n'indique un bug
numérique majeur expliquant à lui seul les erreurs observées en conversation.

Ces premières mesures ne se comparent pas aux repères de l'équipe Opinel
(issue #1, §6) : l'écart était moyenné sur tout le vocabulaire, où la queue
domine ; les deux « références » étaient en BF16 (la ligne « plancher bf16 »,
pas une référence FP32) ; et le texte forcé ne couvrait que 25 positions
proches du protocole. `scripts/check_logit_parity.py` suit désormais la
méthode de l'équipe (voir `evaluation/README.md`) ; ses résultats sont dans
la section du backend MLX.

**Limites.** Une seule exécution déterministe, sur 30 cas ciblés : ce n'est
pas une mesure de fiabilité générale. Les réponses libres ont été jugées par
un assistant (`kind=assistant`), sans validation humaine indépendante. Le preset
`ovh` (échantillonnage) et les graines multiples n'ont pas été évalués. La parité
porte sur trois prompts courts, sans seuil d'acceptation ; le Q8 MLX n'est
pas couvert. **Les hallucinations et les consignes ignorées ne sont pas
corrigées** : ces mesures les situent sans les résoudre.

### Consignes développeur `opinel-fr-v1` (26 novembre 2026)

Mesure faite avec MLX BF16 fusionné sur le GPU du M5, preset `baseline`, graine 0,
sans puis avec le profil, sur la suite principale (30 cas) et la suite témoin (14 cas) :

| | Principale sans | Principale avec | Témoin sans | Témoin avec |
|---|---|---|---|---|
| Réussis | 27 | 23 | 10 | 10 |
| Échecs | 3 | 7 | 4 | 4 |

**Le profil n'est pas retenu : il reste désactivé par défaut.**
- Il supprime les dérives vers un tutoriel ou une recette (« tutoie moi »
  dans les contextes fondue et Python, « arrête de me vouvoyer »). En
  revanche, le modèle n'applique toujours pas le bon registre : il annonce
  « je vous tutoie » ou inverse la demande. Sur la suite témoin, les quatre
  cas de registre libres échouent avec comme sans consignes.
- Il dégrade d'autres consignes. Les réponses deviennent des phrases au lieu
  d'un mot (« kelvin »), les résumés perdent une idée exigée, et le modèle
  ne demande plus le document ou les données manquantes. « vouvoie moi »,
  réussi sans consignes sur MLX, échoue avec.
- Il corrige un fait (fondue moitié-moitié), sur une seule exécution.

Le tutoiement et le vouvoiement semblent donc mal maîtrisés par le modèle
lui-même ; une consigne générale ne suffit pas sur cette taille de modèle.

Sans consignes, MLX et PyTorch donnent 27/30 tous les deux, mais pas sur
les mêmes cas. Avec MLX, la fondue moitié-moitié devient « Gruyère et
Emmental » et « vouvoie moi » réussit. Avec PyTorch, c'est l'inverse. Les
écarts numériques mesurés plus haut sont faibles, mais une quasi-égalité
suffit à faire diverger une génération déterministe. Un seul passage par
réglage ne permet donc pas de départager des variantes à un ou deux cas près.

### Variabilité sans consignes (26 novembre 2026)

Six exécutions de la suite principale et cinq de la suite témoin, toutes sans
consignes. Les variantes déterministes sont PyTorch CPU, MLX fusionné et MLX
de référence ; le preset `ovh` (`medium`, température 0,8) est lancé avec les
graines 0, 1 et 2. Scores : principale 27 à 28/30, témoin 10 à 11/14.
Cas qui échouent au moins une fois :

| Cas | Réussites | Lecture |
|---|---|---|
| `i01` « tutoie moi » (fondue) | 0/6 | recette dans 5 cas, budget épuisé dans 1 |
| `hi01` « tutoie moi » (Python) | 0/5 | tutoriel Python dans 4 cas, budget épuisé dans 1 |
| `hi02` « on peut se tutoyer ? » | 0/5 | accepte, mais continue à vouvoyer |
| `hi05` tutoiement déjà convenu | 0/5 | revient au vouvoiement |
| `hi03` « arrête de me vouvoyer » | 1/5 | continue à vouvoyer |
| `i02` « vouvoie moi » | 3/6 | dépend du moteur et de la graine |
| `f04` unité de longueur | 2/6 | « metro » au lieu de « mètre » avec MLX, y compris sans kernels fusionnés |
| `f01`, `f05`, `u04` | 5/6 | erreur ou écart de format isolés |

**Les demandes de tutoiement échouent de façon stable**, quels que soient le
moteur, les kernels, le raisonnement (`low` ou `medium`) ou l'échantillonnage.
C'est un défaut du modèle lui-même, pas du portage Mac : l'équipe Opinel l'a
confirmé dans l'[issue #1](../../issues/1), l'échec existant déjà dans le
modèle d'origine. **Le moteur ne le corrige donc pas volontairement** (ni
règles de détection, ni réponse fixe) ; ces cas restent dans les suites comme
échecs connus du modèle. Le passage au
vouvoiement est instable, sans échec systématique. Les cas d'incertitude
(ne pas inventer un fichier, un prénom, une citation, un prix) réussissent
dans toutes les exécutions. Le preset `ovh` ne fait pas mieux que `baseline`
de façon mesurable, et il est environ deux fois plus lent.

Les cas témoins ont maintenant été lus. Un futur profil (`opinel-fr-v2`)
devra être ajusté sur la suite principale seulement, et validé sur de
nouveaux cas témoins.

### Contrôle de bout en bout sur GSM8K

`scripts/evaluate_gsm8k.py` reproduit le protocole de la fiche (issue #1,
§13). Il prend les 300 premières questions du jeu de test d'origine, épinglé à
un commit et vérifié par SHA256 dans `.cache/gsm8k/`, et ajoute à chacune
`Give the final answer as a number on the last line.`. Il échantillonne en
0,8 / 0,9 / 40 dans l'ordre de vLLM, avec un budget de 6144 tokens, et retient
le dernier nombre du canal final. Le rapport est écrit après chaque question
et peut être repris avec `--resume`. Repères de la fiche pour N°8 : 94,3 % en
`low` et 95,0 % en `medium`. À n = 300, l'erreur-type est d'environ 1,4 point ;
le résumé signale un écart de plus de 3 points. Compter environ une heure
par mode depuis le terminal du Mac (médiane de 7,7 s par question en `low`,
9,6 s en `medium`) :

```sh
.venv-mlx/bin/python scripts/evaluate_gsm8k.py --reasoning-effort low \
  --output .cache/evaluations/gsm8k-n8-low.json
.venv-mlx/bin/python scripts/evaluate_gsm8k.py --summary .cache/evaluations/gsm8k-n8-low.json
```

## Télégraphe : décodage spéculatif (en cours)

[Télégraphe](https://huggingface.co/maurienne-ai/Olala-v1.0-mini-7A1B-Telegraphe)
est le drafter de l'équipe Opinel : il propose K tokens d'un coup, que la cible
vérifie en une seule passe. `opinel_telegraphe.py` le porte en MLX et
`opinel_speculative.py` fait la boucle greedy (étapes 1 et 2 sur 4 ;
l'échantillonnage et l'intégration au chat restent à faire). Le drafter lit le
flux résiduel de la cible après les couches 1, 9, 17, 25 et 33, la table
d'embedding normalisée de la cible (comme vLLM) et, au choix, sa propre tête
bf16 ou la tête INT8 de la cible. Les poids sont téléchargés à une révision
épinglée et vérifiés par SHA256 ; l'embedding du token masque est lu dans les
octets bruts de `mask_embedding.pt`, sans exécuter le pickle.

Acceptation hors ligne (`scripts/telegraphe_acceptance.py`) : la cible écrit
une continuation greedy de 256 tokens pour 16 prompts variés ; à chaque position,
Télégraphe propose 7 tokens à partir du contexte, comptés tant qu'ils égalent le
choix de la cible. Tokens par passe (acceptés + 1) :

| Configuration | K = 3 | K = 5 | K = 7 | Une passe du drafter |
|---|---|---|---|---|
| Tête bf16 du drafter | 2,81 | 3,43 | 3,82 | ≈ 7,5 ms |
| Tête INT8 de la cible | 2,84 | 3,49 | 3,88 | ≈ 7,4 ms |
| Tête INT8 de la cible, couches 8 bits | 2,82 | 3,43 | 3,81 | ≈ 6,3 ms |
| Fiche Télégraphe (RTX 5090, 2 000 réponses) | 2,96 | 3,70 | 4,20 | |

Selon le prompt, de 2,27 (espagnol) à 3,51 (calcul) à K = 3. Cette mesure part
de chaque position du texte ; la vraie boucle repart plus souvent juste après un
refus, là où c'est difficile, et accepte moins (voir ci-dessous). Sur un même
texte, la boucle et la mesure hors ligne proposent les mêmes tokens.

**Vérification et retour arrière.** La cible traite l'ancre et les K
propositions en une passe ; si seules les `keep` premières positions sont
acceptées, les caches reviennent à l'état après ces `keep` tokens. Pour
Mamba-3, l'état d'avant la passe et les sorties d'`in_proj` sont gardés pendant
la passe, et la récurrence est rejouée sur les `keep` lignes (`m3_prefill`, sans
relire de poids) ; pour l'attention, le cache KV est raccourci et le cache de
décalage reprend le k / v brut (et les coefficients a_k / a_v) de la dernière
position gardée. Contre une passe directe sur les `keep` tokens
(`scripts/check_speculative.py --rollback-check`) : caches KV et de décalage
identiques bit à bit, état Mamba-3 à 1e-6 près (noyau séquentiel contre noyau
par blocs ; les angles de rotation se comparent modulo 2π). À `keep` = 1, la
passe directe est le décodage fusionné, un autre noyau : écart 1e-2, logits
suivants de même top-1 et à la même distance du préremplissage complet.

**Gain mesuré, greedy** (`scripts/check_speculative.py`, 16 prompts × 256
tokens, tête INT8 de la cible, couches du drafter en 8 bits), décodage normal
77,3 tokens/s :

| K | Tokens/s | Gain | Tokens par passe | Drafter + vérification par passe |
|---|---|---|---|---|
| 3 | 80,6 | ×1,04 | 2,65 | 5,6 + 27,2 ms |
| 5 | 70,5 | ×0,91 | 3,07 | 6,9 + 36,4 ms |
| 7 | 52,7 | ×0,68 | 3,27 | 11,1 + 50,2 ms |

La vérification par le chemin de préremplissage d'OVH coûte presque deux pas de
décodage fusionné (saut fixe d'environ 8 ms dès 2 tokens, `scripts/bench_verify.py`,
signalé dans l'issue #1) : le gain reste marginal sans un chemin de vérification
fusionné pour les petits blocs.

**D'où vient le saut** (`scripts/profile_verify.py`). Un seul token passé par le
chemin générique couche par couche (`OLALA_FUSED=0`) coûte déjà 25 ms, contre
14,7 ms par le décodage fusionné, et 4 tokens 26,7 ms : c'est le chemin, pas le
nombre de tokens. Avec une barrière après chaque partie, à 4 tokens : MoE
17,7 ms, Mamba-3 10,1 ms, attention 8,5 ms, le reste 4,5 ms (somme sur les 36
couches, barrières comprises). Parmi les réglages d'OVH, seul
`OLALA_TPA_CACHE=full` aide : 22,5 ms au lieu de 26,8 à 4 tokens, décodage
fusionné inchangé (14,6 ms) ; `OLALA_M3_CHUNKED=0`, `OLALA_MOE_V3=0` et
l'absence de `OLALA_DETERMINISTIC` ne changent rien. Avec ce cache complet, la
boucle greedy donne ×1,15 à K = 3 (88,5 tokens/s, 2,73 tokens par passe,
5,5 + 25,2 ms) et ×0,99 à K = 5 ; le cache complet reste à mesurer sur les
longs contextes, où le factorisé est choisi sur M5.

**Vérification par les noyaux de décodage** (`opinel_verify.py`, après la réponse
de l'équipe dans l'issue #1). Le décodage fusionné d'OVH sur T séquences
indépendantes (`scripts/profile_verify.py --rows`) coûte 14,1 / 17,2 / 26,5 /
40,0 ms pour T = 1 / 2 / 4 / 8. Profil par noyau, avec une barrière après
chacun : de T = 1 à 4, le MoE fusionné passe de 7,0 à 9,9 ms (environ 1 ms par
token, les experts lus en plus), mais les projections denses (`gemv`) de 26 à
48 ms : le GEMV relit les poids pour chaque ligne, alors que
`mx.quantized_matmul` coûte autant pour 1 et pour 4 lignes (2,86 et 2,98 ms pour
36 matrices 7328 × 1536). La passe de vérification prend donc les L tokens du
bloc comme lignes du décodage, avec :

- les projections denses par `mx.quantized_matmul` dès 3 lignes, le MoE fusionné
  jusqu'à 8 lignes ;
- Mamba-3 par un seul `m3_prefill` séquentiel pour le bloc (option : un pas de
  décodage par token, dont les états intermédiaires servent directement au
  retour arrière) ;
- l'attention par les noyaux de décodage, un token après l'autre (causale par
  construction, cache écrit sur place).

| Tokens vérifiés | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| chemin générique (ms) | 24,0 | 25,8 | 25,7 | 32,6 | 35,9 |
| noyaux de décodage (ms) | 18,4 | 20,6 | 21,6 | 30,4 | 36,8 |

Mêmes top-1 que le chemin générique (100 %), écart top-5 0,02 ; après un retour
arrière, la suite est celle d'une passe directe sur les tokens gardés. Ce que
chaque partie coûte à 4 tokens : l'attention générique 7,8 ms pour ses 7
couches (le gros du coût fixe), Mamba-3 en pas successifs 7,2 ms (environ
1,8 ms par token), autant ou plus qu'un `m3_prefill` par bloc.

Boucle greedy, mesures alternées prompt par prompt (le Mac chauffe pendant les
25 minutes du run et ralentit toutes les variantes ; mesurées l'une après
l'autre, les dernières paraissaient plus lentes) :

| Cache d'attention | K = 2 | K = 3 | K = 4 |
|---|---|---|---|
| factorisé (par défaut sur M5) | ×1,16 | **×1,19** | ×1,15 |
| complet (`OLALA_TPA_CACHE=full`) | ×1,08 | ×1,08 | ×1,06 |

À K = 3, une passe coûte le drafter (6,4 ms) plus la vérification (30,4 ms en
boucle, machine chaude).

**Construction du graphe.** Python met environ 2 ms à construire le graphe d'une
vérification avant que le GPU ne démarre. `opinel_verify` lance donc l'évaluation
(`mx.async_eval`) toutes les 3 couches : le GPU traite le début pendant que la
suite se construit. Banc d'essai : 16,9 ms au lieu de 19,9 à 2 tokens, 20,6 au
lieu de 23,0 à 4. Dans la boucle, mesures alternées
(`check_speculative.py --layers-per-submit 0 3`, machine au calme) :

| Envoi au GPU | K = 2 | K = 3 |
|---|---|---|
| à la fin | ×1,17 | ×1,21 |
| toutes les 3 couches | ×1,21 | **×1,24** |

Le gain est plus petit en boucle (0,5 ms par vérification) qu'au banc : la
boucle évalue déjà le drafter et les caches entre deux vérifications. Les
mesures se font machine au calme et sur secteur : sur batterie, ou pendant un
inventaire logiciel (`ndtrack`, `flxfsscan`), le décodage normal tombe de 77 à
43-62 tokens/s et la boucle, plus dépendante du CPU, encore davantage.

**Drafter.** Une passe à K = 3 coûte environ 5,4 ms : la tête 2 ms (233 Mo
INT8 lus à environ 115 Go/s, déjà au plafond de bande passante et constante de 2
à 5 lignes), les 5 couches 2,4 ms (126 Mo, limitées par les lancements de
noyaux), les transitions 0,4 ms. Trois changements exacts (brouillons identiques
à l'ancienne version) : `mx.compile` des parties de taille fixe de chaque
couche, q/k/v et gate/up fusionnées en une matrice chacune, sélection des 16
candidats sans copie négative des logits. Brouillon isolé : 6,3 → 5,6 ms ; dans
la boucle, mesures alternées passe par passe : 5,7 → 5,2 ms. C'est 1,5 % d'une
passe, sous le bruit d'un run à l'autre (×1,20 à K = 3 sur le dernier run, ×1,24
sur le précédent). Une tête réduite aux tokens fréquents ne paierait pas : les
64 000 premiers ids ne couvrent que 94 % des tokens d'un texte français (98,7 %
en anglais), et chaque token hors liste casse la chaîne du brouillon.

**Échantillonnage par rejet** (`rejection_step` dans `opinel_speculative.py`).
Avec température, le drafter tire chaque token dans sa loi q (softmax des scores
de transition sur ses 16 candidats, à la même température) ; la cible garde le
token avec la probabilité min(1, p/q), p étant la loi de vLLM (température,
top-k, top-p, `vllm_law` dans `opinel_mlx.py`) ; au premier refus, le token
suivant est tiré dans max(p − q, 0) renormalisée, et après K acceptations dans
p. La sortie suit exactement la loi de la cible quelle que soit q
(`tests/test_speculative.py`, 60 000 tirages). Avec top-k = 40, la décision se
prend sur le CPU sur 40 valeurs par position. Réglages du chat (0,8 / 0,9 / 40,
`check_speculative.py --temperature 0.8 --top-p 0.9 --top-k 40`, mesures
alternées) :

| K | 2 | 3 | 4 |
|---|---|---|---|
| tokens par passe | 2,26 | 2,61 | 2,82 |
| gain | **×1,15** | ×1,14 | ×1,09 |

L'acceptation reste proche du greedy (2,61 contre 2,67 à K = 3).

En résumé, la boucle greedy plafonne vers ×1,2 sur M5 : une vérification de 4
tokens coûte encore près de deux pas de décodage (27 ms contre 14-15), et le
drafter un peu plus d'un tiers de pas. Les réponses greedy divergent du décodage
normal au bout d'environ un tiers des tokens (médiane), l'effet attendu d'autres
noyaux sur un choix greedy.

## Pistes de l'issue #1 non traitées

Les conseils suivants de l'[issue #1](../../issues/1) restent à faire ou à
mesurer :

- forcer le canal final (`<|channel_end|><|channel_start|>final<|content|>`)
  quand l'analyse épuise son budget, au lieu de jeter l'échange (§11) ;
- charger N°4 directement : ses experts sont déjà au format affine 4 bits de
  MLX, avec des groupes de 32 (§1) ; à faire après validation de N°8 ;
- mesurer `OLALA_TPA_TGS` (nombre de cœurs GPU) et `OLALA_TPA_CACHE=full` contre
  `factored` sur les longs contextes (§8) ;
- au-delà de 32768 tokens : `"slw_wsize": 65536` dans le `config.json` MLX
  reproduit le fork, à valider par un test d'aiguille (§12) ;
- décodage PyTorch MPS sans synchronisation par couche (`moe_decode`, §9) :
  PyTorch reste surtout la référence CPU ;
- Télégraphe (§5) : drafter, boucle greedy et vérification par les noyaux de décodage faits,
  gain ×1,2 à K = 3 en greedy, ×1,15 à K = 2 en échantillonné (rejet exact) ;
  reste l'intégration au chat ;

Les demandes explicites de tutoiement ou de vouvoiement ne figurent pas dans
cette liste : c'est une limite du modèle, que le moteur ne corrige pas.

## Vérification et limites

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv-mlx/bin/python -m unittest discover -s tests -v
# Vérification du modèle réel (charge environ 14 Go) :
.venv/bin/python scripts/check_model.py --device cpu
```

Les 92 tests unitaires passent dans chacun des deux environnements. Les tests
qui demandent PyTorch ou MLX sont explicitement ignorés lorsque la
bibliothèque est absente. Le chargement réel et la
génération avec cache ont été exécutés sur CPU, et une génération MPS a été
confirmée depuis le terminal du Mac. La parité numérique complète avec le serveur
du créateur reste à vérifier.
Les vérifications réelles avec `low` ont donné `4` pour `2 + 2` en français
et en anglais, `56` pour `7 × 8`, et `Paris` pour la capitale de la France.
Sur ces essais, la génération a pris environ 13 à 17 secondes par question,
préremplissage et raisonnement compris, après le chargement du modèle.
Le backend PyTorch ne porte ni les kernels CUDA du fork, ni Télégraphe, ni son
serveur API. Le backend MLX utilise les kernels Metal du module OVH ; le BF16
atteint environ 45–46 tokens/s au décodage dans les échanges rapportés sur le M5.
Les performances Q8 restent à mesurer.

Le modèle amont est sous licence Apache 2.0 ; le téléchargement conserve `LICENSE` et `NOTICE`.
