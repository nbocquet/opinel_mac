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
La génération est déterministe, avec un raisonnement `low` et une limite de
256 tokens en génération simple, 1024 en chat. Ce budget inclut le raisonnement
masqué et la réponse visible. Le mode `disabled` a échoué sur des calculs simples
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
indisponible ; les essais GPU se font depuis le terminal local. Le code limite le cache
RoPE à 2048 positions et ne calcule les logits que pour le dernier token.
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
- `/quit` ou Ctrl-D quitte la session.
- Ctrl-C pendant une réponse l'interrompt et permet de saisir un nouveau message.
  À l'invite, Ctrl-C quitte la session.

L'historique reste en mémoire pendant la session, sans enregistrement sur disque.
Seules les réponses finales complètes sont conservées. Si une génération est
tronquée ou échoue, l'échange n'entre pas dans l'historique ; un avertissement
précise que le texte éventuellement déjà affiché est incomplet. Les échanges les
plus anciens sont retirés, par paires question/réponse, lorsque le prompt dépasse
le budget disponible ; ce retrait est signalé. Avec MLX, la fenêtre totale est
de **8192 tokens** par défaut, dont 1024 réservés à la génération : instructions,
historique et nouveau message disposent donc de 7168 tokens. Avec PyTorch,
la fenêtre reste de 2048 tokens et l'entrée limitée à 1024. Un message seul trop
long est refusé.
Les options `--reasoning-effort low` et `--max-new-tokens 512` fonctionnent aussi en chat.
`--context-length` règle la fenêtre totale (jusqu'à 32768 pour MLX, 2048 pour
PyTorch). Cette fenêtre doit dépasser le budget `--max-new-tokens`.
Une limite explicite plus basse reste respectée ; si elle est atteinte, le message
d'erreur indique le budget épuisé et comment réessayer.

## Backend OVH MLX/Metal avec les poids Opinel

Le backend `mlx` réutilise le module Olala du
[fork OVH mlx-engine](https://github.com/OVHai-LLM/mlx-engine), avec les poids
**Opinel N8**. Il ne télécharge pas les poids du modèle Olala original.
Les sources inchangées et leur licence MIT sont dans `vendor/ovh_olala/` ;
`UPSTREAM.json` fixe le commit `58083ca0ac4ca0654b508f74e1bdf7b6e0e173f4`
et les empreintes des fichiers. Le serveur complet OVH n'est pas nécessaire.

**État de validation :** l'environnement `.venv-mlx` est installé et la génération
BF16 a été confirmée depuis le terminal du M5 : environ 45–46 tokens/s au décodage.
Les 945 tenseurs exportés ont été comparés intégralement, sans différence,
aux valeurs utilisées par le backend PyTorch. La session de développement ne
peut pas initialiser Metal ; la conversion Q8 et sa validation restent à faire
depuis le terminal local. Le contexte de 8192 tokens a été testé avec le vrai
tokenizer ; une génération GPU utilisant toute cette fenêtre reste à mesurer.

Pour essayer les poids BF16 déjà exportés avec le moteur MLX :

```sh
./opinel chat --backend mlx
# Même réglage, explicite :
./opinel chat --backend mlx --context-length 8192
```

Pour produire la version MLX Q8 puis l'utiliser :

```sh
./opinel convert-mlx
./opinel chat --backend mlx
```

Le lanceur choisit l'environnement `.venv-mlx` et utilise le dossier
`models/opinel-n8-mlx-q8/` s'il existe, sinon `models/opinel-n8-mlx-bf16/`.
Le format et le chemin chargés sont affichés. `--model-dir` permet de choisir
explicitement un export ; son fichier `OPINEL_PROVENANCE.json` est obligatoire.
Sans `--backend mlx`, le lanceur conserve le backend PyTorch.

La conversion se fait en deux étapes :

1. Déquantification N8 en BF16, avec conservation des tenseurs FP32 d'origine,
   traduction des noms des projections et vérification intégrale. L'export de
   13,58 Go est écrit par fragments et n'est publié qu'après vérification.
2. Recette OVH Q8 affine, 8 bits par groupe de 64, avec fusion des projections
   et chargement natif strict de contrôle. Cette seconde quantification ajoute
   un arrondi ; elle n'est pas une copie numérique exacte du N8 déquantifié.

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
l'historique borné et le filtrage des canaux. La génération reste déterministe
pour faciliter les comparaisons.

Pour comparer les mêmes questions sur les deux backends, depuis le terminal :

```sh
.venv/bin/python scripts/check_model.py --backend torch --device mps
.venv-mlx/bin/python scripts/check_model.py --backend mlx --device mps
```

Les résultats et durées sont écrits dans `.cache/torch-model-check.json` et
`.cache/mlx-model-check.json`. Les quatre questions sont des contrôles simples,
pas une validation complète de la parité numérique ou de la qualité.

### Diagnostic de la qualité des réponses

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

## Vérification et limites

```sh
.venv/bin/python -m unittest discover -s tests -v
# Vérification du modèle réel (charge environ 14 Go) :
.venv/bin/python scripts/check_model.py --device cpu
```

Les 42 tests unitaires passent avec les dépendances installées. Sans PyTorch,
les tests numériques sont explicitement ignorés. Le chargement réel et la
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
