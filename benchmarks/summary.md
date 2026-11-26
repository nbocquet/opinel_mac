# Synthèse des benchmarks

Généré par `scripts/bench_summary.py` ; protocole dans `benchmarks/README.md`.

| Machine | Date | Puce | Décodage normal | Vérif 4 tokens | Greedy K = 3 | Échantillonné K = 2 | Chat varié |
|---|---|---|---|---|---|---|---|
| m4-mini | 2026-11-26 | Apple M4 | 61,6 tokens/s | 27,8 ms (préremplissage) | ×1,27 | ×1,13 | ×1,07 |
| m5-macbookair | 2026-11-26 | Apple M5 | 65,3 tokens/s | 20,4 ms (décodage) | ×1,22 | ×1,18 | ×1,09 |
