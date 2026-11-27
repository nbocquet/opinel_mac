# Synthèse des benchmarks

Généré par `scripts/bench_summary.py` ; protocole dans `benchmarks/README.md`.

| Machine | Date | Puce | Décodage normal | Vérif 4 tokens | Greedy K = 3 | Échantillonné K = 2 | Chat varié |
|---|---|---|---|---|---|---|---|
| m4-mini | 2026-11-26 | Apple M4 | 61,6 tokens/s | 27,8 ms (préremplissage) | ×1,27 | ×1,13 | ×1,07 |
| m5-macbookair | 2026-11-26 | Apple M5 | 65,3 tokens/s | 20,4 ms (décodage) | ×1,22 | ×1,18 | ×1,09 |
| m4-mini | 2026-11-27 | Apple M4 | 61,0 tokens/s | 28,5 ms (préremplissage) | ×1,32 | ×1,16 | ×1,08 |
| m5-macbookair | 2026-11-27 | Apple M5 | 77,7 tokens/s | 20,6 ms (décodage) | ×1,32 | ×1,24 | ×1,13 |
