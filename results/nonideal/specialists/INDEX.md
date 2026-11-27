# Specialisti p_on (Blocco III): consegna pubblicata

Solo tabelle aggregate, figure di riepilogo, controlli e manifest. Nessun checkpoint, campo, cache o CSV per frame. Percorsi completi sul cluster: `/home/bontempi/official_thesis_nonideal_spec_pkg` (pacchetto), `/home/bontempi/official_thesis_nonideal_spec_runs` (run).

- Sintesi scientifica verificata: [docs/nonideal/05_specialist_results.md](../../../docs/nonideal/05_specialist_results.md).
- `package/`: `COMPLETE.json`, `FINAL_SUMMARY.md`, `INDEX.md`, `package_manifest.json`, `tables/` (`per_run_scenario.csv` 720 righe, `seed_summary.csv`, `diagonal_own_scenario.csv`, `training_diagnostics.csv`), `figures/` (fig1–3 dell'aggregatore). Non copiati: `per_condition.csv` (4 MB) e `macro_by_replica.csv`.
- `delivery_check/`: controllo di sola lettura di 30 run, 60 gate, 720 righe, ricalcolo per frame, hash.
- `reading/`: tabelle derivate da `per_run_scenario.csv` con [scripts/72_nonideal_specialist_reading.py](../../../scripts/72_nonideal_specialist_reading.py). Segno: guadagno = originale − specialista (positivo: specialista migliore); costo sul pulito = specialista − originale.
- `graphics/`: 16 PNG di matrici e dispersioni, `NOTE.md`, manifest, `BASELINE_FIELDS_NEEDED.md`. Le 30 schede per run e le schede di variabilità (83 MB) restano sul cluster in `.../graphics/`.
- `freeze.json`, `submission.json`, `chain_ids.json`, `post_hook_submitted.txt`, `graphics_hook.json` (originale), `graphics_hook_v2.json` (tentativo fallito 30685595 e riuscito 30711670).

SHA: training `b6fc9fefda5a1d4776c87c9285d7187c0d89fc48`, post-processing `b1c095507b803e5fa27ad8ba8bdad2febd8d2541`, grafici `c5963373c94e8ea7fa3a71855b7cb0493c2481b4`.
