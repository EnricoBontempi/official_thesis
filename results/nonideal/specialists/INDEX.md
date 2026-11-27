# Specialisti p_on (Blocco III): indice della consegna

Report: [docs/nonideal/05_specialist_results.md](../../../docs/nonideal/05_specialist_results.md) (Parte 1 per le persone, Parte 2 = appendice tecnica per un altro agente).

## Stato (27 novembre 2026)

| Livello | Stato |
|---|---|
| Campagna numerica | **verificata** (30 run validi, 60/60 gate, 720 righe, ricalcolo per frame): [delivery_check/](delivery_check/delivery_check.md) |
| Grafici | **disponibili** (job 30711670, 179 file verificati); schede complete nella release |
| Rapporto | **consegnato** |
| Lacune residue | pannelli appaiati baseline–specialista (frame 0/1600 contro 1000; richiederebbero una nuova inferenza, non autorizzata); latenza d'inferenza non misurata; controllo del solo batching GLU non eseguito |

## Nel repository

| Percorso | Contenuto |
|---|---|
| `package/` | `COMPLETE.json`, `FINAL_SUMMARY.md`, `INDEX.md`, `package_manifest.json`, `figures/` (fig1–3) |
| `package/tables/per_run_scenario.csv` | 720 righe: run × checkpoint × scenario × ruolo |
| `package/tables/per_condition.csv.gz` | 24 840 righe: per condizione e replica (gzip senza perdita; sha256 del contenuto = manifest) |
| `package/tables/macro_by_replica.csv.gz` | 2 160 righe: per replica (gzip senza perdita) |
| `package/tables/seed_summary.csv`, `diagonal_own_scenario.csv`, `training_diagnostics.csv` | riassunti per seed, proprio scenario, stati dei training |
| `package/tables/diagnostics_by_run_scenario.csv.gz` | 720 × 83: bias, centered-RMSE, regioni, picco, IoU… specialista/originale/differenza |
| `reading/` | tabelle derivate (`own_scenario_*`, `transfer_*`, `baseline_deterioration`) e `figures/` A–E del report |
| `graphics/` | 16 PNG di matrici/seed/regioni, scheda di esempio, `NOTE.md`, `INDEX.md`, `figure_manifest.json`, `BASELINE_FIELDS_NEEDED.md`, log del job riuscito |
| `accounting/` | `jobs_all_attempts.csv`, `phases.csv`, `training_by_run.csv`, `eval_costs_by_run.csv`, `eval_costs_by_cell.csv`, `raw/` (sacct, ledger) |
| `delivery_check/` | controllo di consegna di sola lettura |
| `freeze.json`, `submission.json`, `chain_ids.json`, `post_hook_submitted.txt` | provenienza della catena |
| `graphics_hook.json`, `graphics_hook_v2.json` | hook originale (non modificato) e registro del tentativo fallito 30685595 e di quello riuscito 30711670 |

Script: `scripts/71_…delivery_check.py`, `72_…reading.py`, `73_…compute_accounting.py`, `74_…diagnostics_table.py`, `75_…report_figures.py`.

## Release tecnica (pre-release, non latest)

Tag `nonideal-specialists-data-v1`: <https://github.com/EnricoBontempi/official_thesis/releases/tag/nonideal-specialists-data-v1>

| Asset | sha256 | Dimensione |
|---|---|---|
| `specialists_graphics_cards_and_variability.tar.gz` (30 schede per run + schede di variabilità) | `73c8c2bee396155afc53ab38de02ab666d2bbb2b23734b9ee18a692577b71c5f` | 81 MB |
| `specialists_eval_per_frame_and_provenance.tar.gz` (CSV per frame di 60 valutazioni, gate, provenance, costi) | `81447353ecbd25a4c4ef6ade2c86e2c50482384821ab577f4b3eea86c085bd52` | 180 MB |
| `blockII_per_frame_all.csv.gz` (CSV per frame dei 12 originali) | `cb96687cf3614c5393bc9658ec3469980313413f495f32930fd2e810ffc070b3` | 38 MB |

## Solo sul cluster (non pubblicato)

Checkpoint e directory dei run (`/home/bontempi/official_thesis_nonideal_spec_runs`, 5,2 GB); campi `fields_val_frame1000_best.npz` (30 × ~2,7 MB, derivati dal dataset); `per_condition.csv` e `macro_by_replica.csv` non compressi (contenuto identico ai `.gz`); log Slurm; cache e dataset. Pacchetto: `/home/bontempi/official_thesis_nonideal_spec_pkg`.

SHA: training `b6fc9fefda5a1d4776c87c9285d7187c0d89fc48`, post-processing `b1c095507b803e5fa27ad8ba8bdad2febd8d2541`, grafici `c5963373c94e8ea7fa3a71855b7cb0493c2481b4`.
