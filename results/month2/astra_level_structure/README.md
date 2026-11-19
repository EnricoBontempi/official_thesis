# Astra level/structure — pacchetto tecnico per Neon (A3 tecnico, nessuna interpretazione)

Branch `mission/astra-level`. Contratto, controlli, limiti e comandi: `docs/month2/astra_analysis_handoff.md`.
Dati: valutazione offline dei quattro best S5 originali (seed0) su seen220 e VAL123, 5% velocità, maschere
persistite, repliche 0/1/2 (tre maschere, NON tre seed). Analisi esplorativa: VAL ha influenzato selezione/arresto S5.

Definizione di macro (uniforme): metrica per frame → media sui frame di una condizione (`sim_id`, NaN saltati,
conteggio dei frame validi riportato) → media sulle condizioni = valore della replica. `replica=mean` = media
aritmetica dei tre valori di replica, con `rep_min`/`rep_max`. Seen (20 condizioni) e VAL (3) non sono mai fusi.

## tables/
| File | Contenuto |
|---|---|
| `per_condition_metrics.csv` | per (role, model, treatment, replica, sim_id): media sui frame di tutte le metriche all/off, conteggi regionali medi, offset, `*__n_valid_frames` |
| `macro_metrics_long.csv` | macro per replica 0/1/2 e `mean` (+min/max) di tutte le metriche, domini `all_` (campo completo) e `off_` (16375 siti fuori dalle nove ancore) |
| `macro_headline_wide.csv` | RMSE, MAE, bias firmato, |bias|, centered, replica-mean, all/off, in forma larga |
| `macro_regional_long.csv` | hot/peak/jet/nonhot RMSE, hot IoU, area calda pred/true (replica-mean) |
| `macro_valid_counts_long.csv`, `region_valid_frame_counts.csv` | conteggi medi di pixel validi per regione e frame validi |
| `offsets_per_condition.csv`, `offsets_macro_long.csv` | offset oracle/AO9, errore di stima AO9−oracle (firmato e assoluto), residuo medio sulle ancore, `mse_recovery_vs_oracle` (definizione congelata, per frame) |
| `paired_diff_per_condition.csv`, `paired_diff_macro_long.csv` | differenze appaiate A−B a livello frame: AO9−free, oracle−free, AO9−oracle, on−off (U-Net, GLU, Gappy), U-Net−GLU (p_on e p_off); conteggi dei segni; nessun test di significatività |
| `information_labels.csv` | informazione disponibile a ciascun modello/trattamento (neurali, Gappy on/off, ParamMean, GlobalMean, T-only) |
| `completeness_rows.csv`, `completeness_nan_columns.csv` | righe attese/osservate; NaN legittimi (solo T-only, colonne offset) |
| `figure_scales.json`, `figure_panel_check.csv` | scale comuni, soglie, siti T9; controllo dei 36 pannelli contro l'RMSE del CSV (max 5e-7 K) |
| `aggregation_meta.json` | elenco tabelle e definizione di macro |

Colonne: `model` = run S5 o metodo baseline; `treatment` ∈ free/oracle/ao9 (T-only: `t_only_const`);
`metric` = `all_*`/`off_*` (`bias_K` = pred−true firmato; `abs_bias_K` = |bias| del frame, poi mediato).

## figures/
Sei PNG: tre condizioni VAL × {campi T, errore firmato}, frame 1000, replica 0, quattro modelli neurali ×
{truth, free, oracle, AO9}. Scala T comune a tutti i pannelli (50–2750 K), scala errore simmetrica comune (±700 K),
contorno 2100 K, nove siti T9 visibili. Soglie: hot ≥2100 K, peak ≥2200 K, jet <1800 K. Le baseline non hanno
campi salvati: non sono disegnate.

## verification/
`replay_k0_verification.json` / `_cells.csv` (replay sostitutivo, vedi handoff), `baseline_mask_crosscheck.json`.

## provenance/
`astra_outputs_sha256.txt` (hash degli output Astra su Engaging), `sacct_astra_jobs.psv` (i log Slurm `*.out` sono ignorati da git: restano su Engaging in `~/official_thesis_astra_results/slurm_logs/`).
Non inclusi (pesanti/ignorati da git): checkpoint, per-frame CSV grezzi, i quattro `val_frame1000_rep0_fields.npz`
(restano su Engaging `~/official_thesis_astra_results/level_structure_v1/<run>/`).
