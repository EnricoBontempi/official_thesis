# Missione 02 — Fase 1 (F1.1–F1.3): indice degli artefatti

Report: [../../../mese3/02_virtual_piv/fase1_report_R1.md](../../../mese3/02_virtual_piv/fase1_report_R1.md) · Nota di orientamento: [../../../mese3/02_virtual_piv/orientation_note.md](../../../mese3/02_virtual_piv/orientation_note.md)
Riproduzione: `bash scripts/piv_phase1/run_all.sh` · Checksum: [SHA256SUMS](SHA256SUMS) · Ogni `*provenance.json` registra SHA del codice, versioni e file letti con sha256.

| Cartella | Contenuto | Script |
|---|---|---|
| `f11/` | F1.1: spostamenti delle strutture (`displacement_patches.csv`, `displacement_regressions.csv`, `displacement_subset_means.csv`), validazione sintetica del metodo, lag e traccianti (`displacement_lags_passive_tracers.csv`, `lag_joint_regressions.csv`), mappe medie (`mean_maps_train_conditions.npz`), simmetria (`mean_field_symmetry_centre_lines.csv`), decorrelazione (`decorrelation_drivers.csv`), identità cache–raw | `f11_orientation_check.py`, `f11_analyze.py`, `f11b_method_validation.py`, `f11c_mean_symmetry.py`, `f11d_symmetry_and_decorrelation.py` |
| `f12/` | F1.2: registro dei kernel (`kernel_registry.json/csv`, verifica formule), spettri (`spectra_1d.npz`, `spectra_tail_metrics.csv`), funzioni di struttura, effetto del filtro per regione e condizione (`filter_effect.csv`), D-kern (`dkern_mhg_vs_tophat.csv`, `dkern_width_matched.csv`, `tophat_width_matched.json`), contenuto delle finestre B (`b_window_content.json`), contesto e soglie (`f12_context.json`) | `f12_kernel_registry.py`, `f12_grid_support.py`, `f12b_dkern_width_matched.py`, `f12c_b_window_content.py` |
| `f12_swapcells/` | Sensibilità: stessi calcoli con i passi di cella scambiati | `f12_*.py --swap-cells` |
| `f13/` | F1.3: tabella v_ref,c e σ_u (`vref_sigma_u_table.csv`), coerenza fra pool (`vref_consistency.json`) | `f13_vref.py` |
| `tables/` | Tabelle Markdown generate dai CSV (incluse nel report) | `make_tables.py` |
| `figures/` | Figure 1–9 | `make_figures.py` |

Lettura dei dati: solo TRAIN per F1.1 e F1.2; F1.3 usa anche la sola velocità di VAL (autorizzato). TEST mai letto.
