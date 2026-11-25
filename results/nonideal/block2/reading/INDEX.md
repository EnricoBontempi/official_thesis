# Blocco II — lettura per la discussione

Figure e controlli sulle tabelle già pubblicate dei 12 modelli. Nessuna nuova valutazione, nessuna inferenza, nessuna modifica del pacchetto originale.

Nota: [docs/nonideal/03_block2_reading.md](../../../../docs/nonideal/03_block2_reading.md).
Report della campagna: [docs/nonideal/02_block2_robustness.md](../../../../docs/nonideal/02_block2_robustness.md).
Pacchetto originale: [../package/INDEX.md](../package/INDEX.md).

Base di questa lettura: commit `d4f2e3a58da9c95bb2259f2130cfbe41727df955`, branch di partenza `nonideal/block1-degradation-spec`.
Script: [scripts/63_nonideal_block2_reading.py](../../../../scripts/63_nonideal_block2_reading.py). Legge solo i CSV in `results/nonideal/block2/package/tables/` e riscrive questa cartella.

## Cosa è stato ricalcolato

Dalle tabelle aggregate, con lo script:

- le 48 RMSE medie del report, entro 0,02 K (`tables/rmse_report_check.csv`);
- i punti disegnati, un seed già mediato sulle tre repliche di maschera (`tables/delta_by_seed_plotted.csv`, `tables/g1_vs_g1eq_by_seed.csv`, `tables/seen_g1_by_condition_seed.csv`, `tables/val_g1_g1eq_by_condition_seed.csv`);
- il controllo delle 460 combinazioni famiglia × φ/NH3 × condizione × cella × ruolo, di cui 9 migliorano.

Hash dei CSV sorgente e delle figure prodotte: `tables/file_sha256.csv`.

## Cosa è stato solo letto

- `results/nonideal/block2/VERIFY_OK.json`: 12/12 completi, `problems` vuoto.
- `results/nonideal/block2/package/COMPLETE.json`: controllo di completezza del pacchetto.
- Nel report: 74 088 righe scientifiche, gate R0, ricalcolo indipendente dei macro dal CSV per frame (scarto ≤ 1,4·10⁻¹⁴ K), job della campagna.
- `tables/sites_and_area.csv` del pacchetto: 167–243 siti rimossi, media circa 207 su 819, area dei dischi 25,0–26,0 %. Non è stato ricalcolato dai frame.

`package/tables/per_frame_all.csv.gz` è citato dal manifest e non è in questo checkout. `fields_figure_frames.npz` non è nel repository. Per questo non ci sono pannelli nuovi di temperatura predetta.

## Figure riusate

Dal pacchetto originale, non ridisegnate:

| file | domanda a cui risponde |
|---|---|
| [fig1_macro_rmse_by_cell.png](../package/figures/fig1_macro_rmse_by_cell.png) | RMSE assoluta nei sei scenari, seen220 e VAL123, tre seed |
| [fig2_deterioration_vs_R0.png](../package/figures/fig2_deterioration_vs_R0.png) | aumento rispetto al proprio caso pulito, tre seed |
| [fig5_fields_O-C149-on-s0.png](../package/figures/fig5_fields_O-C149-on-s0.png) e gli altri tre `fig5_fields_*.png` | temperatura vera e modulo dell'errore, scala 0–400 K, siti tolti e contorno dei dischi, seed 0 |

In fig1 e fig2 ogni punto è un seed di training, mediato sulle tre repliche di maschera. `p_on` significa che φ e la percentuale di NH3 sono ingressi; `p_off` che entrambi sono esclusi.

`fig3_val_by_condition.png` e `fig4_level_vs_structure.png` restano nel pacchetto. Per la discussione si usano le figure sotto: la prima non ha legenda, la seconda ha etichette contratte.

## Figure nuove

| file | cosa mostra |
|---|---|
| [fig_g1_vs_g1eq.png](figures/fig_g1_vs_g1eq.png) | lacune contigue contro lo stesso numero di siti a caso, e il rapporto fra i due aumenti di RMSE |
| [fig_bias_centered.png](figures/fig_bias_centered.png) | variazione del bias e della centered-RMSE rispetto al proprio caso pulito |
| [fig_regional_jets_hot.png](figures/fig_regional_jets_hot.png) | aumento di RMSE nei getti (T vera < 1800 K), nella zona calda (≥ 2100 K) e nel picco (≥ 2200 K) |
| [fig_conditions.png](figures/fig_conditions.png) | condizioni seen sotto i dischi; su VAL, dischi e siti a caso, RMSE e bias |

Colori: blu U-Net con φ e NH3, verde U-Net senza, arancio GLU con, rosa GLU senza. Cerchio, quadrato e triangolo sono i seed 0, 1 e 2. Non sono intervalli di confidenza.

## Come rigenerare

```bash
python scripts/63_nonideal_block2_reading.py
```

Lo script si ferma se le RMSE medie non coincidono col report entro 0,02 K, o se i contrasti per condizione non sono 460 con 9 miglioramenti.
