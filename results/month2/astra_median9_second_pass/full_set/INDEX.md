# Insieme completo median9 — indice delle tabelle

Riduzione dei CSV per-frame già validati. Non è il pacchetto dei 12 campi di `../example_frame1000_rep0/`. Seen e VAL non sono mescolati. Nessuna tabella sceglie uno stimatore.

Macro, ovunque: media sui frame di una condizione (`sim_id`), poi media sulle condizioni, poi media delle tre repliche di maschera, con minimo e massimo di replica.

| File | Contenuto |
| --- | --- |
| `tables/per_condition_metrics.csv` | free, oracle, AO9, mean9, median9, per condizione e replica |
| `tables/macro_metrics_long.csv` | stessa macro, lunga |
| `tables/macro_headline_wide.csv` | RMSE, MAE, bias, bias assoluto, centered, campo completo e off-anchor |
| `tables/macro_regional_long.csv` | regioni, area e IoU calda |
| `tables/macro_valid_counts_long.csv` | conteggi regionali |
| `tables/offset_error_per_condition.csv` | offset applicato e distanza dall’oracle |
| `tables/offset_error_macro_long.csv` | stessa quantità in macro |
| `tables/paired_diff_per_condition.csv` | differenza A−B per condizione |
| `tables/paired_diff_macro_long.csv` | differenza A−B in macro, con celle negative e positive |
| `tables/paired_frame_signs.csv` | conteggio dei frame in cui A−B è negativo, positivo o nullo |
| `tables/anchor_residual_per_condition.csv` | nove residui e contributo ad AO9, con riga e colonna |
| `tables/anchor_col95_share_macro_long.csv` | quota della somma dei contributi assoluti che viene dalle due ancore con colonna 95 |
| `tables/t_only_per_condition.csv` | solo il T-only storico |
| `tables/t_only_macro_long.csv` | macro del T-only |
| `tables/information_labels.csv` | che informazione entra in ciascun trattamento |
| `tables/aggregation_meta.json` | gate, SHA e definizione delle differenze |

Per RMSE, MAE e bias assoluto, un valore A−B negativo significa che A è più vicino al vero di B. Un valore positivo è un peggioramento. I due segni restano entrambi.
