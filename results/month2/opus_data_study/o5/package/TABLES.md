# Tabelle O5, lettura descrittiva

Generato da `scripts/29_opus_o5_package.py`. Nessun rango, nessun test, nessuna conclusione. RMSE in K, media sulle condizioni e sulle repliche.

## RMSE globale al 5% per run e checkpoint

| run_id | checkpoint_kind | epoch | seen220 | ref760 | VAL123 |
| --- | --- | --- | --- | --- | --- |
| O-B19-on-s0 | best | 50 | 57.56 | 41.43 | 43.93 |
| O-B19-on-s0 | best_seen_diagnostic | 50 | 57.56 | 41.43 | 43.93 |
| O-B19-on-s0 | last | 620 | 66.32 | 35.13 | 50.25 |
| O-B19-off-s0 | best | 60 | 65.85 | 44.78 | 46.74 |
| O-B19-off-s0 | best_seen_diagnostic | 40 | 65.37 | 48.29 | 48.20 |
| O-B19-off-s0 | last | 630 | 73.95 | 39.02 | 53.60 |
| O-A38-on-s0 | best | 120 | 50.85 | 16.26 | 42.39 |
| O-A38-on-s0 | best_seen_diagnostic | 120 | 50.85 | 16.26 | 42.39 |
| O-A38-on-s0 | last | 630 | 55.87 | 7.16 | 45.50 |
| O-A38-off-s0 | best | 90 | 56.99 | 26.46 | 42.92 |
| O-A38-off-s0 | best_seen_diagnostic | 130 | 56.59 | 19.67 | 44.71 |
| O-A38-off-s0 | last | 660 | 62.83 | 7.84 | 49.48 |
| O-A38-on-s1 | best | 60 | 51.58 | 30.59 | 48.22 |
| O-A38-on-s1 | best_seen_diagnostic | 60 | 51.58 | 30.59 | 48.22 |
| O-A38-on-s1 | last | 630 | 56.84 | 7.07 | 51.33 |
| O-A38-off-s1 | best | 40 | 61.04 | 44.81 | 43.06 |
| O-A38-off-s1 | best_seen_diagnostic | 110 | 54.83 | 20.03 | 44.50 |
| O-A38-off-s1 | last | 610 | 60.98 | 8.79 | 49.17 |
| O-C149-on-s0 | best | 90 | 45.42 | 38.24 | 38.71 |
| O-C149-on-s0 | best_seen_diagnostic | 220 | 39.31 | 20.43 | 38.97 |
| O-C149-on-s0 | last | 660 | 40.75 | 15.07 | 39.90 |
| O-C149-off-s0 | best | 260 | 41.99 | 20.26 | 39.82 |
| O-C149-off-s0 | best_seen_diagnostic | 240 | 41.28 | 20.61 | 40.52 |
| O-C149-off-s0 | last | 710 | 42.88 | 14.95 | 41.92 |

## Delta appaiati al 5% (a meno b), per condizione poi media

Dati: stessa p e seed 0. Seed: A38 seed 1 meno seed 0. p: on meno off. `n_a_lower` conta le condizioni in cui a ha RMSE minore.

| label | checkpoint_a | role | mean_delta | min_delta | max_delta | n_conditions | n_a_lower | n_a_higher |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B19-A38 seed0 p_off | best | seen_train_eval | 8.86 | -12.79 | 32.05 | 20 | 2 | 18 |
| B19-A38 seed0 p_off | best | val_eval | 3.82 | -0.52 | 9.80 | 3 | 1 | 2 |
| B19-A38 seed0 p_off | best_seen_diagnostic | seen_train_eval | 8.78 | 1.47 | 21.97 | 20 | 0 | 20 |
| B19-A38 seed0 p_off | best_seen_diagnostic | val_eval | 3.49 | 1.96 | 6.17 | 3 | 0 | 3 |
| B19-A38 seed0 p_off | last | seen_train_eval | 11.12 | -5.77 | 35.72 | 20 | 1 | 19 |
| B19-A38 seed0 p_off | last | val_eval | 4.12 | 1.67 | 6.63 | 3 | 0 | 3 |
| B19-A38 seed0 p_on | best | seen_train_eval | 6.71 | -0.34 | 26.22 | 20 | 1 | 19 |
| B19-A38 seed0 p_on | best | val_eval | 1.54 | -2.15 | 7.07 | 3 | 2 | 1 |
| B19-A38 seed0 p_on | best_seen_diagnostic | seen_train_eval | 6.71 | -0.34 | 26.22 | 20 | 1 | 19 |
| B19-A38 seed0 p_on | best_seen_diagnostic | val_eval | 1.54 | -2.15 | 7.07 | 3 | 2 | 1 |
| B19-A38 seed0 p_on | last | seen_train_eval | 10.45 | 0.16 | 39.48 | 20 | 0 | 20 |
| B19-A38 seed0 p_on | last | val_eval | 4.75 | 1.27 | 7.70 | 3 | 0 | 3 |
| C149-A38 seed0 p_off | best | seen_train_eval | -15.00 | -41.10 | -3.08 | 20 | 20 | 0 |
| C149-A38 seed0 p_off | best | val_eval | -3.09 | -4.34 | -0.62 | 3 | 3 | 0 |
| C149-A38 seed0 p_off | best_seen_diagnostic | seen_train_eval | -15.31 | -39.52 | -2.26 | 20 | 20 | 0 |
| C149-A38 seed0 p_off | best_seen_diagnostic | val_eval | -4.19 | -8.21 | -0.81 | 3 | 3 | 0 |
| C149-A38 seed0 p_off | last | seen_train_eval | -19.95 | -52.37 | -1.16 | 20 | 20 | 0 |
| C149-A38 seed0 p_off | last | val_eval | -7.56 | -13.93 | -2.28 | 3 | 3 | 0 |
| C149-A38 seed0 p_on | best | seen_train_eval | -5.43 | -22.49 | 7.43 | 20 | 11 | 9 |
| C149-A38 seed0 p_on | best | val_eval | -3.68 | -8.68 | -0.57 | 3 | 3 | 0 |
| C149-A38 seed0 p_on | best_seen_diagnostic | seen_train_eval | -11.54 | -27.81 | -0.60 | 20 | 20 | 0 |
| C149-A38 seed0 p_on | best_seen_diagnostic | val_eval | -3.42 | -8.02 | -0.66 | 3 | 3 | 0 |
| C149-A38 seed0 p_on | last | seen_train_eval | -15.12 | -34.15 | -0.73 | 20 | 20 | 0 |
| C149-A38 seed0 p_on | last | val_eval | -5.60 | -10.71 | -1.45 | 3 | 3 | 0 |
| C149-B19 seed0 p_off | best | seen_train_eval | -23.86 | -60.58 | -5.83 | 20 | 20 | 0 |
| C149-B19 seed0 p_off | best | val_eval | -6.91 | -10.42 | -3.82 | 3 | 3 | 0 |
| C149-B19 seed0 p_off | best_seen_diagnostic | seen_train_eval | -24.09 | -50.29 | -8.91 | 20 | 20 | 0 |
| C149-B19 seed0 p_off | best_seen_diagnostic | val_eval | -7.67 | -14.38 | -2.77 | 3 | 3 | 0 |
| C149-B19 seed0 p_off | last | seen_train_eval | -31.07 | -75.29 | -1.27 | 20 | 20 | 0 |
| C149-B19 seed0 p_off | last | val_eval | -11.68 | -20.56 | -6.33 | 3 | 3 | 0 |
| C149-B19 seed0 p_on | best | seen_train_eval | -12.14 | -39.70 | 3.68 | 20 | 16 | 4 |
| C149-B19 seed0 p_on | best | val_eval | -5.22 | -7.64 | -1.49 | 3 | 3 | 0 |
| C149-B19 seed0 p_on | best_seen_diagnostic | seen_train_eval | -18.25 | -47.82 | -4.36 | 20 | 20 | 0 |
| C149-B19 seed0 p_on | best_seen_diagnostic | val_eval | -4.96 | -8.64 | -0.37 | 3 | 3 | 0 |
| C149-B19 seed0 p_on | last | seen_train_eval | -25.57 | -64.19 | -1.09 | 20 | 20 | 0 |
| C149-B19 seed0 p_on | last | val_eval | -10.35 | -11.98 | -9.15 | 3 | 3 | 0 |
| A38 seed0 on-off | best | seen_train_eval | -6.14 | -26.46 | 4.35 | 20 | 17 | 3 |
| A38 seed0 on-off | best | val_eval | -0.53 | -4.56 | 5.32 | 3 | 2 | 1 |
| A38 seed0 on-off | best_seen_diagnostic | seen_train_eval | -5.74 | -16.89 | 3.28 | 20 | 18 | 2 |
| A38 seed0 on-off | best_seen_diagnostic | val_eval | -2.32 | -9.72 | 4.98 | 3 | 2 | 1 |
| A38 seed0 on-off | last | seen_train_eval | -6.96 | -22.55 | 2.52 | 20 | 19 | 1 |
| A38 seed0 on-off | last | val_eval | -3.98 | -11.58 | 5.15 | 3 | 2 | 1 |
| A38 seed1 on-off | best | seen_train_eval | -9.46 | -23.33 | -1.40 | 20 | 20 | 0 |
| A38 seed1 on-off | best | val_eval | 5.16 | -3.66 | 17.09 | 3 | 1 | 2 |
| A38 seed1 on-off | best_seen_diagnostic | seen_train_eval | -3.25 | -25.26 | 6.45 | 20 | 11 | 9 |
| A38 seed1 on-off | best_seen_diagnostic | val_eval | 3.72 | -8.91 | 15.49 | 3 | 1 | 2 |
| A38 seed1 on-off | last | seen_train_eval | -4.14 | -19.43 | 6.88 | 20 | 17 | 3 |
| A38 seed1 on-off | last | val_eval | 2.16 | -7.64 | 9.45 | 3 | 1 | 2 |
| B19 seed0 on-off | best | seen_train_eval | -8.29 | -27.44 | -1.04 | 20 | 20 | 0 |
| B19 seed0 on-off | best | val_eval | -2.81 | -12.44 | 3.02 | 3 | 1 | 2 |
| B19 seed0 on-off | best_seen_diagnostic | seen_train_eval | -7.81 | -23.92 | 6.65 | 20 | 16 | 4 |
| B19 seed0 on-off | best_seen_diagnostic | val_eval | -4.27 | -16.19 | 2.89 | 3 | 1 | 2 |
| B19 seed0 on-off | last | seen_train_eval | -7.63 | -32.72 | 4.43 | 20 | 17 | 3 |
| B19 seed0 on-off | last | val_eval | -3.35 | -12.93 | 4.75 | 3 | 2 | 1 |
| C149 seed0 on-off | best | seen_train_eval | 3.43 | -6.39 | 9.20 | 20 | 5 | 15 |
| C149 seed0 on-off | best | val_eval | -1.11 | -3.51 | 0.96 | 3 | 2 | 1 |
| C149 seed0 on-off | best_seen_diagnostic | seen_train_eval | -1.97 | -10.32 | 1.22 | 20 | 17 | 3 |
| C149 seed0 on-off | best_seen_diagnostic | val_eval | -1.55 | -2.99 | 0.50 | 3 | 2 | 1 |
| C149 seed0 on-off | last | seen_train_eval | -2.12 | -10.30 | 2.40 | 20 | 16 | 4 |
| C149 seed0 on-off | last | val_eval | -2.02 | -4.67 | 0.90 | 3 | 2 | 1 |
| A38 seed1-seed0 p_off | best | seen_train_eval | 4.05 | -2.98 | 15.77 | 20 | 6 | 14 |
| A38 seed1-seed0 p_off | best | val_eval | 0.14 | -0.86 | 1.09 | 3 | 1 | 2 |
| A38 seed1-seed0 p_off | best_seen_diagnostic | seen_train_eval | -1.76 | -12.42 | 9.76 | 20 | 12 | 8 |
| A38 seed1-seed0 p_off | best_seen_diagnostic | val_eval | -0.21 | -1.93 | 3.07 | 3 | 2 | 1 |
| A38 seed1-seed0 p_off | last | seen_train_eval | -1.85 | -10.83 | 3.99 | 20 | 12 | 8 |
| A38 seed1-seed0 p_off | last | val_eval | -0.31 | -1.63 | 2.10 | 3 | 2 | 1 |
| A38 seed1-seed0 p_on | best | seen_train_eval | 0.73 | -11.24 | 8.25 | 20 | 6 | 14 |
| A38 seed1-seed0 p_on | best | val_eval | 5.83 | -2.17 | 20.79 | 3 | 2 | 1 |
| A38 seed1-seed0 p_on | best_seen_diagnostic | seen_train_eval | 0.73 | -11.24 | 8.25 | 20 | 6 | 14 |
| A38 seed1-seed0 p_on | best_seen_diagnostic | val_eval | 5.83 | -2.17 | 20.79 | 3 | 2 | 1 |
| A38 seed1-seed0 p_on | last | seen_train_eval | 0.97 | -2.77 | 6.87 | 20 | 4 | 16 |
| A38 seed1-seed0 p_on | last | val_eval | 5.83 | 2.53 | 12.27 | 3 | 0 | 3 |

## Selezione: last, best_seen meno best (stesso run)

| label | role | mean_delta | min_delta | max_delta | n_conditions | n_a_lower | n_a_higher |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O-A38-off-s0 best_seen_diagnostic-best | seen_train_eval | -0.40 | -10.76 | 6.64 | 20 | 13 | 7 |
| O-A38-off-s0 best_seen_diagnostic-best | val_eval | 1.79 | -2.34 | 7.38 | 3 | 1 | 2 |
| O-A38-off-s0 last-best | seen_train_eval | 5.84 | -11.50 | 20.33 | 20 | 7 | 13 |
| O-A38-off-s0 last-best | val_eval | 6.56 | -0.87 | 15.51 | 3 | 1 | 2 |
| O-A38-off-s0 last-best_seen_diagnostic | seen_train_eval | 6.24 | -3.30 | 16.16 | 20 | 5 | 15 |
| O-A38-off-s0 last-best_seen_diagnostic | val_eval | 4.77 | 1.47 | 8.13 | 3 | 0 | 3 |
| O-A38-off-s1 best_seen_diagnostic-best | seen_train_eval | -6.21 | -19.44 | 10.08 | 20 | 17 | 3 |
| O-A38-off-s1 best_seen_diagnostic-best | val_eval | 1.44 | -2.52 | 5.25 | 3 | 1 | 2 |
| O-A38-off-s1 last-best | seen_train_eval | -0.06 | -22.50 | 14.27 | 20 | 9 | 11 |
| O-A38-off-s1 last-best | val_eval | 6.11 | 2.09 | 13.91 | 3 | 0 | 3 |
| O-A38-off-s1 last-best_seen_diagnostic | seen_train_eval | 6.15 | -4.84 | 15.79 | 20 | 6 | 14 |
| O-A38-off-s1 last-best_seen_diagnostic | val_eval | 4.67 | 0.50 | 8.65 | 3 | 0 | 3 |
| O-A38-on-s0 best_seen_diagnostic-best | seen_train_eval | 0.00 | 0.00 | 0.00 | 20 | 0 | 0 |
| O-A38-on-s0 best_seen_diagnostic-best | val_eval | 0.00 | 0.00 | 0.00 | 3 | 0 | 0 |
| O-A38-on-s0 last-best | seen_train_eval | 5.02 | -2.56 | 13.89 | 20 | 4 | 16 |
| O-A38-on-s0 last-best | val_eval | 3.11 | -1.82 | 6.27 | 3 | 1 | 2 |
| O-A38-on-s0 last-best_seen_diagnostic | seen_train_eval | 5.02 | -2.56 | 13.89 | 20 | 4 | 16 |
| O-A38-on-s0 last-best_seen_diagnostic | val_eval | 3.11 | -1.82 | 6.27 | 3 | 1 | 2 |
| O-A38-on-s1 best_seen_diagnostic-best | seen_train_eval | 0.00 | 0.00 | 0.00 | 20 | 0 | 0 |
| O-A38-on-s1 best_seen_diagnostic-best | val_eval | 0.00 | 0.00 | 0.00 | 3 | 0 | 0 |
| O-A38-on-s1 last-best | seen_train_eval | 5.26 | -8.76 | 21.11 | 20 | 9 | 11 |
| O-A38-on-s1 last-best | val_eval | 3.11 | -10.33 | 9.93 | 3 | 1 | 2 |
| O-A38-on-s1 last-best_seen_diagnostic | seen_train_eval | 5.26 | -8.76 | 21.11 | 20 | 9 | 11 |
| O-A38-on-s1 last-best_seen_diagnostic | val_eval | 3.11 | -10.33 | 9.93 | 3 | 1 | 2 |
| O-B19-off-s0 best_seen_diagnostic-best | seen_train_eval | -0.48 | -11.95 | 8.93 | 20 | 11 | 9 |
| O-B19-off-s0 best_seen_diagnostic-best | val_eval | 1.46 | 0.14 | 3.75 | 3 | 0 | 3 |
| O-B19-off-s0 last-best | seen_train_eval | 8.10 | -8.17 | 25.82 | 20 | 4 | 16 |
| O-B19-off-s0 last-best | val_eval | 6.86 | 3.70 | 12.34 | 3 | 0 | 3 |
| O-B19-off-s0 last-best_seen_diagnostic | seen_train_eval | 8.58 | -11.25 | 30.03 | 20 | 6 | 14 |
| O-B19-off-s0 last-best_seen_diagnostic | val_eval | 5.40 | 3.56 | 8.59 | 3 | 0 | 3 |
| O-B19-on-s0 best_seen_diagnostic-best | seen_train_eval | 0.00 | 0.00 | 0.00 | 20 | 0 | 0 |
| O-B19-on-s0 best_seen_diagnostic-best | val_eval | 0.00 | 0.00 | 0.00 | 3 | 0 | 0 |
| O-B19-on-s0 last-best | seen_train_eval | 8.76 | -6.40 | 27.38 | 20 | 5 | 15 |
| O-B19-on-s0 last-best | val_eval | 6.32 | -1.18 | 11.84 | 3 | 1 | 2 |
| O-B19-on-s0 last-best_seen_diagnostic | seen_train_eval | 8.76 | -6.40 | 27.38 | 20 | 5 | 15 |
| O-B19-on-s0 last-best_seen_diagnostic | val_eval | 6.32 | -1.18 | 11.84 | 3 | 1 | 2 |
| O-C149-off-s0 best_seen_diagnostic-best | seen_train_eval | -0.70 | -3.70 | 0.68 | 20 | 14 | 6 |
| O-C149-off-s0 best_seen_diagnostic-best | val_eval | 0.70 | -0.22 | 1.19 | 3 | 1 | 2 |
| O-C149-off-s0 last-best | seen_train_eval | 0.89 | -2.20 | 5.60 | 20 | 8 | 12 |
| O-C149-off-s0 last-best | val_eval | 2.10 | 1.19 | 2.91 | 3 | 0 | 3 |
| O-C149-off-s0 last-best_seen_diagnostic | seen_train_eval | 1.60 | -1.49 | 7.00 | 20 | 7 | 13 |
| O-C149-off-s0 last-best_seen_diagnostic | val_eval | 1.40 | -0.01 | 2.41 | 3 | 1 | 2 |
| O-C149-on-s0 best_seen_diagnostic-best | seen_train_eval | -6.11 | -10.53 | -2.34 | 20 | 20 | 0 |
| O-C149-on-s0 best_seen_diagnostic-best | val_eval | 0.26 | -1.01 | 1.12 | 3 | 1 | 2 |
| O-C149-on-s0 last-best | seen_train_eval | -4.67 | -11.64 | 4.34 | 20 | 18 | 2 |
| O-C149-on-s0 last-best | val_eval | 1.19 | -2.69 | 3.41 | 3 | 1 | 2 |
| O-C149-on-s0 last-best_seen_diagnostic | seen_train_eval | 1.44 | -1.38 | 7.05 | 20 | 6 | 14 |
| O-C149-on-s0 last-best_seen_diagnostic | val_eval | 0.93 | -1.69 | 2.29 | 3 | 1 | 2 |

## Metriche regionali non definite su alcuni frame

72 gruppi (run, checkpoint, ruolo, copertura, metrica) hanno frame con metrica regionale non definita (NaN, per esempio picco vuoto). Le medie li saltano. Elenco: `tables/undefined_regional_metric_frames.csv`. Le metriche di base (RMSE, MAE, bias, centered) sono finite ovunque.

## O5 al 5% contro il loop alla stessa epoca (tolleranza 0.05 K)

Train e seen: replica 0. VAL: media delle tre maschere.

| role | n | max_abs_delta_K | n_within |
| --- | --- | --- | --- |
| seen_train_eval | 24 | 0.0005 | 24 |
| train_fit | 24 | 0.0003 | 24 |
| val_eval | 24 | 0.0011 | 24 |
