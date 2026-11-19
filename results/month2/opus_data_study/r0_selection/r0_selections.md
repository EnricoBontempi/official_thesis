# R0 — rianalisi delle quattro storie S5

Selezioni calcolate sulle curve già registrate. Scheduler e early-stop hanno usato VAL: questa tabella non è una validazione indipendente e non crea checkpoint intermedi.

| Run | Regola | Epoca | Update | Score | VAL | seen | Condizioni (RMSE K) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| s5_sparse_unet_pon_seed0 | original_best_val | 60 | 5700 | 42.9641 | 42.9641 | 51.4914 | 0NH3_1.05 44.914, 20NH3_1.1 44.055, 40NH3_0.9 39.923 |
| s5_sparse_unet_pon_seed0 | original_last | 630 | 59850 | 49.4956 | 49.4956 | 56.2171 | 0NH3_1.05 53.724, 20NH3_1.1 52.548, 40NH3_0.9 42.215 |
| s5_sparse_unet_pon_seed0 | min_seen | 70 | 6650 | 50.7661 | 43.2027 | 50.7661 | 0NH3_1.05 43.980, 20NH3_1.1 42.157, 40NH3_0.9 43.471 |
| s5_sparse_unet_pon_seed0 | causal_ma5_first_min | 90 | 8550 | 45.2303 | 46.7194 | 52.2382 | 0NH3_1.05 44.144, 20NH3_1.1 46.260, 40NH3_0.9 49.754 |
| s5_sparse_unet_pon_seed0 | loo_holdout_0NH3_1.05.h5 | 60 | 5700 | 41.9889 | 42.9641 | 51.4914 | 0NH3_1.05 44.914, 20NH3_1.1 44.055, 40NH3_0.9 39.923 |
| s5_sparse_unet_pon_seed0 | loo_holdout_20NH3_1.1.h5 | 60 | 5700 | 42.4188 | 42.9641 | 51.4914 | 0NH3_1.05 44.914, 20NH3_1.1 44.055, 40NH3_0.9 39.923 |
| s5_sparse_unet_pon_seed0 | loo_holdout_40NH3_0.9.h5 | 70 | 6650 | 43.0685 | 43.2027 | 50.7661 | 0NH3_1.05 43.980, 20NH3_1.1 42.157, 40NH3_0.9 43.471 |
| s5_sparse_unet_poff_seed0 | original_best_val | 140 | 13300 | 43.5253 | 43.5253 | 55.2761 | 0NH3_1.05 51.364, 20NH3_1.1 42.286, 40NH3_0.9 36.926 |
| s5_sparse_unet_poff_seed0 | original_last | 650 | 61750 | 47.6716 | 47.6716 | 61.7115 | 0NH3_1.05 60.461, 20NH3_1.1 45.617, 40NH3_0.9 36.937 |
| s5_sparse_unet_poff_seed0 | min_seen | 120 | 11400 | 55.2118 | 45.1110 | 55.2118 | 0NH3_1.05 53.840, 20NH3_1.1 40.979, 40NH3_0.9 40.514 |
| s5_sparse_unet_poff_seed0 | causal_ma5_first_min | 140 | 13300 | 44.4871 | 43.5253 | 55.2761 | 0NH3_1.05 51.364, 20NH3_1.1 42.286, 40NH3_0.9 36.926 |
| s5_sparse_unet_poff_seed0 | loo_holdout_0NH3_1.05.h5 | 150 | 14250 | 39.1572 | 45.1666 | 56.0298 | 0NH3_1.05 57.185, 20NH3_1.1 41.968, 40NH3_0.9 36.346 |
| s5_sparse_unet_poff_seed0 | loo_holdout_20NH3_1.1.h5 | 140 | 13300 | 44.1450 | 43.5253 | 55.2761 | 0NH3_1.05 51.364, 20NH3_1.1 42.286, 40NH3_0.9 36.926 |
| s5_sparse_unet_poff_seed0 | loo_holdout_40NH3_0.9.h5 | 70 | 6650 | 45.9418 | 43.6860 | 56.6833 | 0NH3_1.05 51.253, 20NH3_1.1 40.630, 40NH3_0.9 39.174 |
| s5_adapted_glu_pon_seed0 | original_best_val | 440 | 41800 | 45.8007 | 45.8007 | 56.0591 | 0NH3_1.05 53.499, 20NH3_1.1 54.659, 40NH3_0.9 29.244 |
| s5_adapted_glu_pon_seed0 | original_last | 890 | 84550 | 46.8392 | 46.8392 | 55.4785 | 0NH3_1.05 56.355, 20NH3_1.1 56.024, 40NH3_0.9 28.139 |
| s5_adapted_glu_pon_seed0 | min_seen | 660 | 62700 | 55.3382 | 46.9445 | 55.3382 | 0NH3_1.05 56.847, 20NH3_1.1 55.830, 40NH3_0.9 28.157 |
| s5_adapted_glu_pon_seed0 | causal_ma5_first_min | 480 | 45600 | 46.3328 | 45.9678 | 56.2324 | 0NH3_1.05 54.866, 20NH3_1.1 54.556, 40NH3_0.9 28.481 |
| s5_adapted_glu_pon_seed0 | loo_holdout_0NH3_1.05.h5 | 480 | 45600 | 41.5185 | 45.9678 | 56.2324 | 0NH3_1.05 54.866, 20NH3_1.1 54.556, 40NH3_0.9 28.481 |
| s5_adapted_glu_pon_seed0 | loo_holdout_20NH3_1.1.h5 | 440 | 41800 | 41.3714 | 45.8007 | 56.0591 | 0NH3_1.05 53.499, 20NH3_1.1 54.659, 40NH3_0.9 29.244 |
| s5_adapted_glu_pon_seed0 | loo_holdout_40NH3_0.9.h5 | 360 | 34200 | 52.1981 | 48.0028 | 59.9856 | 0NH3_1.05 53.777, 20NH3_1.1 50.619, 40NH3_0.9 39.612 |
| s5_adapted_glu_poff_seed0 | original_best_val | 650 | 61750 | 47.5354 | 47.5354 | 60.2516 | 0NH3_1.05 54.318, 20NH3_1.1 48.975, 40NH3_0.9 39.313 |
| s5_adapted_glu_poff_seed0 | original_last | 980 | 93100 | 47.8706 | 47.8706 | 59.6894 | 0NH3_1.05 54.196, 20NH3_1.1 50.056, 40NH3_0.9 39.360 |
| s5_adapted_glu_poff_seed0 | min_seen | 930 | 88350 | 59.6835 | 48.2920 | 59.6835 | 0NH3_1.05 54.394, 20NH3_1.1 50.870, 40NH3_0.9 39.612 |
| s5_adapted_glu_poff_seed0 | causal_ma5_first_min | 870 | 82650 | 47.9909 | 48.0424 | 59.7195 | 0NH3_1.05 54.103, 20NH3_1.1 50.574, 40NH3_0.9 39.451 |
| s5_adapted_glu_poff_seed0 | loo_holdout_0NH3_1.05.h5 | 510 | 48450 | 43.9735 | 47.6726 | 61.1410 | 0NH3_1.05 55.071, 20NH3_1.1 48.881, 40NH3_0.9 39.066 |
| s5_adapted_glu_poff_seed0 | loo_holdout_20NH3_1.1.h5 | 850 | 80750 | 46.5208 | 47.9820 | 59.7262 | 0NH3_1.05 53.741, 20NH3_1.1 50.905, 40NH3_0.9 39.301 |
| s5_adapted_glu_poff_seed0 | loo_holdout_40NH3_0.9.h5 | 470 | 44650 | 50.8783 | 47.5875 | 61.1647 | 0NH3_1.05 52.462, 20NH3_1.1 49.295, 40NH3_0.9 41.006 |

Il criterio di ogni riga è `score_name` nel CSV. `original_best_val` e `original_last` restano le selezioni del training.
L'ultima eval di ogni run manca da `eval_history.json` ed è stata unita dal CSV `evals/epoch_XXXX/val_eval_per_frame.csv.gz` e da `progress.json`. Il CSV `train_fit` di quell'epoca non c'è: il train all'arresto non è stato inventato.
Leave-one-condition-out sceglie l'epoca che minimizza la media delle altre due condizioni VAL e riporta la condizione esclusa a quell'epoca. La curva di tutte e tre le condizioni era già visibile al training.
