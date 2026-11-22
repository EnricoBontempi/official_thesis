# Sweep K — indice

Forward e correzioni: `54a3b4d3a177b731ed7eef45030c3a49b0986c16`, checkout detached `/home/bontempi/projects/official_thesis_k_sweep_v2`.
Lettura delle tabelle: `30593985337f6367f5e974807cb54865b00d678f`, job `30081928`.
Manifest `configs/observations/month2_temperature_anchors_nested_v1.json`, SHA256 del file `a9c997c96cb44489dd1fc0d4484304b44d8beddbf00b73c25baf4e1922a19e54`, contenuto `83fd3b5f2a07e5febc787c384285e265cb4764884fcbc6fb90d22218932f6ae6`. Lo stesso SHA256 del file è nei checkout assisted `official_thesis_t`, `official_thesis_t_run`, `official_thesis_t_chain` e `official_thesis_t_eval`.

Nessuno stimatore è selezionato. I CSV per-frame restano sul cluster e non sono in git.

## Matrice

Dodici run, tutti con `COMPLETE.json`. Gate free: 144 JSON neurali e 24 JSON baseline, tutti `ok`, macro 0 K, scarto per-frame 0 K, `mask_hash` senza scarti, riferimenti con path e SHA256. seen220 e VAL123. Checkpoint best e last. Repliche 0/1/2.

| Task | Run |
| --- | --- |
| 30032665_0 | S7-C149-unet-on-s1 |
| 30032665_1 | S7-C149-unet-off-s1 |
| 30032665_2 | S7-C149-unet-on-s2 |
| 30032665_3 | S7-C149-unet-off-s2 |
| 30032665_4 | S7-C149-glu-on-s0 |
| 30032665_5 | S7-C149-glu-off-s0 |
| 30032665_6 | S7-C149-glu-on-s1 |
| 30032665_7 | S7-C149-glu-off-s1 |
| 30032665_8 | S7-C149-glu-on-s2 |
| 30032665_9 | S7-C149-glu-off-s2 |
| 30032665_10 | O-C149-on-s0 |
| 30032665_11 | O-C149-off-s0 |

Baseline fittati, senza refit: GlobalTrainMean p_off, ParamConditionedMean p_on, GappyPOD_free p_on e p_off. T-block con la coppia storica p_off (λ 0.001, w_T 0.5) e p_on (λ 0.01, w_T 0.5). Artefatto `a18f1bf2f4f5a170a04ebd73fb5d438aaa3c3e390cf44e678377381eecfe3e5a`.

## File

- `by_seed.csv` — una riga per metodo, p, seed, checkpoint, ruolo, replica, layout, K, trattamento
- `versus_free.csv` — differenza rispetto al free, peggioramenti inclusi
- `across_seeds.csv` — minimo e massimo fra i seed, dentro una replica
- `per_condition.csv` — 20 condizioni seen e 3 VAL
- `checks.json` — identità K=0 e coincidenza media/mediana a K=1
- `figures/val_common_rmse_vs_K.png` — VAL, best o fitted, media delle tre repliche sul dominio comune. Ridisegnata dagli stessi `by_seed.csv` con una legenda condivisa; il PNG del job `30081928` aveva la legenda solo sul pannello GappyPOD

## Percorsi cluster

- `/home/bontempi/official_thesis_k_sweep_v1`
- `/home/bontempi/official_thesis_k_sweep_baselines_v1`
- `/home/bontempi/official_thesis_k_sweep_package_v1`
- `/home/bontempi/official_thesis_k_sweep_chiusura_v1/BLOCCO.md` registra il primo aggregato `30032666`, prima della lettura corretta
