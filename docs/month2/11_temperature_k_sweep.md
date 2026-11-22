# 11 — Sweep delle correzioni a K temperature (C149)

**Stato:** sweep letto dopo i gate. Nessuna selezione di stimatore. I numeri sotto sono quelli del pacchetto `results/month2/temperature_k_sweep/`.

## 1. Contratto e perimetro

Manifest congelato `configs/observations/month2_temperature_anchors_nested_v1.json`, SHA256 del file `a9c997c96cb44489dd1fc0d4484304b44d8beddbf00b73c25baf4e1922a19e54`, contenuto `83fd3b5f2a07e5febc787c384285e265cb4764884fcbc6fb90d22218932f6ae6`. Tre permutazioni annidate 1 ⊂ 4 ⊂ 9 ⊂ 16 ⊂ 32, una per replica di maschera di velocità, siti fissi su tutti i frame. La griglia storica T9 (righe e colonne 32, 64, 95) è un ponte a parte, `layout_kind=t9_bridge`, e non è uno dei tre layout.

Velocità al 5% persistita, seen220 e VAL123, repliche 0/1/2, checkpoint best e last dei dodici run S7. Nessun best_seen, nessun TEST, nessun training. Il protocollo YAML comune non è parsabile (testo dopo la sequenza `metrics`); i siti usati sono quelli del JSON, non una griglia alternativa.

## 2. Correzioni

Su una sola predizione: free; AO_K con tau = 1, `sum(residui)/(K+1)`; media dei residui senza shrink; mediana ordinaria (K pari: media dei due valori centrali); oracle, media del residuo sull’intero campo, solo diagnostico. A K = 9 le tre correzioni con ancore passano dalle funzioni già testate di median9. A K = 0, AO, media e mediana restituiscono il campo free; l’oracle no. Nessun clipping.

Il T-block di GappyPOD non è una traslazione. Usa gli oggetti C149 già fittati (artefatto `a18f1bf2…`) e la coppia storica p_off (0.001, 0.5), p_on (0.01, 0.5). Il w_T = 5 dell’artefatto non entra nel T-block.

## 3. Metriche e gate

Campo completo, dominio comune (griglia meno i 32 siti, uguale a ogni K), off-anchor(K), errore sulle ancore. Regioni con conteggio: un insieme vuoto resta NaN e viene dichiarato; un RMSE globale non finito ferma il run. Il gate free confronta `s7_rmse_K` con il CSV S7 della stessa chiave (run, checkpoint o metodo, ruolo, replica, sim_id, frame), file e hash espliciti, tolleranza macro 0.05 K. Gli scarti per-frame sono scritti anche quando il macro passa. `COMPLETE.json` solo dopo gate e rilettura del CSV.

## 4. Esecuzione e gate

Checkout dei forward: `/home/bontempi/projects/official_thesis_k_sweep_v2` a `54a3b4d3a177b731ed7eef45030c3a49b0986c16`. Smoke GPU `30032664` e array `30032665` (dodici task, una L40S, `%1`) sono `COMPLETED` con exit 0. Baseline CPU `30032411` e `30032412` pure. I dodici run e il baseline hanno `COMPLETE.json`.

Il primo aggregato, `30032666`, è `FAILED`: cercava `per_frame/` nella radice, mentre i 144 CSV neurali stanno in `<run>/per_frame/`. I campi non sono stati ricalcolati. La lettura `30081928` (`30593985337f6367f5e974807cb54865b00d678f`) ha aperto quei CSV e i 36 CSV baseline. `30081718` era una lettura intermedia, fallita prima di scrivere tabelle perché la chiave di identità non distingueva p_on da p_off.

Gate free contro i CSV S7, chiavi `sim_id` e `frame_index`, path e SHA256 presenti: 144 gate neurali e 24 gate baseline, tutti `ok`. Macro e scarto per-frame massimo 0 K. `n_frames` 220 su seen e 123 su VAL. `mask_hash_mismatch` totale 0. Il manifest usato dai forward ha lo stesso SHA256 del file presente nei checkout assisted `official_thesis_t`, `official_thesis_t_run`, `official_thesis_t_chain` e `official_thesis_t_eval`.

`checks.json`: a K=0, AO, media e mediana hanno lo stesso RMSE pieno e lo stesso centered RMSE del free (differenza 0). A K=1 media e mediana coincidono. Su tutto il primario, lo scarto di centered RMSE fra free e AO/media/mediana resta sotto 1e-13 K. A K=0 l’off-anchor del free coincide col campo intero. Le ancore a K=0 sono NaN su tutte le 852 righe primarie (`anchor_n` 0). Per K>0 l’RMSE di ancora è finito. Il dominio comune primario ha 16352 pixel; il ponte T9 ha `common_n` 0 e solo K=9. Nessun RMSE pieno è NaN. Hot, peak, jet e nonhot sono definiti su ogni riga di questo pacchetto (`n_empty` 0).

## 5. Risultati

Figura: `results/month2/temperature_k_sweep/figures/val_common_rmse_vs_K.png`. VAL, checkpoint best o fitted, dominio comune, media delle tre repliche di velocità. Linea continua p_off, tratteggiata p_on, un tratto per seed. Il free è piatto in K. A K=1 media e mediana salgono su diversi seed; da K=4 restano vicine e, sulla media di replica, sotto il free. La media di replica a K=1 nasconde repliche singole molto più alte: GlobalTrainMean, VAL, replica 2, media e mediana hanno RMSE pieno 253.5 K contro un free di 139.6 K.

RMSE comune del free, VAL, best o fitted, media delle tre repliche. I baseline hanno un solo seed.

| Metodo | p | seed 0 | seed 1 | seed 2 |
| --- | --- | ---: | ---: | ---: |
| sparse_unet | off | 39.80 | 40.38 | 41.54 |
| sparse_unet | on | 38.69 | 42.21 | 38.20 |
| adapted_glu | off | 44.49 | 45.20 | 43.75 |
| adapted_glu | on | 43.14 | 41.49 | 43.52 |
| GappyPOD_free | off | 93.79 | — | — |
| GappyPOD_free | on | 70.55 | — | — |
| ParamConditionedMean | on | 87.73 | — | — |
| GlobalTrainMean | off | 139.61 | — | — |

Il free non cambia con K (scarto massimo dentro ogni identità: 0). GlobalTrainMean è solo p_off e ParamConditionedMean solo p_on, come negli oggetti fittati.

A K=32, VAL, best o fitted, la mediana sul dominio comune sta sotto il free nelle 48 combinazioni metodo × p × seed × replica. Il delta va da −23.93 K (GlobalTrainMean, replica 0) a −0.65 K. Seen, stessa mediana, stessi checkpoint e stesso K: 48/48 sotto il free, delta da −19.22 K a −0.002 K. La tabella sotto include anche il last e ha lo stesso segno a K=32 per la mediana su VAL (0 peggioramenti su 84). Non è un ordinamento fra stimatori: a K=1, VAL, best o fitted, la media ha RMSE pieno più alto del free in 16 righe su 48 (6 U-Net, 6 GLU, 2 GappyPOD, 1 ParamMean, 1 GlobalMean). Il massimo è +114.0 K, GlobalTrainMean replica 2.

Conteggi di righe primarie AO/media/mediana con RMSE pieno maggiore del free (`versus_free.csv`):

| Ruolo | K | AO | media | mediana |
| --- | ---: | ---: | ---: | ---: |
| seen | 1 | 61 | 73 | 73 |
| seen | 4 | 64 | 71 | 58 |
| seen | 9 | 60 | 65 | 32 |
| seen | 16 | 49 | 57 | 18 |
| seen | 32 | 36 | 40 | 7 |
| VAL | 1 | 28 | 28 | 28 |
| VAL | 4 | 26 | 26 | 0 |
| VAL | 9 | 8 | 15 | 0 |
| VAL | 16 | 18 | 19 | 0 |
| VAL | 32 | 2 | 2 | 0 |

A K=0 il conteggio è 0. Ogni cella ha 84 righe: dodici run × due checkpoint × tre repliche, più i baseline fittati (GlobalTrainMean, ParamConditionedMean, GappyPOD sulle due p × tre repliche). Le 3024 righe AO/media/mediana primarie ne contengono 964 con RMSE pieno più alto del free.

Bias pieno, VAL, best o fitted: il free va da −61.4 K a +4.3 K; la mediana a K=32 va da −13.1 K a +3.1 K. L’RMSE sulle ancore, stessa mediana e stesso K, va da 30.1 K a 113.6 K: l’offset scalare non azzera i siti.

Il last neurale, free, VAL, dominio comune, sta sopra il best di 0.16–3.97 K sulle 36 righe replica × run. È il checkpoint secondario, non una scelta.

T-block, VAL, fitted, delta del dominio comune rispetto al GappyPOD free della stessa p. A K=0 il delta è 0 sulle sei righe (due p × tre repliche). A K=1 il p_off va da −21.3 K a +2.6 K e il p_on da −0.11 K a +1.8 K. A K=32 il p_off va da −31.6 K a −26.8 K e il p_on da −8.7 K a −2.8 K.

L’oracle, solo diagnostico, su VAL best/fitted ha delta pieno fra −24.9 K e −0.86 K. Il ponte T9 resta fuori da queste curve.

`per_condition.csv` tiene le 20 condizioni seen e le 3 VAL. `across_seeds.csv` tiene il minimo e il massimo fra i seed dentro una replica. Dettaglio e figura: `results/month2/temperature_k_sweep/INDEX.md`.
