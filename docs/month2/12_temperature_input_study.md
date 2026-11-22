# Studio con temperatura in ingresso (Step 2.7-B): 12 training assisted C149 (22 novembre 2026)

**Stato:** training, valutazioni e pacchetto verificati nel perimetro dichiarato. Metriche = valutazione finale, non del loop. Descrittivo: nessun vincitore, nessun seed o checkpoint scelto per K. Indice: `results/month2/temperature_assisted_c149/package/INDEX.md`. Lo sweep delle correzioni (Cursor) è un lavoro separato e non è qui interpretato.

## 1. Matrice e provenienza
U-Net e GLU adattato × p_on/p_off × model/shuffle seed 0/1/2 = 12 training da zero, un modello per configurazione allenato a K variabile e valutato su K = 0/1/4/9/16/32; C149 (2980 frame), normalizzatore fit-only 760, velocità 5%, T ideali. Contratto comune `10ff6e5`; training SHA `e5136b6` (U-Net: job 30034027, normal Advanced; GLU: job 30034036, preemptable); catena di valutazione SHA `b7e3b1f` (job 30037805 → array 30060516 → aggregazione 30060517); matrice di valutazione congelata sola lettura, SHA256 `1c91ce25…` (`matrix_frozen.json`). Layout T: tre layout casuali fissi del manifest, uno per replica di maschera velocità (variabilità congiunta, non attribuibile ai soli sensori T). Domini: campo intero; dominio comune (griglia senza i 32 siti del layout, uguale per ogni K); off-anchor(K); ancore. Criterio di selezione NUOVO rispetto a S7: media dei sei K della RMSE macro VAL 5% sul dominio comune, un solo best per run; last e best_seen_diagnostic separati. Il free S7 resta riferimento immutato; assisted(K=0) è un altro modello.

## 2. Esito dei training
- **Conclusi per early-stop: 9/12** (6 U-Net a 640–760 epoche, best a epoca 70–250; GLU on-s0 a 950, on-s2 a 770, off-s0 a 930).
- **Fine tranche (1000 epoche) senza plateau accertato: 3/12** — glu-on-s1 (best 43.23 @820, contatore 12/15), glu-off-s1 (41.555 @750, 7/15), glu-off-s2 (41.473 @730, 9/15). LR al minimo 1e-5 dalle epoche 890/940/920; ultime 20 eval piatte (glu-on-s1 43.6–44.0; off-s1 41.7–42.6; off-s2 41.6–42.6), nessun nuovo minimo da 18/25/27 eval. **Proposta (non eseguita):** come in S7, continuare dallo stato completo fino a max 1100 epoche con regola early-stop invariata (poche decine di epoche, ≈ 5–30 min di calcolo l'una), poi ripetere solo la valutazione dei tre run (≈ 6 min l'una) e riaggregare; il best cambierebbe solo con un nuovo minimo stretto. Richiede autorizzazione esplicita perché invalida la matrice congelata per quei tre run.
- Falliti/held/bloccati: 0. Tutti i task Slurm COMPLETED 0:0.
- **Preemption (pool preemptable):** tre GLU preempted. glu-off-s0 1 volta (epoca 109), glu-off-s1 2 volte (55, 59), glu-off-s2 5 volte (epoche 9, 10, 269, 270, 272: nel limite MaxBatchRequeue=5). Ledger: SIGTERM, `attempt_end rc=143`, poi `attempt_start action=resume` da `last.pt` completo; nessun `fresh_restart_no_checkpoint`, nessun `refused`, nessun `partial_interrupt.pt`; storia delle eval contigua (nessun buco/duplicato); le righe duplicate dello stream (1/2/5) sono epoche interrotte rifatte con hash degli indici identico. Nessun restart da zero nascosto.
- Costi (sacct, calcolo / coda): U-Net 0.6–0.8 h di calcolo (coda 0.01–0.8 h); GLU 2.4–3.2 h (coda 0.14 h per i primi, 1.7–3.3 h per i preempted). Valutazione finale 2.3–6 min per run.

## 3. Valutazioni e gate
12/12 `COMPLETE` (27 CSV per-frame per run: 3 checkpoint × 3 ruoli × 3 repliche, sei K per file), 714744 righe. Gate di parità col loop (RMSE macro di dominio comune entro 0.05 K per run/checkpoint/ruolo/replica e, su VAL, per ogni K): **69/69 per ogni run, scarto massimo 0.0 K (non arrotondato), per-frame 0.0 K su 660 + 2214 + 2280 frame confrontati** (stesso percorso numerico e stessi batch del loop). Ricalcolo indipendente dai CSV grezzi di tre macro: scarto ≤ 1.4e-14. Controlli: chiavi uniche e complete, `mask_hash` per riga = payload verificato, hash dei checkpoint e della matrice, forward condivisi.

## 4. Risultati: RMSE macro di dominio comune (K), media dei 3 seed [min–max]
**VAL123, best**
| config | K=0 | K=1 | K=4 | K=9 | K=16 | K=32 |
|---|---|---|---|---|---|---|
| U-Net p_on | 37.7 [35.4–39.7] | 37.7 [35.4–39.7] | 37.7 [35.4–39.7] | 37.6 [35.4–39.7] | 37.6 [35.4–39.7] | 37.6 [35.3–39.6] |
| U-Net p_off | 39.7 [38.7–40.4] | 39.7 | 39.7 | 39.7 | 39.7 [38.7–40.3] | 39.7 [38.6–40.3] |
| GLU p_on | 43.9 [41.5–47.0] | 43.9 [41.5–47.1] | 43.9 [41.5–47.0] | 43.9 [41.5–47.0] | 43.9 [41.5–46.9] | 43.8 [41.6–46.5] |
| GLU p_off | 44.4 [42.9–46.0] | 43.2 [42.0–44.8] | 41.9 [40.9–43.9] | 41.9 [40.6–43.9] | 41.8 [40.4–43.7] | 41.4 [40.0–43.2] |

**VAL123, last**: U-Net p_on 39.8 → 39.8; p_off 43.7 → 43.6; GLU p_on 44.9 → 44.8; GLU p_off 44.5 → 41.8 (K=0 → 32). **seen220, best**: U-Net p_on 43.0 [40.4–48.0] piatta; p_off 44.6 piatta; GLU p_on 50.6 → 49.9; GLU p_off 53.3 → 51.0. **seen220, last**: U-Net 41.5 / 43.6 piatte; GLU p_on 48.2 → 47.7; p_off 52.7 → 50.6. **reference760 (best)**: U-Net 28.1 / 29.4 piatte; GLU p_on 42.0 → 41.8; p_off 42.0 → 40.8.

**Individuali, VAL best (K = 0 / 1 / 4 / 9 / 16 / 32)**
| run | ep. | K=0 | K=1 | K=4 | K=9 | K=16 | K=32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TA-unet-on-s0 | 70 | 37.91 | 37.91 | 37.89 | 37.86 | 37.83 | 37.73 |
| TA-unet-on-s1 | 250 | 35.38 | 35.38 | 35.38 | 35.37 | 35.36 | 35.35 |
| TA-unet-on-s2 | 200 | 39.69 | 39.70 | 39.68 | 39.67 | 39.65 | 39.59 |
| TA-unet-off-s0 | 220 | 40.14 | 40.14 | 40.15 | 40.15 | 40.16 | 40.16 |
| TA-unet-off-s1 | 160 | 38.69 | 38.68 | 38.68 | 38.68 | 38.69 | 38.63 |
| TA-unet-off-s2 | 130 | 40.41 | 40.41 | 40.41 | 40.39 | 40.35 | 40.35 |
| TA-glu-on-s0 | 500 | 41.53 | 41.54 | 41.55 | 41.54 | 41.54 | 41.55 |
| TA-glu-on-s1 | 820 | 43.16 | 43.19 | 43.22 | 43.23 | 43.29 | 43.29 |
| TA-glu-on-s2 | 320 | 47.04 | 47.05 | 47.04 | 47.02 | 46.92 | 46.49 |
| TA-glu-off-s0 | 720 | 46.01 | 44.84 | 43.88 | 43.91 | 43.71 | 43.16 |
| TA-glu-off-s1 | 750 | 42.95 | 41.97 | 40.92 | 41.18 | 41.18 | 41.14 |
| TA-glu-off-s2 | 730 | 44.15 | 42.76 | 40.92 | 40.60 | 40.40 | 40.00 |

**Guadagno rispetto allo stesso run a K=0 (VAL, best, dominio comune; media [min–max], K=32):** U-Net p_on −0.10 [−0.18…−0.03]; U-Net p_off −0.03 [−0.06…+0.02]; GLU p_on −0.13 [−0.54…+0.12]; **GLU p_off −2.94 [−4.15…−1.81]** (già −1.18 a K=1 e −2.46 a K=4, poi quasi piatto). **Livello e struttura (VAL, best):** il guadagno del GLU p_off è soprattutto di livello: bias firmato da −4.6 K (K=0) a +2.0 K (K=32) con centered RMSE 40.1 → 39.2 K; per gli altri modelli bias e centered non cambiano in modo apprezzabile. **Ancore (VAL, best):** RMSE ai K siti osservati 20–60 K, simile all'errore del campo (non confrontabile con K=0, dove non esistono ancore): i modelli non riproducono i valori osservati. **Off-anchor(K)** coincide con il dominio comune entro 0.1 K.

## 5. Riferimento free S7 (modelli diversi, nessuna identità promessa)
RMSE campo intero VAL 5% best: assisted a K=0 vs free S7 per i 12 run in `tables/free_S7_reference_vs_assisted_K0.csv` (es. U-Net on-s1 35.40 vs 42.24; GLU on-s2 47.04 vs 43.54): differiscono nei due sensi, come atteso per training diversi.

## 6. Lettura (descrittiva, senza aggettivi universali)
Nella ricetta studiata (perdita full-field invariata, K uniforme 0–32 per batch, T ideali), **l'ingresso T riduce l'errore solo per il GLU p_off** (circa −3 K a K=32 su VAL, in gran parte come correzione del livello); per U-Net (p_on e p_off) e GLU p_on il guadagno è ≲ 0.2 K su VAL, in media e quasi ovunque, anche a K=32. Ipotesi **non verificata**: con 32 siti su 16384 pixel la perdita full-field assegna ai pixel osservati ~0.2% del peso, quindi il modello può trascurare T; un test di sensibilità (variare i valori T osservati) e l'errore agli stessi siti a K=0 non sono stati calcolati. I checkpoint best sono scelti con il criterio a sei K e, per le U-Net, molto presto (epoca 70–250): non indicano per sé quanto il modello avrebbe appreso a usare T più tardi.

## 7. Limiti e risultati sfavorevoli
VAL123 usata nella selezione; tre coppie layout-maschera = variabilità congiunta; solo T ideali, nessun rumore; i tre GLU a 1000 epoche non hanno plateau accertato (§2); U-Net e GLU p_on non mostrano beneficio; le ancore non sono riprodotte; nessun TEST.

## 8. Cosa resta
(a) Decisione su continuazione dei tre GLU a fine tranche (§2). (b) Diagnostici di sensibilità a T e dell'errore agli stessi siti a K=0 (inferenza, non eseguiti). (c) Confronto con i risultati dello sweep delle correzioni (Cursor), ancora da fare e non qui. (d) Nessuna nuova campagna autorizzata in questo incarico.
