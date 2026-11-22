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
(a) Decisione su continuazione dei tre GLU a fine tranche (§2). (b) Diagnostici di sensibilità a T e dell'errore agli stessi siti a K=0 (inferenza, non eseguiti). (c) Confronto con i risultati dello sweep delle correzioni (Cursor): eseguito, vedi §10 [aggiornamento 22 novembre]; (a) risolto, vedi §9. (d) Nessuna nuova campagna autorizzata in questo incarico.

---
## 9. Aggiornamento del 22 novembre 2026 (pacchetto v2): continuazione dei tre GLU
Il testo e le tabelle sopra restano invariati (stato a 1000 epoche, pacchetto `results/month2/temperature_assisted_c149/package/`, SHA `aa8f65f`). Questo paragrafo registra l'esito della continuazione autorizzata.

**Esecuzione.** Solo TA-glu-on-s1, TA-glu-off-s1, TA-glu-off-s2, da `last.pt` (epoca 1000) fino a un massimo di 1100 epoche, stesse regole (stesso early-stop, schedule, K di training), nessun restart; array 30082569 (task 7, 10, 11; tutti COMPLETED, 6:41, 16:38, 12:35; `mit_normal_gpu`). Tranche 1 preservata in `<run>/tranche1_epoch1000/` con `MANIFEST.json` (12 file, sha256 riverificati; `last.pt` della tranche = `last` della matrice v1 per tutti e tre). I nove altri run non sono stati toccati: checkpoint con sha256 identico alla matrice v1.

| Run | Arresto | Epoche | Best VAL (invariato) | best_seen diagnostico | Tentativi |
| --- | --- | --- | --- | --- | --- |
| TA-glu-on-s1 | plateau (`early_stop_no_rel_improvement_0.001_for_15_evals`) | 1030 | 820 (43.23 K, loop) | 980 (invariato) | 2 |
| TA-glu-off-s1 | idem | 1080 | 750 (41.56 K) | 1000 → 1050 | 4 |
| TA-glu-off-s2 | idem | 1060 | 730 (41.47 K) | 890 → 1050 | 7 |

Tutti e tre hanno raggiunto il criterio di plateau prima di 1100: la domanda aperta al §2 («plateau a 1000 epoche?») ha risposta, non è stata necessaria alcuna estensione oltre il limite. La prima tranche era dunque troncata dal tetto delle 1000 epoche, non da un andamento ancora in discesa che continui: dopo l'epoca 1000 il VAL non ha migliorato il minimo (best invariato a 820/750/730).

**Valutazioni v2.** Matrice congelata `matrix_frozen_v2.json` (sha256 `3663d33820aac446b9621bf35ae49dec26ee4cc6a8379ad5363b0a81e8d922f7`; v1 `1c91ce25…`). Catena 30082570 → array di valutazione 30083486 (12 task COMPLETED) → aggregazione 30083487. Solo i checkpoint cambiati sono stati ricalcolati (`last` dei tre run; `best_seen_diagnostic` di glu-off-s1 e glu-off-s2); tutto il resto è stato riattestato per hash dal v1 (`checkpoint_plan` in `provenance.json`). Gate 69/69 per run, scarto massimo 0.0 K, 12/12 `COMPLETE.json`, 714744 righe. Pacchetto: `results/month2/temperature_assisted_c149/package_v2/` (INDEX nella cartella); tabelle v1 e v2 coincidono in 558 delle 648 celle di `global_by_setting.csv`; le 90 che cambiano sono tutte `last` o `best_seen_diagnostic` dei tre run.

**Effetto numerico (RMSE macro di dominio comune, VAL123, K).** `best` invariato, quindi la lettura principale del §4 e §6 non cambia. `last`: glu-off-s1 K=0 42.85 → 42.60, K=32 41.64 → 41.50; glu-off-s2 K=0 44.58 → 44.12, K=32 40.59 → 40.34; glu-on-s1 K=0 43.75 → 43.71, K=32 43.91 → 43.87. Variazioni da 0.04 a 0.46 K, nello stesso ordine della dispersione tra seed; nessuna decisione ne dipende.

Limiti: i tre run hanno più tentativi per preemption (2, 4, 7); ogni ripartenza da `last.pt` di propria appartenenza, flussi duplicati con hash identici (ledger in `results/month2/temperature_assisted_c149/extension/<run>/`). Il test misto di riuso+calcolo (30082577) è COMPLETED con gate 0.0 K; il riuso puro (30082576) idem.

## 10. Confronto congiunto con lo sweep delle correzioni (passo 2, Cursor)
Pacchetto congiunto: `results/month2/joint_step23/package/INDEX.md` (tabelle in `tables/`). Input: sweep K di Cursor, branch `sweep/c149-temperature-k`, SHA `be18ad007e66fbf2565ad3d5486c6513fc5c3e4c` (`results/month2/temperature_k_sweep/INDEX.md`, report `docs/month2/11_temperature_k_sweep.md`), e il pacchetto assisted v2. Job: launcher 30083178, post-processing 30083497 (COMPLETED, 35 s, CPU). 26 controlli, tutti i bloccanti superati: manifest T identico (`a9c997c9…`), K, repliche, layout `nested_r<replica>`, frame (220/123) identici, hash maschere di velocità identico su 1029 frame, dominio comune 16352 px, conteggi off-anchor 16384−K, unità K, definizione di macro riprodotta a ≤2e-14 K sulle tabelle pubblicate di entrambi i lati, tabelle di Cursor byte-identiche a quelle pubblicate a be18ad0. Nessun forward ripetuto.

Letture descrittive (RMSE macro di dominio comune, VAL123, `best`, media su 3 seed e 3 repliche; tutte le tabelle nell'INDEX; nessun vincitore):
- La media dei residui con K = 1 peggiora sempre (da 40–44 a 65–72 K): una sola ancora trasferisce il suo errore a tutto il campo. Con K ≥ 4 la correzione mediana abbassa il free di circa 5 K (U-Net p_on K=32: 39.70 → 34.54; U-Net p_off 40.57 → 35.40), di 2–3 K per GLU (p_off 44.48 → 41.16; p_on 42.72 → 41.24).
- L'assisted rispetto al free: U-Net p_on 39.70 → 37.66 (già a K=0, quindi per il training diverso e non per le ancore: 37.55 a K=32), U-Net p_off 40.57 → 39.71, GLU p_off 44.48 → 41.43 (a K=32; 44.37 a K=0), GLU p_on 42.72 → 43.78 (peggiore).
- A K=32 la mediana post hoc sul free è più bassa dell'assisted in tutte e quattro le combinazioni famiglia×p (U-Net p_on 34.54 vs 37.55, U-Net p_off 35.40 vs 39.71, GLU p_on 41.24 vs 43.78, GLU p_off 41.16 vs 41.43; quest'ultima entro 0.3 K): nelle condizioni misurate T in ingresso non ha battuto una correzione scalare della stessa informazione. Il confronto confonde però modello diverso e uso delle ancore: la parte dovuta alle ancore si legge solo a parità di modello, `gain_vs_K0_by_run.csv`.
- Limiti: criteri di selezione diversi tra i due lati (S7 vs media su sei K), `best` e `last` non sono le stesse epoche; T ideali senza rumore; tre coppie layout-maschera; VAL123 usata nella selezione dell'assisted; nessun TEST.
