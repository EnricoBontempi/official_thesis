# Studio con temperatura (Step 2.7) — contratto comune e coordinamento Sonnet/Cursor

**Pubblicato il 21 novembre 2026, PRIMA della preparazione lunga.** Branch `temperature/assisted-c149` (derivato da `s7/c149-free-campaign` @ `2879c1b5`). Il contratto è congelato: Cursor e il percorso assisted usano gli stessi file, byte per byte.

## File del contratto
- `configs/experiments/month2_temperature_study_protocol.yaml` — protocollo comune.
- `configs/observations/month2_temperature_anchors_nested_v1.json` — manifest dei tre layout T (indici, coordinate, algoritmo, hash); generato da `scripts/50_t_write_anchor_manifest.py` con geometria C149 (sha nel manifest). Ordine/siti non sono mai stati scelti con T, errori o risultati.

## Cosa fa Cursor (solo questo)
Sweep delle correzioni (free / media_K / mediana_K) sui 12 checkpoint free S7 congelati e sui baseline C149, K={0,1,4,9,16,32}, stessi layout e dominio comune del manifest, griglia T9 storica solo come ponte separato. Non modifica checkpoint o risultati free, non tocca il manifest, non aspetta i training assisted.

## Cosa fa Sonnet
Implementazione T-input (U-Net e GLU), test, smoke, profiling, checkpoint/resume e 12 training assisted (famiglia × p × seed 0/1/2), un modello per (famiglia, p, seed) a K variabile, valutato su tutto lo sweep.

## Piano di assegnazione GPU (totale massimo 4 normal + 4 preemptable, inclusi smoke, sweep e altri job dell'utente)
| Chi | normal Advanced (L40S) | preemptable | Note |
|---|---:|---:|---|
| Cursor (inferenza sui checkpoint free, se serve GPU) | ≤ 1 | 0 | i job CPU (`mit_normal`, `mit_amf_advanced_cpu`) non contano |
| Sonnet (smoke, profiling, training assisted) | ≤ 3 | ≤ 4 | array separati non devono superare insieme i limiti |
Regola: ogni run assisted ha **una sola submission attiva** (nessun duplicato normal/preemptable); l'assegnazione run→pool sarà scritta qui al momento della submission. Preemptable: `mit_preemptable` con account `mit_general` (QoS normal; `mit_amf_advanced_gpu` su preemptable è rifiutato, non ritentato); solo con checkpoint/resume verificati. Prima di sottomettere, controllare `squeue -u bontempi` per i GPU già in uso.

## Assegnazione run → pool (SUBMISSION DEL 22 NOVEMBRE 2026, 00:09 EST; una sola submission attiva per run)
| Pool | Job | Run (array index) | Concorrenza | Risorse |
|---|---|---|---:|---|
| normal Advanced | **30034027** | TA-unet-on-s0/s1/s2 (0,1,2), TA-unet-off-s0/s1/s2 (3,4,5) | 3 | `mit_normal_gpu`, account/QoS `mit_amf_advanced_gpu`, L40S×1, 8 CPU, 64G, 6 h, cap interno 5.5 h |
| preemptable | **30034036** | TA-glu-on-s0/s1/s2 (6,7,8), TA-glu-off-s0/s1/s2 (9,10,11) | 4 | `mit_preemptable`, account `mit_general`, QoS `normal`, L40S×1, 8 CPU, 64G, 2 giorni, cap interno 40 h |
| CPU (watchdog) | 30034038 | — | — | `mit_normal`, `mit_amf_advanced_cpu`, 1 CPU: rilascia task tenuti per MaxBatchRequeue |
GPU di Sonnet: ≤ 3 normal + ≤ 4 preemptable (rispettato dalle concorrenze %3/%4). Cursor ≤ 1 normal. Non sottomettere altri array assisted: gli indici 0–11 hanno già una submission attiva (`t_submit.sh` li rifiuta).

## Esecuzione
- **SHA eseguito (codice e config):** `e5136b69019b769fd5cd1107c78ce791191f8812` (branch `temperature/assisted-c149`); checkout congelato `/home/bontempi/projects/official_thesis_t_run`; matrice `configs/experiments/month2_t_assisted_matrix.json`; output nuovi `/home/bontempi/official_thesis_t_assisted_runs/<TA-run>/` (log `slurm/`, ledger `_ledger/`).
- **Contenuto implementato:** U-Net 9 canali (T normalizzata + M_T, p per nome); GLU con token T tipizzati e vicinato top-k esatto su reticolo (tie-break (distanza, sito) indipendente da forma del batch; GLU free intatto); TRAIN K uniforme per optimizer batch, funzione di (namespace, seed 20261121, update), layout T TRAIN casuali per frame/epoca in namespace separato; selezione = media dei sei K della RMSE macro VAL 5% sul dominio comune (criterio NUOVO rispetto a S7), best unico, last e best_seen separati; diagnostici TRAIN/seen a K={0,9,32}.
- **Test/smoke (PASS):** pytest su GPU (8 test: pesi condivisi bit-identici al free, K=0 ≈ free, T usate, p_off, top-k e predizioni invarianti a forma batch e compagni); smoke per famiglia: kill duro + resume da last.pt, requeue no-op, continuazione per cap di wall-time, confronto continuo/resumed (stato, stream, curve), controlli reali: isolamento da T non osservate e velocità nascoste (bit-identico), posizioni/maschere/token = manifest, K=0, p_off, overfit TRAIN, controllo positivo (a 32 siti osservati l'errore alle ancore scende da 0.54→0.33 U-Net, 0.41→0.29 GLU in unità normalizzate), eval sui sei K letta da disco.
- **Profilo misurato (L40S):** U-Net epoca ≈ 2.3 s + valutazione ≈ 24 s ogni 10 epoche (→ ≈ 1.3 h per 1000 epoche, picco 0.6 GB); GLU epoca ≈ 7.5 s + valutazione ≈ 45 s ogni 10 epoche (→ ≈ 3.3 h per 1000 epoche, picco 4.8 GB). Il costo di valutazione era dominato dalla rilettura del payload per frame (ora decompresso una volta, siti identici verificati). Coda non misurabile in anticipo: nessun tempo di attesa ancora registrato.
- **Continuazione:** per i cap di wall-time il runner registra `stopped_wall_time` e fa `scontrol requeue` (ripresa solo da `last.pt` completo, ledger `continuation_requeue`). Sul pool preemptable una preemption riavvia lo stesso job (`SLURM_RESTART_COUNT>0`): il planner riprende da `last.pt` all'ultimo confine d'epoca, mai da best/partial, e un riavvio senza checkpoint è registrato come `fresh_restart_no_checkpoint` (non come resume). Slurm mette in hold un job requeued più di `MaxBatchRequeue`=5 volte: il watchdog (job CPU 30034038) lo rilascia con `scontrol release` (non risottomette, non rilancia errori). Errori veri (`stopped_error`, checkpoint corrotto) sono rifiutati dal planner.
- **Comandi manuali:**
```
ssh engaging
squeue -u bontempi -r -o "%i %j %T %M %r %N"
sacct -D -X -j 30034027,30034036 -o JobID%18,State,ExitCode,Elapsed,NodeList
R=/home/bontempi/official_thesis_t_assisted_runs
cat $R/_ledger/TA-glu-on-s0.jsonl; tail -f $R/slurm/slurm-30034036_6.out
scontrol release <jobid>        # solo se in hold per MaxBatchRequeue
```

## Stato
Contratto e training assisted pubblicati/sottomessi; nessun training completato al momento della scrittura. Aggiornamenti successivi in questo file.
