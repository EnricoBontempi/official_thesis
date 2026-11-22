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

## Catena di valutazione finale automatica (a PC spento) — sottomessa il 22 novembre, ~01:05 EST
Nessuna dipendenza dal monitor locale: tutto è in coda su Engaging.
| Passo | Job | Cosa fa |
|---|---|---|
| 1. verifica + freeze + sottomissioni | **30037805** (CPU, `afterany:30034027:30034036`) | parte quando i due array di training sono finiti, qualunque esito. Controlla: tutti i task Slurm COMPLETED 0:0; per ognuno dei 12 run `summary.json` terminale (`completed`/`stopped_early_plateau`); nessun `refused` né `fresh_restart_no_checkpoint` nel ledger. Se qualcosa non va scrive `/home/bontempi/official_thesis_t_assisted_eval/INCOMPLETE.json` con le cause e SI FERMA (nessun recupero cieco, nessun rilancio di training). Se va tutto bene: congela la matrice (`matrix_frozen.json`, sola lettura: epoche, SHA file e pesi dei tre checkpoint, forward condivisi, riferimenti del loop) e sottomette 2 e 3 |
| 2. valutazioni | array **dipendente dal passo 1** (id in `chain_submitted.json`), 12 task, `mit_normal_gpu` Advanced, concorrenza `min(3, 4 − GPU normal in uso)` | `scripts/55_t_eval.py` per run: best/last/best_seen distinti, K=0/1/4/9/16/32, reference760/seen220/VAL123, repliche 0/1/2, full-field + dominio comune + off-anchor + ancore; gate di parità col loop (macro RMSE di dominio comune entro 0.05 K per run/checkpoint/ruolo/replica e per K su VAL; riferimento mancante = fallimento); readback; `COMPLETE.json` solo dopo gate e readback, altrimenti `GATE_FAILED.json` |
| 3. aggregazione | job CPU `afterok` sull'array (id in `chain_submitted.json`) | `scripts/56_t_aggregate.py`: rifiuta matrice incompleta o incoerente (hash matrice, chiavi, `mask_hash` vs payload verificato, forward condivisi); tabelle, figure, costi (calcolo/coda), `INDEX.md` |
Output: `/home/bontempi/official_thesis_t_assisted_eval/` (`runs/<TA-run>/`, `package/`, `slurm/`, `matrix_frozen.json`, `chain_submitted.json` o `INCOMPLETE.json`). SHA della catena: `b7e3b1f52bb5beaa3c6494504e2cad0da444d87c` (checkout `/home/bontempi/projects/official_thesis_t_chain`).
**Limiti residui:** (i) se un training fallisce o resta in hold oltre il watchdog, la catena resta in attesa o produce `INCOMPLETE.json`: serve intervento umano; (ii) una preemption ripetuta oltre MaxBatchRequeue richiede il rilascio da parte del watchdog (job 30034038, termina quando non restano job `t_assist`); (iii) la sottomissione dell'array di valutazione dipende dalla disponibilità di GPU normal (condivise con Cursor); (iv) il pacchetto resta sul cluster: va copiato nel repo e committato a mano; (v) se la valutazione fallisce il gate, nessun risultato è validato (diagnostica preservata).
**Verifica dell'evaluator:** smoke su checkpoint in corso (U-Net e GLU, CPU, sottoinsieme di frame): output, readback, forward condivisi, aggregazione in modalità smoke OK; il gate di parità completo è stato provato su un run concluso (vedi sotto).

## Stato
Contratto e training assisted pubblicati/sottomessi; nessun training completato al momento della scrittura. Aggiornamenti successivi in questo file.

---
## AGGIORNAMENTO 22 novembre 2026, ~13:30 EST — esito reale
Training 12/12 COMPLETED (9 early-stop, 3 a 1000 epoche senza plateau: glu-on-s1, glu-off-s1, glu-off-s2), preemption GLU gestite con resume (nessun restart nascosto). Catena: verifica 30037805 (06:41) → array 30060516 (12/12 COMPLETE, gate 69/69, scarto max 0.0 K) → aggregazione 30060517. Pacchetto in `results/month2/temperature_assisted_c149/package/` (INDEX), report `docs/month2/12_temperature_input_study.md`. Finding principale: guadagno da T apprezzabile solo per GLU p_off (≈ −3 K a K=32); U-Net e GLU p_on ≲ 0.2 K. Residui: continuazione dei tre run a fine tranche (da autorizzare), sensibilità a T, confronto con lo sweep di Cursor.

---
## CONTINUAZIONE DEI TRE RUN E CATENA v2 — sottomessa il 22 novembre 2026 (EST)
Autorizzata la continuazione di **soli** TA-glu-on-s1, TA-glu-off-s1, TA-glu-off-s2 da `last.pt` (epoca 1000, update 95000) fino a un massimo di 1100 epoche, stesse regole di training (stesso early-stop, LR, schedule, K di training), nessun restart, gli altri nove run non toccati. Nessun nuovo modello, loss, tuning, TEST.

- Via reale `plan-extend` del launcher (non si è modificato il summary): tranche 1 preservata in `<run>/tranche1_epoch1000/` con `MANIFEST.json` (sha256 di checkpoint, summary, history) e provenance; ledger `extension_start` + `attempt_start action=extend`. Validata prima su una copia (job 30082073, 10 epoche, ledger e history identici sulle prime 100 voci).
- **Array di continuazione 30082569** (indici 7, 10, 11; `mit_normal_gpu` / `mit_amf_advanced_gpu`; concorrenza 3; 3 h; SHA `a86e39c61731586a3343d9d1489b4d3b22fa0547`, checkout `/home/bontempi/projects/official_thesis_t_ext`).
- **Catena v2 30082570** (CPU, `afterany:30082569`): verifica (task COMPLETED; i tre run con estensione, tranche1, epoche in (1000, 1100]; i nove altri con checkpoint sha identico a v1; nessun restart da zero) → congela `matrix_frozen_v2.json` → array di valutazione v2 (12 task; i nove run invariati e i checkpoint invariati sono riattestati per hash dal v1, solo i checkpoint cambiati dei tre run sono ricalcolati) → aggregazione `afterok`. Qualunque anomalia scrive `INCOMPLETE_v2.json` e si ferma, nessun rilancio cieco.
- Radice v2: `/home/bontempi/official_thesis_t_assisted_eval_v2/` (`matrix_frozen_v2.json`, `runs/`, `package/`, `chain_v2_submitted.json`). La matrice v1 (sha256 `1c91ce257e0ae93e6cb909bc23741f08ba7339ea9cb0b3e83677b3b15dff5042`), `runs/` e il pacchetto pubblicato in `aa8f65f` non sono toccati.
- Il riuso nel valutatore è verificato (`--reuse-matrix`): stesso file_sha256, weights_sha256, epoca, `COMPLETE.json` v1 con gate ok. Test: riuso puro su CPU (job 30082576, 27 CSV, gate 69/69, scarto 0.0 K).
- Se a 1100 epoche l'early-stop non scatta, si riportano curve e contatori e **non** si estende oltre.

## CONSEGNA CONGIUNTA PASSI 2 + 3 (post-processing sul cluster, gated)
Input: pacchetto Cursor `~/official_thesis_k_sweep_package_v1` (forward SHA `54a3b4d…`, aggregazione `3059398…`, per-frame in `~/official_thesis_k_sweep_v1/<run>/per_frame`) e pacchetto assisted v2 (sopra). Script `scripts/57_t_joint_package.py` (SHA di consegna nel commit di questo paragrafo), lanciato da `scripts/engaging/t_joint_launch.sbatch` (`afterany` sulla catena v2) che accoda `t_joint.sbatch` dietro l'aggregazione v2. Controlli bloccanti prima di qualunque tabella: manifest sha256 `a9c997c9…` su entrambi i lati; K; repliche/layout `nested_r<replica>`; frame (sim_id, frame_index) identici per ruolo (220 / 123); hash maschera di velocità identico per frame; dominio comune 16384−32 su entrambi; conteggi off-anchor 16384−K; unità K; la definizione di macro riprodotta a 1e-6 K sulle tabelle pubblicate di entrambi i pacchetti. Se un input manca o un controllo fallisce: `JOINT_BLOCCO.md` + `JOINT_STATUS.json` (complete=false) in `~/official_thesis_t_joint_package/`, nessun rilancio, pacchetto non dichiarato completo. Se tutto passa: `INDEX.md` unico con riferimenti esatti ai due pacchetti, tabelle libero / libero+media / libero+mediana / assistito, nessun vincitore, nessun nuovo training. Prova a secco sui dati v1 (job 30083008): tutti i controlli superati.

**Job della consegna congiunta (22 novembre 2026):** launcher `30083131` (`t_joint_launch`, CPU, `afterany:30082570`), SHA di consegna `50903f3c246f01266691a7f4a525b72aba5f656d`, checkout `/home/bontempi/projects/official_thesis_t_joint`, uscita `/home/bontempi/official_thesis_t_joint_package/` (`joint_submitted.json` con l'ID del job congiunto, poi `INDEX.md` + `tables/` oppure `JOINT_BLOCCO.md` + `JOINT_STATUS.json`). Il launcher è idempotente (rifiuta un secondo invio).

**Aggiornamento consegna congiunta (22 novembre 2026):** Cursor ha pubblicato il passo 2 (branch `sweep/c149-temperature-k`, SHA `be18ad007e66fbf2565ad3d5486c6513fc5c3e4c`, indice `results/month2/temperature_k_sweep/INDEX.md`, report `docs/month2/11_temperature_k_sweep.md`). Le tabelle sul cluster (`by_seed`, `across_seeds`, `per_condition`, `versus_free`, `checks.json`) sono byte-identiche alla versione pubblicata; `INDEX.md` e la figura sono stati rigenerati da Cursor dopo (solo testo/figura). Il post-processing ora richiede questa identità come controllo bloccante (`results/month2/joint_step23/cursor_published_be18ad0_sha256.json`). Il launcher 30083131 (ancora pending, nulla sottomesso) è stato sostituito da **30083178** (`afterany:30082570`), SHA di consegna `682fe29496d9852c82af2a5d6fb59f05eb329006`. Nessun forward ripetuto, branch di Cursor non toccato.

---
## ESITO FINALE — 22 novembre 2026 (verificato dopo il rientro)
- Continuazioni 30082569: task 7/10/11 COMPLETED; plateau a 1030/1080/1060 epoche (`early_stop_no_rel_improvement_0.001_for_15_evals`), best invariato; tranche1 con MANIFEST sha256 riverificato; ledger `extension_start` + `attempt_start action=extend` + `attempt_end rc 0`; nove run altri con checkpoint identici a v1.
- Catena v2: 30082570 (18 s) → eval 30083486 (12/12 COMPLETED; solo i checkpoint cambiati ricalcolati) → aggregazione 30083487; matrice `3663d33820aac446b9621bf35ae49dec26ee4cc6a8379ad5363b0a81e8d922f7`; gate 69/69 per run, scarto massimo 0.0 K. Test misto 30082577 COMPLETED (gate 0.0 K).
- Congiunto: launcher 30083178 → job 30083497 COMPLETED; `JOINT_STATUS.json` ok/complete, 26 controlli.
- Pubblicati: `results/month2/temperature_assisted_c149/{package_v2,matrix_frozen_v2.json,chain_v2_submitted.json,extension/}`, `results/month2/joint_step23/package/` (INDEX per Opus/Neon: `results/month2/joint_step23/package/INDEX.md`). Pacchetto v1 (`package/`, `aa8f65f`) non modificato.
- Limite residuo: l'INDEX congiunto rimanda a percorsi del cluster per i CSV per-frame (non in git); `INDEX.md` e figura di Cursor sul cluster sono stati rigenerati da lui dopo le tabelle, ma le tabelle sono byte-identiche a quelle pubblicate a `be18ad0`.
