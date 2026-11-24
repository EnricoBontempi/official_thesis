# Diario della ricerca

Questo file è il diario del progetto di tesi in `official_thesis` (ricostruzione di campi di temperatura da osservazioni sparse in combustione turbolenta). Il precedente progetto esplorativo HONF resta separato in `thesis-research`.

---

## 2026-11-13 — Invio del proposal e avvio del Mese 1

- **Contesto:** Inviata la proposta di tesi (`docs/proposal.md`) alla Prof.ssa Sili Deng e creata la repository `official_thesis` con la roadmap del Mese 1 (`roadmap_month1.md`, commit `75c5db2`, `7dff7d6`). La prof ha risposto che il piano va bene, dicendosi curiosa di vedere i primi aggiornamenti perché un po' scettica sulla possibilità di ricostruire la temperatura *istantanea* dalla sola velocità.
- **Primi controlli sui dati (`Step 1.1`–`1.2` e pilota `Step 1.3`):**
  - Fatta la ricognizione del dataset `RealPDEBench` (sezione combustione) su Hugging Face (`docs/month1/01_remote_reconnaissance.md`).
  - Scoperta importante: negli split ufficiali di RealPDEBench i set `val` e `test` numerici sono vuoti (`[]`), perché loro usano tutte le 30 simulazioni LES (sottocampionate a $64 \times 64$) per fare pre-training di forecasting temporale prima del fine-tuning sui dati sperimentali $\text{OH}^*$. Per il nostro problema dobbiamo quindi definire noi uno split pulito per condizione operativa a piena risoluzione $128 \times 128$.
  - Scaricati 3 shard pilota (`0NH3_1`, `40NH3_1`, `80NH3_1.05`) per controllare formato e canali (`02_pilot_data_audit_report.md`).

---

## 2026-11-14 — Download completo delle 30 condizioni, split e prime baseline

- **Audit completo delle 30 simulazioni LES (`Step 1.3`, `docs/month1/02_data_audit_report.md`):**
  - Scaricati tutti i 30 shard numerici (~58.6 GB, 2001 frame a $4000\text{ Hz}$ per condizione, miscele $\text{NH}_3 \in \{0, 20, 40, 60, 80\}\%$ e $\phi \in [0.75, 1.30]$).
  - **Chiarita la geometria fisica della fetta 2D ($128 \times 128$):** rispetto all'ipotesi iniziale del pilota, le mappe mostrano chiaramente una sezione diametrale simmetrica rispetto alla riga centrale (`row 64`). Due getti freddi di miscela fresca ($T \sim 300\text{ K}$) entrano dal **bordo destro (`col 127`)** e si aprono a V verso sinistra, racchiudendo la zona centrale calda di ricircolo (IRZ, $T \sim 1900\text{--}2280\text{ K}$). I canali di velocità sono `Ch 11` ($v_i$, radiale/trasversale), `Ch 12` ($v_j$, assiale) e `Ch 13` ($v_k$, swirl fuori dal piano). Il modulo $|\mathbf{u}|$ viene sempre ricalcolato da `Ch 11–13` perché `Ch 14` ha piccoli artefatti negativi di interpolazione.
- **Definizione degli split e del protocollo (`Step 1.4`, `docs/month1/03_validation_and_splits_protocol.md`):**
  - Poiché l'autocorrelazione sul fronte di fiamma dura 50–100 frame (`fig04`), abbiamo diviso le 30 condizioni tenendo fuori intere traiettorie (`configs/splits/splits_interp_extrap.yaml`, `fig06`): **20 `train`**, **3 `val`** (`0NH3_1.05`, `20NH3_1.1`, `40NH3_0.9`), **3 `test_interp`** (`20NH3_1.0`, `40NH3_1.15`, `60NH3_0.95`) e **4 `test_extrap`** ai bordi (`20NH3_0.75`, `0NH3_1.3`, `20NH3_1.3`, `80NH3_1.25`).
  - Statistiche di normalizzazione calcolate solo sulle 20 condizioni di `train`.
- **Prima esecuzione delle baseline (`Step 1.5`, commit `567b76f`) e successiva revisione (`analisi_agents.md`, commit `659815a`):**
  - Lanciate le prime baseline non neurali (`GlobalTrainMean`, `ParamConditionedMean`, `LocalHGB`, `GappyPOD`).
  - Facendo revisionare il codice e il PDF di GLU a Fable (Claude) e Sol (ChatGPT), sono emersi due problemi di regolarizzazione nel codice di `baselines.py` (un `ridge = 1e-2` non scalato che schiacciava sia il contributo delle $K$ termocoppie in `GappyPOD` sia la curvatura quadratica in $\phi$ di `ParamMean`) e sono stati chiariti i dettagli esatti del paper GLU (`arXiv:2603.26023`).

---

## 2026-11-15 — Fix delle baseline, risultati definitivi del Mese 1 e report LaTeX per i supervisori

- **Fix delle baseline e risultati verificati (`commit 4fdbcb1`, `docs/month1/04_feasibility_check_report.md`):**
  - Corretto `ParamConditionedMean` (feature centrate attorno a $\phi=1.0$ scelte in `LOCO-CV` sulle 20 condizioni di train) e `GappyPOD` ($R=32$ modi con regolarizzazione scalata sulle varianze modali, più variante con offset sulle ancore `+AO` e proiezione oracolo `OraclePOD`).
  - **Cosa ci dicono davvero i dati del Mese 1:**
    1. **Livello termico medio vs chimica (`fig07`):** a parità di velocità del fluido, la temperatura cambia di $200\text{--}360\text{ K}$ tra $0\%$ e $80\%\text{ NH}_3$. Però basta conoscere i parametri $(\text{NH}_3\%, \phi)$ perché già un modello semplice (`ParamMean`, 0 sensori) stimi il livello medio entro $\sim 18\text{--}19\text{ K}$ (`|Bias|`) in distribuzione e $\sim 43\text{ K}$ in estrapolazione.
    2. **Il vero collo di bottiglia è stimare la forma istantanea del fronte di fiamma:** proiettando la $T$ vera sui 32 modi POD (`OraclePOD`), l'errore di forma (`Centered-RMSE`) scende a $36\text{--}75\text{ K}$ (la base lineare cattura $\sim 80\%$ dell'energia delle fluttuazioni). Ma quando stimiamo quei 32 modi linearmente dal $5\%$ di velocità (`GappyPOD Cond`), il `Centered-RMSE` resta a $67\text{--}139\text{ K}$ (e $205\text{--}360\text{ K}$ lungo i getti e il fronte di fiamma). Il problema non è la base POD, ma il legame fortemente non lineare tra velocità e temperatura nei vortici.
    3. **Fiamme ricche ($\phi \ge 1.20$) e complementarità globale/locale:** l'errore sulla forma del fronte cresce con $\phi$ ($r = 0.73$) sia in training sia in estrapolazione. Sulle fiamme stabili (`0NH3_1.0`) va meglio `GappyPOD` (globale), mentre su quelle ricche e corrugate (`20NH3_1.3`, `80NH3_1.25`) va meglio `LocalHGB` (locale non lineare), motivando l'uso di un'architettura globale-locale come GLU.
    4. **Termocoppie sparse ($K \in \{1\dots 32\}$):** con $K \le 9$ termocoppie i metodi lineari (`+AO` e `+T-block`) non migliorano la forma del fronte; servono $K = 16\text{--}32$ sensori per ridurre il `Centered-RMSE` da $139\text{ K}$ a $119\text{ K}$.
- **Report LaTeX per i supervisori (`overleaf_month1_report/main.tex` e repo `documents_thesis`, commit `3f7c0e2`):**
  - Scritto e compilato il report completo in LaTeX con tutte le figure (`fig01`–`fig10`), le tabelle verificate sui CSV e la proposta in 4 passi per il Mese 2–3:
    - *Step 0:* U-Net da velocità densa $(v_i, v_j) \rightarrow T$ (1 frame vs finestra di 3 frame) per misurare empiricamente quanta informazione sul fronte di fiamma c'è nella velocità prima di introdurre la sparsità;
    - *Step 1:* Adattamento di GLU alla velocità sparsa ($1\%, 5\%, 20\%$) condizionato a $(\text{NH}_3\%, \phi)$;
    - *Step 2:* Modulo residuo a due stadi per assimilare poche termocoppie ($K > 0$) senza farle schiacciare dai token di velocità;
    - *Step 3 (Mese 3):* Base di correzione a basso rango dipendente dallo snapshot (inclusi modi di spostamento del fronte) per rendere utili anche $K = 4\text{--}9$ termocoppie.

---

## 2026-11-16 — Ricezione del codice di GLU da Jason Chen

- **Aggiornamento:** In attesa del riscontro della Prof.ssa Deng e di Linzheng sul report, Jason Chen ci ha girato il repository con il codice ufficiale di GLU (`https://github.com/cosmos2w/GLU_Reconstruction`).
- **Organizzazione (`commit ebaae9d`):** Clonato il codice originale intatto dentro `external/GLU_Reconstruction/` e documentato nel `README.md` come baseline di riferimento del lab (i nostri adattamenti andranno in `src/combustion_recon/`, e i training verranno lanciati su Engaging).
- **Cosa abbiamo visto ispezionando `external/GLU_Reconstruction/`:**
  - Lo Stage-0 (ricostruzione spaziale) usa solo `DomainAdaptiveEncoder`, `phi_mlp_1` e `SoftDomainAdaptiveReconstructor`, saltando il modulo di forecasting temporale.
  - Nel loro `DomainAdaptiveEncoder` c'è già sia il filtro `channel_to_encode` (pensato proprio per osservare solo un sottoinsieme di canali, come la velocità) sia l'argomento `U` per passare i parametri operativi della condizione (`# unused here` nei loro test a singola condizione).
  - La fusione globale-locale nel decoder usa un gate spaziale adattivo (`SpatiallyAwareCLSFusion`) e la testa di varianza è addestrata con loss Gaussian NLL (`training_procedure_pseudocode.tex`).

---

## 2026-11-16 — Preparazione della roadmap del Mese 2

- **Attività (Codex, America/New_York / UTC−05:00):** letti stato remoto, diario, proposal disponibile, roadmap e report del Mese 1, analisi delle architetture, configurazioni, codice e risultati aggregati. Creati [roadmap_month2.md](../roadmap_month2.md) e [contesto/decisioni](month2/00_context_and_decisions.md); aggiornati README e nota di transizione del Mese 1.
- **Sequenza proposta:** protocollo e pipeline verificabili → baseline comparabili → U-Net da velocità densa → audit/adattamento GLU e confronto sparse → poche ancore T, prima con offset semplice → pilot di trasferimento congelato e revisione. Le architetture restano candidate; ogni step ha output, verifiche e criterio per procedere.
- **Cosa emerge dalla verifica:** il divario GappyPOD/Oracle riguarda soprattutto la stima dalla velocità, senza provare un limite fisico irreducibile. AO corregge solo il livello; T-block a K=32 può migliorare anche la struttura. Servono inoltre statistiche fit-only che escludano i frame seen, input separati dai target e controllo della memoria del loader. Dettagli e numeri controllati sui CSV sono nel documento di contesto.
- **Limite sul proposal:** `documents_thesis/main.tex` contiene il report del 15 novembre, identico a `overleaf_month1_report/main.tex`; anche la sua cronologia parte dal report. Non è stato identificato lì il LaTeX inviato il 13 novembre: lo scopo della roadmap segue `docs/proposal.md`, senza attestare identità con l'allegato inviato. Nessun proposal modificato.
- **Stato (al momento di quella preparazione):** roadmap preparata, allora non ancora autorizzata né eseguita. Nessun training/download in quell’attività.
- **Sincronizzazione:** preservata integralmente la semplificazione concorrente del diario arrivata nel commit `a2cebb1`, successivo alla base inizialmente letta `6fd0f15`.

---

## 2026-11-16 — Step 2.1: protocollo pilota versionato

- **Autorizzazione:** Enrico ha approvato la roadmap e autorizzato l’avvio del lavoro (conversazione odierna). Questa autorizzazione operativa è distinta da una ratifica tecnica dei supervisori, che **non** risulta documentata e non è stata inventata.
- **Output:** creati `configs/splits/month2_frame_protocol.yaml` (`month2_frame_protocol_v1`) e `configs/experiments/month2_pilot.yaml` (`month2_pilot_v1`); aggiornato il registro in `docs/month2/00_context_and_decisions.md` (D1–D9, fold, controlli).
- **Protocollo:** split 20/3/3/4 invariato; `train_fit` 38 frame/condizione (`0…1480` step 40); `seen_train_eval` 11 frame (`1600…2000` step 40), disgiunti; VAL/TEST 41 frame step 50; input primario 2C single-frame @5%; K primario 9 con sweep annidato; selezione su VAL; TEST solo dopo freeze; 3C/finestra/testa residua non attivate.
- **Fold:** `20NH3_0.85.h5` risulta sul bordo dell’inviluppo convesso residuo → non usabile come estrapolazione geometrica; `80NH3_1.2.h5` è fuori. Proposta non locked: `40NH3_0.85.h5` + `80NH3_1.2.h5`.
- **Limiti:** nessuna pipeline/modello/training; stats fit-only Mese 2 non calcolate; Engaging previsto ma GPU/allocazione non verificate; 24 GPU-ore = tetto proposto, non allocazione. Dettagli e hash nel registro.

---

## 2026-11-16 — Step 2.2: pipeline dati Month-2 verificata in locale

- **Implementato:** percorso isolato `Month2ReconDataset` + `ShardStore` (decode Arrow per shard, cache compatta di frame T/vi/vj/vk); ruoli `train_fit`/`seen`/`val` e gate TEST; sample `{model_inputs, targets, metadata}`; collate con padding mask; normalizzatore fit-only fold-capable; evaluator in K.
- **Artefatti:** `data/manifests/train_norm_stats_month2_fit.json` (760 frame TRAIN-fit; Mese 1 `train_norm_stats.json` intatto), `month2_frames.json`, `month2_observations.json`; report `docs/month2/01_pipeline_validation.md`; script `scripts/05_validate_month2_pipeline.py`.
- **Verifiche:** 43 pytest OK; batch reale CPU confrontato allo shard; isolamento free/assisted/p_off/2C; TEST denied senza freeze. Picco RSS locale ~6.5 GiB (decode Arrow + workers); cache compatta residua piccola. Engaging/GPU non verificati. Nessun training né baseline fitting.

---

## 2026-11-16 — Step 2.2 correttivo: protezioni API prima del 2.3

- **Contesto:** revisione ChatGPT su `1fc150f`; pipeline reale già OK, ma tre usi impropri dell’API da chiudere.
- **Correzioni:** (1) `build_index` rifiuta sim_id fuori dallo split del ruolo prima di ogni I/O; (2) normalizer/compute verificano frame indices, split hash e `protocol_scientific_sha256` (stats numeriche invariate, meta provenance patch); (3) aggregazione rifiuta config/ruoli misti (`train_fit`≠`seen`); (4) documentato/misurato `prefetch` = 1 decode vs N miss.
- **Verifiche:** 55 pytest; validate-only PASS. Nessun 2.3 / training.

---

## 2026-11-16 — Step 2.3: baseline Month-2 su protocollo comune

- **Contesto:** base `82fcd34`; solo Step 2.3; fitting CPU autorizzato; nessun TEST/freeze/cluster/neurale.
- **Eseguito:** `configs/experiments/month2_baselines.yaml` preregistrato; adapter isolation `month2_baseline_adapters.py`; runner `scripts/06_run_month2_baselines.py` (smoke + full). Fit su 760 TRAIN-fit; eval seen 220 + VAL 123; maschere 0/1/2; 2C 5%; GlobalMean, ParamMean (LOCO `piecewise_phi1`, α=0.01), GappyPOD R=32 p off/on free/AO/T-block, OraclePOD, LocalHGB dense diagnostico.
- **Search:** free λ∈{1e-4…1}; assisted 3×3 a K=9. Selezionati: free p_off λ=0.001, p_on λ=0.01; T-block p_off (0.001, 0.5), p_on (0.01, 0.5). R=64 non lanciato.
- **Numeri guida VAL (RMSE K):** GlobalMean 139.7; ParamMean 88.0; Gappy free p_on 70.6; T-block p_on K=9 67.5 / K=32 63.4; Oracle ~35. Stima ≫ rappresentazione.
- **Controlli:** AO Centered-RMSE invariante; K=0 T-block flag = free; maschere K condivise; adapter≡API storica; contenitori observed-only. Fix locale: pad `t` su `40NH3_1.2.h5` (t_shape=2000 vs shape_t=2001).
- **Costi:** ~32 min wall; peak RSS ~10.4 GiB; 23 shard decode. Pytest mirati 35 passed.
- **Stop:** nessuna Step 2.4 / U-Net / GLU; revisione indipendente prima dei neurali.


---

## 2026-11-16 — Revisioni Opus/Astra e missione di consolidamento prima dei neurali

- **Provenienza:** Enrico ha condiviso le due revisioni indipendenti di `822ed52`; entrambe concludono «pronti dopo correzioni circoscritte». I revisori non disponevano dei raw; i loro controlli non equivalgono a un rerun completo locale.
- **Decisione di Enrico:** affidare a Cursor IDE locale correzioni e replica invariata delle baseline, con arresto prima di implementazioni/training neurali e prima di cambi scientifici. Creato [roadmap_missione.md](../roadmap_missione.md), collegato al piano principale.
- **Ambito:** controlli d'isolamento effettivi, diagnosi del timestamp `40NH3_1.2`, tracciabilità codice/ambiente/dati, contratto YAML, aggregazione e confronto con il run storico. Il vecchio «preregistrato» è una dichiarazione non attestabile cronologicamente; la replica non ne riscrive la storia.
- **Rinviati:** U-Net/GLU, R=64, nuove griglie/τ, esperimenti temporali e modifiche agli split. La futura consegna del codice sarà su branch/PR senza merge automatico.
- **Stato:** piano preparato da ChatGPT/Codex e autorizzato per l'esecuzione futura; in questa attività nessuna modifica di codice, nessun rerun, nessun training.

---

## 2026-11-16 — Missione consolidamento baseline eseguita (COMPLETATA, in attesa revisione)

- **Esecuzione:** branch `mission/baseline-consolidation`; freeze codice `2c800ce` poi replica `consolidation_v1`.
- **Dati:** SHA shard `40NH3_1.2` = inventario remoto; pad generico rimosso; frame 2000 = duplicato di 1999; tempo non validato oltre `t_len`.
- **Replica vs `822ed52`:** metriche per-frame max abs diff = 0; selezioni identiche; classificazione `identical_or_within_tolerance`.
- **Report:** [mission_baseline_consolidation_report.md](month2/mission_baseline_consolidation_report.md). Storico `results/tables/` intatto.
- **Stop:** nessun U-Net/GLU/Step 2.4; attendere Enrico/ChatGPT. Corregge anche la lettura troppo forte di «preregistrato» sul run storico (non attestabile cronologicamente).

---

## 2026-11-17 — Correzione metadati temporali post-revisione Opus

- **Contesto:** Opus su `129f298` — verdetto «accettabile dopo piccola correzione»; metriche storiche/replicate verificate (63 colonne comuni).
- **Correzione:** su `40NH3_1.2` l’associazione campo↔tempo è **non verificata per tutti gli indici**; `time_validated` non è più True su 0–1999; flag espliciti + guardia temporale; metadata dataset; niente rerun/baseline/neurali.
- **Invarianti:** freeze replica `2c800ce` e CSV `consolidation_v1` restano tali; cambia solo la semantica dei metadata del lettore.

---

## 2026-11-17 — Preparazione della missione U-Net densa (Step 2.4)

- **Mandato:** Enrico ha confermato il diagnostico U-Net e richiesto una roadmap dedicata e il push, dopo aver discusso la proposta dettagliata Opus. Questa attività pubblica il piano: nessuna implementazione, verifica raw, fitting o training eseguito qui; nessuna nuova ratifica supervisori.
- **Stato remoto verificato:** main `f34ce8c`; consolidamento completato su `mission/baseline-consolidation` a `3cf7bf0`, ancora non integrato. Il report del branch riporta replica invariata (66.180 righe, differenza massima metriche zero); freeze eseguito `2c800ce`, risultati `129f298`. La correzione successiva riguarda i metadati temporali di tutto lo shard anomalo, senza rerun. Queste evidenze del branch non vengono presentate come già integrate in main.
- **Output:** [roadmap_unet.md](../roadmap_unet.md), dettaglio operativo U0–U7 del solo Step 2.4; collegamenti e stato aggiornati in README, roadmap Mese 2 e registro. Nessuna roadmap concorrente per l'intero Mese 2.
- **Configurazione di partenza:** vi/vj densi single-frame + coordinate + canali costanti NH3/phi; T supervisionata ma assente dagli input; previsione diretta normalizzata, MSE, p_on seed 0. FiLM, residuo ParamMean, tempo, GLU e sensori T fuori missione. Architettura/budget finali da fissare con profiling, non su prestazioni VAL.
- **Confronti:** ParamMean e GappyPOD denso matched; scambio deterministico delle velocità fra istantanee della stessa condizione per verificare l'uso dell'allineamento istantaneo. Nessuna inferenza automatica di limite fisico o robustezza da un singolo seed.
- **Precisazioni rispetto alla proposta Opus:** testa inizializzata con piccoli pesi non nulli per consentire gradienti fin dal primo backward; niente soglia universale “tutte le tre VAL”; niente estensioni automatiche di capacità/frame; distanze fra indici distinte da tempi verificati; lambda denso riselezionato senza dichiarare invalido a priori quello sparse.
- **Workflow:** implementazione/test/overfit → STOP A; profiling + riferimento denso → STOP B; training p_on e interpretazione → STOP C; p_off/repliche con incarico delimitato. Il prossimo prompt a Cursor definirà l'ambito attivo. Pubblicare questo piano non dichiara alcuno di questi step completato.

---

## 2026-11-17 — U0–U2 Dense U-Net (STOP A) su `mission/dense-unet`

- **Eseguito:** branch dedicato con consolidamento `3cf7bf0` + roadmap `adabacc`; config densa distinta; cache 760/220/123 (~277 MiB) da raw locali; DenseUNet b=32 (**4 902 753** param); test `tests/test_unet_dense.py` 11 passed; overfit 8 frame TRAIN (4 condizioni), 500 update CPU, MSE_norm 0.909→0.00162, RMSE_K 316→13.4; checkpoint fuori Git.
- **Report:** [03_dense_velocity_diagnostic.md](month2/03_dense_velocity_diagnostic.md).
- **Non eseguito / non dichiarato:** U3+, profiling completo, GappyPOD denso, training scientifico, selezione VAL, GLU/masked/T-sensor/tempo, Step 2.4 completo.
- **Stop A:** attendere esame Enrico/ChatGPT.

---

## 2026-11-17 — STOP B: correzioni U1–U2 + U3 profiling/GappyPOD denso

- **Correzioni:** isolamento sample→input→forward; validazione cache (identità protocollo, fingerprint dati≠modello, SHA inventario esplicito); contratto YAML consumato/rifiutato; rifiuto gradienti non finiti prima di `optimizer.step`. Commit codice `11aae27` prima degli esperimenti.
- **Profiling (CPU, pesi nuovi):** 2 epoche TRAIN b=32 batch=8; ~93 s train + ~5 s eval; proiezione 200 ep+eval ≈2.6 h; proposta U4 b=32/batch8/CPU/E_max=200 (VAL non usata per scegliere).
- **GappyPOD denso:** `month2_dense_linear_v1`; R=32 K=0 cov=100%; λ selezionati VAL p_off=1e-3 (93.7 K), p_on=1e-2 (70.5 K); ParamMean TRAIN completo 88.0 K; sparse/consolidation intatti.
- **Report:** [03_dense_velocity_diagnostic.md](month2/03_dense_velocity_diagnostic.md).
- **Stop B:** U4 **non** avviato; attendere esame.

---

## 2026-11-17 — Preparazione runner U4 (training non eseguito)

- **Autorizzato:** solo implementazione/test piccoli del percorso epoch-based U4.
- **Implementato:** `training/epoch_loop.py`, CLI `train-u4` (richiede `--authorize-scientific-u4`), candidate YAML `month2_dense_unet_u4_candidate.yaml` (b=32, batch 8, CPU, 200 ep, lr=1e-3, wall 6 h configurabile, niente fallback 3e-4 automatico).
- **Verificato:** `tests/test_unet_epoch_trainer.py` (visite campione, eval/selezione/tie-break, checkpoint, wall/non-finite, rifiuto CLI senza auth).
- **Non eseguito:** training scientifico completo; nessuna nuova eval VAL scientifica; budget non dichiarato approvato.
- **Comando futuro:** vedi report `03_dense_velocity_diagnostic.md` sezione U4.

---

## 2026-11-17 — Preflight Engaging (sola lettura; U4 fermo)

- **Mandato:** ricognizione risorse ORCD/Engaging per fattibilità `official_thesis`; nessun job, allocazione, trasferimento o modifica remota; U4 non lanciato; budget/architettura/config U4 non toccati.
- **Osservato (~16:28–16:31 America/New_York):** SSH OK; account AMF GPU/CPU + `mit_general`; “4 normal + 4 preemptable” = MaxTRESPU `gres/gpu=4` su QoS `mit_amf_advanced_gpu` e `mit_preemptable` (non 8 GPU idle). Job personali: 2R+14PD tutti `dbg4b-*` su `mit_preemptable`/`mit_general`/QoS `normal` (1×H100) — occupazione HONF separata, non consuma i due cap da 4 AMF/preemptable-QoS. Zero job su `mit_amf_advanced_gpu`. Partition `mit_normal_gpu` MaxTime 6 h, PreemptMode OFF; preemptable fino a 2 d con REQUEUE.
- **Software/storage:** nessun checkout `official_thesis` sul cluster; env PyTorch trovato solo `~/envs/honf` (progetto separato); POOL/SCRATCH utente 0/1024 GiB; HOME ~69/200 GiB. Smoke CUDA pendente (serve allocazione).
- **Report:** [engaging_resource_preflight.md](month2/engaging_resource_preflight.md).
- **Stop:** nessun training locale/cluster; nessun `sbatch`/`salloc`/`srun`.

---

## 2026-11-17 — Decisione Enrico: U4 Engaging (prep + un run p_on seed 0)

- **Decisione operativa Enrico:** primo training U-Net su Engaging; max iniziale **1000** epoche; best su VAL; early stop su plateau; se a 1000 ancora migliora → questione aperta. Sostituisce candidato CPU/200; **non** adotta WSD Opus. Nessuna ratifica supervisori inventata. Solo p_on seed 0.
- **Pre-run chiuso:** checkpoint complete vs partial; interrupt non incrementa epoche; `last.pt` non sovrascritto da stato parziale; resume rifiuta incompatibilità; YAML consumato/rifiutato; config effettiva+provenienza; hash payload cache; metriche per-frame; resume solo a confine d’epoca (test continuo vs resume). Schedule: warmup 950 + ReduceLROnPlateau; early stop 200/15/0.1% con reset su LR drop.
- **HONF (sola lettura):** launcher dbg4b in `hetmf-runs/launcher-ops/debug-step4b/`; B2 AMF usa `mit_amf_advanced_gpu` + `mit_normal_gpu`. Nessuna modifica a env/job HONF; env tesi dedicato `~/envs/official_thesis`.
- **Prossimo in questo incarico:** commit prep → checkout/cache Engaging → smoke/profiling GPU → gate 4.5 h → un training se passa.

---

## 2026-11-17 — U4 Engaging eseguito (p_on seed 0) e STOP

- **Avvio:** job `29465586`, `mit_amf_advanced_gpu` / `mit_normal_gpu` / QoS AMF, GPU **L40S**, nodo `node3008`, SHA `50bda7f`, run `u4_pon_seed0_engaging_v1`. Coda ≈132 s; calcolo ≈28.5 min. Smoke/profile job `29465473` (gate ≈0.85 h ≪ 4.5 h).
- **Esito:** `stopped_early_plateau` a **730** epoche (max 1000); best VAL macro RMSE **26.931 K** @ ep. **160**; last 28.055 K; TRAIN@best 13.65 K, seen@best 37.68 K. LR finale 1e-5. Checkpoint hash in `results/.../checkpoint_hashes.json` (`.pt` su Engaging pool).
- **Confronto lineare matched (U3):** GappyPOD denso p_on VAL selezionato ≈70.5 K. Un solo seed; niente U5/p_off.
- **Report:** `results/month2_dense_unet/u4_pon_seed0_engaging_v1/auto_report.md` + aggiornamento `03_dense_velocity_diagnostic.md`.
- **STOP:** nessun U5, p_off, altri seed, GLU, sparse, sensori T o TEST.

---

## 2026-11-17 — U5 snapshot-swap (STOP C)

- **Eseguito:** codice `8549a9b`; job Engaging `29471596` (L40S `node3102`); best U4 @160 verificato (sha `6d456abc…`); replay aligned VAL/seen **identici** a U4 (Δ=0, atol dichiarato 0.05 K).
- **Risultato numerico:** VAL macro 26.93→96.55 K (Δ+69.6); seen 37.68→150.30 (Δ+112.6). Peggioramento soprattutto strutturale; `0NH3_1.05` molto più sensibile. Matched VAL: GappyPOD 70.46, ParamMean 88.01, GlobalMean 139.70.
- **Artefatti:** `results/month2_dense_unet/u5_pon_seed0_snapshot_control_v1/` (pairing, tabelle, figure, `auto_report.md`, `interpretation.md`).
- **Limiti:** VAL contaminata da selezione U4; un seed; niente causalità/ceiling/sparse. **STOP** prima di U6/p_off/altri seed.

---

## 2026-11-17 — Revisione Opus U4/U5 (registrata) e avvio U6 p_off seed 0

- **Revisione esterna (riferita da Enrico, non ratifica supervisori):** «utilizzabile con riserve»; nessun difetto invalidante. Opus ha letto codice/artefatti e ricalcolato metriche dai CSV; senza checkpoint/cache/raw e senza rieseguire l’inferenza.
- **Punti confermati:** replay aligned U5 ≡ U4; pairing/transform input corretti; baseline U5 rifittate (scostamento dall’incarico) ma metriche per-frame ≈ U3 entro ~5.5e-12; VAL non indipendente; best 26.931 K @160 e last 28.055 K @730; nessuna indicazione a prolungare U4.
- **Decisione operativa Enrico:** prima parte di U6 — **un** training denso p_off seed 0 + snapshot-swap sul best, stesso contratto U4 salvo canali p a zero e init fresco. Altri seed rinviati. Engaging autorizzato.
- **Prep codice:** provenance nel run manifest; test isolamento p_off; candidate `month2_dense_unet_u6_candidate.yaml`; `train-u6`; smoke/train/eval Engaging; eval/swap `scripts/10_u6_poff_eval_and_swap.py` riusando pairing U5 e tabelle U3 (no refit).

---

## 2026-11-17 — U6 p_off seed 0 + snapshot control (STOP)

- **Train:** SHA `2511280`; job `29476814` L40S `node3202`; early-stop @600; best VAL **39.669 K @30**; last 44.102 K; updates 57000; ≈24 min. Best ckpt sha `83307308…`.
- **Eval/swap:** SHA `815bb7a` (fix path symlink dopo fail `29479222`); job `29479532`; replay aligned Δ=0; VAL swap 39.67→102.41 (Δ+62.7).
- **Confronto:** Δ best VAL p_off−p_on **+12.7 K**; matched U3 riusati (GappyPOD p_off 93.66; GlobalMean 139.70). Bias e Centered-RMSE entrambi peggiorano senza p; dipendenza istantanea resta forte.
- **Artefatti:** `results/month2_dense_unet/u6_poff_seed0_engaging_v1/` e `.../u6_poff_seed0_snapshot_control_v1/`.
- **STOP (allora):** nessun altro seed / GLU / sparse in quell’incarico; U7 ancora da chiudere.

---

## 2026-11-17 — Revisione Opus U6 e chiusura U7 (documentale)

- **Evento sperimentale U4–U6:** stesso giorno (America/New_York); **registrazione chiusura U7:** sera del 17 novembre 2026.
- **Opus su U6 (`a3a4d8f`, riferito da Enrico):** U6 valido; confronto p_on/p_off utilizzabile con riserve; nessun difetto bloccante; pronti per U7 / prep sparse successiva; nessuna nuova verifica sperimentale richiesta. Lettura codice/CSV; senza checkpoint/cache/raw né re-inferenza. **Non** ratifica supervisori.
- **Correzioni U7:** report 03 non più “U6–U7 not started”; ranking p_off **39.67 K inferiore a GappyPOD p_on 70.46 K** (supera tutte le baseline dense riportate, resta peggiore di U-Net p_on); scomposizione best/last livello–struttura verificata dagli artefatti (`u7_closure_val_level_structure.csv`); provenance `clean:false` = untracked, `dirty_entries` vuoto; swapped senza `src_frame_index` documentato.
- **Decisione operativa Enrico:** missione densa conclusa nel perimetro U0–U6; U7 = consolidamento; p_on/p_off preliminare (seed 0); seed aggiuntivi rinviati; nessun nuovo training denso in chiusura.
- **Stato:** Step 2.4 chiuso con limiti; Mese 2 **non** concluso (2.5–2.9 aperti); merge main non eseguito.
- **Prossimo orientamento (non autorizzazione):** protocollo U-Net sparse (2.6) e audit GLU (2.5); p_on candidato sparse iniziale con confronto p_off previsto; best/last + livello/struttura nei report futuri.


---

## 2026-11-18 — Proposte indipendenti Opus/Astra e roadmap sparse (solo piano)

- **Base:** `87d1e3f40826a9343c3c2ce19d140d481b911619`, missione U-Net densa U0–U7 chiusa; nessun nuovo esperimento in questa attività.
- **Discussione:** lette separatamente le due proposte, poi confrontate con Enrico. Nessun rilievo riferito impone di riaprire il denso; maschere e adattamento GLU richiedono verifiche implementative future. I revisori non avevano raw/cache/checkpoint e non hanno riprodotto l'inferenza.
- **Decisione operativa Enrico:** unica [roadmapsparse.md](../roadmapsparse.md), dettaglio di 2.5–2.6 e conferma sparse pertinente di 2.8. U-Net da zero a 7 canali; primo training al 5%, controlli a 1/20%; maschere condivise persistite e annidate conservando il 5% storico; quattro screening U-Net/GLU p_on/off se fattibili; GLU deterministico p additivo, geometria isotropa, batch omogenei senza padding iniziale. Repliche dei soli contrasti scelti dopo screening.
- **Risorse:** Enrico richiede esplicitamente nessun tetto arbitrario di ore per training/campagna; precedente ceiling proposto 24 GPU-h superato per la missione sparse. Profiling e costo visibile, plateau operativo e tranche iniziale 1000 epoche riesaminabile dalle curve; walltime scheduler distinto, checkpoint/resume. Fino a quattro run concorrenti dopo validazione e verifica degli slot effettivi. Nessun training avviato da questo mandato documentale.
- **Output:** roadmap sparse, riferimenti e stato in roadmap Mese 2/README/registro D4-D8. Documenti e fonti preesistenti letti; controlli documentali, nessuna prova torch/GPU/raw.
- **Stato/STOP:** soltanto S0 pubblicazione; S1–S7 non eseguiti. Prossimo incarico Cursor previsto S1–S2 (osservazioni/audit + U-Net piccola verificata), STOP A prima dei run scientifici. Mese 2 ancora aperto, niente GLU/sparse/assisted/TEST avviati; nessuna ratifica supervisori o merge main.

---

## 2026-11-18 — S1–S2 sparse: maschere persistite, audit GLU, U-Net 7ch (STOP A)

- **Base:** roadmap `5de0d7be…`; branch `mission/dense-unet`; denso U0–U7 non riaperto.
- **S1 maschere:** materializzate 1%⊂5%⊂20% (164/819/3277) per 1103×3=**3309** istanze; join storico 5% **3309/3309**; verified_examples @5% 3/3; payload `data/processed/month2_sparse_eval_masks/` (aggregate SHA `7be78f35…`); manifest versionato; nessun TEST.
- **S1 GLU:** solo audit in `docs/month2/04_glu_release_adaptation.md`; `external/` invariato; wrapper non implementato.
- **S2:** `SparseUNet` 7 canali, **4 903 041** parametri (b=32); cache densa riusata dopo fingerprint; train_mask_seed=20261118 con `set_epoch`; protocollo `month2_sparse_protocol.yaml` rifiuta chiavi sconosciute.
- **Test:** 14 pytest passati (isolamento, contratti, resume complete≠partial, Kelvin, omogeneità evaluator).
- **Overfit diagnostico (CPU, 8 frame train_fit, seed subset 7):** fixed masks 40 ep loss 1.0015→0.1287; variable masks 15 ep 1.0015→0.3828; pesi **non** scientifici. Artefatti `results/month2/sparse_s2_diagnostic_overfit/`.
- **Report:** `docs/month2/05_sparse_reconstruction.md`.
- **STOP A:** niente S3/GLU trainer, niente screening/training 760, niente tuning VAL, niente tetti 24 GPU-h reintrodotti.

---

## 2026-11-18 — S3 GLU adattato + correzioni S1–S2 (STOP B)

- **Correzioni S1:** chiarito che 3309/3309 seed↔payload ≠ equivalenza storica completa; aggiunto join vs CSV consolidamento baseline @5% K=0: **3309/3309** match su `observation_hash` hex; 6618 righe `dense_full_velocity` contate a parte (`month2_sparse_mask_baseline_join_5pct.json`).
- **Correzioni S2:** test isolamento sul percorso campione→input; p_off ricostruito da p sorgente mutati; controlli positivi post-apprendimento; tautologie rimosse; resume diagnostico esplicitamente incompleto vs runner scientifico (requisito S4).
- **S3 GLU:** wrapper `AdaptedDeterministicGLU` (1 154 578 params); `glu_minimal.yaml`; encoder freqs 256 / decoder 64; p additivo bias-free; q isotropo; decoder pubblico; chunk 512 eval max_abs=0; `per_sensor_sigma` risk test eseguito.
- **Overfit diagnostico (8 frame S2, full grid, CPU):** fixed 1.262→0.367 (64 upd); variable 1.262→0.734 (40 upd). Pesi non scientifici.
- **Test (prima chiusura):** 41 passed (dense+sparse+GLU). Report `05` aggiornato.
- **STOP B:** niente S4 profiling/freeze, niente screening, niente training 760.

---

## 2026-11-18 — S4 in corso: runner scientifico + submission profiling

- **Runner:** `sparse_scientific_loop.py` (U-Net/GLU) allinea MSE full-field, AdamW 1e-3/1e-4, clip 1, fp32 no AMP/TF32/compile, batch 8, warmup 950, ReduceLROnPlateau VAL 5%×3 repl, eval/10, first absolute min, early-stop 200/15/0.1% con reset su calo LR, tranche 1000 estendibile, TRAIN/seen diagnostici.
- **Checkpoint/resume:** complete vs partial; identity lock; mask `set_epoch`; buffer geometria GLU; test sintetici su LR/contatori/RNG/history.
- **Profiling submitted (SHA `b844944`):** record in `docs/month2/s4_profiling_submission_record.md`.
  - U-Net **29552531** @ `node3504` L40S → runs `.../s4_profile_sparse_unet_29552531/` + `.../official_thesis/slurm-29552531.out`
  - GLU **29552532** @ `node3507` L40S → runs `.../s4_profile_adapted_glu_29552532/` + `.../slurm-29552532.out`
  - One-shot: entrambi **FAILED** (`ExitCode=1`, `ModuleNotFoundError: einops`). Nessun rilancio/auto-fix in questa turno. HONF 29547931/32 non toccati.
- **S5 preparazione (solo):** quattro candidati YAML + quattro sbatch + `scripts/16_train_sparse_scientific.py`; `scientific_training_authorized: false`; walltime/mem/microbatch **PROVISIONAL**; nessuna submission, nessun afterok.
- **GappyPOD 1/20%:** ancora aperta (artifact POD mancanti).
- **S4 non concluso; freeze scientifico non dichiarato.**

---

## 2026-11-18 — S5 candidates prepared (no submit)

- Index `configs/experiments/month2_sparse_s5_screening.yaml`.
- U-Net/GLU × p_on/p_off seed 0; run-id e directory separati; init fresca; gate `AUTHORIZE_S5_SUBMIT` + CLI `--authorize-scientific-s5`.
- STOP: attesa decisione Enrico su profiling.

---

## 2026-11-18 — S4 dep fix + profiling resubmit (STOP after IDs)

- **Diagnosi attempt-1:** entrambi i job (`29552531`/`29552532`) FAILED in ~14 s su L40S prima di qualsiasi misura; traceback `ModuleNotFoundError: einops` da `external/GLU_Reconstruction/src/models.py` via eager import in `models/__init__.py` (anche il path U-Net). Interprete: `~/envs/official_thesis` (torch `2.8.0+cu128`).
- **Fix env (solo thesis):** `pip install einops==0.8.2` in `/home/bontempi/envs/official_thesis`; HONF/`~/envs/honf` non toccato; torch/CUDA non aggiornati.
- **Fix import:** lazy GLU in `models/__init__.py`; import GLU condizionale in `s4_sparse_profile.py`; helper lazy in `sparse_scientific_loop.py`; preflight `s4_dep_preflight.py` nei launcher.
- **Test locali:** 30 passed (sparse scientific + GLU + sparse U-Net).
- **Resubmit solo profiling** (SHA `740805f`): U-Net **29554165**, GLU **29554166**; nuove directory; attempt-1 preservato; no preemptable; no S5/afterok; no auto-requeue. Record: `docs/month2/s4_profiling_submission_record.md` stato **submitted**. S4 non concluso; freeze non dichiarato.

---

## 2026-11-18 — S4 profiling outcomes + freeze candidate (no S5)

- **Esito:** entrambi COMPLETED `0:0` / `completed_diagnostic` su L40S (`node3008` U-Net, `node3408` GLU); coda≈189 s; compute 68 s / 79 s; SHA `740805f`; env thesis + `einops 0.8.2`.
- **Misure:** U-Net VRAM 0.60 GB, loop 31.6 s; GLU 4.33 GB, 40.7 s; batch 8; microbatch null; query 16384; inferenza 1/5/20% registrata; CPU/GPU ok; smoke loss/grad finiti. RMSE diagnostici non usati per scelta modello.
- **Stima tranche 1000 (con incertezza):** U-Net ~2–3 h, GLU ~3–4.5 h; wall scheduler candidato 6:00:00 + resume; non tetto scientifico.
- **Freeze prereq:** hash cache/stats/mask/geometry ok on-job; resume schedule verificato in test locali (non resume Engaging lungo). GappyPOD 1/20% bloccato (POD fit non persistito); non blocca S5 neurale.
- **Consegna:** `docs/month2/s4_scientific_freeze_candidate.md`; `results/month2/s4_profiling/`; config S5 campi operativi risolti, auth flags false. **Nessun nuovo job / nessuna submission S5.** STOP.

---

## 2026-11-18 — Correzioni circoscritte S3 (nuovo STOP B)

- **Tie-break reale:** sostituito `torch.topk` nel decoder thesis con `topk_distance_then_site_index` (distance↑, site_index↑, stable double-argsort); nessun jitter delle distanze; `external/` intatto. Corretta la descrizione precedente che attribuiva a `topk` la preferenza per indici inferiori. Test N>k, surplus equidistanti, indici attesi, permutazioni, controllo distanze non pari; predizioni invarianti a permutazione (CPU, atol=1e-5, rtol=0). Nessuna riproducibilità universale CPU/GPU dichiarata.
- **Geometria di dominio fissa:** center/L da assi completi `geometry.npz` (o assi sintetici espliciti), persistiti come buffer; errore se mancano; niente inferenza silenziosa dal primo batch. Invarianti A–E: max_abs misurati 0 (subset/batch/chunk/reload) con tol dichiarate. Diagnostici full-grid precedenti: sensors∪full queries ≡ mappa dominio (diff 0); chunking già applicava q prima dello split — non invalidare retroattivamente full/chunk. Sottoinsiemi angolo possono differire (L_diff≈0.060). Metriche overfit non riscritte; nessun tuning σ/VAL.
- **Isolamento T non vacuo:** mutazione T sul campione sorgente + rebuild `build_canonical_observation`; target cambiato, input GLU e predizione identici; firma senza target/G_full; p_off da p sorgente mutati; rtol=0.
- **Test:** 45 passed. Audit `04` / report `05` / diario aggiornati. **STOP B** — nessun S4.

---

## 2026-11-18 — S5 neural freeze + four-run submission (STOP after IDs)

- **Checkpoint fix:** `best.pt` saved after scheduler/early-stop/history updates; selected weights preserved; atomic save; tests strengthened (warmup crossed, LR reduce asserted, early-stop reset, resume best/extend, partial reject).
- **Freeze:** four S5 configs authorized (`scientific_training_authorized` + `s5_submission_authorized`); wall internal 5.5 h under Slurm 6 h; microbatch null; p_off applied per batch.
- **S4:** neural profiling OK; GappyPOD 1/20% still open — S4 not fully closed.
- **Submission:** U-Net p_on **29555700**, p_off **29555701**; GLU p_on **29555702**, p_off **29555703** (SHA `7971343`); no preemptable; no polling; HONF untouched. Record: `docs/month2/s5_neural_freeze_record.md`.

---

## 2026-11-18 — STOP E sparse delivery (S5 done + S6 + Gappy nested)

- **S5 seed 0 (HOME tip `38ac7fe`):** jobs 29567988/90/92/94 COMPLETED early-stop. Best VAL@5%: U-Net p_on **42.964**, p_off 43.526, GLU p_on 45.801, p_off 47.535 K.
- **GappyPOD nested:** job **29601906** (`mit_amf_advanced_cpu`, 128G) COMPLETED 55 s; load-equivalence ok; fit SHA `a6fbdfd2…`; **gate 5% passed** (6/6); 1% e 20% prodotti. OUT `~/official_thesis_gappypod_nested_1_20_amf_v2/` → mirror Git `results/month2/gappypod_nested_1_20/`.
- **S6:** best 1/5/20% riusati dal primo tentativo (fail swap `p_conditioning`); last+swap jobs **29601907–29601910** COMPLETED dopo fix collate. Coverage 27/27 best + 27/27 last ×4 run; swap seen+VAL + T_donor ref.
- **Risultati chiave:** deployable @5% neurali ≪ Gappy; inversione best↔last (U-Net p_on 1°→4°); @1% GLU davanti a U-Net; U-Net p_off peggiora a 20%; swap ΔVAL ~58 K.
- **Consegna revisori:** `docs/month2/05_sparse_reconstruction.md`; `results/month2/s6_sparse_eval/STOP_E_delivery/`; aggiornati `roadmapsparse.md` / `roadmap_month2.md`. **Nessun S7.** Lacune: montage scale comuni; solo seed 0.

---

## 2026-11-18 — Dopo le tre revisioni STOP E: stato, scopo e piano approvato (solo diario)

**Fatti eseguiti (non ricalcolati in questa entry).** Missione U-Net densa U0–U7 chiusa nel perimetro dichiarato ([03](month2/03_dense_velocity_diagnostic.md)). S4 baseline/profiling e freeze neurale eseguiti; S5 quattro training seed 0 conclusi (early-stop). S6 consegnato allo STOP E nel commit `a6b2bad7445170e6d821a14de9f7af9a1905ab2c` (`mission/dense-unet`), con lacune poi segnalate dai revisori. GappyPOD job **29601906** concluso, gate storico 5% passato, nested 1%/20% prodotti. Completamenti S6 **29601907–29601910** (last + snapshot-swap) recuperati nella stessa consegna. Tip di esecuzione Engaging: `38ac7fe`. Report versionato: [05_sparse_reconstruction.md](month2/05_sparse_reconstruction.md); artefatti: `results/month2/s6_sparse_eval/STOP_E_delivery/`.

**Risultati preliminari dal report (seed 0; da non elevare a conclusione multi-seed).** Al 5% VAL best i neurali restano nettamente sotto GappyPOD matched (U-Net p_on ~43 K vs Gappy p_on ~71 K). Lo snapshot-swap indica forte dipendenza dall’istantanea (Δ VAL swap−aligned ~54–58 K, vicino al riferimento T_donatore). Le classifiche cambiano con checkpoint (best vs last), copertura (a 1% GLU davanti a U-Net) e, dove ispezionato, condizione. Un solo seed di training: le tre «repliche» S6 sono **repliche di maschera eval**, non tre seed di modello/shuffle.

**Revisioni indipendenti (Opus, Astra, Neon) — direzioni discusse da Enrico, non risultati dimostrati.**
- *Opus:* verificare quanto quantità di frame e variabilità del training limitino la generalizzazione a nuove istantanee; propone una curva nel numero di frame e repliche controllate.
- *Astra:* separare livello e struttura usando gli attuali checkpoint, con oracle-offset diagnostico e correzione da poche ancore T, prima di scegliere il contrasto da replicare.
- *Neon:* completare diagnostici e valutare successivamente training multidensità, confronto U-Net/GLU e trasferimento fra condizioni.

**Rilievi comuni ancora da verificare/correggere (non risolti in questa turnazione):** protocollo YAML; provenienza del codice di valutazione e degli artefatti riusati; figure complete a scale confrontabili; metriche regionali aggregate; swap GappyPOD mancante nella consegna neurale.

**Scopo ingegneristico (da `docs/proposal.md` e contesto Mese 2).** La tesi mira a ricostruire campi di temperatura in camere di combustione da velocità accessibili (riferimento applicativo: PIV / accesso ottico laser) e, se serve, poche misure termiche (termocoppie). Il setup attuale usa osservazioni ideali su dati numerici e ancore T puntuali come proxy; il problema sperimentale guida scelte ed estensioni. Rigore, verifiche e riproducibilità devono sostenere questo obiettivo: le campagne diagnostiche rispondono a domande utili alla ricostruzione sparsa e alle decisioni successive, non a curiosità isolate.

**Piano approvato da Enrico — ancora da eseguire (nessuna implementazione in questa entry).**
1. Creare `analisi tripla revisore s7.md` con spiegazione completa delle tre proposte, convergenze/divergenze e decisione concordata; conservare bene le proposte Neon per riesame successivo.
2. Creare `roadmapopus.md` sulla quantità di dati e sulla variabilità dei training; matrice di run, controlli e criteri da fissare nella roadmap prima del lancio.
3. Creare `roadmapastra.md` su livello/struttura e contributo di poche ancore T, riusando i checkpoint dove possibile; piano dettagliato nella roadmap.
4. Eseguire le due sotto-missioni nell’ordine **Opus → Astra**, discutere gli esiti e riprendere la decisione S7. Le correzioni tecniche comuni saranno una preparazione condivisa, da fare una sola volta.
5. Dopo queste due missioni, riesaminare Neon. La roadmap Neon **non** è ancora da creare e la sua campagna **non** è autorizzata al lancio.

Numeri di run, soglie e raccomandazioni dettagliate dei revisori **non** sono decisioni già approvate: il disegno resta da definire nei passi successivi.

**Gerarchia e stato.**
- [roadmapsparse.md](../roadmapsparse.md) resta aperta allo **STOP E**; S7 sospeso in attesa delle due sotto-missioni, senza annullare gli esperimenti già eseguiti.
- `roadmapopus.md` e `roadmapastra.md` saranno sotto-missioni di approfondimento che informano la chiusura sparse.
- Il Mese 2 resta aperto; Astra contribuirà anche allo Step 2.7 (ancore T / assisted).
- Dopo Opus e Astra: discutere esiti, decidere il seguito sparse, poi completare gli aperti del Mese 2.
- Nessuna scelta definitiva tra U-Net e GLU; nessun training S7, nessun job e nessun file nuovo in questa turnazione (solo aggiornamento del diario).

---

## 2026-11-19 — Opus wave-1: preparazione, controlli e submission (quattro GPU normal)

**Fatti.** Prima onda Opus autorizzata da Enrico: `O-A38-on/off-s1` e `O-B19-on/off-s0` (SparseUNet, 5%, scientific epoch 760/95, normalizzatore 760 congelato, init fresca). Prep SHA `969b6c78a84465eb89b63bb5b7a0187c908652e5`. Job Engaging **29612010–29612013** sottomessi (account/QoS `mit_amf_advanced_gpu`, partition `mit_normal_gpu`, L40S×1, 64G, 6h). Correzioni comuni riusabili da Astra: YAML protocollo di nuovo parsabile (`authorized_s6_offline` mapping); `require_eval_mask_payload`; scrittura `eval_history.json` anche su early-stop; amp/compile/tf32 espliciti off; gate Gappy rifiuta riferimento vuoto; stream presentazioni + stato sampler in checkpoint; `init_manifest.json`. Tip Engaging codice: `$HOME/projects/official_thesis_opus`; cache/maschere riusate da tip S5 HOME; output `$HOME/official_thesis_opus_runs/`. Dettaglio: [opus_overnight_handoff.md](month2/opus_overnight_handoff.md).

**Controlli eseguiti (locali).** Nesting B19⊂A38 su cache; stream A38 ≡ formula seed DataLoader S5; B19 = due permutazioni/epoca e maschera condivisa per doppia presentazione; resume stream sintetico; p_off isolation; stessi pesi iniziali on/off a seed uguale; contratti YAML Opus; protocollo load. Dichiarato: **95 update/epoca verificati**; **equivalenza scientifica completa con il run Engaging S5 non rivendicata** (nessun bridge replay RMSE).

**Aperto per domani.** O1 (R0) non eseguito; provenienza S6/figure/regionali/swap Gappy e prosa roadmapsparse non chiusi; nessuna attività Astra; niente GLU/S7. Organizzazione: Cursor→Opus, Sonnet→Astra (best S5), Sol coordina.

---

## 2026-11-19 — Opus C149: cache densificata, tip separato, submission afterok

**Fatti.** Sorgenti Arrow verificate in WSL (`data/raw/combustion/hf_dataset/numerical`, 23 shard TRAIN+VAL); assenti su Engaging. Cache C149 costruita in streaming (`scripts/18_build_opus_c149_cache.py`): 2980/220/123; B19⊂A38⊂C149; 760 frame A38 bit-identici alla cache storica; geometria uguale; anomalia `40NH3_1.2` conservata. Normalizzatore storico 760 invariato. Tip esecuzione separato `official_thesis_opus_c149` (wave-1 tip e Astra frozen intatti). Prep SHA `07646e8`. Job CPU verify **29613544**; GPU **29613545/29613546** con `afterok:29613544`. Dettaglio: [opus_overnight_handoff.md](month2/opus_overnight_handoff.md).

**Controlli.** Sampler pool 2980 (residuo/confine/resume; nessuna coda droppata su una permutazione completa); 95 update/epoca; init on/off seed 0 uguale; contratti YAML C149; gate shared-A38. **Non rivendicato:** rebuild raw su Engaging; equivalenza scientifica completa vs S5 Engaging.

**Sei training Opus nuovi:** A38 s1 on/off, B19 s0 on/off (29612010–13), C149 s0 on/off (questa onda).

---

## 2026-11-19 — Opus C149 attempt2: allowlist della cache e risubmission

**Lettura notturna, non un nuovo calcolo.** Wave-1 **29612010–29612013**: Slurm COMPLETED per early-stop, con checkpoint `best`/`last`/`best_seen_diagnostic`. Il verify **29613544** è COMPLETED e il report remoto ha `status=ok` (2980/220/123, 760 frame condivisi a differenza 0). **29613545** e **29613546** sono FAILED in 9–14 s su L40S, prima dell'epoca 0: `SparseUNetCacheDataset` invocava `validate_dense_cache_manifest` senza allowlist, quindi l'unico id ammesso restava `month2_dense_unet_cache_v1`. Il Reason `QOSMaxGRESPerUser` è l'attesa in coda, non la causa dell'exit 1. Directory e `slurm-2961354{5,6}.out` del tentativo 1 sono conservati.

**Correzione eseguita.** `cache_validation_kwargs` passa il contratto del run al dataset: C149 solo `month2_dense_unet_cache_opus_c149_v1`, con conteggi 2980/220/123 e protocollo C149; A38, B19 e il default solo l'id storico. I conteggi servono perché, superato l'id, la stessa funzione avrebbe confrontato 2980 con il default storico 760. Gli altri gate di manifest restano. Test: `tests/test_opus_c149_cache_contract.py` insieme a `tests/test_scientific_epoch_sampler.py`, 22 passed. Nessun training locale.

**Attempt2 sottomesso, non concluso.** SHA eseguito `d6332000491ba0e8983b9e7fa09b34badb191ea4`. Checkout congelato `/home/bontempi/projects/official_thesis_opus_c149_attempt2`; il checkout precedente resta `f061df1`. Job **29645053** (`O-C149-on-s0`) e **29645054** (`O-C149-off-s0`), PENDING alle 09:38:08 EST, `mit_normal_gpu` / `mit_amf_advanced_gpu` / L40S×1 / 64G / 6h. Output nuovo: `/home/bontempi/official_thesis_opus_c149_attempt2_runs/`. Cache e report di 29613544 riusati; aggregate payload `fa1c9358deb3bda0313cd74f9ab6855af87bc969c4069a463073d4c4120f03b4`. Nessun warm start. Wave-1, S5, Astra e HONF non toccati. Opus, sparse e Mese 2 restano aperti. Dettaglio: [opus_overnight_handoff.md](month2/opus_overnight_handoff.md).

---

## 2026-11-19 — Sei training Opus in early-stop; bridge S5 e R0

**Sei training, tutti early-stop.** Slurm COMPLETED con exit 0 non è convergenza: `summary.completed` resta falso e lo stato è `stopped_early_plateau`. VAL ha guidato checkpoint, scheduler e arresto.

| Run | Job | Epoche | Best VAL (K) | Epoca best |
| --- | --- | ---: | ---: | ---: |
| O-A38-on-s1 | 29612010 | 630 | 48.219 | 60 |
| O-A38-off-s1 | 29612011 | 610 | 43.056 | 40 |
| O-B19-on-s0 | 29612012 | 620 | 43.929 | 50 |
| O-B19-off-s0 | 29612013 | 630 | 46.737 | 60 |
| O-C149-on-s0 attempt2 | 29645053 | 660 | 38.712 | 90 |
| O-C149-off-s0 attempt2 | 29645054 | 710 | 39.824 | 260 |

C149 attempt2 è il run scientifico: init condivisa fra on e off a seed 0, cache `fa1c9358deb3bda0313cd74f9ab6855af87bc969c4069a463073d4c4120f03b4`, nessun partial. Il tentativo 1 (29613545/46) resta il fallimento di allowlist. Il checkout attempt2 resta fermo a `d633200`.

**Bridge, due controlli separati.** Il confronto statico e il test sintetico, ripetuto su torch 2.8 di Engaging, mostrano che lo stream A38 è un `randperm` sul seme, mentre S5 ha eseguito `DataLoader(shuffle=True)`, che estrae prima un intero dallo stesso generatore. All'epoca 0 le sequenze iniziano rispettivamente con `[26, 184, 591, …]` e `[193, 743, 248, …]`: stessi 760 frame, tutti i 95 batch diversi. Loss, optimizer, warmup, plateau e TF32 spento coincidono; l'hash iniziale S5 non esiste. Questo basta a non usare S5 come braccio A38 seed 0 del runner nuovo. Il replay di inferenza dei due `best.pt` U-Net al 5% è il job **29670888**, sottomesso e non letto: tolleranza macro 0.05 K, e non prova l'equivalenza del training. Ledger: `results/month2/opus_data_study/bridge_s5/LEDGER.md`.

**R0 eseguito sulle curve, non sui pesi.** Per i quattro S5: minimo seen, leave-one-VAL-condition-out e primo minimo della media mobile causale di cinque eval, con le prime quattro escluse. Best e last originali restano. L'eval di arresto, assente da `eval_history.json`, è stata unita dal CSV e da `progress.json`. Non è una validazione indipendente. Tabelle: `results/month2/opus_data_study/r0_selection/`.

**O5 non preparata e non sottomessa.** Il divario di ordine confonde il confronto dati/seed. Due riferimenti A38 seed 0 nel runner nuovo sarebbero necessari per quel contrasto; non sono stati lanciati. Opus, sparse e Mese 2 restano aperti. Nessun S7, A4 o TEST.

---

## 2026-11-19 — Due riferimenti A38 seed 0 sul runner Opus

**Motivazione.** Il bridge ha mostrato che S5 ha eseguito il DataLoader, mentre i sei training Opus usano lo stream. S5 non può fare da braccio A38 seed 0 di quel runner. Enrico ha autorizzato `O-A38-on-s0` e `O-A38-off-s0`: SparseUNet b32, init fresca, stesso contratto A38 seed 1 con soli `seed` e `shuffle_seed` a 0. Lo stream non è stato modificato per imitare il DataLoader. Nessun GLU, nessuna O5.

**Replay 29670888.** Alle 14:13 EST era RUNNING su L40S da un minuto e mezzo, stdout solo di avvio, stderr vuoto, report assente. Alla submission dei riferimenti, intorno alle 14:19, era ancora in esecuzione. Nessun errore di identità visibile. Non è stato atteso: il replay di inferenza resta distinto dalla differenza di shuffle già accertata.

**Submission.** SHA eseguito `32a59cab21884d8c052f941f526c7cf09367b1f3`. Checkout congelato `/home/bontempi/projects/official_thesis_opus_a38s0`. Output nuovo `/home/bontempi/official_thesis_opus_a38s0_runs/`. Job **29671719** (`O-A38-on-s0`) e **29671721** (`O-A38-off-s0`), entrambi PENDING alle 14:19:40 EST. Risorse del freeze A38: `mit_normal_gpu`, account/QoS `mit_amf_advanced_gpu`, L40S×1, 8 CPU, 64G, 6h, tetto interno 5.5h. Il QoS consente 4 GPU, 64 CPU e 1T per utente; il replay ne occupava una, quindi i due riferimenti rientrano nel tetto e possono partire quando la partizione ha i nodi. Cache storica `month2_dense_unet_cache_v1`, aggregate `710bab909a0efb6cc1b4b4a0d7f7dcc4889472b7adc8076abb94fd529576ab5d`; il gate rifiuta la cache C149. Gli hash consumati vengono scritti in `slurm_context_O-A38-*-s0.txt` all'avvio, prima del training. Wave-1, C149, bridge, Astra, S5 e HONF non sono stati toccati. Opus resta aperto.



---

## 2026-11-19 — Chiusura Astra primaria e diagnostica median9; revisione Neon

**Decisione di Enrico/coordinatore.** Conclusi nel perimetro svolto il blocco Astra primario S5 A0–A3 e l'estensione esplorativa median9. Nessuna nuova inferenza o sweep richiesto per questa chiusura. A4 sui checkpoint Opus resta non eseguita e da decidere dopo Opus. Sparse, S7 e Mese 2 restano aperti. I pacchetti e i report sono sui branch dedicati elencati sotto: questa entry non ne costituisce integrazione in `mission/dense-unet`.

**Primaria S5.** Quattro best originali U-Net/GLU adattato deterministico × p_on/off seed 0, velocità 5%, seen220/VAL123, tre repliche di maschera, T9 fisso [32,64,95]×[32,64,95], rumore 0. Free, oracle-offset e AO9 tau1 dalla stessa predizione. Pacchetto tecnico `86a925af61accdef0b235ac084827cf639c1040c` su `mission/astra-level`; report 07 revisionato dal coordinatore in `82bff35726fb70af9987081802ece3ef718e42bd` su `review/astra-07-report`. Il replay neurale automatico originale aveva riferimenti vuoti; gli originali sono preservati e la verifica sostitutiva copre 24/24 celle tramite S6, con riscontro S5 sulle celle disponibili. Risultato: il livello spiega il crollo di IoU nel caso motivante, ma AO9 non migliora sistematicamente le reti sulle macro.

**Second pass autorizzato, distinto dal primario.** Aggiunti mean9 senza shrink e mediana ordinaria dei nove residui senza shrink, mantenendo pesi, siti e osservazioni. T-only storico separato. Nuova inferenza `44e5e5e0bd0907560d1736a53d357933680b090a`: task GPU **29686178_0–3**, baseline CPU **29686179**. Cursor riferisce tutti COMPLETED 0:0, gate/replay completo accettato, hash delle maschere su ogni riga e residui delle nove ancore persistiti. Aggregazione `04e3b68addc891034f0840c2cbaa141d5ef8c4ac` su `astra/median9-second-pass`, indice `results/month2/astra_median9_second_pass/full_set/INDEX.md`; i 12 esempi frame1000 restano separati dall'insieme completo.

**Revisione Neon.** Report 08: `7cb7bfdc6430a2731c924a31d90897a3aa52a89f`, branch `review/astra-08-median9`, file `docs/month2/08_astra_median9_second_pass_analysis.md`. Neon riferisce ricalcoli delle tabelle aggregate e coerenza del replay primario, nessun errore bloccante e giudizio favorevole alla chiusura senza ulteriori run. Distingue le verifiche Git dagli output grezzi/log/gate remoti, documentati da Cursor. Il coordinatore ha letto report e consegne; non ha rieseguito inferenza, accounting o ricalcolo indipendente delle tabelle.

**Numeri dal report Neon, macro sulle tre maschere, RMSE in K (free / AO9 / median9 / oracle).**

| Modello | VAL | Seen |
| --- | --- | --- |
| U-Net p_on | 42.96 / 43.14 / 38.58 / 38.36 | 51.67 / 53.81 / 51.10 / 50.61 |
| U-Net p_off | 43.53 / 42.86 / 38.49 / 38.00 | 55.44 / 53.78 / 51.79 / 51.17 |
| GLU p_on | 45.80 / 51.12 / 44.88 / 44.61 | 55.51 / 58.48 / 55.56 / 55.09 |
| GLU p_off | 47.54 / 48.20 / 43.16 / 42.72 | 60.31 / 61.83 / 59.25 / 58.51 |

Median9 migliora la RMSE macro rispetto ad AO9 su tutti gli otto metodi, e rispetto a free su 8/8 metodi VAL e 7/8 seen (GLU p_on seen +0.05 K). Sulle reti VAL dista circa 0.2–0.5 K dall'oracle. Mean9 senza shrink peggiora rispetto ad AO9 sulle macro considerate: il miglioramento della mediana non è spiegato dalla sola rimozione dello shrink. Non si assegna una quota causale «100%» allo stimatore e non si seleziona retroattivamente una ricetta diversa per condizione.

**Interpretazione e limiti accettati.** I residui nei due siti (32,95)/(95,95) sono grandi rispetto agli altri siti e sostengono l'ipotesi di sovrarappresentazione degli errori nei getti nella media delle ancore; sono errori del modello, non rumore strumentale. La mediana stima bene l'offset nei casi valutati, ma non è per definizione uno stimatore della media spaziale. Una traslazione non cambia la centered-RMSE; RMSE regionale e IoU possono cambiare. Le reti restano migliori delle baseline congelate anche con median9, senza dimostrazione di indispensabilità universale. Singolo seed, layout unico, T ideali e analisi motivata da VAL già usata in selezione: evidenza esplorativa, non TEST indipendente.

**Rilievi non bloccanti di Neon.** Il filtro `anchor_col == 95` include tre siti, mentre alcune descrizioni parlano di due: la relativa quota non va attribuita ai soli due siti di bordo getto. Le colonne specifiche mean9/median9 sono non applicabili e NaN sulle righe degli altri trattamenti; le tabelle degli errori di offset usano la colonna comune senza tale lacuna. Nessun cambiamento a metriche o artefatti storici in questa entry.

**Seguito concordato.** Attendere e verificare i riferimenti A38 seed0 e il replay Opus, completare O5 e analisi Opus; poi sintesi con Enrico di Astra+Opus, riesame delle proposte Neon già documentate e decisione esplicita su S7. Nessuna chiusura automatica sparse o Mese 2 e nessun nuovo training autorizzato da questa entry.

---

## 2026-11-19 — Esiti A38 seed 0 e replay; O5 preparata e non sottomessa

**Tre job, una lettura.** Tutti COMPLETED con exit 0. Nessun partial, stderr vuoto, nessun NaN, OOM o timeout. I numeri del training sotto sono metriche del loop, non la valutazione O5.

| Job | Run | Nodo | Durata | Arresto | Epoche / update | Best VAL | Best seen, replica 0 |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| 29671719 | O-A38-on-s0 | node2804 | 03:42:22 | `early_stop_no_rel_improvement_0.001_for_15_evals` | 630 / 59850 | 42.389 K all'epoca 120 | stessa epoca, 50.881 K |
| 29671721 | O-A38-off-s0 | node3008 | 03:55:50 | stesso motivo | 660 / 62700 | 42.915 K all'epoca 90 | epoca 130, 56.750 K |
| 29670888 | replay U-Net S5 best 5% | node2804 | 00:06:26 | inferenza conclusa | — | delta 0.0 K, entro 0.05 K | non applicabile |

SHA eseguito dei due training: `32a59cab21884d8c052f941f526c7cf09367b1f3`. Checkout lasciato fermo in `/home/bontempi/projects/official_thesis_opus_a38s0`. Output in `/home/bontempi/official_thesis_opus_a38s0_runs/`. Cache storica `710bab90…`, maschere `2e2d8ff6…`, normalizzatore `678263fb…`. Per on, `best.pt` e `best_seen_diagnostic.pt` hanno epoche uguali e pesi uguali, file diversi. Per off le epoche sono 90 e 130. Le history contengono l'epoca di arresto e il minimo coincide con il summary. `summary.completed` resta falso.

Il replay ha confrontato i due best U-Net S5 al 5% con le macro registrate: delta 0.0 K, `passed` vero, `training_equivalence=not_tested`. I riferimenti non erano vuoti. Non cancella la differenza di ordine fra DataLoader S5 e stream Opus già scritta nel bridge.

**O5 preparata, non lanciata.** La matrice è `configs/experiments/opus_o5_matrix.json`: otto run del runner Opus, con S5 tenuto fuori. reference760 usa gli indici A38 anche su B19 e C149. Coperture 1/5/20% e repliche 0/1/2, queste ultime presenti nel payload. I pesi identici di tre coppie best/best_seen possono condividere il forward e restano identità distinte. Indice: `results/month2/opus_data_study/o5_prep/INDEX.md`. Le curve del loop sono lì, etichettate come tali. R0 non è stata rifatta. Il report scientifico resta a Neon. Opus non è chiuso: O5 non è eseguita, O6 non è iniziata, S7 non è scelto, A4 non è autorizzata.

---

## 2026-11-19 — Array O5 29716019 fallito prima della valutazione

**Otto task, stesso errore.** 29716019_0–7 sono FAILED con ExitCode 1:0, durate 8–12 s, nodi node4108, node4203, node4206, node3007, node4106, node4108, node4203, node3204. Stdout vuoto. Gli otto stderr hanno lo stesso SHA-256. Checkout eseguito ancora `c70565978b27ebdae7ad11a84dcefecf3845dde2`. Nessun `COMPLETE.json`, nessuna riga di valutazione, nessun confronto col loop.

L'import in `scripts/26_opus_o5_eval.py` chiede `combustion_recon.training.normalizer`. La classe `Month2FitOnlyNormalizer` sta in `combustion_recon.data.month2_normalizer`, che è il modulo usato dall'evaluator S6. L'import avviene dopo `mkdir` della directory di run: le otto directory sotto `/home/bontempi/official_thesis_opus_o5_runs/` esistono e sono vuote, quindi il launcher attuale le rifiuterebbe. Non è stato fatto alcun rilancio. Aggregazione, figure e pacchetto Neon non sono stati prodotti. Opus resta aperto.


---

## 2026-11-20 — Decisione di prosecuzione: dieci training free, sweep K, poi T in ingresso

**Decisione approvata da Enrico; aggiornamento documentale del coordinatore.** Dopo lettura e discussione dei report Opus/Astra e della proposta Neon, Enrico sceglie un confronto architetturale più ampio della raccomandazione di sole quattro nuove U-Net. Sono aggiornate le roadmap esistenti e il registro, senza cancellare lo storico né creare una roadmap concorrente. **Nessuna submission, inferenza, verifica Slurm o esecuzione di training effettuata in questo aggiornamento.**

**Chiusura Opus e fonti.** Accettata la consegna tecnica `5373aa3c7633585edcee3d879b3c92bbb4748c11` (branch `opus/o5-eval-attempt2`) e la review scientifica `4f0e2637758de0766d0bb0891fe8ff97eee978c9` (branch `review/opus-06-report`, report 06). L'esecutore riferisce otto eval valide, array 29728465 (0 e 2–7) e sostitutivo 29732630_1; packaging 29771334 dopo fix dei soli controlli NaN del post-processing. La review riporta 72/72 confronti O5/loop entro 0.05 K, massimo 0.0011 K, senza bloccanti. Il coordinatore ha letto report e consegne Git, non ha riletto accounting/checkpoint/raw sul cluster. O6 sceglie C149 e lo stream Opus per il seguito, senza affermare overfitting eliminato o validazione TEST indipendente.

**S7 / primo blocco, temperature-free.** U-Net masked e GLU adattato deterministico, C149=149 frame per 20 condizioni (2980), velocità TRAIN 5%, p_on/off e seed modello/shuffle 0/1/2: **dodici run complessivi**. I due `O-C149-on/off-s0` esistenti sono riusati; **dieci nuovi training approvati = quattro U-Net s1/s2 + sei GLU s0/s1/s2**. I GLU storici A38 non valgono come C149 seed 0. Normalizzatore fit-only 760 congelato, stream 760 presentazioni/95 update per epoca, train_mask_seed=20261118 e payload eval comuni; nessun cambio architetturale o TRAIN multidensità. Regole scientifiche Opus conservate; nuovi percorsi/versioni dedicati. Integrazione GLU nel runner, smoke sul percorso reale, profiling e freeze precedono il lancio, nel prossimo incarico Sonnet. Massimo quattro GPU complessive normal Advanced L40S, limiti e walltime verificati al lancio; nessun fallback general/preemptable.

**Secondo blocco, dopo conclusione e analisi free.** Sweep K=0/1/4/9/16/32 di correzioni AO tau1, mean senza shrink e median ordinaria, dalla stessa predizione congelata e con baseline matched. Layout annidato/coordinate/hash decisi prima dell'esecuzione, senza selezione sui residui. K=9 non è dichiarato ottimale; la griglia storica resta identificata. Nessun training neurale necessario allo sweep; eventuali forward se le predizioni non sono salvate. A4 autonoma prima di S7 non è un prerequisito.

**Terzo blocco, dopo lo sweep.** Provare training che ricevano effettivamente le T osservate e le loro posizioni, sul modello scelto dal confronto free; **se non emerge una scelta chiara fra GLU e U-Net, su entrambi**. Identici K/siti/velocità/frame del secondo blocco per il confronto appreso vs correzione. Forma dell'integrazione, p, semi/numero di training e distribuzione TRAIN-K da congelare prima della relativa esecuzione: il piano non autorizza un numero indefinito di job o un training per ogni K. Il ramo free storico resta preservato.

**Superamento delle proposte precedenti.** La proposta 09 (`9daf1585ba2bd9d264372e62565606a800fbefe5`, `review/astra-opus-s7-proposal`) resta consultiva; non adottati rinvio GLU e sequenza A4 prima della campagna free. Astra primaria/report 07 e median9/report 08 restano conclusi nel perimetro S5, senza riapertura; i nuovi esperimenti T sono Step 2.7 Mese 2. Report e codice sui branch sorgente non sono integrati da questa entry; nessuna modifica a metriche, checkpoint o proposal.

**Regole di conclusione e aperti.** S7 richiede matrice valida, eval e confronto multi-seed, costi, comparatori a pari dati (distinguendo baseline C149 da quelle A38), completamento residuo GappyPOD swap e report STOP F. Un risultato negativo valido o l'assenza di un vincitore non obbligano ad altre campagne. Sparse si chiude dopo questo blocco; Mese 2 resta aperto per sweep e T apprese, decisione separata su eventuale TEST/fold, report e integrazione finale. Test del Mese 1 già esaminati; VAL è sviluppo, non una conferma indipendente. Ratifica dei supervisori distinta. Nessuna modifica a HONF, presentazione, `external/` o checkout remoti.


---

## 2026-11-20 — S7: preparazione, smoke e submission dei dieci training free C149

Branch `s7/c149-free-campaign`, SHA eseguito `f792b73388d43f008e8979f1aad0d1fa961cb559`. Preemptable con account Advanced non consentito dallo scheduler (`mit_amf_advanced_gpu` valido solo su `mit_normal_gpu`); Enrico sceglie `mit_normal_gpu` + Advanced non preemptable. Smoke reali U-Net (29797847) e GLU (29797848): PASS, dopo due errori reali corretti (automount `/orcd/pool` in timeout sui nodi → copia in home con stessi hash; RNG di resume su GPU da riportare su CPU). Array **29799680** (`0-9%4`, L40S, 6h, cap interno 5.5 h): task 0–3 (U-Net) in esecuzione con epoche reali; task 4–9 (GLU) pending. Dettagli e comandi in [s7_training_handoff.md](month2/s7_training_handoff.md). Nessun training concluso, nessuna valutazione; S7, sparse e Mese 2 restano aperti.


---

## 2026-11-21 — S7: valutazioni, residui e chiusura STOP F (temperature-free)

Branch `s7/c149-free-campaign`. Valutazioni 12/12 (SHA `acc8e2b`, matrice `a76bac9b…`), gate loop 18/18 con scarto 0 K; continuazione di glu-on-s1 chiusa a epoca 1080 (best 630 invariato). Residui: swap neurali sui 12 best (job 30027646), swap GappyPOD con oggetti A38 congelati (30027643), baseline C149 con iperparametri congelati e controllo sul sottoinsieme A38 (30027642), diagnostico batch GLU (30027668: nessuna dipendenza dagli altri esempi a B=8 fisso; sensibilità numerica dei pareggi top-k alla forma del batch). Figure dei campi e correzione dell'etichetta centered-vs-bias. Revisori Opus/Neon: U-Net complessivamente preferibile nella ricetta free; decisione del coordinatore: U-Net p_on candidato operativo free, p_off nel confronto, sweep K su entrambe le famiglie e p_on/off, matrice con T dopo lo sweep (decisione di sviluppo post-risultati, non regola preregistrata). **STOP F concluso nel perimetro temperature-free; Mese 2 aperto.** Indice: `results/month2/s7_c149_comparison/package/INDEX.md`; report `docs/month2/10_s7_c149_model_comparison.md`.


---

## 2026-11-22 — Step 2.7-B: 12 training temperature-assisted conclusi e valutati

Branch `temperature/assisted-c149`; contratto `10ff6e5`, training `e5136b6`, catena `b7e3b1f`. U-Net e GLU × p_on/off × seed 0/1/2, K variabile {0,1,4,9,16,32}, criterio di selezione a sei K sul dominio comune. 9 early-stop, 3 GLU a 1000 epoche senza plateau accertato; preemption GLU gestite con resume. Valutazioni 12/12, gate loop 69/69 (scarto max 0.0 K). Guadagno da T ≈ −3 K a K=32 solo per GLU p_off; ≲ 0.2 K per U-Net e GLU p_on. Report `docs/month2/12_temperature_input_study.md`. Sweep delle correzioni (Cursor) separato.

## 2026-11-22 — Step 2.7-B: continuazione dei tre GLU, valutazioni v2 e consegna congiunta con lo sweep K

Array 30082569 (glu-on-s1, glu-off-s1, glu-off-s2, da epoca 1000, max 1100): tutti e tre in plateau (1030, 1080, 1060 epoche), best VAL invariato (820/750/730). Catena v2 30082570 → valutazioni 30083486 (12/12, gate 69/69, scarto 0.0 K) → aggregazione 30083487; matrice v2 `3663d338…`, riuso per hash dei checkpoint invariati. Pacchetto `results/month2/temperature_assisted_c149/package_v2/`; il pacchetto v1 (`aa8f65f`) resta intatto. Consegna congiunta con Cursor `be18ad0`: launcher 30083178, post-processing 30083497, 26 controlli (tutti i bloccanti superati), `results/month2/joint_step23/package/INDEX.md`. Nessun nuovo training, nessun TEST, nessun vincitore.

---

## 2026-11-24 — Nuova priorità: osservazioni di velocità non ideali; A/B/C sospese

**Pianificazione, nessuna esecuzione.** Dopo il report del Mese 2 (inviato il 22 novembre), Linzheng ha chiesto di privilegiare la rilevanza applicativa e il contributo di un domain expert rispetto a nuove ablation sullo stesso benchmark. Secondo quanto riferito da Enrico, il colloquio del 24 novembre con il Prof. D'Errico (ICE, PoliMi) indica interesse, dati soprattutto RANS motoristici, LES e condivisibilità da verificare, nessuna competenza diretta su PIV in combustori ottici; suggeriti EngineBench e archivi Sandia come riferimenti. La scelta tra motori e combustore non è presa.

Pubblicate [roadmaptests.md](../roadmaptests.md) (Blocco I audit e modello dell'osservazione; Blocco II robustezza dei sei checkpoint best free p_on; Blocco III retraining condizionato a una regola fissata prima) e la nota [docs/nonideal/00_ricognizione_osservazioni.md](nonideal/00_ricognizione_osservazioni.md). Verificati: assenza di PIV e di T misurata nella combustione RealPDEBench; vettori mancanti codificati a zero e lacune artificiali in EngineBench (codice letto); nessun dato TNF letto con velocità e T simultanee co-localizzate; canale Mv della U-Net come validità nativa; divieto di N eterogeneo nel batch GLU. Controlli leggeri solo su frame TRAIN.

Roadmap A/B/C (`plan/abc-preparatory-studies` @ `9d2c3af`) **sospese, non cancellate**: nota datata in testa a ciascuna, piano storico integro; secondo Enrico non erano state eseguite. Il vecchio incarico amministrativo A/B/C è superato. Nessun training, inferenza, submission Slurm o accesso a TEST. Termocoppie all'uscita restano un tema separato.

---

## 2026-11-24 — Blocco I: specifica delle osservazioni degradate (preparazione, nessun esperimento)

**Cosa è stato fatto.** Branch `nonideal/block1-degradation-spec` (da `8d72a81`). Completate le verifiche documentali: Appendice B di RealPDEBench letta (B.1/B.2 e Tabella 7 verbatim, B.3/B.4 solo in sintesi); piano, interpolazione, definizione e segno di i/j/k e mesh/SGS risultano non dichiarati nelle fonti consultate; l'orientamento delle componenti è stato riclassificato in verificato / inferito / non dichiarato. Scritta `configs/nonideal/degradation_spec_v1.yaml` (sei celle R0, N1, G1, N1G1, G1-eq, N2; G1 = unione di dischi, area 25 %, diametro 3–10 mm, bordi ritagliati; seed in un namespace separato), il modulo `nonideal_observations.py` e i test CPU. G1 qualificato su geometria e maschere TRAIN (2280 pacchetti): area effettiva 25.0–26.0 %, 165–260 siti rimossi, minimo 559 superstiti, nessun rifiuto. Roadmap aggiornata: rimosso il cancello automatico di retraining, 0.04 K del GLU declassato a osservazione, troncamento al minimo del batch lasciato aperto per il Blocco III.

**Limiti.** I valori numerici sono default di scenario, non misure del nostro combustore; G1 è uno stress test sintetico; nessuna PIV né incertezza qualificata; EngineBench non scaricato. Nessun forward di modelli, nessun VAL/TEST, nessun job. In attesa della revisione di Enrico prima dello smoke GPU.

---

## 2026-11-24 — Blocco II: ampliamento a 12 modelli, implementazione e test CPU (campagna non eseguita)

**Cosa è stato fatto.** Enrico ha approvato la specifica del Blocco I e ampliato il Blocco II a 12 checkpoint congelati (U-Net/GLU × p_on/p_off × seed 0/1/2). Scritta la revisione `configs/nonideal/degradation_spec_v1_12models.yaml` (v1 intatta; operatori, livelli, geometrie e generatori invariati, verificato da test; modelli e hash dalla matrice S7), il modulo di valutazione `nonideal_eval.py`, il task per checkpoint (`60_nonideal_eval.py`: R0 + gate contro il riferimento S7, poi le altre cinque celle, readback), verifica e aggregazione con figure (`61`, `62`) e la catena Slurm (`nonideal_*.sbatch`, `nonideal_submit.sh`). Test CPU: suite completa 289 passati, 12 saltati; selftest con pesi casuali su frame `seen_train_eval` (GLU p_off, U-Net p_off/p_on) con il percorso completo funzionante.

**Cosa NON è stato fatto.** Nessuno smoke GPU, nessuna verifica di identità sul cluster, nessuna sottomissione: dall'ambiente di lavoro non esiste un socket SSH attivo verso Engaging (autenticazione interattiva password + Duo non eseguibile dalla sessione). Nessun risultato su checkpoint reali esiste. Nessun TEST, VAL non aperto, nessun training.

**Prossimo passo.** Aprire la connessione SSH multiplexata (`ssh engaging`) e seguire `docs/nonideal/02_block2_runbook.md`: smoke sui quattro percorsi, poi campagna.
