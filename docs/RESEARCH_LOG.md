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
