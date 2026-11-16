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
