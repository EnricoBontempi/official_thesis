# Diario della ricerca

Questo file è il diario autorevole.
Aggiungere voci cronologiche con data/fuso, autore/tool, provenienza
(eseguito / ispezionato / riferito dall'utente), motivazione, evidenze collegate,
risultati e limiti, ipotesi, decisioni, problemi e prossimi passi.
Correggere con nuove voci datate che citano quella precedente, senza cancellarla.

Il diario riguarda esclusivamente il progetto di tesi in `official_thesis`:
ricostruzione della temperatura da osservazioni sparse in flussi reagenti.
Il progetto HONF e le relative indagini di debugging appartengono a
`thesis-research`, dove restano il loro diario e le loro evidenze.
Non duplicare qui la cronologia dell'altro progetto.

Registrare le date degli eventi, distinguendole dalle date di scrittura delle
voci. Quando una data o un esito non sono verificabili, dichiararlo.
Distinguere proposte, implementazioni, esperimenti eseguiti, risultati verificati
e decisioni approvate dai supervisori.

---

> **Nota sulla ricostruzione retrospettiva (scritta il `2026-11-16`, fuso `America/New_York` / `UTC-05:00`):**  
> Le voci seguenti relative al periodo `2026-11-13` – `2026-11-16` sono state ricostruite retrospettivamente il `2026-11-16` a partire dalla cronologia dei commit di `official_thesis`, dagli artefatti salvati nel repository (`docs/`, `configs/`, `scripts/`, `src/`, `results/`, `overleaf_month1_report/`) e dalla conversazione di progetto con l'utente. Per ciascuna voce sono distinti la data dell'evento, il timestamp del commit Git (quando presente) e la data di questa annotazione retrospettiva.

---

## 2026-11-13 (America/New_York, UTC-05:00) — Invio del thesis proposal, inizializzazione del repository e roadmap del Mese 1

- **Data evento:** Venerdì `2026-11-13` (invio proposta in giornata; commit iniziali alle `17:33:29 -0500` e `17:58:30 -0500`).
- **Data registrazione nel diario:** `2026-11-16` (ricostruzione retrospettiva).
- **Autore/tool:** Utente (Enrico Bontempi) e Antigravity (Gemini).

### Riferito dall'utente

- L'utente ha inviato ufficialmente il documento di proposta della tesi (*"Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion"*) alla Prof.ssa Sili Deng venerdì `2026-11-13` (ora esatta dell'invio email non registrata nel repository).
- Come riferito dall'utente il `2026-11-14`, la Prof.ssa Deng ha risposto positivamente sull'impianto generale esprimendo però una cautela specifica sulla predizione istantanea della temperatura: *"Thank you, Enrico. The plan sounds good. I am a bit skeptical on the instantaneous temperature prediction and would love to see updates when they are available."*
- Tale risposta costituisce un riscontro preliminare sul piano di lavoro; **non** costituisce approvazione finale delle scelte tecniche di split, metriche o architetture del Mese 1–2, che restano in attesa di revisione da parte dei supervisori.

### Motivazione e problema affrontato

- Impostare un repository separato e ordinato (`EnricoBontempi/official_thesis`), distinto dal precedente progetto esplorativo HONF in `thesis-research`, dedicato esclusivamente alla tesi ufficiale sulla ricostruzione spaziale istantanea del campo di temperatura $T$ da osservazioni sparse di velocità $\mathbf{u}$ (con o senza parametri operativi $(\text{NH}_3\%, \phi)$ e ancore sparse di temperatura) sul dataset di combustione turbolenta swirl-stabilizzata $\text{NH}_3/\text{CH}_4/\text{air}$ di *RealPDEBench*.
- Tradurre gli obiettivi del Mese 1 del proposal in una sequenza operativa verificabile (Steps `1.1`–`1.6`).

### Eseguito e artefatti collegati

- Commit `75c5db2` (`2026-11-13 17:33:29 -0500`): inizializzazione del repository con `docs/proposal.md`, `README.md` e `.gitignore` (configurato per escludere dal versionamento Git i file raw pesanti `.arrow`, `.h5`, `.npy`, `.pt`).
- Commit `7dff7d6` (`2026-11-13 17:58:30 -0500`): stesura di `roadmap_month1.md`, articolata in sei passi:
  - `Step 1.1`: scaffolding del pacchetto Python `combustion_recon`, configurazioni e test;
  - `Step 1.2`: ricognizione remota di `AI4Science-WestlakeU/RealPDEBench` su Hugging Face senza scaricare l'intero archivio da ~729 GB;
  - `Step 1.3`: download selettivo e audit fisico/statistico delle traiettorie di combustione numerica LES ($128 \times 128$);
  - `Step 1.4`: blocco degli split per condizione operativa (`train`, `val`, `test_interp`, `test_extrap`), normalizzazione solo su `train` e maschere di osservazione;
  - `Step 1.5`: check di fattibilità leggeri su CPU e baseline non neurali;
  - `Step 1.6`: allineamento con la letteratura di riferimento (`GLU`, `RealPDEBench`, `RF-PINNs`) e sintesi per i supervisori.

### Decisioni e prossimi passi

- Separare rigorosamente il task `temperature_free` (sola velocità sparsa $\mathbf{u} \rightarrow T$, con o senza parametri operativi) dal task `temperature_assisted` ($\mathbf{u}$ sparsa + $K$ sensori sparsi di temperatura $T \rightarrow T$).
- Imporre tre vincoli metodologici fin dallo scaffolding: (1) nessuna condivisione di frame della stessa condizione operativa tra `train` e `val`/`test`; (2) nessuna confusione tra la frazione molare planare LES di $\text{OH}$ (canale 8) e il canale `observed` di chemiluminescenza $\text{OH}^*$ integrata lungo la linea di vista; (3) nessuna rappresentazione dei punti non osservati come zeri fisici.

---

## 2026-11-13 (America/New_York, UTC-05:00) — Step 1.1, Step 1.2 (ricognizione remota di RealPDEBench) e Step 1.3 pilota su 3 traiettorie

- **Data evento:** `2026-11-13` (artefatti generati tra le `18:23` e le `18:56 -0500`; inclusi nel successivo commit cumulativo `567b76f` del `2026-11-14 14:17:08 -0500`).
- **Data registrazione nel diario:** `2026-11-16` (ricostruzione retrospettiva).
- **Autore/tool:** Cursor Agent (esecuzione codice e download su prompt preparati da Antigravity) e Antigravity (ispezione artefatti).

### Problema e motivazione

- Prima di scaricare decine di gigabyte, verificare la struttura esatta del sottoinsieme `combustion/` su Hugging Face (`AI4Science-WestlakeU/RealPDEBench`), la corrispondenza tra i file `.arrow` e i nomi delle simulazioni `sim_id`, gli split ufficiali del benchmark e la fattibilità dello storage locale.

### Eseguito e ispezionato

1. **Scaffolding (`Step 1.1`) e ricognizione remota (`Step 1.2`):**
   - Eseguito `scripts/01_inspect_remote_dataset.py`; prodotti `docs/month1/01_remote_reconnaissance.md`, `data/manifests/remote_file_inventory.json` e copie dei metadati JSON leggeri in `data/manifests/hf_small_metadata/`.
   - Verificato lo spazio disco locale (~880 GB liberi su `/dev/sdd`) e l'ambiente `.venv` (Python 3.13.13, PyTorch 2.15.0, esecuzione CPU-only in questo ambiente WSL poiché `torch.cuda.is_available() == False`).
   - Accertato che il sottoinsieme `combustion/` pesa complessivamente `62.292 GB`, di cui **`58.626 GB`** per i **30 shard `.arrow` numerici** (`combustion/hf_dataset/numerical/data-00000-of-00030.arrow` … `data-00029`, ~1.954 GB ciascuno, 1 traiettoria per shard) e `3.666 GB` per gli 8 shard sperimentali `real` ($\text{OH}^*$).
   - **Risultato critico sugli split ufficiali di RealPDEBench:** nei file ufficiali del benchmark, `val_index_numerical.json` e `test_index_numerical.json` sono **vuoti (`[]`)** e tutte le 30 traiettorie numeriche sono inserite in `train_index_numerical.json` (`remain_params_numerical.json` contiene 30/30 traiettorie), poiché RealPDEBench usa l'intero set numerico a risoluzione dimezzata (`sub_s = 2`, $64 \times 64$) per pre-allenare modelli di forecasting temporale prima del fine-tuning su $\text{OH}^*$. È quindi indispensabile definire nella nostra tesi uno split dedicato per condizione operativa a piena risoluzione $128 \times 128$ (`sub_s = 1`).
2. **Audit pilota su 3 shard (`Step 1.3 pilota`):**
   - Eseguito `scripts/02_audit_pilot_trajectories.py` scaricando inizialmente 3 shard rappresentativi (`0NH3_1.h5`, `40NH3_1.h5`, `80NH3_1.05.h5`, ~5.9 GB totali) e generando `docs/month1/02_pilot_data_audit_report.md`, `data/manifests/pilot_audit_summary.json` e `data/manifests/pilot_shard_map.json`.
   - Confermato che l'ordine dei 30 shard `.arrow` segue l'ordine lessicografico dei `sim_id`, che ogni traiettoria contiene $2001$ frame a $\Delta t = 2.5 \times 10^{-4}\text{ s}$ ($4000\text{ Hz}$, durata $0.5\text{ s}$) su griglia $128 \times 128$ ($x \in [-0.0369, +0.0369]\text{ m}$, $y \in [0.0629, 0.1591]\text{ m}$) con 15 canali fisici (`Ch 10` = `Temperature` in Kelvin; `Ch 11–13` = `Velocity[i, j, k]` in m/s; `Ch 14` = `Velocity_Magnitude`).

### Limiti ed errore di interpretazione iniziale (corretto il 2026-11-14)

- Nel report pilota `docs/month1/02_pilot_data_audit_report.md` (§2–3) era stata ipotizzata un'orientazione geometrica errata del bruciatore (assumendo che il getto entrasse dal basso a $y_{\min}$ verso l'alto $+y$). Tale ipotesi è stata smentita e corretta il giorno successivo (`2026-11-14`) dall'ispezione visiva delle mappe 2D in `docs/month1/02_data_audit_report.md`.

---

## 2026-11-14 (America/New_York, UTC-05:00) — Completamento download e audit delle 30 condizioni (Step 1.3), protocollo di split (Step 1.4), prima esecuzione delle baseline (Step 1.5 v1, poi corretta) e note preliminari su GLU (Step 1.6 v1)

- **Data evento:** `2026-11-14` (`02:41` – `14:17 -0500`, commit `567b76f` alle `14:17:08 -0500`).
- **Data registrazione nel diario:** `2026-11-16` (ricostruzione retrospettiva).
- **Autore/tool:** Cursor Agent (esecuzione download, audit, split e baseline v1) e Antigravity (ispezione figure `fig01`–`fig06`, redazione iniziale di `05_literature_and_glu_notes.md` e `06_supervisor_review_brief.md`).

### Eseguito e ispezionato

1. **Step 1.3 completo e correzione dell'orientazione fisica (`docs/month1/02_data_audit_report.md`, che sostituisce il report pilota):**
   - Scaricati e verificati tramite SHA256 tutti i 30 shard numerici (~58.6 GB) in `data/raw/combustion/hf_dataset/numerical/`. Eseguito `scripts/02_audit_trajectories.py`, producendo `data/manifests/operating_conditions.yaml`, `data/manifests/full30_audit_summary.json`, `results/tables/month1_30_conditions_inventory.csv` e le figure `results/figures/month1_audit/fig01`–`fig05`.
   - **Correzione geometrica verificata sulle figure `fig02`–`fig04`:** la fetta 2D $(128 \times 128)$ è una sezione diametrale simmetrica attorno alla linea orizzontale centrale (**`row 64`**, $y \approx 0.111\text{ m}$). I due getti anulari freddi di reagenti incombusti ($T \sim 250\text{--}400\text{ K}$) entrano dal **bordo destro (`col 127`**, $x \approx +0.037\text{ m}$) e divergono a V verso sinistra, racchiudendo una zona centrale calda di ricircolo interno (**IRZ**, $T \sim 1900\text{--}2280\text{ K}$). `Velocity[j]` (`Ch 12`) è la componente assiale principale ($+25\text{ m/s}$ lungo i rami a V, $-10\text{ m/s}$ nella IRZ); `Velocity[i]` (`Ch 11`) è la componente radiale/trasversale nel piano (antisimmetrica rispetto a `row 64`); `Velocity[k]` (`Ch 13`) è la componente di swirl azimutale fuori dal piano.
   - **Anomalie dei dati documentate:** (a) il canale esportato `Ch 14` (`Velocity_Magnitude`) presenta artefatti numerici di interpolazione con valori negativi fino a $-8\text{ m/s}$, per cui nel nostro codice $|\mathbf{u}| = \sqrt{v_i^2 + v_j^2 + v_k^2}$ viene sempre ricalcolato da `Ch 11–13`; (b) il canale `Ch 10` (`Temperature`) presenta rari outlier di interpolazione fino a $-375\text{ K}$ e $3002\text{ K}$; (c) l'autocorrelazione temporale nel *flame brush* resta elevata per $50\text{--}100$ frame ($\tau_c \approx 14\text{ ms}$, `fig04`).
2. **Step 1.4 — Blocco degli split e del protocollo di valutazione (`docs/month1/03_validation_and_splits_protocol.md`, `scripts/03_lock_splits_and_stats.py`):**
   - Bloccata la partizione delle 30 condizioni in `configs/splits/splits_interp_extrap.yaml` (`fig06_locked_operating_condition_splits.png`):
     - `train` (20 condizioni): $\text{NH}_3 \in \{0, 20, 40, 80\}\%$, $\phi \in [0.85, 1.25]$;
     - `val` (3 condizioni): `0NH3_1.05.h5`, `20NH3_1.1.h5`, `40NH3_0.9.h5`;
     - `test_interp` (3 condizioni interne all'inviluppo convesso di training): `20NH3_1.h5`, `40NH3_1.15.h5`, `60NH3_0.95.h5` (unica condizione al 60% $\text{NH}_3$ presente nel dataset);
     - `test_extrap` (4 condizioni di frontiera esterne al range di training): `20NH3_0.75.h5` (più magra), `0NH3_1.3.h5`, `20NH3_1.3.h5` (più ricche), `80NH3_1.25.h5` (angolo ad alto $\text{NH}_3$ e ricco).
   - Calcolate le statistiche di normalizzazione solo sulle 20 condizioni di `train` (`data/manifests/train_norm_stats.json`, $\mu_T = 2039.3\text{ K}, \sigma_T = 331.6\text{ K}$) e definite le regioni diagnostiche in `configs/splits/evaluation_protocol.yaml` ($\Omega_{\text{high}}: T \ge 2100\text{ K}$, $\Omega_{\text{peak}}: T \ge 2200\text{ K}$, $\Omega_{\text{jet}}: T < 1800\text{ K}$) e i layout di osservazione in `configs/masks/observation_layouts.yaml`.
3. **Step 1.5 (versione iniziale v1, successivamente corretta) e Step 1.6 (versione iniziale v1, successivamente corretta) — Commit `567b76f`:**
   - Eseguita una prima versione di `scripts/04_feasibility_checks.py` (`GlobalTrainMean`, `ParamConditionedMean`, `LocalHGB`, `GappyPOD`) e redatte le prime versioni di `docs/month1/04_feasibility_check_report.md`, `docs/month1/05_literature_and_glu_notes.md` e `docs/month1/06_supervisor_review_brief.md`.

### Problemi ed errori nella versione `567b76f` (diagnosticati e corretti nelle voci successive)

- **Bug di regolarizzazione nelle baseline di `567b76f`:** nella prima implementazione di `GappyPODReconstructor` e `ParamConditionedMeanPredictor` (`src/combustion_recon/models/baselines.py`), una penalità Ridge fissa `1e-2` non scalata rispetto agli autovalori dei predittori e alla norma unitaria dei modi POD (distribuita su $65.536$ entrate) sovra-regolarizzava i modi, annullava artificialmente il contributo delle $K \in \{1\dots 32\}$ ancore di temperatura in `GappyPOD` e schiacciava la curvatura quadratica in $\phi$ di `ParamConditionedMeanPredictor`. Le conclusioni numeriche di `567b76f` sull'inefficacia delle ancore $T$ e sul bias di $+88\text{ K}$ di `ParamMean` erano pertanto viziate da questo artefatto implementativo.
- **Imprecisioni su GLU in `05_literature_and_glu_notes.md` (v1):** non avendo ancora letto il PDF integrale di `arXiv:2603.26023`, Antigravity aveva descritto GLU assumendo token scalari con `variable_id`, fusione lineare globale-locale e campo di importanza dinamico.

---

## 2026-11-14 / 2026-11-15 (America/New_York, UTC-05:00) — Revisione multi-agente (Fable/Claude e Sol/ChatGPT), correzione dei bug nelle baseline e cambio di interpretazione scientifica

- **Data evento:** Dal pomeriggio del `2026-11-14` (`15:15 -0500`, commit `659815a`) alla notte del `2026-11-15` (`01:36 -0500`, commit `4fdbcb1`, seguito dalla verifica di Fable su `4fdbcb1` intorno alle `01:43 -0500`).
- **Data registrazione nel diario:** `2026-11-16` (ricostruzione retrospettiva).
- **Autore/tool:** Revisori esterni indipendenti Fable (Claude) e Sol (ChatGPT) interrogati dall'utente con accesso al PDF di GLU e al repository; Antigravity (sintesi in `analisi_agents.md` e coordinamento); Cursor Agent (esecuzione dei due pass di correzione sul codice e sui CSV).

### Problema e motivazione

- Verificare in modo indipendente la correttezza matematica delle baseline del Mese 1, la fedeltà della descrizione del paper GLU (`arXiv:2603.26023`) e la solidità delle conclusioni prima di preparare il documento per i supervisori.

### Ispezionato ed eseguito (in tre iterazioni)

1. **Iterazione 1 — Revisione di Fable e Sol sul PDF di GLU e sul codice pre-fix (`analisi_agents.md`, commit `659815a` del `2026-11-14 15:15:08 -0500`):**
   - **Correzioni sul paper GLU (`arXiv:2603.26023`, autori: Linzheng Wang, Jason Chen, Nicolas Tricard, Zituo Chen, Sili Deng):**
     - Nel paper GLU originale, i sensori osservano vettori co-locati $\mathbf{u}_i \in \mathbb{R}^c$ (non esiste `variable_id` nel paper originale);
     - L'encoder usa $S$ query latenti apprendibili $L \in \mathbb{R}^{S \times D}$ con doppio passaggio di Cross-Attention ($L' = \text{CrossAttn}(L, H, H)$ e $Z_{\text{local}} = \text{CrossAttn}(H, L', L') + H$), mantenendo tutti gli $N$ token dei sensori più un singolo token globale $\mathbf{z}_{\text{global}}$;
     - Il campo di importanza $\varphi(\mathbf{x}) \sim \text{Beta}(\alpha(\mathbf{x}), \beta(\mathbf{x}))$ dipende solo dalle coordinate spaziali $\mathbf{x}$ (statico) e deforma la distanza nel recupero k-NN del decoder tramite $d_\varphi(\mathbf{y}_j, \mathbf{x}_i) = \|\mathbf{y}_j - \mathbf{x}_i\|_2 / (\varphi_i^\gamma + \epsilon)$;
     - Il benchmark di combustione nel paper GLU riguarda **una sola condizione operativa fissa** ($\text{CH}_4/\text{air}$ stechiometrico su *backward-facing step* in OpenFOAM) in cui l'$1\%$ dei sensori misura **tutti i canali contemporaneamente, inclusa la temperatura $T$**. GLU non ha quindi mai testato né l'inferenza cross-variabile ($\mathbf{u} \rightarrow T$ senza sensori di $T$) né la generalizzazione tra condizioni operative/miscele diverse.
   - **Diagnosi del bug n. 1 (`GappyPODReconstructor`):** Fable e Sol hanno individuato che i modi POD avevano norma unitaria sull'intero vettore di $65.536$ componenti ($128 \times 128 \times 4$), per cui $K \le 32$ ancore di temperatura producevano un termine $A_T^\top A_T \sim K / 65536 \le 5 \times 10^{-4}$, venti volte più piccolo di `ridge_lambda = 1e-2`, venendo azzerate dalla regolarizzazione. Segnalati inoltre l'uso di `hash()` non deterministico per i seed delle maschere, un buffer temporale di soli 25 frame su `seen_train_eval` (inferiore a $\tau_c$) e la necessità di scomporre $\text{RMSE}^2 = \text{Bias}^2 + \text{Centered-RMSE}^2$.
2. **Iterazione 2 — Primo fix su `GappyPOD` e scoperta del bug n. 2 su `ParamConditionedMeanPredictor` (`2026-11-15 00:00` – `01:36 -0500`, commit `4fdbcb1`):**
   - Dopo un primo fix locale su `GappyPOD` (regolarizzazione scalata con le varianze modali $\sigma_r^2$, pesi normalizzati $1/M_u$ vs $w_T/K$, aggiunta della variante `Anchor Offset (+AO)` su ancore $K$ annidate, seed deterministici `zlib.crc32` e buffer di 120 frame tra `train_fit` $[0, 1480]$ e `seen_train_eval` $[1600, 2000]$), un secondo controllo di Fable ha individuato che **`ParamConditionedMeanPredictor` soffriva dello stesso difetto di regolarizzazione**: `ridge_alpha = 1e-2` applicato a feature polinomiali non centrate `[1, n, p, n^2, p^2, n*p]` (inclusa l'intercetta) abbatteva del 94% l'autovalore della curvatura in $\phi$ ($6 \times 10^{-4}$), generando artificialmente circa due terzi del bias di $+88\text{ K}$ in estrapolazione e di $-35/-40\text{ K}$ su `val`/`test_interp` (bias poi ereditato da `GappyPOD_Conditioned`).
   - **Eseguito nel commit `4fdbcb1` (`2026-11-15 01:36:02 -0500`):**
     - Corretto `ParamConditionedMeanPredictor` (`src/combustion_recon/models/baselines.py`) centrando le feature attorno a $(n_0 = 0.30, \phi_0 = 1.0)$, escludendo l'intercetta dalla penalizzazione e selezionando automaticamente base e `ridge_alpha` tramite **Leave-One-Condition-Out CV (LOCO-CV)** sulle 20 condizioni di training (selezionata `piecewise_phi1`, lineare a tratti attorno a $\phi=1.0$, con LOCO RMSE $= 71.96\text{ K}$).
     - Aggiunto il metodo `OraclePOD` (proiezione diretta della $T$ vera sui $R=32$ modi POD di temperatura) per misurare il limite di rappresentazione della base lineare.
     - Rimosso l'ultimo `hash()` residuo in `scripts/03_lock_splits_and_stats.py`, uniformati i frame di valutazione tra tutte le tabelle, centrata la scala di $\nabla_{\text{2D}} \cdot \mathbf{u}$ in `fig07`, uniformata per riga la scala colori di temperatura in `fig10`, generato `results/tables/month1_per_condition_summary.csv` e superati `27/27 pytest`.
3. **Iterazione 3 — Verifica di Fable sul commit `4fdbcb1` e revisione delle conclusioni fisiche (`2026-11-15 ~01:43 -0500`):**
   - Fable ha verificato i CSV e il codice di `4fdbcb1`, confermando l'assenza di ulteriori bug invalidanti e la corrispondenza dei numeri (`ParamMean` replicato entro $1\text{ K}$, `+AO` conforme alla legge teorica della varianza), ma ha evidenziato tre punti cruciali per l'interpretazione:
     1. **Il collo di bottiglia principale è il gap di stima ($\mathbf{u} \rightarrow T$), non la capacità della base POD:** `OraclePOD` ($R=32$) ricostruisce il campo con un `Centered-RMSE` di soli $53.5\text{ K}$ (`seen`), $35.7\text{ K}$ (`val`), $46.4\text{ K}$ (`test_interp`) e $74.6\text{ K}$ (`test_extrap`), catturando il $78\text{--}82\%$ dell'energia delle fluttuazioni attorno a `ParamMean`. Invece `GappyPOD_Conditioned` (stimato dal $5\%$ di velocità) si ferma a $105.9 / 66.7 / 85.5 / 138.5\text{ K}$ di `Centered-RMSE` (recuperando solo il $24\text{--}38\%$ della varianza delle fluttuazioni).
     2. **L'errore di forma (`Centered-RMSE`) scala con $\phi$ in tutti gli split, non solo in estrapolazione:** attraverso tutte le 30 condizioni, `Centered-RMSE` correla fortemente con $\phi$ ($r = 0.73$). Confrontando le 4 condizioni di `test_extrap` con i rispettivi vicini più prossimi nel training set (`results/tables/month1_per_condition_summary.csv`), il `Centered-RMSE` di `GappyPOD_Conditioned` differisce tra $-8\%$ e $+31\%$ (`20NH3_0.75`: $62.0$ vs $47.4\text{ K}$; `0NH3_1.3`: $177.4$ vs $151.3\text{ K}$; `20NH3_1.3`: $167.9$ vs $147.8\text{ K}$; `80NH3_1.25`: $146.9$ vs $160.0\text{ K}$).
     3. **Effetto di $K$ termocoppie (`+AO` vs `+T-block`):** con $K=1\text{--}4$ ancore, l'offset medio (`+AO`) può peggiorare l'RMSE per frame ($153.4 \rightarrow 158.8\text{ K}$ a $K=1$ in `test_extrap`) per via della varianza locale del *flame brush* ($\sigma_c \approx 138.5\text{ K}$), riducendo stabilmente $|\text{Bias}|$ solo per $K \ge 9\text{--}32$ ($43.0 \rightarrow 33.1 \rightarrow 18.5\text{ K}$). Il solve modale congiunto (`+T-block`) riduce anche l'errore di forma a $K=32$ (`Cond + T-block` `Centered-RMSE` $138.5 \rightarrow 119.3\text{ K}$; `Uncond + T-block` RMSE $182.7 \rightarrow 125.5\text{ K}$), ma non produce miglioramenti strutturali per $K \le 9$.
     4. **Metrica `top25pct_iou` accantonata:** nel codice `metrics.py`, `top25pct_iou` sogliava il campo predetto usando il 75° percentile della temperatura *vera*, risultando quindi sensibile al bias di livello; la metrica non viene usata nelle conclusioni per i supervisori.

### Limiti noti delle baseline del Mese 1 (dichiarati e tracciati)

- `LocalHGB` è addestrato su un sottocampione di $60.000$ pixel con patch $3 \times 3$ (baseline locale veloce su CPU, non upper bound non-lineare).
- Gli iperparametri $(\lambda = 10^{-2}, w_T = 5)$ selezionati per `GappyPOD` su `val` toccano gli estremi della griglia grossolana esplorata nello script `04`.
- La normalizzazione $1/M_u$ nell'obiettivo di `GappyPOD` rende la stima lineare quasi invariante alla densità di campionamento della velocità tra $1\%$ e $20\%$.
- Il livello medio predetto in estrapolazione da `ParamConditionedMean` varia tra $-36\text{ K}$ e $+43\text{ K}$ a seconda della forma funzionale scelta (quadratica vs lineare a tratti attorno a $\phi=1.0$).

---

## 2026-11-15 (America/New_York, UTC-05:00) — Stesura, verifica e rifinitura del report LaTeX del Mese 1 per i supervisori

- **Data evento:** `2026-11-15` (`01:50` – `02:12 -0500`, commit `324757b`, `a72459f`, `3f7c0e2` in `official_thesis` e commit `17dd846` in `documents_thesis`).
- **Data registrazione nel diario:** `2026-11-16` (ricostruzione retrospettiva).
- **Autore/tool:** Antigravity (stesura LaTeX e compilazione `pdflatex`/`bibtex`), Fable/Claude (doppio controllo riga per riga contro i CSV) e Utente (sincronizzazione Overleaf).

### Motivazione

- Consegnare ai supervisori un report chiaro, rigoroso e leggibile in formato LaTeX/PDF (evitando elenchi di test software o gergo interno), completo delle 10 figure del Mese 1, delle tabelle verificate dai CSV e di una proposta trasparente per i passi del Mese 2–3 con relativi vantaggi, limiti e domande aperte.

### Eseguito e verificato

- Creata la cartella `overleaf_month1_report/` (`main.tex`, `references.bib`, `figures/fig01`–`fig10`, `main.pdf`) in `official_thesis` (commit `324757b`) e sincronizzata nella repository pubblica dedicata `EnricoBontempi/documents_thesis` per l'importazione su Overleaf.
- Sottoposto `main.tex` a due passaggi di verifica puntuale con Fable (commit `a72459f` alle `02:06:31 -0500` e commit `3f7c0e2` alle `02:12:13 -0500`):
  - Corretta in `references.bib` e `docs/month1/05_literature_and_glu_notes.md` la lista autori di GLU (`Linzheng Wang, Jason Chen, Nicolas Tricard, Zituo Chen, Sili Deng`, eliminando l'errore di trascrizione "Lichuan/Chuan" commesso in precedenza da Antigravity);
  - Verificata la corrispondenza esatta di tutte le celle delle Tabelle 1, 2 e 3 di `main.tex` con `results/tables/month1_feasibility_split_summary.csv` e `results/tables/month1_per_condition_summary.csv`;
  - Riportata l'equazione esatta della varianza di `+AO` con $\tau = 1$: $\text{RMSE}^2(K) \approx \sigma_c^2 (1 + K/(K+\tau)^2) + (\tau/(K+\tau))^2 \text{Bias}^2$;
  - Articolata nella Sezione 4 di `main.tex` la progressione di esperimenti proposta per il Mese 2–3 (senza nomi promozionali):
    - **Step 0 (Empirical Information Ceiling Diagnostic):** U-Net condizionata con FiLM da velocità planare densa ($100\%$) $(v_i, v_j) \rightarrow T$, confrontando un singolo frame istantaneo con una breve finestra temporale di 3 frame ($[t-2, t-1, t]$ o $[t-1, t, t+1]$) per stimare empiricamente il limite informativo della velocità prima di introdurre la sparsità;
    - **Step 1 (Temperature-Free Sparse Reconstruction):** adattamento diretto di GLU ai token di velocità sparsa ($1\%, 5\%, 20\%$) con embedding dei valori che preserva la scala $|\mathbf{u}|$, condizionamento FiLM su $(\text{NH}_3\%, \phi)$ e campo di importanza condizionato ai parametri $\varphi(\mathbf{x}, \mathbf{p})$, confrontato contro una U-Net con canale di maschera binaria esplicita e contro `GappyPOD`;
    - **Step 2 (Two-Stage Residual Assimilation per $K > 0$):** modulo di cross-attention residuo sugli scarti $e_k = T(\mathbf{x}_k) - \hat{T}_0(\mathbf{x}_k)$ alle $K$ termocoppie sopra la predizione Stage-1 $\hat{T}_0(\mathbf{x})$, confrontato contro `+AO` con shrinkage empirico-Bayesiano e Kriging/GPR dei residui;
    - **Step 3 (Estensione metodologica per il Mese 3):** predizione di una base di correzione a basso rango dipendente dallo snapshot $\mathbf{b}_\theta(\mathbf{x} \mid \mathcal{O}_u, \mathbf{p}) \in \mathbb{R}^r$ (inclusi modi linearizzati di spostamento del fronte $\nabla m_\theta \cdot \boldsymbol{\psi}_i$) aggiornata in forma chiusa sulle $K$ termocoppie per testare se una base condizionata alla velocità istantanea renda efficaci piccoli numeri di ancore ($K = 4\text{--}9$).

### Stato decisionale

- Gli Steps 0–3 descritti nel report sono **proposte progettuali per il Mese 2–3**, non implementazioni completate né decisioni già approvate dai supervisori.

---

## 2026-11-16 (America/New_York, UTC-05:00) — Ricezione del codice di riferimento di GLU da Jason Chen e acquisizione in `external/GLU_Reconstruction/`

- **Data evento:** Lunedì `2026-11-16` (comunicazione dell'utente alle `09:22 -0500`; commit `ebaae9d` alle `09:32:29 -0500`; creazione diario vuoto commit `ce30053` alle `09:49:11 -0500`).
- **Data registrazione nel diario:** `2026-11-16 10:05 -0500`.
- **Autore/tool:** Utente (riferito contatto con Jason Chen) e Antigravity (clonazione e ispezione preliminare del codice).

### Riferito dall'utente

- Alla mattina del `2026-11-16`, la Prof.ssa Sili Deng e Linzheng Wang non hanno ancora risposto al report del Mese 1.
- Ha invece risposto **Jason Chen** (PhD nel gruppo Deng e secondo autore del paper GLU), fornendo all'utente il link al repository GitHub del codice di GLU: `https://github.com/cosmos2w/GLU_Reconstruction`.

### Eseguito

- Commit `ebaae9d` (`2026-11-16 09:32:29 -0500`): clonato il repository `https://github.com/cosmos2w/GLU_Reconstruction` nella directory `external/GLU_Reconstruction/` all'interno di `official_thesis` (rimuovendo la sottocartella `.git` interna per mantenerlo tracciato come albero statico di riferimento) e aggiornato `README.md` per specificare che `external/GLU_Reconstruction/` costituisce la baseline originale del laboratorio da mantenere intatta, mentre ogni sviluppo e adattamento della tesi risiederà in `src/combustion_recon/`.

### Ispezionato in `external/GLU_Reconstruction/`

Dall'ispezione di `external/GLU_Reconstruction/README.md`, `training_procedure_pseudocode.tex`, `configs/collinear_re40_stage0.yaml` e `src/models.py` emergono quattro elementi tecnici verificati direttamente sul codice degli autori:
1. **Struttura a due stadi disaccoppiati (`Stage 0` vs `Stage 1`):** quando `Stage: 0` (ricostruzione spaziale), il wrapper `GLU_Bay_DD` (`src/models.py`, righe 2225–2228) congela e salta interamente il propagatore temporale (`latent_traj = G_obs`), usando esclusivamente `DomainAdaptiveEncoder` (`righe 261–391`), il modulo di importanza spaziale `phi_mlp_1` (`righe 2150–2196`) e il decoder `SoftDomainAdaptiveReconstructor` (`righe 1644–2105`).
2. **Supporto nativo per osservazioni parziali tra canali e segnaposto per i parametri operativi `U`:**
   - `DomainAdaptiveEncoder` include già un argomento `channel_to_encode: Optional[List[int]]` (`righe 274–291, 331–338`; impostato in `collinear_re40_stage0.yaml` alla riga 28 con il commento `# Can set for partial observation situation`) che seleziona un sottoinsieme di canali in ingresso all'encoder mentre il decoder predice `N_channels` canali di output.
   - La funzione `DomainAdaptiveEncoder.forward(self, coords_tuv, U, original_phi)` (`riga 321`) riceve già il tensore `U` delle condizioni operative (`optional conditions: [B, N_para]` in `README.md`), marcato dagli autori come `# (unused here)` perché i benchmark del paper erano a singola condizione operativa.
3. **Normalizzazione dei valori e fusione globale-locale nel codice rilasciato:**
   - L'embedding dei valori usa `nn.Linear(N_channels, All_dim)` seguito da `nn.RMSNorm(All_dim)` (`build_pos_value_proj`, `righe 229–238`), anziché `LayerNorm` senza bias.
   - La fusione tra la rappresentazione locale aggregata dai $k$ sensori vicini e il token globale `cls_token` avviene tramite il modulo `SpatiallyAwareCLSFusion` (`righe 122–157`), che calcola un gate sigmoidale dipendente dal punto query `gate = self.gate_net(local_features)` e combina `gate * local_proj + (1 - gate) * cls_proj`.
4. **Chiarimento sulla loss della testa di incertezza (`CalRecVar`):**
   - Il file `external/GLU_Reconstruction/training_procedure_pseudocode.tex` (`righe 23–35`) e `configs/collinear_re40_stage0.yaml` (`righe 67–80`) documentano che nello Stage-0 la varianza predetta $\log \hat{\boldsymbol{\sigma}}^2$ viene addestrata tramite Gaussian NLL eteroschedastica ($\mathcal{L}_{\mathrm{nll}} = \frac{1}{2}\langle (\hat{\mathbf{u}} - \mathbf{u})^2 / (\hat{\boldsymbol{\sigma}}^2 + \varepsilon) + \log \hat{\boldsymbol{\sigma}}^2 \rangle$) con peso `nll_weight: 1.0e-5` e annealing su `nll_anneal_epochs: 500`, mentre la distribuzione Beta $\varphi(\mathbf{x})$ viene supervisionata dalla varianza normalizzata tramite i pesi `bayesian_phi`.

### Decisioni, questioni aperte e prossimi passi

- **Nessun training o adattamento neurale del Mese 2 è stato ancora eseguito.** L'unica attività svolta finora per la transizione al Mese 2 è l'acquisizione e l'ispezione preliminare del codice di riferimento `external/GLU_Reconstruction/`.
- **Questioni ancora da chiarire / in attesa:**
  1. Risposta della Prof.ssa Sili Deng e di Linzheng Wang al report del Mese 1 (conferma dello split `20/3/3/4`, uso di velocità 2C $(v_i, v_j)$ come configurazione primaria, eventuale disponibilità delle portate massiche per condizione ed eventuale clipping degli outlier di interpolazione di $T$).
  2. Data/ora esatta dell'invio del proposal il `2026-11-13` e del successivo invio del report PDF del Mese 1 ai supervisori (da confermare con l'utente se necessario).
- **Prossimi passi concordati con l'utente:**
  1. Far ispezionare il codice `external/GLU_Reconstruction/` (`train.py`, `src/models.py`, `src/dataloading.py`) a un agente di revisione/implementazione ad alta capacità (Opus/Fable o Astra/Sol) per progettare nel dettaglio l'adattamento pulito in `src/combustion_recon/` senza alterare la copia di riferimento in `external/GLU_Reconstruction/`.
  2. Verificare la disponibilità di un ambiente con GPU NVIDIA attiva (es. cluster Engaging o workstation dedicata), dato che l'ambiente WSL locale attuale vede solo la CPU (`torch.cuda.is_available() == False`), prima di lanciare i training dello `Step 0` (U-Net densa) e dello `Step 1` (GLU adattato).
