# Month 2 Detailed Roadmap — Verified Baselines & First Neural Temperature Reconstructor

**Progetto:** Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion  
**Autore:** Enrico Bontempi — MSc Thesis, Deng Lab, MIT  
**Preparata:** 2026-11-16, America/New_York (UTC−05:00)  
**Stato aggiornato 2026-11-17 (chiusura U7):** piano autorizzato operativamente da Enrico; 2.1–2.3 eseguiti. Consolidamento tecnico su `mission/baseline-consolidation` (`3cf7bf0`), integrazione in main ancora pendente. **Step 2.4 (diagnostico U-Net denso):** concluso nel perimetro dichiarato su `mission/dense-unet` (U0–U7; p_on/p_off seed 0 + controlli snapshot; repliche seed rinviate) — dettaglio [roadmap_unet.md](roadmap_unet.md) e report [docs/month2/03_dense_velocity_diagnostic.md](docs/month2/03_dense_velocity_diagnostic.md). Engaging GPU/AMF verificati e usati per U4–U6. **Mese 2 NON concluso** (2.5–2.9 aperti). Nessuna ratifica tecnica documentata dei supervisori; nessun merge su main.  
**Aggiornamento 2026-11-18 (pianificazione):** pubblicata [roadmapsparse.md](roadmapsparse.md), piano operativo approvato da Enrico per 2.5–2.6 e conferma sparse pertinente di 2.8. Nessun tetto arbitrario in ore; profiling/plateau/tranche e walltime scheduler distinti.

**Aggiornamento 2026-11-18 (STOP E):** missione sparse S1–S6 eseguita su Engaging tip `38ac7fe`; screening seed 0 + eval 1/5/20% + swap + Gappy nested gate-pass. Consegna revisori in `docs/month2/05_sparse_reconstruction.md` e `results/month2/s6_sparse_eval/STOP_E_delivery/`. S7 e scelta finalisti **non** avviati.  
**Fonti:** [proposal disponibile](docs/proposal.md), [Mese 1](roadmap_month1.md), [diario](docs/RESEARCH_LOG.md), [report LaTeX aggiornato](overleaf_month1_report/main.tex), [contesto verificato e decisioni](docs/month2/00_context_and_decisions.md).

> **Limite di provenienza:** `documents_thesis/main.tex` è il report del Mese 1 del 15 novembre, identico alla copia sopra, non il proposal inviato il 13 novembre. Non ci sono altri sorgenti inclusi tramite `input/include`; è stata letta la bibliografia. L'identità fra `docs/proposal.md` e l'allegato inviato non è verificata. La ricognizione e le differenze accertabili sono nel [registro](docs/month2/00_context_and_decisions.md#1-provenienza-e-limite-sul-documento-inviato). Non modificare il proposal inviato per adeguarlo al piano.

## Piano corrente di completamento — 20 novembre 2026

**Decisione approvata da Enrico:** tre blocchi sperimentali nell'ordine seguente, poi chiusura documentata. Questo piano prevale sui default storici incompatibili riportati sotto; non cancella esperimenti o decisioni precedenti. **Il presente aggiornamento è solo documentale: nessun job, training o TEST eseguito.**

| Ordine | Blocco e domanda | Matrice / azione | Stato |
| --- | --- | --- | --- |
| 1 | Confronto temperature-free: quale famiglia, quanto conta p e quanto varia il training? | C149=2980 frame, velocità TRAIN 5%, U-Net masked e GLU adattato × p_on/off × seed 0/1/2: **12 run complessivi, 10 nuovi training + 2 U-Net C149 seed 0 riusati**; eval 1/5/20% | S7 approvato, preparazione/submission nel prossimo incarico a Sonnet |
| 2 | Correzione del livello: quanto aggiungono K temperature alla stessa predizione congelata? | Sweep K=0/1/4/9/16/32, media regolarizzata, media senza shrink e mediana, baseline matched; velocità 5%, siti annidati congelati | Step 2.7, dopo conclusione e analisi del blocco 1; nessun nuovo training neurale |
| 3 | T realmente in ingresso: si migliora anche la ricostruzione spaziale oltre la correzione? | Training assisted sul backbone scelto; **se il confronto fra famiglie non giustifica una scelta chiara, su entrambe**. Identici K, siti, dati e osservazioni del blocco 2 | Step 2.7, direzione approvata; matrice/integrazione T/semi/distribuzione TRAIN-K da fissare prima dell'esecuzione |
| 4 | Chiusura Mese 2 | Confronti matched e costi, decisione separata su eventuali fold/pilot TEST, report, integrazione Git e confronto supervisori | Steps 2.8–2.9 aperti |

**S7 è dettagliato unicamente in [roadmapsparse.md](roadmapsparse.md#s7--confronto-c149-multi-seed-e-chiusura-sparse-decisione-20-novembre).** S7/STOP F chiude sparse temperature-free, non i blocchi assisted né l'intero Mese 2. Il conteggio dieci riguarda soltanto i nuovi training free; non è il numero complessivo degli esperimenti futuri.

**Fonti e stato delle sotto-missioni:** Opus O0–O5 consegnati e revisionati ([06](https://github.com/EnricoBontempi/official_thesis/blob/4f0e2637758de0766d0bb0891fe8ff97eee978c9/docs/month2/06_opus_data_and_training_variability.md)); O6 accettata come scelta C149/runner Opus per il seguito, con limiti di seed e VAL dichiarati. Astra primaria ([07](https://github.com/EnricoBontempi/official_thesis/blob/82bff35726fb70af9987081802ece3ef718e42bd/docs/month2/07_astra_level_structure_and_anchors.md)) e diagnostica median9 ([08](https://github.com/EnricoBontempi/official_thesis/blob/7cb7bfdc6430a2731c924a31d90897a3aa52a89f/docs/month2/08_astra_median9_second_pass_analysis.md)) concluse nel perimetro S5. Non si riaprono quelle missioni: le prove successive con T vivono qui. [Proposta Neon 09](https://github.com/EnricoBontempi/official_thesis/blob/9daf1585ba2bd9d264372e62565606a800fbefe5/docs/month2/09_astra_opus_synthesis_and_s7_proposal.md) conservata come documento consultivo, superata dalla decisione di includere anche sei training GLU C149 e di affrontare T soltanto dopo il confronto free. Codice, report e risultati sono ancora su branch distinti: questa pubblicazione non ne costituisce merge.

**Riuso e parità:** C149 mantiene normalizzatore fit-only storico 760 congelato, seen220 e VAL123, protocollo e stream Opus; i riferimenti S5 A38 restano separati. I comparatori principali a pari dati devono identificare/qualificare anche il fit C149 di POD/prior; quelli A38 restano storici. Nessuna modifica a split, target, soglie o dati TEST. Repliche di training e maschere eval restano distinte; VAL già usata in selezione e diagnostica non diventa una conferma indipendente.

**Fuori dall'incarico immediato:** training multidensità, nuove architetture/iperparametri GLU, rumore o termocoppie realistiche, finestre temporali, 3C, nuovi split e TEST. Scelta futura o risultato negativo non comportano campagne aggiuntive automatiche. La ratifica dei supervisori resta distinta dal completamento operativo.


## Scopo e punto di partenza

**Domanda operativa:** un modello non lineare ricava dalle osservazioni di velocità informazioni utili sulle fluttuazioni istantanee di T oltre la media parametrica e GappyPOD? Che cosa migliora aggiungendo poche misure T, separando errore di livello, struttura e cambio di condizione?

Il Mese 1 ha prodotto audit, split, normalizzatori, metriche e baseline corrette. Il diario e il report più recente identificano soprattutto un **gap di stima** fra GappyPOD e la proiezione oracle della T vera sulla base POD; non dimostrano impossibilità fisica della ricostruzione o incapacità generale delle basi lineari. Baseline e documenti sono disponibili; l'accordo tecnico con i supervisori richiesto dallo Step 1.6 è ancora aperto. È disponibile il codice GLU originale. Il diagnostico U-Net denso è stato implementato, allenato e chiuso in 2.4; i neurali sparse e l'adattamento GLU restano da eseguire secondo [roadmapsparse.md](roadmapsparse.md).

Il risultato minimo originario del Mese 2 è una pipeline riproducibile, baseline qualificate sullo stesso protocollo e un primo ricostruttore neurale con confronto temperature-free/assisted su condizioni viste ma temporalmente separate e su VAL. **Il 20 novembre Enrico amplia esplicitamente il lavoro di completamento al confronto C149 multi-seed di entrambe le famiglie e a una prova appresa con T in ingresso dopo lo sweep di correzione.** Un pilot controllato sulle condizioni escluse, dopo congelamento, prepara i mesi successivi; non anticipa un'intera campagna di generalizzazione. “Mese” significa fase: si procede appena i criteri sono soddisfatti, senza attese artificiali e senza spunte basate solo su codice scritto.

### Classi di attività e ordine

- **I — Preparabile subito:** infrastruttura, test, provenienza, audit della release e stima delle risorse, senza interpretare una proposta come approvata.
- **E — Esperimento:** richiede pipeline verificata, protocollo e budget registrati; si conclude con risultati, anche negativi, non soltanto con uno script.
- **S — Decisione scientifica:** registrata con data, fonte e decisore nel registro D1–D9; gli agenti non la dichiarano approvata per deduzione.

| Step | Classe | Risultato atteso | Dipendenza per procedere |
|---|---|---|---|
| 2.1 | I + S | Protocollo concreto e decisioni tracciate | Prima di run scientifici |
| 2.2 | I | Loader, frame, normalizzazione, osservazioni e valutatore sicuri | 2.1 tecnico; D2/D3/D5/D9 prima del freeze |
| 2.3 | I + E | Baseline comparabili e range di costo | 2.2, decisioni applicabili |
| 2.4 | E + S | Primo modello: U-Net densa single-frame | 2.3, D4/D8 |
| 2.5 | I + S | Piano preciso di adattamento della release GLU | In parallelo a 2.2–2.4; chiudere prima di 2.6 GLU |
| 2.6 | I + E | U-Net mascherata e GLU minimale da velocità sparsa | 2.4 e audit 2.5 |
| 2.7 | E + S | Confronto con poche ancore T e AO; modulo appreso solo se giustificato | 2.6, D5/D6 |
| 2.8 | E | Confronti matched e pilot di trasferimento congelato | Selezione su VAL/fold TRAIN conclusa |
| 2.9 | I + S | Report, configurazione selezionata e passaggio di fase | Evidenze e limiti di tutti gli step applicabili |

## Contratto scientifico comune a tutti gli step

### Input, target e significato fisico

- Target: **Temperature, canale 10, in K**, campo numerico LES 128×128 all'istante t. La T completa è disponibile per loss e valutazione; in `temperature_free` nessun suo valore, riassunto per frame o feature derivata entra nel modello.
- Input primario proposto: **Velocity[i], Velocity[j], canali 11 e 12**, vettori co-locati in m/s sulla sezione. Mantenerli identificati per canale: orientazione geometrica interpretata nei report, associazione esatta alle derivate fisiche da verificare prima di qualunque perdita PDE. Non assumere che i/j coincidano automaticamente con x/y dell'array.
- 3C aggiunge esplicitamente canale 13, modifica il budget informativo e resta un'ablazione. In 2C è ammessa solo la norma planare ricavata da 11/12; vietato introdurre vk attraverso velocità 3D, patch o altre feature. Non usare il canale 14 esportato come norma affidabile.
- Coordinate x/y in metri, con trasformazione deterministica documentata; parametri noti `nh3_fraction` in [0,1] e φ. Non passare `sim_id`, indice della condizione, timestamp assoluto o etichette di split come scorciatoie predittive. Metadata e frame restano nel registro dei risultati.
- `temperature_assisted`: stessi input di velocità più **solo K valori T** alle coordinate dichiarate. Per le correzioni deterministiche e per un modulo residuo con backbone congelato, K=0 restituisce esattamente il relativo free. Nel training con T in ingresso del piano 20 novembre, un backbone integralmente riaddestrato può avere un diverso risultato a K=0: conservarlo e confrontarlo esplicitamente con il checkpoint free originale, senza promettere identità tra pesi diversi. Le ancore sono osservazioni numeriche ideali, puntuali e sincronizzate; non simulano ancora risposta temporale, rumore o bias di una termocoppia reale.
- Nessun campo OH/OH*, specie, densità o heat release aggiunto implicitamente. Nessun forecasting, ricostruzione di campi medi in sostituzione del target istantaneo o cambio di dataset senza decisione esplicita.

### Split, frame e fit-only

- Conservare il file `configs/splits/splits_interp_extrap.yaml`: 20 TRAIN, 3 VAL, 3 TEST_INTERP, 4 TEST_EXTRAP, intere condizioni disgiunte. Verificare geometricamente l'inviluppo per le etichette interp/extrap, non soltanto min/max. Una traiettoria per condizione: la diagnostica seen è un blocco temporale separato della **stessa** traiettoria, non una replica indipendente.
- Prima tranche comune: frame `train_fit = 0:1481:40` (38/condizione, 760 totali), `seen_train_eval = 1600:2001:40` (11/condizione, 220 totali), VAL/TEST `0:2001:50` (41/condizione). Il divario tra ultimo frame fit 1480 e primo seen 1600 è 120 frame. Il sottocampionamento non rende automaticamente indipendenti i campioni.
- Il normalizzatore esistente usa solo condizioni TRAIN, ma **tutti** i loro frame, inclusi seen. Mantenerlo come artefatto storico; per il Mese 2 produrre `train_norm_stats_month2_fit.json` su soli frame `train_fit` (ed eventuali frame di input delle finestre autorizzate). Normalizzare anche i target con statistiche fit-only è corretto nel training supervisionato. Vietate normalizzazioni usando la media/std della T del frame da ricostruire.
- POD, ParamMean, preprocessing e calibrazioni apprese rispettano lo stesso sottoinsieme fit. Per un fold pseudo-extrap interno a TRAIN, ricalcolare **tutti** questi oggetti escludendo la condizione trattenuta: la normalizzazione globale delle 20 condizioni non è fit-only per quel fold.
- Incrementare i frame solo con evidenza e decisione esplicita; nuova versione del manifest e riaddestramento dei confronti interessati con gli stessi dati. **Decisione 20 novembre:** adottato C149 verificato in Opus (149 frame/condizione), normalizzatore 760 congelato e comparatori con pool di fit dichiarati. Non confrontare un neurale con più frame e una baseline rimasta a 38 frame/condizione attribuendo tutta la differenza al modello.
- I test del Mese 1 sono già stati esaminati e hanno informato la discussione: dichiararlo. Da qui nessuna scelta di architettura, K, soglia o iperparametro sui test. Non creare nuovi split “più favorevoli”. Eventuali risultati successivi al pilot che ne usino i feedback diventano esplorativi; un'ulteriore conferma indipendente richiede un protocollo concordato, senza promettere un test intatto disponibile.

### Osservazioni e parità dei confronti

- Primo sparse setup: 2C, single-frame, 5% dei **siti** (~819 vettori = ~1638 valori scalari); estensioni essenziali 1% e 20%. Registrare siti, scalari, numero token e K separatamente.
- Maschere di valutazione persistite o ottenute da `stable_seed` su `(seed, sim_id, frame, layout, replica)`, indipendenti da ordine del dataset, worker o modello. K annidati con stessa permutazione: 1⊂4⊂9⊂16⊂32; stessa maschera velocità per K=0 e K>0. Le maschere non dipendono dalla T target.
- Training: variazione casuale di layout con RNG riproducibile per epoca. Iniziare a 5%; poi una sola strategia preregistrata di campionamento fra 1/5/20% per entrambi i modelli, se prevista nel budget. Distinguere cambio di densità visto in training da generalizzazione a densità mai viste.
- Confrontare metodi sullo **stesso pacchetto di osservazioni**, frame, componenti, parametri disponibili e budget di selezione. Dense U-Net e LocalHGB denso sono diagnostici con più informazione, riportati fuori dalla classifica sparse. Unconditional significa assenza di p anche nel mean prior, nelle soglie adattive, nelle statistiche per condizione e nei moduli ausiliari; il normalizzatore globale fit-only resta comune.
- Un vettore PIV osservato può diventare un token `(x,y,vi,vj)` senza raddoppiare i siti. I token scalari eterogenei esistenti restano un'alternativa documentata. Padding sempre con validity mask. Per U-Net usare valori normalizzati riempiti con zero **più maschera esplicita**: lo zero numerico è un segnaposto riconoscibile, non una misura fisica. Non interpolare attraverso valori di velocità nascosti prima di mascherare.
- Le T assistite si valutano anche sui **punti non osservati da sensori T**, oltre al campo intero; riportare separatamente errore alle ancore. Non ottenere un guadagno limitandosi a ricopiare K target.

### Metriche, selezione e costo

- Primaria per checkpoint/tuning: media per condizione della RMSE full-field di VAL; a parità numerica scegliere costo minore. Riportare anche peggior condizione, variazione tra seed e degradi regionali. Distinguerla da RMSE aggregata su tutti i pixel: documentare l'ordine frame→condizione→split.
- Per ogni frame: RMSE/MAE in K, NRMSE con σ_T fit-only, bias firmato, |bias|, Centered-RMSE; l'identità RMSE²=bias²+Centered-RMSE² vale **prima** di mediare, non sui quadrati delle medie in tabella. Salvare mappe del bias temporale e deviazione delle fluttuazioni.
- Regioni fisse già definite: T≥2100 K, T≥2200 K, T<1800 K; RMSE regionale, IoU/Dice ed estensione della regione con prevalenza vera/predetta. Queste regioni sono diagnostiche termiche, non una segmentazione chimica esatta del fronte. Regione vera vuota: RMSE non definita con conteggio zero, mai 0 K; riportare il numero di casi vuoti e l'attuale convenzione both-empty IoU=1 separatamente.
- Livello: media spaziale, P95 e picco con errori; segnalare la sensibilità dei picchi agli outlier LES. Target grezzo come proposta primaria; nessun clipping silenzioso. Una sensitivity con esclusione di valori fuori range fisso può essere secondaria solo dopo D5, applicata identicamente a tutti.
- Non usare `top25pct_iou` esistente come indicatore invariante al bias. Per ora bastano errori centrati + regioni fisse. Un'eventuale nuova metrica rank-based con quantili separati richiede definizione/test/nome distinti prima di usarla.
- Incertezza statistica: seed modello e replica maschere distinti; i frame correlati non sono repliche indipendenti. Se si stimano intervalli, bootstrap a blocchi temporali con dimensione motivata dall'autocorrelazione, riportando l'esiguo numero di condizioni. Nessun p-value fondato su milioni di pixel indipendenti.
- Hardware, tempo di fit e ricerca, picco RAM/VRAM, aggiornamenti, esempi/query visti, parametri e latenza batch=1 a campo completo entrano in ogni run card. Confrontare budget di ottimizzazione e budget di selezione, riportando anche il costo effettivo; stesso numero di epoche non garantisce stesso costo.

## Step-by-Step Execution Plan

### Step 2.1 — Congelare domanda, protocollo e decisioni

- **Stato:** `[x] Concretizzato come default operativo (2026-11-16)`; **classe:** I + S.
- **Cosa è stato chiuso qui:** protocollo frame versionato, contratto pilota versionato, registro D1–D9 aggiornato con autorizzazione Enrico vs assenza di ratifica supervisori, verifica geometrica dei fold, controlli YAML/conteggi.
- **Cosa resta esplicitamente pendente (non spuntato come “approvato dai supervisori”):** ratifica tecnica Deng Lab; D1 provenienza allegato; lock scientifico finale dei fold; definizione operativa delle risorse dei prossimi run (il ceiling proposto 24 GPU-ore è superato per la missione sparse, decisione Enrico 18 novembre; costi U4/U6 già misurati su L40S); Steps 2.5–2.9; merge main.
- **Problema:** quali risultati sono affidabili e quali scelte restano ipotesi? Senza questo passaggio gli agenti possono implementare tre architetture “raccomandate” come se fossero approvate.
- **Motivazione/dipendenze:** chiudere la parte decisionale operativa aperta dello Step 1.6 lato protocollo. Riferimenti e discrepanze sono già raccolti nel registro; non occorre ricominciare l'audit del dataset.
- **Attività concrete (eseguite in 2.1):**
  1. Registrati D1–D9 distinguendo autorizzazione Enrico, default operativi, verifiche tecniche pendenti ed estensioni rinviate; conservata l’assenza di ratifica supervisori documentata. Provenienza proposal lasciata come limite dichiarato, non bloccante.
  2. Creati `configs/splits/month2_frame_protocol.yaml` (`month2_frame_protocol_v1`) e `configs/experiments/month2_pilot.yaml` (`month2_pilot_v1`).
  3. Verificati gli esempi fold: `20NH3_0.85.h5` è **on_boundary** (non estrapolazione geometrica); `80NH3_1.2.h5` è **outside**. Proposta alternativa lean `40NH3_0.85.h5` + `80NH3_1.2.h5` (non locked). Nessun fitting sui fold.
  4. Budget **storico Step 2.1:** tetto **24 GPU-ore** allora proposto, Engaging allora non verificato. **Aggiornamento 18 novembre:** questo ceiling non si applica alla missione sparse; nessun tetto arbitrario in ore, profiling e plateau secondo roadmapsparse.md.
- **Output:** `docs/month2/00_context_and_decisions.md` aggiornato; YAML sopra; entry nel diario.
- **Verifiche/completamento:** parsing YAML OK; conteggi frame e disgiunzione fit/seen OK; nessun claim di normalizzazione Mese 2 già calcolata; nessun “approved supervisori” inventato.
- **Passaggio:** protocollo tecnico sufficiente per 2.2; i run scientifici 2.3–2.8 restano subordinati a risorse verificate e alle decisioni applicabili ancora aperte.

### Step 2.2 — Rendere la pipeline adatta al training e alla verifica del leakage

- **Stato:** `[x] Implementato, verificato su dati reali locali, e correttivo API completato (2026-11-16)`; **classe:** I.
- **Cosa è chiuso qui:** percorso Month-2 isolato (`Month2ReconDataset` + `ShardStore` compact LRU), frame roles, stats fit-only, manifest, collate, evaluator Kelvin, gate TEST, test di isolamento + batch reale CPU; **poi** membership ruoli, provenance stats (scientific hash), aggregazione omogenea, prefetch I/O documentato.
- **Cosa resta pendente / non spuntato (storico 2.2):** picco RSS locale dominato dal decode Arrow (~2 GiB/shard, ~6.5 GiB osservati con DataLoader workers). **Aggiornamento successivo:** Engaging GPU/AMF verificati (preflight + U4/U5); U-Net densa U0–U5 eseguiti — vedi [roadmap_unet.md](roadmap_unet.md).
- **Problema:** il loader Month-1 indicizza/carica in modo non adatto al training sicuro. Il percorso Month-2 lo sostituisce per il lavoro successivo senza alterare gli artefatti storici.
- **Motivazione/dipendenze:** contratto 2.1; non richiede scegliere GLU.
- **Attività concrete (eseguite):**
  1. Shard locali verificati (30 arrow, ~59 GiB) sul workspace WSL; Engaging non verificato. `ShardStore` decodifica un shard alla volta e tiene solo frame compatti (T/vi/vj/vk).
  2. Ruoli `train_fit` / `seen_train_eval` / `val_eval` (+ indici TEST gated); `train_norm_stats_month2_fit.json` su soli 760 frame fit; normalizzatore fold-capable (n condizioni non fissato a 20).
  3. Sample `{model_inputs, targets, metadata}`; free/assisted; p_off senza op_params; 2C senza vk; dense pack zero+mask.
  4. Manifest frame/osservazioni; collate con validity mask; seed maschera indipendenti da ordine/worker.
  5. `month2_evaluator` con denorm in K, aggregazione frame→condizione→split, off-anchor; `top25pct_iou` esclusa dalla selezione.
- **Output/file:** moduli `src/combustion_recon/data/month2_*.py`, `shard_access.py`, `collate.py`, `freeze.py`; `evaluation/month2_evaluator.py`; manifest/stats; `tests/test_month2_pipeline.py`, `tests/test_model_input_isolation.py`; `scripts/05_validate_month2_pipeline.py`; `docs/month2/01_pipeline_validation.md`.
- **Verifiche:** 43 pytest passati; batch reale confrontato allo shard; stats indipendenti OK; TEST denied senza freeze. Dettagli numerici nel report.
- **Passaggio:** pipeline utilizzabile per Step 2.3 sulle risorse locali; prima di run su Engaging ripetere check host/GPU.

### Step 2.3 — Qualificare baseline e diagnostici sul protocollo comune

- **Stato:** `[x] Eseguito e verificato in locale (CPU)`; **classe:** I + E.
- **Problema:** quale miglioramento conta davvero rispetto a predire una media o fare una ricostruzione modale già corretta?
- **Motivazione/dipendenze:** 2.2; i numeri storici 3C/full-grid non possono essere trasferiti automaticamente alla configurazione primaria 2C sparse con nuovi frame/statistiche.
- **Attività concrete:**
  1. Riusare `GlobalTrainMeanPredictor`, `ParamConditionedMeanPredictor` con LOCO e `GappyPODReconstructor`, senza riscriverli. Registrare fit frame, prior, componenti e preprocessing effettivi. Baseline zero-sensori: GlobalMean senza p e ParamMean con p.
  2. Verificare su un sottoinsieme comune il comportamento corretto di MAP, pesi per modalità e AO; distinguere `+AO` (τ=1 storico) da `+T-block`. Ripetere K=0 con input/maschere identici, oltre a K=9 e K=32.
  3. Il tuning storico di λ/w_T tocca i bordi ed è stato effettuato a K=4. Fare una piccola estensione su VAL separatamente per temperature-free e assisted primario K=9: R iniziale 32; al massimo 9 combinazioni λ/w_T per task, includendo il riferimento storico e un ordine di grandezza oltre il bordo rilevante. R=64 soltanto se il gap oracle/VALIDAZIONE lo giustifica entro il budget. Riportare tutto il costo di selezione.
  4. Oracle POD usa la T completa unicamente per distinguere rappresentazione e stima; non entra nella classifica deployable. LocalHGB resta diagnostico dense; se si vuole confrontarlo sparse, definire prima una versione che usi esclusivamente gli stessi dati osservati, senza patch nascoste.
  5. Valutare train-fit, seen e VAL con manifest comune; i risultati storici TEST restano riferimento documentale, non guida al tuning.
- **Output/file:** `scripts/06_run_month2_baselines.py`; `configs/experiments/month2_baselines.yaml`; tabelle `results/tables/month2_baseline_{per_frame,per_condition,summary,search}.csv`; `results/meta/month2_baseline_run_manifest.json`; `docs/month2/02_baseline_validation.md`. CSV Mese 1 non toccati.
- **Verifiche/completamento:** AO Centered-RMSE invariante; K=0 T-block flag = free sullo stesso modello; nessun iperparametro su TEST; Oracle etichettato non-deployable; isolation adapter/parity storica OK; HGB dense in tabella separata. R=64 non eseguito (nessuna necessità automatica).
- **Passaggio:** baseline confrontabili con costo noto; **stop per revisione indipendente** prima di Step 2.4.

### Step 2.4 — Primo neurale: diagnostico da velocità densa single-frame

- **Stato:** `[x] Concluso nel perimetro dichiarato` su `mission/dense-unet` (U0–U7; 2026-11-17). **Limiti espliciti:** una sola coppia seed 0 p_on/p_off; VAL usata per selezione/scheduler; repliche seed rinviate; nessuna ratifica supervisori; merge main non eseguito. **Classe:** E + S. Riferimenti: [roadmap_unet.md](roadmap_unet.md), [docs/month2/03_dense_velocity_diagnostic.md](docs/month2/03_dense_velocity_diagnostic.md).
- **Problema:** una mappa non lineare spaziale usa la velocità meglio delle baseline, prima di complicare la rappresentazione sparse?
- **Motivazione/dipendenze:** 2.3 e D4/D8. Diagnostico ad alta copertura, **non** information ceiling.
- **Attività concrete (eseguite):** U-Net b=32, canali vi/vj+xy+p, T diretto; p_on seed 0 (U4) e p_off seed 0 (U6) su Engaging L40S; snapshot-swap (U5/U6); confronti matched con GlobalMean/ParamMean/GappyPOD denso; report e chiusura U7. FiLM/ParamMean-residual/temporale restano fuori perimetro.
- **Output/file:** `src/combustion_recon/models/unet.py`, `training/*`, `scripts/07_train_reconstructor.py`, `09_dense_snapshot_control.py`, `10_u6_poff_eval_and_swap.py`, configs dense; `results/month2_dense_unet/u{4,5,6}_*/`; report 03. Checkpoint su Engaging pool con hash versionati.
- **Verifiche/completamento:** isolation/overfit/profiling; training scientifici e controlli snapshot eseguiti; metriche e figure versionate. **Non** spuntato come robustezza multi-seed.
- **Passaggio:** verso Step 2.6 sparse e 2.5 audit GLU (orientamento; non autorizzazione automatica). Non promettere che GLU risolva il problema né cancellare il ramo free.

### Step 2.5 — Audit dell'interfaccia GLU e scelta dell'adattamento minimo

- **Stato:** `[ ] Da completare`; **classe:** I + S. Piano operativo ora in [roadmapsparse.md](roadmapsparse.md), S1/S3/S4; audit prima del confronto GLU, nessuna implementazione sparse già dichiarata.
- **Problema:** come riusare la release effettiva senza assumere che paper, codice e nostro task abbiano input/loss identici?
- **Motivazione/dipendenze:** sorgenti già acquisiti in `external/GLU_Reconstruction/`; tutti gli adattamenti devono vivere nel pacchetto di tesi. Non lanciare il trainer upstream con split temporale e preprocessing propri come se implementasse il nostro protocollo.
- **Attività concrete:**
  1. Documentare Stage 0: encoder `DomainAdaptiveEncoder`, decoder `SoftDomainAdaptiveReconstructor`, selezione `channel_to_encode`, `U` inutilizzato, wrapper che salta il propagatore. Mappare canali originali 11/12→indici locali 0/1 e target T separato; il decoder di tesi emette soltanto T.
  2. Provare che nessun percorso `G_full`, sensor supervision, importance target o loss ausiliaria renda T osservata nel ramo free. Vietato passare T nel payload d'inferenza anche se si spera che `channel_to_encode` la scarti.
  3. Annotare differenze della release: Linear con bias + RMSNorm sui valori, fusione gated, self-attention del mixer sensori, `cdist` query×sensori. Testare sensibilità a vettori di uguale direzione ma scala diversa; non affermare collasso matematico esatto dell'implementazione attuale. Un embedding MLP senza normalizzazione immediata dei valori è un adattamento motivato, da confrontare se diventa una rivendicazione.
  4. Primo candidato: backbone globale/locale deterministico, distanza geometrica uniforme (importance=1), niente varianza/Beta o propagatore. Denominarlo **GLU adattato, ablation deterministica**, non riproduzione integrale del paper. Condizionamento semplice confrontabile con 2.4; nessuna aggiunta simultanea di FiLM + prior ParamMean + importance condizionata.
  5. Profilare numero token, self-attention e decoder; chunking delle query con equivalenza numerica. Se il costo a 20% è eccessivo, registrare il limite e ridurre batch/ottimizzare a parità di input; non togliere silenziosamente sensori solo a un concorrente.
- **Output/file:** futuro `docs/month2/04_glu_release_adaptation.md` con tabella modulo→riuso/modifica/esclusione, hash dei sorgenti; `configs/models/glu_minimal.yaml`; test d'interfaccia in `tests/test_glu_adapter.py` prima dei run.
- **Verifiche/completamento:** forme, gradienti, input isolation, input permutati, chunked/full equivalenti in eval; primo protocollo GLU con batch omogenei senza padding e rifiuto esplicito degli input padded, come delimitato in [roadmapsparse.md](roadmapsparse.md); test validità end-to-end obbligatori prima di qualunque futuro supporto padding; nessuna dipendenza da tempo futuro o T target; `external/` identico alla base. Le 100000 epoche e batch=256 della config upstream non vengono copiate come budget ragionevole.
- **Passaggio:** D4 registra quale adattamento implementare. Se l'audit scopre un costo incompatibile, proseguire con U-Net mascherata e registrare GLU rinviato; non considerare completato il confronto GLU.

### Step 2.6 — Ricostruzione temperature-free da velocità sparsa

- **Stato:** primo confronto A38 seed 0 eseguito e consegnato a STOP E; **conferma C149 eseguita: S7/STOP F concluso nel perimetro temperature-free il 21 novembre** (report `docs/month2/10_s7_c149_model_comparison.md`; U-Net p_on candidato operativo free per decisione di sviluppo post-risultati, p_off e GLU restano nel confronto); Mese 2 resta aperto, prossimo blocco sweep K su entrambe le famiglie e p_on/off. Classe I + E. [roadmapsparse.md](roadmapsparse.md) è il piano operativo S7 (10 nuovi training + 2 riusi), autorizzato esplicitamente da Enrico il 20 novembre; questo commit documentale non lancia job.
- **Problema:** quanto errore aggiungono osservazioni incomplete e rappresentazione/architettura, rispetto al diagnostico denso?
- **Motivazione/dipendenze:** 2.4, e 2.5 per GLU. Serve prima una baseline sparse neurale semplice, poi il candidato del laboratorio con gli stessi input.
- **Attività concrete:**
  1. Estendere la stessa U-Net a canali valori+maschera esplicita; mantenere coordinate, capacità e budget documentati. Addestrarla da zero sul pacchetto sparse primario, senza introdurre valori dense nascosti.
  2. Implementare l'adattamento deterministico GLU scelto in `src/combustion_recon/models/glu_reconstructor.py`, usando token vettoriali co-locati o interfaccia scalare con equivalenza delle misure provata. Decoder a coordinate, output T, nessun channel T nei token free.
  3. Partire da 5%, K=0, single-frame. Per ogni backbone confrontare p off/on entro la matrice ridotta approvata, con medesimo prior di output diretto e budget di selezione. Preservare un ramo unconditioned valido anche se il conditioned vince.
  4. Valutare lo stesso modello a 1/5/20% con maschere fissate: distinguere variazione di copertura da variazione di condizione. Non introdurre subito rakes, blocchi mancanti e sweep 0.5–20%; sono estensioni dei mesi successivi.
- **Output/file:** nuovi moduli e test sopra; config `month2_sparse_unet.yaml`, `month2_sparse_glu.yaml`; `docs/month2/05_sparse_reconstruction.md`; tabelle matched con GappyPOD, GlobalMean/ParamMean; curve/coperture e mappe per condizione; run card con RAM/VRAM e latenza completa.
- **Verifiche/completamento:** test di padding/permutazione per set encoder, predizioni identiche se cambiano solo valori nascosti, invarianti su p off, stesso observation hash per ogni confronto. Risultati train/seen/VAL e costo disponibili; non basta “lo script gira”. La versione sparse a 100% può essere uno smoke diagnostico, non richiede una nuova campagna dense.
- **Passaggio:** scegliere un backbone per 2.7 usando VAL e costo; un mancato vantaggio di GLU è un risultato accettabile e può motivare tenere U-Net. Nessun passaggio automatico a un modello più complesso perché “più nuovo”.

### Step 2.7 — Correzione con T e poi T in ingresso: stessi K e stesse osservazioni

- **Stato:** aperto. Astra ha concluso la diagnosi storica a K=9 su S5; **Enrico approva il 20 novembre il seguito in due blocchi, entrambi successivi alla campagna free S7**. La precedente testa residua puramente opzionale non esaurisce più il piano: è prevista una prova di training con T realmente in ingresso. Forma e matrice di quel training saranno congelate prima di eseguirlo.
- **Domanda:** quanto recupera una semplice correzione scalare e quanto aggiunge un modello che apprende a utilizzare la posizione e il valore delle T, soprattutto nella struttura e nei getti?
- **Dipendenze:** conclusione e analisi S7, checkpoint free identificati, regola di selezione del backbone/p documentata. Se non emerge una scelta chiara fra U-Net e GLU, il successivo training assisted include entrambe. Nessuna scelta sul miglior singolo seed né sui risultati TEST.

#### 2.7-A — Sweep delle correzioni: nessun nuovo training neurale

1. Congelare **K=0/1/4/9/16/32** e un manifest di siti annidati 1⊂4⊂9⊂16⊂32. Le posizioni sono scelte dalla geometria, senza T/gradienti/residui o risultati VAL, e persistite con indici/coordinate/hash. Usare esattamente questo payload anche in 2.7-B. K=9 non è un optimum dimostrato: la griglia storica T9 è solo un riferimento convenzionale. Decidere preventivamente se preservarla come sottoinsieme del layout più ampio; se cambia, tenerla distinta dal riferimento Astra originale.
2. Velocità primaria 5%, medesimi siti/valori di velocità, frame e repliche eval 0/1/2, stessi p disponibili per ogni confronto. T numeriche ideali, nessun rumore o tuning del layout. Tutto il target fuori dalle K ancore è ammesso solo in loss/metriche/oracle, mai nell'input della correzione.
3. Dalla **stessa predizione base** calcolare free, AO_K con tau=1 (`delta=sum(residui)/(K+1)`), mean_K senza shrink e median_K ordinaria senza shrink. Per K pari mediana = media dei due valori centrali; per K=0 tutte le correzioni restituiscono esattamente il free. Oracle-offset full-field separato, soltanto diagnostico. Media senza shrink distingue l'effetto dello stimatore dalla rimozione della regolarizzazione.
4. Affiancare GlobalMean/ParamMean a informazione etichettata, GappyPOD+correzioni e GappyPOD+T-block con gli stessi siti T e velocità; artefatti di fit/rango/lambda dichiarati e baseline C149 qualificate per il confronto a pari dati. Nessun refit/tuning silenzioso su VAL. Il T-only costante storico resta un controllo debole di livello, non rappresenta tutti gli interpolatori T-only.
5. Pubblicare globali, off-anchor e alle ancore, centered-RMSE, bias e regioni con conteggi; errore dell'offset rispetto all'oracle, singoli K/seed/condizioni e peggioramenti. Nessun miglioramento monotono imposto o filtraggio dei casi sfavorevoli. Usare lo stesso insieme off-anchor tra metodi a K fissato; registrare che cambia con K, affiancando campo completo o un insieme comune per confrontare diversi K.
6. Elaborare a chunk e riusare il forward fra correzioni/K. Se le predizioni non sono persistite serve inferenza nuova: zero training non significa zero GPU. Salvare residui per sito e metriche sufficienti a riprodurre i confronti, senza richiedere tutte le mappe complete.
7. Gate free/K=0 contro i CSV S7: riferimenti non vuoti, file/hash/chiavi e tolleranza macro RMSE 0.05 K, scarti per-frame ove disponibili. Centered-RMSE invariata per la stessa traslazione/insieme di pixel; regione vuota esplicita, nessun NaN globale. COMPLETE soltanto a verifiche eseguite.

**Consegna:** specifica K/layout congelata, tabelle e figure matched, costi, report distinto dallo storico Astra. La mediana resta originata da una diagnosi esplorativa su VAL; applicarla a nuovi seed non rende indipendente quella validazione. Il risultato dello sweep, anche negativo, non obbliga a cambiare stimatore/layout prima della prova appresa.

#### 2.7-B — Training con osservazioni T realmente in ingresso

1. Usare il backbone scelto dal confronto free; **entrambe le famiglie se il confronto non giustifica un vincitore chiaro**. La scelta di studiare questo blocco è approvata: non è subordinata a ottenere un buon risultato della mediana. Si dettagliano dopo S7/2.7-A numero di training/seed, p, inizializzazione o fine-tuning e rappresentazione degli input prima della submission.
2. Fornire al modello soltanto i K valori T osservati, relative posizioni e validità, oltre alle stesse velocità/coordinate/p consentiti. Per U-Net valori T normalizzati più maschera esplicita o adattamento documentato; per GLU token/canali con tipo di variabile o adattamento equivalente verificato. Non fornire la mappa T completa come feature e non scartare le ancore nel k-NN delle velocità.
3. **Stesso manifest K/siti e stessi pacchetti di osservazioni dello sweep 2.7-A**, senza un layout più favorevole al modello appreso. Congelare la distribuzione TRAIN-K (inclusa gestione di K=0) e il criterio di selezione prima di allenare. È possibile allenare una singola configurazione a K variabile e valutarla su tutto lo sweep: nessun training per ogni K è implicito, né una nuova griglia di architetture/iperparametri.
4. Scelta implementativa da motivare una volta: ingresso T nel backbone oppure modulo residuo che legga effettivamente le ancore; non limitarsi a rinominare una traslazione deterministica come assimilazione appresa. Se si usa un residuo su backbone congelato, l'eventuale vincolo di correzione a media nulla e il ramo offset vanno definiti/testati. Una rete integralmente riaddestrata con T non è automaticamente identica al vecchio free quando K=0: misurarlo e dichiararlo. Conservare sempre i checkpoint temperature-free originali.
5. Confrontare il modello appreso con il relativo free, media/mediana e baseline matched **agli stessi K e siti**, separando effetto delle informazioni T da capacità di sfruttarle spazialmente. Misurare centered-RMSE/getti oltre al livello, guadagni off-anchor, costo e tutti i peggioramenti. Affiancare una sola baseline spaziale residua economica (kernel/RBF o ridge TRAIN) da specificare prima, evitando di attribuire ogni vantaggio alla complessità neurale.
6. Test di isolamento mutando T fuori ancore, gestione dell'insieme vuoto, maschere/validità, permutazioni e nessuna T nascosta nell'input. Statistiche/calibrazioni solo sul TRAIN consentito, nessuna scelta TEST. Le etichette T complete per la loss supervisionata restano lecite.
7. Esito valido anche se il training assisted non supera la correzione semplice: riportare il risultato e selezionare la soluzione giustificata dai dati, senza campagne ulteriori automatiche.

**Output indicativi nuovi:** `results/month2/temperature_k_sweep/`, `results/month2/temperature_input_study/`, specifiche dedicate e report `docs/month2/11_temperature_k_sweep.md` e `docs/month2/12_temperature_input_study.md`. Percorsi futuri da creare; non sovrascrivere report 06–09. Prima di ogni blocco: matrice, selezione, stime e sorgenti/config congelati; il presente commit non autorizza una matrice assisted indefinita né esegue inferenza.

**Completamento 2.7:** sweep di correzione e prova appresa con T eseguiti/analizzati nel perimetro concordato, comparatori e limiti espliciti, scelta free/assisted registrata. Esiti negativi o assenza di una famiglia dominante sono risultati accettabili. Il TEST resta uno step separato.

### Step 2.8 — Confronti essenziali e primo trasferimento controllato

- **Stato:** `[ ] Da eseguire dopo selezione`; **classe:** E.
- **Problema:** il guadagno viene dal modello, dai parametri, da K ancore o dalla quantità di dati? Sopravvive al cambio di condizione?
- **Motivazione/dipendenze:** esperimenti 2.3–2.7, nessun difetto di pipeline aperto. Prepara la campagna dei Mesi 3–4 con una matrice piccola.
- **Attività concrete:**
  1. Completare la matrice sotto su seen/VAL; screening a un seed, poi almeno tre seed per i soli finalisti primari se nel budget D8. Usare medesime repliche di maschera (proposta: tre). Se le ripetizioni non sono fattibili, dichiarare il risultato preliminare e rinviare la chiusura della variabilità, senza fingere robustezza.
  2. Se si usa il controllo di bordo, riaddestrare soltanto la coppia finalista rilevante sui due fold TRAIN definiti in 2.1, con normalizzazione/prior/POD fold-specifici. Nessun tuning di una famiglia diversa per ogni condizione esclusa.
  3. Congelare config, checkpoint, codice, regola di valutazione e metriche prima di un eventuale pilot TEST_INTERP/TEST_EXTRAP concordato in D2. Salvare `month2_selection_freeze.json` con hash e timestamp; un runner eval deve richiedere questo artefatto per leggere i test.
  4. Pilot sul protocollo di frame già definito, per condizione. Distinguere interpolazione della miscela (inclusa unica 60% NH3) da estrapolazione/frontiera di φ. Il solo 60% non stabilisce generalizzazione sistematica a tutte le miscele nuove. Confrontare anche con vicini TRAIN e loro difficoltà intrinseca, non solo medie aggregate di split.
  5. Errori peggiori: mappa/bias, prevalenza hot-region, cold jets e p95. Un caso fallito resta nel report. Se un risultato di TEST induce una modifica, etichettare il seguito esplorativo e non riusare quel test per scegliere il nuovo modello.
- **Output/file:** futuri `scripts/08_evaluate_reconstructor.py`; freeze manifest; `results/tables/month2_matched_{per_frame,per_condition,summary,cost}.csv`; `results/figures/month2/`; `docs/month2/07_matched_pilot_results.md`.
- **Verifiche/completamento:** ogni confronto ha input/data/mask/budget verificabili; assenza di configurazioni scelte sui test; disponibilità di tutti gli errori per condizione e costi; incertezze/repliche dichiarate. Non occorre ottenere una vittoria per chiudere l'esperimento.
- **Passaggio:** evidenza sufficiente a scegliere il prossimo problema (stima da velocità, robustezza a p, uso di poche ancore o costo). Campagna completa di layout e generalizzazione resta futura.

#### Matrice minima, senza prodotto cartesiano di tutte le opzioni

| Blocco | Modelli | Input controllati | Confronto che consente |
|---|---|---|---|
| Riferimenti | GlobalMean, ParamMean | 0 sensori, p off/on dichiarato | Quanto spiega il livello medio |
| Dense diagnostico | LocalHGB; U-Net compatta p off/on | 100% vi/vj, singolo t | Primo guadagno non lineare; separato dai metodi sparse |
| Sparse primario | GappyPOD, masked U-Net, GLU minimale | Stesse 2C al 5%, K=0, p off/on | Architettura e informazione operativa |
| Assisted — correzione | Finalisti free + media/mediana; ParamMean e GappyPOD + correzioni/T-block | Velocità 5%, K annidati 0/1/4/9/16/32 e stessi siti | Valore delle T e limite della traslazione scalare |
| Assisted — appreso (decisione 20 novembre) | Backbone scelto con T in ingresso; entrambe le famiglie se confronto inconcludente | Identici K/siti/osservazioni del blocco correzione, TRAIN-K e selezione congelati | Quanto apprendere dalle T aggiunge rispetto alla correzione e ai comparatori |
| Sweep senza nuova famiglia | Finalista free+AO e baseline | K annidati 0/1/4/9/16/32; copertura 1/5/20% a K=0 | Sensibilità a informazione disponibile |
| Estensioni ancora fuori mandato | 3C; 3 frame causali; rumore/layout realistico; altre architetture | Decisione separata | Risposta a un limite osservato |

Non replicare tre seed per ogni riga di ricerca iperparametrica. Ripetere i finalisti e i loro comparatori essenziali; la capacità della suite si decide prima del pilot, con budget totale visibile.

### Step 2.9 — Chiusura documentata e scelta della fase successiva

- **Stato:** `[ ] Da completare`; **classe:** I + S.
- **Problema:** quale conclusione autorizzano davvero i risultati, e quale singola modifica merita il Mese 3?
- **Motivazione/dipendenze:** evidenze degli step applicabili; distinguere software pronto, esperimenti conclusi e decisioni ancora pendenti.
- **Attività concrete:**
  1. Report breve con domanda, protocollo/versioni, tabelle matched per condizione, mappe, costi, risultati negativi e limiti (una traiettoria per condizione, test già esaminati, ancore ideali, outlier).
  2. Registrare scelta con i supervisori: mantenere free come task principale; dare più peso ad assisted se lo richiedono i risultati; oppure investire prima nel diagnostico single-frame/temporale. Specificare quale evidenza sostiene la scelta.
  3. Se la testa residua non supera AO/kernel in modo coerente su VAL, fermarla. Se GLU non supera la baseline più semplice a costo accettabile, mantenere quella baseline e formulare una domanda mirata prima di aggiungere complessità.
  4. Per il Mese 3 considerare **solo candidati**: importance condizionata, residuo attorno a ParamMean con ablation esplicita, o base di correzione dipendente dallo snapshot. L'incertezza predittiva del modello non equivale al rumore del sensore: non usarla come R di un filtro senza un modello probabilistico coerente e verifica di calibrazione.
  5. Aggiornare diario, README e stato dei singoli step con run ID/commit; lasciare aperte le attività rinviate e la ratifica mancante. Nessuna modifica retroattiva del proposal.
- **Output/file:** futuro `docs/month2/13_month2_review.md` (il numero 08 è già usato dal report median9); questa roadmap aggiornata con evidenze; diario e indice; proposta circoscritta per il Mese 3, evitando roadmap concorrenti. I nomi dei nuovi report vanno verificati contro il remoto prima di crearli.
- **Verifiche/completamento:** un altro agente può ricostruire ogni tabella da commit/config/dati identificati; checkpoint e risultati pesanti hanno riferimento persistente fuori Git; nessuna conclusione di impossibilità fisica da un modello fallito; novità metodologica non dichiarata per il solo uso di FiLM/maschere.
- **Passaggio:** decisione registrata e una domanda successiva falsificabile. Se manca la ratifica scientifica, dichiarare “baseline/primo pilot completati, revisione aperta”, non “Mese 2 approvato”.

## Missioni circoscritte: consolidamento e primo neurale

Opus e Astra hanno revisionato indipendentemente `822ed52` e concluso «pronti dopo correzioni circoscritte». Enrico ha autorizzato il consolidamento con Cursor IDE locale ([roadmap_missione.md](roadmap_missione.md)). **Missione eseguita** su `mission/baseline-consolidation` a `3cf7bf0`: replica `consolidation_v1` numericamente identica allo storico entro le tolleranze fissate; metadati temporali corretti; report in `docs/month2/mission_baseline_consolidation_report.md`. I risultati Step 2.3 in `results/tables/` restano lo storico datato (provenienza incompleta; «preregistrato» non attestabile retroattivamente). Integrazione in main ancora pendente alla ricognizione del piano U-Net.

[roadmap_unet.md](roadmap_unet.md), richiesta da Enrico il 17 novembre dopo discussione della specifica Opus, dettaglia **soltanto 2.4**: implementazione/verifiche piccole → STOP A; profiling e GappyPOD denso → STOP B; primo training p_on e controllo delle istantanee → STOP C; p_off/seed delimitati e report. Non avvia GLU, sparse, assisted o TEST. Piano pubblicato non equivale a esperimenti eseguiti o ratifica supervisori.

## Istruzioni pratiche per gli agenti di coding

1. Prima di ogni task leggere istruzioni correnti, `git status`, remoto aggiornato, diario, questa roadmap e decisioni applicabili. Non importare esperimenti o diario di `thesis-research`/HONF.
2. Prendere un solo step o un sotto-obiettivo esplicito con criterio d'uscita. I percorsi futuri sopra sono proposti, non file già esistenti né autorizzazione a eseguire l'intera fase in autonomia.
3. Prima della modifica cercare moduli/config esistenti e riusarli. Conservare `external/GLU_Reconstruction/` intatto. Code/config/test di tesi sotto `src/`, `configs/`, `scripts/`, `tests/`.
4. Un run parte soltanto dopo smoke CPU, validazione protocollo, dispositivo, costo previsto e regole di arresto verificati e registrati. Per la missione sparse Enrico ha escluso tetti arbitrari di ore (18 novembre): applicare profiling, plateau e tranche riesaminabili di [roadmapsparse.md](roadmapsparse.md), rispettando separatamente il walltime scheduler. CPU locale ammessa se fattibile; GPU/Engaging richiedono verifica effettiva quando usati. Un pilot breve può profilare il costo prima del run completo; mai lanciare il default upstream da 100000 epoche.
5. Run manifest minimo: commit, config/hash, dataset/shard e frame ID/hash, normalizer/prior/POD hash, input ammessi, p/K/2C/3C/tempo, seed training e maschere, device/versioni, ottimizzazioni/query, criterio checkpoint, tempo/RAM/VRAM, output persistenti e stato esito.
6. Non versionare raw, cache o checkpoint pesanti; versionare configurazioni, manifest, tabelle compatte e report. Non far riscrivere gli artefatti Mese 1 agli script Mese 2.
7. Diario: distinguere ispezionato, implementato, eseguito, verificato, proposto e approvato. Spuntare lo step solo con tutti i suoi criteri; annotare esplicitamente optional non eseguiti e motivazione.
8. Prima del push aggiornare il remoto, preservare modifiche concorrenti, risolvere eventuali conflitti leggendo la nuova evidenza; commit coerente e push fast-forward, **mai force**. Verificare lo SHA remoto. Nessun training o invio a persone implicito in un commit documentale.

## Checklist di completamento della fase

- [x] 2.1 — Protocollo pilota e default operativi versionati; autorizzazione Enrico registrata; limite proposal e assenza ratifica supervisori espliciti; fold geometricamente verificati (lock scientifico fold ancora aperto). Non implica GPU verificate né stats fit-only già calcolate.
- [x] 2.2 — Pipeline Month-2 reale verificata in locale (fit-only, isolation, collate, gate TEST) + protezioni API. Engaging/GPU verificati in seguito (preflight + U4/U5); picco RAM locale legato al decode Arrow.
- [x] 2.3 — Baseline CPU sul protocollo Mese 2 (GlobalMean/ParamMean/GappyPOD 2C ±AO/T-block, Oracle, HGB dense diagnostico); ricerca versionata (preregistrazione cronologica storica non attestabile); report `02_baseline_validation.md`. R=64 non eseguito. Consolidamento e replica completati su `mission/baseline-consolidation` a `3cf7bf0`; integrazione ancora pendente.
- [x] 2.4 — Diagnostico denso concluso nel perimetro U0–U7 ([roadmap_unet.md](roadmap_unet.md), report 03). Repliche seed rinviate; optional temporale fuori missione. **Non** implica Mese 2 chiuso né robustezza multi-seed.
- [x] 2.5 — Audit GLU + adattamento deterministico eseguiti (S1/S3; report `04`/`05`). Ratifica supervisori distinta, ancora aperta.
- [x] 2.6 — Primo neurale sparse matched (U-Net/GLU × p_on/p_off seed 0) + GappyPOD nested 1/5/20% gate-pass; confronto consolidato a **STOP E** (`docs/month2/05_sparse_reconstruction.md`, `results/month2/s6_sparse_eval/STOP_E_delivery/`). **Non** implica finalisti multi-seed né chiusura 2.8.
- [ ] 2.7 — Piano ampliato il 20 novembre; **aggiornamento 22 novembre:** S7 concluso a STOP F. Enrico autorizza la matrice assisted completa — **U-Net e GLU × p_on/off × seed 0/1/2 = 12 training nuovi a K variabile** (un modello per configurazione, valutato su K=0/1/4/9/16/32), in parallelo allo sweep delle correzioni (Cursor) e senza attenderne i risultati; questo amplia la scelta della sola famiglia vincente. Contratto comune congelato in `configs/experiments/month2_temperature_study_protocol.yaml` + `configs/observations/month2_temperature_anchors_nested_v1.json`; coordinamento in `docs/month2/temperature_study_handoff.md` (branch `temperature/assisted-c149`). Training sottomessi il 22 novembre, risultati non ancora disponibili; lo sweep delle correzioni resta aperto.
- [ ] 2.8 — Matrice essenziale, variabilità/costo e comparatori C149 matched; risultati S7 e 2.7 da completare. Eventuali fold/pilot TEST restano decisioni separate con freeze, senza dichiarare intatti i TEST già esaminati nel Mese 1.
- [ ] 2.9 — Report e scelta successiva registrati; approvazione dei supervisori distinta dalla preparazione/esecuzione.

**Fuori dal minimo del Mese 2:** forecasting GLU Stage 1; griglie estese di architetture; incertezza/Beta e assimilazione low-rank completa; T_ad/Cantera come nuovo vincolo; PINN su slice 2D non chiusa; dataset esterni/engine; campagna completa di occlusioni, rumore e layout dei Mesi 4–5. Reintrodurli solo con una domanda concreta, risorse ed evidenze.
