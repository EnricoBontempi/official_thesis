# Mese 2 — Contesto verificato e registro delle decisioni

**Data aggiornamento Step 2.1:** 2026-11-16, America/New_York (UTC−05:00)  
**Ambito:** esclusivamente `EnricoBontempi/official_thesis`; nessuna attività HONF.  
**Stato aggiornato 2026-11-17 (U7):** protocollo, pipeline e baseline 2.1–2.3 eseguiti; consolidamento `3cf7bf0` sul branch dedicato (non ancora in main). **Step 2.4 / missione U-Net densa conclusa** su `mission/dense-unet` (U0–U7; p_on/p_off seed 0 + snapshot controls; repliche seed rinviate) — report [03_dense_velocity_diagnostic.md](03_dense_velocity_diagnostic.md). Engaging AMF/L40S verificati e usati. Steps 2.5–2.9 aperti; Mese 2 **non** concluso. Nessuna ratifica tecnica documentata dei supervisori.  
**Aggiornamento 2026-11-18:** approvata operativamente e pubblicata [roadmapsparse.md](../../roadmapsparse.md). **S1–S3 eseguiti** (maschere + join baseline 3309/3309, U-Net sparse, GLU wrapper deterministico + overfit diagnostico); report [05_sparse_reconstruction.md](05_sparse_reconstruction.md). **STOP B** — nessun training scientifico sparse; S4–S7 aperti.  
**Piano operativo:** [roadmap_month2.md](../../roadmap_month2.md).  
**Autorizzazione operativa Enrico:** conversazione del 16 novembre 2026 (approvazione roadmap e avvio lavoro). Distinta da qualsiasi approvazione dei supervisori.

> **Stato corrente (20 novembre 2026):** il registro D1–D9 del 16–18 novembre è storico. L'aggiornamento finale **§11** e il [piano Mese 2](../../roadmap_month2.md#piano-corrente-di-completamento--20-novembre-2026) prevalgono sulle scelte successivamente modificate; nessun job viene eseguito da questo aggiornamento.

## 1. Provenienza e limite sul documento inviato

Ricognizione precedente del remoto `main` a `6fd0f15` (con diario semplificato preservato da `a2cebb1`); roadmap pubblicata in `ef4b708`.

- Letti diario, proposal Markdown, roadmap Mese 1, report Mese 1, configurazioni e codice pertinenti.
- `documents_thesis/main.tex` è il report del Mese 1 del 15 novembre, non il proposal inviato il 13 novembre. **Non è verificata l'identità fra `docs/proposal.md` e l'allegato inviato.**
- **Step 2.1:** nessuna nuova ricerca estesa dell'allegato; il limite di provenienza resta dichiarato e **non bloccante** per il protocollo pilota. Scopo operativo da `docs/proposal.md`. Nessun proposal modificato.

## 2. Cosa è completato e cosa rimane aperto

| Elemento | Evidenza | Stato e limite |
|---|---|---|
| Audit 30 condizioni, split 20/3/3/4, baseline corrette, report LaTeX, codice GLU intatto | artefatti Mese 1 / `external/` | Completati tecnicamente; sign-off supervisori dello Step 1.6 ancora aperto. |
| Roadmap Mese 2 | `roadmap_month2.md`, commit `ef4b708` | Preparata; **autorizzata operativamente da Enrico** (16 nov 2026); non equivale a ratifica supervisori. |
| Protocollo frame + contratto pilota | YAML Step 2.1 | Default operativi; ora consumati da `Month2ReconDataset`. |
| Statistiche fit-only Mese 2 | `data/manifests/train_norm_stats_month2_fit.json` | **Calcolate** su 20×38 frame `train_fit`. Storico Mese 1 intatto. |
| Pipeline dati Month-2 | moduli `month2_*`, report `01_pipeline_validation.md` | **Verificata in locale** (batch reale, isolation, gate TEST). |
| Baseline Mese 2 (Step 2.3) | `month2_baselines_v1`, report `02_baseline_validation.md` | Eseguite in locale CPU; storico in `results/tables/`. |
| Consolidamento post-revisione | `consolidation_v1`, report `mission_baseline_consolidation_report.md` | **Completata**: isolation/YAML/timeline corretti; replica identica entro tolleranza; in attesa Enrico/ChatGPT. |
| Modelli / training neurali Mese 2 | U0–U7 denso chiusi; S1–S3 sparse su `mission/dense-unet` | Step 2.4 concluso. Sparse U-Net + GLU wrapper diagnostici (STOP B); screening/profiling S4+ aperti. |
| GPU/allocazione Engaging | preflight + U4–U6 | **Verificati e usati** (AMF / L40S). Il ceiling storico proposto 24 GPU-ore è superato per la missione sparse (Enrico, 18 novembre): nessun tetto arbitrario in ore; profiling/plateau e walltime scheduler distinti. |

## 3. Lettura scientifica dei risultati corretti

La domanda è quanto della **temperatura istantanea** sia prevedibile da velocità parzialmente osservata, quanto aggiungano parametri noti e poche osservazioni termiche, e come cambino gli errori su condizioni escluse dal training. La temperatura è disponibile come target numerico supervisionato anche quando è assente dagli input.

La tabella seguente è stata controllata contro `results/tables/month1_feasibility_split_summary.csv`; sono **risultati storici 3C**, non benchmark già equivalenti a un futuro modello 2C. Unità degli errori: K; split storico `test_extrap`, velocità al 5% per GappyPOD.

| Metodo storico | RMSE | Bias firmato medio | Media del bias assoluto per frame | Centered-RMSE |
|---|---:|---:|---:|---:|
| GappyPOD conditioned, K=0 | 153.4 | +41.2 | 43.0 | 138.5 |
| Stesso + AO, K=32 | 140.4 | −0.8 | 18.5 | 138.5 |
| Stesso + T-block, K=32 | 130.2 | +34.8 | 36.1 | 119.3 |
| GappyPOD unconditioned + T-block, K=32 | 125.5 | +5.7 | 17.6 | 123.5 |

- Gap GappyPOD vs OraclePOD: problema di **stima** dalla velocità sulla base lineare R=32, non prova di limite fisico universale.
- AO lascia invariato Centered-RMSE; T-block a K=32 può migliorare anche la forma.
- LocalHGB storico è diagnostico denso; non confrontarlo come se usasse le stesse osservazioni sparse.
- `top25pct_iou` esistente **non** è metrica di forma invariante al bias: esclusa dalla selezione del pilota.

Il report LaTeX del 15 novembre prevale sulle conclusioni datate in sintesi più vecchie (`05_literature_and_glu_notes.md`, `06_supervisor_review_brief.md`, `analisi_agents.md`).

## 4. Differenze fra proposta iniziale, aggiornamento LaTeX e codice

| Tema | Proposal Markdown | Report LaTeX / codice successivo | Trattamento nel protocollo pilota v1 |
|---|---|---|---|
| Primo modello | Deterministico, scelto col laboratorio | U-Net densa prima di GLU | Matrice: diagnostico denso → sparse U-Net / GLU minimale; scelta architetturale ancora aperta ai supervisori (D4). |
| Tempo | Istantanea | 3 frame | **Single-frame primario**; finestra causale **non attivata**. |
| Condizionamento | Confronto centrale | FiLM / residuo ParamMean | Confronto **p off/on** richiesto; meccanismo candidato FiLM, non ratificato. |
| GLU | Approccio informante | Codice acquisito | Audit/adattamento in Step 2.5; non implementato qui. |
| Ancore / assimilazione | Fallback esplicito | Modulo residuo | Primario K=9 + sweep annidato; AO prima; testa appresa **non attivata**. |
| Test | Campagna Mese 4 | Già esaminati in Mese 1 | Selezione solo VAL; TEST solo dopo freeze. |

## 5. Autorizzazione, default operativi e registro D1–D9

### 5.1 Autorizzazione operativa (Enrico)

| Campo | Valore |
|---|---|
| Decisore | Enrico Bontempi |
| Data | 2026-11-16 |
| Fonte | Conversazione operativa (orientamento + mandato Step 2.1) |
| Contenuto | Approvazione della roadmap Mese 2 e autorizzazione ad avviare il lavoro, a partire dalla concretizzazione del protocollo |
| Non implica | Ratifica tecnica dei supervisori; disponibilità GPU; calcolo già avvenuto delle stats fit-only; implementazione pipeline |

### 5.2 Assenza di ratifica supervisori

Non risulta, nei documenti della repository, un’approvazione tecnica specifica di Deng Lab / supervisori su protocollo 2C, primo modello, uso dei test già esaminati, ancore ideali, budget o fold. Il riscontro preliminare della professoressa sul piano generale (diario 13 nov) e la condivisione del codice GLU da Jason **non** costituiscono tale ratifica. Le righe D2–D8 restano **senza ratifica supervisori documentata**.

### 5.3 Registro D1–D9 (aggiornato Step 2.1)

Legenda stato: **default operativo** = usato per preparare protocollo/pipeline; **verifica tecnica pendente** = da misurare/implementare; **estensione rinviata** = fuori dal minimo attivo; **limite dichiarato** = aperto ma non bloccante per 2.1/2.2 infrastrutturale; **scelta scientifica aperta** = non chiusa silenziosamente.

| ID | Decisione | Impostazione registrata il 2026-11-16 | Classificazione | Responsabile / note |
|---|---|---|---|---|
| D1 | Identità proposal 13 nov | Scopo da `docs/proposal.md`; identità con allegato inviato **non verificata**; nessuna nuova ricerca estesa in 2.1 | Limite dichiarato; non bloccante | Enrico; aperta come provenienza |
| D2 | Split, metriche, test | Conservati 20/3/3/4; selezione = media per condizione della RMSE full-field su VAL; TEST esclusi dalla selezione; freeze richiesto per eval test | Default operativo Enrico; **senza ratifica supervisori** | Supervisori + Enrico |
| D3 | Osservazioni primarie | Single-frame; velocità 2C ch. 11/12; sparse primario 5% siti (~819); controlli eval 1/5/20%; 3C **non attivata** | Default operativo; 3C = estensione rinviata | Supervisori + Enrico |
| D4 | Primo modello / GLU | Matrice ridotta: U-Net densa diagnostica → U-Net mascherata + GLU deterministico minimale; condizionamento on/off (candidato FiLM) | **U7:** diagnostico denso chiuso. **18 novembre:** sintesi Opus/Astra approvata da Enrico in roadmapsparse.md: U-Net a 7 canali, GLU deterministico p additivo, quattro screening on/off, maschere condivise/annidate; implementazione e run ancora aperti. Nessuna ratifica supervisori. | Supervisori + Enrico |
| D5 | Target e ancore T | Target T ch.10 grezza in K; ancore ideali sincronizzate; outlier da segnalare; nessun clipping silenzioso | Default operativo; **senza ratifica supervisori** | Supervisori + Enrico |
| D6 | Temperatura assistita | Primario K=9; sweep annidato 0/1/4/9/16/32; prima AO; testa residua appresa **non attivata** | Default operativo; testa = estensione con gate | Supervisori + Enrico |
| D7 | Finestra temporale | Non attivata nel pilota primario | Estensione rinviata (gate D7) | Supervisori + Enrico |
| D8 | Risorse e budget | Host previsto Engaging; tetto iniziale proposto **24 GPU-ore** = ceiling da rivalutare | **18 novembre, Enrico:** nessun tetto arbitrario di ore per run/campagna sparse; ceiling 24 GPU-h proposto superato per questa missione. Profiling, plateau, tranche iniziale 1000 epoche riesaminabile e resume; walltime scheduler separato. Fino a 4 run concorrenti solo dopo validazione e verifica slot reali. U4/U6≈28.5/24 min L40S restano misure storiche. | Enrico + referente risorse |
| D9 | Frame seen e fold | `train_fit` vs `seen_train_eval` disgiunti; future stats fit-only solo su `train_fit`; fold interni con refit obbligatorio; candidati geometrici proposti sotto | Default operativo protocollo; stats **non ancora calcolate**; fold **non locked** | Enrico; ratifica D2 ancora aperta |

## 6. Verifica geometrica dei fold pseudo-extrap (solo metadata TRAIN)

Spazio dei parametri: `(nh3_fraction, φ)` parsato dagli `sim_id` TRAIN in `splits_interp_extrap.yaml`.  
Criterio: il punto escluso deve stare **fuori** dall’inviluppo convesso delle 19 condizioni rimanenti. Controllare solo min/max per asse **non basta** (tutti i candidati restano dentro il bounding box asse-allineato del TRAIN residuo).

| Candidato | `(nh3, φ)` | vs hull rimanente | Estrapolazione geometrica? | Esito |
|---|---|---|---|---|
| `20NH3_0.85.h5` (esempio roadmap) | (0.20, 0.85) | **on_boundary** (spigolo fra `0NH3_0.85` e `40NH3_0.85`) | No | **Scartato** come fold di estrapolazione geometrica |
| `80NH3_1.2.h5` (esempio roadmap) | (0.80, 1.20) | **outside** | Sì | **Valido** come candidato |
| `40NH3_0.85.h5` (alternativa proposta) | (0.40, 0.85) | **outside** | Sì | Proposta come sostituto lean di `20NH3_0.85` |

Altri TRAIN risultati `outside` in leave-one-out (metadata only): `0NH3_0.85`, `0NH3_1.25`, `20NH3_1.25`, `40NH3_0.85`, `80NH3_0.9`, `80NH3_1.2`.

**Coppia proposta (non locked):** fold A = `40NH3_0.85.h5`; fold B = `80NH3_1.2.h5`.  
La scelta definitiva resta una **scelta scientifica aperta** se si vuole un criterio diverso (es. massimizzare distanza, privilegiare angolo CH4, ecc.); **non blocca** il protocollo primario VAL/seen né lo Step 2.2.  
Nessun fitting/training/valutazione prestazioni eseguito sui fold in questo step. Per ogni fold futuro: rifittare normalizer, POD e prior escludendo la condizione trattenuta.

## 7. Contratto concretizzato (file)

| File | ID | Ruolo |
|---|---|---|
| [configs/splits/month2_frame_protocol.yaml](../../configs/splits/month2_frame_protocol.yaml) | `month2_frame_protocol_v1` | Indici frame fit/seen/eval; regole fit-only; split 20/3/3/4 solo referenziato |
| [configs/experiments/month2_pilot.yaml](../../configs/experiments/month2_pilot.yaml) | `month2_pilot_v1` | Input 2C, sparsità, K, seed, selezione VAL, matrice ridotta, estensioni off |

Hash di riferimento (non autoreferenziali):

- `configs/splits/splits_interp_extrap.yaml` → `acb87bf95852e7849f88a270af8cbd211ad999d0039c35885232b7bc2a99e7b9`
- `configs/splits/evaluation_protocol.yaml` → `85a5a2437ec6c2257a01931987907ef7129b9858a2546c9cb33bac8fc99a2950`
- `configs/masks/observation_layouts.yaml` → `a0344e87d5a98cd1203cc0e9b258aa0859e80659e2817877602328973fd1bbb1`
- `data/manifests/train_norm_stats.json` (storico Mese 1) → `37bf7e8579812930d37e816d1689b89a3a91666e56af66d24b08fd958cfa9b76`

## 8. Controlli eseguiti in Step 2.1

- Parsing YAML dei due nuovi file: OK (`yaml.safe_load`).
- `train_fit`: 38 indici `0,40,…,1480`; `seen_train_eval`: 11 indici `1600,…,2000`; intersezione vuota; gap 120.
- VAL/TEST schedule: 41 indici `0,50,…,2000`.
- `sim_id` del protocollo coincidono col file split referenziato (nessuna riscrittura membership).
- Canali 10/11/12, K nested, gruppi matrice e flag estensioni off: coerenti fra registro e `month2_pilot.yaml`.
- Nessun file afferma che `train_norm_stats_month2_fit.json` esista già.

## 9. Cosa non è chiuso (utile al prossimo step)

**Chiuso per Step 2.2:** frame fit vs seen; stats fit-only; isolation input; gate TEST; collate; batch reale locale.

**Non bloccanti per Step 2.3 locale:** D1 provenienza; ratifica supervisori; lock fold; D4/D7/3C/testa; Engaging (si può partire in locale).

**Prima di run scientifici su cluster / confronti costosi:** verificare host/GPU e budget (D8); audit GLU prima del confronto GLU (2.5); gestire il picco RAM del decode Arrow (prefetch selettivo / meno worker).


## 10. Aggiornamento operativo 2026-11-17 — Missione U-Net

Questo aggiornamento prevale sugli stati pendenti storici delle sezioni 5–9 per il solo avanzamento qui descritto; non riscrive quanto fosse stato verificato il 16 novembre.

- **D4:** dopo proposta Opus e discussione con Enrico, primo esperimento = U-Net densa single-frame, output T diretto, coordinate esplicite e concatenazione dei canali NH3/phi. p_on principale; p_off successivo; FiLM e mean prior rinviati. Dettaglio in [roadmap_unet.md](../../roadmap_unet.md). **Chiusura U7:** diagnostico denso eseguito; non ratifica supervisori.
- **D8 (storico 17 nov mattina):** profiling prima di fissare larghezza/epoche/dispositivo; GPU/Engaging da verificare; 24 GPU-ore proposta. **Aggiornamento U7:** Engaging verificato e usato (L40S); costi U4/U6 misurati.
- **D9:** stats fit-only già calcolate in 2.2, nessun refit richiesto dalla missione. Fold ancora non attivati.
- **Consolidamento:** dipendenza remota `3cf7bf0`, ancora fuori main; conserva replica e storia dei risultati.
- **D7:** single-frame; associazione campo-tempo non verificata su tutto `40NH3_1.2`. Finestre temporali fuori missione.
- **Confronti densi:** ParamMean e GappyPOD R=32 con medesima velocità completa; controllo di scambio istantanee. Nessuna modifica dei comparatori storici sparse e nessun TEST.
- **Stato (STOP B, storico):** U0–U3 eseguiti; U4–U7 allora non eseguiti.
- **Stato (U7, 2026-11-17 sera):** U0–U7 chiusi nel perimetro denso dichiarato — [03_dense_velocity_diagnostic.md](03_dense_velocity_diagnostic.md). Prossimo orientamento: 2.5 audit GLU e 2.6 U-Net sparse (non autorizzati da questa chiusura).


## 9. Decisione sparse — 2026-11-18 (America/New_York)

- **Fonte/decisore:** discussione con Enrico dopo due proposte indipendenti Opus/Astra, approvazione della sintesi e richiesta esplicita di scrittura/push. Non ratifica del laboratorio.
- **D3/D4:** [roadmapsparse.md](../../roadmapsparse.md) dettaglia 2.5–2.6: training solo 5%, eval 1/5/20% con maschere condivise/persistite e annidate mantenendo il 5% storico; U-Net nuova a 7 canali; GLU adattato deterministico, p con proiezione additiva senza bias, coordinate isotrope e sigma condivisa; batch omogenei senza padding iniziale e rifiuto esplicito dei casi non supportati. Quattro screening seed 0, poi due configurazioni da confermare con seed 1/2. Decisioni numeriche/config da verificare e congelare prima dei run, senza tuning occulto.
- **D8:** Enrico esclude tetti arbitrari di ore, anche di campagna. Costo misurato e comunicato; plateau operativo e tranche riesaminabile, non termine orario scientifico. Il walltime reale dello scheduler si rispetta con checkpoint/resume. Quattro slot normal riferiti consentono concorrenza potenziale, da verificare sul momento; nessun job lanciato qui.
- **Stato:** solo S0 piano; prossimo incarico previsto S1–S2 con STOP A. Steps 2.5–2.9 e Mese 2 restano aperti. Nessun training, raw download o modifica upstream per questa pubblicazione.


## 11. Decisione corrente — C149 entrambe le famiglie, poi sweep K e T in ingresso (20 novembre 2026)

**Fonte:** conversazione operativa con Enrico Bontempi, che approva la sequenza e chiede al coordinatore di aggiornare/pushare le roadmap prima dell'incarico a Sonnet. Non ratifica dei supervisori.

| Decisione | Impostazione corrente | Stato |
| --- | --- | --- |
| D3 / D9 — Dati e osservazioni free | C149 2980 frame (149×20), velocità 2C TRAIN 5%, seen220/VAL123 invariati; stream Opus 760 presentazioni/95 update; stats fit-only 760 congelate | Approvata per S7 |
| D4 — Famiglie e repliche | U-Net masked e GLU adattato deterministico, ciascuno p_on/off e seed init/shuffle 0/1/2 | **12 run totali, 10 nuovi + 2 U-Net C149 s0 riusati**; nessun GLU C149 già disponibile |
| D8 — Risorse | Max quattro GPU complessive, L40S normal Advanced; limiti attuali da verificare, checkpoint/resume e profiling reali, nessun tetto scientifico in ore o fallback general/preemptable | Preparazione verificata e submission dei dieci training approvate; prossimo incarico Sonnet |
| D5 / D6 — Prima prova T | Dopo conclusione/analisi S7, sweep K=0/1/4/9/16/32 di AO tau1, mean e median su predizioni congelate + baseline matched; T ideali | Direzione approvata; manifest layout/selezione da congelare prima dello sweep |
| D4 / D6 — T in ingresso | Dopo lo sweep, training che legga effettivamente i valori T osservati/posizioni; backbone scelto oppure entrambe le famiglie se confronto inconcludente; **stessi K/siti/osservazioni del confronto correzione** | Esperimento scelto; matrice, semi, p e TRAIN-K da specificare prima dell'esecuzione; non un numero indefinito di training autorizzati |
| D2 — TEST | Nessuna apertura attuale; eventuale pilot soltanto con decisione distinta e freeze | Invariato; TEST Mese 1 già esaminati, non dichiararli indipendenti/intatti |
| Chiusure | Opus O0–O5 e review accettati; O6 sceglie C149. Astra primaria e median9 concluse nel perimetro S5; A4 autonoma non richiesta | Sparse/S7 e Mese 2 aperti |
| Confine del seguito | Multidensità, nuove architetture, 3C/temporale, rumore/layout strumentali e altri sweep | Fuori dall'incarico immediato |

**Modifiche esplicite al piano:** S7 non è più limitato a quattro repliche di due configurazioni; include sei GLU C149 nuovi oltre alle quattro U-Net. Il training con T non resta una testa opzionale attivata soltanto se la mediana funziona male: Enrico sceglie di provarlo dopo il confronto delle correzioni. La specifica architetturale assisted resta da fissare e il ramo free originale va preservato.

**Tracciabilità delle evidenze:** [Opus 06](https://github.com/EnricoBontempi/official_thesis/blob/4f0e2637758de0766d0bb0891fe8ff97eee978c9/docs/month2/06_opus_data_and_training_variability.md), [Astra 07](https://github.com/EnricoBontempi/official_thesis/blob/82bff35726fb70af9987081802ece3ef718e42bd/docs/month2/07_astra_level_structure_and_anchors.md), [median9 08](https://github.com/EnricoBontempi/official_thesis/blob/7cb7bfdc6430a2731c924a31d90897a3aa52a89f/docs/month2/08_astra_median9_second_pass_analysis.md), [proposta consultiva 09](https://github.com/EnricoBontempi/official_thesis/blob/9daf1585ba2bd9d264372e62565606a800fbefe5/docs/month2/09_astra_opus_synthesis_and_s7_proposal.md). Il rinvio GLU suggerito da 09 non è stato adottato. Nessun risultato storico modificato; report/codice ancora su branch distinti.

**Regole di fine:** [S7](../../roadmapsparse.md) chiude il confronto free e sparse a STOP F; [2.7](../../roadmap_month2.md) completa correzione e T apprese; 2.8–2.9 completano valutazioni/decisioni/report Mese 2. Risultati negativi validi e assenza di vincitore sono esiti accettabili, senza campagne aggiuntive automatiche. Prima dell'analisi S7 fissare la regola di scelta del seguito con metriche per seed/condizione e costi. Nessun cambiamento al proposal inviato e nessuna approvazione del laboratorio inventata.
