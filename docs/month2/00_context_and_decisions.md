# Mese 2 — Contesto verificato e registro delle decisioni

**Data:** 2026-11-16, America/New_York (UTC−05:00)  
**Ambito:** esclusivamente `EnricoBontempi/official_thesis`; nessuna attività HONF.  
**Stato:** ricognizione documentale e proposta operativa; nessuna approvazione o esecuzione del Mese 2 attestata da questo documento.  
**Piano operativo:** [roadmap_month2.md](../../roadmap_month2.md).

## 1. Provenienza e limite sul documento inviato

Ricognizione del remoto `main` a `6fd0f15eecc267f45e68f151fb298e607dd4322f`:

Prima della pubblicazione è arrivato il commit concorrente `a2cebb179712d3bf0160d3fe674b26af45d9fae9`, che semplifica il solo diario. È stato letto e preservato integralmente come nuova base; le note retrospettive dettagliate citate sotto sono state ispezionate nella versione precedente e restano nella cronologia. La nuova versione indica Engaging come host previsto dei training: questo aggiorna la destinazione pianificata, senza attestare GPU/allocazione già disponibili.

- Letti [diario](../RESEARCH_LOG.md), [proposal Markdown](../proposal.md), [roadmap Mese 1](../../roadmap_month1.md), tutti i sette report in `docs/month1/`, [analisi_agents.md](../../analisi_agents.md), configurazioni, implementazioni e test pertinenti, metadati e CSV aggregati del Mese 1, sorgenti del report LaTeX e parti pertinenti della release GLU.
- Nell'albero remoto non risultano `AGENTS.md`, una roadmap Mese 2 o una roadmap generale separata dal piano semestrale del proposal. Esiste un solo branch remoto, `main`; la cronologia ispezionata usa commit diretti. Valgono le istruzioni della sessione e il vincolo del README di conservare intatto `external/GLU_Reconstruction/`.
- `documents_thesis` è la repository separata [EnricoBontempi/documents_thesis](https://github.com/EnricoBontempi/documents_thesis). A `17dd8468ce567a8b2d87ea34b355a8cb692757f0`, `main.tex` è intitolato **Month 1 Progress Report & Baseline Feasibility Study**, non thesis proposal. `main.tex` e `references.bib` coincidono esattamente con le copie in `official_thesis/overleaf_month1_report/`. Non ci sono `\\input` o `\\include` di altri sorgenti; sono referenziate dieci figure e la bibliografia `references.bib`, letta anch'essa.
- La cronologia completa disponibile di `documents_thesis` contiene tre commit del **15 novembre**, dall'iniziale `c5b7017` al finale `17dd846`; l'albero iniziale contiene già il report. Non vi si trova una versione LaTeX identificabile come il proposal inviato il **13 novembre**.
- L'invio del proposal il 13 novembre è riferito dall'utente e annotato nel diario; `docs/proposal.md` è presente dal commit iniziale `75c5db2` dello stesso giorno. Questo è il riferimento di scopo disponibile. **Non è verificata l'identità fra quel Markdown e l'allegato effettivamente inviato.** Non sono attestabili differenze testuali con un LaTeX inviato assente dalle due repository ispezionate.
- È stato controllato anche il contenuto iniziale del precedente `thesis_proposal.tex` disponibile fra i documenti personali: è una bozza del 6/7 novembre con più applicazioni candidate, non una copia identificabile del proposal ufficiale del 13 novembre. Non viene usato per stabilire lo scopo attuale.

La roadmap usa quindi il proposal Markdown per lo scopo, il report LaTeX del 15 novembre come **aggiornamento proposto**, e codice/config/CSV come evidenza tecnica. Recuperare e identificare l'allegato realmente inviato è una voce aperta di provenienza, non un motivo per inventare approvazioni o impedire la preparazione della roadmap. Non è stato modificato alcun proposal o report LaTeX.

## 2. Cosa è completato e cosa rimane aperto

| Elemento | Evidenza ispezionata | Stato e limite |
|---|---|---|
| Audit di 30 condizioni LES, 2001 frame, 128×128 | `02_data_audit_report.md`, manifest `full30_audit_summary.json`, inventario | Completato e registrato nel Mese 1; raw da ~58.6 GB esclusi da Git. Disponibilità e checksum sul futuro host di training da riverificare, senza riscaricare automaticamente. |
| Split 20/3/3/4 per condizione | YAML, script 03 e test | Definito tecnicamente; non equivale ad approvazione dei supervisori. Gli attuali test verificano intervalli/frontiere, non l'intero inviluppo convesso. |
| Normalizzazione | `train_norm_stats.json`, script 03 | Solo condizioni TRAIN, ma su tutti i loro 2001 frame, inclusi quelli della diagnostica seen. Serve una versione limitata a `train_fit` per una diagnostica temporale indipendente. Nessuna evidenza di normalizzazione su condizioni VAL/TEST. |
| Baseline corrette | commit `4fdbcb1`, script 04, `baselines.py`, CSV e meta | Eseguite nel Mese 1; POD selezionata **R=32**, non R=64. Ricontrollo documentale dei CSV effettuato; esperimenti non rieseguiti in questa attività. |
| Test software | `tests/`, diario | Il diario riporta 27 test passati dopo le correzioni. Questo non sostituisce un nuovo controllo sull'ambiente che eseguirà il Mese 2. |
| Accordo scientifico | diario 13 e 16 novembre | Riscontro preliminare positivo della professoressa riferito dall'utente, con cautela sulla temperatura istantanea; nessuna approvazione tecnica specifica documentata. Il diario riferisce risposta di Jason con il codice GLU, non un'approvazione della suite sperimentale. |
| Step 1.6 | roadmap e brief | Letteratura e materiali preparati; la componente “Supervisor Sign-Off” è **aperta** anche se lo step porta una spunta di completamento. |
| Codice GLU | import `ebaae9d`, albero `external/` | Acquisito e ispezionato; nessun adattamento neurale di tesi o training del Mese 2 presente. |
| Risorse | report e diario | CPU-only sul precedente WSL; Engaging previsto nel diario aggiornato, nessuna GPU/allocazione sul cluster verificata qui. |

## 3. Lettura scientifica dei risultati corretti

La domanda è quanto della **temperatura istantanea** sia prevedibile da velocità parzialmente osservata, quanto aggiungano parametri noti e poche osservazioni termiche, e come cambino gli errori su condizioni escluse dal training. La temperatura è disponibile come target numerico supervisionato anche quando è assente dagli input.

La tabella seguente è stata controllata contro `results/tables/month1_feasibility_split_summary.csv`; sono **risultati storici 3C**, non benchmark già equivalenti a un futuro modello 2C. Unità degli errori: K; split storico `test_extrap`, velocità al 5% per GappyPOD.

| Metodo storico | RMSE | Bias firmato medio | Media del bias assoluto per frame | Centered-RMSE |
|---|---:|---:|---:|---:|
| GappyPOD conditioned, K=0 | 153.4 | +41.2 | 43.0 | 138.5 |
| Stesso + AO, K=32 | 140.4 | −0.8 | 18.5 | 138.5 |
| Stesso + T-block, K=32 | 130.2 | +34.8 | 36.1 | 119.3 |
| GappyPOD unconditioned + T-block, K=32 | 125.5 | +5.7 | 17.6 | 123.5 |

- La base POD a R=32 rappresenta T meglio di quanto il suo stimatore ricavi dalla velocità: `OraclePOD_Conditioned` ha Centered-RMSE ~74.6 K in estrapolazione, rispetto a ~138.5 K di GappyPOD conditioned. L'oracle legge T completa: è un diagnostico della **specifica base lineare**, non un modello ammissibile o un limite fisico universale.
- AO è un offset uniforme: lascia invariato l'errore centrato per costruzione. A K=1 può peggiorare RMSE; bias firmato medio vicino a zero non implica piccolo errore di livello per ogni frame. Il T-block può modificare anche la forma e a K=32 la migliora: non è corretto affermare che tutte le baseline lineari correggano soltanto il livello.
- LocalHGB legge patch 3×3 di **velocità densa**, mentre GappyPOD legge osservazioni sparse. Il loro confronto storico non isola l'architettura a parità di informazione.
- La sovrapposizione di velocità simili a temperature diverse suggerisce ambiguità, ma non prova non identificabilità della mappa dal **campo completo** di velocità. Un singolo modello denso che fallisce non prova un limite irreducibile.
- La difficoltà varia anche fra condizioni TRAIN, specialmente con φ: il divario fra medie di split non misura da solo una penalità di extrapolazione.
- `top25pct_iou` attuale usa P75(T vera) anche per sogliare la predizione: non è invariante agli offset. Il diario la accantona; non usarla come metrica di forma nella nuova selezione.

Il report LaTeX successivo prevale sulle conclusioni datate ancora presenti in `05_literature_and_glu_notes.md`, `06_supervisor_review_brief.md` e `analisi_agents.md` (numeri precedenti, R=64, “bias risolto”, “POD incapace di deformare”). Questi file rimangono evidenze storiche; la roadmap non ne assume le raccomandazioni come decisioni.

## 4. Differenze fra proposta iniziale, aggiornamento LaTeX e codice

| Tema | Proposal Markdown | Report LaTeX / codice successivo | Trattamento nel Mese 2 |
|---|---|---|---|
| Primo modello | Deterministico, scelto con il laboratorio | Report propone U-Net densa prima di GLU | Diagnostico denso limitato, da concordare; non “ceiling” fisico. |
| Tempo | Ricostruzione istantanea | Report propone 3 frame causali o simmetrici | Single-frame primario; finestra causale opzionale come **input più ricco**, futuro solo offline separato. |
| Condizionamento | Confronto centrale; studio completo nel Mese 3 | Report propone FiLM e residuo attorno a ParamMean già all'inizio | Toggle minimo per confronto pilota; nessun residuo condizionato nascosto nel ramo unconditioned. Studio esteso nel Mese 3. |
| GLU | Approccio informante | Codice acquisito il 16 novembre | Audit delle interfacce e adattamento minimo; non occorre reimplementare da una descrizione del paper. |
| Incertezza/importanza | Non necessarie al primo modello | Report propone Beta condizionata e assimilazione | Prima MSE deterministica e distanza geometrica uniforme; optional separati solo dopo un risultato che li richieda. Identificare esplicitamente l'ablation rispetto alla release. |
| Test fra condizioni | Campagna completa nel Mese 4 | Risultati esplorativi già letti nel Mese 1 | Selezione su VAL e fold TRAIN; un pilot congelato se concordato, campagna estesa rinviata. I test storici non sono più “mai visti”. |

Nel codice GLU rilasciato `build_pos_value_proj` usa Linear **con bias** + RMSNorm, e `SpatiallyAwareCLSFusion` una fusione gated. `channel_to_encode` esiste già; l'argomento `U` dell'encoder non è usato. Il wrapper Stage 0 salta il propagatore, ma il suo `forward` riceve anche `G_full`: l'adattamento deve dimostrare che T target non entra nelle predizioni. Il mixer dei sensori è un TransformerEncoder, quindi non si può assumere costo solo O(NS) dai cross-attention; il decoder materializza distanze query×sensori con `cdist`. Questi sono motivi concreti per audit, test e profiling prima di aumentare token o copertura.

## 5. Decisioni da registrare prima delle rispettive attività

Nessuna riga seguente è approvata al momento della stesura. Chi esegue compila data, decisore, fonte verificabile e conseguenza sui run; un semplice commit di codice non vale come approvazione scientifica.

| ID | Decisione | Proposta iniziale e conseguenza | Responsabile / stato |
|---|---|---|---|
| D1 | Identità del proposal del 13 novembre | Identificare commit/percorso o copia dell'allegato; fino ad allora scopo da `docs/proposal.md`, limite dichiarato | Enrico; aperta |
| D2 | Split, metriche e uso dei test già esaminati | Conservare 20/3/3/4; selezione solo su VAL/fold TRAIN; nessuna rivendicazione di test intatto | Supervisori + Enrico; aperta |
| D3 | Osservazioni primarie | 2C co-locate (11,12), singolo frame, 5% iniziale; 3C separata, nessuna velocità 3D derivata da vk nascosto | Supervisori + Enrico; aperta |
| D4 | Primo modello e riferimento GLU | U-Net compatta densa, poi stessa rete mascherata e GLU deterministico minimale; scegliere una sola modalità di condizionamento iniziale | Supervisori + Enrico; aperta |
| D5 | Target e ancore T | T LES grezza in K come primaria per continuità, outlier segnalati; ancore ideali sincronizzate, non termocoppie reali già validate | Supervisori + Enrico; aperta |
| D6 | Temperatura assistita | Primario K=9, sweep K=0/1/4/9/16/32 su stesso backbone; prima AO, poi eventuale correzione spaziale | Supervisori + Enrico; aperta |
| D7 | Finestra temporale | Solo opzionale, causale [t−2,t−1,t] con target T(t), Δt reale 0.25 ms; decidere se il guadagno vale l'input aggiuntivo | Supervisori + Enrico; aperta |
| D8 | Risorse e budget | Engaging come host previsto; verificare allocazione e profilare; tetto iniziale proposto 24 GPU-ore per pilot complessivo, senza assumere risorse già assegnate | Enrico + referente risorse; aperta |
| D9 | Gestione dei frame seen e fold interni | Nuove statistiche fit-only versionate, dati di training comuni fra modelli; fold norm/POD/ParamMean ricalcolati senza condizione esclusa | Enrico, con ratifica del protocollo D2; aperta |

Le attività infrastrutturali della roadmap si possono preparare senza scegliere un'architettura nuova; i run scientifici dipendono dalla registrazione delle decisioni applicabili. Questo task ha autorizzato solo preparazione della documentazione e commit/push, non l'avvio di quei lavori.
