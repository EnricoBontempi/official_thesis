> **Nota del 24 novembre 2026.** **SOSPESA — NON ESEGUIRE.** La priorità corrente è la definizione del problema applicativo e lo studio di robustezza a osservazioni non ideali descritto in [roadmaptests.md](roadmaptests.md). Eventuale riattivazione soltanto dopo riesame e decisione esplicita di Enrico.
>
> Il piano storico sotto questa nota è conservato integralmente. Secondo quanto riferito da Enrico, non è stato eseguito.

# Roadmap B — Generalizzazione a condizioni escluse dal training

**Preparazione successiva al Mese 2 · 22 novembre 2026 · America/New_York**  
**Responsabile scientifico:** Enrico Bontempi  
**Esecutore previsto:** Hydrogen  
**Coordinamento:** nuova chat amministratrice delle attività A/B/C.

## Stato e mandato

- [x] Piano B preparato su richiesta di Enrico e pubblicato per il passaggio alla nuova chat.
- [ ] B0 — Contratto verificato e congelato.
- [ ] B1 — Implementazione, controlli di isolamento e smoke reale.
- [ ] B2 — Training dei riferimenti dei due fold.
- [ ] B3 — Valutazione delle condizioni escluse.
- [ ] B4 — Analisi e consegna del contratto per A/C.

**Questa pubblicazione è solo documentale. Non sottomette job, non autorizza implicitamente un esecutore a eseguire la campagna e non costituisce la roadmap complessiva del Mese 3.** La nuova chat assegnerà esplicitamente l'esecuzione entro la matrice sotto, dopo aver controllato lo stato reale e le risorse condivise. Nessuna apertura TEST, nuova famiglia o estensione automatica.

Il blocco sperimentale del Mese 2 è concluso e il report è stato inviato ai supervisori. B non è un requisito retroattivo per quella chiusura. La direzione applicativa resta in definizione con Enrico, il supervisore Polimi e il gruppo MIT; nuovi dati non sono necessari per questo studio preparatorio.

## 1. Domanda e risultato atteso

**L'errore di livello termico osservato su poche condizioni VAL si ripete quando escludiamo dal training altre condizioni operative, oppure dipende principalmente dai casi già studiati?**

B deve misurare, senza introdurre un rimedio:
1. errore globale, livello e struttura su due ulteriori condizioni escluse;
2. dispersione fra tre training e fra tre maschere di osservazione;
3. confronto con frame nuovi di condizioni rimaste in training;
4. se il problema risulta abbastanza coerente da motivare un seguito sul livello oppure se prevalgono errori locali/eterogenei.

Non si allena una nuova architettura e non si cerca di battere il risultato del Mese 2. Due fold mirati non stimano una legge generale, non costituiscono una cross-validation esaustiva e non dimostrano né possibilità né impossibilità fisica della ricostruzione.

### Perché non basta riusare i checkpoint esistenti

I checkpoint C149 hanno visto tutte le 20 condizioni TRAIN originali. Valutarli sulle due condizioni scelte sotto sarebbe una diagnostica interna, non una prova di esclusione. Occorre riaddestrare ciascun riferimento dopo aver rimosso l'intera condizione anche dal fit di normalizzazione e di eventuali prior.

## 2. Fonti e continuità col lavoro concluso

Leggere prima di implementare:

| Fonte | Versione / percorso |
|---|---|
| Sintesi delle revisioni e osservazioni di Enrico | [reviewmese2.md](reviewmese2.md), commit di origine 9a6277f271652a2b6416cad14a6609ddb2c1a021 |
| Proposal e domanda generale | docs/proposal.md |
| Split originale | configs/splits/splits_interp_extrap.yaml |
| Verifica geometrica storica dei candidati | docs/month2/00_context_and_decisions.md, §6 |
| Confronto free S7 | 2879c1b5270be30fb10375f0b300a50cd37d5413; docs/month2/10_s7_c149_model_comparison.md |
| Ricetta di riferimento | configs/experiments/s7/S7-C149-unet-on-s1.yaml, come presente a d20232ef04873e94708995d119bedc2f6eca2f50 |
| Protocollo frame C149 storico | configs/splits/month2_frame_protocol_opus_c149.yaml |
| Stato finale dei confronti con T | d20232ef04873e94708995d119bedc2f6eca2f50; results/month2/joint_step23/package/INDEX.md |
| Report consegnato ai supervisori | EnricoBontempi/documents_thesis, versione del 22 novembre; non modificarlo in B |

Usare i moduli S7 esistenti per modello, sampler, maschere, metriche, checkpoint/resume. Non copiare alla cieca il dataset validator o la normalizzazione storica: il contratto fold-specifico deve propagarsi a ogni apertura del dataset.

Il file storico month2_temperature_study_protocol.yaml ha un errore di parsing già documentato. Non correggerlo in place né importarlo come configurazione eseguibile. Il protocollo B sarà un file nuovo, parsabile e con proprio identificativo.

## 3. Disegno minimo fissato dalla roadmap

### 3.1 Due esclusioni separate, non simultanee

| ID fold | Condizione esclusa da tutto il fit | Condizioni ammesse al fit | Motivazione |
|---|---|---:|---|
| B-lean | 40NH3_0.85.h5 | Le altre 19 del TRAIN originale | Candidato di frontiera lean già proposto sulla sola geometria dei parametri |
| B-rich | 80NH3_1.2.h5 | Le altre 19 del TRAIN originale | Candidato di frontiera ad alta NH3 e phi, già proposto sulla sola geometria |

In B0 ricalcolare l'inviluppo convesso in (frazione NH3, phi) delle 19 condizioni rimaste. I due candidati risultavano esterni nella verifica storica; confermarlo con elenco esatto e coordinate. Non basta il bounding box. Distinguere esterno, bordo e interno con tolleranza numerica registrata.

La scelta dei candidati è deliberata, non una ricerca dei casi con errore massimo. Se membership, geometria o disponibilità non corrispondono, fermare la preparazione e consegnare la discrepanza: non sostituire autonomamente il fold con un caso più favorevole.

Le due condizioni escluse sono diverse fra loro, non un campione rappresentativo di ogni miscela o regime. Non si studia la nuova percentuale NH3 del TEST storico.

### 3.2 Matrice dei training

| Modello | Input | Fold | Seed | Totale |
|---|---|---|---|---:|
| U-Net masked b32 del riferimento S7 | vi/vj 2C osservate al 5%, coordinate, p_on; K=0 | B-lean e B-rich | 0, 1, 2 | **6 training nuovi** |

p_on = phi e percentuale NH3 forniti come parametri operativi noti; non si stima p dal target. p_off non appartiene al blocco minimo. **B non misura quindi l'effetto causale di conoscere p.** Un confronto p_on/off richiederebbe una matrice aggiuntiva esplicitamente decisa, non sei run introdotti automaticamente.

Nessun warm start dai modelli S7: hanno già visto la condizione ora esclusa. Nessun GLU, T in ingresso, nuovo stimatore o tuning della loss in B.

## 4. Separazione fra fit, selezione e valutazione esterna al fold

### 4.1 Ruoli per ciascun fold

| Ruolo | Membership / frame | Uso ammesso |
|---|---|---|
| fit | 19 condizioni × frame 0..1480, passo 10 = **2831 frame** | Ottimizzazione supervisionata |
| norm_fit | Stesse 19 × frame 0..1480, passo 40 = **722 frame** | Fit di tutte le statistiche, mantenendo la convenzione storica a 38 frame/condizione |
| train_reference | I 722 frame sopra | Diagnostica del fit |
| seen_retained | 19 condizioni × frame 1600..2000, passo 40 = **209 frame** | Diagnostica temporale; nessuna riselezione |
| selection_val | VAL123 originale, invariata | Scheduler, early-stop e scelta best |
| outer_fold | Solo condizione esclusa × frame 0..2000, passo 50 = **41 frame** | Valutazione dopo congelamento dei checkpoint |
| original_TEST | Tutti i TEST_INTERP/TEST_EXTRAP originali | **Nessuna lettura di valori/campi o valutazione** |

L'esclusione riguarda **ogni frame e ogni variabile** della condizione trattenuta dai dati usati per fit, calibrazioni, prior o scelte di iperparametri. Il dataset outer è separato dal dataset di training; il fatto che i raw siano nello stesso archivio non rende lecito usarli per statistiche aggregate.

L'outer fold può fornire soltanto metadati per membership, geometria e indice dei frame prima del freeze. Un eventuale controllo di integrità che legge il file non può alimentare statistiche, scelta dei parametri o ispezione di prestazioni.

### 4.2 Cosa significa “indipendente” qui

VAL123 rimane un insieme di sviluppo già ripetutamente osservato. Non diventa una nuova validazione indipendente.

Le due condizioni outer erano TRAIN nel Mese 2 e sono già state analizzate storicamente, anche con baseline. Pertanto sono **nuove esclusioni dai nuovi fit**, non dati mai visti dal ricercatore. Il disegno protegge dall'uso diretto di quelle traiettorie per allenare o selezionare i nuovi checkpoint; non cancella l'adattamento della ricerca al dataset.

Questo è uno studio di generalizzazione di sviluppo a esclusione di condizione. Non va presentato come TEST finale cieco, nested CV completa o conferma indipendente dell'intera ricerca.

### 4.3 Momento di apertura dei risultati outer

Il loop non calcola metriche outer e non usa i loro valori per scheduling o arresto. Congelare prima tutti i checkpoint della matrice B, i criteri e il codice di valutazione; poi valutare entrambi i fold.

Per impedire che il primo risultato cambi gli ultimi training, il job outer deve dipendere dalla chiusura verificata dei sei run. La presenza di artefatti di selezione non abilita automaticamente una valutazione scientifica outer.

Per A/C: condividere subito il contratto e i riferimenti fold-specifici quando disponibili. Se si vuole un confronto successivo il meno adattato possibile, congelare la specifica delle varianti prima di leggerne i risultati outer. Qualsiasi variante progettata dopo averli osservati sarà etichettata esplorativa; non è vietata, ma non è una conferma cieca.

## 5. Normalizzazione e dati senza contaminazione

Per ogni fold rifittare su norm_fit:
- statistiche di velocità e target;
- qualsiasi trasformazione appresa di p;
- ogni prior o artefatto fittato effettivamente usato.

Coordinate e unità fisiche della griglia possono restare fissate dalla geometria, se non dipendono da target aggregati.

**Non riusare il normalizzatore storico 760:** contiene la condizione esclusa. Non usare cache normalizzate globalmente come se fossero già conformi al fold. Preferire array in unità fisiche e costruire nuove cache/statistiche fold-specifiche. Se occorre invertire una trasformazione storica, dimostrare e documentare fedeltà numerica e assenza di trasformazioni irreversibili; in caso contrario partire dai dati fisici disponibili. Non ricostruire i raw o leggere interi archivi pesanti sul login.

Un file deve dichiarare gli ID di condizioni e frame realmente consumati dal fit. Gli hash da soli non provano l'esclusione. Testare il percorso effettivo, non soltanto la lista nel YAML.

Non rifittare POD o GappyPOD per abitudine: il blocco minimo non li usa. Se un successivo comparatore li richiede, deve avere artefatti separati per fold ed essere autorizzato nel relativo incarico.

## 6. Ricetta di training e checkpoint

Riutilizzare la U-Net free S7: stessa architettura, canali, loss MSE full-field e nessuna T osservata nell'input. Fissare:
- seed init/shuffle 0/1/2; seed maschere TRAIN 20261118;
- training con copertura 5%;
- scientific epoch = **760 presentazioni e 95 update**, batch 8, sullo stream deterministico applicato ai 2831 frame ammessi;
- AdamW lr 1e-3, weight decay 1e-4, clip norm 1, warmup 950 update;
- ReduceLROnPlateau: fattore 0.5, patience 5 eval, soglia relativa 0.001, lr minimo 1e-5;
- valutazione selection_val ogni 10 epoche su tre repliche;
- early-stop dopo almeno 200 epoche, 15 eval senza miglioramento relativo 0.001, contatore resettato alla riduzione LR;
- fp32, AMP/TF32/compile disattivati.

760 presentazioni non significa un passaggio completo del pool. Conservare lo stream e registrare update e presentazioni, senza far dipendere la definizione di epoca dal numero di condizioni.

**Checkpoint primario:** primo minimo assoluto della macro RMSE full-field di selection_val al 5%, media delle tre repliche. La soglia di miglioramento dell'early-stop non cambia il criterio del best.

**Secondario:** last a plateau. Non creare o usare un best_outer. best_seen_diagnostic non è richiesto e non può sostituire il primario.

Budget: tranche iniziale 1000 epoche. Per un run non in plateau, è prevista al massimo una estensione fino a 1200 con regole e stato invariati, se esplicitamente inclusa nell'incarico di esecuzione. Nessun reset o nuova ricerca LR. Se a 1200 manca il plateau, conservare e dichiarare run a budget esaurito; non dichiararlo fallito o convergente per convenienza e non estendere ancora senza decisione. A/C dovranno adottare lo stesso limite quando confrontati a pari budget.

Le differenze full-training S7 vs fold includono rimozione di una condizione, statistiche diverse e variazione del pool: **non sono una stima isolata dell'effetto della quantità di dati**.

## 7. Osservazioni e protocollo di valutazione comune

### Primario B

K=0, velocità 5%, repliche 0/1/2. Non eseguire in B lo sweep 1/20% o K: sono altri assi, non necessari a rispondere alla domanda minima.

Per selection_val e seen_retained riusare gli stessi siti persistiti del protocollo storico, filtrando gli ID esclusi. Per outer_fold generare, se non già disponibili, payload deterministici nuovi, target-indipendenti, con namespace B e seed pubblicato; conservarli con hash per frame/replica. Non equiparare stessi seed a stessi siti: verificare gli indici effettivi. Il manifest B è l'autorità per tutte le varianti future di quel fold.

### Metriche e unità

Per frame e replica, in kelvin:
- RMSE, MAE, bias firmato e valore assoluto del bias;
- centered-RMSE, calcolata dopo rimozione del bias del singolo frame;
- media spaziale di T vera e predetta;
- RMSE nei getti (T vera <1800 K), regione calda (>=2100 K), picco (>=2200 K);
- hot IoU e errore di area, con convenzioni di regione vuota esplicite.

Oracle di livello = sottrazione del bias dello stesso frame. È solo diagnostico e non un metodo in inferenza. L'identità RMSE² = bias² + centered² va verificata per frame, non sulle medie macro.

Aggregazione: media dei frame entro condizione, poi media uniforme delle condizioni nel ruolo, poi repliche; seed individuali più media/min/max e deviazione standard descrittiva. Pubblicare le due condizioni outer separatamente; un'eventuale media fra i due fold è solo un riepilogo di due casi selezionati.

Aggiungere p95 della RMSE sui frame, per seed/condizione/replica con definizione del quantile, non ricavandolo dalle medie. Non trattare frame o pixel correlati come repliche indipendenti per test statistici.

### Compatibilità con A/C

B misura il campo intero come primario K=0. Il contratto dovrà anche specificare il dominio comune che esclude i 32 siti T per la futura valutazione di C; nel confronto cross-K C userà quello stesso insieme a ogni K. Conservare il manifest T storico a9c997c96cb44489dd1fc0d4484304b44d8beddbf00b73c25baf4e1922a19e54, scelto senza target.

B non applica correzioni T né ne legge i valori come feature. I siti di C possono essere condivisi come geometria. Non confrontare la RMSE full-field B con quella common-domain C senza ricalcolare il riferimento sul medesimo dominio.

A userà identico split, normalizzatore, maschere, stream, seed e regola di selezione del corrispondente riferimento B. C, se valutata su un outer fold, dovrà usare **il backbone di quel fold** e adattare il correttore solo sulle condizioni ammesse; la U-Net S7 congelata ha già visto quelle condizioni outer e non è un comparatore a esclusione valida.

Su un outer fold, per C sono lecite soltanto le K T esplicitamente osservate durante l'inferenza, se autorizzate dalla sua roadmap; non l'intero target, né il suo uso per fit o selezione. A/C possono preparare codice e lavorare sul protocollo storico mentre B gira, dichiarando che quei risultati restano sviluppati sulle VAL note.

## 8. Sequenza operativa e criteri di uscita

### B0 — Contratto, prima dei calcoli

**Output proposti nuovi** (da creare, non artefatti già esistenti):
- configs/experiments/generalization_b_protocol_v1.yaml;
- configs/splits/generalization_b_folds_v1.json;
- docs/preparatory/b_generalization_handoff.md;
- results/preparatory/b_generalization/prep/.

Verificare parsabilità e risoluzione di ogni path; congelare membership, fit/selection/outer, metriche, matrice sei run, seed, stop e asset di normalizzazione. Esportare per A/C un contratto con versione/hash e un changelog. Una modifica successiva sostanziale richiede una versione nuova, non sovrascrittura.

**Uscita:** due fold geometricamente verificati, conteggi attesi, nessun TEST nel percorso di esecuzione, regole comprensibili e assenza di decisioni implicite lasciate al trainer.

### B1 — Preparazione e smoke, prima dell'array

- Isolamento: mutare dati/campioni della condizione esclusa in un dataset sintetico non cambia statistiche, prior o stream di training. Verificare che i loader usino effettivamente il pool ammesso e stats del fold.
- Il runner non può leggere outer o TEST come eval_role durante training; un namespace diverso non deve aggirare la protezione.
- Verificare p_on e assenza di target nascosti; maschere persistite obbligatorie in valutazione.
- Provare il vecchio evaluator su un checkpoint storico col suo normalizzatore storico e riferimento non vuoto: macro RMSE entro 0.05 K; scarto per-frame dichiarato. **Non** pretendere replay storico dopo aver cambiato normalizzatore.
- Uno smoke reale GPU per ciascun fold, con output non scientifici separati: training, VAL, checkpoint/resume e readback. Nessun forward pesante sul login. Smoke outer solo sintetico prima del freeze, non una prima lettura delle prestazioni reali.
- Verificare che un'interruzione riprenda optimizer/scheduler/RNG/sampler/storia dall'ultima epoca completa. Mai ripartire da best o da zero sotto etichetta resume.
- Profilare il costo e verificare le risorse prima della submission scientifica.

**Uscita:** contratto consumato realmente, smoke validi, matrice pronta, codice/config pushati e checkout remoto congelato. Se il mandato autorizza già la matrice, procedere dopo il gate tecnico senza un'altra richiesta puramente rituale.

### B2 — Sei training, senza valutazione outer nel loop

Sottomettere soltanto la matrice autorizzata. Output nuovi e non sovrascrivibili; log e ledger separati per run; checkpoint e statistiche fold-specifiche. Conservare tentativi interrotti e continuazioni senza contarli come seed aggiuntivi.

Chiudere i sei run con summary, progress, history, stop e checkpoint coerenti. Fallimento tecnico, plateau e budget esaurito sono stati diversi. Un errore non viene convertito in successo da uno script di aggregazione.

**Uscita:** matrice congelata dei best e last con hash e criteri; stato di plateau/budget dichiarato. Nessuna esclusione di seed sfavorevoli.

### B3 — Valutazione congelata e aggregazione

Dopo B2:
1. replay del best/last su selection_val contro il loop, riferimento completo e tolleranza macro 0.05 K, scarti per-frame;
2. valutazione su outer_fold e seen_retained, con train_reference solo diagnostico;
3. verifica chiavi, cardinalità, hash maschere e domini, regioni vuote e metrica indipendente su un piccolo campione;
4. aggregazione solo con tutte le verifiche richieste eseguite.

Per outer non esiste un RMSE storico da riprodurre: usare controlli di completezza e correttezza delle metriche, senza inventare un gate di replay. Un gate mancante non è PASS implicito.

Salvare metriche per-frame e chiavi sufficienti a riaggregare; non servono tutte le mappe. Per ogni fold/seed, fissare il frame qualitativo 1000 e mostrarlo con scale comuni fra comparatori. Aggiungere il peggior frame individuato dopo la valutazione, etichettato come tale, senza usarlo per tuning.

**Uscita:** pacchetto completo oppure BLOCK_REPORT.md con input o verifiche mancanti. Le dipendenze Slurm non sostituiscono la verifica degli artefatti.

### B4 — Lettura scientifica e consegna

Rispondere in linguaggio semplice:
- quale fold mostra bias coerente fra seed e quale no;
- quanto del margine diagnostico riguarda livello e quanto struttura;
- se last cambia il quadro rispetto a best selezionato su VAL;
- se la condizione esclusa è difficile anche nel riferimento storico addestrato con quella condizione, precisando che questo è un confronto diagnostico con contaminazione intenzionale, non un metodo held-out;
- quale singola domanda resta giustificata per A o C.

Nessuna soglia di miglioramento in K va inventata dopo i risultati per decidere successo. Riportare segni, grandezze, dispersione e condizioni individuali. Non chiamare “significativo” un confronto privo di disegno inferenziale adeguato.

**Consegna prevista:** results/preparatory/b_generalization/INDEX.md, tabelle e figure, docs/preparatory/b_generalization_report.md e handoff aggiornato, tutto pubblicato con percorsi remoti degli artefatti pesanti. Stato finale distinto da approvazione umana della direzione della tesi.

## 9. Risorse, automazione e collaborazione

Stimare inizialmente **sei training U-Net dell'ordine di alcune ore ciascuno**, usando come riferimento le 3.7–4.2 ore della campagna S7 free; non usare il tempo della ricetta assisted. Il costo del nuovo percorso fold-specifico va profilato: non promettere un'ora esatta di fine. Coda, preemption e normalizzazione hanno costi separati.

Sono utilizzabili, nei limiti reali da verificare:
- mit_normal_gpu con account/QoS mit_amf_advanced_gpu, L40S;
- mit_preemptable con account mit_general e QoS ammesso, L40S, se resume reale è verificato.

Non combinare Advanced con una partizione che non lo ammette. I quattro slot normal e quattro preemptable menzionati da Enrico sono risorse **condivise fra A/B/C**, non quote per ciascun agente. La chat amministratrice assegna gli slot prima delle submission. Nessuna modifica a job o checkout altrui.

Predisporre, quando autorizzata, una catena cluster training → verifica → freeze → eval → aggregazione, indipendente da PC/IDE. Se si autorizza soltanto la submission, consegnare ID/risorse/percorsi e fermarsi senza polling locale. Se la catena si blocca, conservare output e spiegazione; non avviare recuperi indefiniti o duplicare run.

Hydrogen possiede codice e contratto B in un worktree dedicato, indicativamente branch preparatory/b-generalization. Sonnet (C) e Cursor (A) consumano il contratto pubblicato senza modificarlo in parallelo. Non scrivere contemporaneamente roadmap comuni, report consegnato o diario: la sintesi globale spetta alla chat amministratrice.

## 10. Confini e stop

Fuori dal minimo B:
- campagna su tutte le 20 esclusioni, tuning di p, FiLM o ParamMean;
- modello che predice il bias, nuova architettura, GLU o loss regionale;
- T in input, K sweep, rumore e layout fisici;
- altri dataset o TEST originali;
- conclusioni di impossibilità o promessa di pubblicabilità.

Una difficoltà di cache, normalizzazione o evaluator richiede una correzione tecnica tracciata e smoke mirato, non una modifica silenziosa del disegno scientifico.

**STOP B finale:** i due fold e i tre seed sono valutati e documentati; contratto e riferimenti sono utilizzabili da A/C; limiti e budget sono espliciti. Se l'esito non sostiene un errore di livello generale, il risultato negativo chiude comunque la domanda nei due casi provati.

La prossima decisione appartiene a Enrico: usare questa evidenza insieme alle attività A/C e al contributo dei supervisori per definire la direzione applicativa. Pubblicare B non chiude né anticipa quella decisione.
