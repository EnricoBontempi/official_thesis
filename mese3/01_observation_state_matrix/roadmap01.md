# Roadmap 01 — Observation–State Matrix

**Data:** 25 novembre 2026  
**Missione:** `mese3/01_observation_state_matrix/`  
**Stato:** roadmap discussa nella chat di missione; template A/B/C approvati e pubblicati in [templates_abc.md](templates_abc.md). Aggiornamento 26 novembre 2026: raccolta iniziale revisionata; integrazione F3 proposta, in attesa di revisione del coordinatore e discussione con Enrico ([observation_state_matrix.md](observation_state_matrix.md)).  
**Base organizzativa:** branch `plan/month3-missions`, struttura iniziale `d30ec497`.

## 1. Obiettivo

Costruire una base documentata per capire **quali grandezze descrivono una fiamma, quali segnali possiamo acquisire, come vengono misurati, quali limiti hanno, quali stati possiamo cercare di ricostruire e come validarli**.

La missione serve anche a definire meglio il problema della tesi. Non assume che la scelta finale debba essere temperatura istantanea da velocità sparse, combustione turbolenta, RealPDEBench o un metodo ML specifico. Questi sono il punto di partenza e il patrimonio di lavoro esistente, non criteri per scartare alternative.

La copertura sarà progressiva: ampia nella mappa iniziale, approfondita dove le evidenze indicano un problema diagnostico utile. Non promettiamo un inventario esaustivo di ogni variabile e articolo di combustione.

**Tre risultati collegati:**
1. **A — Matrice delle misure:** grandezza → tecnica → segnale → accesso spaziale/temporale → errori e limiti.
2. **B — Inventario dei dati:** cosa esiste e cosa possediamo davvero, con geometria, tempi, variabili e formati verificati.
3. **C — Catalogo dei problemi:** osservazioni disponibili → stato desiderato → approcci esistenti → validazione → fattibilità e informazioni mancanti.

La matrice non deve diventare un documento infinito che blocca gli altri filoni.

## 2. Apertura scientifica e distinzioni necessarie

La ricognizione considera, quando pertinenti:
- fiamme laminari e turbolente; premiscelate, non premiscelate e parzialmente premiscelate;
- combustori, fiamme da laboratorio e problemi motoristici;
- esperimenti, simulazioni e dati sintetici;
- stati istantanei, medi, statistici, periodici o condizionati alla fase;
- grandezze locali, profili, sezioni, proiezioni e volumi;
- ricostruzione, stima di quantità aggregate e fusione di misure;
- metodi diagnostici e fisici, inversioni classiche, metodi statistici e ML.

Questi assi si annotano separatamente. **RANS/URANS/LES/DNS descrivono il trattamento numerico; laminare/turbolento descrive il regime fisico.** Non sono alternative sullo stesso asse. La ricognizione deve poter confrontare, per esempio, l'utilità di un problema su campi medi RANS con quella di uno su campi istantanei LES, senza trattarne i target come equivalenti.

L'indicazione attuale di Linzheng resta proseguire sul problema di combustione turbolenta ed esplorare i dati motore. L'apertura di questa matrice è una scelta di ricognizione discussa con Enrico: non attribuisce ai supervisor un cambio di direzione e non lo attua automaticamente. Eventuali alternative saranno portate alla discussione con evidenze.

## 3. Struttura A — Matrice delle misure

**Una riga per coppia grandezza–tecnica in una configurazione identificabile.** Se piano/volume, regime o configurazione cambiano sostanzialmente caratteristiche e limiti, usare righe distinte.

| Campo | Contenuto richiesto |
|---|---|
| ID e contesto | Identificativo stabile; tipo di fiamma/apparato e condizioni rilevanti |
| Grandezza fisica | Quantità di interesse, definizione e unità |
| Tecnica | Strumento o diagnostica; eventuale elaborazione necessaria |
| Segnale acquisito | Che cosa viene registrato realmente; relazione con la grandezza fisica e ipotesi di conversione |
| Natura del dato | Scalare, vettore, componenti, campo o quantità integrata |
| Supporto spaziale | Punto, linea, piano, proiezione o volume; coordinate, campo di vista, risoluzione effettiva e spessore se pertinente |
| Supporto temporale | Scatto singolo/sequenza/media; frequenza, esposizione, risposta, ritardo e sincronizzazione |
| Copertura e validità | Regioni inaccessibili, campionamento, lacune, valori invalidi e criteri di rigetto |
| Errori e incertezza | Bias, componente casuale, correlazioni, dipendenza dalle condizioni, calibrazione; definizione della quantità riportata |
| Disponibilità pratica | Dove la tecnica è usata e con quali vincoli; evidenza per quell'apparato, non un sì/no universale |
| Qualificazione della misura | Come accuratezza e incertezza sono state valutate; riferimento e suoi limiti |
| Fonti e stato dell'evidenza | Riferimenti puntuali; documentato, inferito o mancante |

Distinguere errore su un valore osservato, assenza di osservazione, risoluzione finita e risposta temporale. Non ridurli tutti a rumore gaussiano.

Famiglie iniziali da esplorare, senza renderle un elenco chiuso: temperatura; velocità e sue componenti; pressione; composizione/specie; segnali ottici; grandezze derivate o integrate quali indicatori di zona di reazione, rilascio termico e statistiche. Registrare se una quantità è misurata, inferita da un segnale o ricavata da altre variabili.

## 4. Struttura B — Inventario dei dati

**Una scheda per dataset e configurazione**, collegata agli ID della matrice A.

Registrare:
- fonte, versione, accessibilità effettiva, licenza e vincoli di condivisione;
- apparato, combustibile, regime, condizioni operative e natura sperimentale/numerica;
- variabili, definizioni, unità, canali e preprocessing;
- coordinate, orientamento, griglia, mesh, sezione/proiezione/volume e scale fisiche;
- distinzione fra dimensione fisica e componenti disponibili: una griglia 2D non dimostra un flusso fisicamente 2D;
- timestamp, durata, frequenza effettiva, gap, duplicati, continuità e associazione campo–tempo;
- sincronizzazione e co-localizzazione fra modalità; stesso scenario non implica stesso stato istantaneo;
- struttura degli array, dimensioni, organizzazione di traiettorie e snapshot, formati file e metadati;
- disponibilità dichiarata, contenuto ispezionato e possesso/accesso effettivo, tenuti distinti;
- target di riferimento possibili, loro scala di risoluzione e limiti.

Campo vettoriale, mappa, tabella e CSV non sono categorie equivalenti: separare contenuto fisico, struttura numerica e formato di memorizzazione.

### Verifica comune RealPDEBench

Riutilizzare audit e codice esistenti prima di proporre altre verifiche. Ricostruire lo stato aggiornato, senza promuovere inferenze storiche a fatti:
- geometria della sezione, assi e definizione/segno delle componenti i/j/k;
- campi grezzi, osservazioni, canali effettivamente conservati nelle cache e sottocampionamento;
- tempi originali rispetto ai tempi delle cache; incoerenze fra documentazione e dati;
- anomalia già documentata di `40NH3_1.2`: gap temporale interno, duplicato finale e associazione campo–tempo non verificata;
- distinzione fra campi LES, specie numeriche, eventuali proxy ottici e misure OH*;
- significato dell'abbinamento simulazione–esperimento e validazione effettivamente disponibile.

Per gli altri dati, inclusi RANS/URANS motoristici e casi laminari, produrre schede se emergono fonti pertinenti. Non assumere dati PoliMi disponibili senza conferma.

Questa fase è documentale. Eventuali controlli sui payload, download pesanti o nuove elaborazioni saranno proposti separatamente; non serve aprire TEST per preparare l'inventario.

## 5. Struttura C — Catalogo dei problemi e degli approcci

Una riga sintetica per problema; breve scheda di approfondimento solo quando utile.

| Campo | Contenuto |
|---|---|
| Bisogno | Quale limite diagnostico o domanda fisica rende utile la ricostruzione |
| Osservazioni | Segnali, geometria, tempi, qualità e parametri conosciuti a inferenza |
| Stato desiderato | Grandezza, scala, supporto spaziale e natura istantanea/media/statistica |
| Informazione mancante | Cosa non è osservato; ambiguità plausibili e ipotesi necessarie |
| Approcci esistenti | Diagnostici, fisici, classici, statistici e ML; cosa è stato effettivamente dimostrato |
| Validazione | Riferimento, simultaneità, indipendenza, incertezza del riferimento, metriche e limiti |
| Fattibilità | Dati e competenze disponibili; accessi e verifiche mancanti; dipendenze |
| Evidenze | Collegamenti alle righe A, schede B e fonti |

**Input/output/riferimento sono ruoli del singolo problema**, non proprietà permanenti di una variabile.

Distinguere validazione su simulazione, confronto con misure indipendenti, coerenza delle osservazioni e accordo statistico. Un residuo fisico basso o una buona riproduzione degli input non dimostra da solo l'accuratezza dello stato nascosto. Un riferimento RANS medio non valida fluttuazioni istantanee; una sezione 2D non fornisce da sola il riferimento per un volume 3D.

## 6. Fasi operative e punti di discussione

| Fase | Attività | Consegna | Criterio per proseguire |
|---|---|---|---|
| F0 — Impostazione | Tradurre questa roadmap in template comuni e domande di ricerca; assegnare ID e responsabilità | Schema A/B/C e istruzioni comuni per gli agent | Formato comprensibile; campi ignoti ammessi; nessun target imposto |
| F1 — Base e primi casi | Riordinare audit RealPDEBench; leggere criticamente i due paper di Jason; compilare prime righe | Inventario iniziale, due schede critiche e matrice v0.1 | Distinzioni misurato/simulato/inferito e principali lacune esplicite; niente attesa di completezza |
| F2 — Scouting parallelo | Cercare diagnostiche, dati e problemi per famiglie; includere esempi oltre il contesto attuale | Rapporti tematici, registro delle ricerche e proposte di righe A/B/C | Copertura delle famiglie iniziali oppure motivazione delle lacune; fonti puntuali |
| F3 — Verifica e integrazione | Deduplicare; controllare affermazioni decisive e numeri; risolvere o registrare contraddizioni | Matrice v0.2, inventario e catalogo coerenti | Ogni conclusione che orienta una scelta ha evidenza o un'incertezza dichiarata |
| F4 — Sintesi e confronto | Confrontare opportunità e limiti; formulare domande mirate per esperti/supervisor | Sintesi leggibile, alternative motivate e prossime verifiche proposte | Discussione con Enrico; eventuali cambi di problema distinti dalla ricognizione |
| F5 — Manutenzione | Integrare evidenze dagli altri filoni e nuove fonti quando cambiano una decisione | Versioni successive e stato delle domande aggiornato | Aggiornamenti mirati; nessuna revisione esaustiva obbligatoria |

F1 e F2 possono sovrapporsi appena il formato F0 è definito. Le altre missioni possono procedere con le informazioni disponibili e le ipotesi esplicite.

**Punto di arresto della prima ricognizione:** abbiamo una mappa iniziale delle famiglie, le due schede di Jason, l'inventario RealPDEBench, problemi collegati alle evidenze e un elenco preciso delle informazioni mancanti. Dopo un primo giro di ricerca e di riferimenti citati/citanti, estendere lo scouting solo dove una lacuna può cambiare una valutazione. Nuovi paper che ripetono lo stesso punto si registrano senza moltiplicare schede equivalenti.

## 7. Organizzazione degli agent

Questa chat coordina il lavoro. Il coordinatore prepara prompt con perimetro, fonti iniziali, formato e consegna; **Enrico avvia gli agent e riporta qui gli output**, che vengono discussi e integrati. La pubblicazione di questa roadmap non avvia agent né ricognizioni automaticamente.

| Incarico | Responsabilità | Confini |
|---|---|---|
| Dati | Audit documentale RealPDEBench; scouting di dataset alternativi, inclusi campi medi e casi laminari | Nessuna disponibilità presunta, accesso a TEST o campagna numerica |
| Velocità e pressione | Tecniche, segnali, copertura, risoluzione, errori, riferimenti e problemi pertinenti | Coordinamento con missione PIV; nessun modello virtuale deciso qui |
| Temperatura | Diagnostiche puntuali/ottiche, accuratezza, tempi, vincoli e riferimenti | Coordinamento con fusione termica; nessun sensore/layout fissato |
| Specie e ottica | Distinzione segnali–stati, diagnostiche, grandezze derivate, sezioni e proiezioni | Riutilizzare lavoro della missione ottica; nessuna campagna ottica |
| Revisione trasversale | Controllo indipendente di affermazioni decisive, ruoli input/target e validazione | Dopo i primi output, con accesso alle fonti originali |

La ripartizione è adattabile; un incarico può essere diviso o accorpato. Prima di avviare scouting sovrapposti, verificare le consegne già esistenti nelle chat dedicate.

### Contratto comune degli output

Ogni rapporto deve includere:
1. sintesi comprensibile e contesto di applicabilità;
2. righe proposte per A/B/C con ID e fonti;
3. affermazioni documentate, inferenze e informazioni mancanti separate;
4. numeri con unità, definizione dell'errore, condizioni e pagina/sezione/tabella di origine;
5. stato di accesso alle fonti: testo integrale, abstract o fonte secondaria;
6. bibliografia con DOI/URL, ricerche effettuate, esclusioni rilevanti e lacune;
7. implicazioni per i nostri dati senza forzare ogni risultato su RealPDEBench.

Uno scout cerca ampiezza; i paper che sostengono conclusioni importanti ricevono una lettura approfondita. Un full text non accessibile resta segnalato: non si ricostruiscono dettagli metodologici dall'abstract. Nessun numero universale di rumore, frequenza o costo senza contesto. Nessuna pretesa di esaustività.

## 8. Schede comuni dei paper di Jason

Riferimenti iniziali:
- Nie et al., *Reconstructing temperature fields from OH distribution and soot volume fraction in turbulent flames using an artificial neural network*, DOI [10.1016/j.combustflame.2023.113182](https://doi.org/10.1016/j.combustflame.2023.113182).
- Liu et al., *Neural network-based 3D reconstruction of temperature and velocity for turbulent flames from 2D measurements*, DOI [10.1016/j.combustflame.2025.114454](https://doi.org/10.1016/j.combustflame.2025.114454).

Usare il testo integrale, fornito da Enrico o reperito legalmente, prima di chiudere le schede. Per ciascuno registrare problema, apparato/regime, misure o simulazioni, target, geometria, tempi, metodo, informazioni ausiliarie, split, metriche, riferimento e limiti.

Controlli prioritari:
- quali quantità sono realmente disponibili a inferenza, incluse normalizzazione e recupero della scala assoluta;
- differenza fra segnale OH, OH* e relative misure;
- distinzione fra sezioni planari e proiezioni integrate;
- che cosa valida il risultato e quali trasferimenti restano soltanto proposti;
- che cosa è riutilizzabile e quali dati mancherebbero a noi.

Le schede sono comuni alle altre missioni: non duplicarne versioni divergenti.

## 9. Gestione delle evidenze e dei documenti

Per ogni affermazione usare:
- **D — Documentato:** fonte identificata, specificando se ispezione diretta, fonte primaria o rapporto preesistente.
- **I — Inferenza:** ragionamento e ipotesi espliciti.
- **M — Mancante/non verificato:** domanda concreta e possibile via di verifica.

Separare stato dell'evidenza e stato del lavoro. Un rapporto ricevuto non equivale a una fonte primaria verificata. Dove le fonti discordano, conservare entrambe e indicare conseguenze pratiche.

Output previsti, da creare nelle fasi corrispondenti nella cartella della missione:
- `observation_state_matrix.md`: vista sintetica A/B/C, definizioni e collegamenti;
- `data_inventory.md`: schede B, audit riutilizzati e verifiche mancanti;
- `papers/`: due schede comuni di Jason ed eventuali altri approfondimenti necessari;
- `scouting/`: rapporti degli agent e riferimenti delle ricerche, senza copiare interi articoli.

Questi sono output pianificati, non file già prodotti. Tabelle strutturate aggiuntive si introducono solo se aiutano l'uso effettivo. Niente file vuoti di facciata.

La roadmap tiene checklist e decisioni operative; il diario canonico `docs/RESEARCH_LOG.md` conserva gli aggiornamenti scientifici quando si integrano risultati. Non creare un secondo diario. Codice e risultati storici restano nei percorsi canonici.

## 10. Perimetro e stato iniziale

La presente pubblicazione riguarda **solo questa roadmap**. Non esegue scouting, esperimenti, training o inferenze; non scarica dataset; non modifica altri filoni o `documents_thesis`; non contatta autori o supervisor. Gli incarichi successivi saranno formulati nella chat sulla base di questa sequenza. Contatti con Franzelli da coordinare con Deng come indicato da Linzheng.

Una proposta alternativa può emergere anche fuori dal regime o dataset attuale; la sua valutazione appartiene a questa missione, l'eventuale cambio operativo sarà discusso separatamente.

- [x] Obiettivo A/B/C e apertura del perimetro discussi con Enrico.
- [x] Roadmap iniziale definita per la pubblicazione.
- [x] F0 — Template A/B/C e regole comuni approvati e pubblicati: [templates_abc.md](templates_abc.md).
- [ ] F0 — Provare il formato su un esempio A/B/C collegato nella chat.
- [ ] F0 — Preparare i prompt del primo giro e assegnare etichette/responsabilità; nessun agent avviato.
- [ ] F1: inventario iniziale, schede Jason, matrice v0.1. *Raccolta iniziale revisionata dal coordinatore: [data_inventory.md](data_inventory.md), [papers/nie_2024.md](papers/nie_2024.md), [papers/liu_2025.md](papers/liu_2025.md); PR d'origine non mergiate.*
- [ ] F2: scouting tematici. *Primo rapporto revisionato: [scouting/measurement_scouting.md](scouting/measurement_scouting.md); PR d'origine non mergiata.*
- [ ] F3: verifica e integrazione. *Integrazione proposta in [observation_state_matrix.md](observation_state_matrix.md), in attesa di revisione del coordinatore e discussione con Enrico.*
- [ ] F4: sintesi e discussione.
- [ ] F5: aggiornamenti incrementali secondo necessità.

## Fonti di contesto già identificate

- [Indice delle missioni](../README.md) e [README locale](README.md).
- [Proposal](../../docs/proposal.md): contesto iniziale, non vincolo di esclusione della ricognizione.
- [Diario canonico](../../docs/RESEARCH_LOG.md): messaggi di Linzheng/Jason del 25 novembre e distinzione dalle decisioni successive.
- [Ricognizione osservazioni](../../docs/nonideal/00_ricognizione_osservazioni.md): qualificazione delle evidenze geometriche e diagnostiche.
- [Audit temporale](../../docs/month2/mission_baseline_timestamp_audit.md).
- [Confronto free](../../docs/month2/10_s7_c149_model_comparison.md) e [studio con T](../../docs/month2/12_temperature_input_study.md).
- [Sciacchitano 2019](https://doi.org/10.1088/1361-6501/ab1db8): riferimento iniziale per la diagnostica di velocità, da coordinare con missione PIV.

I riferimenti iniziali non costituiscono una revisione bibliografica già completata.
