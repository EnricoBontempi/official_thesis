# Template comuni A/B/C — Observation–State Matrix

**Stato:** template approvati da Enrico nella chat di missione e pubblicati per F0.  
**Roadmap:** [roadmap01.md](roadmap01.md).  
Questa è la struttura da riutilizzare per le consegne. Non contiene schede scientifiche già compilate; la prova su un esempio e i prompt degli incarichi restano da preparare.

## Uso e perimetro

Copiare soltanto il template pertinente al proprio incarico. Una riga breve alimenta la vista generale; la scheda contiene gli approfondimenti necessari. Non occorre compilare ogni campo prima di condividere una prima versione: scrivere **M — non verificato**, con la domanda aperta. Usare **N/A**, motivandolo, per campi non applicabili.

La ricognizione può includere fiamme laminari o turbolente, dati sperimentali o numerici, stati medi o istantanei e approcci fisici, classici, statistici o ML. Il punto di partenza della tesi non è un criterio di esclusione. Regime fisico, trattamento numerico e natura del target si descrivono separatamente.

Non duplicare gli approfondimenti già prodotti dalle missioni PIV, fusione termica o ottica: collegarli e integrare le informazioni pertinenti.

### Identificativi e collegamenti

- Misura: `A-<tema>-<incarico>-<numero>`.
- Dataset/configurazione: `B-<tema>-<incarico>-<numero>`.
- Problema: `C-<tema>-<incarico>-<numero>`.
- Fonte locale al rapporto: `S01`, `S02`, ecc.

Il coordinatore assegna un'etichetta di incarico prima dell'avvio per evitare collisioni. Conservare gli ID durante le revisioni; eventuali accorpamenti devono mantenere il rinvio agli ID originali. Una proposta può collegare righe già esistenti; se un collegamento non è ancora noto, indicarlo senza inventare una scheda.

## A — Come si misura una grandezza

**Unità della scheda:** grandezza + tecnica + configurazione. Separare le configurazioni quando cambiano sostanzialmente accesso spaziale/temporale o limiti. La sola etichetta “temperatura” o “PIV” non identifica una misura.

### Riga sintetica da copiare

| ID | Grandezza e contesto | Tecnica e segnale | Spazio e copertura | Tempo | Errori e limiti | Disponibilità e fonti |
|---|---|---|---|---|---|---|
| A-… | … | … | … | … | … | … |

### Scheda da copiare

**ID e titolo:** A-… — …  
**Sintesi:** poche frasi che spieghino che cosa si misura, dove è utile e il limite principale.

| Campo | Contenuto da compilare |
|---|---|
| Grandezza e contesto | Definizione, unità, apparato/tipo di fiamma e condizioni rilevanti |
| Tecnica e segnale | Strumento e segnale registrato; passaggi e ipotesi per ricavare la grandezza; quantità direttamente acquisita o inferita |
| Spazio | Scalare/vettore/componenti; punto, linea, piano, proiezione o volume; coordinate, campo di vista, risoluzione effettiva, spessore se pertinente |
| Copertura e validità | Distribuzione delle misure, regioni non osservate, lacune, rigetto dei valori e maschere di validità |
| Tempo | Scatto/sequenza/media; frequenza, esposizione, risposta, ritardo, sincronizzazione e simultaneità |
| Errori e limiti | Incertezza, bias, correlazioni, interferenze, calibrazione e dipendenze dalle condizioni; numeri solo con definizione e fonte |
| Qualificazione della misura | Come accuratezza e incertezza sono state controllate; riferimento utilizzato e suoi limiti |
| Disponibilità pratica | Esempi documentati e vincoli tecnici; contesto al quale vale la disponibilità |
| Fonti e aperti | Fonti puntuali, evidenze D/I/M e domande ancora irrisolte |
| Collegamenti | Schede B e C pertinenti, se disponibili |

Tenere distinti errore del valore, mancanza di osservazioni, risoluzione finita e risposta temporale. Distinguere supporto fisico e numero di componenti: un campo vettoriale osservato su un piano non è automaticamente un flusso fisicamente bidimensionale.

## B — Che cosa contiene un dataset

**Unità della scheda:** dataset + configurazione/versione identificabile. Separare modalità o configurazioni con geometria e tempi diversi quando serve.

### Riga sintetica da copiare

| ID | Dataset/configurazione | Contesto e provenienza | Variabili | Spazio e tempo | Struttura dei file | Accesso, riferimento e lacune |
|---|---|---|---|---|---|---|
| B-… | … | … | … | … | … | … |

### Scheda da copiare

**ID e titolo:** B-… — …  
**Sintesi:** poche frasi su cosa offre il dataset e cosa è stato realmente verificato.

| Campo | Contenuto da compilare |
|---|---|
| Identità e accesso | Fonte, versione, DOI/URL, licenza; pubblico/su richiesta; accesso effettivo e vincoli di condivisione |
| Contesto fisico e provenienza | Apparato, combustibile, condizioni e regime; esperimento/simulazione/sintetico; trattamento numerico e natura istantanea/media/statistica |
| Variabili | Nomi, definizioni, unità e canali; misurate, simulate, derivate o sintetizzate; preprocessing |
| Spazio | Coordinate, orientamento, mesh/griglia, supporto, risoluzione e componenti; geometria fisica rispetto a quella dei dati |
| Tempo | Scatti/sequenze, timestamp, durata e frequenza; gap, duplicati, continuità, associazione campo–tempo e sottocampionamenti |
| Relazione fra modalità | Co-localizzazione, simultaneità, sincronizzazione e registrazione; stesso scenario rispetto a stesso stato istantaneo |
| Struttura dei file | Formati, dimensioni degli array, ordine assi/canali, unità, metadati e organizzazione di traiettorie/snapshot |
| Riferimenti utilizzabili | Quantità disponibili per valutare ricostruzioni, qualità/scala del riferimento e limiti |
| Verifiche e lacune | Separare dichiarato dalla fonte, ispezionato direttamente ed effettivamente disponibile; domande e possibili verifiche |
| Collegamenti | Righe A e problemi C pertinenti; audit preesistenti da riutilizzare |

“Campo vettoriale”, “array” e “CSV” descrivono livelli diversi: riportare contenuto fisico, struttura numerica e formato del file separatamente. Non presumere accesso o sincronizzazione dalla sola esistenza di un dataset.

Per RealPDEBench, seguire anche le verifiche comuni della roadmap: geometria, variabili, cache rispetto ai grezzi, temporalità e distinzione fra simulazioni e misure. Riutilizzare gli audit prima di richiedere nuovi controlli; questa scheda non autorizza letture di TEST o elaborazioni scientifiche.

## C — Quale problema possiamo formulare

**Unità della scheda:** un problema con osservazioni e target definiti. Gli approcci sono possibilità documentate, non soluzioni garantite.

### Riga sintetica da copiare

| ID | Bisogno | Input disponibili | Stato desiderato | Approcci esistenti | Validazione | Fattibilità e aperti |
|---|---|---|---|---|---|---|
| C-… | … | … | … | … | … | … |

### Scheda da copiare

**ID e titolo:** C-… — …  
**Sintesi:** poche frasi sul limite diagnostico, sulla quantità desiderata e sull'evidenza di fattibilità.

| Campo | Contenuto da compilare |
|---|---|
| Bisogno | Limite diagnostico o domanda fisica; perché è utile recuperare questa quantità |
| Input disponibili | Segnali e informazioni realmente disponibili durante l'uso, incluse coordinate, parametri operativi, calibrazioni e storia temporale |
| Stato desiderato | Grandezza, unità, supporto, scala e natura istantanea/media/statistica; informazioni non osservate e ipotesi necessarie |
| Approcci esistenti | Diagnostica, inversione fisica, metodo classico/statistico/ML; cosa è dimostrato e cosa è solo proposto |
| Validazione | Riferimento, provenienza, indipendenza, simultaneità/co-localizzazione ove richieste, incertezza, metriche e limiti della conclusione |
| Fattibilità e aperti | Dati e competenze disponibili; informazioni/accessi mancanti; ambiguità e verifiche che possono cambiare la valutazione |
| Fonti e collegamenti | Evidenze D/I/M, fonti puntuali e ID delle schede A/B |

Input, target e riferimento sono ruoli del problema, non proprietà fisse della variabile. Segnalare informazioni termiche o altre quantità del target eventualmente usate per normalizzazione, recupero della scala assoluta o preprocessing.

Tenere separati validazione numerica, confronto sperimentale, accordo statistico e coerenza delle osservazioni. Esplicitare i limiti del riferimento e del protocollo. Non attribuire a un target medio la validazione delle fluttuazioni istantanee o a una sezione la validazione di un intero volume.

## Regole comuni sulle evidenze

Per le affermazioni importanti, indicare lo stato a livello di singola affermazione, non soltanto dell'intera scheda:

| Marcatura | Significato |
|---|---|
| **D — Documentato** | Fonte identificata; dichiarare se letta direttamente, fonte primaria o rapporto preesistente |
| **I — Inferenza** | Ragionamento, premesse e condizioni espliciti |
| **M — Mancante/non verificato** | Domanda concreta e possibile via di verifica |

- Citare DOI/URL e pagina, sezione, figura o tabella quando disponibili.
- Dichiarare se la lettura riguarda testo integrale, sezioni specifiche, abstract o fonte secondaria. Disporre del PDF non equivale ad averlo letto interamente.
- Per ogni numero riportare unità, definizione, condizioni e riferimento; distinguere bias, deviazione standard, RMSE, intervalli e livelli di copertura.
- Non presentare valori di errore, frequenza o costo come universali. Non convertire ordini di grandezza in specifiche del nostro apparato.
- Non ricostruire dettagli mancanti dall'abstract. Conservare le contraddizioni fra fonti e indicarne le conseguenze.
- Aprire alle alternative: spiegare la pertinenza di casi laminari, medi o RANS senza escluderli per distanza dal punto di partenza.
- Scrivere una sintesi comprensibile: cosa abbiamo imparato e quale domanda resta aperta.
- Non forzare tutti i risultati su RealPDEBench o su un metodo ML.

## Fonti e consegna di un rapporto

### Registro fonti da copiare

| ID fonte | Citazione e DOI/URL/versione | Accesso e parti lette | Localizzatore | Affermazioni/schede supportate |
|---|---|---|---|---|
| S01 | … | … | … | … |

### Struttura della consegna

1. Sintesi comprensibile e contesto di applicabilità.
2. Righe sintetiche A/B/C pertinenti all'incarico; nessun obbligo di inventare righe nelle altre categorie.
3. Schede di approfondimento necessarie, con ID e collegamenti.
4. Fonti e stato della lettura.
5. Breve registro delle ricerche: query, percorsi consultati, esclusioni rilevanti e lacune.
6. Informazioni mancanti e prossime verifiche proposte.

Queste sono istruzioni comuni, non prompt di incarico. Perimetro, responsabilità, fonti iniziali e modalità di consegna verranno specificati nei prompt. Gli agent vengono avviati da Enrico; nessuno scouting, esperimento o contatto esterno viene avviato con la pubblicazione dei template.
