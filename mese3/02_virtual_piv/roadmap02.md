# Roadmap 02 — PIV virtuale

**Data:** 25 novembre 2026  
**Missione:** `mese3/02_virtual_piv/`  
**Stato:** piano in cinque step concordato con Enrico; esecuzione da avviare un passo alla volta nella chat di missione.  
**Base organizzativa:** branch `plan/month3-missions`, struttura iniziale `d30ec497`.

## Obiettivo e collegamento con la tesi

Costruire osservazioni di velocità motivate da una configurazione PIV sperimentale documentata e rappresentabile con i nostri dati LES, quindi valutare come queste osservazioni influenzano la ricostruzione della temperatura istantanea con U-Net e GLU.

Il punto di partenza è RealPDEBench: campi LES della combustione, osservazioni ideali di velocità campionate al 5% casuale e T numerica come riferimento. La PIV introduce caratteristiche da qualificare: disposizione dei vettori e campo di vista, risoluzione spaziale finita, regioni mancanti, errori e incertezza della misura. La missione deve chiarire quali contano nel caso scelto e come rappresentarli.

**Consegna finale:** un modello virtuale di osservazione con fonti, parametri e approssimazioni espliciti, accompagnato da una valutazione della ricostruzione. La fedeltà sarà riferita a uno scenario identificabile, senza dichiarare una riproduzione universale della PIV o una validazione sperimentale della T ricostruita.

Lo studio precedente, seguito dalla [missione 00](../00_chiusura_studio_degradazioni/README.md), ha introdotto rumore e lacune sintetiche, consultando anche EngineBench. È una base preliminare: non ha qualificato una configurazione PIV del combustore e non includeva filtraggio spaziale o campionamento regolare. I risultati e i 30 specialisti restano nella loro campagna separata.

## Organizzazione e avanzamento

Questa chat è il punto di coordinamento della missione 02: il coordinatore prepara i prompt, Enrico avvia gli agent e riporta le consegne, poi si discutono evidenze e decisioni prima del passo successivo.

**La pubblicazione della roadmap non avvia agent, scouting, download di dataset, implementazioni, inferenze o training.** Gli step vengono avviati con incarichi successivi. I due scouting dello step 2 possono partire in parallelo; la verifica dello step 3 segue la loro consegna. Le fasi successive dipendono dalle decisioni e dai risultati precedenti.

| Step | Attività | Consegna attesa | Discussione prima di proseguire |
|---|---|---|---|
| 1 | Lettura critica di Sciacchitano e collegamento con la tesi | Scheda del paper, implicazioni e domande per gli scouting | Concordare le domande e il formato delle evidenze |
| 2 | Scouting parallelo di dataset e letteratura PIV | Inventario dei dataset e quadro documentato dei problemi della misura | Scegliere dati e ipotesi da verificare direttamente |
| 3 | Verifica delle evidenze sui dataset | Rapporto dei problemi effettivamente osservabili, con statistiche e limiti | Scegliere lo scenario sperimentale rappresentabile |
| 4 | Specifica e costruzione delle osservazioni virtuali LES | Contratto di osservazione, implementazione e qualificazione degli input | Approvare prima il contratto, poi discutere gli input qualificati e il piano di valutazione |
| 5 | Valutazione delle reti esistenti ed eventuali training mirati | Risultati, limiti e decisione sugli interventi utili | Campagna e budget da definire dopo lo step 4; training da decidere dopo le prime valutazioni |

## Step 1 — Leggere Sciacchitano e capire come usarlo

Fonte obbligatoria: A. Sciacchitano (2019), *Uncertainty quantification in particle image velocimetry*, Measurement Science and Technology 30, 092001, DOI [10.1088/1361-6501/ab1db8](https://doi.org/10.1088/1361-6501/ab1db8).

Leggere il testo integrale e produrre una spiegazione comprensibile che colleghi la catena della misura PIV al nostro problema di ricostruzione. Estrarre:
- differenze fra osservazione numerica puntuale e vettore PIV;
- ruolo di finestra di interrogazione, passo dei vettori, sovrapposizione e supporto del foglio laser;
- errori casuali, bias, vettori spuri, invalidazione, correlazioni ed effetti del processamento;
- distinzione fra errore e incertezza, stime a priori e a posteriori;
- cosa richiede immagini di particelle, calibrazione o riferimenti indipendenti e cosa può essere verificato sui soli campi di velocità;
- implicazioni per le osservazioni virtuali, senza fissare ancora operatori o livelli numerici.

**Consegna:** scheda con riferimenti a sezioni/pagine, rilevanza per la storia della tesi, limiti dello studio nonideal e lista di domande per lo step 2. La ricognizione iniziale svolta nella chat è materiale preparatorio, non chiusura automatica di questo step.

**Punto di discussione:** concordare quali informazioni chiedere agli scout e come registrarle.

## Step 2 — Due scouting ampi e complementari, in parallelo

### Agente A — Dataset PIV

Cercare archivi PIV sperimentali sul web, privilegiando combustione e fiamme swirl ma includendo altri flussi quando offrono evidenze utili. Partire anche da EngineBench e dai riferimenti emersi nello step 1, verificando ciò che è realmente accessibile.

Per ogni dataset registrare:
- apparato, flusso, combustibile, condizioni e pertinenza alla tesi;
- DOI/URL, versione, licenza, disponibilità dichiarata e accesso effettivo;
- immagini di particelle, vettori istantanei, medie/RMS, sequenze e altre modalità disponibili;
- coordinate, unità, componenti, griglia, campo di vista, risoluzione e tempi;
- flag di validità, maschere, indicatori di qualità e incertezza;
- processamento: validazione, filtraggio, interpolazione, riempimento e padding;
- forma dei file/array, documentazione e campioni effettivamente ispezionati;
- possibili utilizzi e informazioni mancanti.

Non confondere dati citati in un articolo, dati scaricabili e dati già posseduti. Eventuali campioni ispezionati devono avere una provenienza identificabile. Concordare nei prompt i limiti dei download e dell'ispezione: nessun download massivo automatico.

### Agente B — Problemi e caratteristiche della PIV sperimentale

Cercare review, paper metodologici ed esperimenti per ricostruire quali limitazioni si presentano, perché e come appaiono nei dati. Dare priorità a fonti primarie e misure in combustione; approfondire riferimenti citati e citanti quando servono.

Per ciascun fenomeno distinguere:
- meccanismo fisico, ottico o algoritmico;
- condizioni in cui compare e rilevanza per il caso;
- effetto su valori, copertura, risoluzione o dipendenze spazio-temporali;
- eventuali ordini di grandezza, frequenze e metodi con cui sono stati misurati;
- tracce osservabili nelle immagini, nei vettori e nei metadati;
- trattamento sperimentale e conseguenze sul dato rilasciato.

Nessuna lista universale dei “tre problemi principali” assunta in anticipo. Rumore, outlier e lacune sono ipotesi da approfondire, non conclusioni già stabilite.

### Qualità e consegna dei due scouting

Richiesta una ricerca ampia e approfondita, con bibliografia tracciabile, registro delle ricerche, esclusioni motivate e lacune. La qualità dipende dalla copertura delle domande e dalle evidenze utilizzabili, non dal solo numero di titoli.

Leggere integralmente i lavori decisivi; segnalare quando è accessibile solo abstract o materiale secondario. Separare dati documentati, inferenze e informazioni mancanti. Riportare numeri con unità, definizione e contesto: una quantità “±” non è automaticamente una deviazione standard gaussiana.

**Consegne:** inventario dei dataset e rapporto della letteratura, collegati tramite identificativi e fonti. Il primo giro può chiudersi quando copre le domande concordate oppure documenta precisamente ciò che manca; ulteriori ricerche devono risolvere lacune utili alla scelta.

**Punto di discussione:** selezionare i dataset accessibili e le affermazioni da controllare nello step 3.

## Step 3 — Verificare i problemi sui dati

Assegnare a un agente verificatore i due rapporti, le fonti e i dataset selezionati. Deve confrontare le affermazioni della letteratura con il contenuto effettivo dei dati, cercando anche fenomeni non previsti.

Per ogni coppia dataset–fenomeno classificare:
1. presente e misurabile nei dati ispezionati;
2. documentato dagli autori ma trattato/corretto prima del rilascio;
3. non riscontrato nel campione ispezionato, con ambito della verifica;
4. non verificabile con il materiale disponibile.

**L'assenza di buchi nei vettori finali non prova l'assenza di rigetti nella misura:** i dati possono essere stati ripuliti e interpolati. Uno zero non dimostra da solo una misura mancante; la variabilità turbolenta non misura da sola il rumore. Per stimare errori servono riferimenti, prove dedicate o indicatori d'incertezza qualificati.

Quantificare, dove supportato:
- numero/frazione di vettori validi, invalidati e sostituiti;
- dimensioni fisiche, posizione, contiguità e persistenza delle lacune;
- passo, risoluzione e copertura;
- outlier, bias, incertezza e correlazioni soltanto con un metodo giustificato;
- variazioni fra condizioni e frame.

Indicare sempre campione, denominatore, unità, metodo e limiti. Una frequenza vale per il dataset e le condizioni analizzate: non diventa una frequenza universale della PIV.

**Consegna:** elenco verificato dei fenomeni, cause documentate, frequenze quando stimabili, esempi rappresentativi e discrepanze rispetto allo scouting. Controlli riproducibili, senza training o inferenze delle nostre reti.

**Punto di discussione:** scegliere insieme uno scenario coerente, o poche alternative motivate. Non combinare automaticamente difetti e parametri provenienti da apparati incompatibili.

## Step 4 — Specificare e costruire le osservazioni virtuali

Usare il rapporto dello step 3 e la verifica comune RealPDEBench per definire come osservare i campi LES nello scenario scelto.

### 4A — Contratto da discutere prima dell'implementazione

Definire:
- piano, componenti, campo di vista e regione della T da ricostruire;
- layout, passo, origine della griglia e risoluzione spaziale;
- eventuale approssimazione del filtraggio in piano;
- regioni inaccessibili, invalidazione e trattamento dei vettori mancanti;
- modello dell'errore, significato dell'incertezza, correlazioni e bias inclusi;
- ordine delle trasformazioni in unità fisiche e normalizzazione;
- rappresentazione equivalente per U-Net e GLU, inclusi vettori fuori dai nodi LES;
- controlli necessari per distinguere layout, quantità di osservazioni, filtraggio ed errore;
- provenienza e categoria di ogni parametro: documentato nel riferimento, trasferito con ipotesi, oppure scelto come sensibilità;
- fenomeni esclusi perché non rappresentabili o non necessari.

Una sola slice non consente di ricostruire arbitrariamente media attraverso lo spessore del foglio, traiettorie delle particelle o ottica tridimensionale. Non introdurre dettaglio numerico oltre quello supportato dalla griglia LES e dalle informazioni sui suoi filtri.

Lacune dipendenti da T o dalla fiamma richiedono una giustificazione separata e controlli sull'informazione portata dalla maschera. T resta il riferimento nascosto del compito; ogni uso di T nel generatore deve essere dichiarato e discusso prima.

**Punto di discussione:** approvare il contratto e la fattibilità prima di implementare.

### 4B — Implementazione e qualificazione degli input

Implementare lo scenario concordato, conservando intatti i dati sorgente, le configurazioni e i risultati storici. Documentare per ogni trasformazione cosa cambia e perché, versioni, seed e convenzioni.

Qualificare geometria, supporto del filtro, maschere, conteggi, unità e proprietà statistiche previste, con esempi visivi e controlli leggeri su TRAIN. Nessuna ottimizzazione del modello di misura sulle prestazioni delle reti; nessun accesso a TEST. Le specifiche delle osservazioni si fissano prima delle valutazioni comparative.

**Consegna:** specifica versionata, codice, verifiche e rapporto degli input virtuali con limiti espliciti.

**Punto di discussione:** valutare gli input ottenuti e definire la campagna dello step 5.

## Step 5 — Valutare le reti; decidere eventuali training mirati

La campagna sarà definita dopo lo step 4: niente numero di training, matrice completa o budget fissati ora.

Prima valutare U-Net e GLU già allenate, con osservazioni e riferimenti condivisi. Riutilizzare, dove applicabili, pipeline, metriche e controlli della missione 00; documentare gli adattamenti richiesti dal nuovo compito. Stabilire prima delle esecuzioni checkpoint, condizioni, repliche, controlli e modalità di confronto.

Analizzare accuratezza globale, livello termico, struttura, zone osservate/non osservate e singole condizioni. Usare confronti appaiati quando si isola un effetto; distinguere variabilità della misura, seed dei modelli e incertezza della ricostruzione. Il protocollo deve esplicitare il riuso di VAL per selezione e sviluppo. L'eventuale valutazione finale su TEST richiede una decisione successiva esplicita.

Poi discutere se servono retraining mirati o un miglioramento metodologico. La missione 00 orienta le ipotesi e può evitare duplicazioni, ma non garantisce che le tendenze restino valide con layout e risoluzione diversi. Non è prevista automaticamente un'altra campagna da 30 training.

**Consegna:** rapporto delle reti esistenti e proposta motivata degli eventuali training; se autorizzati successivamente, risultati della campagna e conclusioni sullo scenario.

## Coordinamento con le altre missioni e documenti

Riutilizzare la [roadmap 01](../01_observation_state_matrix/roadmap01.md), i suoi [template A/B/C](../01_observation_state_matrix/templates_abc.md) e gli output disponibili: matrice delle misure, inventario dei dati e catalogo dei problemi. Non ripetere l'audit generale RealPDEBench; segnalare le informazioni necessarie per PIV, comprese convenzioni delle componenti, geometria della slice, filtri e ricampionamento. Un campo non verificato resta aperto.

Gli scouting PIV devono fornire elementi riutilizzabili dalla missione 01. Coordinare incarichi sovrapposti prima di avviarli. Lo stesso contratto di osservazione potrà servire a DMF-Gen e fusione termica, senza avviarne qui le campagne.

Riferimenti di contesto:
- [README locale](README.md) e [indice Mese 3](../README.md).
- [Ricognizione osservazioni](../../docs/nonideal/00_ricognizione_osservazioni.md).
- [Studio di robustezza sintetico](../../docs/nonideal/02_block2_robustness.md).
- [Handoff specialisti](../../docs/nonideal/03_specialist_handoff.md) e [limiti dello smoke e budget](../../docs/nonideal/04_specialist_smoke_criteria_and_budget_policy.md).
- [Diario canonico](../../docs/RESEARCH_LOG.md), incluse le indicazioni dei supervisori del 25 novembre.

Gli output degli step saranno creati nella cartella della missione quando prodotti, senza file vuoti preventivi; codice e risultati nei percorsi canonici della repository. La roadmap mantiene l'avanzamento operativo; gli aggiornamenti scientifici si registrano nel diario canonico quando si integrano risultati. `documents_thesis` resta una fonte consultabile, senza modifiche.

## Stato iniziale

- [x] Obiettivo e sequenza in cinque step concordati con Enrico.
- [x] Roadmap preparata per la pubblicazione sul branch di pianificazione.
- [ ] Step 1 — Lettura strutturata di Sciacchitano e discussione.
- [ ] Step 2 — Preparazione dei prompt e avvio dei due scouting paralleli.
- [ ] Step 3 — Verifica sui dataset e sintesi.
- [ ] Step 4A — Scelta dello scenario e approvazione del contratto.
- [ ] Step 4B — Implementazione e qualificazione delle osservazioni.
- [ ] Step 5 — Definizione ed esecuzione della valutazione; eventuali training da decidere.

Nessun agente o nuovo esperimento è avviato dalla pubblicazione di questo documento.
