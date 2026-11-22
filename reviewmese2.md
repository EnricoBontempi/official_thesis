# Review Mese 2 — sintesi delle tre revisioni e proposte per il seguito

**Data:** 22 novembre 2026  
**Progetto:** Enrico Bontempi — MIT, Deng Lab  
**Tesi:** *Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion*

## 1. Scopo e stato delle decisioni

Questo documento interno riunisce le revisioni indipendenti di Astra, Opus e Neon sul confronto temperature-free, sullo sweep di correzioni con poche temperature e sui training con temperatura in ingresso. Conserva attribuzioni e divergenze per rendere tracciabile la discussione.

Tutti e tre i revisori considerano **chiudibile il blocco sperimentale del Mese 2 senza ulteriori training o inferenze obbligatorie**. Non hanno individuato errori materiali bloccanti negli artefatti accessibili. Questo giudizio non equivale a un audit diretto dei checkpoint e dei CSV per-frame sul cluster, né alla ratifica dei supervisori.

Enrico ha approvato il passaggio alla chiusura documentale: preparazione della presente sintesi, successivo report per i supervisori e aggiornamento dei documenti di progetto. **Nessuna delle tre direzioni del Mese 3 è ancora approvata; nessun nuovo esperimento è autorizzato da questa nota.**

Il report destinato ai supervisori sarà autonomo e scientifico, senza nomi degli agenti o retroscena operativi. Questa nota è invece il verbale interno della revisione.

## 2. Fonti congelate e perimetro della verifica

| Fonte | Versione |
|---|---|
| Pacchetto congiunto e assisted v2 | [d20232ef](https://github.com/EnricoBontempi/official_thesis/tree/d20232ef04873e94708995d119bedc2f6eca2f50) |
| Indice congiunto | [results/month2/joint_step23/package/INDEX.md](https://github.com/EnricoBontempi/official_thesis/blob/d20232ef04873e94708995d119bedc2f6eca2f50/results/month2/joint_step23/package/INDEX.md) |
| S7 temperature-free e residui | [2879c1b5](https://github.com/EnricoBontempi/official_thesis/tree/2879c1b5270be30fb10375f0b300a50cd37d5413) |
| Sweep K | [be18ad00](https://github.com/EnricoBontempi/official_thesis/tree/be18ad007e66fbf2565ad3d5486c6513fc5c3e4c) |
| Review Neon | [e51b2a9e — nota integrale](https://github.com/EnricoBontempi/official_thesis/blob/e51b2a9e6eebaec87339bc9d5c0b2ad53d921c17/docs/month2/reviews/month2_joint_review_neon.md) |
| Review Astra | Documento consegnato: `month2_joint_review_codex(1).md`; branch dichiarato `review/month2-joint-codex-20261122` rimasto al commit base perché la pubblicazione è stata bloccata |
| Review Opus | Documento consegnato: `month2_joint_review_claude_opus55.md`, patch e `check_joint.py`; commit locale dichiarato `1e789e8`, non pubblicato |

Le ultime due note sono state lette integralmente nella sessione di coordinamento; non si dichiara che siano archiviate nel repository o raggiungibili tramite i relativi commit locali. Il loro eventuale deposito integrale resta distinto dalla pubblicazione di questa sintesi.

I revisori hanno rieseguito riconciliazioni e statistiche sulle tabelle aggregate disponibili in Git, verificato identità matematiche e hash e letto parti del codice. Hanno invece soltanto consultato gli esiti dei gate sul cluster. Checkpoint, nuova inferenza e gran parte dei CSV per-frame non erano accessibili direttamente. Il coordinatore ha letto le tre note; i numeri seguenti sono una sintesi delle loro verifiche, non un nuovo ricalcolo indipendente dei dati grezzi.

## 3. Cosa è stato completato

### 3.1 Ricostruzione senza temperatura

Dodici run C149: U-Net masked e GLU adattato, ciascuno con e senza parametri operativi p, tre seed di training. Dieci training nuovi e due U-Net seed 0 riusati con provenienza verificata. C149 comprende 2980 frame TRAIN, 149 per ciascuna delle 20 condizioni; la normalizzazione storica fit-only su 760 frame è rimasta congelata.

Training con velocità osservata al 5%; valutazioni a 1%, 5%, 20%, tre repliche delle maschere, ruoli reference760, seen220 e VAL123. Best, last e best_seen_diagnostic restano identità separate. Completati anche snapshot-swap, baseline C149, diagnostico batch GLU e figure. S7/STOP F è chiuso nel perimetro temperature-free.

### 3.2 Correzione delle predizioni congelate

Sweep K = 0, 1, 4, 9, 16, 32 con AO tau=1, media e mediana dei residui osservati. L’offset viene aggiunto uniformemente al campo; oracle è un riferimento diagnostico che usa il campo vero.

Tre layout casuali fissi, annidati e indipendenti dalla temperatura; ciascuno associato a una replica delle osservazioni di velocità. Il dominio comune esclude sempre i 32 siti, anche a K=0, per confrontare K diversi sugli stessi pixel. La T9 storica resta un ponte separato.

### 3.3 Temperatura direttamente in ingresso

Dodici nuovi training: entrambe le famiglie, p_on/off, seed 0/1/2. K variabile per batch tra i sei valori, siti TRAIN casuali, valutazione sugli stessi layout dello sweep. La selezione assisted usa la media sui sei K della metrica VAL di dominio comune e differisce dal criterio free.

Tutti i training sono conclusi per plateau. Tre GLU sono stati continuati da epoca 1000 e si sono fermati a 1030, 1080 e 1060, senza cambiare i best. Il pacchetto v2 preserva v1 e riusa gli output invariati dopo verifica degli hash. Gli esecutori riportano gate validi e controlli congiunti di compatibilità superati.

## 4. Risultati condivisi dai revisori

VAL123, best, velocità al 5%, RMSE macro di dominio comune in kelvin; media sui tre seed e sulle tre repliche. I confronti regionali e le eccezioni richiedono le tabelle sorgente.

| Configurazione | Free | Free + mediana K=4 | Free + mediana K=32 | Assisted K=0 | Assisted K=32 | Assisted: variazione K=32 meno K=0 |
|---|---:|---:|---:|---:|---:|---:|
| U-Net p_on | 39.70 | 34.85 | 34.54 | 37.66 | 37.55 | −0.10 |
| U-Net p_off | 40.57 | 35.84 | 35.40 | 39.75 | 39.71 | −0.03 |
| GLU p_on | 42.72 | 41.64 | 41.24 | 43.91 | 43.78 | −0.13 |
| GLU p_off | 44.48 | 41.73 | 41.16 | 44.37 | 41.43 | −2.94 |

### 4.1 U-Net e GLU hanno vantaggi diversi

Nelle ricette provate, U-Net ha minore errore medio e strutturale al 5%, migliore seen220 e minore costo di training. La centered-RMSE VAL free è circa 34–35 K contro circa 41 K del GLU. Il GLU presenta vantaggi nel livello termico e nella regione calda, nonché una migliore tenuta in diversi confronti fuori densità, soprattutto all’1%.

Non è un verdetto universale sulle architetture: capacità, implementazioni e traiettorie di ottimizzazione sono diverse. I range dei seed si sovrappongono per alcune metriche. L’indice di seed comune non rende appaiate le inizializzazioni tra famiglie.

### 4.2 La mediana corregge il livello, non la struttura

Su VAL la mediana recupera gran parte del margine dell’oracle scalare già a K=4. Per U-Net p_on, passare da 4 a 32 ancore riduce ancora la RMSE di soli 0.31 K nei layout provati. Questo non stabilisce un numero universale di sensori.

Il beneficio è concentrato nelle condizioni con forte bias. Su seen220 è più piccolo e talvolta negativo: U-Net p_on passa da 42.32 a 42.99 K a K=4, e a 42.08 K a K=32. Una traslazione uniforme lascia invariata la centered-RMSE; può anche peggiorare una regione pur migliorando la metrica globale.

A K=1 media e mediana coincidono. Il forte peggioramento medio su VAL è dominato dal sito sfavorevole della replica 2, mentre le altre due repliche migliorano. Non va descritto come peggioramento in ogni layout o come proprietà di qualsiasi singola misura.

### 4.3 La ricetta assisted ha benefici limitati, con una chiara eccezione

Il confronto causale più informativo sulla disponibilità di T è tra K diversi dello **stesso checkpoint assisted**. Il vantaggio della U-Net assisted sul free originale è già quasi interamente presente a K=0 e non può essere attribuito alle ancore.

GLU p_off migliora rispetto al proprio K=0 in tutti i tre seed, con circa −2.94 K globali e −0.97 K di centered-RMSE su VAL a K=32. Su seen supera la propria alternativa free+mediana. Le altre tre configurazioni hanno benefici globali piccoli.

Su VAL, a K≥4, free+mediana è migliore nella media di gruppo della rispettiva rete assisted; per GLU p_off il margine a K=32 è piccolo e l’ordine varia tra seed. Non è quindi una dominanza per ogni singolo run.

Non è dimostrato che le reti ignorino i valori T, che non possano assimilarli o che il GLU abbia imparato a riconoscere un outlier. Un confronto alle ancore fra modelli diversi non sostituisce una perturbazione controllata dei valori sullo stesso modello. GappyPOD con T-block mostra inoltre che le osservazioni possono contribuire anche alla struttura in una ricostruzione congiunta.

## 5. Proposta Astra — migliorare i getti freddi nel task senza T

### Domanda

Una loss che dia maggiore peso agli errori nei getti freddi migliora la struttura della ricostruzione senza penalizzare il resto del campo?

### Esperimento proposto

Una sola variante della U-Net free p_on C149, velocità 5%, con un termine regionale sui pixel TRAIN con T vera <1800 K. Fissare una sola ponderazione prima dei run e conservare dati, osservazioni, tre seed, budget e regola di checkpoint del comparatore.

La T vera definisce la loss durante il training; non è un input richiesto in inferenza. Valutare RMSE nei getti, con e senza rimozione del bias globale, RMSE globale e centrata, zone calde e risultati per condizione.

### Motivazione

La correzione scalare arriva vicino al limite di livello ma lascia elevati errori locali nei getti. Astra propone di intervenire sul residuo spaziale più evidente, mantenendo il task originario temperature-free e cambiando una sola componente della ricetta.

### Osservazioni del coordinatore

È un’ablation circoscritta e interpretabile. Tuttavia soglia regionale e peso possono specializzare il metodo sul dataset o privilegiare una metrica a danno di altre. Non è overfitting dimostrato: è un rischio da valutare con un disegno di generalizzazione. L’utilità su altre condizioni, geometrie e dataset non segue dal miglioramento su VAL attuale. Occorre motivare la priorità dei getti rispetto alla domanda più generale sulla ricostruzione da osservazioni incomplete.

## 6. Proposta Opus — verificare la generalizzazione dell’errore di livello

### Domanda

Sulle condizioni operative escluse dal training, il livello termico è sbagliato in modo sistematico e prevedibile da p o dalla velocità, oppure l’evidenza attuale dipende da pochi casi?

### Sequenza proposta

1. Misurare il bias su ulteriori condizioni escluse, con protocollo e regole di selezione congelati. Possibili strumenti: fold leave-condition-out sulle condizioni TRAIN, che richiedono training, oppure un pilot separatamente autorizzato su dati tenuti da parte. La storia dei TEST già esaminati nel Mese 1 deve essere preservata.
2. Solo se emerge un comportamento sufficientemente sistematico, provare un singolo intervento mirato, per esempio una predizione residua attorno a ParamMean. Confrontare livello e struttura separatamente.

### Motivazione

Una o due condizioni VAL spiegano gran parte del vantaggio della mediana. p_on non elimina il bias freddo della U-Net su 40NH3_0.9. Prima di ottimizzare ancora su questi casi, Opus suggerisce di verificare che il problema sia trasferibile e meriti un meccanismo dedicato.

### Osservazioni del coordinatore

È la proposta che affronta più direttamente il rischio di adattare la ricerca alle tre condizioni VAL già osservate. Richiede però una definizione precisa di cosa resta indipendente da selezione e sviluppo; non basta chiamare un insieme “fold” o “TEST”. Un solo caso di bias non prova una legge generale. Se il bias non risultasse prevedibile con i metodi provati, ciò non dimostrerebbe che l’informazione termica sia necessariamente indispensabile: dati, ottimizzazione e rappresentazione resterebbero spiegazioni alternative.

La scelta di usare o meno TEST non è stata presa. Non è un requisito per chiudere gli esperimenti già svolti.

## 7. Proposta Neon — assimilazione spaziale dei residui su backbone congelato

### Domanda

Le T sparse possono migliorare la struttura, oltre al livello, se vengono usate esplicitamente come residui rispetto alla predizione free?

### Meccanismo proposto

Congelare U-Net p_on free. Nei siti osservati calcolare:

`r_k = T_obs(x_k) − T_free(x_k)`.

Un correttore riceve residui, coordinate ed eventualmente contesto dello snapshot e produce una mappa spaziale da sommare al campo. A K=0 la correzione deve essere esattamente nulla.

Si distingue sia dalla mediana, che restituisce un solo offset, sia dal training end-to-end già provato, che riceveva T grezza tramite canali o token e poteva cambiare anche la ricostruzione di base.

### Sequenza proposta

1. Baseline spaziale economica: RBF/kernel/ridge oppure POD dei residui, con fit e scelte sui dati TRAIN. Stessi K e layout per il confronto principale.
2. Un solo modulo neurale residuo leggero, tre seed, confrontato contro free, free+mediana e baseline spaziale. Cercare miglioramenti sul dominio comune e nella centered-RMSE, monitorando getti, zone calde, seen e singole condizioni.

### Motivazione

Il backbone U-Net è già competitivo nella struttura. Un modulo dedicato può verificare se l’insuccesso della concatenazione T dipenda dalla formulazione, senza riaddestrare l’intera rete o alterare K=0. La baseline economica aiuta a capire se il vantaggio richieda davvero apprendimento neurale.

### Osservazioni del coordinatore

È una direzione coerente con la ricostruzione condizionata da misure incomplete e potenzialmente riutilizzabile con coordinate e layout diversi. Questa trasferibilità resta da dimostrare, in particolare con rumore e sensori fisici.

Non si garantisce che una misura locale debba correggere solo l’intorno: la propagazione va verificata rispetto alla fisica e ai dati. Non è appropriato imporre in futuro consistenza esatta con misure rumorose senza un modello dell’incertezza.

Un fallimento di una RBF o di un modulo non dimostrerebbe che poche T non contengano informazione strutturale. Il confronto su residui richiede attenzione al fit: i residui del backbone sui suoi stessi frame TRAIN possono essere molto più piccoli e diversi da quelli su condizioni nuove. Il disegno deve esplicitare questo possibile scarto.

Neon propone di rinviare TEST al Mese 4. È una raccomandazione del revisore, **non una decisione approvata**. Le sue indicazioni di costo “circa 4×” non vanno generalizzate: per la campagna free i tempi osservati erano circa 4 h U-Net contro 5.5–7.8 h GLU; i rapporti cambiano fra campagne.

## 8. Confronto delle direzioni

| Direzione | Obiettivo principale | Prima attività | Nuovo training | Rischio principale da controllare |
|---|---|---|---|---|
| Astra | Migliore struttura nei getti senza T | Congelare una sola loss regionale | Sì, variante U-Net su tre seed | Specializzazione alla regione/soglia e compromessi sul campo |
| Opus | Verificare generalizzazione del bias | Disegno di condizioni escluse e valutazione congelata | Sì per fold; non per sola inferenza su checkpoint esistenti e dati ammessi | Poche condizioni, storia della selezione, uso improprio di TEST |
| Neon | Assimilare struttura da poche T | Baseline residua spaziale | Baseline non necessariamente neurale; poi modulo su tre seed | Vantaggio apparente da interpolazione, scarto TRAIN/condizioni nuove, robustezza |

Non sono tre campagne da eseguire automaticamente. Si può scegliere una direzione metodologica principale e adottare un controllo di generalizzazione per valutarla. L’ordine resta da discutere con i supervisori.

## 9. Generalità della tesi e rischio di adattamento a VAL

Enrico ha esplicitamente richiesto che il seguito non diventi una successione di accorgimenti per abbassare l’errore sulle condizioni correnti. La domanda di fondo rimane:

**Quanto e come osservazioni sparse consentono di ricostruire la singola istantanea, e quanto il metodo regge quando cambiano osservazioni e condizioni?**

La mediana è uno stimatore generale, ma è stata scelta dopo un fallimento osservato su VAL. La replica su nuovi seed e layout del medesimo problema non costituisce una conferma indipendente su condizioni o dataset nuovi. Non si può concludere che quattro termocoppie bastino in generale.

Numero e posizione delle misure, rumore, risposta dinamica e vincoli fisici dei sensori, condizioni operative e un eventuale altro dataset sono possibili assi di verifica, non impegni sperimentali già approvati. Occorre scegliere quali siano centrali per la tesi, evitando una lista aperta.

Il riferimento di lavoro suggerito concordemente dai revisori è U-Net free p_on al 5%, con p_off quando p non è disponibile, e la variante free+mediana per K≥4 nei layout ideali provati. È una raccomandazione post-risultati da registrare nella chiusura; non stabilisce un modello definitivo né una garanzia di miglioramento su ogni condizione.

## 10. Correzioni e chiusura documentale

Da consolidare prima della consegna finale, senza nuovi esperimenti obbligatori:

1. **Erratum YAML:** il protocollo congelato non è parsabile alla riga 72. Conservare l’originale identificato dagli hash; descrivere l’errore e, se necessario per riuso, creare una revisione parsabile distinta. Le review non hanno trovato discrepanze numeriche attribuibili al difetto; il codice eseguito usava configurazioni e manifest separati e registrava l’identità del protocollo.
2. **Report 12, precisazioni:** K=1 peggiora le medie di configurazione, non ogni replica; il fallimento medio è dominato dalla replica 2. I benefici mediani GLU a K=32 sono circa 1.48 K p_on e 3.32 K p_off, non indistintamente “2–3 K”.
3. **Baseline residua spaziale prevista ma omessa:** registrare l’omissione e il perimetro delle conclusioni; non affermare che le T non contengano struttura. Decidere esplicitamente l’eventuale rinvio al seguito.
4. **Scelte operative:** registrare free/assisted e alternative preservate come decisioni di sviluppo, non selezione preregistrata.
5. **Riferimenti secondari:** annotare gli `extended_runs.*.last_epoch: null` nel manifest congiunto dove necessario; le epoche reali sono nei riepiloghi e nelle tabelle. Verificare la voce p95 prevista dalla roadmap e riportarla come disponibile, rinviata o non ricostruibile, senza inventare quantili da medie aggregate.
6. **Sintesi:** preparare `docs/month2/13_month2_review.md` o il documento di chiusura concordato, con risultati positivi e negativi, mappe, costi e limiti. Aggiornare poi roadmap, registro, README e diario in modo coerente.
7. **Consegna supervisori:** preparare il report nella repository `documents_thesis`, mantenendo accessibile il Mese 1 e verificando la compilazione. Questo passaggio non è eseguito dalla presente nota.
8. **Stato:** distinguere “blocco sperimentale completato”, “report pronto/pubblicato”, “report inviato” e “discusso/ratificato dai supervisori”. Non registrare come avvenute le ultime due attività prima che accadano.

Le limitazioni da conservare nella sintesi sono: tre condizioni VAL già usate nello sviluppo e nella selezione; tre seed; correlazione temporale; layout e maschere accoppiati; T ideali senza rumore; criteri diversi free/assisted; differenza geometrica GLU free/assisted; capacità diverse tra famiglie. Nessun TEST è stato usato nei tre esperimenti correnti, ma resta la storia dei TEST esaminati nel Mese 1.

## 11. Domande da porre ai supervisori

1. Il seguito deve privilegiare il task solo velocità o l’assimilazione di poche misure T?
2. Quale evidenza di generalizzazione è prioritaria prima di un altro intervento metodologico: nuove condizioni escluse, nuovi layout, rumore o altro dataset?
3. Fra loss regionale, studio del bias e assimilazione residua, quale domanda produce il contributo scientifico più coerente con la tesi?
4. Quali vincoli realistici dei sensori vanno introdotti, e quando?
5. Quale perimetro considerano sufficiente per ratificare la chiusura del Mese 2 e concordare il Mese 3?

**Esito della sintesi:** accordo sulla sufficienza degli esperimenti del Mese 2; tre alternative motivate per il seguito; nessuna nuova campagna avviata o scelta senza discussione.
