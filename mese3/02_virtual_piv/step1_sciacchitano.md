# Missione 02 — Step 1: scheda critica integrata di Sciacchitano (2019)

**Versione:** v2, integrazione della lettura del coordinatore con le schede indipendenti di Opus e Neon fornite da Enrico.  
**Stato:** step 1 approvato e dichiarato completato da Enrico il 25 novembre 2026 (America/New_York), dopo il confronto delle tre letture. Prossimo passo: preparare gli incarichi dello step 2 nella chat di missione.  
**Perimetro:** lettura del paper e traduzione in domande per gli scouting. La pubblicazione riguarda questa scheda e gli aggiornamenti documentali di stato. Nessun parametro definitivo, codice di osservazione, esperimento, training o inferenza.

## 1. Fonte, lettura e convenzioni

A. Sciacchitano, *Uncertainty quantification in particle image velocimetry*, Measurement Science and Technology **30** (2019), 092001. DOI: [10.1088/1361-6501/ab1db8](https://doi.org/10.1088/1361-6501/ab1db8).

Fonte effettivamente letta: il PDF integrale fornito da Enrico, `Sciacchitano_2019_Meas._Sci._Technol._30_092001.pdf`, versione pubblicata. Il file ha **32 pagine PDF**: una copertina editoriale senza numero e le pagine stampate **1–31**. Nel seguito **p. n indica sempre il numero stampato**; in questa copia la pagina PDF è n + 1. Per esempio, la figura 4 è a p. 5, cioè pagina PDF 6. Questa corrispondenza non va trasferita automaticamente a copie con copertine universitarie aggiuntive.

Letti abstract, corpo dell'articolo (§§1–6), tabelle e bibliografia. Controllate visivamente, oltre all'estrazione testuale, la catena di misura in figura 4, le equazioni 6–7, la tabella 4, l'esempio di verifica in figura 13 e la pagina dell'equazione 16. I lavori citati dalla review **non sono stati letti integralmente in questa attività**: quando ne riferisco un risultato, l'evidenza è la sintesi di Sciacchitano, non una verifica del lavoro originale. Non è stata svolta la ricerca estesa prevista nello step 2. Per questa integrazione sono state lette entrambe le schede complete e riletta la v1; verificati nuovamente sul PDF i passaggi su overlap e distanza 2h, le definizioni di incertezza, i requisiti dei metodi UQ e le possibili anomalie di notazione alle pp. 21–23. Le schede di Opus e Neon sono contributi di lettura, non fonti sperimentali aggiuntive. Il confronto e le correzioni sono riassunti al §13.

Per rintracciare la copia letta: SHA-256 `bb3879b3d54a0660365b7458d239d085d658fa2bf70f782d38015d88f5a65b1d`.

**Etichette usate:**

- **D — Documentato:** contenuto letto nel paper, oppure contesto della tesi già documentato, identificando quale dei due.
- **I — Inferenza/proposta:** conseguenza ragionata per il nostro lavoro; non un'affermazione dell'autore sulla nostra tesi.
- **M — Mancante:** informazione che serve ancora a verificare o scegliere qualcosa.

Le etichette descrivono il sostegno di un'affermazione, non il suo grado di utilità. Una proposta può essere utile senza essere già validata.

## 2. Che cosa ci serve portare a casa

**Il risultato di questo step è imparare a specificare e verificare l'osservazione che daremo alle reti.** Sciacchitano ci aiuta a distinguere gli effetti della misura e a riconoscere le evidenze necessarie per descriverli. I dataset e gli esperimenti dello step 2 dovranno poi fornire il contesto e i valori specifici.

Il paper è una review metrologica, soprattutto sulla PIV planare a due o tre componenti. Non propone una rete per ricostruire T, un dataset di combustione o un generatore universale di degradazioni. Gli esempi su vortici, getti e altri flussi servono a discutere le misure e la loro incertezza.

La sua utilità per noi è diretta: studiamo uno stato nascosto partendo da un'osservazione di velocità. Prima di attribuire significato sperimentale alla ricostruzione dobbiamo esplicitare **che cosa la velocità in ingresso rappresenta e che cosa della misura è stato perso o alterato**.

Cinque messaggi per la missione:

1. **La misura attraversa una catena.** Flusso, particelle, ottica, immagini, stima degli spostamenti e validazione concorrono al dato finale.
2. **Valore incerto, valore spurio, osservazione assente e struttura non risolta sono situazioni differenti.** Non si rappresentano automaticamente con lo stesso rumore.
3. **L'incertezza dipende dall'esperimento e può cambiare nello spazio, nel tempo e fra componenti.** Una costante può essere un'ipotesi semplificatrice, non una proprietà universale della PIV.
4. **Per verificare una proprietà servono i dati adatti.** Una griglia finale completa non rivela da sola né errori reali né storia dei vettori sostituiti.
5. **Una misura più credibile non richiede di imitare ogni sorgente d'errore.** Richiede scegliere uno scenario coerente, dichiarare ciò che modelliamo e spiegare ciò che rimane fuori. Quest'ultimo punto è la nostra proposta operativa.

## 3. Spiegazione della misura e del vocabolario

### 3.1 Dalle particelle alla velocità

**D — Paper, §3, pp. 6–7, eq. 6.** La PIV ricava la velocità stimando lo spostamento di gruppi di particelle fra immagini. La conversione dipende dal fattore di scala e dalla separazione temporale degli impulsi. Nella notazione del paper:

\[
u=\frac{\Delta x}{\Delta t}=\frac{\Delta X}{M\Delta t},
\]

dove ΔX è lo spostamento sul piano immagine, Δx quello nello spazio fisico e M il fattore di ingrandimento con unità coerenti. Non basta conoscere un errore in pixel per conoscerlo in m/s.

**I — Per noi.** Dobbiamo distinguere almeno tre tempi: separazione fra i due impulsi che formano una misura PIV; intervallo fra campi PIV successivi; intervallo fra snapshot LES disponibili. Non si sostituiscono fra loro. Il Δt dei file LES non decide automaticamente il Δt di una PIV virtuale. Il paper assume la separazione fra impulsi piccola rispetto ai tempi caratteristici del flusso, così da trascurare il troncamento temporale (p. 7). Se nello step 4 useremo un solo snapshot, la misura sarà approssimata come istantanea: non dichiareremo di aver simulato il moto delle particelle fra due immagini.

### 3.2 Errore e incertezza

**D — Paper, §1.2, pp. 3–4, eq. 1–5.** L'errore è la differenza fra misura e valore vero; normalmente il secondo è ignoto. L'incertezza ne descrive la conoscenza disponibile. Il paper distingue componente casuale e sistematica dell'errore: una componente sistematica può variare con la sorgente che la produce, senza essere un offset costante ovunque.

L'incertezza standard è una scala assimilabile a una deviazione standard; l'incertezza espansa aggiunge un fattore di copertura. Il collegamento fra una banda e una probabilità richiede informazioni sulla distribuzione. I consueti fattori 1, 2 e 3 hanno la nota interpretazione circa 68%, 95% e 99.7% **nell'ipotesi gaussiana**.

**I — Per noi.** Un esperimento che scrive “±0.8 m/s” non ci ha ancora autorizzato a generare rumore gaussiano con σ = 0.8 m/s. Dobbiamo sapere che cosa indica quel numero, a quale componente si applica, con quale copertura e quali sorgenti comprende.

La distinzione ISO fra valutazioni di tipo A e B, richiamata nella nota a p. 4, riguarda come si valuta l'incertezza: non va sovrapposta automaticamente a casuale/sistematica. Le eventuali classi di scenario della nostra documentazione storica devono avere un nome esplicito per non confondersi con questa terminologia.

### 3.3 Risoluzione, passo e campo di vista

**D — Paper, §§2.3 e 3.1.1, pp. 6–8.** Finestra, pesatura e algoritmo influenzano la risposta spaziale. A p. 8 il filtraggio spaziale è collegato a errori di modulazione; la review richiama anche limiti all'equivalenza fra correlazione e media mobile.

**I — Distinzione operativa per noi:**

| Grandezza | Domanda cui risponde |
|---|---|
| Campo di vista | Quale regione è osservata? |
| Passo dei vettori | A quale distanza sono le posizioni restituite? |
| Finestra di interrogazione | Quale regione dell'immagine entra nella stima? |
| Risposta/risoluzione effettiva | Quali variazioni del campo sono effettivamente conservate? |
| Spessore del foglio laser | Quale estensione fuori dal piano contribuisce al segnale? |

Come semplice relazione geometrica, per finestre regolari della stessa dimensione e senza ulteriori ricampionamenti, il passo è dimensione della finestra moltiplicata per uno meno la frazione di sovrapposizione. Questa relazione non descrive da sola la risposta spaziale dell'algoritmo e va verificata per i file effettivamente rilasciati.

Un aumento del numero di vettori ottenuto sovrapponendo finestre non prova un aumento equivalente dell'informazione spaziale indipendente. Nello scouting dobbiamo raccogliere separatamente queste grandezze.

## 4. Registro delle evidenze rilevanti

Tutte le righe di questa sezione sono **D — lette nella review**, con i limiti specificati. Le conseguenze per RealPDEBench sono nella sezione 7.

| ID | Evidenza e significato | Dove nel paper | Limiti da mantenere |
|---|---|---|---|
| P01 | Un valore universale di 0.1 pixel è troppo semplificato; l'incertezza varia fra esperimenti e all'interno dello stesso esperimento. | §1.1 p. 2; §4.3 pp. 19–20; §6 pp. 25–26 | La review non fornisce la mappa d'incertezza del nostro combustore. |
| P02 | La catena comprende accesso ottico, risposta delle particelle, illuminazione, imaging, analisi, validazione e riduzione dei dati. La figura include sostituzione dei vettori e problemi di indice di rifrazione. | §2 pp. 4–6; fig. 4 p. 5 | La figura inventaria sorgenti; non ne misura la frequenza o la dominanza nella combustione. |
| P03 | Disallineamento, prospettiva, calibrazione e variazione dell'ingrandimento attraverso uno spessore finito possono produrre errori. | §2.1.1 pp. 4–5 | Servono geometria e calibrazione dell'apparato; una slice non le contiene. |
| P04 | Gli errori di temporizzazione influenzano la velocità; il loro peso dipende dalla separazione fra impulsi. | §2.1.2 p. 5; §3 p. 7, eq. 6–7 | I casi e i numeri citati dipendono dai laser; non sono un errore temporale universale. |
| P05 | La velocità delle particelle può differire da quella del fluido; la risposta dipende dalle proprietà delle particelle e dal moto del fluido. | §2.1.3 p. 5 | Non si ricava una distribuzione di slip dai soli valori istantanei planari. |
| P06 | Il peak locking può spostare la stima verso spostamenti interi in pixel ed è legato a dimensione dell'immagine delle particelle e processamento. | §2.1.4 pp. 5–6; §3.1.1 pp. 7–8 | Non equivale a rumore gaussiano additivo e non è dimostrato presente nel nostro caso. |
| P07 | Rumore di acquisizione e variazioni fra immagini delle intensità delle singole particelle influenzano la stima. Variazioni globali dell'illuminazione non hanno automaticamente lo stesso effetto. | §§2.1.4–2.1.5 p. 6 | “Più luminosità della fiamma = σ prefissata” non è una legge dedotta dalla review. |
| P08 | Gradienti e moto fuori piano possono aumentare gli errori: particelle che entrano/escono dal foglio non trovano necessariamente corrispondenza. La deformazione delle finestre può mitigare effetti dei gradienti in piano. | §2.2 p. 6; §3.1.1 p. 8 | Non implica che un certo valore della componente fuori piano generi sempre un buco, né che il difetto sia al centro della fiamma. |
| P09 | Outlier isolati e cluster di outlier pongono problemi diversi di identificazione. | §2.3 p. 6; fig. 4 p. 5 | Nessuna percentuale tipica universale di rigetto o legge di generazione delle lacune. |
| P10 | Il filtraggio spaziale può introdurre bias; una media mobile è un modello con condizioni di validità, non una definizione generale della PIV. | §3.1.1 p. 8 | La review riporta una condizione specifica su ampiezza dello spostamento sinusoidale e diametro immagine; non va reinterpretata come soglia in m/s della nostra LES. |
| P11 | Le immagini sintetiche consentono controllo e riferimento noto, ma possono sottostimare gli errori sperimentali perché idealizzano immagini e flusso. | §3.1.2 pp. 8–9; §4.1 pp. 14, 16–18 | Generare immagini di particelle non garantisce di per sé maggiore fedeltà del modello virtuale. |
| P12 | Metodi diversi di UQ hanno requisiti e punti ciechi diversi: alcuni non rilevano outlier o peak locking. | §3.2 pp. 9–14; tab. 4 p. 15 | Un indicatore di qualità o una stima UQ non certifica l'assenza di altre sorgenti. |
| P13 | Per valutare una stima UQ serve un errore noto o stimato tramite riferimento più accurato; contano corrispondenza spaziale e temporale e incertezza del riferimento. | §4.1 pp. 14, 16–18; fig. 12 p. 16 | Due misure non simultanee di un flusso instazionario non danno l'errore istantaneo per differenza. |
| P14 | La copertura misura la frazione di errori contenuti nelle bande; il confronto deve usare il livello nominale corretto. Il confronto RMS ha ipotesi proprie. | §4.1 pp. 17–18 | RMS simili non garantiscono da soli una buona incertezza per ogni vettore. |
| P15 | La propagazione a quantità derivate dipende da sensibilità e covarianze. Errori spazialmente o temporalmente vicini possono essere correlati. | §5 p. 20, eq. 13; §5.1 pp. 21–22, eq. 15 | Correlazione delle velocità e correlazione dei loro errori non sono la stessa cosa. |
| P16 | In un esempio per la vorticità, la correlazione a una separazione specifica modifica l'incertezza della derivata. | §5.2 pp. 22–23, eq. 16 | Il risultato citato per 50% di overlap riguarda ρ(2h) in quel setup; non prova che tutti gli errori fra vettori adiacenti siano indipendenti sotto il 50%. |
| P17 | La stereo-PIV richiede propagazione attraverso geometria e ricostruzione; la componente fuori piano può avere incertezza diversa. La volumetria introduce ulteriori sorgenti. | §§5.3–5.4 pp. 23–24; fig. 17 p. 21 | Due componenti esportate da una stereo-PIV non ereditano automaticamente lo stesso modello d'errore di una PIV a camera singola. |
| P18 | Sorgenti come slip, calibrazione e timing possono non essere quantificate dall'analisi delle immagini; i bias da scale non risolte restano importanti. | §2 p. 4; §6 p. 26 | Le conclusioni sui limiti dei metodi sono riferite alla letteratura considerata nel 2019; gli sviluppi successivi vanno cercati nello step 2. |

### 4.1 Tre letture particolarmente utili per evitare scorciatoie

**Finestre e media mobile.** A p. 8 la review riporta, richiamando Theunissen (2012), un comportamento da media mobile per piccoli rapporti A/dτ, con A ampiezza dello spostamento sinusoidale e dτ diametro dell'immagine delle particelle. Non sono ampiezza della velocità e dimensione della finestra. La conseguenza per noi è chiedere quale risposta spaziale sia ragionevole approssimare; il paper non impone di aggiungere un box filter di dimensione arbitraria.

**Correlazioni.** Alle pp. 22–23 la quantità ρ(2h) è la correlazione dell'errore a distanza 2h, necessaria per la derivata centrale. Il valore nullo riportato per overlap del 50% o inferiore non descrive automaticamente ρ(h). Prima di usare numeri di correlazione bisogna sapere distanza, direzione, componente e algoritmo.

**Lacune.** La review discute outlier e validazione e include sostituzione dei vettori nella figura 4. Non dimostra una forma, posizione o frequenza dei buchi nelle fiamme swirl. “Buchi grandi al centro della fiamma” resta una possibile osservazione da cercare e verificare, non un risultato di questo step.

### 4.2 Specificità della combustione: che cosa chiedere oltre questa review

Le altre letture rendono utile esplicitare questa lacuna. **D — Paper:** l'indice di rifrazione compare nella figura 4; variazioni di temperatura, densità e viscosità sono richiamate al §2.2. Le applicazioni in combustione sono menzionate al §4.3, ma la review non sviluppa una caratterizzazione sistematica degli errori nelle fiamme.

**M/I — Domande di scouting, non risultati dimostrati da Sciacchitano per il nostro caso:**

| Fenomeno da approfondire | Domanda e tipo di evidenza cercata |
|---|---|
| Espansione e densità del seeding | Come cambia la concentrazione di traccianti fra reagenti e prodotti? Quali flussi sono inseminati? Ci sono immagini o statistiche che colleghino il fenomeno a incertezza o rigetti? |
| Gradienti di indice di rifrazione / beam steering | Sono stati misurati spostamenti apparenti o distorsioni in configurazioni comparabili? Con quale geometria, scala e riferimento? |
| Fondo luminoso, riflessi e sporcamento delle finestre | Quale residuo rimane dopo filtri ottici e correzione del background? Produce valori errati, regioni escluse o entrambi? |
| Risposta e integrità dei traccianti | Quali materiali, dimensioni e condizioni termiche? Quali evidenze di slip, termoforesi, evaporazione o altra alterazione, se pertinenti a quel tracciante? |
| Posizione dei problemi rispetto alla fiamma | Esistono diagnostiche simultanee e registrate, per esempio OH-PLIF, che permettano un confronto? Non assimilare il segnale a una T vera senza un metodo qualificato. |

Non assumiamo che tutti questi fenomeni siano presenti, dominanti o modellabili. Nello scouting sarà utile separare fenomeni evitati dal progetto sperimentale, corretti nel processamento e ancora presenti nel dato finale. Un elenco di possibili cause non è una misura della loro frequenza.

### 4.3 Formule: anomalie della fonte da non correggere implicitamente

Il controllo proposto nelle altre letture è utile. Alcune anomalie sono presenti nella pagina stampata, non soltanto nell'estrazione del testo.

| Passaggio | Controllo sulla copia letta | Uso nella missione |
|---|---|---|
| Eq. 14, p. 21 (PDF 22) | Il membro sinistro mostra U al quadrato e il destro una radice di termini quadratici: la notazione appare dimensionalmente incoerente. | Non implementata e non riprodotta come formula operativa. Se necessaria, verificare l'originale citato ed eventuali correzioni. |
| Eq. 16, p. 22 (PDF 23) | Nel denominatore compare d, mentre nel testo associato viene definito h come passo. | Manteniamo solo la dipendenza qualitativa dalla correlazione a distanza 2h; il denominatore va chiarito nella fonte originaria prima dell'uso. |
| Relazione geometrica stereo, §5.3 p. 23 (PDF 24) | La pagina stampata riporta tan(θ). La proposta che manchi un numeratore nell'estrazione non è dimostrata: bisogna controllare convenzione angolare e lavoro citato. | Non deduciamo fattori numerici universali fra componenti e non sostituiamo arbitrariamente tan con cot. |

Queste domande non bloccano lo scouting: le formule non sono necessarie per definire ora l'operatore virtuale. La review resta utile senza trattare ogni espressione tipografica come una specifica da copiare nel codice.

## 5. Come si stima l'incertezza: cosa possiamo chiedere ai dati

### 5.1 Due classificazioni da non confondere

**D — Paper, §§3–3.2, pp. 7–14.**

- **A priori:** valuta prestazioni o dipendenze attese per un algoritmo e una classe di condizioni, mediante teoria, simulazioni d'immagini o prove controllate.
- **A posteriori:** cerca una stima locale e istantanea per il dataset acquisito.
- Fra i metodi a posteriori, **indiretti** e **diretti** distinguono il ricorso a relazioni precalibrate dall'estrazione dell'informazione dai dati delle immagini/correlazioni. Un metodo a posteriori può quindi richiedere una calibrazione a priori.

### 5.2 Sintesi dei metodi della review

Questa tabella serve a interrogare i dataset; non sceglie un metodo da implementare. Fonte: §3.2 pp. 9–14 e tabella 4 p. 15.

| Metodo | Informazione richiesta | Che cosa restituisce nella review | Limite decisivo per lo scouting |
|---|---|---|---|
| Superficie d'incertezza | Calibrazione su immagini sintetiche per uno specifico processamento; stima locale delle sorgenti selezionate | Bande anche asimmetriche e contributi casuali/sistematici | Dipende dalle sorgenti incluse e dalla calibrazione; va ricostruita al cambiare dei parametri. Non copre automaticamente scale non risolte. |
| Metriche SNR della correlazione | Piano di correlazione o metriche conservate, più relazione empirica calibrata | Principalmente incertezza del modulo dello spostamento; possibile contributo degli outlier | Un rapporto fra picchi non è già un'incertezza in m/s, né una probabilità che il vettore sia corretto. Passare alle componenti richiede ulteriori ipotesi. |
| Particle disparity / image matching | Coppie d'immagini, stima dello spostamento e identificazione delle particelle | Incertezza per componente; la tabella distingue anche contributi casuali/sistematici | Identificazione delle particelle incerta; numero finito di immagini di particelle; difficoltà con outlier e peak locking. |
| Correlation statistics | Informazione delle immagini e del processamento iterativo della correlazione | Incertezza standard per componente | Non basta un CSV di velocità; nel quadro della review outlier e peak locking restano punti ciechi. |
| Moment of correlation | Informazione del piano di correlazione | Stima da forma e momenti della correlazione | Discretizzazione e risoluzione del picco influenzano anche la stima d'incertezza. |
| Error sampling | Ripetizioni cambiando aspetti dell'esperimento o del processamento | Soprattutto incertezza sistematica di quantità statistiche | Costoso; non fornisce automaticamente il rumore istantaneo di ciascun vettore. |

Le etichette “nessuna ipotesi sul flusso o imaging” riportate per alcuni metodi nella tabella originale non significano assenza di ipotesi algoritmiche o copertura di ogni errore. Il testo esplicita condizioni e limiti. In particolare il metodo correlation statistics è descritto per processamento iterativo multi-pass con deformazione delle finestre e predictor-corrector giunto a convergenza (§3.2.2.2, p. 11). Anche la stima UQ ha variabilità propria per il numero finito di particelle: i valori 5–25% della tabella 4 sono variabilità relativa della stima di incertezza, non direttamente errore percentuale della velocità.

### 5.3 Perché un campo d'incertezza va a sua volta verificato

**D — Paper, §4.2, pp. 18–19; figura 13 p. 16.** La review riporta il confronto di Sciacchitano et al. (2015) su un getto con moto fuori piano nel core. Nella figura 13(g), per il livello nominale del 68%, le coperture indicate sono circa 3% per uncertainty surface, 64% per particle disparity, 51% per peak ratio e 65% per correlation statistics. Si tratta di quel caso e di quelle implementazioni: non di una graduatoria universale dei metodi. Sono dati riportati dalla review, non nuove misure di questo lavoro di lettura.

**I — Per noi.** “Il dataset ha una colonna uncertainty” è un punto di partenza. Servono metodo, unità, componenti, livello, validazione e sorgenti escluse. Una mappa uniforme e una non uniforme possono avere la stessa media ma implicazioni diverse per la ricostruzione. **Una mappa di incertezza standard per vettore non è una mappa degli errori realizzati e non contiene automaticamente le covarianze degli errori fra vettori.** La correlazione spaziale delle ampiezze di incertezza non sostituisce la correlazione degli errori. Anche rielaborare le immagini con overlap diversi non fornisce da solo l'errore vero: per quantificarne la correlazione serve un riferimento, un benchmark noto o un metodo ulteriormente giustificato.

### 5.4 Conversioni e statistiche: due regole pratiche

**I — Derivazione dall'eq. 6, non parametro scelto.** Se ℓpx è la scala fisica per pixel e si considera soltanto la dispersione dello spostamento, mantenendo fissi scala e tempo:

\[
s_{u,\mathrm{disp}}=\frac{\ell_{\mathrm{px}}}{\Delta t_{\mathrm{PIV}}}\,s_{\Delta X}.
\]

Questa formula non comprende automaticamente calibrazione, timing, bias, errori di orientamento o correlazioni. L'eq. 7 del paper presenta una propagazione più ampia, sotto ipotesi di indipendenza delle variabili considerate. Una percentuale richiede inoltre il denominatore: velocità locale, bulk, fondo scala o altra grandezza.

**I — Identificabilità dell'errore.** In un flusso turbolento la velocità misurata varia anche perché varia la velocità fisica. La RMS dei campi non è la deviazione standard dell'errore di misura. La review distingue incertezza delle statistiche e incertezza delle osservazioni istantanee (§§4.1 e 5.1): i due problemi non si scambiano semplicemente usando più frame. Anche la correlazione temporale del moto fisico, che influenza il numero effettivo di campioni per una statistica, va distinta dalla correlazione temporale dell'errore. Non scegliamo una delle due sulla base dell'altra. Non assumiamo inoltre che qualunque errore gonfi sempre varianze o covarianze: con dipendenze fra errore e segnale l'effetto richiede un'analisi specifica.

## 6. Che cosa rappresenta già lo studio nonideal

**D — Contesto della tesi.** Questa sezione usa le specifiche e i report già letti durante la preparazione della missione, sul branch iniziale di pianificazione. Non costituisce una nuova verifica del cluster o dello stato dei 30 specialisti.

| Elemento esistente | Utilità già presente | Che cosa non qualifica |
|---|---|---|
| Velocità LES ai siti casuali al 5% | Compito controllato con target T numerico e input riproducibili | Layout e risoluzione di un esperimento PIV |
| Rumore gaussiano i.i.d., σ = 0.5/1.5 m/s | Sensibilità a due livelli di perturbazione additiva, con unità fisiche esplicite | Distribuzione, covarianza e intensità dell'errore di uno specifico apparato |
| Lacune a dischi, circa 25% dell'area | Effetto di una geometria sintetica contigua | Frequenza, posizione, forma e persistenza di invalidazioni sperimentali |
| Rimozione casuale a pari numero di siti | Confronto utile fra due modalità di perdita d'osservazioni | Decomposizione causale universale del danno; cambiano comunque le posizioni mancanti |
| Rumore più lacune | Interazione in uno scenario controllato | Dipendenza comune da seeding, gradienti o condizioni ottiche |
| Maschere/token espliciti per siti mancanti | Distinzione corretta fra assenza e valore numerico nullo | Eventuale provenienza da interpolazione di un vettore rilasciato |
| Specialisti | Evidenza sul training nei regimi sintetici già definiti, quando la campagna sarà verificata | Trasferimento garantito a un nuovo layout e modello di misura |

**I — Cosa rimane acquisito.** Gli strumenti di confronto, il trattamento esplicito dei mancanti e l'esperienza della campagna sono riutilizzabili. Non occorre cancellare o rinominare i risultati precedenti. La nuova missione cambierà il significato delle osservazioni in ingresso e dovrà registrarlo.

Fonti di contesto: `docs/nonideal/00_ricognizione_osservazioni.md`; `01_block1_handoff.md`; `02_block2_robustness.md`; `03_specialist_handoff.md`; `04_specialist_smoke_criteria_and_budget_policy.md`; `configs/nonideal/degradation_spec_v1_12models.yaml`. La roadmap operativa è `mese3/02_virtual_piv/roadmap02.md`, pubblicata con commit `c79c470`.

## 7. Dal paper alle decisioni della missione

La prima colonna richiama evidenze **D**; la seconda è **I — proposta per la tesi**; la terza indica **M**. La quarta descrive una possibile verifica futura, non attività già eseguita.

| Evidenza del paper | Implicazione per il nostro problema | Informazione ancora necessaria | Possibile verifica |
|---|---|---|---|
| Catena di misura e sorgenti multiple, P02 | Definire se modelliamo un output PIV prima o dopo validazione e sostituzione | Livello di processamento del riferimento | Metodo dell'esperimento, README e flag dei file |
| Geometria e componenti, P03/P17 | Verificare la corrispondenza fra canali LES e misura rappresentata | Piano fisico, assi e segni i/j/k; 2D2C o stereo-PIV | Audit comune missione 01; metadati/autori quando necessario |
| Risposta spaziale, P10 | Separare filtro e posizioni dei vettori; valutare una semplificazione planare | Finestra, pesatura, algoritmo e risoluzione fisica | Metodi dell'esperimento e studi di risposta del processamento |
| Filtraggio e scale non risolte, P10/P18 | Non pretendere che un dettaglio sub-griglia sia rappresentato fedelmente | Filtro LES e passaggio alla griglia 128×128 | Riutilizzare audit dei dati; registrare ciò che resta ignoto |
| Conversione spostamento–velocità, P04 | Usare parametri ottici/temporali coerenti per tradurre gli errori | Scala fisica/pixel, Δt fra impulsi e definizione dell'incertezza | Paper, calibrazione e metadati |
| Outlier e validazione, P09/P12 | Distinguere errore residuo, invalidazione e vettore sostituito | Maschere prima/dopo validazione e criteri | Confronto dei livelli di processamento; analisi dei flag |
| Sorgenti locali e variabili, P01/P08 | Considerare l'eterogeneità soltanto se motivata da evidenza utilizzabile | Mappe UQ o statistiche regionali qualificate | Campioni rappresentativi e metodo UQ documentato |
| Correlazioni degli errori, P15/P16 | Un processo i.i.d. è una scelta di scenario da motivare | Covarianze o riferimenti sufficienti a stimarle | Errori rispetto a riferimento, prove d'immagini o studi specifici; non correlazione del solo campo u |
| Accesso ottico nella catena, P02 | Distinguere bordo del campo di vista da rigetto interno | Geometria fisica e supporto delle maschere | Coordinate, disegno apparato e campi di validità |
| Risposta delle particelle e moto fuori piano, P05/P08 | Non introdurre automaticamente leggi di dropout da T o dalla terza componente | Particelle, foglio laser, tempi, orientamento e dati sufficienti | Fonte primaria dedicata; verifiche comuni dei canali |
| Verifica della misura, P13/P14 | Dare un grado di affidabilità alle proprietà trasferite | Riferimento, sincronizzazione e incertezza del riferimento | Lettura del protocollo e delle prove di validazione |
| Propagazione a quantità derivate, P15 | Studiare in futuro quanto la misura perturba la T predetta | Modello di errore qualificato e protocollo sulle reti | Valutazione successiva al contratto di osservazione |

### 7.1 Limiti specifici dei nostri dati

**D — Contesto preesistente:** griglia 128×128, passo circa 0.581 mm in una direzione e 0.758 mm nell'altra. Questi numeri sono quelli riportati nell'audit già disponibile; non sono un nuovo calcolo in questa lettura. Componenti, piano fisico, filtri e ricampionamento devono essere riusati dall'audit comune, con il loro stato di verifica.

**I — Conseguenze:**

- Una finestra quadrata in celle non è quadrata in millimetri. La scala fisica va esplicitata.
- Finestre che occupano pochissimi nodi LES non consentono di caratterizzare accuratamente ogni possibile risposta PIV. Ricampionare più finemente non crea informazione fisica.
- Una sola slice non dà la distribuzione della velocità attraverso lo spessore del foglio, né le traiettorie tridimensionali delle particelle. Una terza componente in una sezione, se disponibile, non sostituisce un volume.
- Una PIV ottenuta in un altro apparato può motivare una classe di osservazioni. Non dimostra che identici parametri descrivano il combustore di RealPDEBench.
- U-Net e GLU devono ricevere gli stessi valori, posizioni e informazione di validità. Eventuali adattamenti fra griglia fisica PIV e griglia U-Net sono parte del modello di osservazione, non dettagli neutri.

### 7.2 Tre incertezze diverse nella tesi

**I — Distinzione per il nostro programma di ricerca, non contributo specifico del paper:**

1. Incertezza della misura di velocità, oggetto centrale della review.
2. Sensibilità della T predetta a quella misura incerta, dipendente anche dalla rete.
3. Incertezza dello stato nascosto compatibile con le osservazioni, più errori del modello e limiti del riferimento LES.

Ripetere perturbazioni d'ingresso in una rete deterministica studierebbe una componente del punto 2. Non dimostrerebbe una distribuzione completa e calibrata della T vera. La dispersione fra seed di training è un'altra quantità ancora. Questa separazione servirà anche per il coordinamento con DMF-Gen.

## 8. Cosa chiedere all'agente A — Dataset PIV

**Mandato:** cercare accessi sperimentali concreti e accertare quali evidenze consentono. Priorità a combustione e swirl; dataset di altri flussi utili per verifiche metrologiche restano pertinenti, con trasferibilità dichiarata. Archivi di sole statistiche e archivi sintetici vanno classificati separatamente, senza scartarli o presentarli come misure istantanee.

Per ogni candidato compilare questa scheda, collegabile all'inventario della missione 01:

| Campo | Informazione da recuperare |
|---|---|
| Identità | Nome, autori, DOI/URL, versione, accesso e licenza; dati trovati, accessibili e ispezionati separati |
| Apparato e flusso | Geometria, reagente/non reagente, miscela, pressione e condizioni principali |
| Tipo di misura | PIV planare 2C, stereo 3C, volumetrica o PTV/LPT; non dedurlo dalla sola forma degli array |
| Livello dei dati | Coppie d'immagini, vettori prima/dopo validazione, sostituzioni, campo ricampionato, sole medie/RMS |
| Geometria | Assi e unità, piano, campo di vista, griglia e orientamento delle componenti |
| Risoluzione | Scala ottica/pixel con unità e convenzione, finestre di ogni passaggio, pesatura, overlap, passo e risoluzione dichiarata; eventuali spostamenti tipici e massimi in pixel |
| Tempi | Δt fra impulsi, frequenza dei campi, durata, timestamp e sincronizzazione delle modalità |
| Particelle e ottica | Materiale e dimensioni fisiche; quali flussi sono inseminati; concentrazione e diametro dell'immagine delle particelle in pixel, tenuti distinti; foglio laser, filtri e vincoli d'accesso |
| Validità | Flag, NaN, codici, zero, padding, regione esterna al dominio e vettori rimpiazzati: significati documentati |
| UQ/qualità | Metodo, unità, componente, copertura, statistiche locali/globali, sorgenti coperte e validazione |
| Elaborazione | Software/versione e algoritmo; criteri e soglie di validazione, filtri, interpolazione, riempimenti e ricampionamento; provenienza delle diverse versioni dello stesso campo |
| Riferimento | Moto imposto o altra misura; co-localizzazione, simultaneità e incertezza; altre modalità simultanee, registrandone segnale e ruolo senza promuoverle a riferimento T |
| Ispezione | File/campioni letti, formato e forma effettivi, provenienza e limiti dell'ispezione |

**Domande decisive:** possiamo distinguere vettore misurato e interpolato? Possiamo separare FOV, ostacolo e rigetto? Esistono immagini originali o mappe UQ? Ci sono abbastanza condizioni e frame per descrivere variabilità senza generalizzare da un esempio? Si può ispezionare un campione senza scaricare l'intero archivio?

Ogni dataset deve terminare con un giudizio circoscritto: utile per layout; utile per lacune; utile per UQ; utile solo come precedente metodologico; oppure insufficiente per la verifica richiesta. La mancanza di T simultanea non rende il dataset inutile per qualificare l'osservazione di velocità, ma impedisce di usarlo come riferimento sperimentale diretto della nostra ricostruzione di T.

## 9. Cosa chiedere all'agente B — Letteratura e problemi reali

**Mandato:** trasformare la lista di sorgenti della review in conoscenza contestualizzata, con priorità alla combustione. Approfondire anche sviluppi successivi al 2019 quando incidono sulle scelte. Non cercare una classifica universale precostituita.

Per ogni fenomeno cercare:

1. Meccanismo e condizioni che lo rendono rilevante, distinguendo ciò che avviene nel flusso, nell'ottica e nel processamento.
2. Effetto osservabile: bias, dispersione, outlier, invalidazione, copertura, risposta spaziale o temporale.
3. Quantificazione: cosa è stato misurato, con quali unità, denominatore, campione, riferimento e livello d'incertezza.
4. Dipendenza da configurazione, componenti, condizioni operative, spazio e tempo.
5. Trattamento usato e che cosa rimane dopo validazione, sostituzione e filtraggio.
6. Dati necessari a verificare la conclusione, eventuale archivio associato e limiti del trasferimento alla LES.

**Priorità tematiche:**

- Risposta spaziale dei moderni algoritmi e relazione fra finestra, overlap, passo e risoluzione effettiva.
- Misure in fiamme swirl: approfondire le domande del §4.2, incluse espansione del seeding, beam steering, fondo luminoso e risposta dei traccianti. Verificare singolarmente presenza e importanza nel riferimento scelto; nessuna legge di dropout da T è già fissata.
- Rigetti isolati e regioni contigue, persistenza delle lacune e metodi di sostituzione. Cercare statistiche prima del riempimento.
- UQ per componente, eterogeneità, bias e correlazioni; metodi di verifica e sorgenti non comprese.
- Quali semplificazioni di osservazione sono già state impiegate per ricostruzione o assimilazione, con quale validazione.

**Le frequenze devono restare contestuali:** frazione di quali vettori, in quali frame e condizioni, prima o dopo quale filtro? “Il 10% è mancante” senza queste risposte non è una specifica utilizzabile.

## 10. Quali verifiche saranno possibili nello step 3

Questa tabella è una proposta di metodo (**I**), fondata sui requisiti discussi nella review. Serve a evitare promesse che i dati non consentono.

| Materiale disponibile | Che cosa possiamo verificare | Che cosa non è identificato da quel materiale soltanto |
|---|---|---|
| Coordinate e vettori finali | Layout, passo, supporto; valori non finiti secondo convenzioni note; contenuto numerico rilasciato | Errore reale per vettore, rigetti storici, incertezza delle immagini, origine di uno zero |
| Flag di validità e sostituzione con definizioni | Frazioni, forme e distribuzioni dei rigetti/sostituiti nel campione | Verità della classificazione del validatore, errori residui e cause fisiche di ogni rigetto |
| Sequenze di maschere con tempi | Persistenza, dimensioni e spostamenti delle lacune; variabilità fra condizioni | Una frequenza universale per tutte le PIV |
| Coppie d'immagini e metadati | Ispezione di seeding/background/riflessi; eventuale rielaborazione e UQ con metodo appropriato | Velocità vera del gas e tutte le sorgenti nascoste |
| Piani di correlazione/metriche | Applicazione o controllo del metodo compatibile con quella informazione e calibrazione | Qualsiasi metodo UQ a scelta; conversione universale qualità→σ |
| Mappe UQ con protocollo | Distribuzione delle stime, componenti, regioni più incerte e definizione delle bande | Errore realmente realizzato; copertura empirica senza riferimento |
| Riferimento più accurato, simultaneo e co-localizzato | Stima degli errori, bias, dipendenze e verifica della copertura, con incertezza del riferimento | Errore esatto se il riferimento o l'allineamento sono imperfetti |
| Sole medie e RMS | Campo medio, statistiche pubblicate e loro supporto | Buchi e outlier di ogni frame, rumore istantaneo, coppie di stati T–u |

Per ogni verifica registrare anche “non verificabile”, anziché sostituire il dato mancante con una stima intuitiva. Un eventuale controllo di divergenza nulla sulla sezione non è un criterio universale di errore: nella nostra combustione 3D a densità variabile la divergenza planare non deve in generale annullarsi. I metodi volumetrici per flussi incomprimibili discussi al §5.4 non si trasferiscono automaticamente alla slice reagente.

### 10.1 Qualità, validità ed errore: registrazioni separate

**I — Regola pratica per l'inventario:** un flag del validatore registra la decisione di un algoritmo; una metrica SNR descrive la correlazione; una banda UQ stima l'incertezza sotto un protocollo; l'errore rispetto a un riferimento misura un residuo con i limiti di quel riferimento. Non sostituire una colonna con l'altra.

Per le lacune si possono misurare geometria e frequenza anche da maschere ben definite, senza possedere necessariamente tutti i vettori pre-validazione. Per attribuirne la causa servono altre evidenze. Per il bias di risoluzione, rielaborare immagini con finestre diverse misura una sensibilità al processamento; quantificare l'errore rispetto al campo fisico richiede un riferimento o una caratterizzazione aggiuntiva. Dati processati con provenienza documentata possono essere perfettamente utili: il processamento non li rende falsi.

## 11. Riferimenti da affidare allo scouting, non già verificati negli originali

Queste piste provengono dal testo e dalla bibliografia della review. I titoli sono identificatori bibliografici. Non attestano che un archivio sia oggi scaricabile o che i paper dimostrino un risultato trasferibile al nostro combustore.

| Priorità e domanda | Lavoro citato da Sciacchitano | Dove emerge nella review |
|---|---|---|
| Molto alta: ricostruzione e PIV nei combustori | Saini, Arndt e Steinberg (2016), *Development and evaluation of gappy-POD as a data reconstruction technique for noisy PIV measurements in gas turbine combustors*, Exp. Fluids 57, 122 | §4.3 p. 20; bibliografia p. 30 |
| Alta: base sperimentale per verificare UQ | Neal et al. (2015), *Collaborative framework for PIV uncertainty quantification: the experimental database*, MST 26, 074003 | §4.1 pp. 16–17; bibliografia p. 29 |
| Alta: confronto di metodi e punti ciechi | Sciacchitano et al. (2015), *Collaborative framework for PIV uncertainty quantification: comparative assessment of methods*, MST 26, 074004; Boomsma et al. (2016), *A comparative experimental evaluation of uncertainty estimation methods for two-component PIV*, MST 27, 094006 | §4.2 pp. 18–19; figure 13–14 pp. 16–17 |
| Alta: validazione e cluster di outlier | Westerweel e Scarano (2005), *Universal outlier detection for PIV data*, Exp. Fluids 39, 1096–1100; Masullo e Theunissen (2016), *Adaptive vector validation in image velocimetry to minimise the influence of outlier clusters*, Exp. Fluids 57, 33 | §2.3 p. 6 |
| Alta: risposta spaziale | Scarano (2003), *Theory of non-isotropic spatial resolution in PIV*; Lavoie et al. (2007), *Spatial resolution of PIV for the measurement of turbulence*; Theunissen (2012), studio della risposta a spostamento sinusoidale | §3.1.1 p. 8; bibliografia pp. 28, 30–31 |
| Alta: propagazione e correlazioni | Sciacchitano e Wieneke (2016), *PIV uncertainty propagation*, MST 27, 084006 | §§5.1–5.2 pp. 21–23 |
| Mirata: metodo UQ molto utilizzato | Wieneke (2015), *PIV uncertainty quantification from correlation statistics*, MST 26, 074002 | §3.2.2.2 p. 11 e §4.2 p. 19 |
| Mirata: combustione e condizioni reali | Ebi e Clemens (2016), *Experimental investigation of upstream flame propagation during boundary layer flashback of swirl flames*, Combust. Flame 168, 39–52 | §4.3 p. 20; bibliografia p. 27 |
| Mirata: limiti di immagini e apparati | Reuss et al. (2002), errori d'imaging attraverso cilindro motore trasparente; Scharnowski e Kähler (2016a, 2016b), perdita di correlazione/coppie | Tab. 2 p. 11; tab. 1 p. 9; bibliografia pp. 30 |

Altre piste esplicitate dalle letture esterne: Andreini et al. (2016), *Effect of slot injection and effusion array on the linear heat transfer coefficient of a scaled lean-burn combustor with representative swirling flow*, J. Eng. Gas Turbines Power 138, 041501; Andreini et al. (2017), *Hybrid RANS-LES modelling of the aerothermal field in an annular hot streak generator for the study of combustor-turbine interaction*, stessa rivista 139, 021508. Sono citati al §4.3 p. 20 e in bibliografia p. 26. Vanno verificati apparato e dati, senza assumere dai titoli una fiamma identica alla nostra. Carr et al. (2009), *Spatially correlated precision error in digital particle image velocimetry measurements of turbulent flows*, Exp. Fluids 47, 95–106 (bibliografia p. 27), è una pista specifica per la correlazione degli errori.

La priorità è una proposta per distribuire il lavoro: si può cambiare se emerge una fonte più pertinente o più verificabile. I risultati degli scouting già avviati nella missione 01 vanno consultati prima di duplicare letture.

## 12. Scelte aperte e criterio di chiusura dello step

### 12.1 Che cosa il paper permette di stabilire

**D:** esistono sorgenti distinte lungo la catena di misura; metodi e indicatori hanno domini di validità; risoluzione finita, bias e correlazioni non si riducono a una singola σ; validare le stime richiede dati e riferimenti appropriati.

**I:** la missione deve costruire uno scenario motivato e compatibile con la LES; gli scouting devono cercare evidenze capaci di determinare quel modello, anziché una raccolta indistinta di difetti.

### 12.2 Che cosa rimane da decidere

- Esperimento/configurazione di riferimento e trasferibilità alla nostra combustione.
- Piano e componenti effettivamente rappresentabili.
- Campo di vista, layout, passo e scala di risoluzione.
- Livello del dato da simulare: prima o dopo rigetto e sostituzione.
- Distribuzione, ampiezza, correlazione e possibili bias dell'errore.
- Geometria, frequenza, persistenza e cause delle osservazioni mancanti.
- Quali effetti escludere esplicitamente perché secondari, non quantificati o non rappresentabili.
- Controlli e successiva campagna sulle reti; nessun numero di training è fissato qui.

La review non stabilisce quale difetto domini nel nostro scenario, non quantifica la legge delle lacune nelle fiamme e non predice quanto perderanno U-Net e GLU. Non valida i valori di rumore o le geometrie della campagna precedente. Non dimostra che la velocità determini univocamente T.

### 12.3 Domande prioritarie per il passo successivo

1. Quale esperimento reagente pertinente documenta insieme campo di vista, risoluzione, componenti, validazione e incertezza?
2. Quali archivi consentono di ispezionare dati prima del riempimento e distinguere osservazioni originali, rigettate e sostituite?
3. Dove abbiamo soltanto affermazioni generali e dove una misura riproducibile del fenomeno e della sua frequenza?
4. Quale parte di quella misura può essere rappresentata dalla nostra griglia e dalla slice disponibile?
5. Qual è la semplificazione minima che conserva il limite diagnostico scelto e consente un confronto interpretabile delle reti?

### 12.4 Stato della consegna

Le tre letture sono state confrontate e le integrazioni verificate sono incluse in questa v2. La v1 e gli originali ricevuti restano separati. Nessuna delle tre letture è usata come conferma sperimentale indipendente: la fonte comune è la stessa review.

**Chiusura dello step 1:** Enrico ha approvato la scheda integrata e autorizzato la pubblicazione il 25 novembre 2026 (America/New_York). Sono consegnati spiegazione, evidenze rintracciabili, implicazioni separate dalle ipotesi e requisiti per entrambi gli scouting. Lo step 1 è completato. I prompt e l'avvio degli scout dello step 2 saranno coordinati separatamente nella chat di missione; questa pubblicazione non li avvia.

## 13. Esito del confronto fra le tre letture

**Scelta della base:** mantenuta la struttura della v1 del coordinatore, che separa registro delle evidenze, conseguenze per la LES, requisiti dei dataset e decisioni. Da Opus e Neon sono state integrate le specificazioni utili, senza concatenare tre riassunti equivalenti. I contributi si sovrappongono: l'attribuzione sotto indica dove una precisazione è emersa in modo particolarmente esplicito.

| Contributo o divergenza | Decisione nella v2 | Motivo e controllo |
|---|---|---|
| Opus: stato di processamento e sostituzione dei vettori | Rafforzati §8 e §10.1: tracciare provenienza, criteri/soglie, flag e versioni; distinguere validità, qualità, UQ ed errore | Figura 4 p. 5, §2.3 p. 6 e requisiti UQ della tabella 4 p. 15 |
| Neon, anche Opus: aspetti specifici della combustione poco sviluppati dalla review | Aggiunto §4.2 con domande su seeding, beam steering, luminosità, finestre e traccianti | Sono domande di ricerca M/I, non nuovi risultati sulla nostra fiamma |
| Entrambi: intervallo finito della misura e proprietà delle particelle | Esplicitati limite istantaneo al §3.1 e distinzione fra dimensione fisica e diametro immagine al §8 | §3 p. 7 e §2.1.3 p. 5 |
| Entrambi: UQ per vettore e variabilità della stima | Rafforzato §5.2; specificato che mappa UQ e covarianza degli errori sono informazioni differenti | §3.2 pp. 9–14, tabella 4 e §5.1 p. 22 |
| Opus: i.i.d. interpretato come caso overlap ≤50% | Non accolto. Conservata e resa esplicita la cautela su ρ(2h), senza dedurne ρ(h) o indipendenza generale | §5.2 p. 23: distanza e condizioni del risultato sono essenziali |
| Formulazioni di entrambi su PIV come media su finestra | Accolte soltanto come approssimazione da motivare; non come equivalenza esatta universale | §3.1.1 p. 8 e suo limite A/dτ |
| Entrambi: anomalie delle formule | Aggiunto registro §4.3; non corretti silenziosamente i simboli né introdotte formule operative | Controllo delle pagine renderizzate 21–23; formule originali citate ancora da verificare se necessarie |
| Alcune formulazioni: immagini sintetiche sempre ottimistiche, bias sempre gonfia statistiche, necessità di un preciso set di file per qualsiasi verifica | Mantenute affermazioni condizionate e requisiti dipendenti dalla domanda | La review segnala limiti e casi; non prova queste generalizzazioni in ogni modello o dataset |
| Neon: dati non processati come dati non “truccati” | Sostituita la dicotomia con provenienza del processamento | Una misura validata o interpolata può essere utile se il suo significato è documentato |
| Opus: assenza di numeri fisici nella review | Non mantenuta come affermazione assoluta | Figura 15 p. 18 contiene anche esempi di incertezza in m/s; resta vero che non ne ricaviamo parametri qualificati del nostro combustore |
| Entrambi: propagazione Monte Carlo attraverso la rete | Mantenuta come possibilità futura, con distinzione fra sensibilità all'ingresso e incertezza completa di T | Collegamento metodologico I; non risultato di Sciacchitano su reti neurali |
| Entrambi: verifiche interne della slice | Rinviate all'audit comune della missione 01; gli scouting possono procedere con le lacune dichiarate | Evitare duplicazioni e non promuovere orientamento o disponibilità dei canali a fatti senza verifica |

Il confronto non cambia l'obiettivo approvato: costruire un modello di osservazione semplice, motivato e verificabile. Rende più precise le domande necessarie per arrivarci.
