# Analisi delle tre revisioni STOP E e decisione prima di S7

**Data della decisione e della registrazione:** 18 novembre 2026, America/New_York.  
**Branch:** `mission/dense-unet`.  
**Consegna esaminata dai revisori:** `a6b2bad7445170e6d821a14de9f7af9a1905ab2c`.  
**Decisione di Enrico registrata nel diario:** `925f556823214e37f3abf9d73b41d6e5ed815e58`.  
**Stato:** analisi documentale del passo 1. Il disegno esecutivo delle due sotto-missioni sarà fissato nei successivi `roadmapopus.md` e `roadmapastra.md`. La proposta Neon rimane documentata per il riesame successivo.

## 1. Scopo ingegneristico e decisione approvata

La tesi affronta la diagnostica delle camere di combustione con accesso di misura limitato. Vogliamo stimare un campo istantaneo di temperatura, comprese posizione ed estensione delle regioni calde, sfruttando velocità accessibili mediante PIV e, quando necessario, poche misure termiche. Laser, accesso ottico e termocoppie costituiscono il riferimento applicativo richiamato da Enrico. La domanda utile è quali informazioni e quale trattamento delle osservazioni consentano una ricostruzione affidabile quando cambiano condizioni operative e copertura.

Nel setup corrente velocità e temperature provengono da campi numerici, con osservazioni puntuali ideali. Le ancore T sono un primo modello delle misure termiche disponibili. Risposta temporale, volume di misura, rumore, perturbazione della fiamma e limiti ottici richiederanno una qualificazione sperimentale successiva. La [proposta](docs/proposal.md), in particolare sezioni 1–5, definisce il collegamento fra incompletezza delle misure, trasferimento e utilità della ricostruzione.

I controlli metodologici devono sostenere questo obiettivo e risolvere incertezze che cambiano una decisione applicativa. La disponibilità di tempo e risorse consente una campagna più ampia del minimo suggerito inizialmente dall'assistente. Ogni blocco deve comunque avere una domanda, un confronto interpretabile e un esito decisionale.

**Enrico ha approvato la seguente sequenza.**

1. Documentare tutte e tre le revisioni in questa analisi.
2. Preparare ed eseguire una sotto-missione Opus sulla quantità di dati e sulla variabilità dei training.
3. Preparare ed eseguire una sotto-missione Astra su livello, struttura e poche ancore termiche.
4. Discutere entrambi gli esiti, riesaminare le proposte Neon e decidere come riprendere S7.

Le correzioni comuni saranno una preparazione condivisa da effettuare una volta. La decisione approva le domande delle due sotto-missioni e il loro ordine. Numeri dei run, soglie, budget di aggiornamenti, checkpoint e dettagli delle matrici sotto riportate restano **proposte dei revisori**, da fissare nelle roadmap prima dell'esecuzione.

La [roadmap sparse](roadmapsparse.md) resta aperta allo STOP E, con la scelta del seguito S7 sospesa in attesa delle due sotto-missioni. Anche la [roadmap Mese 2](roadmap_month2.md) resta aperta. Astra contribuisce già al nucleo dello Step 2.7, ma completare il suo diagnostico non chiuderà automaticamente tutti gli impegni del Mese 2. Il filone Neon verrà riesaminato dopo Opus e Astra. La sua roadmap e la sua campagna restano da decidere.

## 2. Fonti, attribuzioni e perimetro di questo documento

L'analisi sintetizza le tre relazioni integrali consegnate da Enrico, il confronto in chat e la decisione registrata nel [diario](docs/RESEARCH_LOG.md). Le relazioni sono identificate qui sotto con nome dell'allegato e SHA-256 dei byte ricevuti. I nomi locali identificano le fonti consegnate, senza implicare che gli allegati siano versionati in questa repository.

| Revisore attribuito da Enrico | Allegato integrale usato | SHA-256 |
| --- | --- | --- |
| Opus | `Revisione indipendente STOP E — ricostruzione sparse di T (commit a6b2bad).md` | `156095d0f9b5725adb9313bcdd6f1c0a11a4762f4d8b59413f08a532b39f5adf` |
| Astra | `Revisione_indipendente_STOP_E_a6b2bad(1).md` | `5545161c20f857db2673258f2922899ac80e62dff1abac48f3bad012fd84e2a0` |
| Neon | `Testo incollato(20261119-042531).txt` | `18862ccd9982ef34b2861d8befcc740d066426674c3376f3d6cc6b7787a4947f` |

Il riepilogo breve incollato nel messaggio che accompagnava Astra riprendeva la proposta Opus. L'attribuzione Astra in questo documento segue la relazione integrale, che propone oracle-offset e AO9.

I risultati di partenza sono nel [report sparse](docs/month2/05_sparse_reconstruction.md), nel [pacchetto STOP E](results/month2/s6_sparse_eval/STOP_E_delivery/) e nella [consegna GappyPOD](results/month2/gappypod_nested_1_20/). Il [diagnostico denso](docs/month2/03_dense_velocity_diagnostic.md) e l'[audit GLU](docs/month2/04_glu_release_adaptation.md) completano il contesto.

I numeri e i rilievi descritti come ricalcolati dai revisori mantengono quell'attribuzione. Questo passo documentale non riesegue training, inferenza o audit del codice. Concordanza fra relazioni, verifica numerica e interpretazione meccanicistica sono livelli distinti di evidenza.

Opus e Astra riferiscono riaggregazioni di CSV e lettura del codice, con assenza di checkpoint/cache per un replay neurale. Neon riferisce anche controlli sulla cache locale e sulle maschere VAL, ma nessuna re-inferenza dai checkpoint. I tre riferiscono coerenza delle metriche principali e nessun percorso di leakage dimostrato nei percorsi letti. Questa conclusione riguarda il materiale esaminato.

## 3. Punto di partenza sperimentale

Il task corrente è single-frame, due componenti di velocità co-locate, griglia 128×128 e temperatura come target. `K=0` significa nessuna T in ingresso. `p_on` riceve NH3 e rapporto di equivalenza phi, `p_off` li esclude. Fit e statistiche usano `train_fit`, con 760 frame, 38 per ciascuna delle 20 condizioni. `seen_train_eval` contiene 220 frame successivi delle stesse condizioni. VAL comprende 123 frame di tre condizioni escluse dal fit. I neurali sono allenati al 5% di siti osservati e valutati anche a 1% e 20%, con tre repliche delle maschere.

La metrica primaria è la media delle RMSE spaziali per frame, poi per condizione e per replica. Il best è scelto su VAL al 5%. Le tre maschere misurano sensibilità del layout a pesi fissi. La variabilità fra seed di training resta non misurata nello screening.

| Metodo | Best VAL 5% | Last VAL 5% | Best VAL 1% | Best VAL 20% | Delta swap VAL |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Net p_on | 42.96 | 49.50 | 75.20 | 44.32 | +58.0 |
| U-Net p_off | 43.53 | 47.67 | 73.70 | 54.08 | +57.8 |
| GLU adattato p_on | 45.80 | 46.84 | 68.08 | 46.84 | +53.6 |
| GLU adattato p_off | 47.54 | 47.87 | 67.20 | 48.52 | +55.6 |
| GappyPOD p_on | 70.60 | — | 71.37 | 70.46 | Non consegnato |
| GappyPOD p_off | 94.80 | — | 98.44 | 93.81 | Non consegnato |

Valori in K, arrotondati dalla consegna. Lo swap usa replica 0 e checkpoint best. Le valutazioni a densità diversa usano sempre il best selezionato al 5%.

S5 è concluso per early stopping. GappyPOD job 29601906 ha superato i sei confronti del gate storico 5% prima delle valutazioni nested. I best S6 provengono dal primo tentativo, fallito successivamente nello swap. Last e swap neurali sono stati recuperati nei job 29601907–29601910. I revisori chiedono di qualificare meglio questa catena di provenienza.

`38ac7fe` identifica S5. Gli script di valutazione e le riparazioni successivi richiedono un'identificazione propria, attraverso commit, hash o patch verificabili. La formula sintetica «tip di esecuzione S5/S6/Gappy» non risolve questa distinzione.

## 4. Opus — domanda sui dati e sulla variabilità del training

### 4.1 Che cosa vuole sapere precisamente

**Aumentare il numero di istantanee di training migliora la ricostruzione di nuove istantanee, oltre alle differenze introdotte da seed e scelta del checkpoint?**

La decisione ingegneristica collegata è se investire prima in maggiore copertura del comportamento della fiamma nei dati supervisionati oppure in una diversa architettura, osservazione o strategia di condizionamento. Se poche istantanee producono una ricostruzione che funziona soprattutto sui frame già visti, la sola classifica U-Net/GLU offre una guida limitata per una diagnostica utilizzabile su nuove misure.

### 4.2 Evidenze e ipotesi del revisore

Opus riporta, al 5%, train_fit last 7.2/7.8 K per U-Net on/off contro seen last 56.2/61.6 K. GLU ha train_fit last 27.1/30.3 K e seen 55.0/59.9 K. L'overfitting è particolarmente evidente nella U-Net. Opus legge il divario come segnale di un regime povero di istantanee, con 38 frame per condizione contro 1481 indici disponibili nella finestra di fit.

Questa causa rimane un'ipotesi. Frame aggiuntivi ravvicinati possono essere correlati e offrire poco contenuto nuovo. Cambiare quantità di dati può anche interagire con durata dell'ottimizzazione e ricampionamento delle maschere. Opus riferisce controlli sul denso che non mostrano una semplice deriva del livello medio sufficiente a spiegare tutto il divario.

### 4.3 R0 — rianalisi proposta prima dei training

Opus propone di leggere le valutazioni per epoca già presenti su Engaging e calcolare il risultato VAL dell'epoca selezionata su seen, una selezione leave-one-VAL-condition-out e la sensibilità dei minimi a una media mobile di cinque valutazioni. Vuole capire quanto il vantaggio best U-Net dipenda dal minimo precoce e stretto, a differenza delle curve GLU più regolari. Chiede anche correzione YAML, testi e hash dei sorgenti eseguiti.

Queste rianalisi possono cambiare la lettura del ranking. Le traiettorie originali restano influenzate da VAL tramite scheduler e arresto. Un checkpoint riselezionato non costituisce una nuova valutazione indipendente dal procedimento originale. L'eventuale criterio dei futuri training deve essere fissato prima della nuova campagna.

### 4.4 Matrice originale proposta da Opus

| Braccio | Frame per condizione | Totale fit | Seed modello | Condizionamento | Nuovi run suggeriti |
| --- | ---: | ---: | --- | --- | ---: |
| Riferimento S5 | 38, passo 40 | 760 | 0 | on e off | 0 |
| A, metro del rumore | 38, passo 40 | 760 | 1 | on e off | 2 |
| B, riduzione dei dati | 19, passo 80 | 380 | 0 | on e off | 2 |
| C, aumento dei dati | 149, passo 10 | 2980 | 0 | on e off | 2 |

Architettura proposta U-Net b=32, copertura TRAIN 5%, loss e ricetta di base S5. Opus suggerisce di mantenere il normalizzatore fit-only esistente per il confronto. Per pareggiare warmup, frequenza delle valutazioni e pazienze in aggiornamenti, propone unità di 95 aggiornamenti, ossia 760 campioni estratti dal pool di ciascun braccio. Questo richiede di esplicitare campionamento e definizione dell'epoca nella roadmap.

A e B possono usare la cache attuale. C richiede l'accesso a frame aggiuntivi nella finestra TRAIN e una cache estesa. Seen 1600–2000 deve restare disgiunto. La fattibilità del recupero dei raw va verificata, considerato l'incidente sul pool.

Le misure proposte comprendono fit/seen/VAL, best/last e selezione alternativa, tre coperture, bias, errore centrato, regioni e costo. Opus suggerisce una soglia pari a due volte la differenza seed 0–1 come metro dell'effetto dati. È una regola proposta dal revisore, da riesaminare nel disegno finale. Due seed forniscono un primo confronto descrittivo, senza una stima robusta dell'incertezza né una soglia di significatività automatica.

### 4.5 Esiti e decisioni proposti

| Esito | Interpretazione possibile | Decisione suggerita da Opus |
| --- | --- | --- |
| 380 peggiora e 2980 migliora seen e VAL in modo coerente | Più esempi supervisionati aiutano nel regime corrente | Adottare un regime più ricco, poi rivalutare le famiglie |
| Togliere dati peggiora, aggiungerli aiuta poco | Possibile saturazione o forte correlazione dei frame aggiunti | Cercare un regime sufficiente e spostare l'attenzione su osservazioni e trasferimento |
| I seed cambiano il risultato quanto le differenze oggi discusse | Ranking seed 0 instabile | Repliche mirate prima di conclusioni comparative |
| La riselezione del checkpoint cambia il ranking | Sensibilità al procedimento di selezione | Documentarla e fissare un criterio futuro adeguato |
| I frame aggiuntivi non sono disponibili | La curva verso l'alto rimane aperta | Usare A/B come evidenza indiretta e dichiarare il limite |

Un miglioramento con più frame sostiene l'utilità di quei dati con quella ricetta. Un miglioramento assente non prova un limite informativo fondamentale della velocità al 5%.

### 4.6 Altre proposte Opus da conservare

Opus propone successivamente training a copertura variabile e validazione incrociata per condizione, per studiare p su più condizioni escluse. Propone inoltre un estimatore lineare supervisionato, per esempio ridge fra coefficienti POD di velocità e T con rango più alto, se vogliamo attribuire una parte del vantaggio neurale alla non linearità. Vorrebbe sospendere investimenti GLU fino a una ricetta più stabile, poi rivalutarlo una volta. Queste indicazioni restano proposte, senza scelta definitiva di architettura.

## 5. Astra — domanda su livello, struttura e poche ancore termiche

### 5.1 Che cosa vuole sapere precisamente

**Quanto del fallimento delle regioni calde deriva da un offset del livello termico? Quanto recupereremmo correggendolo perfettamente e quanto di quel recupero è ottenibile con nove temperature osservate?**

La decisione ingegneristica è se poche misure termiche consentano di trasformare una buona ricostruzione spaziale da PIV in una stima termica utile, oppure se rimanga un errore spaziale che richiede un altro trattamento. Questa domanda è direttamente collegata allo Step 2.7 della roadmap Mese 2.

### 5.2 Il caso che motiva il controllo

Per `40NH3_0.9`, Astra riporta i seguenti valori best al 5%, mediati sulle maschere.

| Misura | U-Net p_on | GLU p_on |
| --- | ---: | ---: |
| RMSE campo, K | 39.923 | 29.244 |
| Bias firmato medio, K | -29.352 | -3.250 |
| Errore centrato medio, K | 26.802 | 28.551 |
| Area vera T >= 2100 K | 78.49% | 78.49% |
| Area predetta T >= 2100 K | 24.44% | 76.34% |
| IoU regione calda | 0.310 | 0.925 |

U-Net ha errore centrato leggermente inferiore ma sottostima il livello. La soglia fissa trasforma questo errore in una forte perdita di area calda. Astra vuole misurare l'IoU dopo correzione, anziché dedurla dal solo errore centrato. Un buon errore centrato non garantisce esattezza di posizione, ampiezza o geometria.

### 5.3 Esperimento originale proposto da Astra

Usare i quattro best@5% seed 0 già esistenti, su seen e VAL, con le stesse velocità al 5% e tre maschere. Nessun nuovo training nella prima fase. Per ogni predizione confrontare tre trattamenti.

| Trattamento | Informazione usata | Domanda a cui risponde |
| --- | --- | --- |
| Free K=0 | Velocità, coordinate, eventuale p | Qual è la prestazione attuale? |
| Oracle-offset | Tutta la T vera per stimare l'offset medio del frame | Quanto del difetto è eliminabile con una sola traslazione uniforme? |
| Assisted AO9 | Nove T in siti prefissati e indipendenti dal target | Quanto di quel recupero è ottenibile con osservazioni termiche limitate? |

L'oracle aggiunge `mean(T_true - T_pred)` a tutto il campo. Il suo RMSE per frame coincide con l'errore centrato originale sullo stesso dominio. È un riferimento diagnostico che usa informazione completa del target e va presentato come tale.

Per AO9 Astra propone la formula già esistente, con tau=1 congelato.

\[
\delta_9=\frac{\sum_{k=1}^{9}[T(x_k)-\widehat T(x_k)]}{9+1},
\qquad \widehat T_{AO}=\widehat T+\delta_9.
\]

I siti sono quelli del protocollo ancore, uguali fra metodi. Non devono essere scelti usando T vera o errore. ParamMean+AO e GappyPOD assisted devono essere confrontati con le stesse osservazioni e configurazioni congelate. La presenza di temperature cambia il contenuto informativo del task, quindi free e assisted mantengono confronti propri.

### 5.4 Misure e controlli che rendono la prova interpretabile

- Replay del K=0 prima delle correzioni, con checkpoint, cache e maschere identificati e precisione esplicita.
- Metriche per frame, condizione e maschera, inclusi bias firmato/assoluto, RMSE, errore centrato, regioni fisse, area calda e IoU/Dice.
- Errori full-field e fuori dalle ancore, insieme agli errori nei siti osservati.
- Invarianza dell'errore centrato sotto una correzione uniforme sullo stesso dominio. Identità oracle RMSE = errore centrato verificata prima dell'aggregazione.
- Pannelli frame 1000 di tutte e tre le condizioni VAL, errore firmato a scale comuni e contorno 2100 K, includendo il caso problematico.
- Nel completamento diagnostico S6, swap GappyPOD e distanza fra predizione swapped e T del donatore, mantenendo il layout del destinatario.

Se le predizioni sono già salvate, parte dell'analisi può essere eseguita direttamente su quelle. In caso contrario serve inferenza dai checkpoint. La durata va misurata sul percorso effettivo.

### 5.5 Esiti e decisioni proposti

| Esito | Lettura | Decisione suggerita da Astra |
| --- | --- | --- |
| Oracle e AO9 recuperano gran parte del difetto caldo | Il livello è un limite importante e correggibile con poche T in questo setup | Privilegiare successivamente U-Net on/off e sviluppare free/assisted |
| Oracle recupera, AO9 no | Livello correggibile, ma quelle ancore o lo stimatore non lo identificano bene | Analizzare rappresentatività dei siti e residui, poi valutare K |
| GLU mantiene un vantaggio regionale/spaziale importante anche dopo oracle-offset | L'offset non spiega tutto il divario | Privilegiare U-Net p_on contro GLU p_on nelle repliche |
| Il K=0 non riproduce il riferimento per un errore identificato | Problema della valutazione da risolvere | Correggere e rivalutare gli stessi checkpoint |

Astra preferisce questa diagnosi prima di fissare la coppia S7, con fiducia moderata. La decisione di Enrico colloca la sotto-missione dopo Opus. La roadmap Astra dovrà specificare se confrontare soltanto i checkpoint S5 originali oppure anche checkpoint nuovi di Opus. La versione basata sugli S5 resta il riferimento della proposta originale.

## 6. Neon — proposte da preservare per il riesame successivo

### 6.1 Le domande precise

Neon pone quattro domande collegate al comportamento applicativo.

1. Il peggioramento a 1% e 20% si riduce addestrando direttamente a densità variabili?
2. A parità di questa esposizione, quale famiglia mantiene accuratezza, livello e stabilità migliori?
3. Il risultato U-Net/GLU seed 0 si ripete cambiando inizializzazione?
4. Poche ancore T correggono il livello abbastanza da spostare la priorità verso un sistema assisted?

La sua enfasi è la capacità di usare osservazioni che cambiano quantità e geometria, con attenzione al trasferimento a miscele escluse e alla ricostruzione delle regioni termiche rilevanti.

### 6.2 Analisi aggiuntive e ipotesi

Neon riferisce RMSE rispetto alla media temporale oracolo della condizione pari a circa 119.3 K per `0NH3_1.05`, 37.2 K per `20NH3_1.1` e 29.2 K per `40NH3_0.9`. Usa questi dati per sottolineare l'eterogeneità delle tre VAL. Il riferimento oracolo conosce già le T della condizione esclusa e serve a caratterizzarne la variabilità, senza costituire una baseline utilizzabile con le sole osservazioni di inferenza.

Per U-Net ipotizza che l'aumento dei siti attivi modifichi le statistiche delle prime convoluzioni rispetto alle coordinate e ai parametri, favorendo errori di livello. Per GLU ipotizza interazioni fra numero di token, banda fissa e raggio dei top-k vicini. La lettura del codice e le variazioni di bias motivano questi meccanismi, ma non ne isolano la causalità. Un fallimento residuo dopo un singolo training multidensità non dimostrerebbe da solo un limite strutturale dell'architettura.

Neon segnala inoltre `d_model=32` con feed-forward Transformer ereditato a 2048, che concentrerebbe gran parte dei parametri GLU negli FFN pur mantenendo un latente stretto. Propone di verificarne l'opportunità prima di future estensioni. La configurazione corrente resta quella dello screening da documentare e conservare.

La relazione usa talvolta phi anche per il campo di importanza del metodo GLU. Quel campo è distinto dal rapporto di equivalenza phi che compare nei parametri operativi della combustione. Una futura roadmap dovrà usare nomi non ambigui.

### 6.3 N1, parte diagnostica proposta

Neon propone swap GappyPOD al 5% replica 0 su seen/VAL, tabella regionale, pannelli delle tre VAL con scale comuni, correzioni delle curve e della provenienza. Propone inoltre AO sui quattro best con tau=1 e K in `{0,1,4,9,16,32}`, includendo baseline assisted matched e metriche off-anchor. Questo sweep è più ampio del controllo iniziale AO9 di Astra. Le due proposte possono riusare componenti comuni, ma non vanno sommate implicitamente in un unico mandato già approvato.

### 6.4 N1, quattro training originali proposti

| Run | Modello | p | Copertura TRAIN | Seed modello/shuffle | Domanda |
| --- | --- | --- | --- | --- | --- |
| R1 | U-Net attuale | on | 5% fisso | 1 | Stabilità del minimo precoce e del profilo d'errore |
| R2 | GLU adattato attuale | on | 5% fisso | 1 | Stabilità del vantaggio regionale e fuori densità |
| R3 | U-Net attuale | on | 1/5/20% campionate per batch | 0 | Effetto della sola esposizione multidensità |
| R4 | GLU adattato attuale | on | 1/5/20% campionate per batch | 0 | Capacità di sfruttare più sensori dopo esposizione matched |

Neon mantiene il seed delle maschere TRAIN nel confronto fra inizializzazioni. Per R3/R4 propone una densità omogenea nel batch, compatibile con i token GLU senza padding. Prima di GLU al 20% richiede uno smoke memoria/tempo e, se necessario, microbatch verificato a batch effettivo 8.

Per R1/R2 mantiene selezione su VAL 5%. Per R3/R4 propone selezione sulla media delle tre coperture, da chiamare esplicitamente nuovo protocollo. Cambiano quindi esposizione TRAIN e obiettivo di selezione. Se si vuole attribuire l'effetto soltanto al training multidensità, la futura roadmap dovrà controllare anche questo fattore o dichiarare che confronta ricette complessive.

### 6.5 Scenari e direzioni successive proposti

| Esito ipotizzato | Direzione suggerita da Neon |
| --- | --- |
| GLU multidensità offre il profilo migliore | Considerarlo backbone e studiare un miglioramento locale mirato |
| U-Net multidensità mantiene accuratezza e costo favorevoli | Portare avanti U-Net e conservare GLU come comparatore |
| Persistono errori sensibili alla densità | Valutare una correzione specifica, per esempio input Voronoi/distanza, convoluzioni normalizzate o banda locale adattiva |
| AO recupera gran parte dell'errore di livello | Dare centralità a poche ancore e al trasferimento del livello |

Neon propone poi un fold di pseudo-estrapolazione interno alle condizioni TRAIN, citando condizioni estreme già identificate nei documenti. Normalizzatori, POD e prior dovrebbero essere rifittati escludendo le condizioni tenute fuori. Uno o più fold aiutano a misurare il trasferimento, ma una variazione di fold non sostituisce automaticamente una replica del seed sullo stesso split.

Fra le idee future conserva importanza adattiva, banda GLU dipendente dai vicini, latente più ampio, rappresentazioni più adatte alla sparsità e integrazione di poche T. Il loro valore andrà misurato rispetto a un limite specifico con un controllo matched. Neon raccomanda di discutere la direzione col laboratorio prima della campagna Mese 3 e del freeze TEST.

**Stato deciso da Enrico:** queste proposte sono conservate. La loro priorità verrà riesaminata dopo Opus e Astra, alla luce dei nuovi risultati. Nessuna roadmap Neon viene creata in questo passo.

## 7. Convergenze e divergenze fra i tre pareri

### 7.1 Punti di convergenza

- Le riaggregazioni dei revisori riproducono le metriche principali della consegna.
- Al 5% le ricette neurali superano nettamente le baseline esaminate.
- Lo swap sostiene specificità dell'istantanea e richiede un'interpretazione distinta dalla causalità fisica.
- Il ranking fine e l'effetto di p restano sensibili a condizione, seed non replicato e selezione del checkpoint.
- Livello termico e struttura vanno letti separatamente, insieme alle regioni calde.
- Le curve 1/5/20% di modelli allenati al 5% includono uno scarto dalla densità di training.
- Il GLU corrente è un adattamento deterministico cross-variable, con capacità e componenti diverse dal metodo originale.
- Provenienza, figure complete e completamenti diagnostici richiedono attenzione prima dei run successivi.

### 7.2 Differenze che incidono sul prossimo passo

| Aspetto | Opus | Astra | Neon |
| --- | --- | --- | --- |
| Prima incertezza | Quantità e varietà delle istantanee supervisionate | Errore di livello contro errore spaziale | Generalizzazione alla densità e confronto delle famiglie |
| Prova principale | Curva 19/38/149 frame e secondo seed | K=0, oracle-offset, AO9 sui pesi attuali | Replica p_on di entrambe le famiglie e training multidensità |
| Nuovi training nella prima proposta | Sei U-Net | Nessuno nel primo controllo | Quattro run U-Net/GLU |
| GLU nell'immediato | Rivalutarlo dopo aver stabilizzato la ricetta | Tenerlo come alternativa se il vantaggio spaziale persiste | Includerlo subito nel blocco multidensità |
| Parametri operativi | Misurarli poi su più condizioni escluse | Usare la diagnosi del livello per scegliere il contrasto | Preservare p_on e verificare successivamente pseudo-estrapolazione |
| Ancore termiche | Dopo il regime di dati | Primo esperimento mirato | Parte diagnostica con sweep K |
| Selezione | Rianalisi con criteri alternativi | Best primario conservato per la diagnosi | Criterio nuovo sulle tre coperture per multidensità |

Nessuno dei tre propone come unico passo immediato la semplice esecuzione automatica della vecchia coppia S7 on/off con seed 1 e 2. La scelta di Enrico è dare spazio sia alla domanda Opus sia a quella Astra, poi tornare alla direzione Neon e alla conferma sparse.

## 8. Preparazione tecnica comune da pianificare una volta

I rilievi seguenti sono attribuiti alle revisioni. Questo documento non certifica di averli corretti. Le roadmap dovranno ricondurli a controlli eseguibili e a una consegna identificata.

| Rilievo | Revisori che lo segnalano | Verifica o risultato atteso dalla preparazione |
| --- | --- | --- |
| YAML sparse non parsabile | Opus e Astra | Sintassi e schema validi, inclusa la chiave S6 ammessa dal loader, con test pertinente |
| Riferimento SHA S5 usato anche per script successivi | Tutti | Identificare training, evaluator, patch e artefatti riusati con fonti verificabili |
| Reuse best controllato principalmente per esistenza | Opus e Astra | Collegare CSV, checkpoint, configurazioni, maschere e tentativo d'origine |
| Fallback maschere e controlli payload insufficientemente imposti | Astra | Identità dei payload realmente consumati e fallimento esplicito quando manca il layout richiesto |
| Gate Gappy con riferimento vuoto può passare con zero controlli | Astra | Riferimento non vuoto e insieme delle chiavi attese imposto. Il gate consegnato contiene sei verifiche reali |
| Flag precisione S6 non esplicitati come S5 | Astra | Policy applicata e registrata per confronti e replay successivi |
| Pannelli della sola prima VAL, scale e titoli incompleti | Astra e Neon, con rilievi affini Opus | Tutte le condizioni, scale comuni, errore firmato e soglie dichiarate |
| Tabelle regionali dichiarate assenti ma colonne presenti | Tutti | Aggregazione cross-method con conteggi validi e descrizioni coerenti |
| Curva loss legge una colonna non presente nel percorso esaminato | Neon | Figura generabile dalle colonne effettive e provenienza della trasformazione |
| Ultima eval assente dal JSON standalone | Tutti nei rispettivi controlli | Collegare storia completa e checkpoint all'epoca effettiva, evitando last errati |
| Swap Gappy mancante e interpretazione del donatore da completare | Tutti, con priorità diversa | Completamento o rinvio esplicito. Misura pred_swapped contro T_donor se adottata |

La preparazione preserva i risultati storici e distingue correzioni di contratto da cambiamenti scientifici. Un replay diventa necessario se serve a colmare una lacuna concreta o se una correzione cambia gli output. Rifare tutti i training non è una conseguenza automatica di un problema documentale.

## 9. Punti metodologici da fissare nelle due roadmap

1. **Domanda e misura del successo.** Dichiarare quale miglioramento avrebbe rilevanza per la ricostruzione da osservazioni limitate, includendo nuove istantanee, regioni calde e costo delle misure T.
2. **Controllo dei fattori.** Nell'esperimento dati, specificare quantità di frame, correlazione temporale, aggiornamenti, statistiche e seed. Nell'esperimento ancore, specificare siti, rumore ideale, K, regolarizzazione e informazioni consentite a ogni variante.
3. **Confronto appaiato.** Confrontare le stesse condizioni, frame e maschere dove applicabile. Più pixel e frame correlati non creano automaticamente più unità indipendenti.
4. **Selezione e valutazione.** Distinguere criterio primario congelato, rianalisi esplorative e nuovi protocolli. Riselezionare pesi dopo aver visto un risultato non cambia il riferimento storico.
5. **Conclusioni proporzionate.** Overfitting osservato non prova che più dati bastino. Assenza di miglioramento non prova impossibilità fisica. AO riuscito non certifica ancora una termocoppia reale. Uno sweep K=0/assisted mantiene visibile il cambiamento di informazione.
6. **Risorse operative.** Dimensionare da profiling e verificare account/QoS Advanced espliciti. Nei run riusciti CPU la coppia era `mit_amf_advanced_cpu` su `mit_normal`, nei run GPU `mit_amf_advanced_gpu` su `mit_normal_gpu`. Gli account disponibili e i limiti vanno ricontrollati al lancio. Registrare SHA prima dell'esecuzione, output distinti e checkpoint. Dopo la submission comunicare ID e percorsi e fermarsi, secondo il vincolo di Enrico sul polling. Nessun tetto scientifico arbitrario in ore.

Il suggerimento iniziale dell'assistente di limitare il primo confronto dati a p_on con due regimi e due seed era un'alternativa discussa. La scelta successiva di Enrico autorizza due sotto-missioni più ampie. Il sottoinsieme inizialmente suggerito non va trattato come matrice già approvata al posto della proposta Opus.

## 10. Collegamento con le roadmap e criterio di ripresa

| Livello | Stato dopo questa decisione | Come prosegue |
| --- | --- | --- |
| Missione U-Net densa U0–U7 | Conclusa nel perimetro dichiarato | Evidenza di riferimento con i suoi limiti |
| Sparse S4–S6 | Esecuzioni consegnate, rilievi di revisione aperti | Preparazione comune e completamenti circoscritti |
| Sotto-missione Opus | Approvata come direzione, piano da scrivere | Roadmap sulla quantità dei dati e variabilità |
| Sotto-missione Astra | Approvata come direzione, piano da scrivere | Roadmap livello/struttura/ancore dopo Opus |
| Proposte Neon | Conservate per riesame | Decidere priorità dopo entrambe le sotto-missioni |
| Sparse S7 | Decisione sospesa | Scegliere conferme o revisione motivata del percorso con le nuove evidenze |
| Mese 2 | Aperto | Completare gli impegni residui 2.7–2.9 e la sintesi, secondo lo stato effettivo |

Al termine di Opus e Astra, il confronto decisionale dovrà dire se il regime di dati modifica la prestazione sulle nuove istantanee, quanto il livello sia recuperabile con poche T e quali limiti rimangano cambiando copertura o condizione. Questi esiti determineranno se privilegiare multidensità, conferma architetturale, condizionamento o assisted.

La scelta del modello principale e l'eventuale contributo metodologico rimangono aperti. Le revisioni aiutano a preparare le decisioni, mentre la discussione con i supervisori scientifici mantiene il proprio ruolo. Il prossimo deliverable autorizzato nella sequenza è la preparazione delle roadmap, a partire da Opus. Questo passo 1 consegna l'analisi e conserva le proposte senza avviare esperimenti.
