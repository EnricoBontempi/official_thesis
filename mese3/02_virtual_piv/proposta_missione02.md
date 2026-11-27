# Missione 02 — Proposta: osservazioni PIV per la nostra LES e comportamento delle reti

> **Nota (27/11/2026):** per la definizione delle osservazioni e per la campagna questa proposta è **superata** da [roadmap_tecnica_piv.md](roadmap_tecnica_piv.md). Resta valida la scelta del GTMC come ancoraggio dei parametri. Sono abbandonati il trasferimento dei rigetti da Cardiff e i gruppi di training qui elencati.

**Data:** 27 novembre 2026 · **Branch:** `piv/experiment-proposal` · **Base:** `2f2faff588b4268d9c46f0efa8e324e9af359cc4`
**Stato:** proposta a parole da discutere con Enrico. Non è la roadmap tecnica: non contiene comandi, configurazioni né la matrice definitiva dei run. Non autorizza download, job, training, inferenze o accesso a TEST.

**Obiettivo della missione (già deciso, qui non ridiscusso):**
1. rendere le osservazioni di velocità della LES il più fedeli possibile a una misura PIV sperimentale pertinente alla combustione;
2. studiare come U-Net e GLU ricostruiscono la T istantanea da queste osservazioni.

**Materiale riusato** (consultato senza merge; le teste dei branch coincidono con gli SHA indicati):
- missione 00: `nonideal/specialist-results` @ `cb77e6ee8ade2e66ddd229bb1364267f4d0dd9e5`, `docs/nonideal/05_specialist_results.md` e tabelle;
- scouting dataset: `piv/step2-dataset-scouting` @ `ee9948e227f0d9272ae064cc8601c5e5399c28b4`;
- scouting letteratura: `piv/step2-literature-scouting` @ `2cbf5abd5fd44f8cb6f82c0be933f61e7340ddc5`;
- documenti della missione: [scheda Sciacchitano](step1_sciacchitano.md) e [roadmap 02](roadmap02.md);
- missione 01: [inventario](../01_observation_state_matrix/data_inventory.md) e [matrice](../01_observation_state_matrix/observation_state_matrix.md);
- risultati del Mese 2 con velocità densa: `docs/month2/03_dense_velocity_diagnostic.md`.

La proposta alternativa sullo studio dei limiti della misura è conservata in [proposta_esperimenti_v1.md](proposta_esperimenti_v1.md). Da lì riprendo le parti utili: fonti verificate, stime di costo, lettura della missione 00.

Uso tre etichette:
- **evidenza:** documentata in una fonte letta;
- **trasferimento:** valore documentato in un altro apparato e portato sulla nostra LES con una regola dichiarata;
- **ipotesi:** scelta non sostenuta da una fonte, da dichiarare e, dove possibile, verificare.

---

## 1. In breve

Propongo di rappresentare **una misura sola, ben documentata**: la stereo-PIV del combustore modello GTMC del DLR, una fiamma swirl confinata. La descrive Stöhr et al. (2012, Combustion and Flame 159); ho letto io le sezioni sperimentali del manoscritto.

Lo stesso articolo descrive due sistemi sullo stesso combustore, con lo stesso software:
- un sistema a **5 Hz che vede tutta la sezione** della camera;
- un sistema a **5 kHz limitato a 30×30 mm**.

Il primo è il riferimento principale: dà campi istantanei indipendenti su tutta la sezione, come il nostro compito a singolo frame. Il secondo è la variante che un esperimento risolto nel tempo deve accettare. Prenderli dallo stesso lavoro evita di costruire un apparato immaginario con pezzi di esperimenti diversi.

**Le osservazioni cambiano molto rispetto al vecchio 5 % casuale.** Non più 819 valori esatti in punti sparsi, ma:
- un vettore per ogni nodo della griglia dentro il campo di vista;
- ogni vettore è una media su circa 1,3 mm;
- un errore casuale correlato fra vettori vicini, di ampiezza derivata dalla fonte;
- il campo di vista come principale zona senza dati.

Rigetti e sostituzioni dei vettori non sono documentati per il GTMC. Li propongo come **variante separata**, con struttura presa da un altro dataset (Cardiff, isotermo) e dichiarata come trasferimento.

**Sulle reti propongo tre gruppi di esperimenti.**
1. Prima, test dei modelli esistenti: economici, dicono quanto e perché i modelli attuali soffrono il cambio di osservazione.
2. Poi un gruppo di **nuovi training sulle osservazioni PIV**, con un riferimento ideale riallenato con la stessa ricetta. È la parte centrale: le reti attuali hanno visto solo il 5 % casuale.
3. Infine, training aggiuntivi solo dove i test mostrano che servono.

Dimensione indicativa: circa 35–40 training U-Net e 12–15 GLU, cioè lo stesso ordine della missione 00. Rispetto alla 00 cambiano la distribuzione fra famiglie e la motivazione di ciascun gruppo (§6).

Prima di tutto ciò servono alcune **verifiche indispensabili** (§4). Le principali:
- capire quali componenti di velocità contiene davvero la slice;
- quanto conta un filtro di 1,3 mm sulla nostra griglia;
- qualificare il generatore delle osservazioni prima di usarlo sulle reti.

---

## 2. Quale misura PIV proponiamo di rappresentare, e perché

### 2.1 La scelta: stereo-PIV del GTMC DLR

Che cosa dice la fonte (evidenza; Stöhr et al. 2012, manoscritto DLR elib 77484, §§2.3.2–2.3.3, letto da me):

**Sistema a 5 Hz:**
- foglio laser che copre "the entire vertical section of the combustion chamber", spesso circa 1 mm;
- due camere stereo con filtro passa-banda a 532 nm "per ridurre l'influenza della luminosità di fiamma";
- particelle di TiO₂ di 1 µm nell'aria;
- software DaVis, correlazione multi-scala, finestra finale 16×16 px con overlap del 50 %, "risoluzione spaziale in piano di 1,3×1,3 mm²";
- incertezza casuale in piano da "±0,3 m/s (10 kW) a ±1,4 m/s (35 kW)", ricavata dall'incertezza di ±0,1 px nella localizzazione del picco;
- incertezza della componente fuori piano circa tre volte maggiore;
- errore da slip delle particelle stimato al massimo al 2 %.

**Sistema a 5 kHz (stesso combustore, stesse particelle, stessi algoritmi):**
- campo di vista "limited to a region of 30×30 mm²";
- risoluzione 1,5×1,5 mm²;
- incertezza ±0,5 m/s;
- fuori piano circa il doppio.

Oberleithner et al. (2015, Combustion and Flame 162, §§2.2 e 2.5, manoscritto letto da me) confermano la stessa catena sul GTMC. Per il 5 Hz: 16×16 px, 1,3 mm, overlap 50 %, ±0,8 m/s. Per il 10 kHz: risoluzione 1,0 mm e campo ridotto.

Perché questa misura:
1. **È pertinente alla combustione** e al nostro tipo di fiamma: swirl confinato, camera di 85×85 mm, che è lo stesso ordine della nostra slice di 73,8×96,2 mm.
2. **Le particelle solide sopravvivono nei gas combusti**, quindi la velocità è misurata anche nei prodotti. Con l'olio che evapora (flashback UT Austin) la velocità esisterebbe solo nel gas freddo: è un'altra misura, che lascio fuori dal riferimento (§2.3).
3. **I parametri vengono da un unico articolo** che descrive entrambi i sistemi: niente assemblaggi fra apparati.
4. **Il sistema a 5 Hz fornisce snapshot indipendenti a pieno campo**, esattamente il tipo di dato su cui facciamo ricostruzione a singolo frame.
5. **Il compromesso frequenza/campo di vista è documentato nello stesso lavoro**, quindi possiamo rappresentarlo senza inventarlo.

Limiti da dichiarare subito:
- il GTMC brucia CH₄/aria (10–35 kW, φ 0,65), non NH₃/CH₄;
- la geometria del nostro combustore non è nota;
- non esistono dati vettoriali GTMC pubblici con cui confrontare il nostro generatore.

Rappresentiamo quindi **una classe di misura documentata**, non una riproduzione dell'esperimento DLR.

### 2.2 Che cosa della misura entra e che cosa resta fuori

| Caratteristica | Nel riferimento? | Motivo |
|---|---|---|
| Campo di vista (pieno campo; 30×30 mm nella variante) | Sì | Evidenza diretta |
| Griglia regolare dei vettori | Sì | Finestra e overlap documentati; il passo è dedotto |
| Risposta spaziale (media pesata su ~1,3 mm) | Sì | Evidenza; la forma del filtro è un trasferimento |
| Errore casuale e sua correlazione | Sì | Ampiezza come trasferimento; correlazione da una relazione documentata in metrologia |
| Rigetti e sostituzioni | **Variante separata** | Il GTMC non ne pubblica statistiche (§2.4) |
| Terza componente (stereo) | Rinviata | Prima va verificato che k sia la componente fuori piano (§4) |
| Spessore del foglio, errori propri della stereo, rifrazione, luminosità, slip | No | Non rappresentabili su una slice, oppure piccoli o non quantificati in modo trasferibile. Elencati come esclusioni |

### 2.3 Alternative considerate e perché non come riferimento

- **DLR a 5 bar** (Karmarkar et al., preprint letto dallo scout): combustore pressurizzato a etilene, campo di 54×22 mm, finestra di 1,9 mm, incertezze a posteriori con statistica non definita. Utile come **configurazione di test non vista** (§5.3), non come riferimento.
- **Flashback swirl UT Austin, con olio:** velocità solo nel gas incombusto (evaporazione riportata a circa 555 K in quelle condizioni). Interessante, ma richiede di usare T per generare la maschera e si appoggia a un solo esperimento. Resta un'estensione eventuale.
- **Cardiff, bruciatore swirl, PIV isoterma:** non è una misura in fiamma, ma è l'unico archivio con i vettori a tre stadi di validazione. Lo uso per la **variante dei rigetti**, non come apparato di riferimento.
- **Neal et al. (getto non reagente):** non è combustione. Serve come banco metrologico per verificare la forma dell'errore (§4).
- **EngineBench e TCC-III:** motore ottico. Lo trascinato è già studiato; per l'acceso il contenuto non è verificato. Fuori dalla missione.

### 2.4 Rigetti e sostituzioni: perché una variante separata

In una PIV reale una parte dei vettori viene rigettata e spesso sostituita per interpolazione. Ignorarlo renderebbe l'osservazione ottimista. Però:
- per il GTMC sappiamo solo che i buchi di un singolo vettore venivano interpolati (Boxx et al. 2010, tramite lo scout);
- nessuna fonte letta dà frazioni di rigetto per frame in una fiamma swirl stabilizzata.

L'unico dato verificabile è Cardiff. In 4 frame su 514 e con un decoder non ufficiale, lo scout trova:
- una regione fissa di circa 800 vettori su 4941 senza misura;
- un ulteriore 15–20 % di vettori con il codice interpretato come "sostituito".

Proposta: una **variante "riferimento + validazione trasferita da Cardiff"**, sempre riportata separatamente e con l'etichetta di trasferimento. Avrà due forme:
- **con flag:** vettori rigettati come mancanti espliciti;
- **senza flag:** vettori sostituiti con la regola di processamento documentata da Cardiff, presentati come misurati.

Non la chiamerò "PIV GTMC realistica". Dipende dalla verifica del decoder Cardiff (§4); senza quella non si propone.

---

## 3. Come cambiano le osservazioni rispetto al vecchio 5 %

| Aspetto | Prima (missione 00) | Proposta | Natura della scelta |
|---|---|---|---|
| Posizioni | 819 punti casuali (5 %), cambiati a ogni epoca | Un vettore per nodo LES dentro il campo di vista: circa 16 000 a pieno campo, circa 2 000 nel 30×30 mm | Il passo GTMC dedotto (circa 0,65 mm) è vicino alla cella (0,58×0,76 mm): **trasferimento** con approssimazione dichiarata |
| Valore del vettore | Valore puntuale esatto della LES | Media pesata del campo LES su circa 1,3 mm (filtro gaussiano; un top-hat come alternativa da confrontare) | Scala: **evidenza**. Forma: **trasferimento** dalla metrologia DaVis di Wieneke, non verificata sul GTMC |
| Errore | Gaussiano indipendente, σ = 0,5 o 1,5 m/s, scelti come scenari | Errore correlato fra vettori vicini, con la stessa forma del filtro; ampiezza dell'ordine di 0,8 m/s | Ampiezza: **trasferimento** del rapporto GTMC fra incertezza e velocità massima (1,4 m/s su circa 80 m/s) alla velocità massima della nostra LES (p99,9 circa 47 m/s). Correlazione: relazione documentata nella metrologia (Wieneke), da verificare con Neal. Interpretare ± come deviazione standard è un'**ipotesi** |
| Zone senza dati | Dischi sintetici al 25 % dell'area | Fuori dal campo di vista; fascia di bordo dove il filtro esce dal dominio; nella variante, rigetti con struttura Cardiff | Campo di vista: **evidenza**. Rigetti: **trasferimento** da apparato isotermo |
| Informazione di validità | Maschera esplicita (U-Net) o token rimossi (GLU) | Uguale; nella variante senza flag i vettori sostituiti sembrano misurati | Scelta di confronto |
| Componenti | vi, vj | vi, vj, interpretate come componenti in piano | **Ipotesi** da verificare |
| Tempo | Singolo snapshot | Singolo snapshot; la misura è approssimata come istantanea | Come nella 00 |

**Conseguenza per le reti.**
- La U-Net vedrebbe una maschera quasi tutta piena invece del 5 %: una situazione mai incontrata in training.
- Il GLU riceverebbe migliaia di token invece di 819.

Le reti attuali sono quindi fuori distribuzione anche prima di considerare filtro ed errore.

**Un dato da tenere presente.** Con velocità densa ideale la U-Net raggiunge circa 26,9 K di RMSE sulla VAL (Mese 2, un seed), contro circa 40 K con il 5 % casuale. La nuova osservazione è meno fedele punto per punto, ma molto più ricca dentro il campo di vista. Non è detto che le reti peggiorino: potrebbero migliorare a pieno campo e peggiorare molto fuori dal campo ridotto. È proprio ciò che vogliamo misurare, senza aspettarci un esito.

**Che cosa non facciamo.** Livelli e dimensioni non vengono scelti in base a quanto danneggiano le reti. Se filtro ed errore alla scala documentata hanno un effetto piccolo, lo riportiamo come tale. Le sensibilità usano solo valori documentati, come gli estremi ±0,3–1,4 m/s o la finestra di 1,9 mm del DLR a 5 bar, non valori scelti per allargare la forbice.

---

## 4. Verifiche preliminari indispensabili

Servono a rendere credibile la rappresentazione prima di qualunque valutazione delle reti. Ne propongo sei. Le prime tre sono **bloccanti**; le altre tre condizionano parti specifiche della proposta.

1. **Quali componenti di velocità contiene la slice (bloccante).** La documentazione non dice come i/j/k si orientano rispetto al piano. Propongo un controllo sui dati TRAIN: confrontare lo spostamento apparente delle strutture nei campi scalari fra frame successivi (0,25 ms) con vi, vj e vk.
   - Se i/j sono in piano, la rappresentazione 2C è giustificata.
   - Se k risulta fuori piano, diventa possibile in seguito anche la variante stereo con tre componenti.
   - T e specie si usano solo per questa qualifica, mai come input.
   - Se il controllo non è conclusivo, l'orientamento resta un'ipotesi dichiarata e la domanda va agli autori tramite Enrico.

2. **Che cosa fa un filtro di 1,3 mm sulla nostra griglia (bloccante).** Il filtro copre circa due celle. La LES ha già un filtro e un ricampionamento propri, ignoti. Prima delle reti va misurato, su TRAIN e senza modelli:
   - quanto cambia la velocità filtrando a 1,0–1,9 mm;
   - dove avviene il cambiamento (vicino al fronte, nei getti);
   - quanto contenuto ha la LES a quelle scale.

   Dice in anticipo se il filtro è un effetto apprezzabile o soltanto nominale su questi dati, ed evita di sopravvalutarlo. Una stima della ricognizione indica 0,6 m/s RMS per una media su 3×3 celle.

3. **Qualificazione del generatore (bloccante).** Prima di qualunque confronto, controllare che le osservazioni generate abbiano le proprietà dichiarate:
   - risposta del filtro;
   - ampiezza e correlazione dell'errore;
   - geometria dei campi di vista;
   - conteggi e unità;
   - ordine delle operazioni: filtro in m/s, campionamento, errore, rigetti, normalizzazione;
   - identità delle osservazioni date a U-Net e GLU.

   La qualificazione certifica solo queste proprietà, non la fedeltà a una fiamma reale.

4. **Cardiff: decoder e statistiche dei rigetti (per la variante §2.4).**
   - Validare il decoder non ufficiale confrontando la media dei 514 campi decodificati con la statistica salvata da DynamicStudio nello stesso database.
   - Verificare la regione fissa contro la maschera definita nel database.
   - Ricavare la scala fisica.
   - Misurare, sui 514 frame, frazione e dimensione dei cluster di vettori con codice di rigetto e di sostituzione.

   Il significato dei codici resterà un'interpretazione finché manca la documentazione ufficiale. Serve scaricare circa 0,8 GB di vettori tramite HTTP Range: **download da autorizzare**, non eseguito.

5. **Neal: forma dell'errore (per la correlazione).**
   - Elaborare le immagini del sistema di misura con un software aperto e documentato.
   - Confrontarle con il riferimento HDR simultaneo, allineando il suo kernel di 5 coppie.
   - Verificare se l'errore è correlato come prevede la relazione con il filtro.
   - Risolvere la discrepanza aperta sulla correlazione al 50 % di overlap.

   Limite: è un getto non reagente, e l'HDR ha a sua volta un filtro, outlier interpolati senza flag e un ricampionamento. Serve scaricare 272 MB: **da autorizzare**. Senza questa verifica la correlazione resta un trasferimento dichiarato, non verificato.

6. **Letture mancanti.**
   - Runyon et al. 2018: scala e piano Cardiff.
   - Boxx et al. 2010: pratica di interpolazione sul GTMC.
   - Wieneke 2015 e il capitolo sulla risposta spaziale della sua tesi, per le configurazioni DaVis al 50 % di overlap.
   - Neal et al. 2015.

---

## 5. Test sui modelli esistenti

### 5.1 Quali modelli

- I **12 modelli originali al 5 %** (U-Net e GLU, con e senza contesto NH₃/φ, tre seed), cioè la base del Blocco II.
- Le **due U-Net dense del Mese 2** (con e senza contesto, seed 0). Sono state allenate su velocità ideale ovunque e non hanno un canale di maschera, quindi si possono provare solo su osservazioni a pieno campo senza buchi.
- I **30 specialisti** della 00 non sono prioritari. Una valutazione economica dei G1/G1-eq sul campo ridotto è possibile, ma non la ritengo necessaria.

### 5.2 Quali test e perché

1. **Osservazione PIV completa** (pieno campo e 30×30 mm; nella variante anche con rigetti) su seen e VAL. Misura quanto i modelli attuali reggono il passaggio alla misura PIV: è la domanda dello step 5 della roadmap.
2. **Scala di osservazioni**, aggiungendo un elemento alla volta: velocità densa ideale → con filtro → con errore → con campo di vista ridotto → con rigetti. Mostra **dove** i modelli esistenti si rompono: cambio di layout, filtro, errore o copertura.
3. **Stessi 819 punti del vecchio layout, ma con valori filtrati e con l'errore PIV.** Separa l'effetto della fisica della misura, che cambia i valori, dall'effetto del cambio di layout, che manda i modelli al 5 % fuori distribuzione. È il ponte diretto con la missione 00.

### 5.3 Che cosa possono e non possono dire

Questi test misurano la **robustezza di modelli costruiti per un'altra osservazione**. Un peggioramento forte non dice che la PIV contiene poca informazione su T, ma che questi modelli non la sanno usare. Per questo servono anche i nuovi training (§6).

Costo stimato: pochi GPU-h, scalando dal Blocco II (1,56 GPU-h per 12 modelli × 6 scenari × 3 repliche). Il GLU con migliaia di token può costare di più in valutazione.

---

## 6. Nuovi training proposti

### 6.1 Perché servono, alla luce della missione 00

La missione 00 ha mostrato che gli specialisti allenati da zero su un singolo degrado sintetico:
- non recuperano in modo consistente su condizioni mai viste;
- in media costano sul caso pulito;
- variano molto fra seed.

Valeva però per **degradi di un layout invariato** e con **selezione sulla VAL pulita**. Qui il caso è diverso:
- cambia l'osservazione stessa: densa, filtrata, con campo di vista;
- il "caso pulito" (velocità esatta) non esiste in nessun esperimento;
- un modello pensato per la PIV deve essere allenato sull'osservazione che riceverà.

Non è un'augmentation aggiuntiva: è il nuovo compito.

I limiti della 00 entrano nel disegno così:
- **stessa ricetta per riferimento e PIV.** Il riferimento ideale si riallena con la ricetta dei modelli PIV, invece di confrontarli con originali allenati diversamente;
- **selezione coerente con il compito:** sulla VAL dell'osservazione con cui il modello è allenato; il checkpoint `last` si riporta sempre;
- **più seed dove costano poco:** nella 00 la dispersione fra seed valeva 1–4 K. Propongo 5 seed per la U-Net e 3 per il GLU, che costa di più;
- **stesso batching per tutti i GLU,** più un piccolo ponte che lo collega agli originali (§6.2, gruppo E);
- **niente regime singolo come unica strategia:** un training su più configurazioni (§6.2, gruppo D) è proposto solo se i test mostrano che serve.

### 6.2 Gruppi di training

**A. Riferimento a pieno campo (5 Hz) e riferimento ideale denso, con la stessa ricetta.**
- U-Net: 5 seed per ciascuno, 10 run. GLU: 3 seed per ciascuno, 6 run.
- **Contributo:** misura che cosa la PIV a pieno campo permette di ricostruire con un modello allenato per lei, e quanto si perde rispetto alla velocità ideale densa. È la parte centrale della missione.
- Il riferimento ideale denso riallenato serve anche da **ancoraggio al Mese 2** (26,9 K, un seed, ricetta diversa).

**B. Campo ridotto 30×30 mm (5 kHz) e il suo controllo ideale.**
- U-Net: 5 seed per la PIV a campo ridotto e 5 per la velocità ideale nello stesso campo, 10 run. GLU: 3 seed sulla PIV a campo ridotto.
- **Contributo:** misura la ricostruzione con il campo di vista dei sistemi ad alta frequenza, dentro e fuori dalla finestra osservata. Il controllo ideale nello stesso campo separa la perdita dovuta alla copertura da quella dovuta alla qualità dei vettori.
- La posizione del campo nella slice non è documentata per il nostro combustore. Propongo poche posizioni fissate prima dei risultati, senza sceglierle in base alle prestazioni. Il training è su una posizione; le altre si usano in valutazione.
- Il GLU qui ha circa 2 000 token ed è economico.

**C. Variante con rigetti trasferiti da Cardiff, con e senza flag.**
- U-Net: 5 seed per ciascuna forma, 10 run.
- **Contributo:** dice se rigetti e sostituzioni dalla struttura documentata cambiano la ricostruzione. Dice anche se conta sapere quali vettori sono stati sostituiti: un'indicazione concreta su come le misure andrebbero rilasciate.
- **Condizione:** solo se la verifica Cardiff del §4 va a buon fine. Il GLU si aggiunge solo se la U-Net mostra un effetto chiaro.

**D. Training su più configurazioni (condizionato).**
- Indicativamente 5 seed U-Net e 3 GLU.
- **Quando:** solo se i test incrociati mostrano un costo forte nel passare da una configurazione all'altra (modelli A provati sul campo ridotto, modelli B a pieno campo, entrambi sulla finestra di 1,9 mm e sugli estremi documentati dell'errore).
- **Contenuto:** campionare in training le configurazioni documentate.
- **Contributo:** risponde al limite principale della 00, in cui gli specialisti a regime singolo non trasferivano. È giustificato solo se il problema si ripresenta qui.

**E. Ponte di ricetta GLU.**
- 3 seed, circa 27 GPU-h ai costi della 00.
- **Contenuto:** GLU ideale al 5 % con il batching della nuova campagna.
- **Contributo:** separa la differenza di ricetta dalla differenza di osservazione e chiude il confondente del batching lasciato aperto dalla 00.

**Dimensioni indicative.**
- Circa 30 U-Net sicure (A, B, C), più 5 condizionate (D).
- Circa 12 GLU sicuri (A, B, E), più 3 condizionati.

Non è una matrice definitiva: potrà ridursi se le verifiche preliminari mostrano un filtro nominale, o crescere di pochi run se serve separare filtro ed errore in training.

**Costo indicativo.**
- U-Net: 1–5 GPU-h per run, dalla profilazione densa del Mese 2 fino al costo degli specialisti con valutazioni frequenti e preemption.
- GLU: 9 GPU-h per run a 819 token. A pieno campo con circa 16 000 token il costo è incerto (fino a 10 volte); va profilato. Un'opzione è usare un nodo su due come layout dei token, dichiarata come tale.
- Totale ordine di grandezza: **150–600 GPU-h**, contro 210 della missione 00.

### 6.3 Che cosa non propongo

- Ripetere la matrice dei 30 specialisti con nuovi livelli di rumore indipendente.
- Sweep di livelli scelti per danneggiare le reti.
- Nuove architetture o augmentation prima di aver visto i risultati.
- Generare immagini di particelle dalla slice: non c'è un volume né le traiettorie.
- DMF-Gen o fusione termica come parte di questa missione. Lo stesso modello di osservazione potrà servire a entrambe in seguito.

---

## 7. Come leggeremo i risultati

**Riferimento e validazione.** Tutta l'accuratezza è misurata rispetto alla T della LES: è **validazione numerica**. Cardiff, Neal e la letteratura GTMC qualificano parti della misura, non la temperatura. Nessuna fonte trovata offre T sperimentale simultanea alla PIV nella nostra configurazione: non parleremo di validazione sperimentale della T.

**VAL e TEST.**
- La VAL è già stata usata per selezione e sviluppo: resta il terreno di sviluppo.
- Prima di qualunque apertura di TEST scriveremo i confronti e le metriche principali. TEST si apre una sola volta, su decisione esplicita di Enrico.

**Repliche.**
- Per le affermazioni sui training conta il seed.
- Per la generalizzazione contano le condizioni: 3 in VAL, 7 in TEST, quindi intervalli larghi.
- Frame della stessa condizione e repliche di maschera o rumore non sono repliche indipendenti.

**Metriche.** Oltre alla RMSE globale (con bias e parte centrata), quantità utili alla combustione:
- errore nella zona calda e nei getti freddi;
- sovrapposizione della zona calda;
- distanza fra isoterme vere e predette;
- errore vicino al fronte;
- per la PIV: errore dentro e fuori dal campo di vista, e in funzione della distanza dal vettore valido più vicino.

Tutto è riportato anche per condizione.

---

## 8. Assunzioni che restano e rischi

- **Orientamento delle componenti:** ipotesi finché la verifica 1 del §4 non lo chiarisce.
- **Geometria:** la slice è trattata come "sezione della camera" e la posizione del campo di 30×30 mm è arbitraria. I risultati vanno riportati per posizione.
- **Errore:** interpretare ±0,1 px come deviazione standard e trasferirlo come frazione della velocità massima sono ipotesi. L'errore reale in fiamma è probabilmente più grande nei prodotti, dove le particelle sono più rade; il riferimento non lo rappresenta.
- **Filtro:** la forma è trasferita da metrologia su immagini sintetiche. Sulla nostra griglia l'effetto potrebbe essere piccolo, e il filtro implicito della LES è ignoto.
- **Rigetti:** la variante usa un apparato isotermo. In fiamma i rigetti possono essere più frequenti e legati al fronte, ma non abbiamo dati per quantificarlo.
- **Non rappresentabili:** spessore del foglio, errori stereo, rifrazione, luminosità di fiamma (anche NH₂* nelle fiamme ad ammoniaca, non quantificata).
- **Costo:** il GLU con molti token resta incerto fino alla profilazione.

---

## 9. Punti da discutere con Enrico

1. GTMC DLR a 5 Hz come riferimento e 5 kHz 30×30 mm come variante dello stesso articolo.
2. Rigetti come variante separata trasferita da Cardiff, con e senza flag.
3. Solo due componenti finché l'orientamento non è verificato.
4. Le sei verifiche preliminari, e in particolare l'autorizzazione ai download Cardiff (circa 0,8 GB) e Neal (272 MB).
5. I test sui modelli esistenti, incluso il ponte "819 punti con valori PIV".
6. I gruppi di training A–E, i seed (5 U-Net, 3 GLU) e la condizione per D.
7. GLU a pieno campo con tutti i token o un nodo su due, da decidere dopo la profilazione.
8. Ordine di grandezza del budget e priorità rispetto alle missioni 03 e 04.

Dopo l'approvazione si preparerà la roadmap tecnica da affidare a Sonnet IDE.
