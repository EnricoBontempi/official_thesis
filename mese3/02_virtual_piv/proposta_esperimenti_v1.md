# ALTRA PROPOSTA — Studio dei limiti della misura PIV

> **Proposta alternativa conservata per discussione futura. Non costituisce il piano approvato della missione 02.**
>
> La proposta della missione 02 da discutere con Enrico è in [proposta_missione02.md](proposta_missione02.md).

---

# Missione 02 — Proposta di esperimenti v1 (da discutere con Enrico)

**Data:** 27 novembre 2026 · **Branch:** `piv/experiment-proposal` · **Base:** `2f2faff588b4268d9c46f0efa8e324e9af359cc4`
**Natura:** proposta scientifica iniziale. **Non** è la roadmap esecutiva, non contiene configurazioni di training e non autorizza download, job, training, inferenze o accesso a TEST. Numeri di run e costi sono stime per la discussione.

**Consegne consultate senza merge** (verificato con `git fetch` il 27/11 che le tre teste dei branch coincidono con gli SHA indicati: nessuna revisione successiva):

| Consegna | Branch @ SHA | File letti |
|---|---|---|
| Missione 00, specialisti | `nonideal/specialist-results` @ `cb77e6ee8ade2e66ddd229bb1364267f4d0dd9e5` | `docs/nonideal/05_specialist_results.md` (Parte 1 e Appendici A–N); `results/nonideal/specialists/INDEX.md`; ricalcolati da me guadagni e costi da `package/tables/per_run_scenario.csv`, più `reading/transfer_row_vs_column_decomposition.csv`, `accounting/training_by_run.csv` |
| Scouting dataset v2 | `piv/step2-dataset-scouting` @ `ee9948e227f0d9272ae064cc8601c5e5399c28b4` | `mese3/02_virtual_piv/scouting/step2_datasets.md`; elenco di script e manifest (non rieseguiti) |
| Scouting letteratura v2 | `piv/step2-literature-scouting` @ `2cbf5abd5fd44f8cb6f82c0be933f61e7340ddc5` | `mese3/02_virtual_piv/scouting/step2_literature.md` |

Letture nel checkout: `mese3/README.md`; `mese3/02_virtual_piv/{README,roadmap02,step1_sciacchitano}.md`; `mese3/01_observation_state_matrix/{data_inventory,observation_state_matrix}.md` e `updates/piv_2026-11-26.md`; `docs/proposal.md`; `docs/nonideal/00_ricognizione_osservazioni.md`, `02_block2_robustness.md`, `03_specialist_handoff.md`, `04_specialist_smoke_criteria_and_budget_policy.md`; `docs/month2/03_dense_velocity_diagnostic.md` (risultati con velocità densa); voce del 25/11 di `docs/RESEARCH_LOG.md`. Nella repository non esistono `AGENTS.md`.

Etichette usate: **D** documentato (fonte indicata); **I** mia inferenza; **M** mancante. Livelli di lettura come negli scouting: **FT** testo integrale; **FT-v** integrale di un manoscritto o preprint; **AB** abstract; **SEC** riferito da altri; **SN** snippet.

---

# PARTE 1 — Per Enrico

## 1. Il problema in una pagina

Finora le reti hanno ricevuto la velocità **esatta** della LES in 819 punti casuali (5 %). Una PIV reale dà un'osservazione quasi opposta:

- **densa** dentro il campo di vista, con un vettore ogni 0,5–1 mm circa (dell'ordine della cella LES);
- **filtrata**: ogni vettore è una media pesata su circa 1–2 mm, non un valore puntuale;
- **limitata** a un campo di vista, spesso piccolo nei sistemi ad alta frequenza;
- **affetta da errori correlati** fra vettori vicini;
- **validata e spesso riempita**: alcuni vettori sono rigettati e sostituiti, a volte senza un flag che lo dica.

Lo studio della missione 00 ha esplorato rumore indipendente e lacune a dischi **sul layout al 5 % casuale**. Non ha toccato copertura densa, filtro, campo di vista, correlazione degli errori né sostituzione. Sono proprio le caratteristiche che distinguono la PIV.

C'è un dato già in casa che cambia la prospettiva. Con velocità **densa ideale** la U-Net p_on raggiunge 26,9 K di RMSE sulla VAL (Mese 2, U4, un seed), contro circa 40 K con il 5 % casuale. Quindi una PIV realistica non è necessariamente "peggio" del nostro caso attuale. Guadagna copertura, perde fedeltà puntuale e porzioni di dominio. **Non sappiamo quale effetto prevalga, né quale limitazione della misura controlli l'inferenza di T.** È questa la domanda che propongo.

## 2. Proposta raccomandata

> **Q1. Quanto della temperatura istantanea in una fiamma swirl turbolenta NH₃/CH₄ si può inferire da una PIV planare realistica? Quale limitazione della misura controlla l'errore: campo di vista, risposta spaziale, errore casuale correlato oppure validazione e sostituzione? Un training coerente con la misura recupera la perdita, e resta affidabile se il modello di misura è sbagliato?**

**Configurazione di riferimento: una sola famiglia di apparati documentata.** Propongo la stereo-PIV del Gas Turbine Model Combustor del DLR, come descritta da Stöhr et al. (2012). Ho letto io le sezioni sperimentali del manoscritto (FT-v, §§2.3.2–2.3.3). Lo stesso articolo descrive due sistemi sullo stesso combustore:

- **"pieno campo" a 5 Hz:** foglio laser che copre "the entire vertical section of the combustion chamber"; finestra finale 16×16 px con overlap 50 %, risoluzione dichiarata 1,3×1,3 mm²; incertezza casuale in piano "±0,3 m/s (10 kW) – ±1,4 m/s (35 kW)" ricavata da ±0,1 px; foglio di circa 1 mm;
- **"alta velocità" a 5 kHz:** campo di vista "limited to a region of 30×30 mm²"; risoluzione 1,5×1,5 mm², ±0,5 m/s, stesso algoritmo.

Per la ricostruzione **istantanea a singolo frame**, che è il nostro compito, il sistema a 5 Hz è perfettamente pertinente: dà snapshot indipendenti a pieno campo. Il sistema a 5 kHz rappresenta il compromesso fra frequenza e copertura che un esperimento risolto nel tempo deve accettare. Prendere entrambi dallo stesso lavoro evita di comporre un apparato fittizio con parti di esperimenti diversi.

**Perché serve.**

1. **Diagnostica.** La PIV è molto più diffusa della termometria planare nei combustori swirl. Capire che cosa una PIV realistica dice su T, e dove smette di dirlo, è una domanda di diagnostica della combustione. Non è soltanto una domanda di benchmark ML.
2. **Metodo.** Risponde ai limiti identificati nella missione 00: osservazioni non legate a un apparato, selezione sulla VAL pulita, confondente di batching nel GLU, dispersione fra seed paragonabile agli effetti.
3. **Progetto sperimentale.** Il risultato atteso è una mappa dell'inferibilità di T: per zona calda, getti freddi, posizione delle isoterme e distanza dal campo di vista. Può aiutare a decidere campo di vista e risoluzione, e a valutare l'utilità di rilasciare i flag di validità.

**Ipotesi falsificabili (dettagli al §B.1).**

- **H1, copertura contro fedeltà:** a pieno campo, filtro ed errore ai livelli documentati costano poco rispetto alla velocità densa ideale; il campo di vista di 30×30 mm costa molto di più. È falsificata se filtro più errore costano quanto o più della riduzione del campo di vista.
- **H2, training coerente:** una rete allenata su velocità ideali perde accuratezza sulla PIV realistica in modo che il training sulla misura recupera. È falsificata se la perdita da disallineamento resta entro la dispersione fra seed: risultato utile, coerente con la missione 00.
- **H3, metadati:** dare alla rete i flag dei vettori invalidati riduce l'errore rispetto agli stessi dati riempiti senza flag. È falsificata se la differenza resta entro la dispersione fra seed.
- **H4, robustezza:** un modello allenato sulla configurazione di riferimento degrada su configurazioni documentate ma non viste (per esempio finestra di 1,9 mm o un'altra posizione del campo di vista). L'eventuale training su una famiglia di operatori riduce questo degrado.

**Non prometto** che filtro o errore abbiano un effetto grande. Una prima stima della ricognizione indica che una media 3×3 celle cambia la velocità di 0,63 m/s RMS, cioè circa il livello N1, che costava 0,2 K. Se l'effetto è piccolo, lo riporteremo come tale: dire che la PIV densa a pieno campo basta per T con questi modelli sarebbe di per sé un risultato.

## 3. Che cosa abbiamo già imparato (e che cosa no)

| Conclusione della missione 00 | Orienta davvero | Limitata dal disegno |
|---|---|---|
| Le lacune contigue G1 costano di più (VAL +4,1…+4,3 K, p_on); N1 i.i.d. 0,5 m/s è trascurabile (+0,2 K) | Il danno dipende dalla **struttura** delle osservazioni mancanti più che da un rumore piccolo e indipendente | Layout al 5 % casuale. Dischi e σ non qualificati. Nessun filtro, nessuna correlazione. Non dice che le lacune domineranno con PIV densa |
| Gli specialisti non recuperano su VAL: guadagno medio positivo in 2 celle su 10 (+0,29 e +0,10 K); 8 run su 30 positivi | Allenare su **un solo** regime sintetico, da zero, senza campioni puliti, non è una strategia affidabile **per questi regimi** | Selezione sulla VAL **pulita**, non sullo scenario; 3 seed; regimi ipotetici; non prova l'inutilità del retraining |
| Costo sul pulito sempre positivo su VAL (U-Net 0,5–3,9 K; GLU 1,3–4,8 K) | Un modello specializzato può perdere generalità | Stessa selezione; seed non appaiati in senso causale |
| Il comportamento dipende dalla riga (regime di training: 67–82 % della varianza fra celle su VAL) più che dallo scenario valutato (9–17 %) | La "firma" del training conta più del test: occorre separare l'informazione disponibile dalla ricetta | È una scomposizione descrittiva, non un test |
| seen e VAL divergono (GLU last: specialisti migliori sul seen, peggiori su VAL) | Valutare sempre su condizioni mai viste | seen e VAL differiscono per condizioni, numero e difficoltà: il divario non prova da solo sovradattamento |
| Dispersione fra seed: originali VAL pulita U-Net 38,2–42,2 K, GLU 41,5–43,5 K; guadagni da −9,5 a +3,2 K su VAL | Effetti sotto qualche K richiedono più seed e confronti appaiati | Tre seed non danno intervalli di confidenza |
| GLU originali a batch 8, specialisti a microbatch 1 con accumulo | Nel nuovo disegno tutti i GLU devono seguire la stessa ricetta | Il confronto originale-specialista GLU resta confuso |
| Costo: 210,1 GPU-h, 95 % training; U-Net circa 4,5 GPU-h/run, GLU circa 8,9 GPU-h/run (incluse valutazioni periodiche e preemption); circa 53 h di calendario | Base per stimare i costi | Nessuna latenza d'inferenza misurata; i tempi di valutazione includono osservazioni, I/O e metriche |

## 4. Esperimenti proposti (vista d'insieme)

| Blocco | Domanda | Costo indicativo | Stato |
|---|---|---|---|
| **V — Verifiche preliminari** (step 3 e 4) | Il generatore rappresenta proprietà vere? Che cosa contiene la LES alla scala del filtro? | CPU; download ≈1,1 GB da autorizzare | **Necessario prima di tutto** |
| **C1 — Checkpoint esistenti** | Quanto perdono i modelli attuali su osservazioni PIV realistiche? | ≈3–10 GPU-h | Necessario, economico |
| **C2 — Scala di misura, U-Net** | Quale modulo della misura costa e quanto, con training coerente? | 7 celle × 5 seed = 35 run, ≈40–170 GPU-h | **Cuore della conclusione** |
| **C3 — GLU sul sottoinsieme decisivo + ponte di ricetta** | Le conclusioni dipendono dall'architettura? Il batching spiegava qualcosa? | 9 + 3 run, ≈50–830 GPU-h (incertezza dominata dal numero di token) | Necessario in forma ridotta; profilazione prima |
| **C4 — Disallineamento e generalizzazione** | Che cosa succede se il modello di misura è sbagliato o la configurazione è nuova? | Solo valutazioni, ≈5–20 GPU-h | Necessario |
| **C5 — Training su una famiglia di operatori** | Riduce il costo del disallineamento? | 5 U-Net + 3 GLU | **Condizionato** a C4 |
| **E1 — Tracciante che evapora** | Il confine delle particelle porta informazione su T? | ≈15 run U-Net | Estensione |
| **E2 — Terza componente (stereo)** | Che cosa aggiunge vk? | 1–2 celle | Estensione, bloccata da verifiche |

Ordine di grandezza del nucleo (V, C1–C4): **≈100–1000 GPU-h**. Con GLU su token ridotti, o solo sul campo di vista piccolo, si scende verso 150–300 GPU-h. Il dettaglio è al §B.6.

## 5. Decisioni da discutere

1. **Domanda principale Q1** (contro Q2 tracciante e Q3 progetto del campo di vista, §B.1).
2. **Riferimento GTMC del DLR (Stöhr 2012):** accettarlo come classe di misura, pur essendo CH₄/aria a combustibile e scala diversi dai nostri.
3. **Solo 2C in piano** finché orientamento e componenti della slice non sono verificati (§B.3.0). La stereo 3C resta un'estensione.
4. **Autorizzazioni per lo step 3:** vettori Cardiff (≈0,8 GB tramite HTTP Range) e XML; regione "unsteady core" di Neal (272 MB); un nuovo tentativo su Cambridge Oxycoal. TCC-III fired non ora.
5. **Livello del dato:** mancanti espliciti con flag come caso principale, dati riempiti senza flag come contrasto (H3).
6. **Modulo di validazione:** non esistono statistiche di rigetto del GTMC. Propongo di trattarlo come asse di sensibilità con struttura ricavata da Cardiff (isoterma), **separato** dal riferimento. Va deciso se accettarlo così.
7. **GLU:** sottoinsieme di celle e passo dei token (ogni nodo o un nodo su due).
8. **Statistica:** 5 seed U-Net e 3 GLU; selezione sulla VAL **dello stesso operatore**, riportando anche `last`; preregistrazione prima di un'eventuale apertura di TEST.
9. **E1 (olio):** decidere dopo la verifica V5 della frazione di gas freddo nella LES.
10. **Budget e calendario:** pool GPU, finestra temporale e priorità rispetto alle missioni 03 e 04.

---

# PARTE 2 — Tecnica

## A. Lettura critica del patrimonio esistente

### A.1 Originali contro specialisti

Numeri ricalcolati da `per_run_scenario.csv` @ `cb77e6e`, checkpoint best, ruolo VAL. Guadagno medio sul proprio regime (min/max sui tre seed):

- U-Net: N1 −0,85 (−1,02/−0,70); N2 −2,65; G1 −1,88; G1-eq −0,42; N1G1 +0,29 (−1,19/+3,24).
- GLU: N1 −1,26; N2 +0,10 (−2,67/+2,67); G1 −2,18; G1-eq −1,90; N1G1 −2,81 (−9,53/+1,92).
- Costo sul pulito medio: U-Net +2,15 K, GLU +3,01 K.

Coincidono con il report.

**Che cosa orienta.** Non conviene rifare specialisti da zero su un solo regime con selezione sulla VAL pulita. Il nuovo disegno cambia tre cose:

1. il modello di riferimento si riallena con la **stessa ricetta** delle celle degradate, invece di confrontarlo con originali allenati diversamente;
2. la selezione usa la **VAL dello stesso operatore**: il compito valutato è quello per cui si allena;
3. l'osservazione di training è quella che un esperimento fornirebbe, non un degrado aggiunto a un layout irrealistico.

**Che cosa non orienta.** I regimi N1/N2/G1 erano ipotesi di classe B/C su un layout al 5 %. La loro gerarchia non si trasferisce automaticamente a osservazioni dense e filtrate. In particolare, con osservazioni dense le "lacune" diventano bordi del campo di vista e cluster di rigetto, con informazione circostante molto più ricca.

### A.2 seen e VAL

seen (20 condizioni TRAIN, frame 1600–2000) misura soprattutto l'adattamento alle condizioni viste. VAL (3 condizioni) misura la generalizzazione, ma è stata usata per selezione e sviluppo. Il divario non separa sovradattamento, difficoltà diversa, effetto della selezione e deriva numerica.

**Proposta:** continuare a riportarli separati. Le conclusioni metodologiche si basano su VAL e, alla fine, su TEST preregistrato (§B.5).

### A.3 Guadagno sul degrado e costo sul pulito

Nel nuovo problema il "pulito" (velocità puntuale esatta) **non è un caso d'uso**: nessuna misura lo fornisce. Il costo sul pulito resta utile come diagnostica di generalità, ma il confronto principale è fra **informazione disponibile** sotto operatori diversi (training e test coerenti, §B.4 C2) e **robustezza al disallineamento** (training su un operatore, test su un altro, C4).

### A.4 Selezione dei checkpoint

La selezione sulla VAL pulita era coerente con il protocollo S7, ma non con il compito degli specialisti. Il report mostra che best e last cambiano il quadro (su VAL il last è peggiore del best in tutte le 10 celle; sul seen il last del GLU ha guadagno positivo in tutti i regimi).

**Proposta:** selezione sulla VAL dell'operatore di training (repliche di maschera e di rumore fissate); riportare anche il checkpoint last e la curva VAL come diagnostica; nessuna selezione post hoc.

### A.5 Variabilità fra seed

La dispersione dei guadagni su VAL va da qualche decimo a ~10 K (GLU N1G1). La baseline originale varia di ~4 K fra seed U-Net. Con 3 seed, una differenza fra medie ha un errore standard di circa 1–1,5 K se la deviazione standard fra seed è 1,5–2 K (I). Effetti sotto ~3 K diventano difficili da sostenere.

**Proposta:** 5 seed per la U-Net nel nucleo (costo contenuto); stessa inizializzazione e stesso stream di presentazioni per indice di seed fra le celle, così che il confronto per indice abbia un appaiamento parziale; valutazione appaiata a parità di frame e realizzazioni.

### A.6 Batching del GLU

Il confondente nasce dal confronto fra originali (batch 8 in un forward) e specialisti (microbatch 1 con accumulo). Con osservazioni dense il numero di token varia meno fra esempi (campo di vista fisso, rigetti variabili), ma non è costante.

**Proposta:**

- tutti i GLU della nuova campagna con la stessa ricetta, a microbatch 1 con accumulo, salvo nuova decisione dopo la profilazione;
- un **ponte di ricetta** di 3 run (GLU ideale al 5 %, stessa ricetta) per collegare la nuova campagna agli originali. È lo stesso controllo già proposto e non eseguito nel documento 04.

### A.7 Costi documentati

210,1 GPU-h misurati in totale, di cui 200,7 per i training. U-Net 67,9 GPU-h per 15 run; GLU 132,9 GPU-h per 15 run. Preemption: 30,2 GPU-h di tentativi interrotti.

Il dense U-Net del Mese 2 (U4) ha richiesto circa 28,5 min per 730 epoche nella profilazione, con valutazioni meno frequenti. Il costo per run dipende quindi molto dal piano di valutazione. I tempi di valutazione (53–82 ms per riga) non sono latenze d'inferenza.

## B. Proposta tecnica

### B.1 Alternative considerate

| | Q1 — Limiti della PIV e inferibilità di T (**raccomandata**) | Q2 — Osservabilità definita dal tracciante | Q3 — Progetto del campo di vista |
|---|---|---|---|
| Bisogno diagnostico | Sapere che cosa una PIV planare dice sulla T istantanea e quale limite della misura conta | Con olio la velocità esiste solo nel gas incombusto, ma il bordo delle particelle è un marcatore del fronte | Scegliere dove mettere un campo di vista limitato e con quale risoluzione |
| Osservazioni → stato | Velocità 2C filtrata, campionata, rumorosa, con campo di vista e validità → T nella slice | Velocità nei reagenti + maschera delle zone senza particelle → T anche nei prodotti | Velocità in una finestra → T dentro e fuori la finestra |
| Ispirazione sperimentale | GTMC DLR, Stöhr 2012 (5 Hz pieno campo / 5 kHz 30×30 mm), FT-v letto | Flashback swirl UT Austin, olio d'oliva ~555 K (std 10 K) in quelle condizioni (PIVL:L28, FT); CPIV (AB) | GTMC 5 kHz/10 kHz; DLR 5 bar (54×22 mm, L37 FT-v) |
| Ipotesi falsificabile | H1–H4 (§B.4) | La maschera aggiunge informazione su T oltre la velocità, misurata contro controlli con maschera spostata o rimescolata | L'errore cresce con la distanza dal campo di vista con una scala propria; alcune posizioni sono sistematicamente migliori |
| Che cosa aggiunge alla 00 | Filtro, layout denso, campo di vista, errore correlato, sostituzione, legati a un apparato | Lacune dipendenti dalla fiamma, mai modellate, con controllo esplicito dell'informazione | Un asse di progetto, non un degrado |
| Novità possibile (da verificare, §B.7) | Scomposizione quantitativa dei limiti PIV nell'inferenza di T istantanea in una LES reagente | Uso del bordo del tracciante come osservazione termica implicita | Media; vicina a lavori di posizionamento ottimo dei sensori |
| Rischio | Effetti di filtro/errore forse piccoli sulla griglia 128×128 (comunque un risultato) | Base empirica di un solo esperimento; la soglia usa T nel generatore; la LES potrebbe avere poco gas sotto ~600 K | Dipende da geometria e orientamento non verificati; meno legato alla combustione |

**Raccomandazione:** Q1 come domanda principale. Q3 entra in Q1 come asse del campo di vista: se la copertura domina, diventa il messaggio centrale. Q2 è un'estensione (E1) da decidere dopo la verifica V5.

Il miglioramento metodologico candidato è il training su una famiglia di operatori (C5). Risponde a un limite identificato: gli specialisti a regime singolo non trasferiscono. È proposto **solo se** C4 mostra un costo di disallineamento rilevante. In sé è domain randomization e non è una novità per definizione.

### B.2 Che cosa rimane da chiudere prima delle campagne (step 3 e 4)

#### Step 3 — Verifiche sui dataset

| ID | Verifica | Dati | Esito utile | Limite che resta |
|---|---|---|---|---|
| V1 | **Cardiff, decoder:** confrontare la media dei 514 campi decodificati con la statistica `V.*` salvata da DynamicStudio nello stesso database; confrontare la regione NaN con la maschera `D.4xd8dily`; leggere la scala mm/px da XML o paper (Runyon et al. 2018) | Tre cartelle `.vectors` (≈0,8 GB), statistica, XML; HTTP Range | Decoder validato da un prodotto del software stesso, non da un nostro script | Il significato dei codici resta **inferito** (16 = sostituito) finché manca la documentazione. La coerenza delle medie non certifica i bit di stato |
| V2 | **Cardiff, statistiche:** per frame, frazione dei codici 1 e 16 fuori dalla regione fissa; dimensioni dei cluster (componenti connesse) in vettori e, se nota la scala, in mm; posizione rispetto ai bordi; entità delle modifiche ai valori | Come V1 | Distribuzione empirica di rigetti e sostituzioni in una PIV swirl **isoterma** con processamento documentato | 5 Hz: nessuna persistenza temporale misurabile oltre le regioni fisse. Nessuna trasferibilità dimostrata alla fiamma |
| V3 | **Neal, residui:** produrre i vettori MS con un software aperto e documentato (per esempio OpenPIV; finestre 16/32 px, overlap 50/75 %); allineare al kernel di 5 coppie dell'HDR; calcolare la distribuzione del residuo MS−HDR, la frazione di outlier e l'autocorrelazione ρ(d), ρ(2d) per componente e direzione | "Unsteady core" (272 MB) o "turbulent" (782 MB) | Verifica sperimentale (non reagente) del rumore coerente con il filtro (§B.3.4) e della discrepanza aperta su ρ(2d) al 50 % (0 nella review, 0,11 in Sciacchitano e Wieneke 2016) | Il residuo contiene errore e filtro dell'HDR, interpolazione degli outlier non marcati e ricampionamento a 1 px. Non è la verità puntuale |
| V4 | **Evidenza reagente:** nuovo tentativo su Cambridge Oxycoal (PIV ad alta velocità reagente dichiarata); decidere su TCC-III fired (Globus, azione manuale) | Accesso | Se riesce, una prima statistica di rigetto in presenza di fiamma | Apparati diversi (carbone; motore SI) |
| V0 | **Letture mirate:** Runyon et al. 2018 (scala e piano Cardiff); Neal et al. 2015; Wieneke 2015; capitolo 2 della tesi di Wieneke per la risposta nelle configurazioni DaVis 7.2/8.0 vicine a quelle GTMC | Accesso OA, IOP con challenge | Parametri del filtro meno arbitrari | — |

#### Step 4 — Verifiche sulla LES prima del contratto (solo TRAIN, CPU)

| ID | Verifica | Metodo | Perché |
|---|---|---|---|
| V5 | **Contenuto alla scala del filtro** | Spettri e funzioni di struttura di vi, vj; differenza fra campo e campo filtrato con kernel di 1,0–1,9 mm; quota della differenza nelle zone di forte gradiente | Stimare in anticipo l'ordine di grandezza dell'effetto del filtro e separare il limite di rappresentazione della LES. Include la frazione di area sotto soglie di T (600/800/1200 K) per decidere E1 |
| V6 | **Verifica cinematica delle componenti** | Spostamento apparente delle strutture scalari (per esempio frazioni molari nei getti freddi) fra frame a 0,25 ms, confrontato con (vi, vj) e vk | Verificare dai dati se i/j giacciono nel piano x/y e con quale segno; per esclusione, se k è fuori piano. Prerequisito di E2 e di ogni affermazione sull'orientamento |
| V7 | **Qualificazione del generatore** | §B.5.1 | Prima delle valutazioni comparative |

V6 usa T e specie **solo** per qualificare il generatore su TRAIN, mai come input. Ha limiti noti: diffusione e reazione spostano le strutture scalari, ed è affidabile soprattutto nei getti freddi. Se non è conclusiva, l'orientamento resta M e si chiede agli autori tramite Enrico.

### B.3 Osservazioni più fedeli: elemento per elemento

Ogni riga riporta: fonte → evidenza → verifica ancora necessaria → rappresentazione sulla LES → limite. Il **riferimento R** usa soltanto elementi del GTMC (Stöhr 2012; Oberleithner 2015 come conferma della stessa catena). Gli elementi da altri apparati sono **assi di sensibilità separati**, mai combinati in un unico "apparato realistico".

#### B.3.0 Piano e componenti

- **Fonte ed evidenza:** RealPDEBench (audit: canali 11/12/13 = Velocity i/j/k; definizione rispetto a x/y/z, segno e piano **non dichiarati**, D/M). La GTMC è stereo, 3C sul piano centrale; fuori piano circa 3× (5 Hz, α≈20°) e circa 2× (5 kHz) l'incertezza in piano (Stöhr 2012, FT-v).
- **Verifica:** V6.
- **Rappresentazione:** R usa solo le due componenti dell'input attuale (vi, vj), interpretate come componenti in piano **per ipotesi dichiarata**.
- **Limite:** una stereo-PIV dà 3C; usare 2C scarta informazione realmente misurabile (E2). La terza componente su una slice non rappresenta comunque un volume, né la media attraverso il foglio di circa 1 mm.

#### B.3.1 Campo di vista e regioni escluse

| Elemento | Fonte → evidenza | Verifica necessaria | Rappresentazione | Limite |
|---|---|---|---|---|
| Pieno campo (R-full) | Stöhr 2012 §2.3.2 (FT-v): foglio su "the entire vertical section"; camera 85×85 mm (scouting) | Corrispondenza fra slice (73,8×96,2 mm) e sezione: M | Tutta la slice osservata, salvo l'eventuale fascia di bordo del filtro (vettori il cui kernel esce dal dominio: esclusi, non estrapolati) | La geometria del nostro combustore è ignota. "Pieno campo" significa tutta la slice, non la stessa camera |
| Campo ridotto 30×30 mm (R-30) | Stöhr 2012 §2.3.3 (FT-v): "limited to a region of 30×30 mm²" | Posizione fisica nel GTMC rispetto al bruciatore: non trasferibile | Quadrato di 30×30 mm (≈52×40 celle, ≈12,7 % dell'area) in **3 posizioni fissate prima dei risultati**: per esempio una griglia regolare di centri, oppure posizioni definite dal campo medio di velocità TRAIN, mai dalle prestazioni delle reti | Posizione senza corrispondenza fisica verificata: risultati per posizione, non "la" posizione GTMC |
| Riflessi e pareti | Toronto/DLR, metà campo scartata (PIVL:L05, FT-v); vicino a parete solo media (L24, FT) | Presenza di pareti nella slice: M | Non nel riferimento | Se la slice contiene pareti, R è ottimista vicino ai bordi |

#### B.3.2 Layout e passo

- **Fonte ed evidenza:** 16×16 px con overlap 50 % e 1,3 mm di risoluzione dichiarata (Stöhr §2.3.2; Oberleithner §2.2, FT-v). Il passo di circa 0,65 mm è una **I** ricavata da finestra e overlap, non un valore stampato.
- **Verifica:** nessuna sui dati GTMC (non pubblici).
- **Rappresentazione:** un vettore per nodo LES (0,581×0,758 mm), ovvero il passo PIV approssimato dalla griglia. Evita interpolazioni fuori nodo e dà alle due famiglie esattamente gli stessi valori e posizioni: U-Net come immagine con maschera di validità, GLU come token alle coordinate dei nodi.
- **Limite:** passo anisotropo invece che quadrato. Un vettore per nodo con finestra di 1,3 mm implica un sovracampionamento: l'informazione indipendente è minore del numero di vettori (step 1, §3.3).

#### B.3.3 Risposta spaziale

- **Fonte ed evidenza:**
  - GTMC: DaVis 7.2/8.0 "multi-scale cross-correlation", 16×16 px, 1,3 mm (FT-v);
  - Wieneke, nelle configurazioni DaVis multi-pass studiate su immagini sintetiche: filtro gaussiano-cappello messicano, L_sr(R_A) ≈ 0,7 volte la finestra (32 px, 4 passate, 75 %); al 50 % di overlap L_sr(R_A) ≈ 0,93 e L_sr dal rumore ≈ 1,2 volte la finestra (PIVL:L01–L02, FT della tesi);
  - top-hat solo per il single-pass a finestra quadrata.
- **Verifica:** V0 (Wieneke per DaVis 8 al 50 %); V5 (contenuto della LES a quelle scale).
- **Rappresentazione:** convoluzione in mm del campo LES con kernel di riferimento **gaussiano con L_sr = 1,3 mm**, cioè la risoluzione dichiarata interpretata come L_sr (I). Due alternative di sensibilità dichiarate: top-hat di 1,3 mm e MHG di Wieneke. Nessuna dimensione scelta in base al danno sulle reti.
- **Limite:**
  - il kernel copre circa 2 celle;
  - il campo 128×128 ha già un filtro implicito (LES più ricampionamento) **ignoto**: l'effetto simulato del filtro PIV sulla LES non coincide con l'effetto sulla fiamma reale, che ha più contenuto a piccola scala;
  - la media attraverso lo spessore del foglio (≈1 mm) e gli errori di gradiente fuori piano (L13) non sono rappresentabili su una slice.

#### B.3.4 Errore casuale, correlazioni e bias

- **Fonte ed evidenza:**
  - ±0,1 px a priori → ±0,3–1,4 m/s secondo la potenza (5 Hz), ±0,8 m/s (Oberleithner 5 Hz), ±0,5 m/s (5 kHz) (FT-v). Non è dichiarato se siano incertezze standard o espanse. Non sono validati con un riferimento;
  - un caso a posteriori in combustione (DLR 5 bar, L37) dà intervalli con statistica non definita;
  - correlazione: nel modello lineare di Wieneke l'autocorrelazione del rumore coincide con quella del filtro (L01, FT; sintetico DaVis);
  - bias: slip ≤2 % come stima a priori (Stöhr §2.3.2); rifrazione in fiamma fino al 4 % sistematico e 6 % casuale in un caso di propano (L30, solo AB); peak locking non documentato in fiamma.
- **Verifica:** V3 (forma e correlazione sui residui di Neal); conversione al nostro flusso (sotto).
- **Rappresentazione:**
  - errore = rumore bianco filtrato con lo stesso kernel della risposta, normalizzato a deviazione standard σ: la correlazione non è un parametro libero;
  - σ di riferimento trasferito come **frazione della scala di velocità**: nel GTMC 1,4 m/s per v≈80 m/s (35 kW) dà circa 1,75 %; applicato al p99,9 di |u| della nostra LES (47 m/s) dà circa 0,8 m/s (I). Il ± è trattato come incertezza standard **per ipotesi**;
  - sensibilità: estremi del rapporto documentato, solo dove il denominatore è dichiarato;
  - bias non nel riferimento: piccoli (slip) o non quantificati in modo trasferibile (rifrazione). Esclusi con motivazione esplicita.
- **Limite:** la conversione presume che Δt sia scelto in proporzione alla velocità massima. L'errore reale in fiamma è eteroschedastico (meno particelle nei prodotti), mentre il riferimento è omoschedastico. Il caso dipendente da T appartiene a E1.

#### B.3.5 Invalidazione, sostituzione e processamento

- **Fonte ed evidenza:**
  - GTMC: interpolazione dei buchi di un solo vettore e filtri POD per le figure (Boxx 2010, L21, FT-v tramite lo scout). Nessuna frazione di rigetto pubblicata;
  - Cardiff (isoterma): codice 16 interpretato come sostituzione, 15–20 % dei vettori fuori dalla regione fissa in 4 frame su 514 (decoder non ufficiale);
  - TUM, microPIV di flashback H₂: <10 % invalidi nelle zone inseminate, fino al 20 % interpolati (L22, FT).
- **Verifica:** V1–V2. Senza V1 i numeri di Cardiff restano "frazioni di codici", non di sostituzioni.
- **Rappresentazione** (asse V, **separato da R**):
  - **V-flag:** vettori invalidati con struttura spaziale (isolati e cluster) ricavata dalle statistiche Cardiff (V2), in unità fisiche se la scala è verificata; resi come mancanti espliciti (U-Net: Mv=0; GLU: token rimosso);
  - **V-sub:** gli stessi vettori sostituiti con la regola documentata (media mobile 5×5 iterata), senza flag;
  - frazione: valore Cardiff come base, più un livello superiore dichiarato come sensibilità.
- **Limite:**
  - isoterma: in fiamma i rigetti possono essere più frequenti e legati alla fiamma (L22, L29);
  - Cardiff e GTMC sono apparati diversi, quindi R+V **non** è "la PIV GTMC" ma una combinazione dichiarata;
  - la GPOD temporale e altri riempimenti che usano altri istanti sono esclusi (singolo frame).

**Ordine delle operazioni:** campo LES in m/s → (campo di vista) → filtro in mm → campionamento ai nodi → errore correlato → invalidazione → (sostituzione) → normalizzazione con statistiche fit-only. Tutto è fissato e qualificato prima delle valutazioni comparative.

#### B.3.6 Disponibilità dei dati reagenti

| Classe | Contenuto |
|---|---|
| **Accessibile e ispezionato** | Medie e RMS MRB Darmstadt (layout, convenzioni NaN/0); file 3C di Yang et al. (natura, unità e zeri M) |
| **Solo descritto in letteratura** | GTMC 5 Hz/5 kHz/10 kHz (parametri letti); DLR 5 bar con UQ a posteriori (L37); Toronto/DLR; UT Austin flashback; TUM microPIV; cause qualitative di lacune ad alta pressione (Saini, L23) |
| **Mancante** | PIV istantanea pubblica in una fiamma swirl reagente con flag e UQ; qualunque dettaglio PIV in fiamme NH₃/CH₄; T sperimentale simultanea alla PIV per la nostra configurazione |

### B.4 Gruppi di esperimenti

Celle di operatore usate sotto (tutte 2C):

- **I-full:** velocità puntuale ideale su tutti i nodi;
- **I-30:** ideale nel campo di vista ridotto;
- **F-full:** filtro e layout del riferimento;
- **FE-full:** F + errore correlato;
- **R-full** (= FE-full) e **R-30:** riferimento completo a pieno campo e a campo ridotto (R-30 con 1,5 mm e la σ trasferita da ±0,5 m/s del sistema a 5 kHz);
- **R+V-flag**, **R+V-sub:** riferimento più validazione con flag o con sostituzione senza flag.

#### C1 — Valutazione dei checkpoint esistenti

| Campo | Contenuto |
|---|---|
| Domanda | Quanto perdono i modelli attuali su osservazioni PIV? Serve un retraining? |
| Confronto | 12 originali al 5 % (U-Net/GLU, p_on/p_off, 3 seed) e U-Net densa U4/U6 (solo dove applicabile: input denso senza maschera, quindi solo I-full ed F/FE-full) su tutte le celle |
| Fisso | Checkpoint, frame, realizzazioni dell'operatore (seed di valutazione nuovo, mai usato in training) |
| Cambia | Operatore |
| Metriche | §B.5.3 |
| Esito | Perdita grande o strana (per esempio la densità della maschera fuori distribuzione per la U-Net al 5 %) → il training coerente è necessario. Perdita piccola → H2 indebolita già qui |
| Dipendenze | Generatore qualificato (V7) |
| Nota | È una misura di **robustezza** di modelli fuori distribuzione, non dell'informazione disponibile. Il GLU accetta qualunque insieme di token, ma il passaggio da 819 a migliaia di token è anch'esso uno spostamento |

#### C2 — Scala di misura con training coerente (U-Net)

| Campo | Contenuto |
|---|---|
| Domanda | Quale modulo della misura costa e quanto, quando la rete è allenata e selezionata sullo stesso operatore? (H1, H3) |
| Celle | I-full, F-full, R-full, R-30, I-30, R+V-flag, R+V-sub: 7 celle × 5 seed = **35 run**. Si aggiunge FE-full se V5 indica che filtro ed errore vanno separati (+5) |
| Fisso | Architettura, ricetta (stessa per tutte), split, frame, inizializzazione e stream per indice di seed, normalizzazione fit-only, seed delle realizzazioni (namespace TRAIN, SELEZIONE e VALUTAZIONE distinti come nella 00) |
| Cambia | Solo l'operatore, in training e in valutazione |
| Controlli | I-full replica U4 con la nuova ricetta (ancoraggio al Mese 2). I-30 contro R-30 separa la copertura dalla qualità a campo ridotto. V-flag contro V-sub isola il valore del flag a parità di vettori. ParamMean e GappyPOD (C2-lin, CPU) sulle stesse celle, come baseline lineare indipendente dall'architettura |
| Osservazioni condivise | Identiche fra seed e fra famiglie per (condizione, frame, replica) |
| Esito a favore di H1 | R-full − I-full piccolo (entro la dispersione fra seed) e R-30 − R-full grande. Contro: l'opposto |
| Esito per H3 | V-sub − V-flag > 0 oltre la dispersione, concentrato nelle regioni sostituite |
| Dipendenze | Contratto approvato (4A), V1–V3 per l'asse V, V5, profilazione dei tempi |

#### C3 — GLU sul sottoinsieme decisivo e ponte di ricetta

| Campo | Contenuto |
|---|---|
| Domanda | Le conclusioni di C2 dipendono dall'architettura? |
| Celle | I-full, R-full, R-30 × 3 seed = 9 run; ponte: GLU ideale al 5 % con la nuova ricetta × 3 seed |
| Fisso | Come C2; stessa ricetta GLU per tutte le celle |
| Scelta da discutere | Token a ogni nodo (fino a 16 384 a pieno campo) oppure un nodo su due (≈4 096), dichiarato come layout. R-30 ha circa 2 100 token ed è economico |
| Esito | Stesso ordinamento dei moduli → conclusione più generale. Ordinamento diverso → la conclusione è specifica dell'architettura e va riportata così |
| Dipendenze | Profilazione su GPU (≈2 GPU-h) prima di fissare il passo dei token |

#### C4 — Disallineamento e generalizzazione (solo valutazioni)

| Campo | Contenuto |
|---|---|
| Domanda | H2 e H4: costo di un operatore sbagliato e generalizzazione a configurazioni e condizioni non viste |
| Matrice | Modelli di C2/C3 × operatori di test: tutte le celle di C2, più configurazioni **documentate ma non usate** in training: finestra di 1,9 mm e passo di 0,95 mm (DLR 5 bar, L37), altre posizioni del campo di 30×30 mm, σ agli estremi documentati, maschere di rigetto prese direttamente dai frame Cardiff dopo V2 |
| Esito | Costo diagonale contro fuori diagonale; I-full testato su R-full contro R-full su R-full misura il valore del training coerente (H2) |
| Dipendenze | C2, C3 |

#### C5 — Training su una famiglia di operatori (condizionato)

| Campo | Contenuto |
|---|---|
| Domanda | Un training che campiona l'operatore entro gli intervalli documentati riduce il costo fuori diagonale senza perdere sulla diagonale? |
| Quando | Solo se C4 mostra un costo di disallineamento oltre la dispersione fra seed |
| Celle | 1 distribuzione × 5 seed U-Net (+3 GLU); test su configurazioni tenute fuori dalla distribuzione |
| Esito | Riduzione del costo fuori diagonale → raccomandazione metodologica motivata da un limite misurato |

#### E1 — Tracciante che evapora (estensione, Q2)

| Campo | Contenuto |
|---|---|
| Domanda | Con olio la velocità esiste solo nel gas freddo: quanta informazione su T porta il confine delle particelle? |
| Celle | Maschera T < T_s, con T_s come sensibilità attorno ai ~555 K riportati (UT Austin, in quelle condizioni); controllo con la stessa maschera presa da un altro frame della stessa condizione (informazione di forma senza allineamento); controllo con maschera indipendente da T a parità di area. ≈15 run U-Net |
| Esito | Errore con maschera allineata ≪ errore con maschera spostata → il confine è un'osservazione termica implicita (ed è anche ciò che fa la CPIV) |
| Dipendenze | V5: se la frazione di gas sotto 600 K è trascurabile, E1 perde senso su questa LES |
| Limite | Usa T nel generatore; il bordo reale non è un'isoterma (evaporazione finita, trasporto, soglia d'immagine) |

#### E2 — Terza componente (estensione)

Celle R-full con vk e σ_k = 3 σ (Stöhr §2.3.2). Solo dopo che V6, o gli autori, confermano che k è la componente fuori piano.

#### Attività che non aggiungono evidenza utile ora

- Ripetere la matrice dei 30 specialisti con nuovi livelli i.i.d.
- Scegliere σ o dimensioni delle finestre in base al danno sulle reti.
- Nuove architetture o più augmentation prima che C2 identifichi un limite.
- Generare immagini di particelle dalla slice (non c'è un campo 3D né traiettorie; P11).
- Elaborare le immagini Cardiff con molti algoritmi diversi.
- DMF-Gen o fusione termica come requisiti: sono missioni separate. Lo stesso contratto di osservazione potrà servire a entrambe.

### B.5 Protocollo e validazione

#### B.5.1 Quattro livelli da tenere distinti

1. **Qualifica del generatore.** Verificare che le osservazioni prodotte abbiano le proprietà dichiarate:
   - funzione di trasferimento del filtro;
   - σ e autocorrelazione del rumore rispetto al kernel;
   - frazioni, cluster e posizioni dei rigetti rispetto a V2;
   - conteggi, unità e ordine delle operazioni;
   - equivalenza delle osservazioni fra U-Net e GLU (hash).

   Qualifica **solo** le proprietà confrontate.
2. **Accuratezza della ricostruzione** rispetto alla T LES: validazione **numerica**.
3. **Evidenza sperimentale sulla misura:** Cardiff (stadi di processamento), Neal (residui), letteratura GTMC. Qualifica parti dell'operatore, non la T.
4. **Validazione sperimentale dello stato nascosto:** **non disponibile.** Nessuna risorsa trovata fornisce T simultanea alla PIV nella nostra configurazione. Nessun risultato di questa missione va chiamato validazione sperimentale della T.

#### B.5.2 Sviluppo e conferma

- **Sviluppo:** TRAIN (20 condizioni), seen (blocco temporale successivo) e VAL (3 condizioni), già usata per selezione e sviluppo. Selezione sulla VAL dell'operatore. Anche la scelta delle posizioni del campo di 30×30 mm va fissata prima di vedere le metriche.
- **Conferma:** prima di qualunque apertura di TEST (3 interpolazione + 4 estrapolazione), preregistrare in un documento versionato:
  - ipotesi;
  - celle e confronti;
  - metriche primarie (RMSE macro; RMSE nei getti; IoU della zona calda);
  - unità di replica e analisi.

  TEST si apre una sola volta, con decisione esplicita di Enrico. Fino ad allora le conclusioni sono "di sviluppo".
- **Repliche:**
  - per affermazioni sul **metodo** (training coerente, moduli), l'unità è il **seed di training**;
  - per la **generalizzazione**, l'unità è la **condizione**: solo 3 in VAL e 7 in TEST, quindi bootstrap gerarchico sulle condizioni con intervalli larghi dichiarati;
  - frame della stessa condizione (correlati, Δt di 12,5 ms fra frame VAL) e repliche di maschera o rumore **non** sono campioni indipendenti: servono a stabilizzare le medie, non a ridurre l'errore standard.

#### B.5.3 Metriche

- **Globali:** RMSE macro (frame → condizione → media uniforme), bias, RMSE centrata, MAE.
- **Combustione:**
  - RMSE nella zona calda (T ≥ 2100 K) e nei getti (T < 1800 K);
  - IoU della zona calda;
  - errore sul picco e sul p95;
  - **distanza media fra isoterme vere e predette** (per esempio 1500 K), da fissare prima;
  - errore condizionato a |∇T|, cioè alla regione del fronte.
- **Specifiche della misura:**
  - errore dentro e fuori dal campo di vista;
  - **errore in funzione della distanza dal vettore valido più vicino**, la quantità utile al progetto sperimentale;
  - errore nelle regioni invalidate o sostituite.
- **Per condizione:** sempre, con attenzione alla frazione di NH₃.

#### B.5.4 Incertezza (opzionale)

Monte Carlo sulle realizzazioni dell'operatore (rumore e rigetti) a modello fisso. Quantifica la **sensibilità della T predetta alla realizzazione della misura** (step 1, §7.2, punto 2): né l'incertezza completa dello stato nascosto, né la dispersione fra seed.

La verifica è possibile perché la verità è nota: si confronta la dispersione Monte Carlo con la variazione osservata della predizione fra realizzazioni indipendenti e si misura quanta parte dell'errore totale ne è spiegata. L'incertezza posteriore di T resta compito della missione 03.

### B.6 Stima delle risorse

Ipotesi dichiarate:

- U-Net 1–5 GPU-h per run. Estremo basso dalla profilazione densa U4 (28,5 min per 730 epoche), estremo alto dagli specialisti (4,5 GPU-h, con valutazioni ogni 10 epoche e preemption).
- GLU 8,9 GPU-h per run con 819 token. Con osservazioni dense da ×1 a ×10, secondo token, ricerca dei vicini e memoria: **incertezza dominante, da profilare**.
- Valutazioni scalate dal Blocco II (1,56 GPU-h per 12 modelli × 6 celle × 3 repliche), con un fattore fino a ×5 per input densi nel GLU.

| Blocco | Run | GPU-h (intervallo) | Necessità |
|---|---|---|---|
| V (V1–V7), C2-lin | 0 | CPU; download ≈1,1 GB | necessario |
| Profilazione GPU | smoke | ≈2–4 | necessario |
| C1 | 0 | 3–10 | necessario |
| C2 | 35 (+5) | 35–200 | necessario |
| C3 + ponte | 9 + 3 | 9×(9–90) + 27 ≈ 110–840 | necessario in forma ridotta |
| C4 | 0 | 5–20 | necessario |
| C5 | 5 + 3 | 5–25 + 27–270 | condizionato |
| E1 | 15 | 15–75 | estensione |
| E2 | 5–10 | 5–50 | estensione |
| **Nucleo (V, C1–C4)** | ≈50 | **≈155–1070**; con GLU a token ridotti o solo su R-30 tipicamente 150–300 | — |

Per confronto, la missione 00 è costata 210 GPU-h in circa 53 h di calendario con fino a 8 GPU. Il nucleo con U-Net completa e GLU ridotto è dello stesso ordine. Il GLU denso a pieno campo può moltiplicare il costo: per questo propongo di decidere il passo dei token dopo la profilazione, senza escluderlo in anticipo.

### B.7 Fonti e novità

**Verificate da me in questa attività:**

- Stöhr et al. 2012, Combust. Flame 159, manoscritto DLR elib 77484, §§2.3.2–2.3.3 (FT-v, testo estratto localmente): parametri 5 Hz e 5 kHz citati sopra.
- Oberleithner et al. 2015, Combust. Flame 162, manoscritto elib 96808, §§2.2 e 2.5 (FT-v): 16×16 px, 1,3 mm, 50 %, ±0,8 m/s, fuori piano ×3 (5 Hz); 10 kHz con 1,0×1,0 mm e campo ridotto.

Il resto proviene dagli scouting, con il loro livello di lettura conservato. Non ho riletto Wieneke, Neal e i report OSTI.

**Lavori più vicini (ricerca mirata del 27/11, non esaustiva):**

| Lavoro | Livello | Che cosa fa | Differenza da Q1 |
|---|---|---|---|
| Barwey, Hassanaly, Raman, Steinberg, arXiv:1909.13669 (Combust. Sci. Technol. secondo la ricerca, non verificato) | AB | Da OH-PLIF a PIV 3C con CNN in un combustore swirl sperimentale | Direzione opposta; nessuna T; limiti di misura impliciti, non controllati |
| Barwey, Raman, Steinberg, arXiv:2003.03662 | AB | Sovrapposizione d'informazione fra OH-PLIF e PIV simultanee; serve un intorno di circa due scale integrali | Domanda sull'informazione vicina alla nostra, ma su marcatori di fiamma, non T quantitativa |
| Volk et al., arXiv:2503.23801 (DOI collegato 10.1016/j.icheatmasstransfer.2025.109284, rivista inferita) | AB + parte del FT | PINN: T dal piano "PIV" in convezione di Rayleigh–Bénard da DNS; 3C sul piano senza filtro di finestra nel testo letto; test con rumore e omissione dei dati di bordo (da sintesi) | Non reagente; operatore PIV senza risposta spaziale |
| Liu et al. 2025, Combust. Flame 281, 114454 (scheda nella missione 01) | FT secondo la scheda della 01 | T e velocità 3D da piani 2D sintetici | Filtro di finestra, rumore, outlier e vettori mancanti non valutati (scheda della 01) |
| Nie et al. 2024, Combust. Flame 259, 113182 | Scheda della 01 | T da OH e soot, sperimentale | Altra modalità |

**Valutazione della novità (I, da confermare).** Nelle ricerche svolte non ho trovato un lavoro che misuri, in una fiamma turbolenta reagente, come i limiti di una PIV planare (con parametri tracciabili a un apparato) influenzino l'inferenza della T istantanea, separando copertura, risposta, errore e validazione, e controllando training coerente e disallineamento. Le ricerche sono state due ricerche web e la lettura di tre abstract. Prima di scrivere un manoscritto serve una ricerca bibliografica mirata, con risultati citanti di Barwey 2020, Liu 2025, Volk 2025 e i lavori di super-risoluzione e gap filling PIV con DNS.

Una pubblicazione su Combustion and Flame **non è garantita**. Il punto debole prevedibile per i revisori è l'assenza di validazione sperimentale di T; il punto di forza deve essere l'interpretazione diagnostica. Liu et al. 2025 mostra che studi interamente sintetici sono stati pubblicati in quella rivista, ma non dimostra che il nostro lo sarebbe.

### B.8 Limiti principali

- La T è numerica (LES), con filtro e ricampionamento ignoti. Sull'orientamento della slice ci sono solo ipotesi. Nessuna validazione sperimentale della T.
- L'operatore approssima una misura istantanea su un solo snapshot. Non sono rappresentabili: spessore del foglio, gradienti fuori piano, moto delle particelle fra impulsi, errori specifici della stereo, rifrazione e luminosità di fiamma.
- I parametri GTMC provengono da un combustore CH₄/aria diverso. Il modulo di validazione viene da una PIV isoterma. La trasferibilità è dichiarata, non dimostrata.
- Con 3 condizioni VAL e 7 TEST la generalizzazione fra condizioni ha intervalli inevitabilmente larghi.
- Le stime di costo per il GLU denso sono incerte fino alla profilazione.

### B.9 Collegamento con la roadmap

Questa proposta copre i contenuti degli step 3 (V1–V4), 4A (contratto: §B.3 e ordine delle operazioni), 4B (V5–V7) e 5 (C1–C5, E1–E2) della [roadmap 02](roadmap02.md). Non sostituisce le discussioni previste dalla roadmap.

Dopo la discussione con Enrico si potranno preparare la roadmap esecutiva e la matrice dettagliata, da affidare all'agente nell'IDE. Questo documento non modifica la roadmap, i report esistenti, la Matrix o il diario.
