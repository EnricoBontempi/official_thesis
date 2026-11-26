# Missione 02 — Step 2, agente A: inventario dei dataset PIV

**Data:** 26 novembre 2026 (ricerche e accessi eseguiti fra le 00:50 e le 02:00 UTC circa).
**Branch:** `piv/step2-dataset-scouting`, base `5aec249`.
**Perimetro:** archivi, documentazione e piccoli campioni di dati PIV. Nessun operatore virtuale, parametro definitivo, training, inferenza o accesso a TEST. Nessun dataset, PDF di terzi o file scaricato è stato committato: i campioni sono rimasti nello scratchpad della sessione.
**Letture iniziali:** `roadmap02.md`, `step1_sciacchitano.md` (in particolare §§8, 10, 11), `README.md` della missione, `docs/nonideal/00_ricognizione_osservazioni.md`, `mese3/01_observation_state_matrix/{README,templates_abc}.md`. Nella repository non ci sono `AGENTS.md`. La missione 01 al momento contiene solo i template, senza un inventario compilato: questo rapporto non lo duplica, ma ne adotta la scheda B.

**Identificativi.** Uso `B-PIV-S2D-nn` (B = dataset/configurazione secondo il template comune; S2D = step 2, scout dei dataset). L'etichetta è provvisoria: il coordinatore può riassegnarla mantenendo il rinvio. Le fonti locali sono `S01`, `S02`, … (§7).

**Etichette di evidenza** (dalla scheda dello step 1): **D** documentato (si indica sempre se da metadato, README, paper letto o file ispezionato); **I** inferenza con premesse esplicite; **M** mancante o non verificato.

**Livelli di accesso**, riportati separatamente per ogni dataset:
- **A0** citato in un paper o in un sito, ma dati non trovati o non raggiungibili;
- **A1** metadati del repository letti (titolo, licenza, elenco file);
- **A2** documentazione del dataset letta (README, PDF descrittivo, XML di processamento);
- **A3** campione di dati aperto e controllato (struttura, coordinate, valori mancanti, flag).

---

## 1. Sintesi

**Che cosa abbiamo trovato.** Ho registrato 34 candidati e ne ho aperto un campione per 7 (A3: S2D-01, 02, 03, 04, 09, 10, 11). Il risultato principale è negativo, ma utile: **non ho trovato un archivio pubblico con vettori PIV istantanei in una fiamma swirl reagente, rilasciato con flag di validità/sostituzione documentati e mappe di incertezza.** Le misure reagenti più vicine al nostro caso sono:
- non accessibili: GTMC del DLR, combustori ad alta pressione, dataset usati per gap-filling e ML;
- accessibili solo come dati di figura o medie: bruciatori di Darmstadt;
- accessibili come dati grezzi, ma isotermi: bruciatore swirl di Cardiff;
- in un apparato molto diverso: motore ottico TCC-III alimentato a propano, il cui archivio Deep Blue è protetto da un controllo anti-bot che non ho potuto superare.

**Tre archivi permettono però verifiche che finora non potevamo fare.**

1. **Cardiff, bruciatore swirl generico (B-PIV-S2D-01).** È un database Dantec DynamicStudio con 514 coppie di immagini grezze, la maschera, gli XML con tutti i parametri e i vettori a **tre livelli di processamento**: PIV adattiva, validazione per intervallo e validazione a media mobile con sostituzione. Ogni vettore ha un **codice di stato**. Nel frame ispezionato, il campo finale non contiene buchi né NaN, ma il **28,9%** dei suoi vettori ha stato "sostituito". Fra questi ci sono anche gli 800 vettori che nel primo livello erano NaN nella regione mascherata. È la prova concreta che *un campo completo può contenere vettori sostituiti*. Il campo è isotermo: utile per **lacune, validazione e sostituzione**, non per gli effetti della fiamma.
2. **Database PIV-UQ di Neal et al. 2015 (B-PIV-S2D-02, 4TU, CC0).** Contiene immagini grezze del sistema di misura e un campo di riferimento "HDR" più accurato, già mappato sulle coordinate del sistema di misura, con un flag. Serve per **errore, incertezza e correlazioni** rispetto a un riferimento simultaneo e co-localizzato, in un getto non reagente. Limite documentato: nel riferimento gli outlier sono stati **sostituiti per interpolazione senza un flag distinto**.
3. **International PIV Challenges (B-PIV-S2D-05).** Immagini sperimentali e sintetiche, la convenzione comune dei flag (0 non valido, 1 valido, 2 interpolato), un caso con riflessi di parete, sfondo e maschera, e casi di risoluzione spaziale. La verità di riferimento dei casi sintetici è disponibile **solo su richiesta email**: non la useremo senza una decisione esplicita, perché l'incarico esclude contatti con gli autori.

**Altri risultati concreti dall'ispezione dei file:**
- **Le convenzioni dei mancanti non sono uniformi, nemmeno nello stesso dataset.** Nelle medie PIV del bruciatore multi-regime di Darmstadt, i file CH4 codificano la regione esterna con NaN in entrambe le componenti. I file H2 usano invece **vx = 0 e vy = NaN nelle stesse celle** (18 file su 18 coerenti con questa regola). Altri archivi documentano lo zero come maschera (cilindro di Shang e Tu), oppure usano NaN con descrizione esplicita (motore H2 di Darmstadt), o flag 0 con valori 0 (Neal).
- **RealPDEBench, sistemi fluidi (B-PIV-S2D-04).** La PIV reale della stessa benchmark che usiamo non ha flag né maschere nello schema. Nei 14 frame ispezionati non ci sono NaN né zeri. Il dato su Hugging Face è ricampionato a 64×128 e il passo temporale nei file è **5 ms**, mentre la documentazione dichiara 400 Hz e Δt = 2,5 ms. Non posso dire se ci siano stati riempimenti: non c'è un flag che lo indichi.
- **Configurazioni reagenti documentate ma senza dati pubblici.** GTMC del DLR (Saini e Arndt 2016): finestra circa 1,0 mm, passo circa 0,5 mm, foglio con FWHM circa 1 mm, 10 kHz. Combustore a 5 bar (arXiv:2011.14662): finestra 24×24 px con overlap del 50%, risoluzione 1,9 mm, passo 0,95 mm e UQ per correlation statistics di 0,1–1,1 m/s per le componenti nel piano. Sono precedenti metodologici, non dati verificabili.

**Che cosa cambia per lo step 3.** Possiamo misurare per davvero frazioni, forme e persistenza di rigetti e sostituzioni (Cardiff), ed errori rispetto a un riferimento (Neal). In combustione restano invece verificabili solo layout e campo di vista, con poche condizioni e statistiche medie. La **frequenza delle lacune in una fiamma swirl reagente resta non verificata** con i dati pubblici trovati. Le eventuali verifiche reagenti richiedono un accesso più oneroso (TCC-III fired via Globus; Cambridge Oxycoal quando il repository tornerà raggiungibile), da autorizzare nella chat di coordinamento.

---

## 2. Tabella comparativa

Legenda delle colonne: *Exp/Sint* = esperimento o immagini sintetiche; *Livello* = che cosa contengono i file; *Accesso* = dichiarato / ottenuto da me (A0–A3).

| ID | Dataset | Flusso | Exp/Sint | Misura | Livello dati | Flag / UQ | Licenza | Accesso dich. → ottenuto | Utilità principale per lo step 3 |
|---|---|---|---|---|---|---|---|---|---|
| S2D-01 | Cardiff generic swirl burner, CH4 55 kW (Runyon et al.) | Swirl, **PIV isoterma**; fiamma solo con OH-PLIF/OH* | Exp | 2D2C (I: singola camera) | Immagini grezze + vettori su 3 livelli + statistiche | Stato per vettore (codici); nessuna UQ | CC BY 4.0 | Aperto, 7,3 GB → **A3** | **Lacune, validazione, sostituzione** |
| S2D-02 | PIV-UQ database (Neal, Sciacchitano, Smith) | Getto rettangolare, aria, non reagente | Exp | Misura 2D2C + riferimento stereo HDR | Immagini grezze MS + riferimento vettoriale con flag | Flag 0/1; riferimento per calcolare l'errore | CC0 | Aperto, 0,27–1,39 GB per regione → **A3** | **Errore, UQ, correlazioni** |
| S2D-03 | Darmstadt multi-regime burner (Li, Doğrudil, Dreizler 2024) | Fiamme H2 e CH4 multi-regime | Exp | PIV planare (M: componenti/tipo dal paper) | **Solo medie e RMS** su 3 FOV × 3 posizioni | NaN/0 incoerenti fra file; nessuna UQ | CC BY-NC 4.0 | Aperto, 8 MB per zip → **A3** | Layout/FOV in fiamma; convenzioni dei mancanti |
| S2D-04 | RealPDEBench, sistemi fluidi reali (cylinder…) | Scie in canale ad acqua | Exp | PIV time-resolved 2C | Vettori finali, ricampionati 64×128 | Nessun flag/maschera nello schema | CC BY-NC 4.0 (card HF) | Aperto, ≈36 GB solo cylinder real → **A3** (byte range) | Coerenza con la nostra benchmark; layout; tempi |
| S2D-05 | International PIV Challenges 2001–2014 | Vari (vortice, getti, canale, periodic hill, micro, stereo) | Exp + Sint | 2D, TR, stereo, tomo | Immagini; istruzioni con griglie e flag; verità di riferimento sintetica su richiesta | Convenzione 0/1/2; maschera e sfondo nel caso 2001-C | Non dichiarata sul sito | Aperto (immagini) → **A2** | Fenomeni a livello immagine; risoluzione (sintetici) |
| S2D-06 | TCC-III "Fired Full View" (Schiffmann, Sick, Reuss 2018) | Motore ottico SI, **propano/aria stechiometrico, acceso** | Exp | 2D2C high-speed | Vettori (dichiarati) + pressione | M | M (non letta) | Dichiarato aperto via Globus, ≈19,6 GB → **A0/A1 da fonte secondaria** (Deep Blue bloccato) | Unico candidato reagente istantaneo trovato; da sbloccare |
| S2D-07 | EngineBench (Oxford TPSRG), da TCC-III motored | Motore ottico **trascinato** | Exp | 2D2C | Vettori HDF5, zero = mancante, padding a 128×128 | Nessun flag; errore medio 1,5–8% dichiarato | Incoerente: CC BY 4.0 / CC BY-NC-SA 4.0 | Kaggle con account → **A2** (codice, paper, audit precedente) | Precedente per lacune artificiali; già auditato |
| S2D-08 | Cambridge Oxycoal Swirl Burner (Balusamy, Gao, Kamal, Lowe) | Swirl, polverino di carbone + CH4, reagente e non | Exp | PIV, high-speed PIV, LDA, Mie, OH-PLIF | Dichiarati file di testo di medie e high-speed PIV | M | CC BY (da snippet) | Dichiarato aperto → **A0** (repository 503 durante la sessione) | Potenziale caso reagente con traccianti diversi |
| S2D-09 | Darmstadt pressurized side-wall quenching burner (Johe et al. 2022) | Fiamma–parete premiscelata, 1 e 3 bar | Exp | HS-PIV/OH-PLIF | Solo dati di figura (.mat) | — | CC BY-NC 4.0 | Aperto → **A3** | Solo esempio di formati di figura; insufficiente |
| S2D-10 | Darmstadt H2 SI engine, cycle-to-cycle (Ye et al. 2026) | Motore ottico a idrogeno, acceso | Exp | PIV + Mie | Dati di figura: medie condizionate, centri vortice per ciclo | NaN documentato "outside the valid region"; IC95 su TKE | CC BY 4.0 | Aperto → **A3** | Convenzione dei mancanti in un motore acceso; insufficiente per lacune istantanee |
| S2D-11 | Swirling flame, translated dual-plane stereo-PIV 20 kHz (Yang et al., Opt. Lett. 2020) | Fiamma swirl | Exp | Stereo 3C su 26 piani traslati | Un file Tecplot 3D3C + vorticità | Zeri esatti a blocchi, non documentati | CC BY 4.0 | Aperto, 6,8 MB → **A3** | Solo geometria/campo medio (M: istantaneo o medio) |
| S2D-12 | Shang & Tu, PIV of flow past a cylinder | Cilindro, canale ad acqua, Re 413 | Exp | 2D2C, DaVis 8.1.2, 20 Hz | Vettori (.mat 1,08 GB) | **Zero = regione mascherata (documentato)** | CC BY 4.0 | Aperto → **A1/A2** | Esempio di maschera documentata; non scaricato |
| S2D-13 | DLR GTMC, S-PIV 10 kHz (Saini e Arndt 2016) | Swirl, gas turbine model combustor | Exp | Stereo 3C | Dati non pubblici | — | — | **A0** (parametri letti nel paper) | Precedente di configurazione; non verificabile |
| S2D-14 | Combustore swirl a 5 bar, sPIV/OH-PLIF (arXiv:2011.14662) | Swirl, 5 bar | Exp | Stereo 3C, 10 kHz | Dati non pubblici | UQ per correlation statistics dichiarata | — | **A0** | Precedente di UQ in combustione |
| S2D-15 | Swirl combustor SPIV ad alta resa vettoriale (CVAE, arXiv:2312.06461) | Swirl | Exp | Stereo | Dati non pubblici | Lacune artificiali G = 50/75% | — | **A0** | Precedente metodologico di lacune |
| S2D-16 | Swirl combustor OH-PLIF + 3C PIV usato in ML (arXiv:1904.00546, 1909.13669, 2003.03662, 2607.13193) | Swirl, regimi attached/detached | Exp | Stereo 3C time-resolved | Dati non pubblici trovati | M | — | **A0** | Solo precedente; nessun link dati |
| S2D-17 | RAFT-PIV dataset (Lagemann et al.) | Sintetico, "Problem class 2" | Sint | — | Coppie di immagini + spostamento esatto (TFRecord) | Verità esatta | CC BY 4.0 | Aperto, 12,2 GB → **A1** | Rifiuto da sintetici; eventuale risposta del processamento |
| S2D-18 | SPID / RSPID (Machado, Rocha) | Sintetico, 6 flussi analitici | Sint | — | Immagini + flusso ottico esatto | Verità esatta; parametri di rumore e fuori piano | CC BY 4.0 | Aperto, 2,1 / 16,7 GB → **A1** | Come sopra |
| S2D-19 | Full-domain POD from PIV asynchronous patches (Tirelli et al.) | Getto e cubo | Exp | PIV a patch | Vettori | M | CC BY 4.0 | Aperto, 12,7 GB → **A1** | Bassa priorità |
| S2D-20 | Darmstadt optical engine, motored (Schmidt et al.) e flow bench | Motore trascinato / banco di flusso | Exp | PIV | Vettori .mat / medie h5 | M | CC BY-NC 4.0 | **Accesso ristretto** (401) → **A1** | Escluso per accesso |
| S2D-21 | Cardiff CH4–H2 pressurized swirl burner | Swirl, fino a 0,33 MPa | Exp | OH-PLIF, OH*, pressione; **PIV non elencata** | — | — | CC BY 4.0 | Elenco file vuoto via API → **A1** | Escluso: niente PIV |
| S2D-22 | KIT fuel-flexible combustors (3 record Zenodo, Singh et al.) | Swirl/low-swirl, Jet-A1/H2 | Exp + CFD | **Nessuna PIV**: OH*, LBO, emissioni, CFD | — | — | CC BY 4.0 | **A1** | Escluso |
| S2D-23 | Multinozzle LPP combustor (Ruan et al., Energy Fuels 2021, SI) | Combustore multi-iniettore | Exp | HS-PIV | **Solo video** | — | CC BY-NC 4.0 | **A1** (listing zip) | Escluso |
| S2D-24 | Sydney swirl / TNF swirl | Swirl | Exp | LDV, Raman/Rayleigh | Statistiche | — | — | Già trattato in `00_ricognizione` §4 | Escluso: niente PIV |
| S2D-25 | Cambridge stratified swirl burner (LDA) e Cambridge–Sandia PIV | Swirl stratificato CH4 | Exp | **LDA**; PIV citata dai report Sandia | Statistiche LDA | — | — | **A0** per la PIV | Escluso finché la PIV non è trovata |
| S2D-26 | Boomsma et al. 2016, UQ comparativa 2C | Getto, cilindro | Exp | Due PIV + LDV di riferimento | Dati non trovati | — | — | **A0** | Solo letteratura |
| S2D-27 | Ebi e Clemens 2016, flashback di fiamme swirl | Swirl, flashback | Exp | TR stereo-PIV | Dati non trovati (tesi UT) | — | — | **A0** | Solo letteratura |
| S2D-28 | Darmstadt metal dust flames, fluorescent tracers (Krenn et al. 2026) | Fiamme di polveri metalliche | Exp | PIV con traccianti fluorescenti | Dati di figura .h5 | M | CC BY-NC 4.0 | **A1** | Pista sul problema dei traccianti; non ispezionato |
| S2D-29 | Darmstadt multi-mode H2 flames, Raman/Rayleigh + PIV/OH-LIF (Shi et al. 2026) | H2 multi-modo | Exp | PIV/OH-LIF + Raman | Dati di figura | — | CC BY-NC 4.0, **restrictedAccess** | **A1** | Escluso per accesso |
| S2D-30 | Harvard Dataverse: wakes behind bluff bodies, DNS + PIV | Scie | Exp + DNS | PIV | M | M | M | **A1** (titolo) | Pista per confronto PIV–DNS; non ispezionato |
| S2D-31 | Johns Hopkins Turbulence Database | DNS | Sorgente per immagini sintetiche | — | Campi DNS | — | — | Non pertinente come PIV | Escluso |
| S2D-32 | Cambridge swirl-flame site (OH-PLIF, blow-off) | Swirl premiscelato | Exp | Nessuna PIV dichiarata nel risultato di ricerca | — | — | — | **A0** | Escluso |
| S2D-33 | Cardiff ammonia–coke-oven-gas swirl e NH3/H2 | Swirl NH3 | Exp | Emissioni, chemiluminescenza; PIV non citata | — | — | — | **A0/A1** (snippet) | Escluso, M per PIV |
| S2D-34 | McGill H2-enriched swirl flames (HS-PIV, open access paper) | Swirl H2 | Exp | HS-PIV | Dati non verificati | — | — | **A0** | Da controllare se serve |

---

## 3. Schede dei dataset

Le schede seguono il template B della missione 01, ridotto ai campi pertinenti. **Dove non ho trovato un'informazione scrivo "M — non verificato".** Non uso parametri di un apparato per descriverne un altro.

### B-PIV-S2D-01 — Cardiff: lean methane flame stability in a premixed generic swirl burner

**Sintesi.** È il dataset più ricco di **storia del processamento** che ho trovato. Il database DynamicStudio contiene immagini grezze, maschera, impostazioni di acquisizione e di analisi, e i vettori dopo ogni passo di validazione, con un codice di stato per vettore. La PIV è **isoterma** (dichiarato: "detailed isothermal characterization of the burner velocity field using particle image velocimetry"); la combustione è descritta da OH-PLIF, OH* e pressione dinamica, acquisiti in sessioni diverse.

| Campo | Contenuto |
|---|---|
| Identità | Runyon, Marsh, Bowen, Pugh, Giles, Morris. DOI [10.17035/d.2017.0011500240](https://doi.org/10.17035/d.2017.0011500240), figshare id 27051964, v1 (2024-09-18), **CC BY 4.0**. Paper: Exp. Therm. Fluid Sci., DOI 10.1016/j.expthermflusci.2017.11.019 (**non letto**). [S01] |
| Accesso | Dichiarato: download aperto di un unico zip da 7,30 GB (7 301 821 358 byte). Ottenuto: **listing completo dello zip via richieste HTTP Range** (7049 membri) ed estrazione di README, di tutti i 15 XML della parte PIV e di 4 file `.vectors` (frame #0 a ciascun livello). Contenuto ispezionato: vedi §4.1. **A3** |
| Apparato e condizioni | D (metadato): bruciatore swirl generico, CH4–aria premiscelato, 55 kW, pressione atmosferica, ingresso a ~290 K, comportamento verso il blowoff. Geometria, numero di swirl e piano PIV: M — nel paper. |
| Tipo di misura | D (XML): camera Dantec HiSense Mk II, 1344×1024 px, 12 bit, `cameraFrameMode = DoubleFrame`; laser Litron Nano PIV, temporizzatore BNC 575 (da `Run.xml`). I: 2D2C a camera singola (un solo dispositivo camera nel database). |
| Livello dei dati | D (listing): `PIVImages/20170303_143734/` contiene 514 immagini doppie grezze (2,83 GB); 514 immagini mascherate (`I.4xd8gr7l`, "Image Masking"); la definizione della maschera (`D.4xd8dily`, "Define Mask"); un ramo di "Image Arithmetic" (44 immagini); 514 vettori **AdaptivePIV**, 514 **ValRange** ("Range Validation") e 514 **ValMovAvg** ("Moving Average Validation"); una statistica (`V.*`). Altri due rami di analisi su sottoinsiemi: uno con 514 vettori AdaptivePIV non validati ulteriormente, uno con 3 vettori per livello. |
| Condizioni e snapshot | D: una sola acquisizione PIV (2017-03-03), 514 coppie. I: una condizione isoterma. Il paper può contenere altre condizioni: M. |
| Coordinate e griglia | D (file `.vectors`, decoder euristico, §4.1): array 81×61 di vettori con passo **16 px** (`iaStep = {X=16,Y=16}`). La conversione in unità fisiche non è documentata in chiaro: un campo `pixelToVelocityFactor` esiste nel binario; un valore candidato (6,0067) non è verificato. **M** la scala mm/px. |
| Finestra / overlap / passo | D (XML di `A.4xdm0h6c`): finestra adattiva **min 32×32, max 64×64 px**, passo 16 px, `bAdaptSize = True`, `iParticleCount = 10`, `iMaxIterations = 10`, `windowType = None`, `filterType = None`. L'overlap non è un parametro separato e **non lo ricavo** dal passo, perché la finestra è adattiva. |
| Tempi | D (`Run.xml`, gruppo *defaults*): `timeBetweenPulses = 1.5E-05` s, `triggerRate = 5` Hz. Timestamp per immagine con passo costante di 2 000 000 unità (I: coerente con 5 Hz se in tick .NET da 100 ns; unità non verificata). |
| Particelle e ottica | M — non negli XML letti. Probabilmente nel paper (non letto). |
| Formato | D: database DynamicStudio 3.41; vettori serializzati in formato binario .NET (`Dynamix.Datasets.Vectors.VectorMap`). Ogni vettore ha i campi `U, V, W` (double), `X, Y, Z` (single), `Status`, `HP1, WP1, HP2, WP2` (int), `U2, V2, Exp` (double). Lettura senza DynamicStudio: solo con un decoder ad hoc (§4.1). |
| Validazione e sostituzione | D (XML): nel passo adattivo `bValidateVectorsLastStep = True`, `bSubstituteVectorsLastStep = True`, intervallo `fMinU/fMaxU = ±16`, `fMinSNR = 4`, controllo dell'altezza del picco disattivato (`bCheckPeakHeight = False`; `fMinPkHeight = 0.25`; `bCheckPkHeightRatio = False`, `fMinPHR = 1.15`), vicinato 5×5, `fMaxSingleGrad = 0.1`, `fMaxTotalGrad = 0.2`. Validazione per intervallo: `minLength = -40`, `maxLength = 40` (unità M). Media mobile: `acceptance = 0.14`, `iterations = 3`, area 5×5, `substitute = True`, `validateOnBoundary = True`. |
| Flag | D (file): codici di stato 0, 1, 16 e 17. I (non documentato dal dataset): 0 = accettato; 1 = disabilitato o rigettato, con valori NaN; 16 = sostituito; 17 = combinazione dei due bit. L'interpretazione si basa su quali valori cambiano fra i livelli (§4.1). |
| UQ | Nessuna mappa di incertezza. `HP1/HP2` hanno l'aspetto di altezze del primo e secondo picco di correlazione (I, non verificato): un'eventuale metrica tipo rapporto fra picchi richiederebbe una verifica dedicata e **non è un'incertezza**. |
| Riferimento indipendente | Nessuno. |
| Modalità simultanee | OH-PLIF, OH* e pressione dinamica in sessioni del 2015; PIV nel 2017. **Non simultanee** con la PIV. |
| Verifiche consentite | Frazione, forma, contiguità e persistenza dei vettori rigettati e sostituiti in 514 frame; separazione fra regione mascherata, rigetto per intervallo e rigetto per media mobile; effetto della sostituzione sui valori; rielaborazione delle immagini grezze. |
| Informazioni mancanti | Scala fisica, posizione e orientamento del piano; documentazione dei codici di stato; nessuna misura reagente; una sola condizione PIV. |
| Giudizio | **Utile per lacune, validazione e sostituzione** (forte); utile per il layout di un swirl burner (debole, isotermo); **non** utile per UQ o per gli effetti della fiamma. |

### B-PIV-S2D-02 — PIV-UQ database: getto rettangolare transizionale (Neal et al. 2015)

**Sintesi.** È il benchmark sperimentale costruito apposta per valutare l'UQ della PIV, citato da Sciacchitano (2019, §4.1). Un sistema di misura (MS) e un sistema ad alta gamma dinamica (HDR) acquisiscono simultaneamente lo stesso piano. Il riferimento HDR è già mappato sulle coordinate pixel del MS, per calcolare direttamente l'errore.

| Campo | Contenuto |
|---|---|
| Identità | Neal, Smith, Sciacchitano. Tre record su 4TU.ResearchData, tutti v1, **CC0**: turbulent region [10.4121/13649885](https://doi.org/10.4121/13649885) (781,6 MB); unsteady inviscid core [10.4121/13663376](https://doi.org/10.4121/13663376) (271,8 MB); steady inviscid core [10.4121/13341842](https://doi.org/10.4121/13341842) (1 394,1 MB). Paper: Neal et al., MST 26 (2015) 074003 (**non letto in questa attività**). [S02] |
| Accesso | Ottenuto: metadati API dei tre record; listing dello zip "turbulent region" via HTTP Range; estratti `Database description.pdf` (6 pp., **letto integralmente**) e `data description.txt`; estratti in streaming i primi 3 file di riferimento HDR di H016 e le prime 2 immagini MS di H016. **A3** |
| Apparato | D (PDF §1): getto rettangolare, AR 7,2 (72,8 × 10,2 mm), u₀ = 5 m/s, Re_h ≈ 3000, laboratorio EFDL della Utah State University. Fumo di glicerina-acqua, gocce con mediana 1 µm. Laser 40 mJ a 1 kHz, **foglio di 1,7 mm** misurato. Campionamento 4–10 kHz. Non reagente. |
| Tipo di misura | D: MS = TR-PIV **2D2C** a camera singola (HighSpeedStar 5, 1024×1024 px, 10 bit, M = 0,122, **7,2 px/mm**, immagine delle particelle ≈1,5 px). HDR = stereo TR-PIV (due HighSpeedStar 6, M = 0,535, **26,7 px/mm**, immagine delle particelle 2,5–3 px), usabile anche come 2D2C. |
| Livello dei dati | D: **immagini grezze MS** (tif a 16 bit), **campo di riferimento HDR** (Tecplot ASCII) e PNG di esempio. I vettori MS **non** sono forniti: vanno calcolati dall'utente. |
| Condizioni | D: regioni a) nucleo stazionario e strato limite laminare (densità di inseminazione 0,3–3,0 particelle/mm³; f# 2,8–8); b) nucleo instazionario e roll-up; e) turbolenza sviluppata (x/h = 19). Nella regione turbolenta i set H015/H016/H017 corrispondono a immagini defocalizzate, a fuoco e leggermente defocalizzate. Numero di frame per set: M (zip annidati deflate, non listati). |
| Processamento del riferimento | D (PDF §1.3): DaVis 8.1.6; correlazione sliding-average su un kernel di 5 coppie; finestra finale 32×32 px HDR (**1,42 × 1,42 mm**); overlap 75%; spurious rilevati con universal outlier detection e **rimossi e sostituiti per interpolazione lineare**; nessuno smoothing né spaziale né temporale. |
| Coordinate e formato | D (PDF §§3–4 e file): `X, Y, Vx, Vy, Vz, flag`, **tutto in pixel del MS** (velocità in px per intervallo). Mappatura HDR→MS con dilatazione e traslazione, senza rotazione. Campione: zona 168×167, X 311–478, Y 192–358, **passo 1 px MS** (I: griglia ricampionata, perché la finestra HDR con overlap del 75% corrisponde a un passo nativo più grande; non verificato). |
| Flag | D (PDF): "1 for valid vectors, 0 for invalid vectors (e.g. outside of the measurement domain)". D (campione): 7134 flag 0 su 28 056, **tutti con velocità esattamente 0**; 450 flag 0 all'interno del rettangolo dei validi. I: i vettori HDR sostituiti dall'UOD **non sono distinguibili** con questo flag. |
| UQ | Nessuna mappa UQ fornita: il dataset serve a valutare i metodi. Errore del MS stimabile come differenza dal riferimento HDR, che ha errore relativo circa 3–4 volte inferiore (D, PDF §1.2: argomentazione basata sul rapporto delle risoluzioni). |
| Riferimento | **Simultaneo, co-localizzato** (stessa linea di vista per camera MS e camera 1 HDR, PDF §3), con incertezza propria non nulla. |
| Verifiche consentite | Errore del MS (dopo un processamento a nostra scelta e documentato) rispetto all'HDR; copertura di bande UQ; correlazione spaziale degli errori al variare di finestra e overlap; effetto di inseminazione e focus (regioni a ed e). |
| Mancanze | Non reagente; nessun flag di sostituzione nel riferimento; vettori MS da calcolare (la scelta del software diventa parte del protocollo). |
| Giudizio | **Utile per UQ, errori e correlazioni**; utile come precedente di layout; **non** per lacune in fiamma. |

### B-PIV-S2D-03 — Darmstadt multi-regime burner (MRB): medie PIV

| Campo | Contenuto |
|---|---|
| Identità | Li, Doğrudil, Dreizler. TUdatalib, item `e6d83b56-0a21-49b2-8076-c6113cc5b680`, issued 2025-05-13, **CC BY-NC 4.0**; supplemento a Proc. Combust. Inst. 2024, DOI 10.1016/j.proci.2024.105759 (**non letto**). [S03] |
| Accesso | Ottenuto: metadati; scaricato `Data_MRB_PIV_Avg.zip` (8,26 MB). `Data_MRB_PIV_RMS.zip` (8,1 MB) e `Data_MRB_OH_Avg.zip` (163 MB) non scaricati. **A3** |
| Contenuto | D (file): 18 `.mat` `PIV_avg_{CH4,H2}_FOV{1,2,3}_Phi{140,220,350}`, ciascuno con `vx`, `vy` (≈154×272) e vettori di coordinate `x2D`, `y2D`. Il significato di "Phi140/220/350" non è documentato nel dataset (M; I: forse equivalence ratio ×100, non verificato). |
| Griglia | D: passo 0,0436 in x e y, nelle unità delle coordinate; estensione circa 11,8 × 6,7 unità per FOV. **Unità non dichiarate nei file** (M). |
| Mancanti | D: nei file CH4, 6546–6787 celle NaN in entrambe le componenti; nei file H2, le **stesse celle hanno vx = 0 e vy = NaN** (verificato sui 9 file H2: zeri di vx coincidenti con NaN di vy). La regione mancante è a bande sui bordi (I: fuori dal campo valido; causa non documentata). |
| Livello | **Sole medie** (e RMS non ispezionate). Nessun istantaneo, flag, UQ o parametro di processamento nei file. |
| Giudizio | Utile come esempio di **layout e FOV in fiamme H2/CH4** e di **convenzioni dei mancanti incoerenti**; insufficiente per lacune istantanee, rumore e UQ. |

### B-PIV-S2D-04 — RealPDEBench, sistemi fluidi con PIV reale

| Campo | Contenuto |
|---|---|
| Identità | AI4Science-WestlakeU/RealPDEBench, Hugging Face, `version.json`: data_version **2.0.1** (2026-09-18), formato `lazy_slicing_v2`. Licenza CC BY-NC 4.0 (secondo l'audit precedente; non riletta in questa sessione). [S04] |
| Accesso | Ottenuti: README, `version.json`, `cylinder/channels.json`, `dataset_info.json` e i primi 2 MB più la coda (326 kB) di `cylinder/hf_dataset/real/data-00000-of-00073.arrow` (784,8 MB) via HTTP Range. **A3 parziale** |
| Contenuto dichiarato | D (README e docs): cylinder, controlled_cylinder, fsi, foil; "Real: Particle Image Velocimetry (PIV) in a circulating water tunnel"; cylinder: 92 traiettorie reali, 3990 frame, "128×256" reali (full), Δt = 2,5×10⁻³ s, 400 Hz per 20 s; D = 30 mm; Re 1800–12000. `version.json`: "Spatial subsampling applied during conversion". La v2.0.1 corregge "inconsistent unit conversions in a few trajectories" in 12 shard reali. |
| Schema | D (file): colonne `sim_id`, `u`, `v`, `vo`, `x`, `y`, `t` come blob binari, più le forme. **Nessun flag, maschera o incertezza.** u e v float32, x, y e t float64. |
| Campione | D: shard 0, riga 0 (`10031.h5`): u e v **64×128** per frame; 14 frame di u e 31 di v letti; **nessun NaN, nessuno zero**; u fra −0,03 e 0,37 (unità M; I: m/s). x da 0,0029 a 0,3246 con passo **2,533 mm**, y da 0,0061 a 0,1779 con passo **2,726 mm** (I: metri). t da 0,05 a 19,995 s con passo costante **0,005 s** (3990 valori). |
| Discrepanze | Passo temporale nei file 5 ms contro Δt = 2,5 ms e 400 Hz della documentazione; risoluzione 64×128 nei file contro "128×256 (full resolution)". Non riconciliate. |
| Processamento PIV | M — finestra, overlap, validazione, riempimento della regione del cilindro e dell'ombra: non documentati nelle fonti lette. |
| Giudizio | Utile per **layout, tempi e convenzioni** della benchmark che già usiamo e per un confronto con le simulazioni appaiate (non allineate frame a frame, secondo la documentazione). **Non** utile per lacune o UQ: il dato non dice che cosa sia stato rigettato o riempito. |

### B-PIV-S2D-05 — International PIV Challenges (2001, 2003, 2005, 2014)

| Campo | Contenuto |
|---|---|
| Identità | Sito ufficiale [pivchallenge.org](https://www.pivchallenge.org/). Risultati: Stanislas et al. 2003 (MST 14 R63), 2005 (Exp Fluids 39), 2008 (Exp Fluids 45); Kähler et al. 2016 (Exp Fluids 57:97). Licenza dei dati non dichiarata sul sito (M). [S05] |
| Accesso | Pagine 2001, 2005 e 2014 lette; istruzioni dei casi 4B, 4E e 4F scaricate e lette. Immagini non scaricate (taglia: 4F 4,9 MB; caso 3A 550 MB; le richieste HEAD sugli altri zip del 2014 sono scadute). **A2** |
| Casi rilevanti | **2001-C**: immagini reali di un impeller con forte riflesso di parete, **sfondo e maschera forniti**. **2005-A**: immagini sintetiche per la risoluzione spaziale; "has correct vector information… please contact organizer by e-mail". **2005-C**: getto reale TR a 10 000 fps, frame straddling a 10 µs, senza verità. **2014-B**: periodic hill, TR 2D a 2 kHz, 1044 frame 1280×800 a 16 bit, cw-laser da 5 W, sfere cave di 10 µm, con le difficoltà dichiarate (ampia gamma dinamica, fuori piano, piccole immagini delle particelle). **2014-E**: anello vorticoso, stereo TR, calibrazione su 7 piani. **2014-F**: una coppia 1024×1024. |
| Flag | D (istruzioni 4B e 4F): "0 Not valid, 1 Valid, 2 Interpolated"; griglie imposte (per esempio 4F: 121×121 nodi, passo 8 px, origine dichiarata); per PTV "a displacement of 0 and a Flag 0 may be specified". I risultati dei partecipanti non sono sul sito. |
| Giudizio | Utile per **fenomeni a livello d'immagine** (riflessi, maschere, fuori piano) e come **precedente di convenzione dei flag**. Utile per la risposta spaziale solo con la verità sintetica, oggi su richiesta. Non reagente. |

### B-PIV-S2D-06 — TCC-III "Fired Full View" (e Motored Full View)

| Campo | Contenuto |
|---|---|
| Identità | Schiffmann, Sick, Reuss (2018). DOI [10.7302/Z2H41PCD](https://doi.org/10.7302/Z2H41PCD) (fired); motored [10.7302/Z2MS3QP3](https://doi.org/10.7302/Z2MS3QP3). Deep Blue Data, University of Michigan. |
| Accesso | **Non riuscito**. Il DOI risolve verso Deep Blue, che risponde 403 con una pagina di challenge anti-bot ("Enable JavaScript and cookies"); il web-fetch non risolve l'host. Tutto il resto viene da **snippet di ricerca** (fonte secondaria): dataset di ≈19,6 GB in due file, trasferimento via Globus. **A0/A1 secondario** [S06] |
| Contenuto (secondario) | "two-dimensional two-component velocity fields from various measurement planes with maximized field of view, in-cylinder pressure… geometry"; "high frame rate particle image velocimetry"; acceso con **propano/aria stechiometrico, 40 kPa MAP, 1300 rpm**. Paper collegato: Schiffmann, Reuss, Sick, IJER 2018, variabilità ciclica dell'innesco (**non letto**). |
| Mancanze | Formato, flag, validazione, campo di vista, inseminazione e trattamento della regione bruciata: tutto M. |
| Giudizio | È **l'unico candidato con vettori istantanei in presenza di fiamma** e un rilascio pubblico dichiarato. Apparato molto diverso dal nostro (motore SI omogeneo, non swirl stabilizzato). Da sbloccare solo se lo step 3 deve osservare lacune reali in presenza di fiamma (§5). |

### B-PIV-S2D-07 — EngineBench

**Riuso dell'audit esistente** (`docs/nonideal/00_ricognizione_osservazioni.md` §3), senza duplicarlo. Nuovi elementi verificati oggi:
- pagina Oxford: dati TCC-III "released under the Creative Commons Attribution 4.0" e "Our license: CC BY-NC-SA 4.0" nella stessa pagina; il DOI Kaggle è 10.34740/KAGGLE/DS/5000332;
- paper arXiv:2406.03325v1 (HTML letto per parole chiave): dati **motored**, quattro piani PIV, HDF5; "average velocity error of 1.5−8%" ripreso da Schiffmann et al.;
- l'API Kaggle non restituisce metadati senza autenticazione: il download richiede un account (M per contenuto e versione).

**Giudizio:** precedente metodologico per le lacune artificiali; non è una misura reagente; nessun flag. [S07]

### B-PIV-S2D-08 — Cambridge Oxycoal Swirl Burner (CCB)

| Campo | Contenuto |
|---|---|
| Identità | "Velocity and species imaging data for Cambridge Oxycoal Swirl Burner (CCB)", Balusamy, Gao, Kamal, Lowe; Apollo, item `a2075a6c-7666-4554-96e6-7352bade861b`; DOI 10.17863/CAM.96660 (da una pagina KIT, letta solo come snippet); **CC BY** (snippet). |
| Accesso | **Non riuscito**: `www.repository.cam.ac.uk` ha risposto 503 a tutte le richieste (HTML e API), ripetute a distanza di circa 30 minuti; la negoziazione del DOI non ha restituito metadati. **A0** [S08] |
| Contenuto (secondario) | Secondo gli snippet: PIV, LDA, Mie, OH-PLIF; geometrie CCB1 (senza bluff body) e CCB2 (con); cartella PIV con file di testo delle velocità medie, casi non reagenti e reagenti inseminati con allumina, file di testo di high-speed PIV con casi a carbone; un file di formato per cartella. |
| Dal paper letto | Balusamy et al., Exp Fluids 56:108 (2015), PDF Springer (16 pp., letto per parole chiave): velocità del gas con allumina calcinata da 1 µm in tutti i flussi; **con carbone, le particelle di carbone sono lo scatterer** nel flusso portante (senza allumina). Quindi si misura la velocità di traccianti diversi. In quel paper le velocità sono per lo più LDV; i parametri della PIV (Balusamy, Schmidt, Hochgreb 2013, Exp Fluids 54:1534) non sono accessibili (paywall). |
| Giudizio | **Promettente per una verifica reagente swirl**, ma oggi **non verificabile**. Da ritentare (§5). |

### B-PIV-S2D-09 — Darmstadt pressurized side-wall quenching burner (Johe et al. 2022)

TUdatalib `b4c48efc-…`, CC BY-NC 4.0, sette `Fig*.mat`. Ne ho scaricati tre (Fig6, Fig8, Fig11):
- Fig6 e Fig8: struct `a`/`b` con `x`, `y`, `xc`, `data` su griglie di circa 110×197; valori fra 0,03 e 1,93, senza NaN; grandezza e unità **non dichiarate** nel file;
- Fig11: "FlameBrush" uint8 a 1 e 3 bar.

Sono **dati di figura**, non vettori istantanei. Paper non letto. **Giudizio:** insufficiente. **A3** [S09]

### B-PIV-S2D-10 — Darmstadt H2 spark-ignition engine (Ye et al. 2026)

TUdatalib `e156e984-…`, **CC BY 4.0**. README letto e due HDF5 aperti.

- Velocità medie condizionate (cicli HC/LC) su griglia 170×171, con attributi `units = m/s` e descrizione "**NaN outside the valid region**" (22 363 NaN per campo).
- Per ogni frame, posizioni dei centri del vortice nei cicli validi: per esempio 353 cicli su 900 a −90,4 °CA per 0EGR. Il README rinvia al paper per altri dati.

**Giudizio:** buon esempio di **metadati espliciti dei mancanti** in un motore acceso; niente campi istantanei completi nei file ispezionati. **A3** [S10]

### B-PIV-S2D-11 — Swirling flame, dual-plane stereo-PIV traslato (Yang et al., Opt. Lett. 45, 2020)

Figshare 12904478, `techplotdata.dat` (6,8 MB), CC BY 4.0, scaricato integralmente.

- Struttura: zona Tecplot 26×61×81 (128 466 punti), x da −40 a 40, y da 0 a 60, z da −26 a 24 con passo 2; variabili `u, v, w, vorx, vory, vorz`.
- Valori mancanti: **8,7%** dei punti con velocità esattamente nulla, concentrati in piani alterni (≈17–18% in z = −26, −22, −18…, 0% in z = −24, −20…).
- Non dichiarati: unità, natura istantanea o media (I: probabilmente non istantaneo, perché ottenuto da piani traslati) e significato degli zeri.

**Giudizio:** utile solo come geometria 3C di una fiamma swirl; non per lacune o UQ. **A3** [S11]

### B-PIV-S2D-12 — Shang & Tu, PIV del cilindro (Zenodo 20765567)

CC BY 4.0, un `.mat` da 1,08 GB (non scaricato). La descrizione Zenodo dichiara:
- DaVis 8.1.2, 20 Hz, u e v in m/s, x e y in m;
- "**some of the values in (u,v) are exactly zeros (0s); these regions were masked out prior to PIV processing**": sezione del cilindro e ombre.

**Giudizio:** buon esempio documentato di **zero usato come maschera**, da contrapporre ai casi in cui lo zero non è documentato. **A1/A2** [S12]

### B-PIV-S2D-13 / 14 / 15 / 16 — Configurazioni reagenti documentate ma senza dati pubblici

| ID | Fonte letta | Parametri documentati (D, dalla fonte) | Dati |
|---|---|---|---|
| S2D-13 | Saini e Arndt, AIAA 2016-0436, versione eLib DLR (13 pp., letta per parole chiave); la versione Exp Fluids 57:122 non è letta | S-PIV in un gas turbine model combustor; due CMOS LaVision HSS 8; 532 nm, 2,6 mJ/impulso, Δt fra impulsi 5–15 µs "depending on conditions", **10 kHz**; foglio alto ≈40 mm con **FWHM ≈1 mm**; TiO2; DaVis 8.0; finestra finale ≈1,0 mm, **passo ≈0,5 mm**; FOV ≈49×38 mm ridotto dagli accessi ottici; lacune **artificiali** G = 15/25/50% su un set atmosferico "non-gappy". Cause delle lacune reali elencate come motivazione: basso SNR, luminosità di fondo da soot e fiamma, inseminazione non uniforme. | Non pubblici (A0) |
| S2D-14 | arXiv:2011.14662 (HTML, sezioni diagnostiche) | 5 bar; sPIV, OH-PLIF; DaVis 8.4; finestra finale 24×24 px con overlap del 50%; **risoluzione 1,9 mm, passo 0,95 mm**; incertezza degli istantanei da correlation statistics di DaVis **0,1–1,1 m/s nel piano e 0,3–2,3 m/s fuori piano** (testo del paper; livello di copertura da verificare nel paper) | Nessuna data availability trovata (A0) |
| S2D-15 | arXiv:2312.06461 (abstract e HTML) | SPIV in un combustore swirl con "very high vector yield"; lacune **artificiali** G ≈ 50% e 75%, distribuite nel tempo con profilo gaussiano | Nessun link (A0) |
| S2D-16 | arXiv:1904.00546, 1909.13669, 2003.03662, 2607.13193 (abstract e HTML) | OH-PLIF e PIV 3C simultanei in un combustore swirl, regimi attached e detached | Nessun link dati trovato (A0) |

Questi numeri descrivono **quegli apparati**: non vanno trasferiti a RealPDEBench senza una decisione esplicita (regola dello step 1, §7.1). Il secondo scout può approfondire i paper. [S13–S16]

### B-PIV-S2D-17 / 18 — Benchmark di immagini sintetiche

- **RAFT-PIV** (Zenodo 4432496, CC BY 4.0): 12,2 GB in TFRecord, "Problem class 2"; il contenuto dettagliato è nel paper (Nature Machine Intelligence; non letto). **A1**
- **RSPID e SPID** (Zenodo 7832205 e 7935215, CC BY 4.0): immagini 665×630 a 8 bit; raggio delle particelle 1–4 px; 15–32 particelle per unità di densità dichiarata; fattore di spostamento; rumore 1–15; deviazione standard fuori piano 0,01–0,05; sei flussi analitici; verità esatta. **A1**

**Giudizio:** utili soltanto se lo step 3 deve isolare la risposta di un algoritmo con verità nota. Sciacchitano (P11) avverte che le immagini sintetiche possono sottostimare gli errori sperimentali. [S17, S18]

### B-PIV-S2D-19 … 34 — Esclusioni e piste minori

Sono riassunte nella tabella del §2 e nel registro del §6. Le esclusioni dipendono da: assenza di PIV (21, 22, 24, 32, 33), solo video (23), accesso ristretto (20, 29), dati non trovati (25, 26, 27, 34) oppure priorità bassa e dimensioni elevate (19, 28, 30). JHTDB (31) è una sorgente di campi DNS per generare immagini, non una misura PIV.

---

## 4. Campioni realmente ispezionati

Metodo comune: download in una directory temporanea della sessione, mai nella repository; lettura con Python isolato (`-I`), `numpy`, `scipy.io`, `h5py`, `pyarrow` e `pdfplumber`. Per i grandi zip ho usato richieste **HTTP Range**: si legge la directory centrale dello zip e si estraggono soltanto i membri scelti. Per i membri compressi annidati, ho decompresso in streaming soltanto i primi MB. SHA-256 (primi 16 caratteri esadecimali) dei file aperti, per rintracciarli.

### 4.1 Cardiff (S2D-01)

- **Byte trasferiti:** directory centrale (≈1,3 MB); README (2,6 kB); 15 XML; 4 `.vectors` da circa 0,5 MB. Totale circa 3,3 MB su 7,3 GB.
- **File:** `AdaptivePIV#0.vectors` (`a68ec3a378b91cbc`), `ValRange#0.vectors` (`cd10c56520cc2c75`), `ValMovAvg#0.vectors` (`dc9fa0769e019f96`) del ramo `I.4xd8gr7l` (immagini mascherate); `Run.xml` (`bc6758d0a8c89b96`); README (`c709f97bc6d616d3`).
- **Decoder.** Il formato è una serializzazione binaria .NET. Ho scritto un decoder euristico che riconosce i record della classe `Vector`: 14 membri, passo costante di **98 byte** fra i record, verificato su 4939 intervalli. Il primo elemento, che contiene la definizione della classe, non viene decodificato. Restano quindi 4940 vettori su 81×61 = 4941. **Il decoder è stato verificato solo per coerenza interna:** passo, coordinate in multipli di 16 px, valori NaN dove ci si attende la maschera. Non è stato confrontato con un export ufficiale di DynamicStudio.
- **Risultati per il frame #0:**

| Livello | Stato 0 | Stato 1 (NaN) | Stato 16 | Stato 17 | Valori finiti |
|---|---:|---:|---:|---:|---:|
| AdaptivePIV | 3632 | 800 | 508 | 0 | 4140 |
| ValRange | 3515 | 917 | 506 | 2 | 4140 |
| ValMovAvg | 3512 | 0 | **1428** | 0 | **4940** |

- **Transizioni fra livelli:**
  - degli 800 vettori NaN (stato 1) del livello adattivo, tutti diventano stato 16 e **ricevono un valore** dopo la media mobile;
  - dei 3632 con stato 0, 117 vengono rigettati dalla validazione per intervallo e 120 finiscono sostituiti;
  - i valori di U cambiano fra il livello adattivo e quello finale in 122 vettori, tutti con stato finale 16.
- **Lettura (I):** il campo finale è completo, ma il 28,9% dei vettori è stato sostituito, compresa una regione che nel livello adattivo era NaN e mascherata. Senza il codice di stato, il campo finale sembrerebbe privo di lacune.
- **Coordinate:** X fra 0 e 1200 px e Y fra 0 e 960 px con passo 16, per i vettori con coordinate salvate. I vettori di stato 1 e 16 hanno X = Y = 0: le coordinate vanno ricavate dall'indice di array (I).
- **Limiti:** un solo frame; significato dei codici non documentato; scala fisica non decodificata; il campo è isotermo.

### 4.2 Neal, regione turbolenta (S2D-02)

- **File:** directory centrale; `Database description.pdf` (`da5292d1485bee75`, letto integralmente); `data description.txt`; da `H016/HDR_Data_H016.zip` (compresso e annidato), i primi 2 MB decompressi in streaming → `B00003.dat` (`88a58e6e3a6184e9`), `B00004.dat`, `B00005.dat`; da `MS_Images_H016.zip`, i primi 6 MB → `B00001.tif` (`475b5252dc471d17`), `B00002.tif`.
- **Controlli sul riferimento:** l'header Tecplot corrisponde al PDF; zona I = 168, J = 167; passo 1 px; flag {0: 7134, 1: 20 922}; flag 0 ⇔ vettore esattamente nullo (7134 su 7134), nessun vettore nullo con flag 1, nessun NaN; nei validi, Vx ≈ 2,22 ± 0,40 px, Vy ≈ 0,00 ± 0,30 px, Vz ≈ 0,07 ± 0,37 px. Queste dispersioni comprendono la variabilità turbolenta: **non sono errore**.
- **Controlli sulle immagini MS:** 512×512 uint16 (il PDF dichiara un sensore 1024×1024; I: immagini ritagliate); massimo 792–880 (sensore a 10 bit); mediana circa 16–17 conteggi.
- **Osservazione:** il primo file HDR del set è `B00003`. I: coerente con un kernel di 5 coppie che non produce i primi due campi, non verificato.

### 4.3 Darmstadt MRB (S2D-03)

Lo zip `Data_MRB_PIV_Avg.zip` (`6ba8ff6b2a42b954`, 8,26 MB) è stato aperto integralmente, con i 18 file controllati. Risultati al §3, scheda S2D-03. Esempi: `PIV_avg_CH4_FOV1_Phi140.mat` (`434cdb51bb77e704`) ha 6734 NaN in vx e vy; `PIV_avg_H2_FOV2_Phi350.mat` (`4782fceda8db1968`) ha 6623 zeri in vx nelle stesse celle dei 6623 NaN di vy.

### 4.4 RealPDEBench, fluidi (S2D-04)

- **File:** `version.json` (`e2f5bdb727ec0941`); testa (`36d0fb00708d5330`, 2 MB) e coda dello shard 0 della PIV reale del cilindro.
- **Metodo:** schema Arrow letto con `pyarrow`; buffer del record batch ricostruiti dai metadati (37 buffer, 2 righe per batch: per ogni riga, u = 130 744 320 byte = 3990 × 64 × 128 × 4); poi lettura diretta dei primi frame di u e v e dei vettori x, y e t.
- **Risultati:** al §3, scheda S2D-04.
- **Limite:** una sola traiettoria e 14–31 frame. L'assenza di NaN non dice nulla su eventuali riempimenti.

### 4.5 Altri campioni

- **Johe et al. 2022 (S2D-09):** `Fig6.mat` (`f69a1d253f28d55b`), `Fig8.mat`, `Fig11.mat`.
- **Ye et al. 2026 (S2D-10):** `README.txt` (`cdaf2e886f75ee32`), `fig_IgnFlowField.h5` (`d4fb87e9024b0d6d`), `fig_MainVortex_inst.h5`, `fig_Appendix_InstVortex.h5`.
- **Yang et al. 2020 (S2D-11):** `techplotdata.dat` (`6777998561e71f9d`).
- **PIV Challenge 2014 (S2D-05):** istruzioni dei casi B, E e F (`bcf207b5…`, `29dcda5c…`, `dc97763b…`).
- **Paper letti come PDF** (per parole chiave): Balusamy et al. 2015 (Springer OA) e Saini e Arndt 2016 (eLib DLR).

---

## 5. Selezione per lo step 3

La selezione indica **per quale verifica** serve ogni dataset. Le dimensioni sono quelle da scaricare per una verifica utile; nessun download è stato eseguito oltre ai campioni.

| Priorità | Dataset | Verifica proposta | Materiale da ottenere | Costo e condizioni | Che cosa **non** potrà dire |
|---|---|---|---|---|---|
| 1 | **S2D-01 Cardiff** | Frazione per frame di vettori rigettati, sostituiti e mascherati nei 514 frame; contiguità e persistenza delle regioni; separazione fra maschera, intervallo e media mobile; entità delle modifiche introdotte dalla sostituzione; confronto fra rami con finestre diverse (20–40 contro 32–64 px) | Le tre cartelle `.vectors` del ramo principale (≈263 MB l'una, ≈0,8 GB in tutto) via HTTP Range; opzionalmente le immagini grezze (2,83 GB) | Decoder euristico da consolidare (controllo su più frame e, se possibile, sul paper); chiarire la scala fisica | Effetti della fiamma; incertezza; frequenze in un'altra configurazione |
| 2 | **S2D-02 Neal** | Errore del MS rispetto all'HDR dopo un processamento documentato; dipendenza da focus e inseminazione; correlazione spaziale degli errori a più distanze, componenti e overlap; copertura di un eventuale metodo UQ | Regione "unsteady core" (271,8 MB) per partire, oppure "turbulent" (781,6 MB) | Serve un software PIV dichiarato e riproducibile per i vettori MS: è una scelta da approvare | Comportamento in fiamma; validità per altri apparati |
| 3 | **S2D-04 RealPDEBench fluidi** | Layout, passo e tempi effettivi della PIV della nostra benchmark; ricerca di regioni con valori costanti o lisci attorno al cilindro (indizio, non prova, di riempimento); chiarimento della discrepanza di Δt | 1–2 shard reali (≈0,8 GB l'uno) o letture parziali | Leggero; resta senza flag | Rigetti storici; errore |
| 4 | **S2D-03 Darmstadt MRB** (+ RMS) | Estensione e forma delle regioni non valide nelle medie in fiamma H2/CH4; FOV e passo; documentare le convenzioni NaN/0 | 2 zip da 8 MB | Molto leggero; unità e significato di "Phi" da recuperare dal paper | Lacune istantanee; rumore |
| 5 | **S2D-05 PIV Challenge** 2001-C e 2014-B | Ispezione a livello d'immagine di riflessi, sfondo e maschera, e di inseminazione e fuori piano in un caso TR | Immagini 2001-C (pochi MB) e 2014-B (dimensione M) | Nessuna verità sperimentale; verità sintetica solo su richiesta | Errori quantitativi senza riferimento |
| 6 | **S2D-06 TCC-III Fired** | Unica occasione trovata di osservare **vettori istantanei e lacune in presenza di fiamma** in un rilascio pubblico | ≈19,6 GB via Globus, dopo aver superato la challenge di Deep Blue: serve un'azione di Enrico (browser e Globus) | Oneroso; prima vanno letti metadati e README | Trasferibilità a uno swirl NH3/CH4 |
| 7 | **S2D-08 Cambridge Oxycoal** | Lacune e high-speed PIV in uno swirl reagente; effetto del cambio di tracciante (allumina o carbone) | Riprovare l'accesso al repository; dimensione M | Dipende dalla disponibilità del server | Rappresentatività per fiamme gassose |

**Raccomandazione.** Per lo step 3 propongo di partire da **Cardiff (lacune e sostituzione)** e **Neal (errore e correlazioni)**: sono i due casi in cui una verifica statistica è davvero possibile, anche se nessuno dei due è reagente. In parallelo, una lettura leggera di RealPDEBench fluidi e di Darmstadt MRB. Se la discussione conclude che serve un'evidenza reagente diretta sulle lacune, l'unico accesso concreto trovato è TCC-III Fired, che richiede un intervento manuale.

**Risposte alle domande decisive della scheda dello step 1 (§8):**

| Domanda | Dove la risposta è "sì" | Dove resta "no" o M |
|---|---|---|
| Possiamo distinguere vettore misurato e interpolato? | Cardiff (codici di stato, significato inferito); convenzione PIV Challenge (flag 2, ma solo nei risultati dei partecipanti, non pubblici) | Neal (sostituzioni non marcate nel riferimento); RealPDEBench, EngineBench, Darmstadt (nessun flag) |
| Possiamo separare FOV, ostacolo e rigetto? | Cardiff (maschera definita, poi validazioni separate); Shang e Tu (zero = maschera documentata); Ye et al. (NaN "outside the valid region") | Darmstadt MRB, Yang (zeri e NaN non documentati); RealPDEBench (niente) |
| Esistono immagini originali o mappe UQ? | Immagini: Cardiff, Neal, PIV Challenge | **Nessuna mappa UQ rilasciata** fra i dataset accessibili; un'UQ in combustione è solo descritta in un paper (S2D-14) |
| Abbastanza condizioni e frame? | Neal (più regioni, focus e inseminazione); RealPDEBench (molte traiettorie) | Cardiff: 514 frame ma una sola condizione; Darmstadt MRB: solo medie |
| Ispezione senza scaricare tutto? | Sì per zip su 4TU, Figshare/S3, TUdatalib e Arrow su HF (HTTP Range) | No per Deep Blue (challenge) e Kaggle (account) |

---

## 6. Registro delle ricerche, esclusioni e problemi d'accesso

### 6.1 Ricerche eseguite (in ordine approssimativo)

1. **Ricerca web** (WebSearch, modalità standard ed extended):
   - PIV e swirl flame su Zenodo; Neal 2015, database; EngineBench, Kaggle e TCC-III; Cambridge Oxycoal; International PIV Challenge; ammonia–methane swirl PIV a Cardiff; Darmstadt TSF e TUdatalib; DLR GTMC e PRECCINSTA; TCC-III su Deep Blue;
   - frasi di data availability per swirl PIV/OH-PLIF; RealPDEBench HF; Cambridge–Sandia; Boomsma 2016; HS-PIV in fiamme premiscelate e H2; Saini/Arndt/Steinberg; disponibilità dei dati DLR GTMC; TCCIII Fired Full View; Balusamy 2013.
2. **API dei repository:**
   - Zenodo `/api/records` (tipo dataset) con le query: "PIV flame", "PIV combustor", "PIV swirl", "PIV burner", "velocimetry flame", "stereo PIV combustion", "PIV OH-PLIF", "PIV ammonia", "PIV hydrogen flame", "PIV uncertainty", "PIV challenge", "particle image velocimetry uncertainty quantification", "PIV synthetic images ground truth". La ricerca full-text di Zenodo è molto rumorosa: i risultati pertinenti sono stati filtrati a mano;
   - 4TU (API v2: record e ricerca "PIV uncertainty");
   - Figshare (API v2: "PIV flame", "PIV swirl burner", "particle image velocimetry combustion", "PIV combustor", "stereo PIV flame", "PIV OH PLIF");
   - TUdatalib (API DSpace 7: "PIV", "particle image velocimetry", "velocimetry", "swirl", "flame", "burner", "OH-PLIF");
   - Harvard Dataverse ("PIV flame", "PIV combustor");
   - Mendeley Data (API pubblica: nessun risultato interpretabile, M);
   - Hugging Face (alberi del repository RealPDEBench).
3. **Pagine lette:** progetto EngineBench di Oxford; pivchallenge.org (2001, 2005, 2014); docs di RealPDEBench (cylinder); database swirl di Sydney (pagina introduttiva).
4. **Paper usati per interpretare i dati** (vedi §7): solo le parti necessarie, dichiarate scheda per scheda.

### 6.2 Problemi d'accesso

| Risorsa | Problema | Effetto |
|---|---|---|
| Deep Blue (Michigan: TCC-III fired, motored, CFD Input) | 403 con challenge JavaScript per le richieste dirette; l'host non è risolvibile dal web-fetch | S2D-06 descritto solo da snippet |
| www.repository.cam.ac.uk (Apollo) | 503 "No server is available" per circa 30 minuti; nessun metadato dalla negoziazione del DOI | S2D-08 da snippet e paper |
| Kaggle | API senza autenticazione vuota; nessuna credenziale richiesta, per scelta | EngineBench non riscaricato; riuso dell'audit precedente |
| TUdatalib | Alcuni item `restrictedAccess` (401: S2D-20, S2D-29); i download richiedono uno user-agent da browser | Esclusioni per accesso |
| Cardiff CH4–H2 (figshare 27052006) | Elenco dei file vuoto via API | Contenuto M; PIV non citata nella descrizione |
| Paywall | Balusamy et al. 2013; molti paper di combustione | Parametri PIV del CCB M |
| Rete della sessione | Gli host degli archivi non erano inizialmente consentiti | Aggiunti, con approvazione dell'utente, host esatti di repository dati (Zenodo, Figshare/S3, 4TU, TUdatalib, HF e CDN, Deep Blue, Cambridge, pivchallenge.org, ecc.) all'ambiente personale della sessione |

### 6.3 Esclusioni motivate

- **Senza PIV:** S2D-21, 22, 24, 32, 33.
- **Solo video:** S2D-23.
- **Accesso ristretto:** S2D-20, 29.
- **Dati non trovati:** S2D-13–16 (configurazioni utili come precedente), 25, 26, 27, 34.
- **Rimandati per dimensione o priorità:** S2D-17, 18, 19, 28, 30 (registrati con dimensione, non scaricati).
- **Non scaricati per dimensione,** pur essendo pertinenti: Cardiff completo (7,3 GB), Neal steady core (1,39 GB), RAFT-PIV (12,2 GB), SPID (16,7 GB), RealPDEBench completo (≈36 GB solo per la PIV reale del cilindro), Shang e Tu (1,08 GB).

### 6.4 Che cosa **non** è stato fatto

- Nessuna statistica su interi dataset: è il compito dello step 3.
- Nessun processamento PIV di immagini.
- Codici di stato DynamicStudio: interpretati, non certificati.
- Paper dei dataset principali non letti integralmente: Runyon et al. 2018, Neal et al. 2015 (il PDF descrittivo del database sì), Li et al. 2024, Johe et al. 2022, Ye et al. 2026, Kähler et al. 2016.
- Nessun contatto con autori. Nessuna modifica ad altri file della repository.
- **Non pretendo di aver censito tutti i dataset PIV esistenti.** Archivi istituzionali non interrogati sistematicamente: per esempio mediaTUM, DepositOnce, ETH Research Collection, Bristol, Toronto, DaRUS e repository giapponesi o cinesi.

---

## 7. Bibliografia e collegamenti

| ID | Riferimento | Accesso e parti lette | Schede |
|---|---|---|---|
| S01 | Runyon J., Marsh R., Bowen P., Pugh D., Giles A., Morris S., *Lean methane flame stability in a premixed generic swirl burner: isothermal flow and atmospheric combustion characterization*, dataset, Cardiff University, [doi:10.17035/d.2017.0011500240](https://doi.org/10.17035/d.2017.0011500240) (figshare 27051964). Paper: Exp. Therm. Fluid Sci., [doi:10.1016/j.expthermflusci.2017.11.019](https://doi.org/10.1016/j.expthermflusci.2017.11.019) | Metadati, README, XML, 4 file di vettori; paper non letto | S2D-01 |
| S02 | Neal D.R., Smith B.L., Sciacchitano A., *PIV uncertainty — Database of the turbulent region / unsteady inviscid core / steady inviscid core of a transitional jet*, 4TU.ResearchData, [10.4121/13649885](https://doi.org/10.4121/13649885), [10.4121/13663376](https://doi.org/10.4121/13663376), [10.4121/13341842](https://doi.org/10.4121/13341842). Paper: Neal et al., MST 26 (2015) 074003, [doi:10.1088/0957-0233/26/7/074003](https://doi.org/10.1088/0957-0233/26/7/074003) | PDF descrittivo letto integralmente; campioni; paper non letto | S2D-02 |
| S03 | Li T., Doğrudil M., Dreizler A., *Data for "Macroscopic flame and flow structures in hydrogen and methane multi-regime combustion"*, TUdatalib, item [e6d83b56-…](https://tudatalib.ulb.tu-darmstadt.de/items/e6d83b56-0a21-49b2-8076-c6113cc5b680); paper [doi:10.1016/j.proci.2024.105759](https://doi.org/10.1016/j.proci.2024.105759) | Metadati e zip delle medie; paper non letto | S2D-03 |
| S04 | AI4Science-WestlakeU, *RealPDEBench*, [Hugging Face](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench), data v2.0.1; [docs cylinder](https://realpdebench.github.io/datasets/cylinder); arXiv:2601.01829 (audit precedente) | README, version, schema, campione dello shard | S2D-04 |
| S05 | [pivchallenge.org](https://www.pivchallenge.org/); Stanislas et al. 2003 (MST 14 R63), 2005 (Exp Fluids 39:170), 2008 (Exp Fluids 45:27); Kähler et al. 2016 (Exp Fluids 57:97, [doi:10.1007/s00348-016-2173-1](https://doi.org/10.1007/s00348-016-2173-1)) | Pagine del sito e istruzioni 4B, 4E, 4F; paper non letti | S2D-05 |
| S06 | Schiffmann P., Sick V., Reuss D., *TCCIII Fired Full View*, Deep Blue Data, [doi:10.7302/Z2H41PCD](https://doi.org/10.7302/Z2H41PCD); *TCCIII Motored Full View*, [doi:10.7302/Z2MS3QP3](https://doi.org/10.7302/Z2MS3QP3); *TCC-III CFD Input Dataset*, [hdl 2027.42/108382](https://deepblue.lib.umich.edu/handle/2027.42/108382) | Solo snippet di ricerca (pagine bloccate) | S2D-06 |
| S07 | Baker S.J. et al., *EngineBench*, arXiv:2406.03325v1; [pagina Oxford](https://eng.ox.ac.uk/tpsrg/research/enginebench); [GitHub sambkr/EngineBench](https://github.com/sambkr/EngineBench); Kaggle [10.34740/KAGGLE/DS/5000332](https://doi.org/10.34740/KAGGLE/DS/5000332); audit in `docs/nonideal/00_ricognizione_osservazioni.md` §3 | Pagina, README, HTML del paper per parole chiave | S2D-07 |
| S08 | Balusamy S., Gao Y., Kamal M.M., Lowe A. (dataset CCB), Apollo, DOI 10.17863/CAM.96660 (da snippet); Balusamy S. et al., Exp Fluids 56:108 (2015), [doi:10.1007/s00348-015-1965-z](https://doi.org/10.1007/s00348-015-1965-z); Balusamy S., Schmidt A., Hochgreb S., Exp Fluids 54:1534 (2013) | Repository non raggiungibile; paper 2015 letto per parole chiave; paper 2013 non accessibile | S2D-08 |
| S09 | Johe P. et al., *Characterization of flow field and combustion dynamics in a novel pressurized side-wall quenching burner using highspeed PIV/OH-PLIF measurements*, TUdatalib item b4c48efc-… | Metadati e 3 file di figura | S2D-09 |
| S10 | Ye P., Erhard J., Welch C., Dreizler A., Boehm B., *Data for "Cycle-to-cycle variations in a hydrogen spark ignition engine under EGR diluted and lean conditions"*, TUdatalib item e156e984-… | README e 3 HDF5 | S2D-10 |
| S11 | Yang Z. et al., *20 kHz dual-plane stereo-PIV measurements on a swirling flame using a two-legged burst-mode laser*, Opt. Lett. 45 (2020) 5756, [doi:10.1364/OL.404002](https://doi.org/10.1364/OL.404002); dati [figshare 10.6084/m9.figshare.12904478](https://doi.org/10.6084/m9.figshare.12904478) | File completo; paper non letto | S2D-11 |
| S12 | Shang J., Tu J., *Particle image velocimetry (PIV) data of flow past a cylinder*, Zenodo [10.5281/zenodo.20765567](https://doi.org/10.5281/zenodo.20765567) | Descrizione del record | S2D-12 |
| S13 | Saini P., Arndt C.M., Steinberg A.M., *Development and evaluation of gappy-POD as a data reconstruction technique for noisy PIV measurements in gas turbine combustors*, AIAA 2016-0436 ([eLib DLR 102428](https://elib.dlr.de/102428/)); Exp Fluids 57:122 (2016) | PDF AIAA per parole chiave | S2D-13 |
| S14 | arXiv:2011.14662, *Impact of precessing vortex core dynamics on the thermoacoustic instabilities in a swirl stabilized combustor* | HTML, sezione diagnostica | S2D-14 |
| S15 | arXiv:2312.06461, *Variational Auto-Encoder Based Deep Learning Technique For Filling Gaps in Reacting PIV Data* | Abstract e HTML per parole chiave | S2D-15 |
| S16 | arXiv:1904.00546; arXiv:1909.13669; arXiv:2003.03662; arXiv:2607.13193 | Abstract e HTML per parole chiave | S2D-16 |
| S17 | Lagemann C. et al., *Dataset — Deep Recurrent Optical Flow Learning for PIV Data*, Zenodo [10.5281/zenodo.4432496](https://doi.org/10.5281/zenodo.4432496) | Metadati | S2D-17 |
| S18 | Machado M., Rocha D., *RSPID* [10.5281/zenodo.7832205](https://doi.org/10.5281/zenodo.7832205) e *SPID* [10.5281/zenodo.7935215](https://doi.org/10.5281/zenodo.7935215) | Metadati | S2D-18 |
| S19 | Tirelli I. et al., *Full-domain POD modes from PIV asynchronous patches*, Zenodo [10.5281/zenodo.15365451](https://doi.org/10.5281/zenodo.15365451) | Metadati | S2D-19 |
| S20 | TUdatalib: *Velocity Fields for Standard Operating Points — Motored Engine* (b8402527-…); *Darmstadt engine flow bench velocity data* (2082d0a2-…); *In-situ gas-phase velocity field measurements in metal dust flames* (2ad1d316-…); *Local thermochemical states in turbulent H2-air multi-mode flames* (dafd4cea-…) | Metadati | S2D-20, 28, 29 |
| S21 | Runyon J. et al., *Experimental analysis of confinement and swirl effects on premixed CH4-H2 flame behavior in a pressurized generic swirl burner*, [doi:10.17035/d.2017.0031562739](https://doi.org/10.17035/d.2017.0031562739) | Metadati | S2D-21 |
| S22 | Singh S. et al., record Zenodo 23037088, 20071084, 23062725 (KIT) | Metadati | S2D-22 |
| S23 | Ruan C. et al., Energy Fuels (2021), SI figshare [10.1021/acs.energyfuels.0c03165.s002](https://doi.org/10.1021/acs.energyfuels.0c03165.s002) | Listing dello zip | S2D-23 |
| S24 | Database swirl di Sydney, [pagina](https://web.aeromech.usyd.edu.au/thermofluids/swirl.php); `00_ricognizione_osservazioni.md` §4 | Pagina introduttiva; audit precedente | S2D-24 |
| S25 | Cambridge Stratified Swirl Burner, LDA (Apollo, item 22792653-…) | Solo snippet | S2D-25 |
| S26 | Boomsma A. et al., MST 27 (2016) 094006 | Solo abstract APS e snippet | S2D-26 |
| S27 | Ebi D., Clemens N.T., Combust. Flame 168 (2016) 39–52; tesi UT Austin | Solo snippet | S2D-27 |
| S28 | Sciacchitano A., MST 30 (2019) 092001, tramite `step1_sciacchitano.md` | Scheda dello step 1 | Regole d'interpretazione |

---

## Appendice — collegamento con la missione 01

Le righe B dei candidati prioritari (S2D-01, 02, 03, 04, 05) possono essere copiate nell'inventario della missione 01 con questi ID. Per RealPDEBench, la scheda S2D-04 aggiunge all'audit comune **solo** gli elementi sui sistemi fluidi:
- versione 2.0.1;
- risoluzione nei file 64×128;
- passo temporale di 5 ms nei file, contro i 2,5 ms dichiarati;
- assenza di flag nello schema.

Non tocca la combustione né la slice LES usata dalla tesi.
