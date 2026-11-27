# Observation–State Matrix

**Stato:** prima versione accettata con Enrico il 26 novembre 2026; **F3 conclusa**. Aggiornamenti intermedi dalle missioni 02 (26 novembre) e 04 (27 novembre) integrati (F5). F4, confronto complessivo, rinviata alle evidenze pertinenti delle altre missioni; missione aperta.
**Ultimo aggiornamento:** 27 novembre 2026 · **Branch di pubblicazione:** `plan/month3-missions`. Provenienza storica F3 in §1/§8; fonti degli aggiornamenti PIV/termico in §9/§10.
**Roadmap:** [roadmap01.md](roadmap01.md), manutenzione F5 · **Formato:** [templates_abc.md](templates_abc.md)
**Natura:** integrazione documentale delle consegne revisionate F1/F2 e degli scouting intermedi della 02 e della 04. Nessun esperimento, training, inferenza, download di dati o lettura dei TEST in questa integrazione; nessun contatto esterno. Le ispezioni di campioni effettuate dagli scout sono attribuite ai loro rapporti.

---

## 0. Guida alla lettura

Questo documento è l'**indice ragionato** della missione. Collega tre viste:

- **A — Matrice delle misure (§2):** come si misura una grandezza: tecnica, segnale realmente acquisito, supporto spazio-temporale, errori e limiti, disponibilità.
- **B — Inventario dei dati (§3):** che cosa esiste e che cosa abbiamo davvero.
- **C — Catalogo dei problemi (§4):** osservazioni disponibili durante l'uso → stato desiderato → approcci → validazione → che cosa manca.

Il centro è A; C la completa. **Input, target e riferimento sono ruoli del singolo problema C**, non proprietà di una variabile. I dettagli restano nei rapporti collegati (§1, §9 e §10): qui si riportano solo le informazioni necessarie per confrontare le righe. Se un numero perde il suo contesto in sintesi, è omesso e si rimanda alla scheda.

Ogni numero vale **per l'apparato, la simulazione o la condizione indicati**. Non è un parametro della tecnica in generale né del nostro combustore.

### Legende

| Asse | Codici |
|---|---|
| Evidenza (per affermazione) | **D** documentato nella fonte indicata · **I** inferenza con premesse esplicite · **M** mancante/non verificato. Una conclusione non diventa D solo perché compare in una consegna revisionata: si conserva la marcatura del rapporto d'origine |
| Profondità della fonte | **TI** testo integrale · **SZ** sezioni dichiarate · **AB** solo abstract · **SN** snippet, metadati o fonte secondaria · **RP** rapporto/audit preesistente della repository, non rieseguito · **LC** controllo locale diretto eseguito nella consegna F1 · **F3** controllato durante il consolidamento iniziale · **M02/M04** riportato dalle consegne delle missioni 02/04, con profondità originale e autore dei controlli espliciti |
| Natura del dato | **MIS** misura sperimentale · **NUM** campo numerico · **SIN** osservazione sintetica o proxy derivato da una simulazione |
| Supporto spaziale | **Pt** punto · **Ln** linea · **Pl** piano (sezione) · **Pr** proiezione (integrale lungo la vista) · **Vol** volume |
| Imperfezioni (da SCO §2) | **E** errore sul valore (casuale, bias, correlato) · **F** filtraggio spaziale/risoluzione finita · **R** risposta e integrazione temporale · **L** lacune (assenza di osservazione, rigetto, campo di vista, durata) |
| Accesso (B) | **LOC-I** presente localmente e ispezionato · **LOC** presente localmente, contenuto non caratterizzato · **SCOUT-I** campione ispezionato dagli scout delle missioni 02/04, possesso sul computer di Enrico non verificato · **DICH** dichiarato dalla fonte, non posseduto · **RICH** disponibile su richiesta, non richiesto · **CIT** solo citato o visto in snippet · **NESS** nessuna evidenza nel perimetro consultato |
| Validazione (C) | **VN** numerica, contro simulazione · **VS** sperimentale, contro misura indipendente · **VST** statistica o integrata (PDF, medie, spettri) · **VC** coerenza con gli input o con un residuo fisico. Si indica sempre se il riferimento è lo **stesso stato simultaneo** o solo lo **stesso scenario** |

### Citazioni e identificativi

Le fonti locali dei rapporti hanno numerazioni indipendenti: `S01` di un rapporto non è `S01` di un altro. Qui ogni fonte porta il prefisso del rapporto d'origine:

| Prefisso | Rapporto | Esempio |
|---|---|---|
| `INV` | [data_inventory.md](data_inventory.md) | `INV:S07`, `INV:B-data-audit-005`, `INV §7 C1` |
| `NIE` | [papers/nie_2024.md](papers/nie_2024.md) | `NIE:S01 p. 3`, `NIE §6` |
| `LIU` | [papers/liu_2025.md](papers/liu_2025.md) | `LIU:S01 Fig. 6`, `LIU §3.2` |
| `SCO` | [scouting/measurement_scouting.md](scouting/measurement_scouting.md) | `SCO:S04 §3.7`, `SCO A-001` |
| `PIV1` | [Scheda Sciacchitano della 02](../02_virtual_piv/step1_sciacchitano.md) | Evidenze della review mediate dalla scheda approvata |
| `PIVL` / `PIVD` | [Scouting letteratura / dataset della 02, versioni fissate](updates/piv_2026-11-26.md#1-fonti-e-profondità) | `PIVL:L37`, `PIVD:S2D-01` |
| `TH1` / `THM` / `THD` | [Consegne termiche della 04, versioni fissate](updates/thermal_2026-11-27.md) | Pratica, modelli e dataset; THD:D01 ecc. |

Gli ID originali delle righe sono conservati. Righe nuove, suddivise o accorpate usano `A-consolidation-NNN`, `B-consolidation-NNN`, `C-consolidation-NNN`; le corrispondenze sono in §5.

Nelle celle dei collegamenti si usano forme brevi, senza ambiguità: `-scout-` = `-measurements-scout-`; `-cons-` = `-consolidation-`; `A-nie-`, `B-nie-`, `C-nie-` = `…-jason-nie-`; `A-liu-`, `C-liu-` = `…-jason-liu-`. Esempio: `A-scout-013` = `A-measurements-scout-013`.

---

## 1. Provenienza delle consegne integrate

I quattro file sono stati importati dai commit indicati, senza mergiare le PR e senza modificare i branch d'origine. Nella base `plan/month3-missions` @ `c79c470` nessuno dei quattro file esisteva: nessun confronto o fusione con versioni preesistenti è stato necessario.

| Consegna | PR | Branch @ commit | File importato | Natura e profondità | Modifiche in F3 |
|---|---|---|---|---|---|
| Inventario dati | [EnricoBontempi/official_thesis#1](https://github.com/EnricoBontempi/official_thesis/pull/1) | `matrix/f1-data-audit` @ `a7bbb5761b34c256f79b5eaa21feff40de617474` | [data_inventory.md](data_inventory.md) | Audit documentale; controlli locali leggeri (LC) su metadati e `t` di 23 shard non-TEST; paper RealPDEBench da audit preesistenti (RP) | Due note circoscritte (§8): policy `40NH3_1.2`; unità delle dimensioni OH\* |
| Paper di Jason | [EnricoBontempi/official_thesis#2](https://github.com/EnricoBontempi/official_thesis/pull/2) | `matrix/f1-jason-papers` @ `07b4423782804e1b39d31051847d3b5ae650758b` | [papers/nie_2024.md](papers/nie_2024.md), [papers/liu_2025.md](papers/liu_2025.md) | TI dei PDF dell'editore forniti da Enrico; supplementari non letti | Nie: aggiunto l'URL corretto dell'articolo (§8); Liu: nessuna |
| Scouting misure | [EnricoBontempi/official_thesis#3](https://github.com/EnricoBontempi/official_thesis/pull/3) | `matrix/f2-measurement-scouting` @ `13ec384b782bd6ab14c8ee1a6463b044fe5e8ac1` | [scouting/measurement_scouting.md](scouting/measurement_scouting.md) | 12 fonti open access lette in TI o SZ; altre AB o SN, marcate | Nessuna |

---

## 2. Matrice A — misure e osservazioni

33 righe in sette famiglie. Le righe A.7 sono **osservazioni sintetiche o proxy**, non diagnostiche sperimentali disponibili. Ogni riga rimanda alla scheda dettagliata tramite la fonte.

Colonne comuni: **Grandezza e contesto** · **Tecnica → segnale** (con natura e componenti) · **Spazio** (supporto, copertura, risoluzione) · **Tempo** · **Errori e limiti** (tipo E/F/R/L) · **Disponibilità · fonte · evidenza**.

### A.1 Temperatura

| ID | Grandezza e contesto | Tecnica → segnale | Spazio | Tempo | Errori e limiti | Disponibilità · fonte · evidenza |
|---|---|---|---|---|---|---|
| A-measurements-scout-001 | T del gas [K]; pool fire di metanolo da 1 m (NIST), non premiscelato, puffing ≈ 1,37 Hz | Termocoppia tipo S, filo 50 µm, giunto ≈ 153 µm → f.e.m. = T del **giunto**; T gas = T giunto + correzione radiativa + correzione inerziale (bilancio sul giunto). MIS, scalare | Pt; 46 posizioni assiali/radiali. In un caso calcolato diverso, la lettura risente di gradienti su alcuni mm tramite la conduzione nei fili (SCO:S05) | 60 Hz × 120 s; τ = 57 ± 3 ms a (z = 30 cm, r = 0), circa ∝ 1/T media. La fonte riporta «720 measurements» contro 7200 campioni calcolati: incoerenza non risolta | **R:** a (30 cm, 0), test 3, la correzione inerziale sposta la media di 0,2 K ma la std passa da 237 K (giunto) a 385 K (gas) (Tab. 5). **E:** correzione radiativa fino a 55 K; incertezza combinata 5 % sulla media e 26 % sulla std. La correzione usa velocità del gas **simulate (FDS)**: misura corretta dipendente da un modello. Conduzione trascurata in S04, rilevante nel caso di S05: differenza di applicabilità, non conflitto dimostrato | Tecnica diffusa (SCO:S05, I per i limiti d'uso); numeri solo per questo fuoco · SCO:S04 SZ, S05 TI · D |
| A-measurements-scout-002 | T [K], 7 specie maggiori, φ locale; fiamme CH₄/aria turbolente premiscelate e stratificate (slot burner di Cambridge, misure Sandia) | Raman (spettri) + Rayleigh (532 nm) + CO-LIF a due fotoni; due piani OH-PLIF incrociati per l'orientazione del fronte. Catena T ← Rayleigh + composizione Raman: I. MIS, scalari multipli | Ln 1D; superpixel 0,104 mm (Raman) e 0,020 mm (Rayleigh). Il gradiente è solo la proiezione lungo la linea; 3D solo con i piani incrociati | Singolo colpo; gate Raman 3,9 µs FWHM; frequenza di ripetizione M; nessuna sequenza | **E:** in fiamma premiscelata di riferimento (φ = 0,97, T = 2185 K) precisione σ 0,75 % e accuratezza 2 % su T; φ 2,2 %/5 % (Tab. 2). In turbolenza φ dei prodotti ≈ +0,02 sopra il nominale. **F:** media spaziale su gradienti > 2000 K/mm. Apparato con 1,8 J/impulso totali | Laboratorio dedicato; dati di questo tipo nell'archivio TNF (B-data-audit-007) · SCO:S03 TI della **versione sottomessa** (la pubblicata può differire) · D |
| A-measurements-scout-003 | T planare | Rayleigh planare / filtered Rayleigh: intensità ∝ densità × σ_eff → T. MIS | Pl | M | **E-bias** da σ_eff dipendente dalla composizione (I); numeri rimossi perché non attribuibili | Sola pista · SCO:S25 SN · I/M |
| A-measurements-scout-004 | T planare; combustore modello per turbine a gas, pulsante, 1 atm | Two-line OH-PLIF, rapporto di due PLIF di OH. MIS | Pl, **solo dove c'è OH**: lacuna correlata con lo stato (L, I dal principio) | Media in fase | Accuratezza M | Titolo e sede D · SCO:S16 SN (metadati) · I/M |
| A-jason-nie-001 | T del gas [K]; getti turbolenti non premiscelati di toluene con soot, bruciatore JHC di Adelaide (Re 10 000–15 000) | nTLAF a indio seminato: due ICCD (410/451 nm), T dal rapporto delle fluorescenze con calibrazione. MIS, scalare | Pl centrale; foglio 16 mm × ~0,25 mm; finestra usata 70 × 10 mm a 0,089 mm/px, poi filtro gaussiano 0,5 × 0,5 mm². Copertura legata alla presenza di indio: M | Scatto singolo; 1 Hz, imposto dalla sincronizzazione del sistema combinato; gate M | **E:** ±120 K come std delle immagini istantanee; non è detto se per pixel o per immagine, né se prima o dopo il filtro (M). Accuratezza della media legata alla calibrazione. Limiti dichiarati: interferenze soot/PAH, difficoltà a kHz | Sistema di laboratorio; dati su richiesta · NIE:S01 TI pp. 1–3 · D (valori), M (definizione dei ±120 K) |
| A-TH-001 | T media del gas, HYLON Phase 2, swirl H₂/aria confinato, 1 atm | Doppia giunzione → estrapolazione a superficie nulla; MIS **già corretta**, non letture grezze | Traversate a sei quote; supporto dichiarato ±4 mm; esclusione del nucleo in parte di Flame A.1 | Solo medie; frequenza, durata e τ M | **E/F/L:** ±10% dichiarato per il metodo; gradienti e perturbazione della CRZ; lega e diametri M. TC/Raman/PIV in campagne separate: differenze non isolano il solo errore TC | Campioni letti dallo scout; fonte primaria mediata da THD:D01 §3.1/§4.4 e THM §3.6bis · M04 D/I/M |
| A-TH-002 | T del gas, Delft jet-in-hot-coflow chiuso (flameless) | TC S inguainata Pt-10%, diametro esterno 3 mm, piastra al posto di una finestra; correzione convettiva/radiativa. MIS | Traversata radiale, media su un diametro di sonda; accesso da porte | Acquisizione 10 Hz, traversata 0,62 mm/s; dato pubblicato mediato, τ M | **E/F/R:** correzione con h da PIV **non reagente in altra condizione**; T parete non definita nelle parti lette. 10 Hz non dimostra banda termica 10 Hz | Configurazione documentale, nessun dataset digitale individuato · THM §3.3 TS · M04 D/M |
| A-TH-003 | T media del gas nella zona di ricircolo esterna relativamente fredda, EM2C swirl CH₄/H₂ | TC K 1,5 mm attraverso barra che sostiene il quarzo; MIS, non corretta | Punti nella ORZ; posizioni precise M | Equilibrio dell'apparato dopo circa 45 min; non è τ della sonda. Simultaneità PIV M | **E:** errore radiativo dichiarato ≤5% per T<1200 K nel caso; non trasferibile al nucleo caldo. τ in situ M | Documento Mercier et al.; originale della TC non letto · THM §3.4 TS · M04 D/M |
| A-TH-004 | **T della parete interna**, HYLON Phase 2 | Fosfori termografici LIP, decadimento di Mg₃.₅FGeO₅:Mn, calibrazione contro TC K; MIS, scalare di superficie | Punti su backplane, quarzo e metallo; non T gas | 50 Hz, media di 100 colpi per punto | Incertezza dichiarata 2,50%; campagne separate dalle altre diagnostiche. Può vincolare condizioni al contorno, non sostituire una misura gas | Profili letti dallo scout · THD:D01 §3.1 · M04 D |


**Catena termica (04).** Lettura della giunzione e stima corretta del gas sono prodotti diversi. Lo scambio radiativo netto può dare scarti di entrambi i segni; inerzia, conduzione e protocollo non si riducono a rumore. Un primo ordine con τ costante è un'approssimazione da qualificare; frequenza di acquisizione e banda del sensore sono distinte. Gas, pareti, ingresso e scarico restano grandezze separate. [Fonti, profondità e limiti](updates/thermal_2026-11-27.md).

### A.2 Velocità

| ID | Grandezza e contesto | Tecnica → segnale | Spazio | Tempo | Errori e limiti | Disponibilità · fonte · evidenza |
|---|---|---|---|---|---|---|
| A-measurements-scout-006 | u 3C [m/s] nel piano centrale; DLR Gas Turbine Model Combustor, swirl CH₄/aria, 1 atm. SCO:S01: 10 kW, φ = 0,75; SCO:S02: 10/20/35 kW. Iniezione separata, fiamme «largely premixed» (S02, da Raman citato) | Stereo-PIV (TiO₂, **solo aria inseminata**) → coppie di immagini di particelle → vettori (multi-pass, finestra finale 16 px, overlap 50 %); OH-PLIF simultaneo come marcatore (A-measurements-scout-008). MIS, vettore 3 componenti su piano | Pl; foglio ≤ 0,7 mm. Risoluzione 0,94 mm e spaziatura 0,47 mm (S01, 5 kHz); 1,5 mm (5 kHz) e 3 mm (10 kHz) in S02: **valori non riconciliati**, nessun valore "tipico". FOV 30 × 30 mm² (5 kHz) o 40 × 40 mm² (10 kHz) in una camera di 85 × 85 mm² | 5 kHz: 0,8 s per record (memoria delle camere); 10 kHz: 8192 registrazioni ≈ 0,8 s (I, calcolo) | **E:** casuale ±0,5/0,8/1,2 m/s per le tre fiamme di S02, derivato **solo** da ±0,1 px sul picco di correlazione; bias non quantificati. **F:** finestra di interrogazione. **L:** FOV, durata, rigetti M; seeding solo nell'aria (lacune nei prodotti: I) | Apparato DLR; accesso ai dati M (B-consolidation-002) · SCO:S01 TI (manoscritto), S02 SZ · D/I |
| A-measurements-scout-007 | u 3C volumetrica; getto d'aria **non reattivo** (bruciatore della serie Sandia, D = 7,45 mm, Re = 13 500) | Tomo-PIV a 4 camere, ricostruzione MART, correlazione 24³ voxel; dato già con vettori spuri sostituiti e filtro spaziale. MIS regolarizzata, vettore 3C | Vol 12,3 × 2,5 × 16,5 mm³; interrogazione 393 µm, spaziatura 98,4 µm (sovracampionata) | 10 kHz; sequenza usata di 50 ms | **F:** volume di interrogazione ≫ spaziatura. **E:** covarianza dell'errore assunta nota; valori M. **L:** lacune originali già sostituite. In fiamma (altro lavoro, SN): profondità tipica 1:10 rispetto al FOV e ostacoli elencati | Accesso M (B-consolidation-003) · SCO:S12 SZ; S20 SN · D (getto), D-sec (fiamma) |
| A-measurements-scout-012a | u 2C nel piano di swirl inferiore; motore ottico TCC-III **trascinato** (senza combustione), 1300 rpm, 40 kPa (EngineBench `LSP_small`) | PIV → vettori 50 × 49 con padding a 128 × 128. MIS, 2 componenti | Pl; spaziatura ≈ 1,25–1,4 mm (SN). Vettori mancanti = **zeri esatti**: la maschera `u==0 \| v==0` non distingue lo zero vero. Lacune del benchmark artificiali (poligoni fino al 25 %) | Per fase (gradi di manovella) | **E:** «average velocity error 1,5–8 %», ripreso da Schiffmann et al. 2016; nessun indicatore per vettore. Finestra, overlap e validazione M | Pubblico, **non scaricato**; licenza da chiarire · INV:B-data-audit-006 (codice letto, RP); SCO:S27 RP · D/M |
| A-measurements-scout-012b | Motore ottico TCC-III **con combustione** (fired), deposizioni UMich Deep Blue | Da snippet: velocità PIV (2D2C in una deposizione), pressione nel cilindro, immagini OH\* | M | M | Contenuto, parametri, sincronizzazione PIV–OH\*–pressione, licenza: M | Solo snippet; **risorsa distinta** da 012a · SCO:S26 SN · M |
| A-PIV-update-001 | Velocità 3C nel piano; combustore DLR doppio swirl, etilene/acetone, 5 bar (configurazione distinta da A-scout-006) | Stereo-PIV, TiO₂ 0,5 µm nei due flussi d'aria; DaVis 8.4 multi-pass, finestra finale 24×24 px, overlap 50%. MIS | Pl; FOV circa 54×22 mm², risoluzione dichiarata 1,9 mm, passo 0,95 mm | 10 kHz, δt fra impulsi 10 µs, record 1 s | **E:** correlation statistics a posteriori: intervalli 0,1–1,1 m/s in piano, 0,3–2,3 m/s fuori piano; statistica, copertura e aggregazione M, nessuna validazione indipendente riportata. Non sono σ. **L:** parte dell'oscillazione fuori dal FOV in una condizione | D da preprint letto dallo scout, versione pubblicata non verificata; dati pubblici non individuati · M02, PIVL:L37 FT-v |
| A-PIV-update-002 | Velocità nel gas incombusto; flashback swirl CH₄/H₂ atmosferico, UT Austin | Stereo-PIV con goccioline di olio d'oliva ~1 µm; il bordo particelle/assenza di particelle è usato come marcatore. MIS | Pl; la copertura dipende dalla sopravvivenza del tracciante | Acquisizione a 4 kHz nel sistema descritto; dati pubblici M | **L:** olio che evapora nel preriscaldamento, niente velocità nei prodotti con questo tracciante nel caso descritto. ~555 K (std 10 K) è il valore riportato in quelle condizioni, non un'isoterma esatta né una soglia universale. Una maschera sintetica da T è un'ipotesi che introduce informazione sul target | D per l'apparato; I per una maschera LES. Sezioni del rapporto OSTI lette dallo scout; articolo Ebi & Clemens non letto · M02, PIVL:L08/L28 FT |

**Precisazioni comuni alla PIV, dalla consegna intermedia 02.** PIV1 e PIVL:L01–L02 distinguono passo, risposta spaziale e correlazione degli errori: i risultati multi-pass DaVis su immagini sintetiche non giustificano un filtro top-hat universale o rumore indipendente. PIVD:S2D-01/02 mostra perché distinguere immagini, vettori prima/dopo validazione, sostituzioni e riferimento. La mancanza di NaN nel prodotto finale non dimostra copertura originaria completa. I numeri di A-scout-006 sono stime a priori con copertura non definita nelle parti lette (PIVL:L04); A-PIV-update-001 è un esempio a posteriori distinto. Separazione fra impulsi PIV, intervallo fra campi e Δt LES sono tempi distinti: un modello su un solo snapshot approssima una misura istantanea. Una mappa UQ non fornisce gli errori realizzati né le loro covarianze. Parametri di apparati diversi non sono combinati. Dettagli e lacune nell'[aggiornamento PIV](updates/piv_2026-11-26.md).


### A.3 Pressione

| ID | Grandezza e contesto | Tecnica → segnale | Spazio | Tempo | Errori e limiti | Disponibilità · fonte · evidenza |
|---|---|---|---|---|---|---|
| A-consolidation-006 | p nel cilindro; motori a combustione interna (da A-measurements-scout-009, parte motore) | Trasduttore piezoelettrico → carica, pressione **relativa** → *pegging* per l'assoluto. MIS, scalare | Pt a parete | Per grado di manovella | **E:** thermal shock (deriva intraciclo), errori di pegging; entità M. Grandezze derivate per ciclo (IMEP, HRR apparente a zona singola, CA50): definizioni standard, non verificate su fonti lette (I) | Possibile ponte con dati motore (I) · SCO:S23 SN non attribuito · M |
| A-consolidation-007 | Pressione dinamica/acustica in combustori (da A-measurements-scout-009, parte combustore) | Microfoni o trasduttori remoti su linea semi-infinita → tensione. MIS | Pt a parete | kHz | **E:** funzione di trasferimento della linea fra pressione interna e lettura remota (SCO:S24a); incertezza della funzione di trasferimento di fiamma stimata da serie finite (S24b) | SCO:S24a–b SN · D-sec/M |

### A.4 Composizione, specie e particolato

| ID | Grandezza e contesto | Tecnica → segnale | Spazio | Tempo | Errori e limiti | Disponibilità · fonte · evidenza |
|---|---|---|---|---|---|---|
| A-measurements-scout-008 | **Segnale** OH-PLIF come marcatore di zona di reazione / gas combusto; DLR GTMC (stesso apparato di 006) | PLIF A-X (1,0) Q1(7) a 283,2 nm, ≈ 100 µJ/impulso a 5 kHz (≈ 2,5 mJ a 10 Hz nello stesso bruciatore), gate 500 ns → fluorescenza. MIS, segnale, **non** [OH] | Pl; foglio ≈ 0,4 mm, altezza ≈ 40 mm | 5 kHz, interlacciato con la PIV | **E:** qualitativo senza correzioni di quenching e frazione di Boltzmann (correggibili colpo per colpo con Raman/Rayleigh simultaneo, SCO:S07). Lettura degli autori: assenza di OH = gas sotto ≈ 1500 K | SCO:S01 TI; S07 SZ · D |
| A-jason-nie-002 | Segnale OH-PLIF [u.a.]; fiamme di Nie (JHC, toluene) | Eccitazione 283,58 nm, ICCD con filtro 310 nm; nessuna calibrazione né correzione del quenching. MIS, segnale | Pl centrale; foglio 15 mm × 0,3 mm | Singolo colpo, 1 Hz, simultaneo con nie-001 e nie-003 | Non quantitativo per [OH] (D); incertezza non riportata (M) | NIE:S01 TI pp. 3, 11 · D |
| A-jason-nie-003 | Frazione volumetrica di soot [ppm]; fiamme di Nie | LII planare a 1064 nm (~0,3 J/cm²), segnale prompt 435 ± 20 nm; calibrazione per estinzione in fiamma di Santoro (coefficiente 4,90). MIS, scalare | Pl; foglio 20 mm × 0,25 mm | Singolo colpo, 1 Hz | **E:** precisione ±10 %, **esclusa** l'incertezza del coefficiente di estinzione. Limite generale (SCO:S13 §3.2.2): la calibrazione con l'estinzione vale solo se E(m) è lo stesso nelle due condizioni | NIE:S01 TI p. 3; SCO:S13 SZ · D |
| A-measurements-scout-010 | Specie stabili e radicali; fiamme laminari premiscelate a 10–100 Torr | Sonda di quarzo (orifizio 250–500 µm) + MBMS → correnti ioniche per m/z; frazioni molari tramite sezioni d'urto di ionizzazione. MIS | Profilo 1D di punti lungo l'asse | Stazionario | **E-bias:** perturbazione termica fino a 500 K (150 K e ≈ 200 K in altre fiamme); **bias di posizione:** prelievo qualche diametro di orifizio a monte; sezioni d'urto dei radicali incerte | Laboratori di cinetica; non trasferibile tale e quale in turbolenza (I) · SCO:S09 SZ · D |
| A-consolidation-008 | T e f_v del soot (da A-measurements-scout-011, pirometria); motore diesel heavy-duty | Pirometria a due colori → emissione integrata. MIS | Pr | Per ciclo | **E-bias** verso il soot caldo: con gradienti di 600 K lungo la vista, T +200 K rispetto alla media pesata sulla massa e f_v −50 % (condizione HTC); circa la metà in LTC. Valutazione su **distribuzioni di soot simulate**, non misure 3D | SCO:S15 AB · D (abstract) |
| A-consolidation-009 | T del soot (da A-measurements-scout-011, DBI-TI); motore diesel ottico | Diffuse back-illumination extinction → mappe 2D di T soot integrate. MIS | Pr | M | Dichiarata più vicina alla media pesata sulla massa; incertezza con Monte Carlo | SCO:S14 AB · D (abstract) |

### A.5 Misure integrate lungo la vista e tomografie

| ID | Grandezza e contesto | Tecnica → segnale | Spazio | Tempo | Errori e limiti | Disponibilità · fonte · evidenza |
|---|---|---|---|---|---|---|
| A-measurements-scout-005 | T mediata sul percorso, P, densità colonna di CO e CO₂; fireball dopo detonazione di PETN (**non** una fiamma da bruciatore: analogo metodologico) | LAS: QCL su CO (≈ 2008,5 cm⁻¹) e ICL su CO₂ (≈ 2394,8 cm⁻¹) → spettro di assorbanza → fit spettroscopico; T del CO usata nel fit del CO₂. MIS | Pr: una linea, 83 mm sopra la carica; nessuna risoluzione lungo il percorso | 1 MHz (CO), 500 kHz (CO₂); allineamento dei tempi esperimento–CFD ≈ 2,5 µs | **F/E:** media lungo il percorso (T di CO e CO₂ differiscono al più di 200 K secondo la CFD); P «only an estimate»; bande sperimentali = range su 4 prove | SCO:S10 SZ; review S11 non letta · D |
| A-measurements-scout-013 | HRR integrale o locale dalla chemiluminescenza; fiamme turbolente magre premiscelate e parzialmente premiscelate (tesi Lauer, TUM) | Camere intensificate: OH\* (≈ 310 nm) + canale di fondo CO₂\* (≈ 456 nm); gamma 0,7 del firmware da correggere. MIS, intensità integrata | Pr: nessuna informazione normale al piano della camera; deconvoluzione solo per **medie** assisimmetriche (sulle istantanee resta un errore non eliminabile) | Medie su 2048 coppie di immagini; PIV a 1 kHz per la correzione | Intensità integrale ↔ HRR integrale; HRR **locale** non rappresentato in turbolenza senza correzione con modello di strain e velocità. Altri contesti (counterflow, sensore puntuale, SCO:S17) solo SN: differenza di applicabilità, non conflitto dimostrato | SCO:S08 SZ · D |
| A-consolidation-001 | Intensità OH\* **sperimentale**; fiamme swirl NH₃/CH₄/aria di RealPDEBench, stesse condizioni operative nominali delle LES | Chemiluminescenza OH\* ad alta velocità → `observed (T,H,W)`, 1 canale; unità, scala, filtro, esposizione: M | Pr per principio della tecnica (I); 128 × 128 dichiarata; FOV, orientamento e rapporto con la sezione LES: M | 2001 frame dichiarati; card «4000 Hz per 1 s» incompatibile con 2001 × 0,25 ms = 0,5 s; «2000 fps» da sintesi (SN); `t` reali non letti | Non caratterizzato. Nessuna registrazione campo per campo con la LES | **Dichiarato sull'hub, non posseduto** · INV:B-data-audit-002 (metadati LC; paper RP/SN) · D (esistenza, schema), M (resto) |
| A-consolidation-004 | Campo 3D di **emissione di chemiluminescenza** (intensità emessa per unità di volume) | Tomografia della chemiluminescenza (CTC): molte viste integrate → ricostruzione. MIS | Vol da Pr | Istantaneo | Qualità legata a numero e disposizione delle viste; validazione su fantasmi (M). **Ricostruisce l'emissione, non T né HRR** | SCO:S22 SN · D-sec (target), M (resto) |
| A-consolidation-005 | Campo 3D di **indice di rifrazione** → densità → T solo con ipotesi su indice–composizione | BOS tomografico: spostamenti apparenti di uno sfondo da molte viste. MIS | Vol da Pr | Istantaneo | T dedotta tramite ipotesi; numero di viste M | SCO:S21 SN · D-sec/M |

### A.6 Grandezze derivate

| ID | Grandezza e contesto | Tecnica → segnale | Spazio | Tempo | Errori e limiti | Disponibilità · fonte · evidenza |
|---|---|---|---|---|---|---|
| A-measurements-scout-014 | HRR locale [W/m³]; fiamme CH₄/aria premiscelate magre (φ = 0,7), Ka fino a 253; **DNS** Bunsen e round jet | Inversione fisica: T + 7 specie maggiori (+OH) "misurabili con Raman/Rayleigh" → reattore omogeneo vincolato → stato completo → HRR. Osservazioni **imposte sulla DNS**, non misurate | Ln (in pratica) | Singolo colpo | Bunsen, 225 campioni agli zero-crossing del modo esplosivo: errore relativo medio locale dell'HRR 70,83 % senza OH e 14,24 % con OH; integrato sui profili entro 1 %. Round jet: 62,59 %/12,91 % locale, 10,89 %/0,51 % integrato. Errori locali che si compensano | Valutazione solo numerica · SCO:S07 SZ · D |

### A.7 Osservazioni sintetiche e proxy (non diagnostiche sperimentali)

| ID | Grandezza e contesto | Tecnica → segnale | Spazio | Tempo | Errori e limiti | Disponibilità · fonte · evidenza |
|---|---|---|---|---|---|---|
| A-jason-liu-001 | T [K] su sezioni piane; DNS di fiamme premiscelate CH₄/aria (casi L, M, H, S) | **Virtuale:** valore della DNS campionato sul piano. Tecnica sperimentale imitata non specificata. SIN, scalare. **Sezioni di campi DNS, non proiezioni ottiche** | Pl; 8 piani per direzione, 2 famiglie (16 piani); 128² o 360² px, spaziatura ≈ griglia DNS (24,5–33 µm) | Istantaneo, tutti i piani simultanei | Rumore, filtro, spessore del foglio, bias e maschere **non descritti né valutati** (assenza probabile, non dichiarata) | Solo nel paper · LIU:S01 TI pp. 3, 6 · D/I/M |
| A-jason-liu-002 | u [m/s] su sezioni piane, stessi casi | **Virtuale:** valore DNS; PIV citata solo qualitativamente. Componenti per piano: presumibilmente 3 (I/M). SIN | Gli stessi 16 piani, co-localizzati con liu-001 | Come liu-001 | Filtro di finestra, rumore, outlier, vettori mancanti non valutati | LIU:S01 TI pp. 5–6 · D/I/M |
| A-consolidation-002 | `observed` nei file numerici di RealPDEBench: intensità «surrogate-mapped» | Mappatura dalla simulazione non descritta nelle fonti lette. SIN/proxy. **Non** è OH\* sperimentale né il canale 8 (frazione molare di OH) | Griglia 128 × 128 della LES; piano o proiezione: M | 2001 frame, come la LES | Contenuto mai caratterizzato (solo scansione NaN/Inf) | Presente localmente, non caratterizzato · INV:B-data-audit-003 · D (esistenza), M (metodo) |
| A-consolidation-003 | Velocità planare 2C (vi, vj) campionata dalla LES RealPDEBench in siti sparsi: osservazioni di progetto della tesi | Valori LES ai siti di maschere sparse (1/5/20 % = 164/819/3277 siti), con eventuali degradazioni sintetiche dichiarate. SIN, 2 componenti; vk assente nelle cache | Pl della sezione LES; orientamento fisico e segno delle componenti M | Frame singoli (A38: ogni 40 indici; C149: ogni 10) | **Non è una PIV:** filtro di finestra, lacune e errori correlati esistono solo se aggiunti sinteticamente. La campagna attuale è uno studio sintetico preliminare, non una validazione PIV | Locale (cache TRAIN/VAL) · INV:B-data-audit-004 LC; diario 25/11 · D |

**Famiglie censite senza riga** (solo pista, nessuna configurazione abbastanza precisa): CARS (SCO:S29 SN), fosfori termografici in aerosol (SCO:S19 SN), LDV puntuale (Sydney swirl in INV:B-data-audit-007, RP), analizzatori dei fumi (non coperto), calorimetria O₂ e flussimetri radiativi per HRR globale (contesto di SCO:S04), prodotto OH·CH₂O PLIF come indicatore di HRR (SCO:S07 §1, senza apparato), tomografia ad assorbimento (review non letta).

---

## 3. Vista B — dataset e configurazioni

27 righe. Dettagli in [data_inventory.md](data_inventory.md) (B-data-audit-…), nelle schede dei paper (B-jason-…) e nello scouting §7 (B-consolidation-…, non schede ispezionate). Le sette nuove righe B-PIV-S2D-… rimandano a PIVD (fonti fissate in §9), senza copiare i payload. **Una risorsa citata o un campione ispezionato da uno scout non è un dataset posseduto sul computer di Enrico.**

| ID | Dataset/configurazione | Provenienza: natura · regime · trattamento | Variabili | Spazio e tempo | Struttura/formato | Accesso | Riferimenti utilizzabili · lacune principali | A/C |
|---|---|---|---|---|---|---|---|---|
| B-data-audit-001 | RealPDEBench combustion, campi numerici, 30 condizioni NH₃ 0–80 % × φ (copertura non cartesiana), hub v2.0.1 | NUM; LES 3D implicita instazionaria (STAR-CCM+, EDC); swirl turbolento; **regime di miscelazione non determinato**; stati istantanei su **una sezione esportata** | 15 canali: P, HRR, 7 frazioni molari, T, Velocity[i/j/k], ch14 (≠ modulo euclideo). Unità non dichiarate; K e m/s inferiti | 128 × 128; x ∈ [−36,9; 36,9] mm, y ∈ [62,9; 159,1] mm; Δx ≈ 0,581 mm, Δy ≈ 0,758 mm. 2001 frame, Δt = 0,25 ms (0,5 s); finestre assolute diverse fra condizioni | Arrow IPC; 1 riga = 1 traiettoria = 1 shard (≈ 2,1 GB); `numerical (T,H,W,15)` float32; `x, y, t` float64 1D | LOC-I (30 shard; `t` letto su 23 non-TEST) | T e u LES come riferimento **numerico**, co-localizzati per costruzione. **Nessun volume 3D.** Piano, orientamento fisico, definizione e segno di i/j/k, SGS, mesh, interpolazione: M | A-cons-002/003; C-cons-006/007/009/010 |
| B-data-audit-002 | RealPDEBench, OH\* sperimentale, 30 traiettorie in 8 shard | MIS; stesse condizioni operative nominali | 1 canale, intensità OH\* | 128 × 128 e 2001 frame dichiarati; `t`, FOV, geometria non verificati | Arrow; ≈ 3,93 GB (3,66 GiB) dall'inventario remoto | **DICH** (assente in locale) | Confronto con la LES possibile solo statistico/spettrale/integrato: **stesso scenario, non stesso stato** | A-cons-001; C-cons-008 |
| B-data-audit-003 | `observed` surrogate nei file numerici | SIN | 1 canale | Come 001 | Primo array della riga | **LOC** (non caratterizzato) | Mappatura e unità M | A-cons-002; C-cons-009 |
| B-data-audit-004 | Cache di progetto A38 / C149 e maschere sparse | Derivato da 001, solo TRAIN/VAL | vi, vj normalizzate, T in K, p = (NH₃, φ) normalizzati; **vk assente** | 128 × 128; frame ogni 40 (A38) o ogni 10 (C149) indici | `.npy` float32 `(N,128,128)`, manifest in Git | LOC-I (payload gitignored) | Il flag di associazione campo–tempo registra l'accettazione del loader, non una verifica indipendente | A-cons-003; C-cons-006 |
| B-data-audit-005 | Traiettoria anomala `40NH3_1.2` (shard 22) | Come 001 | Come 001 | `t` di 2000 campioni con passo doppio a 1098→1099 (LC); frame 2000 = 1999 (RP, non rieseguito) | Come 001 | LOC-I | Policy attuale: singolo frame per indice consentito; `field_time_association_verified=False` per tutti gli indici; multi-frame/forecasting rifiutati (F3: verificato nel codice). Dichiarare la lacuna non risolve l'associazione | C-cons-006/007 |
| B-data-audit-006 | EngineBench, TCC-III **motored** | MIS, PIV senza combustione | u, v | 50 × 49 vettori (padding 128 × 128); piano di swirl inferiore; 5205 snapshot `LSP_small`; fase in CAD | Kaggle/GitHub | **DICH** (pubblico, non scaricato); licenza da chiarire | Nessun riferimento termico o scalare; il riferimento è la PIV stessa | A-scout-012a; C-cons-011 |
| B-data-audit-007 | Archivi TNF/Sandia swirl (Sydney swirl, TECFLAM) | MIS; fiamme non premiscelate | LDV puntuale 2–3C; Raman/Rayleigh/LIF; campagne separate | Punti e profili | File separati | **CIT** (TECFLAM su richiesta); non scaricati | Nessun dataset letto con V e T simultanei e co-localizzati a singolo colpo | A-scout-002 |
| B-data-audit-008 | Dati motore PoliMi (RANS/URANS) | NUM attesa; contenuto ignoto | — | — | — | **NESS** (nel perimetro consultato) | Un target RANS è un campo medio, non istantaneo (I) | C-cons-012 |
| B-data-audit-009 | Casi laminari e campi medi (voce iniziale) | — | — | — | — | Nessun archivio laminare nuovo ispezionato in questa integrazione; per **medie sperimentali PIV** vedi ora B-PIV-S2D-03 | La voce iniziale non copre lo scouting PIV aggiunto in §9 | — |
| B-jason-nie-001 | Adelaide JHC: immagini simultanee T/OH/SVF, 3 condizioni (Re, YF) | MIS; getti turbolenti non premiscelati con soot | T nTLAF [K] filtrata; segnale OH-PLIF [u.a.]; SVF [ppm] | Pl centrale; 789 × 113 px a 0,089 mm/px; finestre HAB di 10 mm fra 241 e 306 mm; scatti indipendenti a 1 Hz. Usate: Set-I 1707 train / 200 test; Set-II altre 2 × 798 train e 400 test | M | **RICH** («upon a reasonable request»; nessuna richiesta inviata) | T sperimentale simultanea e co-planare, ±120 K (definizione M); co-registrazione M | A-nie-001…003; C-nie-001 |
| B-jason-liu-001 | DNS L/M/H, fiamme premiscelate planari in turbolenza forzata (S3D) | NUM, DNS; CH₄/aria φ = 0,7, 300 K, meccanismo a 2 passi; Ka 32–280 | T, u usate; altre non descritte | 768 × 128 × 128 a 24,5 µm; sottodominio 5,0 × 3,14 × 3,14 mm³; 120 snapshot (M); Δt M | M | **CIT** (disponibilità non dichiarata nel testo) | Volume 3D esatto rispetto al modello numerico | A-liu-001/002; C-liu-001/002 |
| B-jason-liu-002 | DNS caso S, fiamma premiscelata swirlata con pareti | NUM, DNS; CH₄/aria φ = 0,7, 800 K; U = 100 m/s, S = 1 | T, u | 480³ su 16³ mm³ (≈ 33 µm); sottodominio 12³ mm³; 120 snapshot; Δt M | M | **CIT** | Come B-jason-liu-001 | A-liu-001/002; C-liu-001/002 |
| B-consolidation-001 | Deposizioni UMich Deep Blue, TCC-III **fired** | MIS (da snippet) | Velocità PIV, pressione nel cilindro, immagini OH\* (attribuzione alle singole pagine incerta) | M | M | **CIT** (solo snippet; nulla ispezionato) | Sincronizzazione PIV–OH\*–pressione M | A-scout-012b, A-cons-006 |
| B-consolidation-002 | DLR GTMC: PIV/OH-PLIF 5–10 kHz; altre campagne con LDV e Raman | MIS; swirl CH₄/aria | u 3C, segnale OH-PLIF; LDV e Raman in campagne citate | Vedi A-scout-006 | M | **CIT** (accesso M) | Nelle parti lette nessuna T simultanea alla PIV | A-scout-006/008 |
| B-consolidation-003 | Tomo-PIV del getto Sandia (ref. di SCO:S12) | MIS, non reattivo | u 3C | Vedi A-scout-007 | M | **CIT** | — | A-scout-007; C-scout-004 |
| B-PIV-S2D-01 | Cardiff, Runyon et al. | MIS; PIV **isoterma** in bruciatore swirl; OH-PLIF/OH*/pressione in altre acquisizioni | Immagini, maschera, vettori a tre stadi e codici 0/1/16/17 | 514 coppie dichiarate; scout: 4 frame × 3 stadi, 81×61 vettori; scala fisica M | DynamicStudio binario .NET, XML | **SCOUT-I**, CC BY 4.0 | Semantica codici M; 16 interpretato come sostituzione, non certificato. Decoder non ufficiale. Nessuna UQ o T simultanea · PIVD:S2D-01 | A.2; C-PIV-update-001 |
| B-PIV-S2D-02 | Neal et al., PIV-UQ | MIS; getto rettangolare non reagente | Immagini MS 2C + riferimento stereo HDR con flag | Simultanei e co-localizzati; HDR con correlazione su **5 coppie**, mappatura e interpolazione | TIF MS; Tecplot ASCII HDR; unità px MS/intervallo | **SCOUT-I**, CC0; campioni, non archivio intero | Vettori MS da produrre. Riferimento più accurato, con filtro e incertezza propri; non verità puntuale istantanea · PIVD:S2D-02 | A.2; C-PIV-update-002 |
| B-PIV-S2D-03 | Darmstadt MRB, Li et al. | MIS; fiamme CH₄/H₂ | Medie di u,v; RMS e OH medi dichiarati non ispezionati | 18 file di medie letti; unità coordinate M | MATLAB .mat, mappe e coordinate | **SCOUT-I**, CC BY-NC 4.0 | Layout di medie; non lacune/rumore istantanei. Mancanti: nei file H₂ vx=0 e vy=NaN sulle stesse celle · PIVD:S2D-03 | A.2 |
| B-PIV-S2D-04 | RealPDEBench **fluidi**, cylinder in acqua | MIS; non reagente, distinto da combustion | u,v,vo,x,y,t; nessun flag/UQ nello schema | Campione 64×128, Δt file 5 ms contro 2,5 ms documentati | Arrow, lettura parziale di uno shard | **SCOUT-I** parziale; CC BY-NC 4.0 da audit precedente | Nessun NaN/zero nei frammenti letti non dimostra assenza di riempimenti. Nessuna modifica all'audit temporale della LES combustion · PIVD:S2D-04 | A.2 |
| B-PIV-S2D-05 | PIV Challenge | MIS/SIN secondo il caso | Immagini e istruzioni di benchmark | Configurazione da scegliere | Vari; istruzioni consultate | **DICH**, payload non ispezionati; truth sintetica su richiesta non ottenuta; licenza M | 0/1/2 = convenzione di sottomissione, non flag verificati nei dati rilasciati · PIVD:S2D-05 | A.2 |
| B-PIV-S2D-11 | Yang et al. 2020, fiamma swirl | MIS; piani stereo-PIV traslati | u,v,w e vorticità | File 26×61×81; simultaneità, istantaneo/media e unità M | Un file Tecplot letto integralmente | **SCOUT-I**, CC BY 4.0 | Zeri non qualificati; non è dimostrato un volume istantaneo di riferimento; paper non letto · PIVD:S2D-11 | A.2 |
| B-PIV-S2D-35 | DaRUS RQL, Jose/Geigle/Hampp | MIS, frammenti d'immagine da combustore fuligginoso; codice di segmentazione | Immagini di traccianti/soot, **nessun campo vettoriale** | Parametri di misura M | ZIP/PNG e codice, solo metadati API letti | **DICH**, guestbook non compilato; CC BY 4.0/MIT secondo record | Pista per fenomeni d'immagine, non per percentuali di rigetto vettoriale/UQ · PIVD:S2D-35 | A.2 |
| B-TH-D01 | HYLON TNF Phase 2-1, gennaio 2026 | MIS; swirl confinato H₂/aria, Flame A.1/L.1 | TC estrapolata, Raman media/RMS, LIP pareti, PIV media/RMS, CAD | Traversate/profili, campagne separate; **non serie simultanee** | CSV/TXT, PDF, CAD, immagini | **SCOUT-I**; licenza riuso non esplicita, restrizione NO-LIF nel PDF | Layout/statistiche, non validazione istantanea PIV–T; diametri, lega e raw TC M · THD:D01 | A-TH-001/004; C-cons-007 |
| B-TH-D02 | NIST RSE 2007 | MIS; compartimento d'incendio | TC nude/aspirate, pareti, velocità da sonde bidirezionali | Δt 30 s; coppie TC a stesse coordinate nominali, velocità sfalsate 9–10 cm | CSV, documentazione/FDS | **SCOUT-I** | Confronto fra strumenti, aspirata non assunta esatta; non dinamica rapida/PIV. Anomalia osservata non equivale a guasto dimostrato · THD:D02 | A.1; C-scout-001 |
| B-TH-D04 | NIST Pool Fires/MaCFP | MIS e stime da modello; più fiamme di pozza | TC/TC_RMS misurati; TG/TG_RMS compensati | Profili medi/RMS gas; nessuna serie raw gas depositata nel materiale letto | 13 CSV campionati | **SCOUT-I** | Correzioni non riferimento indipendente; confronto diametri confuso da condizioni/traverse · THD:D04 | A-scout-001; C-scout-001 |
| B-TH-D09 | Waterloo Methanol/MaCFP | MIS; pool fire | TC S 50 µm + LDA 2C, momenti congiunti velocità–T | Acquisizione simultanea 125 Hz, 40 000 campioni; **solo statistiche rilasciate** | CSV per quota; compositi con righe fuse | **SCOUT-I** | Compensazione inerziale già applicata; può confrontare statistiche congiunte, non stati istantanei · THD:D09 | C-scout-001; C-cons-007 |
| B-TH-D06 | Katurji in-fire | MIS; configurazione diversa dal combustore swirl | Serie TC e NaN osservati | 50 Hz nel dataset descritto | Campioni ispezionati dallo scout | **SCOUT-I** | Variabilità fisica, inerzia e rumore non separati; non identifica da sola τ o rumore elettronico · THD:D06 | C-scout-001 |


**Riferimenti e simultaneità: sintesi.** Neal aggiunge un riferimento sperimentale simultaneo di **velocità**, con supporto e incertezza propri (C-PIV-update-002). Fra le risorse censite, B-jason-nie-001 contiene un riferimento termico **sperimentale** simultaneo e co-localizzato con altre modalità, ed è disponibile solo su richiesta. Waterloo aggiunge statistiche congiunte da LDA–TC simultanee; HYLON profili termici e PIV di campagne separate; NIST RSE serie lente con velocità da sonde sfalsate. Non forniscono una nuova serie T–PIV simultanea e co-localizzata per validare la ricostruzione istantanea nel combustore swirl. I riferimenti numerici (B-data-audit-001, B-jason-liu-…) sono simultanei per costruzione, ma input e target provengono dallo stesso modello. Per RealPDEBench, LES e OH\* sperimentale condividono la condizione operativa, non l'istante.

---

## 4. Catalogo C — problemi

20 problemi. I due nuovi C-PIV-update-… riguardano la qualificazione di misure/elaborazioni, non nuovi training di ricostruzione. Nessuno è classificato come "migliore per la tesi": il confronto spetta a F4. Le righe C-consolidation-006…012 formulano problemi già presenti nei rapporti (inventario §5 e §9, schede §11, diario del 25/11) collegandoli ai dati posseduti; non sono nuove proposte di esperimenti.

### 4.1 Vista d'insieme

| ID | Osservazioni → target | Dati | Validazione documentata | Stato |
|---|---|---|---|---|
| C-jason-nie-001 | OH-PLIF + SVF planari → T planare istantanea | MIS | VS (stesso stato) + VST | Dimostrato nello stesso apparato; uso del target a inferenza ambiguo |
| C-jason-liu-001 | 16 sezioni ideali di T → T 3D | NUM/SIN | VN | Dimostrato su DNS ideale |
| C-jason-liu-002 | 16 sezioni ideali di u + T 3D ricostruita → u 3D | NUM/SIN | VN, VST, VC | Dimostrato su DNS; vincolo fisico da chiarire |
| C-measurements-scout-001 | Serie da termocoppie lente → fluttuazioni o statistiche di T gas | MIS | Propagazione degli errori; VS non trovata | Proposta |
| C-consolidation-001 | Proiezione OH\* (+ velocità) → HRR integrale o mediato | MIS | VST (solo in media) | Documentato per l'integrale |
| C-consolidation-002 | Piani PLIF OH (+CH₂O) → indicatore di HRR locale | MIS | M | Evidenza sottile |
| C-consolidation-003 | Linee Raman/Rayleigh (+OH) → HRR locale lungo la linea | SIN su DNS | VN | Dimostrato solo su DNS |
| C-measurements-scout-003 | Poche misure line-of-sight → profili o statistiche di T/specie | MIS + SIN | Confronto con misure sintetiche (non indipendente) | Proposta |
| C-consolidation-004 | Molte viste di chemiluminescenza → emissione 3D (CTC) | MIS | M (fantasmi) | Pista |
| C-consolidation-005 | Molte viste BOS → indice di rifrazione 3D (→ densità, T) | MIS | M | Pista |
| C-measurements-scout-004 | PIV/tomo-PIV sparse → stato LES completo (assimilazione) | MIS + NUM | VS parziale (dati non assimilati) | Dimostrato su getto inerte |
| C-consolidation-006 | Velocità 2C sparse sintetiche → T sulla sezione LES | SIN/NUM | VN | Problema attuale della tesi |
| C-consolidation-007 | Velocità sparse + poche T puntuali lente → T o livello termico | SIN/NUM | VN | Da formulare (missione 04); requisito temporale aperto |
| C-consolidation-008 | OH\* sperimentale RealPDEBench → HRR/statistiche termochimiche | MIS + NUM | VST (stesso scenario, non stesso stato) | Dati non posseduti |
| C-consolidation-009 | `observed` surrogate → T o HRR sulla sezione LES | SIN/NUM | VN | Mappatura del proxy ignota |
| C-consolidation-010 | Frazione molare OH numerica → T o HRR sulla sezione LES | NUM | VN | Analogo sintetico di Nie, senza modello di misura |
| C-consolidation-011 | PIV in motore con lacune → velocità nelle zone mancanti | MIS | Contro la PIV stessa (non indipendente) | Benchmark esistente, non scaricato |
| C-consolidation-012 | Dati RANS/URANS PoliMi → ? | NUM | M | Non formulabile oggi |
| C-PIV-update-001 | Immagini/vettori Cardiff a più stadi → storia delle lacune e modifiche del processamento | MIS isoterma | Confronto fra stadi, nessuna verità indipendente | Candidato per verifiche della 02, protocollo non deciso |
| C-PIV-update-002 | Immagini MS Neal + riferimento HDR → residui e valutazione UQ | MIS non reagente | Riferimento simultaneo più accurato, con incertezza e filtri propri | Candidato per verifiche della 02, processamento MS da definire |

### 4.2 Schede

**C-jason-nie-001 — T planare istantanea da OH-PLIF e soot (sperimentale)**
- *Bisogno:* termometria planare istantanea, anche ad alta velocità, in fiamme turbolente con soot dove FRS/TLAF sono limitati (NIE:S01 pp. 1–2, D).
- *Osservazioni durante l'uso:* segnale OH-PLIF non calibrato + SVF da LII, simultanei e co-planari; nessun parametro operativo. **Statistiche di T per la de-standardizzazione ambigue** (NIE §6: p. 6 contro pp. 7–8): non si può stabilire se a test entri la media per immagine della T misurata. Né leakage dimostrato né assenza di leakage.
- *Target e scala:* T istantanea [K] sullo stesso piano, 70 × 10 mm; risoluzione effettiva ≈ 0,5 mm (target filtrato).
- *Approcci documentati:* U-Net con instance normalization (D); GAN/cGAN peggiori (nei supplementari, M). Diagnostiche dirette: FRS, TLAF, 2D-CARS.
- *Riferimento e validazione:* **VS, stesso stato:** nTLAF simultanea, tecnica diversa dagli input. Std delle deviazioni ≈ 57 K in distribuzione (Tab. 2); 136–140 K su due condizioni escluse (Tab. 3), con test confuso da cambi di altezza e, in un caso, di Re e YF insieme; 53–58 K se quelle condizioni entrano nel training (Tab. 5, non generalizzazione). **VST:** PDF e PDF congiunte, che non validano le strutture istantanee. Il confronto 57 K / ±120 K **non** consente un limite sul rumore del riferimento senza ipotesi non verificate (NIE §7.1). Validation set e baseline semplici non descritti.
- *Mancante:* supplementari (URL dell'articolo ora corretto, supplementari non recuperati); coppia (μ, σ) usata a test; definizione dei ±120 K; co-registrazione delle camere; prova su dati reali ad alta velocità. Per noi: B-jason-nie-001 solo su richiesta; i 15 canali RealPDEBench non contengono OH-PLIF né grandezze soot (INV:B-data-audit-001); se la fiamma NH₃/CH₄ produca soot: M.
- *Collegamenti:* A-jason-nie-001…003; B-jason-nie-001; schema riutilizzato da C-cons-010; missione [05](../05_optical_diagnostics/README.md).

**C-jason-liu-001 — T 3D da sezioni piane di T (DNS)**
- *Bisogno:* dati 3D di T in fiamme turbolente; diagnostiche 3D limitate e costose (LIU:S01 pp. 1–2, D).
- *Osservazioni durante l'uso:* 16 sezioni **ideali** di T (8 + 8 piani) nelle stesse posizioni del training, stessa configurazione di fiamma; normalizzazione M. Sono sezioni di campi DNS, non proiezioni ottiche.
- *Target e scala:* T istantanea 3D [K] su 5,0 × 3,14 × 3,14 mm³ (planari) o 12³ mm³ (S); risoluzione dell'output M.
- *Approcci documentati:* VQ-VAE + diffusione discreta (adattato da Corona-Figueroa et al. 2023). Su un operatore diverso (proiezioni): CTC, NeRF/NVRT, citati (vedi C-cons-004).
- *Riferimento e validazione:* **VN** contro la DNS volumetrica esatta; indipendente dagli input solo fuori dai piani. Con 8 piani per direzione, L2 relativo di T ≈ 2,0 % (caso M) e ≈ 4,8 % (caso S): **letture grafiche** della Fig. 6, incertezza di lettura ≈ ±0,3 punti. Split casuale di 120 snapshot consecutivi (rischio di interpolazione temporale; Δt M). L/H solo qualitativi; nessuna baseline numerica.
- *Mancante:* volumi 3D di training della stessa configurazione; modello di misura delle sezioni; fattibilità sperimentale di 16 piani simultanei (citazioni [47–49], di tomografia di chemiluminescenza, non lette). Per noi: RealPDEBench esporta una sola sezione 2D, quindi **nessun riferimento volumetrico** (INV:B-data-audit-001; chiude la verifica rinviata in LIU §11).
- *Collegamenti:* A-jason-liu-001; B-jason-liu-001/002.

**C-jason-liu-002 — u 3D da sezioni di u e T 3D ricostruita (DNS, PINN)**
- *Osservazioni durante l'uso:* u sulle 16 sezioni (componenti presumibilmente 3, cioè equivalenti a una PIV stereo: I/M); T 3D ricostruita da C-jason-liu-001; p = 0,1 MPa e W = 28 g/mol costanti; μ(T); coordinate e tempo.
- *Target e scala:* u(x, y, z, t) istantanea sul sottodominio.
- *Approcci documentati:* WN-PINNs con separazione fra basso e alto numero d'onda; PINN standard citati come peggiori, senza numeri.
- *Riferimento e validazione:* **VN:** con 8 piani per direzione, L2 relativo ≈ 4–5 % (M) e ≈ 5,5–7,5 % (S) per componente, da lettura grafica della Fig. 6 (incertezza ≈ ±1–2 punti). **VST:** medie condizionate a T per L/H. **VC:** residuo della quantità di moto; un residuo basso non prova lo stato nascosto.
- *Mancante:* trattamento di pressione dinamica e continuità (l'Eq. 13 stampata non le mostra: richiesta di chiarimento, non difetto accertato); supporto temporale di una "testing instance" (uno o più snapshot: non descritto); robustezza a osservazioni 2D2C, filtro PIV e rumore. Costo: 20 min di A100 per istanza.
- *Collegamenti:* A-jason-liu-002; B-jason-liu-001/002; C-jason-liu-001; missioni [02](../02_virtual_piv/README.md) e [04](../04_thermal_fusion/README.md) (propagazione di un errore su T in un vincolo fisico).

**C-measurements-scout-001 — Fluttuazioni di T del gas da termocoppie lente**
- *Bisogno:* le termocoppie sono diffuse, ma la risposta lenta altera le fluttuazioni misurate (A-scout-001).
- *Osservazioni durante l'uso:* serie di T del giunto da uno o pochi giunti; geometria del giunto; velocità locale stimata (in SCO:S04 presa da una simulazione FDS).
- *Target e scala:* T del gas puntuale con le sue fluttuazioni, oppure statistiche (std, PDF).
- *Approcci documentati:* correzione inerziale da bilancio energetico (S04); compensazione a due fili (citata, non letta); stima statistica o appresa (I).
- *Riferimento e validazione:* nel caso letto, propagazione degli errori e verifica del codice su una sinusoide e su FDS; nessuna termometria ottica simultanea. Possibili: VS con Rayleigh/CARS co-localizzati (non trovata); VN con sensore sintetico su DNS/LES.
- *Mancante:* coppie sperimentali termocoppia–ottica; τ e ruolo della conduzione nelle condizioni di interesse (τ dipende da diametro, velocità e T).
- *Collegamenti:* A-measurements-scout-001; C-cons-007; missione [04](../04_thermal_fusion/README.md).

- *Aggiornamento 04:* THM §2 distingue bilancio non lineare e primo ordine approssimato; THD:D04 rilascia TG/TG_RMS da modello, quindi non ne valida indipendentemente la risposta. Waterloo (D09) offre statistiche congiunte da misure simultanee, non serie raw; Katurji (D06) serie con mancanti, ma non separa fisica e rumore. Il confronto resta da progettare nella 04.

**C-consolidation-001 — HRR da proiezione di chemiluminescenza OH\*** *(da C-measurements-scout-002, osservazione OH\*)*
- *Osservazioni durante l'uso:* immagini OH\* da una vista, con canale CO₂\* per il fondo; per la correzione di Lauer anche velocità PIV (strain).
- *Target e scala:* HRR integrale [W], istantaneo o medio; distribuzione **mediata** sulla sezione mediana per deconvoluzione assisimmetrica. HRR locale istantaneo non rappresentato senza correzione.
- *Approcci documentati:* proporzionalità integrale; correzione con modello di strain (SCO:S08). Rapporti OH\*/CH\* in altri contesti (SCO:S17, SN).
- *Riferimento e validazione:* **VST:** tecnica di riferimento costruita con chemiluminescenza, PIV, OH-PLIF e bilancio di entalpia, con ingressi non simultanei: **validazione solo in media** (S08, cap. 5).
- *Mancante:* full text di Hardalupas & Orain 2004 e Ayoola et al. 2006; dati con OH\* e campo di riferimento co-registrati.
- *Collegamenti:* A-scout-013; A-cons-001; C-cons-008; missione [05](../05_optical_diagnostics/README.md).

**C-consolidation-002 — Indicatore di HRR locale da piani PLIF (OH, CH₂O)** *(da C-measurements-scout-002, osservazione PLIF)*
- *Osservazioni durante l'uso:* segnali PLIF di OH ed eventualmente CH₂O sullo stesso piano, simultanei.
- *Target e scala:* HRR locale istantaneo su un piano, come indicatore.
- *Approcci documentati:* prodotto OH·CH₂O come marcatore (SCO tabella §3; S07 §1).
- *Riferimento e validazione:* senza correzione colpo per colpo di quenching e frazione di Boltzmann (possibile con Raman/Rayleigh simultaneo) il prodotto non è una misura quantitativa (S07 §1). Nessuna validazione specifica letta: M.
- *Mancante:* fonte primaria sulla validazione dell'indicatore; nessun PLIF nei dati censiti posseduti.
- *Collegamenti:* A-scout-008; A-jason-nie-002; missione [05](../05_optical_diagnostics/README.md).

**C-consolidation-003 — HRR locale da linee Raman/Rayleigh (+OH) con reattore vincolato** *(da C-measurements-scout-002, osservazione Raman)*
- *Osservazioni durante l'uso:* T e 7 specie maggiori, con o senza OH, lungo una linea a singolo colpo.
- *Target e scala:* HRR locale lungo la linea; HRR integrato sul profilo.
- *Approcci documentati:* reattore omogeneo a P e T costanti con specie maggiori vincolate (SCO:S07).
- *Riferimento e validazione:* **VN** solo su DNS (Ka fino a 253) con incertezze imposte. Errori in A-scout-014: locale 70,8 % senza OH, 14,2 % con OH (Bunsen); integrato entro 1 %; errori locali che si compensano.
- *Mancante:* risultati della §5 di S07 (analisi delle incertezze sperimentali) non estratti; validazione sperimentale; apparato Raman dedicato (A-scout-002).
- *Collegamenti:* A-scout-014, A-scout-002.

**C-measurements-scout-003 — Profili o statistiche di T/specie da poche misure line-of-sight** *(ambito ristretto in F3: le tomografie sono in C-cons-004 e 005)*
- *Osservazioni durante l'uso:* una o poche linee di assorbimento (LAS), intensità di chemiluminescenza o pirometria; geometria dei fasci; tempi.
- *Target e scala:* profilo, campo o statistiche lungo il percorso di T o specie.
- *Approcci documentati:* confronto nello spazio delle misure con misure **sintetiche** calcolate integrando la CFD lungo la stessa linea (SCO:S10).
- *Riferimento e validazione:* il confronto sintetico–sperimentale qualifica **insieme** misura e modello (accordo «reasonable» a tempi brevi); non è un riferimento indipendente.
- *Mancante:* review S11 e Cai & Kaminski; operatore pesato dallo stato e non lineare (pirometria, I da S14–S15); limiti di accesso ottico.
- *Collegamenti:* A-scout-005, A-scout-013, A-cons-008/009.

**C-consolidation-004 — Emissione 3D di chemiluminescenza da molte viste (CTC)** *(da C-measurements-scout-003)*
- *Osservazioni durante l'uso:* molte proiezioni simultanee di chemiluminescenza.
- *Target e scala:* emissione per unità di volume, istantanea. **Non è T né HRR:** un'interpretazione termochimica richiede un passaggio ulteriore (cfr. C-cons-001).
- *Approcci documentati:* CTC (Floyd & Kempf 2011; Unterberger et al.); NeRF/NVRT citati da Liu come approcci diversi dal suo.
- *Riferimento e validazione:* fantasmi numerici (SN, M).
- *Mancante:* tutto oltre gli snippet. Per l'OH\* di RealPDEBench, se confermato a vista singola, servirebbero più viste o ipotesi di simmetria (LIU §11, I/M).
- *Collegamenti:* A-cons-004, A-cons-001.

**C-consolidation-005 — Indice di rifrazione 3D da BOS tomografico** *(da C-measurements-scout-003)*
- *Osservazioni durante l'uso:* spostamenti apparenti di uno sfondo da molte viste.
- *Target e scala:* indice di rifrazione 3D istantaneo; densità e T solo con ipotesi sulla relazione indice–composizione.
- *Approcci documentati:* BOS tomografico (Grauer et al. 2018, SN).
- *Riferimento e validazione:* M.
- *Mancante:* numero di camere, accuratezza, validazione.
- *Collegamenti:* A-cons-005.

**C-measurements-scout-004 — Stato LES completo coerente con misure sparse ad alta frequenza (assimilazione)**
- *Osservazioni durante l'uso:* velocità tomo-PIV o PIV, con covarianza dell'errore assunta nota; OH-PLIF nel caso reattivo (SCO:S28, SN).
- *Target e scala:* stato LES completo, con incertezza d'ensemble.
- *Approcci documentati:* EnKF con localizzazione su getto inerte (SCO:S12); applicazione reattiva su getto DME (S28, SN).
- *Riferimento e validazione:* errore rispetto a dati non assimilati dello stesso esperimento (VS parziale); limite fissato dall'incertezza sperimentale.
- *Mancante:* costo computazionale; caso reattivo solo secondario; nessun dataset posseduto.
- *Collegamenti:* A-scout-007; B-cons-003; missione [03](../03_dmf_gen/README.md) (tema dell'incertezza).

**C-consolidation-006 — T istantanea sulla sezione LES da velocità planare 2C sparse (problema attuale)**
- *Bisogno:* stimare la distribuzione di T quando le osservazioni non danno una mappa termica ([proposal](../../docs/proposal.md) §1–2).
- *Osservazioni durante l'uso:* vi, vj sintetici ai siti di maschere sparse (A-cons-003), eventualmente degradati sinteticamente; parametri operativi (NH₃, φ) se il modello è condizionato; nessuna T.
- *Target e scala:* T istantanea [K] sulla sezione 128 × 128 (Δx ≈ 0,58 mm, Δy ≈ 0,76 mm).
- *Approcci documentati:* U-Net e GLU (lavoro esistente); baseline della proposal (campo medio, metodo non neurale).
- *Riferimento e validazione:* **VN** contro T LES, simultanea e co-localizzata per costruzione; split per condizione operativa (20 train / 3 val; 7 TEST non toccati). Nessuna validazione sperimentale possibile con i dati attuali.
- *Mancante:* orientamento fisico, definizione e segno delle componenti (INV §7 C1); configurazione e parametri del modello PIV ancora da scegliere nella missione 02; vk assente nelle cache (osservazioni 2C). Lo scouting ora motiva la distinzione fra copertura, risposta, errore e validazione/sostituzione ([aggiornamento 02](updates/piv_2026-11-26.md)); non qualifica ancora un operatore per la LES. Una maschera generata da T contiene informazione sul target e va discussa esplicitamente.
- *Collegamenti:* B-data-audit-001/004/005; A-cons-003; missioni [00](../00_chiusura_studio_degradazioni/README.md) e [02](../02_virtual_piv/README.md).

**C-consolidation-007 — Livello termico o T da velocità sparse e poche T puntuali lente (fusione)**
- *Osservazioni durante l'uso:* velocità come in C-cons-006, più T in pochi punti con risposta temporale e incertezza di un sensore reale; numero e posizioni **non fissati**.
- *Target e scala:* T istantanea sulla sezione, oppure livello termico o statistiche.
- *Approcci documentati:* da definire nella missione 04 (baseline semplici prima di metodi appresi). Modello di sensore documentato solo in un contesto diverso (A-scout-001).
- *Riferimento e validazione:* **VN** su LES.
- *Mancante — aspetti temporali:* un sensore lento richiede sequenze di frame. Per le 22 traiettorie non-TEST con coordinate regolari l'associazione indice↔tempo è **accettata dal loader**, senza anomalie dimostrate, ma **non verificata in modo indipendente**: servono ipotesi dichiarate e controlli proporzionati all'uso (INV §9; §6.2). `40NH3_1.2` è esclusa dagli usi multi-frame dalla policy attuale; dichiararne la lacuna non risolve l'associazione né autorizza eccezioni. `t` dei 7 TEST non letto; τ realistico per il nostro combustore M.
- *Collegamenti:* B-data-audit-001/005; A-scout-001; C-scout-001; missione [04](../04_thermal_fusion/README.md).

- *Aggiornamento 04:* distinguere input grezzo di giunzione, T gas corretta e statistica pubblicata; parete/ingresso/scarico non sono T locali del gas. THM/THD documentano layout e supporti (HYLON, Delft, EM2C), ma nessun parametro universale o sensore scelto. I 0,5 s disponibili richiedono valutazione con τ, inizializzazione, transitorio e obiettivo; ambiente radiativo e perturbazione intrusiva non sono qualificati dalla sola slice. Validazione numerica, statistica o fra diagnostiche ancora da concordare; nuove risorse B-TH non danno riferimento istantaneo T–PIV nel nostro apparato.

**C-consolidation-008 — HRR o statistiche termochimiche da OH\* sperimentale RealPDEBench**
- *Osservazioni durante l'uso:* immagini OH\* sperimentali (A-cons-001), una per istante, per condizione operativa.
- *Target e scala:* aperto: HRR integrale o mediato, statistiche, eventualmente T. Linzheng: il target non deve necessariamente essere il campo istantaneo completo di T (diario 25/11).
- *Approcci documentati:* da ricognizione della missione 05; in letteratura C-cons-001.
- *Riferimento e validazione:* **stesso scenario, non stesso stato:** le LES condividono la condizione operativa nominale, non l'istante. Possibile solo un confronto statistico, spettrale o integrato (VST); nessuna validazione sperimentale di T.
- *Mancante:* dati non posseduti (scaricare uno shard è un passo separato da autorizzare); FOV, orientamento, scala, `t` reali; contraddizione «4000 Hz per 1 s» contro 2001 frame.
- *Collegamenti:* A-cons-001; B-data-audit-002; missione [05](../05_optical_diagnostics/README.md).

**C-consolidation-009 — T o HRR da proxy ottico `observed` sulla LES**
- *Osservazioni durante l'uso:* intensità `observed` «surrogate-mapped» (A-cons-002).
- *Target e scala:* T o HRR LES sulla sezione.
- *Approcci documentati:* nessuno specifico.
- *Riferimento e validazione:* **VN**; input e target dalla stessa simulazione, quindi non indipendenti.
- *Mancante:* mappatura e unità del proxy (INV §7 Q2); il proxy non include una risposta ottica documentata.
- *Collegamenti:* B-data-audit-003; A-cons-002; missione [05](../05_optical_diagnostics/README.md).

**C-consolidation-010 — T o HRR da frazione molare di OH numerica sulla LES**
- *Osservazioni durante l'uso:* canale 8 (frazione molare OH) della LES sulla sezione. Non equivale al segnale OH-PLIF (quenching, calibrazione) né all'intensità OH\*.
- *Target e scala:* T o HRR LES sulla sezione.
- *Approcci documentati:* schema di Nie (inferenza da modalità "facili"), applicato senza le sue modalità sperimentali (NIE §11, I).
- *Riferimento e validazione:* **VN**, stessa simulazione.
- *Mancante:* un modello di misura (quenching, integrazione, rumore) per renderlo un problema diagnostico; senza di esso è un problema numerico diverso da Nie.
- *Collegamenti:* B-data-audit-001; C-jason-nie-001; missione [05](../05_optical_diagnostics/README.md).

**C-consolidation-011 — Velocità nelle zone mancanti di una PIV in motore (EngineBench, motored)**
- *Osservazioni durante l'uso:* PIV 2C nel piano di swirl inferiore con lacune (originali codificate come zero, di benchmark artificiali); fase in CAD.
- *Target e scala:* u, v nelle regioni mancanti, istantanee per fase.
- *Approcci documentati:* benchmark EngineBench (metodi non riesaminati qui).
- *Riferimento e validazione:* la PIV stessa nelle zone mascherate artificialmente: stessa modalità, con il proprio errore (1,5–8 %); nessun riferimento termico.
- *Mancante:* dati non scaricati; licenza; ambiguità zero/mancante. Le deposizioni *fired* (A-scout-012b) sono una risorsa distinta e non verificata.
- *Collegamenti:* A-scout-012a; B-data-audit-006; missione [02](../02_virtual_piv/README.md).

**C-consolidation-012 — Problema motore su dati RANS/URANS PoliMi (non formulabile oggi)**
- *Osservazioni durante l'uso:* M. Possibile ponte tramite pressione nel cilindro e grandezze per ciclo (A-cons-006, I).
- *Target e scala:* con RANS, stati medi o d'insieme, non istantanei (I).
- *Approcci documentati:* Jason riferisce un lavoro di digital twin per ammoniaca svolto interamente su RANS/URANS (diario 25/11).
- *Riferimento e validazione:* un target RANS medio non valida fluttuazioni istantanee; misure sperimentali associate M.
- *Mancante:* contenuto, geometria, variabili, risoluzione in angolo di manovella, formato, condivisibilità (INV §7 Q6).
- *Collegamenti:* B-data-audit-008; A-cons-006; quartier generale con Enrico.


**C-PIV-update-001 — Qualificare lacune e sostituzioni attraverso gli stadi Cardiff**
- *Bisogno:* capire che cosa rappresenti un vettore assente, rigettato o modificato nel dato utilizzato dalla ricostruzione.
- *Osservazioni:* maschera, XML e vettori ai tre stadi della stessa acquisizione isoterma; campione scout di 4 frame, non tutti i 514.
- *Target:* distribuzione e persistenza dei codici e delle modifiche dei valori; non il campo di velocità vero.
- *Approccio possibile:* confronto appaiato fra stadi e statistiche descrittive dopo verifica della semantica. Nessun metodo o processamento approvato qui.
- *Riferimento/validazione:* coerenza degli stadi della stessa misura; nessuna verità indipendente, nessuna validazione della PIV in fiamma.
- *Mancante:* documentazione codici e verifica indipendente del decoder; scala fisica prima di misure in mm. Il 28,9–33,3% con codice 16 sui quattro frame non è una frazione certa di sostituzioni.
- *Collegamenti:* B-PIV-S2D-01; PIVD:S2D-01/§4.1; missione 02.

**C-PIV-update-002 — Residui e incertezza rispetto al riferimento HDR Neal**
- *Bisogno:* confrontare una misura PIV con un sistema simultaneo più accurato, distinguendo errore, filtraggio e correlazione.
- *Osservazioni:* immagini MS e vettori HDR già mappati, nel getto non reagente.
- *Target:* residui MS−HDR e loro struttura; eventuale valutazione della copertura UQ rispetto al riferimento qualificato.
- *Approccio possibile:* processamento MS da scegliere e documentare; allineamento dei supporti temporali/spaziali; confronto per componente. Non è un esperimento già deciso.
- *Riferimento/validazione:* misura HDR simultanea/co-localizzata, **non verità puntuale istantanea**. Usa correlazione su cinque coppie, mappatura e interpolazione di outlier senza flag separato. Il residuo contiene contributi di entrambi i sistemi e del loro diverso supporto; non identifica automaticamente la distribuzione dell'errore MS.
- *Mancante:* protocollo MS, supporto confrontabile, trattamento del riferimento e sua incertezza; trasferibilità alla combustione non dimostrata.
- *Collegamenti:* B-PIV-S2D-02; PIVD:S2D-02/§4.2; PIV1; missione 02.

---

## 5. Corrispondenze degli ID e collegamenti

### 5.1 Righe A suddivise o assorbite

| ID originale | Destinazione in questa matrice | Motivo |
|---|---|---|
| A-measurements-scout-009 (pressione) | A-consolidation-006 (cilindro), A-consolidation-007 (pressione dinamica in combustore) | Sensori, segnali e limiti diversi |
| A-measurements-scout-011 (soot) | Parte LII → limite generale aggiunto ad A-jason-nie-003; pirometria → A-consolidation-008; DBI-TI → A-consolidation-009 | Piano contro proiezione; tecniche e apparati diversi |
| A-measurements-scout-012a / 012b | Conservate separate | Motored contro fired: risorse distinte |
| Tabella SCO §3, righe CTC e BOS senza ID | A-consolidation-004, A-consolidation-005 | Target diversi (emissione / indice di rifrazione) |
| INV:B-data-audit-002, 003, 004 | A-consolidation-001, 002, 003 (vista "misura/osservazione"); le schede B restano | Il dataset e l'osservazione che contiene sono viste distinte |

Tutte le altre righe A del consolidamento iniziale (13 di SCO, 5 di Jason) mantengono l'ID originale. A-PIV-update-001/002 sono aggiunte dalla letteratura della 02 (PIVL:L37 e L08/L28); non rinumerano le righe iniziali.

### 5.2 Righe C suddivise o ristrette

| ID originale | Destinazione | Motivo |
|---|---|---|
| C-measurements-scout-002 (HRR da osservabili indiretti) | C-consolidation-001 (proiezione OH\*), 002 (piani PLIF), 003 (linee Raman/Rayleigh) | Osservazioni, ipotesi e riferimenti diversi. L'ID originale resta come rimando |
| C-measurements-scout-003 (inversione line-of-sight) | Ristretto a poche misure integrate; tomografie in C-consolidation-004 (CTC) e 005 (BOS) | Target diversi: emissione, indice di rifrazione, T/specie |
| Problemi descritti in INV §5/§9, NIE §11, LIU §11 senza ID C | C-consolidation-006…012 | Collegare i problemi ai dati effettivamente posseduti o assenti |

### 5.3 Righe B nuove

B-consolidation-001…003 derivano dalla tabella SCO §7 «Dataset segnalati (non ispezionati)», che non conteneva schede B: deposizioni TCC-III fired (S26), DLR GTMC (S01/S02), tomo-PIV del getto Sandia (S12).

Le sette B-PIV-S2D-01/02/03/04/05/11/35 mantengono gli ID del rapporto PIVD. C-PIV-update-001/002 esplicitano le due possibili verifiche Cardiff/Neal proposte da PIVD, senza assegnarle o avviarle. Altri candidati restano nel rapporto, senza moltiplicare le righe della matrice.

### 5.4 Registro delle fonti

Ogni rapporto mantiene il proprio registro, letto alla profondità dichiarata:
- [data_inventory.md §8](data_inventory.md#8-fonti-e-stato-della-lettura) (INV:S01–S16);
- [nie_2024.md §13](papers/nie_2024.md#13-fonti-consultate-e-registro-delle-ricerche) (NIE:S01–S04);
- [liu_2025.md §13](papers/liu_2025.md#13-fonti-consultate-e-registro-delle-ricerche) (LIU:S01–S04);
- [measurement_scouting.md §9](scouting/measurement_scouting.md#9-fonti-e-stato-della-lettura) (SCO:S01–S29).

Fonti comuni a più rapporti, da non contare due volte: Nie et al. 2024 (NIE:S01 = LIU:S04, letto in TI in entrambi i casi); il rapporto preesistente `docs/nonideal/00_ricognizione_osservazioni.md` (INV:S08 = SCO:S27, RP), da cui derivano le informazioni su EngineBench e TNF.

Riferimento bibliografico corretto in F3: Nie et al. 2024, articolo su ScienceDirect <https://www.sciencedirect.com/science/article/pii/S0010218023005576> (URL fornito dal coordinatore, non riverificato da questo ambiente). I supplementari restano non recuperati.

---

## 6. Lacune ordinate

Le missioni indicate sono quelle che *potrebbero* approfondire il punto. Non sono assegnazioni e non avviano attività.

### 6.1 Impediscono di valutare un problema

| Lacuna | Problemi bloccati | Possibile approfondimento |
|---|---|---|
| Statistiche (μ, σ) usate a test per de-standardizzare T in Nie; supplementari non letti | C-jason-nie-001 (uso a inferenza, confronto dei numeri) | 01 / 05 (supplementari dall'URL corretto) |
| Pressione dinamica, continuità e supporto temporale del PINN di Liu; componenti osservate per piano | C-jason-liu-002 | 01; 02 per le componenti |
| OH\* RealPDEBench non posseduto: FOV, geometria, scala, `t` | C-consolidation-008 | 05, con l'inventario dati (download da autorizzare) |
| Mappatura del proxy `observed` | C-consolidation-009 | 05 |
| Contenuto dei dati motore PoliMi | C-consolidation-012 | Quartier generale / Enrico |
| Contenuto e sincronizzazione delle deposizioni TCC-III fired | Eventuale problema motore sperimentale (non ancora in C) | 01 (inventario dati) |
| Validazione dell'indicatore OH·CH₂O e applicabilità locale di OH\* (Hardalupas & Orain, Ayoola) | C-consolidation-001/002 | 05 |

### 6.2 Necessarie prima di progettare un esperimento

| Lacuna | Perché | Possibile approfondimento |
|---|---|---|
| Orientamento fisico, definizione e segno di Velocity[i/j/k], piano della sezione RealPDEBench | Interpretazione fisica di un modello PIV virtuale e del confronto con apparati reali | 01 / 02 (domande agli autori, via Enrico) |
| Uso temporale delle traiettorie. Stato attuale: (a) 22 traiettorie non-TEST con coordinate regolari, associazione indice↔tempo **accettata dal loader** e nessuna anomalia dimostrata; (b) verifica indipendente dell'esportazione **non effettuata** per alcuna traiettoria: lacuna di verifica, non errore dimostrato; (c) `t` dei 7 TEST non letto; (d) duplicati di frame non controllati sulle altre 26 traiettorie; (e) `40NH3_1.2`: anomalia documentata, la policy attuale rifiuta il multi-frame; dichiararne la lacuna non risolve l'associazione né autorizza eccezioni | Un esperimento multi-frame (sensori lenti, forecasting) deve dichiarare l'ipotesi indice↔tempo e applicare controlli **proporzionati all'uso** (INV §9: regolarità di `t` sulla finestra impiegata, continuità dei soli frame usati, scala temporale del sensore rispetto a Δt = 0,25 ms). Il confronto con il raw HDF5 o con gli autori serve solo se l'esito dipende in modo decisivo dal tempo assoluto. Non è un prerequisito universale | 04, nel protocollo del proprio esperimento |
| PIV: configurazione, risposta spaziale, correlazione, validità/sostituzione e bias in fiamma | Meccanismi e apparati documentati da PIV1/PIVL/PIVD, ora integrati (§9); non stabiliti parametri trasferibili alla LES o una percentuale qualificata di rigetti nello swirl reagente stabilizzato | 02: scegliere la configurazione; verifiche proporzionate sui candidati selezionati. Cardiff: codici/decoder; Neal: processamento MS e supporto HDR. Nessun operatore deciso qui |
| τ, ambiente radiativo, stato iniziale e livello del dato termico | THM documenta forme e configurazioni; risposta in situ e parametri trasferibili alla LES non identificati. Dati già compensati non validano il modello che li produce | 04: specifica da discutere con provenienza dei parametri; [consegna intermedia integrata](updates/thermal_2026-11-27.md) |
| Regime di miscelazione della LES (INV §7 C9) | Qualsiasi uso del legame T–φ locale | 01 (fonte primaria) |
| Qualificazione del riferimento: ±120 K Nie non definiti; UQ per vettore non disponibile nei dati EngineBench/GTMC consultati; in PIVL:L37 esistono stime a posteriori ma statistica e copertura M; HDR Neal filtrato/interpolato | Una stima UQ pubblicata non è una mappa disponibile né un riferimento esatto; residuo misura–riferimento e errore di ricostruzione restano distinti | 02 / 05, secondo il confronto scelto |

### 6.3 Approfondimenti utili, non bloccanti per questa prima matrice

- Fonti primarie sulla pressione nel cilindro (pegging, thermal shock) e sulle linee di pressione dinamica (SCO §11, punto 1) — 01, utile per i dati motore.
- Review LAS e tomografia ad assorbimento (Goldenstein 2017; Cai & Kaminski 2017) — 05.
- Versione pubblicata di Barlow et al. 2009 (numeri di Tab. 2) e §5 di Hartl et al. 2019 — revisione trasversale.
- CARS, fosfori termografici, analizzatori dei fumi, dataset laminari e a campi medi — 01 (F5).
- Liu: Δt fra gli snapshot, valori numerici della Fig. 6, robustezza a rumore e filtro; Nie: ablazione degli input, validation set — 01.
- Tracciabilità SHA256 per shard dei raw (INV §7 C4) — 01, nessuna conseguenza scientifica dimostrata.

---

## 7. Che cosa si vede già dalla matrice

Osservazioni trasversali, non classificazioni dei problemi:
- **Segnale ≠ grandezza** in quasi ogni riga A: termocoppia → T del giunto; PIV → particelle; PLIF → fluorescenza dipendente da quenching; chemiluminescenza → fotoni integrati.
- **I quattro difetti E/F/R/L hanno esempi documentati distinti**, non riducibili a rumore gaussiano (A-scout-001, 006, 007, 010).
- **Il supporto spaziale separa i problemi**: sezioni (Liu, Nie), proiezioni (OH\*, LAS, pirometria), linee (Raman), volumi (tomo-PIV, CTC, BOS). Una sezione non fornisce il riferimento di un volume.
- **Il riferimento va qualificato per grandezza e supporto**: B-jason-nie-001 offre T sperimentale simultanea su richiesta; B-PIV-S2D-02 offre un riferimento di velocità simultaneo più accurato, con filtro e incertezza propri. Nessuna nuova risorsa PIV aggiunge T sperimentale simultanea alla velocità per validare direttamente la nostra ricostruzione.
- **I numeri riportati sono esempi di apparati specifici**, non parametri già validati per il nostro combustore.

---

## 8. Note di consolidamento e correzioni

**Importazione.** File copiati senza modifiche dai commit di §1 (commit `2dfaa99` di questo branch). Nessun file con lo stesso nome nella base: nessuna sovrascrittura. Durante il lavoro `plan/month3-missions` è avanzata a `5aec249` (missione 02, step 1, e voce del diario): la base aggiornata è stata integrata con un merge, senza toccare quei file; l'avanzamento non riguarda i quattro rapporti importati.

**Correzioni nei rapporti importati** (tutte marcate nel testo come note F3):
1. *data_inventory.md §9, punto 6 (`40NH3_1.2`).* La versione originale indicava come alternativa per usi temporali «dichiarare esplicitamente la lacuna 1098→1099». Corretto: la policy attuale (verificata in `src/combustion_recon/data/temporal_quality.py` e `shard_access.py`) consente solo l'uso a singolo frame per indice e rifiuta il multi-frame senza associazione verificata; dichiarare la lacuna non risolve l'associazione e non autorizza un'eccezione.
2. *data_inventory.md, scheda B-data-audit-002.* Le dimensioni OH\* erano in unità binarie etichettate MB/GB: aggiunta la conversione in byte dall'inventario remoto (≈ 3,93 GB = ≈ 3,66 GiB per gli 8 shard). Nessuna conseguenza scientifica.
3. *nie_2024.md §13.* Aggiunto l'URL corretto dell'articolo (PII `S0010218023005576`), indicato dal coordinatore; la tabella degli URL tentati, con il PII congetturato, resta invariata come registro storico. Non implica il recupero dei supplementari.

**Scelte di consolidamento.**
- Righe divise quando cambiano tecnica, supporto o apparato (§5); nessun valore "tipico" ricavato da apparati diversi (es. le risoluzioni PIV di S01 e S02 restano separate).
- Il sottoproblema "HRR da osservabili indiretti" è diviso per osservazione (proiezione OH\*, piani PLIF, linee Raman), perché riferimenti e ipotesi differiscono.
- Le osservazioni di Liu e i proxy RealPDEBench sono raccolti in A.7 come **SIN**.
- Marcature D/I/M conservate dai rapporti; le affermazioni da snippet o abstract sono marcate SN/AB anche quando il rapporto le usa come D.

**Verifiche incrociate risolte.** NIE §11 e LIU §11 rinviavano all'audit B comune due domande: (a) RealPDEBench non contiene un riferimento volumetrico (una sola sezione esportata, INV:B-data-audit-001, D [P][A]); (b) nessuno dei 15 canali è un segnale OH-PLIF o una grandezza soot (INV:B-data-audit-001, D [A]). Se la fiamma produca soot resta M.

**Incoerenze delle fonti conservate senza correzione.** «720 measurements» contro 7200 campioni (SCO:S04); risoluzioni PIV 0,94 mm (S01) e 1,5 mm (S02); card RealPDEBench «4000 Hz per 1 s» contro 2001 frame × 0,25 ms; scale integrale ≈ 5,75 µs e di Kolmogorov ≈ 2,07 ns in Nie, giudicate implausibili ma non verificate (NIE §2.2); orientamento del combustore nei diversi audit (INV §7 C1). SCO A-013 descrive l'OH\* come «presente in RealPDEBench»: va letto come *dichiarato sull'hub*, non posseduto (B-data-audit-002).

**Verifiche svolte in F3 e loro limiti.**
- Lette integralmente le quattro consegne e i documenti di contesto (README, roadmap, template, proposal, voce del diario del 25/11).
- Controllo dei calcoli derivati (Δx, Δy, durate, dimensioni di finestre e volumi, rapporti di errore) contro i valori dei rapporti: nessuna discrepanza oltre a quelle elencate.
- Lettura del codice della policy temporale (`temporal_quality.py`, `shard_access.py`) e delle dimensioni OH\* in `data/manifests/remote_file_inventory.json`.
- **Nessuna fonte primaria esterna riletta** in F3: i numeri delle tecniche sono riportati come nei rapporti, alla profondità da loro dichiarata. L'URL corretto di Nie non è stato riverificato da questo ambiente.
- Nessun dato aperto, nessun TEST letto, nessun download.

---

## 9. Aggiornamento intermedio dalla missione 02 — 26 novembre 2026 (F5)

Integrate la scheda approvata Sciacchitano e le due consegne di scouting revisionate, individuate sui branch della 02. [Nota di integrazione, fonti fissate e limiti](updates/piv_2026-11-26.md). Nessuna fusione dei branch di scouting e nessuna modifica ai loro rapporti.

| Fonte | Versione utilizzata | Contenuto trasferito |
|---|---|---|
| PIV1 | [step1_sciacchitano.md](../02_virtual_piv/step1_sciacchitano.md), base `a894257497de1524566186c22b66e02bd7b7b8ee` | Risposta/passo, errori correlati, rigetto/sostituzione e limiti UQ |
| PIVL | [step2_literature.md](https://github.com/EnricoBontempi/official_thesis/blob/2cbf5abd5fd44f8cb6f82c0be933f61e7340ddc5/mese3/02_virtual_piv/scouting/step2_literature.md), `2cbf5abd5fd44f8cb6f82c0be933f61e7340ddc5` | Precisazioni A.2; A-PIV-update-001/002; lacune quantitative e di trasferibilità |
| PIVD | [step2_datasets.md](https://github.com/EnricoBontempi/official_thesis/blob/ee9948e227f0d9272ae064cc8601c5e5399c28b4/mese3/02_virtual_piv/scouting/step2_datasets.md), `ee9948e227f0d9272ae064cc8601c5e5399c28b4` | Sette righe B selezionate dai 35 candidati; C-PIV-update-001/002 come problemi di qualificazione, non esperimenti avviati |

**Verifica e limiti dell'integrazione:** lettura dei rapporti e confronto dei passaggi trasferiti con le rispettive schede/registri; nessuna nuova lettura dei paper originali, decodifica, ispezione di campioni o verifica del disco locale. Le sette ispezioni di campioni riportate da PIVD appartengono allo scout. Questo aggiornamento non costituisce la verifica estesa dello step 3 della 02.

**Stato delle decisioni:** scouting 02 concluso secondo Enrico; protocolli, configurazioni PIV e training da discutere nella missione 02. F5 attiva anche prima dei training e aggiornabile dopo nuovi risultati. F4 rinviata al confronto delle evidenze pertinenti delle altre missioni; attesa la consegna intermedia 04. Nessuna decisione di cambio di problema della tesi.

---

## 10. Aggiornamento intermedio 04 — 27 novembre 2026 (F5)

Integrati TH1, modelli v2 `c2980d3` e dataset v2 `cf318f6`: [fonti fissate e sintesi critica](updates/thermal_2026-11-27.md). Quattro righe A e cinque B aggiunte; aggiornati due problemi C esistenti. Gli ID B-TH-Dxx conservano il numero Dxx del rapporto termico; A-TH-001…004 identificano configurazioni, senza rinumerare gli ID preesistenti.

La consegna intermedia è ricevuta secondo Enrico; i documenti originali mantengono lo stato di proposte da discutere nella 04. Nessun parametro o scenario approvato da questa integrazione. I controlli sui campioni appartengono allo scout: nessun nuovo audit locale, download di payload, lettura di TEST o esperimento. Il tipo finale di validazione resta aperto.

F5 prosegue con nuove evidenze e risultati; le consegne intermedie 02/04 sono entrambe integrate. Il confronto complessivo F4 si prepara con Enrico; scelta e implementazione delle osservazioni termiche restano nella 04, secondo la sua roadmap.
