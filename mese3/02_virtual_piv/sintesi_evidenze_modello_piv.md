# Missione 02 — Sintesi trasversale delle evidenze PIV e modello di osservazione per la LES

**Data:** 27 novembre 2026 · **Branch:** `piv/experiment-proposal`
**Stato:** sintesi da discutere con Enrico prima di approvare la rappresentazione delle osservazioni. Non contiene una campagna di training, una matrice dei run o una roadmap tecnica. Nessun operatore implementato, nessun download di dataset, nessun job, nessun accesso a TEST.

**Materiale incrociato.** Le consegne sugli altri branch sono state lette senza merge; le teste dei branch coincidono con gli SHA indicati.

- Scouting letteratura: `piv/step2-literature-scouting` @ `2cbf5abd5fd44f8cb6f82c0be933f61e7340ddc5`, schede L01–L37 (in seguito **PIVL:Lnn**).
- Scouting dataset: `piv/step2-dataset-scouting` @ `ee9948e227f0d9272ae064cc8601c5e5399c28b4`, schede S2D-nn e campioni del §4 (in seguito **PIVD:S2D-nn**).
- [Scheda Sciacchitano](step1_sciacchitano.md) (**P01–P18**) e [roadmap 02](roadmap02.md).
- [Inventario](../01_observation_state_matrix/data_inventory.md) e [matrice](../01_observation_state_matrix/observation_state_matrix.md) della missione 01 (**INV**, **OSM**).
- Le mie due proposte, conservate come materiale di lavoro: [proposta_esperimenti_v1.md](proposta_esperimenti_v1.md) e [proposta_missione02.md](proposta_missione02.md).

**Letture mirate svolte per questa sintesi.** Ho riletto direttamente dalle fonti primarie i passaggi decisivi, usando PDF pubblici di articoli e tesi, non di dataset. Le pagine sono quelle **del PDF**, salvo dove indico la pagina stampata.

| Sigla | Fonte | Parti rilette |
|---|---|---|
| **STO12** | Stöhr, Boxx, Carter, Meier, *Combust. Flame* 159 (2012), manoscritto DLR elib 77484 | §§2.3.2–2.3.3, p. 3 |
| **OBE15** | Oberleithner et al., *Combust. Flame* 162 (2015), manoscritto elib 96808 | §§2.2, 2.5, pp. 5–7 |
| **BOX10** | Boxx et al., *Combust. Flame* 157 (2010), manoscritto elib 64569 | pp. 15, 24, 31 |
| **SAI16** | Saini, Steinberg, Arndt, AIAA 2016-0436, elib 102428 | §I p. 3, §II pp. 4 e 6, §III p. 7 |
| **LAN12** | Lange et al., 16th Lisbon Symposium 2012, elib 77085 | pp. 5–6, 9 |
| **KAR20** | Karmarkar et al., arXiv:2011.14662v1 | §2.2.1, p. 5 |
| **UTA1** | Clemens (PI), rapporto DOE OSTI 1253136 | §§3.2–3.3.1, pp. 11–13 |
| **UTA2** | Clemens (PI), rapporto DOE OSTI 1506058 | pp. 33–34 |
| **BAU15** | Baumgartner, tesi TU München 2015 | p. 49 stampata (PDF 69), p. 92 (PDF 112), pp. 122, 127, 129 (PDF 142, 147, 149) |
| **WIE17** | Wieneke, tesi TU Delft 2017 | §2.4, Tab. 2.2/2.4 (PDF 29–39); §4.3 (PDF 89–90) |

Le altre fonti sono usate al livello di lettura dichiarato dagli scouting: **FT**, **FT-v**, **AB**, **SN**, **SEC**.

**Etichette.**
- **[file]** osservato in un file di dati ispezionato (dallo scout dataset);
- **[articolo]** descritto in una pubblicazione letta;
- **[ipotesi]** non documentato;
- **[I]** mia inferenza, con premesse dichiarate.

Nella proposta di modello:
- **sostenuto** = evidenze convergenti in fonti pertinenti;
- **trasferimento** = valore documentato altrove e portato sulla LES con una regola esplicita;
- **sensibilità** = aspetto non identificato, esplorato in un intervallo dichiarato;
- **non rappresentabile** = richiede informazione che una slice LES non contiene.

---

## 1. In breve: che cosa abbiamo imparato, come proponiamo di trasformare i dati, che cosa resta incerto

**Che cosa abbiamo imparato.** Mettendo insieme circa trenta fonti fra articoli, tesi e dataset, la PIV in combustione ha alcuni tratti che ritornano in tutti i lavori letti:

1. **Il dato è una griglia regolare di vettori.** Nei lavori DLR la "risoluzione" dichiarata coincide con la finestra finale e il passo è la metà (overlap 50 %). Nei combustori swirl letti le finestre vanno da circa 0,8 a 1,9 mm: dalla cella LES (0,58×0,76 mm) a circa tre celle.
2. **Ogni vettore è una media pesata dello spostamento delle particelle**, non un valore puntuale. La forma della media dipende dall'algoritmo.
3. **Il campo di vista è limitato dall'accesso ottico e dalla frequenza di acquisizione.** I sistemi a pochi Hz possono vedere tutta la sezione della camera. Quelli a kHz vedono di solito una parte, spesso decentrata.
4. **L'errore casuale ha un'ampiezza fissata dalle scelte di progetto:** scala ottica, separazione fra impulsi, spostamento massimo in pixel. In molti lavori è dichiarato a priori come ±0,1 px, non misurato. Quando è misurato contro un riferimento (getto non reagente) vale circa 0,06 px. Fra vettori vicini è correlato.
5. **I vettori invalidi vengono rimossi e quasi sempre sostituiti.** Il dato rilasciato è di solito completo, e i rigetti non si vedono più.
6. **In fiamma, ciò che manca o peggiora dipende soprattutto dal tracciante:**
   - l'olio evapora e lascia senza misura il gas combusto;
   - le particelle solide sopravvivono, ma diventano più rade dove il gas si espande;
   - con i gradienti di densità peggiora la qualità delle immagini.

**Che cosa non abbiamo trovato.** Nessuna fonte pubblica quantifica, in una fiamma swirl a pressione atmosferica, la frazione di vettori rigettati o sostituiti per frame, né la forma e la persistenza delle lacune interne. Le uniche percentuali disponibili vengono da una microPIV di flashback in idrogeno e da una PIV isoterma (Cardiff). Né l'una né l'altra si trasferiscono alla nostra fiamma.

**Quindi proponiamo di trasformare le velocità LES così:**

1. **Nucleo comune (sostenuto dalle fonti):**
   - campo di vista;
   - griglia regolare;
   - media pesata in millimetri;
   - errore casuale correlato, con ampiezza derivata dal progetto della misura e scalata sulla velocità massima di ciascuna condizione;
   - vettori di bordo esclusi.
2. **Due configurazioni coerenti:**
   - **A:** PIV a basso rateo, pieno campo, particelle solide;
   - **B:** PIV ad alto rateo, campo di vista ridotto, particelle solide.

   Una variante **"olio"**, in cui il gas combusto resta senza misura, è un regime alternativo e non un difetto da aggiungere.
3. **Sensibilità dichiarate** per ciò che le fonti indicano come meccanismo ma non quantificano:
   - rigetti interni (sparsi e a cluster);
   - peggioramento nel gas espanso;
   - perdita di coppie per moto fuori piano.
4. **Due livelli di dato:**
   - "validato con mancanti espliciti";
   - "rilasciato, con i buchi riempiti come nella pratica documentata".

**Che cosa resta incerto:**
- l'orientamento delle componenti di velocità della slice (missione 01: non dichiarato);
- quanto contenuto ha la LES alla scala della finestra;
- la frequenza dei rigetti in fiamma swirl atmosferica;
- la correlazione dell'errore al 50 % di overlap (le fonti divergono: ρ(2d) = 0 oppure 0,11);
- la legge con cui la densità di particelle e il moto fuori piano aumentano errore e rigetti.

Alcune di queste lacune si chiudono con verifiche mirate ed economiche. Per altre si procede dichiarando un'ipotesi e un intervallo di sensibilità (§5).

---

## 2. Che cosa abbiamo imparato davvero: tabella delle evidenze

### 2.1 Livello del dato nelle fonti

Una distinzione preliminare, perché cambia ciò che ogni fonte può dimostrare.

| Livello | Dove l'abbiamo | Che cosa permette di vedere |
|---|---|---|
| Immagini di particelle | Cardiff (S2D-01), Neal (S2D-02), PIV Challenge (S2D-05): disponibili, non elaborate | Cause a livello d'immagine; rielaborazione |
| Vettori prima e dopo validazione, con codici | Solo Cardiff, 4 frame su 514, decoder non ufficiale [file] | Che cosa la validazione rimuove e sostituisce |
| Vettori finali con riferimento più accurato | Neal: riferimento HDR, outlier sostituiti senza flag [file + PDF descrittivo] | Residui rispetto a un'altra misura |
| Vettori finali senza flag | RealPDEBench fluidi (S2D-04): nessun NaN né zero in 63 frame [file]; EngineBench: zeri come mancanti (INV B-006) | Layout e tempi; nulla sui rigetti |
| Solo medie o dati di figura | MRB Darmstadt (S2D-03), Johe (S2D-09), Ye (S2D-10) [file] | Campo di vista e convenzioni dei mancanti; nulla sull'istantaneo |
| Parametri e trattamenti descritti | GTMC DLR, Toronto/DLR, DLR 5 bar, UT Austin, TUM, EM2C [articolo] | Progetto della misura, cause qualitative, alcune percentuali di quell'esperimento |

**Conseguenza.** Per le fiamme abbiamo descrizioni e parametri, non vettori pre-validazione. Un campo finale senza NaN, come quello della PIV reale di RealPDEBench fluidi, non dimostra assenza di rigetti: la stessa pratica documentata (BOX10 p. 31; BAU15 p. 49 stampata; Neal PDF §1.3) riempie i buchi prima del rilascio.

### 2.2 Caratteristica per caratteristica

| Caratteristica | Che cosa mostrano insieme le fonti | Condizioni | Convergenze e divergenze | Natura | Fonti (sezione/pagina/campione) |
|---|---|---|---|---|---|
| **Layout** | Griglia cartesiana regolare; passo = finestra × (1 − overlap) quando la finestra è fissa | Tutti i lavori letti | Convergente. La convenzione DLR dichiara la finestra finale come "risoluzione" e il passo come metà: 1 mm / 0,5 mm (GTMC 5 kHz); 1,0 / 0,5 (Toronto/DLR 10 kHz); 1,9 / 0,95 (DLR 5 bar). Per GTMC 5 Hz è dichiarata solo la finestra (1,3 mm, 50 %): passo ≈0,65 mm [I]. Con finestra adattiva (Cardiff) l'overlap non è definito | [articolo]; [file] per Cardiff 81×61 a passo 16 px | BOX10 p. 15; SAI16 p. 7; KAR20 p. 5; STO12 §2.3.2; OBE15 §2.2; PIVD:S2D-01 §4.1 |
| **Campo di vista** | Pieno campo a basso rateo; campo ridotto ad alto rateo per energia dei laser, memoria e sensore delle camere; decentramento per i montanti; esclusioni per riflessi | 5 Hz: "entire vertical section" (GTMC). 5–10 kHz: 30×30 mm (GTMC 5 kHz), ≈49×38 decentrato e poi solo x>0 (Toronto/DLR), ≈54×22 (DLR 5 bar), 13,5×27 (UT Austin), 4,15×4,15 (microPIV TUM) | Convergente sul meccanismo. Le dimensioni dipendono dall'apparato: non c'è un valore tipico. Sulla nostra slice (73,8×96,2 mm) i campi kHz coprono circa il 13–17 % (PIVL:L10, [I]) | [articolo] | STO12 §§2.3.2–2.3.3; OBE15 §2.5; SAI16 p. 7; KAR20 p. 5 e §3; UTA1 p. 12; BAU15 p. 49 st. |
| **Regioni escluse fisse** | Riflessi da finestre e pareti, saturazione, vicinanza alle pareti; zone usate solo in media | Toronto/DLR (riflessi, metà campo scartata); DLR 5 bar (PIV valutabile solo oltre 1 mm dalla parete, "only a mean velocity map provides useful velocity data"); TUM (riflessi confinati sotto 0,2 mm con foglio inclinato e vernice nera) | Convergente: esistono, sono fisse nel tempo e legate alla geometria. Le dimensioni dipendono da pareti e mitigazione | [articolo] | SAI16 p. 7; LAN12 p. 9; BAU15 p. 122 st. |
| **Risoluzione e risposta spaziale** | Il vettore è una media pesata. Single-pass a finestra quadrata ≈ top-hat. Multi-pass DaVis: due funzioni distinte, una per la risposta alle lunghezze d'onda e una per l'autocorrelazione del rumore, entrambe gaussiane con circa 30–40 % di cappello messicano | Wieneke: immagini sintetiche, DaVis 8, finestre gaussiane, 4 passate | Specifico dell'algoritmo. Con 32 px e 50 %: L_sr(rumore) = 38,4 px (≈1,2 W), L_sr(risposta) = 29,8 px (≈0,93 W); con 75 %: 29,7 e 22,9 px. I sistemi GTMC usano DaVis 7.2/8.0 multi-scala, ma pesatura e passate non sono dichiarate | [articolo] sintetico | WIE17 §2.4 e Tab. 2.4 (PDF 31, 39); PIVL:L01–L02 |
| **Spessore del foglio** | 0,4–1,7 mm: integrazione fuori piano dello stesso ordine della finestra in piano | GTMC ≈1 mm (5 Hz), ≤0,7 mm (5 kHz); Toronto/DLR 1 mm FWHM; UT Austin 1 mm; Neal 1,7 mm | Convergente sull'ordine di grandezza | [articolo] | STO12 §2.3.2; BOX10 p. 15; SAI16 p. 7; UTA2 p. 33; PIVD:S2D-02 |
| **Errore casuale: ampiezza** | Nei lavori DLR atmosferici: ±0,1 px × scala / Δt, non verificato e senza statistica dichiarata. Δt scelto per la velocità massima della condizione (GTMC 5 Hz: 6 µs a 35 kW, 20 µs a 10 kW) → incertezza in m/s che cresce con la potenza (±0,3 → ±1,4 m/s). Misure contro riferimento in getto non reagente: deviazione standard 0,060 px (u, v). A posteriori (DLR 5 bar): 0,1–1,1 m/s in piano, statistica non definita | Atmosferico, TiO₂ (GTMC); getto in aria (Neal); 5 bar (DLR) | Convergente su **0,05–0,1 px** come ordine di grandezza in buone condizioni. Divergente sulla definizione: "±" senza statistica, deviazione standard misurata, intervalli a posteriori | [articolo]; [file] per i campioni Neal, senza errore calcolato | STO12 §2.3.2; BOX10 p. 15; OBE15 §2.2; KAR20 p. 5; PIVL:L16 (WIE17 §4.4.2); P01 |
| **Errore: dipendenza spaziale** | Con un solo Δt l'errore assoluto in pixel è circa uniforme, quindi l'errore relativo cresce dove il flusso è lento (ricircoli). Cresce anche con il moto fuori piano, la scarsità di particelle e i gradienti d'indice | GTMC: "relatively coarse velocity resolution (≈±0.3 m/s) at low velocity" motiva il filtro POD; DLR 5 bar: ricircolo interno "not adequately resolved… too slow for the chosen pulse interval"; Valori: incertezza alta nei pennacchi (non reagente) | Convergente sul meccanismo; nessuna mappa d'incertezza in fiamma disponibile | [articolo] | BOX10 p. 24; LAN12 p. 6; PIVL:L25; WIE17 §3.1.7 (L12) |
| **Correlazione degli errori** | Errori di vettori vicini correlati per sovrapposizione delle finestre e per il predittore multi-pass | Sintetico DaVis | ρ(2d) ≈ 0,45 al 75 %; valori "rappresentativi" 0 / 0,11 / 0,45 per overlap 0/50/75 %. La review dà ρ(2h) = 0 per overlap ≤50 % nel caso citato: **divergenza non risolta** | [articolo] sintetico | WIE17 §4.3 (PDF 89–90); P16; PIVL:L15 |
| **Bias** | Slip ≤2 % come stima a priori (TiO₂ 1 µm, 80 m/s); slip trascurabile (St = 0,03). Rifrazione in fiamma: 4 % sistematico e 6 % casuale in propano (solo AB). A pressione: bias massimo 0,2 m/s (metodo non descritto). Peak locking: non documentato in fiamma. In getto freddo con riferimento: errore medio −0,005/−0,021 px | Vari | Nessuna evidenza di un bias grande e trasferibile; i candidati sono piccoli o non quantificati in modo utilizzabile | [articolo]; AB per Vanselow | STO12 §2.3.2; OBE15 §2.2; UTA2 p. 34; PIVL:L14, L16, L30 |
| **Outlier** | Vettori spuri da poca informazione; isolati e a cluster. Test mediano 3×3 normalizzato (ε = 0,1); filtri di intervallo; media mobile. I test a vicinato fisso perdono cluster | Saini (GTC); TUM; Cardiff (XML); review | Convergente sulla pratica; nessuna frequenza generale (P09) | [articolo]; [file] per gli XML Cardiff | SAI16 p. 6; BAU15 p. 49 st.; PIVD:S2D-01; P09; PIVL:L19 |
| **Regioni mancanti** | Vedi §3 | — | — | — | — |
| **Sostituzione** | Interpolazione dei buchi di un vettore (GTMC); interpolazione 3×3 dei vettori rimossi (TUM); media mobile con `substitute = True` (Cardiff); interpolazione lineare nel riferimento Neal; filtri POD e GPOD che usano l'intera serie temporale | Pratica diffusa | Convergente: il dato rilasciato è di solito completo. I flag sono incoerenti: NaN, zero, flag 0/1/2, codici non documentati | [articolo]; [file] | BOX10 p. 31; BAU15 p. 49 st.; PIVD:S2D-01 §4.1, S2D-02, S2D-03; SAI16 p. 4 |

---

## 3. Approfondimento: regioni senza misura e vettori invalidi

### 3.1 Cause, manifestazione ed evidenza

| Causa | Come si manifesta nello spazio | Nel tempo | Evidenza in combustione | Scale o frazioni documentate | Effetto del processamento |
|---|---|---|---|---|---|
| **Accesso ottico e campo di vista** | Regione esterna fissa; campo decentrato per i montanti | Fissa | Forte [articolo]: GTMC, Toronto/DLR, DLR 5 bar, UT Austin | Vedi §2.2: dimensioni dell'apparato | Spesso il campo viene ritagliato e la regione non compare nel file (SAI16 p. 7) |
| **Bordo del campo di vista e della finestra** | Primo vettore a mezza finestra dal bordo; fascia non valutabile | Fissa | Implicita in tutti i lavori; UT Austin: primo vettore a circa 0,5 mm dalla parete | ≈ mezza finestra | Il riempimento al bordo è raro; codice di validazione "on boundary" in Cardiff |
| **Riflessi e flare da pareti e finestre** | Fasce o zone vicine alle superfici; rumore aumentato | Fissa (dipende dal foglio) | Forte [articolo]: DLR 5 bar oltre 1 mm dalla parete; Toronto/DLR, x<0 scartato; TUM <0,2 mm con mitigazione | 0,2 mm – metà campo, secondo mitigazione e geometria | Esclusione o ritaglio; in alcune zone si usa solo la media (LAN12 p. 9) |
| **Gas combusto senza traccianti (olio)** | Tutta la regione oltre il fronte di evaporazione | Segue la fiamma: intermittente e legata al flusso | Forte per quell'apparato [articolo]: UT Austin, flashback swirl CH₄/H₂, "olive oil droplets… vaporized in the preheat zone", ≈555 K (std 10 K) misurati in quello studio | Il bordo dipende da olio, dimensione delle gocce, storia termica, soglia d'immagine (PIVL:L28) | Il bordo è usato come marcatore del fronte (UTA1 p. 13); nessun vettore nel combusto |
| **Calo della densità di particelle nel gas espanso (particelle solide)** | Più vettori interpolati nella zona di reazione; transizione netta di densità al fronte | Segue la fiamma | Media [articolo]: microPIV TUM, H₂/aria flashback, "percentage of interpolated vectors is higher in the reaction zone due to low seeding density"; tesi TU Delft con conteggi di particelle più bassi nel combusto (PIVL:L29) | **Non quantificata in fiamma swirl.** TUM: <10 % invalidi nelle zone con particelle, fino al 20 % interpolati in prove con una scia senza particelle (microscala, flashback H₂) | Interpolazione 3×3: rigetti invisibili nel rilascio (BAU15 p. 49 st.) |
| **Svuotamento centrifugo del seeding nei nuclei di vortice** | Zona scura al centro del moto rotante, **anche senza fiamma** | Segue la struttura del flusso | Debole e qualitativa [articolo]: TUM, immagini Mie in un tubo con swirl ("probably a result of low seeding density in this region due to centrifugal forces") | Nessuna | — |
| **Inseminazione non uniforme e flussi non inseminati** | Vuoti nel campo delle particelle | Intermittenti | Affermazione [articolo]: Saini, "non-uniform particle seeding can result in voids"; nei sistemi DLR è inseminata l'aria (STO12; KAR20: entrambi i flussi d'aria) | Nessuna | GPOD (Saini); interpolazione |
| **Basso rapporto segnale/rumore** (laser a kHz a bassa energia, fondo di fuliggine e luminosità) | "Significant gaps" nei campi processati. I gap artificiali di prova, costruiti "in a manner similar to the gaps found in real experimental data", sono raggruppati in cluster; la geometria non è specificata | Numero di mancanti variabile fra snapshot (assunto gaussiano nei test artificiali) | Affermazione [articolo] per combustori **ad alta pressione** (SAI16 p. 3, p. 6); filtri a 532 nm contro la luminosità (STO12) | **Nessuna statistica dei gap reali**: SAI16 usa gap artificiali G = 15/25/50 % su un set atmosferico "with essentially no missing vectors" (p. 4) | GPOD con l'intera serie |
| **Perdita di coppie per moto fuori piano** | Errore casuale aumentato e rigetti dove la componente normale al foglio è grande | Segue il flusso | Media [articolo]: UT Austin, "out-of-plane velocity was on the same order as the in-plane… limiting factor… was out-of-plane correlation loss"; GTMC posiziona la vita del fascio "to minimize noise arising from inter-frame particle dropout" | Nessuna frazione documentata; legge di perdita di coppie **non letta** (Keane & Adrian) | Validazione e interpolazione |
| **Gradienti d'indice e striature** | Striature d'intensità, picchi di correlazione allungati; validazione bassa vicino a pareti calde | Casuale (striature) | Media [articolo] **ad alta pressione**: UT Austin a 3 atm, aumenta con la pressione; DLR 5 bar | Nessuna frazione | Filtro d'immagine (sfondo mobile di 16 px + normalizzazione min-max) che "greatly improves" il dato (UTA2 p. 34) |
| **Range dinamico (Δt unico)** | Regioni lente misurate male (errore relativo alto), non necessariamente invalidate | Fissa per configurazione | Media [articolo]: DLR 5 bar, ricircolo interno non risolto; GTMC, POD per il rumore a bassa velocità | Nessuna frazione | Filtro POD per le figure (BOX10 p. 24) |
| **Processamento in isotermo (Cardiff)** | Regione fissa di ≈800 vettori su 4941 senza valore al primo stadio; poi 117–270 rigettati dal filtro di intervallo e 490–541 vettori con codice 16 già al primo stadio | 799 posizioni comuni ai 4 frame (fissa); il resto varia | **Isoterma, non in fiamma** [file, decoder non ufficiale, 4 frame su 514] | Codice 16 sul 28,9–33,3 % dei vettori finali; 15,2–20,4 % fuori dalla regione fissa | Media mobile 5×5 con sostituzione: campo finale completo, valori cambiati solo dove il codice è 16 |

### 3.2 Che cosa ne segue

1. **Le regioni senza misura documentate meglio sono fisse e geometriche:** campo di vista, bordi, riflessi. Sono le uniche con una base solida per la morfologia.

2. **In fiamma ci sono due meccanismi documentati legati alla combustione:**
   - assenza totale di misura nel gas combusto con olio (UT Austin);
   - aumento dei vettori interpolati dove la densità di particelle cala (TUM).

   Sono **regimi diversi**, determinati dalla scelta del tracciante. Non sono difetti da sommare.

3. **Rigetti interni intermittenti sparsi e a cluster** sono certamente presenti prima della validazione: la pratica di rimozione e sostituzione lo presuppone. Però:
   - nessuna fonte in fiamma swirl atmosferica ne fornisce frazione, dimensioni o persistenza;
   - in Saini i cluster descrivono gap artificiali costruiti per imitare quelli reali, in un lavoro motivato da combustori ad alta pressione;
   - la percentuale TUM riguarda la microscala, in prossimità di pareti e in flashback di H₂.

4. **Persistenza temporale:** documentata solo per le regioni fisse (Cardiff: stesse posizioni nei 4 frame; il sistema a 5 Hz non permette di seguire altro). Per le lacune legate alla fiamma la persistenza segue il flusso, come mostrano le descrizioni TUM, ma non è quantificata. Per la ricostruzione a singolo frame la persistenza non è comunque necessaria.

5. **Il ruolo di Cardiff.** Mostra come una catena di validazione reale trasforma un campo con buchi in un campo completo, e che la frazione di vettori sostituiti può essere grande anche in un flusso isotermo. Serve a capire il processamento e a qualificare la rappresentazione della sostituzione. **Non giustifica né la frazione né la forma dei rigetti in fiamma**; le sue percentuali non vanno trasferite.

---

## 4. Traduzione sulla nostra LES

**Vincoli dalla missione 01** (INV B-data-audit-001, §7):
- griglia 128×128 con celle 0,581×0,758 mm;
- assi verificati: colonne = x, righe = y;
- **orientamento fisico e definizione di i/j/k non dichiarati**;
- vk presente nei file raw ma non nelle cache;
- filtro e ricampionamento della LES non dichiarati;
- T in K, velocità in m/s per inferenza dagli intervalli.

La proposta usa due componenti, vi e vj, interpretate come in piano **per ipotesi**, finché la verifica G1 del §5 non le qualifica.

### 4.1 Struttura: nucleo comune e configurazioni

| Livello | Contenuto | Perché |
|---|---|---|
| **Nucleo comune** | Griglia regolare; media pesata in mm; errore casuale correlato con ampiezza dal progetto; esclusione dei vettori di bordo; due livelli di dato | Proprietà che convergono in tutte le fonti pertinenti |
| **Configurazione A** | Basso rateo, pieno campo, particelle solide, finestra di circa 1,3 mm | GTMC 5 Hz (STO12 §2.3.2; OBE15 §2.2). È il tipo di dato più vicino al nostro compito a singolo frame (snapshot indipendenti) |
| **Configurazione B** | Alto rateo, campo ridotto, particelle solide, finestra 1,0–1,5 mm | GTMC 5 kHz/10 kHz (STO12 §2.3.3; OBE15 §2.5), coerente con Toronto/DLR e UT Austin per il campo ridotto |
| **Variante tracciante "olio"** | Nessuna misura nel gas oltre il fronte di evaporazione | UT Austin (UTA1 p. 13). Regime alternativo, da usare solo se la LES ha abbastanza gas freddo |
| **Sensibilità** | Rigetti interni; peggioramento nel gas espanso; perdita di coppie fuori piano; forma del filtro; correlazione al 50 % | Meccanismi documentati, quantità non identificate |

A e B **non** combinano apparati diversi: sono i due sistemi dello stesso combustore e dello stesso articolo. Le altre fonti servono a mostrare che le proprietà scelte ricorrono, non a fornire parametri da sommare.

### 4.2 Elemento per elemento

Ogni riga segue lo schema: evidenze convergenti → proprietà da riprodurre → rappresentazione proposta → parametri o intervalli sostenibili → assunzioni e verifiche.

**(1) Campo di vista e regioni escluse** — *A: sostenuto. B: sostenuto per la dimensione; sensibilità per la posizione.*
- *Evidenze:* pieno campo a basso rateo (STO12 §2.3.2); campo ridotto ad alto rateo, 30×30 mm (STO12 §2.3.3), da confrontare con 49×38, 54×22 e 13,5×27 mm altrove; decentramento per l'accesso ottico (SAI16 p. 7).
- *Proprietà:* in A tutta la slice è osservata; in B una finestra di 30×30 mm, cioè circa 52×40 celle e il 13 % dell'area.
- *Rappresentazione:* maschera geometrica fissa. In B, poche posizioni fissate **prima** di ogni valutazione, definite in coordinate della slice e non sulla base delle prestazioni.
- *Assunzioni:* la geometria della camera e la posizione del bruciatore non sono note (INV C1), quindi la posizione in B non ha corrispondenza fisica: risultati riportati per posizione. Non sappiamo se la slice contenga pareti, quindi le esclusioni da riflessi **non** sono nel modello di base. Se la missione 01 o gli autori indicano pareti nella slice, una fascia esclusa di almeno circa 1 mm (LAN12 p. 9) diventa una sensibilità.

**(2) Layout e passo** — *sostenuto, con approssimazione dichiarata.*
- *Evidenze:* griglia regolare, passo pari a metà finestra nelle configurazioni DLR (0,5 / 0,65 / 0,95 mm).
- *Proprietà:* un vettore ogni ~0,5–0,65 mm; l'informazione indipendente è minore del numero di vettori (P10, step 1 §3.3).
- *Rappresentazione proposta:* un vettore per nodo LES (passo 0,581×0,758 mm). Alternativa: una griglia PIV a passo isotropo con interpolazione del campo filtrato, che però introduce un'interpolazione aggiuntiva e complica l'equivalenza fra U-Net e GLU.
- *Assunzioni:* passo anisotropo invece di quadrato. Ritengo accettabile la differenza (0,65 contro 0,58/0,76 mm), ma va dichiarata.

**(3) Risposta spaziale** — *scala sostenuta; forma come trasferimento con sensibilità.*
- *Evidenze:* finestra finale 1,0–1,9 mm nei combustori swirl letti; GTMC A 1,3 mm, B 1,5 mm (5 kHz) o 1,0 mm (10 kHz). La forma dipende dall'algoritmo (WIE17 §2.4–2.6).
- *Proprietà:* media pesata su circa una finestra, con attenuazione delle strutture sotto qualche millimetro.
- *Rappresentazione:* convoluzione in millimetri, rispettando l'anisotropia delle celle. Kernel di riferimento **gaussiano/MHG con L_sr della risposta ≈ 0,9 W** (DaVis multi-pass al 50 %, WIE17 Tab. 2.4). Alternativa di sensibilità: **top-hat di lato W** (single-pass). Vettori di bordo esclusi quando il kernel esce dal campo.
- *Parametri:* W = 1,3 mm (A); 1,0–1,5 mm (B). Per lo stesso sistema GTMC a 5 kHz BOX10 (p. 15) dichiara 1,0 mm con passo 0,5 mm e STO12 (§2.3.3) 1,5×1,5 mm²: la discrepanza non è spiegata nelle fonti (PIVL:L04) e giustifica l'intervallo.
- *Assunzioni e verifiche:* la pesatura e il numero di passate dei sistemi GTMC non sono dichiarati. Il kernel copre circa due celle e la LES ha già un filtro implicito ignoto, quindi l'effetto simulato può essere piccolo (verifica G2). La media attraverso lo spessore del foglio (~1 mm) è **non rappresentabile**.

**(4) Errore casuale: ampiezza** — *trasferimento con regola esplicita, più sensibilità.*
- *Evidenze:* ordine 0,05–0,1 px; Δt scelto in base alla velocità massima della condizione. Nel GTMC a 5 Hz, con 16 px = 1,3 mm (≈81 µm/px) e Δt = 6 µs a 35 kW, v ≈ 80 m/s dà ≈6 px di spostamento massimo [I, dai valori di STO12]. ±0,1 px corrisponde allora a ≈1,35 m/s, coerente con il ±1,4 m/s dichiarato.
- *Proprietà:* l'errore in m/s è **circa costante nel campo** e **proporzionale alla velocità massima della condizione**, perché Δt è scelto condizione per condizione. L'errore relativo è massimo nei ricircoli lenti.
- *Rappresentazione:* σ_u = σ_px × v_max,cond / d_max, applicato per componente in piano, con **v_max,cond calcolato per condizione** (per esempio un quantile alto di |u| planare, solo su TRAIN per le condizioni di training).
- *Parametri:* σ_px nell'intervallo 0,05–0,1 px (dal valore misurato in getto freddo a quello convenzionale a priori); d_max ≈ 6 px [I, dal GTMC]. Con il p99,9 di |u| pooled TRAIN ≈ 47 m/s (ricognizione §1.3) si ottiene σ ≈ 0,4–0,8 m/s.
- *Assunzioni:* "±0,1 px" **non** è trattato come σ gaussiana dichiarata. È un estremo convenzionale dell'intervallo; la natura statistica della distribuzione resta da verificare con Neal (G4). In fiamma l'errore cresce dove le particelle sono rade o il moto fuori piano è forte: vedi (7) e (8).

**(5) Errore casuale: correlazione** — *trasferimento con divergenza dichiarata.*
- *Evidenze:* errori correlati fra vettori vicini; in DaVis multi-pass la lunghezza di correlazione del rumore è ≈1,2 W al 50 % (WIE17 Tab. 2.4). ρ(2d) "rappresentativo" = 0,11 al 50 % (WIE17 §4.3), contro 0 nel caso riportato dalla review (P16).
- *Proprietà:* rumore spazialmente correlato su circa una finestra, con autocorrelazione che decresce rapidamente (WIE17 PDF 29).
- *Rappresentazione:* rumore bianco filtrato con un kernel la cui autocorrelazione abbia L_sr ≈ 1,2 W, poi rinormalizzato a σ. Sensibilità con rumore indipendente per nodo (ρ = 0), che rappresenta l'estremo della review.
- *Verifica:* G4 (residui Neal). Se non eseguita, la correlazione resta un trasferimento da metrologia sintetica.

**(6) Bias** — *escluso dal modello di base, con motivazione.*
- Slip ≤2 % come stima a priori; nessun bias trasferibile documentato in fiamma atmosferica; peak locking non documentato in fiamma; rifrazione quantificata solo in abstract.
- Riportare l'esclusione come scelta. Non introdurre bias arbitrari.

**(7) Peggioramento nel gas espanso (particelle solide)** — *sensibilità, usa T nel generatore.*
- *Evidenze:* calo della densità di particelle attraverso il fronte e più vettori interpolati nella zona di reazione (BAU15 pp. 122, 127 stampate; PIVL:L29). Il legame con il rapporto di densità è un'inferenza fisica: le particelle seguono la massa, ρ ∝ 1/T a pressione costante.
- *Proprietà:* nel combusto, meno particelle per finestra → errore e rigetti maggiori.
- *Rappresentazione possibile:* probabilità di rigetto e/o ampiezza dell'errore crescenti con un indicatore di densità relativa derivato da T.
- *Parametri:* nessuno documentato in fiamma swirl, quindi solo intervalli di sensibilità.
- *Assunzioni:* **la maschera porterebbe informazione su T** (step 1; roadmap §4A). Servono controlli che separino il contenuto informativo della maschera, per esempio una maschera della stessa forma presa da un altro frame. Non fa parte del modello di base.

**(8) Perdita di coppie per moto fuori piano** — *sensibilità, condizionata alla verifica G1.*
- *Evidenze:* nel swirl la componente fuori piano è dello stesso ordine di quella in piano e limita Δt (UTA1 p. 12); il meccanismo aumenta l'errore casuale e la perdita di corrispondenza (WIE17 §3.1.7, PIVL:L12).
- *Proprietà:* errore e rigetti crescono con |v_normale| Δt / spessore del foglio.
- *Rappresentazione possibile:* usa **vk** (non T) per modulare errore o rigetti.
- *Parametri:* con Δt dal punto (4) e foglio di ~1 mm la frazione di particelle che escono è dell'ordine di |vk| / v_max × 0,5 mm / 1 mm [I]. La legge esatta va letta in Keane & Adrian, non letto.
- *Assunzioni:* richiede che k sia la componente normale alla slice (G1). Senza G1 non si rappresenta.

**(9) Rigetti interni sparsi e a cluster, e sostituzione** — *sensibilità; non trasferire le percentuali di Cardiff o TUM.*
- *Evidenze:* la pratica di validazione e sostituzione è universale nelle fonti; frequenze in fiamma swirl atmosferica assenti (§3.2).
- *Proprietà:* nel livello "validato" alcuni vettori mancano, isolati o a piccoli gruppi. Nel livello "rilasciato" sono stati riempiti con interpolazione locale (3×3, media mobile) **senza traccia**.
- *Rappresentazione:* due output dello stesso generatore:
  - **validato con flag:** mancanti espliciti (U-Net: maschera = 0; GLU: token rimosso);
  - **rilasciato:** gli stessi vettori sostituiti con interpolazione locale e presentati come misurati.

  La morfologia proposta: rigetti isolati più cluster di pochi vettori, in un **intervallo di sensibilità esplicitamente non calibrato**. Il caso nullo corrisponde al dato GTMC rilasciato, in cui sono documentati solo i buchi di un singolo vettore riempiti (BOX10 p. 31).
- *Parametri:* nessun valore centrale sostenuto. Gli intervalli vanno dichiarati come scenario, fissati prima delle valutazioni e mai in base al danno.
- *Uso di Cardiff:* qualificare che la regola di sostituzione riprodotta si comporti come quella di una catena reale (valori finali vicini alla media locale: PIVD:S2D-01 §4.1(c)), non fissare le frazioni.

**(10) Variante "olio"** — *regime alternativo; trasferimento con T nel generatore.*
- *Evidenze:* UT Austin, l'olio d'oliva evapora a ≈555 K in quello studio; nessun vettore nel combusto (UTA1 pp. 11, 13).
- *Rappresentazione:* osservazione solo dove T < T_e, con T_e come sensibilità attorno al valore riportato e un bordo non netto.
- *Assunzioni:* il bordo reale non è un'isoterma (evaporazione finita, trasporto delle gocce, soglia d'immagine; PIVL:L28). Usa T e richiede controlli sull'informazione della maschera. Ha senso solo se la LES contiene abbastanza gas freddo (G6).

**Ordine delle operazioni proposto:** velocità LES in m/s → selezione del campo di vista → convoluzione in mm → campionamento ai nodi ed esclusione dei bordi → errore correlato → rigetti (solo nelle sensibilità) → eventuale sostituzione (livello "rilasciato") → normalizzazione con statistiche fit-only.

### 4.3 Riepilogo per classe di scelta

| Classe | Elementi |
|---|---|
| **Sostenuto** | Griglia regolare con passo ≈ metà finestra; scala della finestra; campo pieno (A) o ridotto (B) e loro dimensioni; esclusione dei bordi; errore proporzionale alla velocità massima della condizione; esistenza di validazione e sostituzione e il loro effetto sul dato rilasciato |
| **Trasferimento da giustificare** | Forma del filtro (DaVis sintetico); lunghezza di correlazione dell'errore; σ_px tra 0,05 e 0,1 px e d_max ≈ 6 px; passo PIV approssimato dai nodi LES; 2C in piano |
| **Sensibilità** | Posizione del campo in B; top-hat contro MHG; ρ(2d) fra 0 e 0,11; rigetti interni e loro morfologia; peggioramento nel gas espanso (T nel generatore); perdita di coppie fuori piano (vk); variante olio |
| **Non rappresentabile** | Spessore del foglio e gradienti fuori piano; errori propri della stereo; rifrazione e striature; luminosità di fiamma (incluse NH₂*/NO₂* nelle fiamme con ammoniaca, solo SN); traiettorie delle particelle fra impulsi; regioni non inseminate per il flusso del combustibile (quale inlet sia inseminato nella LES non è noto) |

---

## 5. Che cosa manca davvero

| ID | Lacuna | Quale decisione blocca | Verifica mirata | Indispensabile? |
|---|---|---|---|---|
| G1 | Definizione e segno di i/j/k rispetto alla slice | Interpretare vi, vj come in piano; usare vk (perdita di coppie, eventuale 3C) | Su TRAIN: confronto fra lo spostamento apparente delle strutture scalari tra frame consecutivi e (vi, vj, vk). Se non conclusivo, domanda agli autori tramite Enrico | **Sì** per (8) e per la 3C. Per il nucleo 2C si può procedere dichiarando l'ipotesi, ma la verifica è economica e va fatta |
| G2 | Contenuto della LES alla scala di 1–2 mm | Quanto pesa davvero la risposta spaziale; top-hat contro MHG | Su TRAIN: spettri o funzioni di struttura di vi e vj; differenza fra campo e campo filtrato con W = 1,0–1,9 mm, per regione (getti, fronte) | **Sì**, prima di fissare il nucleo: evita di dichiarare un effetto nominale come se fosse fisico |
| G3 | Qualificazione del generatore | Qualunque uso comparativo | Controlli delle proprietà dichiarate: trasferimento del filtro, σ e autocorrelazione dell'errore, maschere, ordine delle operazioni, identità delle osservazioni fra U-Net e GLU | **Sì** |
| G4 | Statistica e correlazione dell'errore; divergenza su ρ(2d) al 50 % | (4) forma della distribuzione; (5) lunghezza di correlazione | Neal: elaborare le immagini MS con un software aperto e documentato a 50 % e 75 % di overlap; residui rispetto all'HDR; ρ(d) e ρ(2d) | No: si procede con la sensibilità ρ = 0 / filtrato. Consigliata, perché chiude una divergenza esplicita |
| G5 | Frequenza e morfologia dei rigetti in fiamma swirl atmosferica | Valore centrale di (9) e (7) | Non esiste un dataset pubblico verificato. Tentativo su Cambridge Oxycoal (swirl reagente con PIV ad alta velocità dichiarata, allumina e carbone) per vedere se rilascia mancanti espliciti. TCC-III fired descrive un motore SI, non uno swirl | No: si procede con sensibilità non calibrate e lo si dichiara. La verifica è utile solo se Oxycoal fornisce vettori pre-sostituzione |
| G6 | Frazione di gas sotto ~600 K nella LES | Se la variante olio ha senso | Su TRAIN: frazione d'area sotto soglie di T per condizione | Solo per la variante olio |
| G7 | Legge di perdita di coppie | Forma quantitativa di (8) | Lettura mirata di Keane & Adrian (perdita di coppie in-plane e out-of-plane) | No; necessaria solo se (8) entra |
| G8 | Pesatura e passate dei sistemi GTMC | Scelta fra MHG e top-hat | Letture mirate: Meier 2010 (§2.4) e Steinberg et al. 2013 per il sistema a 10 kHz | No: coperta dalla sensibilità top-hat/MHG |
| G9 | Decoder Cardiff e semantica dei codici | Uso di Cardiff per qualificare la sostituzione | Confronto della media dei 514 campi decodificati con la statistica salvata da DynamicStudio nello stesso database; confronto della regione NaN con la maschera definita | No; utile solo per qualificare la regola di sostituzione |
| G10 | Presenza di pareti nella slice e geometria della camera | Esclusioni fisse da riflessi | Missione 01 o autori | No: assenza assunta e dichiarata |

**Non serve un nuovo scouting generico.** Le verifiche G1, G2, G3 e G6 si fanno sui nostri dati TRAIN, senza download e senza reti. Le altre sono controlli mirati con una domanda precisa.

**Manca una base quantitativa** per i rigetti in fiamma swirl e per l'effetto della densità di particelle. Non rende impossibile il modello: lo rende un **nucleo sostenuto con sensibilità dichiarate**, da presentare così.

### 5.1 Eventuali nuovi download (non avviati, da autorizzare)

| Download | Dimensione | Per quale verifica |
|---|---|---|
| Neal, regione "unsteady core" (4TU, CC0) | 271,8 MB | G4 |
| Cardiff, tre cartelle `.vectors`, statistica `V.*` e XML via HTTP Range (figshare, CC BY 4.0) | ≈0,8 GB | G9 |
| Cambridge Oxycoal (Apollo) | Ignota; nuovo tentativo di accesso | G5 |

Per questa sintesi ho scaricato soltanto i **PDF di articoli e tesi** elencati in testa (STO12, OBE15, BOX10, SAI16, LAN12, KAR20, UTA1, UTA2, BAU15, WIE17, più Meier 2010, Valori 2019, la tesi TU Delft e Barbosa 2008 non usati in modo decisivo). Sono rimasti nello scratchpad della sessione e non sono committati.

---

## 6. Che cosa cambia rispetto alle due proposte precedenti

- La [proposta_missione02.md](proposta_missione02.md) prendeva il GTMC come apparato di riferimento e Cardiff come fonte della variante dei rigetti. Questa sintesi **mantiene il GTMC come ancoraggio dei parametri**, ma mostra che le proprietà del nucleo convergono anche nelle altre fonti (Toronto/DLR, DLR 5 bar, UT Austin, TUM).
- **Non** propone più di trasferire struttura e frazioni dei rigetti da Cardiff. Cardiff serve solo a qualificare la regola di sostituzione; i rigetti diventano una sensibilità non calibrata.
- L'ampiezza dell'errore è ora legata alla **velocità massima di ciascuna condizione** tramite il Δt documentato, invece che a un valore unico sul dataset.
- Il filtro distingue **risposta e correlazione del rumore** come due funzioni diverse (WIE17), invece di usare lo stesso kernel per entrambe.
- Sono aggiunti due meccanismi legati al flusso, **senza usare T**: perdita di coppie fuori piano (con vk, dopo G1) e svuotamento centrifugo (solo qualitativo, non incluso).
- [proposta_esperimenti_v1.md](proposta_esperimenti_v1.md) resta un'alternativa di ricerca; le sue verifiche V1–V7 sono riprese qui come G1–G9.

Nessuna campagna di training è proposta in questo documento.
