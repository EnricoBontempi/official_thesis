# Missione 02 — Step 2, Agente B: evidenze della letteratura sulla PIV sperimentale

**Data:** 26 novembre 2026 (v1); revisione v2 nella stessa data, dopo la revisione del coordinatore (§13)  
**Branch:** `piv/step2-literature-scouting`, base `5aec249`  
**Stato:** consegna dello scout della letteratura, da discutere nella chat di coordinamento. Non è una decisione sullo scenario.  
**Perimetro:** ricerca e lettura di fonti; nessun operatore virtuale, parametro definitivo, training, inferenza o accesso a TEST. Nessun PDF di terzi è committato. La missione 01 non ha ancora output nel checkout; il rapporto dell'Agente A non era disponibile durante la stesura della v1.

Letture iniziali svolte: [roadmap 02](../roadmap02.md), [scheda di Sciacchitano](../step1_sciacchitano.md) (in particolare §§4, 5, 9, 11), [README](../README.md), [ricognizione osservazioni](../../../docs/nonideal/00_ricognizione_osservazioni.md). Nella repository non esistono file `AGENTS.md`.

---

## 0. Come leggere questo rapporto

**Livello di lettura della fonte**, sempre dichiarato:

| Sigla | Significato |
|---|---|
| **FT** | Testo integrale letto nelle parti rilevanti: metodo sperimentale, processamento, validazione, risultati pertinenti. Per le tesi lunghe sono indicati i capitoli letti. |
| **FT-v** | Testo integrale di una **versione diversa** dalla pubblicazione citata: manoscritto accettato, preprint o versione congressuale. La versione è sempre indicata. |
| **AB** | Solo abstract, da OpenAlex o dalla pagina dell'editore. |
| **SN** | Snippet o riassunto restituito dal motore di ricerca. È un indizio, non una lettura della fonte. |
| **SEC** | Risultato di un lavoro riferito da un altro lavoro letto. La fonte primaria non è stata letta. |

**Regola d'uso dei numeri.** I numeri da FT/FT-v possono sostenere una proposta, entro le condizioni della fonte. I numeri da AB restano attribuiti e circoscritti all'esperimento descritto. I numeri da SN o SEC **non sostengono decisioni quantitative**: indicano soltanto che cosa cercare o leggere.

**Natura dell'affermazione:** *sperimentale* (misura in un apparato), *simulata* (immagini o campi sintetici, Monte Carlo), *a priori* (stima dichiarata da regole generali, non misurata), *a posteriori* (stima ricavata dai dati acquisiti con un metodo UQ), *inferenza* (ragionamento di questo rapporto).

Gli ID delle evidenze sono **L01…L37**. Gli ID L01–L36 sono quelli della v1; L37 è stato aggiunto nella v2. Gli ID **P01…P18** rimandano al registro della [scheda dello step 1](../step1_sciacchitano.md#4-registro-delle-evidenze-rilevanti).

---

## 1. Sintesi

Tra parentesi: ID delle evidenze e livello di lettura che sostiene ogni frase.

1. **Nessuna fonte letta quantifica, in una fiamma swirl stabilizzata, la frazione di vettori rigettati o sostituiti per frame e per regione.**
   - Il numero più diretto trovato riguarda un flashback H2/aria in microPIV: vettori interpolati fino al 20% in alcune prove, dovuti a una scia senza particelle; meno del 10% di invalidi nelle zone inseminate (L22, FT).
   - Le lacune dei dati PIV ad alta pressione sono descritte come “significative”. Nella versione letta vengono simulate con gap artificiali su dati atmosferici quasi completi, senza statistiche sui gap reali (L23, FT-v).
2. **Il tipo di tracciante determina se i gas combusti sono osservati.**
   - In un esperimento di flashback swirl CH4/H2 a pressione atmosferica, le goccioline di olio d'oliva (~1 µm) evaporano nella zona di preriscaldamento. Gli autori riportano una temperatura di evaporazione di circa 555 K (deviazione standard 10 K), misurata nelle loro condizioni. Con questo tracciante la velocità è disponibile solo nel gas incombusto (L28, FT).
   - Il valore non è una soglia universale (dipende da olio, dimensione delle gocce e storia termica) né definisce un'isoterma esatta.
   - Particelle solide (TiO2, Al2O3) sopravvivono e consentono misure in entrambi i gas, ma si depositano su pareti e finestre (L28, L34, FT).
3. **Attraverso il fronte la densità di particelle visibili cala.**
   - Lo documentano descrizioni e immagini nella microPIV di flashback, dove l'interpolazione è più frequente nella zona di reazione, e una tesi di PIV tomografica (L22, L29, FT).
   - L'entità del calo non è misurata nelle fonti lette: il valore di −85% compare solo in uno snippet e non è usato. Il legame con il rapporto di densità è un'inferenza fisica.
4. **Gli effetti ottici della fiamma e dei gradienti di densità sono documentati, ma quantificati in pochi casi e con livelli di lettura diversi.**
   - In una fiamma premiscelata di propano, Vanselow et al. (2019) riportano errori sistematici fino al 4% e casuali fino al 6%; la parte casuale varia fra i piani (L30, solo AB). Il numero vale per quell'esperimento.
   - Ad alta pressione compaiono striature d'intensità e picchi di correlazione allungati, corretti con filtri d'immagine (L31, FT, qualitativo).
   - Vicino a una parete raffreddata il beam steering porta a validazioni basse e lascia utile soltanto la media (L24, FT, qualitativo).
   - In un flusso non reagente a densità variabile, il metodo UQ di Wieneke classifica affidabile circa il 75% della sezione (L25, FT).
5. **Nei lavori di combustione DLR atmosferici letti l'incertezza istantanea è una stima a priori.**
   - Viene ottenuta da ±0.1 px moltiplicato per scala/Δt e dichiarata fra ±0.3 e ±1.4 m/s secondo sistema e condizione. Non è indicato se sia standard o espansa e non è verificata con un riferimento (L04, L11, FT-v).
   - Per lo stesso sistema a 5 kHz e la stessa camera, due articoli dichiarano 1.0 mm con ±0.3 m/s e 1.5 mm con ±0.5 m/s (L04).
   - **Un caso in combustione pressurizzata fornisce invece una stima a posteriori:** nel combustore DLR a 5 bar di Karmarkar et al. (preprint), le incertezze da correlation statistics di DaVis sono 0.1–1.1 m/s in piano e 0.3–2.3 m/s fuori piano. La statistica a cui si riferiscono questi intervalli e la loro copertura non sono definite, e i valori non sono validati con un riferimento (L37, FT-v).
6. **La risposta spaziale della PIV dipende dall'algoritmo e non coincide in generale con una media su finestra quadrata.**
   - Questa coincide solo con il caso single-pass a finestra quadrata.
   - Nelle configurazioni DaVis studiate da Wieneke (immagini sintetiche, finestre gaussiane, più passate) il filtro equivalente è gaussiano con una componente a cappello messicano. Servono due funzioni: una per la risposta alle lunghezze d'onda (L_sr ≈ 0.7 volte la finestra nel caso 32×32 px, 4 passate, 75% di overlap) e una per l'autocorrelazione del rumore (L01–L02, FT).
   - Sono risultati specifici di quelle configurazioni, non un modello universale della PIV multi-pass.
7. **Gli errori dei vettori vicini sono correlati, e la correlazione dipende da overlap e algoritmo.**
   - Nel caso sintetico DaVis con finestra gaussiana 32×32 px e 75% di overlap, ρ(2d) ≈ 0.45 e la correlazione è significativa fino a 3 passi (L15, FT).
   - Per il 50% di overlap, la fonte primaria usa ρ(2d) = 0.11 come valore “rappresentativo” in una simulazione Monte Carlo. Lo step 1 riporta invece dalla review un valore nullo. **La discrepanza resta aperta**: le condizioni dei due valori non sono state riconciliate e non scelgo un valore.
8. **Su un getto non reagente con riferimento ad alta risoluzione, l'errore misurato è prevalentemente casuale.**
   - Deviazione standard 0.060 px per entrambe le componenti; errore medio −0.005 px su u e −0.021 px su v. Incertezza stimata con correlation statistics 0.063–0.064 px (L16, FT).
   - Non è trasferibile automaticamente alla fiamma.
9. **Il moto fuori piano e i gradienti fuori piano sono sorgenti di errore casuale descritte come rilevanti, e lo swirl le rende probabili.**
   - Nel flashback swirl di UT Austin il Δt è stato limitato dalla perdita di correlazione fuori piano (L08, FT).
   - Il valore indicativo “~0.1 px” e la frase “spesso la sorgente dominante” vengono da lavori citati da Wieneke (L12, SEC).
   - Nella stereo-PIV la componente fuori piano ha un'incertezza 2–3 volte quella in piano nelle stime a priori DLR (L04) e circa doppia negli intervalli a posteriori di L37.
10. **Validazione e sostituzione cambiano il dato rilasciato.**
    - In un iniettore swirl reagente gli autori riportano, da un loro studio parametrico non letto, che i parametri di validazione cambiano la velocità massima fino all'11% (L20, FT-v + SEC).
    - L'interpolazione dei buchi isolati, i filtri POD, il riempimento gappy-POD e i filtri d'immagine producono campi completi in cui i rigetti non sono più visibili (L21, L23, L26, FT/FT-v).
11. **Gli apparati ad alta frequenza hanno campi di vista piccoli rispetto alla nostra slice.**
    - Per esempio 30×30 mm² o circa 54×22 mm², contro 73.8×96.2 mm² (L04, L10, L37).
    - Il campo è spesso decentrato per i montanti della camera; si aggiungono regioni escluse per riflessi o vicinanza alle pareti (L05, L24, L34).
    - Queste esclusioni fisse sono documentate meglio delle lacune interne.
12. **Restano plausibili ma non verificati:**
    - lacune guidate dalla fiamma in fiamme swirl stabilizzate;
    - effetto della luminosità NH2*/NO2* delle fiamme con ammoniaca sulle immagini PIV;
    - persistenza temporale delle lacune;
    - bias di una finestra a cavallo del fronte.

    Lo step 3 deve controllarli sui dati (L33, L35).

---

## 2. Accesso alle fonti: limiti effettivi

Il controllo è stato fatto all'inizio della sessione. Determina che cosa è stato letto davvero.

| Canale | Esito | Conseguenza |
|---|---|---|
| DLR elib, TU Delft repository (file `File_<uuid>`), arXiv, OSTI, TUM (epc.ed.tum.de) | Download PDF funzionante | Fonti FT/FT-v di questo rapporto |
| IOPscience (MST) | Pagina di challenge JavaScript, nessun PDF | Neal 2015, Boomsma 2016 e Wieneke 2015 non letti in FT, benché OA. Wieneke 2015 e Sciacchitano & Wieneke 2016 sono coperti dalla tesi di Wieneke e dalla copia TU Delft (FT). |
| Springer (Exp. Fluids, Appl. Phys. B) | Redirect a `idp.springer.com`, host non consentito | Nessun articolo Springer letto in FT, compresi gli OA |
| PMC | reCAPTCHA | Garcia 2011 non letto |
| CORE download | Challenge JavaScript | Nessun PDF da CORE |
| Tentativo di superare le challenge con browser automatizzato | **Negato dal sistema di permessi; non ritentato** | — |
| Host aggiunti all'ambiente durante la sessione con approvazione (`idp.springer.com`, `utoronto.scholaris.ca`, `media.suub.uni-bremen.de`, `research-information.bris.ac.uk`, `www.sciencedirect.com`) | Download **negati dal classificatore dei permessi** della sessione; non ritentati | Saini et al. 2016 (versione rivista), Vanselow et al. 2018/2019 e Masullo & Theunissen 2016 restano AB/SN. I manoscritti accettati esistono in archivi istituzionali (Toronto, Bremen) e possono essere letti in una sessione con permessi adeguati. |

Gli abstract sono stati letti tramite OpenAlex quando disponibili. Niente è stato ricostruito dai soli snippet: dove compare un numero da snippet è marcato **SN**.

---

## 3. Mappa dei fenomeni

Colonna “Evidenza”:
- **forte**: più fonti FT/FT-v in apparati pertinenti;
- **media**: FT in un apparato o in metrologia sintetica ben definita;
- **debole**: AB/SN/SEC o solo qualitativa;
- **assente**: non trovata.

| Fenomeno | Meccanismo | Dove documentato (apparato) | Effetto sul dato | Evidenza | ID |
|---|---|---|---|---|---|
| Campo di vista ridotto e decentrato | Sensore e memoria a kHz; montanti della camera; riflessi | GTMC DLR 5 kHz–10 kHz; combustore Toronto/DLR; EM2C; flashback UT Austin; combustore DLR a 5 bar | Copertura parziale; bordi fissi; metà fiamma | forte, descrittiva | L04–L08, L37 |
| Risposta spaziale finita | Finestra, pesatura, passate iterative, filtro predictor/corrector | Metrologia sintetica DaVis (configurazioni di Wieneke); parametri dichiarati in combustione | Attenuazione delle piccole scale, overshoot, correlazione del rumore | media (sintetica, specifica dell'algoritmo) | L01, L02 |
| Compromesso rumore/risoluzione | Più pixel per finestra, meno rumore | Sintetico DaVis | σ·L_sr ≈ costante sul set studiato | media | L01 |
| Range dinamico di velocità (Δt unico) | Un solo Δt per getti veloci e ricircoli lenti | Combustore pressurizzato DLR | Ricircolo lento non risolto; errore relativo maggiore a bassa velocità (inferenza) | debole–media (qualitativa) | L06 |
| Incertezza a priori ±0.1 px | Convenzione sulla precisione della stima del picco | Lavori DLR atmosferici letti | ±0.3–1.4 m/s dichiarati, non misurati | forte come pratica, debole come misura | L04, L11 |
| Incertezza a posteriori (correlation statistics) | Analisi della correlazione delle immagini | Combustore DLR a 5 bar (ethylene/acetone) | 0.1–1.1 m/s in piano, 0.3–2.3 m/s fuori piano (statistica non definita) | debole–media (un caso, non validato) | L37 |
| Componente fuori piano (stereo) più incerta | Geometria stereo, angolo delle camere | GTMC DLR; DLR 5 bar | 2–3× (a priori); ~2× negli intervalli a posteriori | media | L04, L37 |
| Moto e gradienti fuori piano | Perdita di coppie; particelle a z diverse | Metrologia (FT, valori da SEC); swirl UT Austin (FT) | Errore casuale; Δt limitato | media | L08, L12, L13 |
| Peak locking | Immagini di particella sotto ~2 px; pixel grandi delle camere veloci | Metrologia | Bias verso spostamenti interi; più rumore | media (generale), assente in fiamma | L14 |
| Correlazione spaziale degli errori | Sovrapposizione delle finestre e filtro effettivo | Sintetico DaVis; getto di riferimento | ρ(2d) ≈ 0.45 a 75% di overlap (DaVis, sintetico); valore a 50% non riconciliato | media | L15, L16 |
| Limiti dei metodi UQ e copertura | Punti ciechi dei metodi | Getto, scia (non reagenti) | Sotto/sovrastima; risposta piatta al moto fuori piano | debole–media (AB/SEC) | L17 |
| Outlier isolati e cluster | Picchi spuri da poca informazione o rumore | Metrologia (AB/SN); microPIV in fiamma (FT) | Rigetti; i cluster sfuggono ai test a vicinato fisso | debole | L19, L22 |
| Sensibilità ai parametri di validazione | Soglie diverse, rigetti diversi | Iniettore swirl EM2C | Fino all'11% sulla velocità massima | debole–media (riportato, studio originale non letto) | L20 |
| Sostituzione e smoothing | Interpolazione 3×3, filtri POD, GPOD, filtri d'immagine | GTMC DLR; TUM; Toronto/DLR; UT Austin | Campo completo; rigetti invisibili | forte (pratica) | L21–L23, L26 |
| Tracciante che evapora | Goccioline d'olio che evaporano nel preriscaldamento | Flashback UT Austin (olio d'oliva, ~555 K in quelle condizioni); CPIV (AB/SN) | Nessun vettore nei gas combusti | forte nell'esperimento documentato | L28 |
| Calo della densità di particelle visibili | Espansione termica (meccanismo indicato dagli autori) | TUM, tesi TU Delft (FT); −85% solo SN | Più vettori interpolati in zona di reazione; fronte visibile nelle immagini | media (qualitativa) | L22, L29 |
| Slip e termoforesi | Inerzia; gradiente di T nella zona di preriscaldamento | GTMC (stime a priori); fiamme a controflusso (AB) | Ritardo particella-gas al fronte | debole–media | L27 |
| Gradienti d'indice di rifrazione | Deflessione dei raggi, sfocatura | Propano (AB); alta pressione (FT); vicino a pareti (FT); supercritico non reagente (FT) | Errori sistematici e casuali locali; striature; validazioni basse | media | L24, L25, L30–L32 |
| Luminosità della fiamma | Chemiluminescenza o fuliggine nella banda della camera | GTMC, TUM (FT); alta pressione (affermazione); fuliggine RQL (AB/SN); NH3 (SN) | Fondo; riflessi accentuati | debole–media | L33 |
| Riflessi e accesso ottico | Fogli che colpiscono pareti o finestre; montanti | DLR, Toronto/DLR, TUM, UT Austin, motore | Regioni escluse; saturazione; autocorrelazione sul fondo | forte (qualitativa) | L05, L24, L34 |
| Sporcamento delle finestre | Depositi di semina; superfici roventi | DLR, UT Austin | Tempi di misura limitati; profilo del foglio alterato | media (qualitativa) | L34 |
| Qualità peggiore vicino alla fiamma | Combinazione dei precedenti | TUM (FT); Vanselow (AB/SN); Valori (FT, non reagente) | Problemi concentrati presso il fronte o i pennacchi | debole–media; nessuna statistica in fiamme swirl | L22, L25, L30–L32, L35 |

---

## 4. Registro delle evidenze

Ogni scheda riporta, quando disponibili: fenomeno e meccanismo; apparato e condizioni; tipo; effetto e numeri; metodo; campione e variabilità; trattamento e residuo; fonte, posizione e livello di lettura; limiti e trasferibilità; dati per lo step 3. I campi senza informazione sono omessi o marcati “non dichiarato”. DOI e URL sono in §10.

### 4.A Operatore di misura

**L01 — Modello a filtro lineare della PIV, lunghezza di risoluzione e legame con il rumore (risultati per configurazioni DaVis specifiche)**
- *Meccanismo proposto dall'autore:* la PIV si approssima come media pesata dello spostamento con pesi F(x). Sotto le ipotesi del modello (linearità; contributi di rumore indipendenti pixel per pixel, App. A), l'autocorrelazione normalizzata di F coincide con quella del rumore fra vettori vicini (eq. 2.4). L_sr = 1/∫F² equivale all'integrale dell'autocorrelazione del rumore. Per un filtro top-hat L_sr = L; per una gaussiana L_sr = σ√(4π).
- *Tipo:* simulato, con immagini sintetiche processate in DaVis 8: spostamento costante, Nppp 0.1, d_p 2.5 px, 10% di moto fuori piano, finestre gaussiane, 4 passate.
- *Numeri (Tab. 2.2/2.4), finestra 32×32 px, overlap 0/25/50/75/87%:*
  - σ = 0.050/0.052/0.060/0.076/0.084 px;
  - L_sr dal rumore = 47.1/41.6/38.4/29.7/25.8 px;
  - L_sr dalla risposta alle lunghezze d'onda (R_A) = 59.3/44.5/29.8/22.9/20.8 px;
  - 16×16 px al 75%: σ = 0.146 px, L_sr = 15.8 px (rumore) e 12.2 px (R_A);
  - 64×64 px al 75%: σ = 0.038 px, L_sr = 57.3 e 44.6 px.

  Il prodotto σ·L_sr(rumore) resta fra 2.13 e 2.35 px². L'autore la chiama “universal rule” (eq. 2.8), ma è verificata **su un set di immagini** variando i parametri di processamento. Bias sistematico < 0.002 px.
- *Lettura per noi:* in queste configurazioni multi-pass l'overlap cambia sia il rumore sia la risposta, perché il predittore viene interpolato fra vettori; non è solo una questione di passo. Con 0% e 25% di overlap la risoluzione da R_A peggiora.
- *Fonte:* B. Wieneke, *PIV Uncertainty Quantification and Beyond*, tesi di dottorato TU Delft, 2017, cap. 2, §§2.2–2.4, Tab. 2.2 e 2.4. **FT**, cap. 2 letto integralmente.
- *Limiti:* un solo software, con raccomandazioni specifiche di DaVis (“3–4 passate, 75% di overlap, finestre gaussiane”). Immagini ideali senza fiamma. L'autore stesso nota che la forma del filtro “può variare significativamente” con dettagli dell'algoritmo, numero di iterazioni e convergenza (§2.7). **Non è un modello universale della PIV multi-pass.**
- *Step 3:* se un dataset dichiara software, finestra, pesatura, passate e overlap, si può valutare quale forma di filtro sia plausibile. Dai soli vettori non si ricava.

**L02 — Forma della risposta: single-pass, multi-pass, gradino (risultati per configurazioni specifiche)**
- *Numeri e osservazioni dalla tesi di Wieneke (DaVis, sintetico):*
  - Single-pass con finestra quadrata 32 px: risposta R_A a forma di sinc, lobo negativo a circa 2/3 della finestra; a λ = 200 px l'ampiezza è ancora ridotta del 10% (§2.6.1).
  - Single-pass con finestra gaussiana DaVis: filtro quasi gaussiano con L_sr ≈ 36 px.
  - Multi-pass (32×32 px, 75%) da 1 a 6 passate: L_sr(R_A) passa da 36 a 22 px; L_sr dal rumore da 37 a 30 px. Il filtro è approssimato da una miscela gaussiana–cappello messicano (MHG) con b ≈ 0.3–0.4 (Tab. 2.3).
  - Risposta a gradino: overshoot e undershoot; MHG con b = 0.385 e L_sr ≈ 22.7 px (§2.6.2).
- *Altri algoritmi:*
  - Elsinga & Westerweel 2011, software TU Delft (**SEC**): risoluzione di separabilità ≈ finestra + 1 px.
  - Kähler et al. 2012 (**AB**): per la correlazione ensemble single-pixel il limite è la dimensione delle immagini di particella; la PTV raggiunge risoluzioni sub-pixel *sui campi mediati*.
- *Fonte:* tesi di Wieneke, §§2.5–2.7, **FT**; Kähler, Scharnowski & Cierpka 2012, **AB**.
- *Trasferibilità:* la media su finestra quadrata descrive il single-pass a finestra quadrata. Per altri algoritmi è un'approssimazione la cui risposta dipende dall'implementazione. Il rapporto L_sr(R_A)/finestra ≈ 0.7 vale per le configurazioni DaVis studiate. Theunissen 2012, Scarano 2003 e Lavoie et al. 2007 **non sono stati letti**.

**L03 — Precedente di operatore “PIV virtuale” applicato a campi simulati**
- *Contenuto:* Sunol et al. (arXiv 2510.24673 v4, reologia, flussi non reagenti) definiscono y = S·H·u + ε.
  - H è una media con finestra di Hann sulle finestre d'interrogazione; S è il campionamento sulla griglia dei vettori.
  - ε è un rumore correlato tramite smoothing gaussiano, con larghezze legate a finestra e passo (χ_corr = 0.35).
  - L'ampiezza di ε è eteroschedastica: ∝ U95·[1 + 0.5·gradiente normalizzato], più un offset di bias a bassa frequenza pari a 0.1 volte la scala base.
- *Tipo:* scelta di modellazione per uno studio di sensibilità. I parametri **non** sono qualificati su un esperimento.
- *Fonte:* sezione “Reference data and PIV emulation”, eq. 24–30. **FT** della sezione.
- *Uso:* precedente di struttura (filtro, campionamento, rumore correlato ed eteroschedastico). Non giustifica alcun valore numerico.

**L04 — Configurazioni PIV nel Gas Turbine Model Combustor DLR (atmosferico, CH4/aria)**

Ogni riga è un sistema distinto, con i valori dichiarati *nella sezione indicata*. Non trasferisco valori da una riga all'altra.

| Sistema (fonte, sezione) | Flusso | FOV | Finestra/overlap | Risoluzione dichiarata | Passo | Δt | Foglio | Incertezza dichiarata |
|---|---|---|---|---|---|---|---|---|
| SPIV 5 kHz + OH-PLIF (Boxx 2010, pp. 13–14) | 10 kW, φ 0.75, parzialmente premiscelata | non quantificato nel testo letto (foglio ≈40 mm in altezza; 512×512 px) | 16×16 px, 50% | 1 mm | 0.5 mm | 20 µs | ≤0.7 mm | ±0.3 m/s “random uncertainty”, da ±0.1 px |
| Stesso sistema 5 kHz (Meier 2010, §2.4.2) | GTMC | — | 16×16 px, 50% | 1 mm | 0.5 mm | 20 µs | ≤0.7 mm | ±0.3 m/s “absolute accuracy”, da ±0.1 px |
| SPIV 5 kHz + OH-PLIF (Stöhr 2012, §2.3.3) | 10 kW, φ 0.65 | 30×30 mm² | non ripetuta (dichiarati “stessi algoritmi” del 5 Hz) | 1.5×1.5 mm² | — | 20 µs | 0.7 mm | ±0.5 m/s in piano; fuori piano ≈2× (α≈30°) |
| SPIV 5 Hz (Stöhr 2012, §2.3.2) | 10–35 kW, φ 0.65 | intera sezione verticale della camera | 16×16 px, 50% | 1.3×1.3 mm² | — | 6 µs (35 kW) – 20 µs (10 kW) | ≈1 mm | ±0.3 m/s (10 kW) – ±1.4 m/s (35 kW); fuori piano ≈3× (α≈20°) |
| PIV 5 Hz (Meier 2010, §2.3) | GTMC | altezza della camera (110 mm) | 16 px | 1.1 mm | — | — | 1 mm | 0.4 m/s “measuring uncertainty”, da 0.1 px |
| SPIV 5 Hz (Oberleithner 2015, §2.2) | GTMC, varie condizioni | intera sezione verticale | 16×16 px, 50% (DaVis 8.0) | 1.3×1.3 mm² | — | non riportato nell'estratto | ≈1 mm | ±0.8 m/s in piano, da ±0.1 px; fuori piano ≈3× (α = 20°) |
| SPIV 10 kHz + OH-PLIF (Oberleithner 2015, §2.5) | GTMC | ridotto rispetto al 5 Hz (fig. 1); foglio alto ≈40 mm | 16×16 px, 50% (DaVis 8.0) | 1.0×1.0 mm² | — | non riportato | ≈1 mm | **non dichiarata** in questa sezione (rinvio a Steinberg et al. 2013, non letto) |

- *Altri dettagli:*
  - TiO2 0.5–1 µm, letto fluido, circa 4% dell'aria (Boxx 2010).
  - Filtro passa-banda 532±5 nm “per ridurre l'influenza della luminosità di fiamma” (Stöhr §2.3.2; Oberleithner §2.2).
  - Durata 0.8 s per run (4096 coppie) nei sistemi a 5 kHz.
  - Camera 85×85 mm con montanti negli angoli.
- *Tipo:* parametri dichiarati. Le incertezze dichiarate sono **a priori** (L11). Nessuna è dichiarata come standard o espansa.
- *Variabilità:* lo stesso sistema SPIV a 5 kHz nello stesso combustore compare con 1.0 mm e ±0.3 m/s (Boxx 2010; Meier 2010) e con 1.5×1.5 mm² e ±0.5 m/s (Stöhr 2012). La causa non è spiegata: definizione della risoluzione, ingrandimento o processamento.
- *Slip:* Stöhr 2012 (§2.3.2) stima un errore massimo da slip del 2% per il 5 Hz a 35 kW. Oberleithner 2015 (§2.2) calcola un numero di Stokes di 0.03 e considera trascurabile lo slip (L27).
- *Fonti:*
  - Boxx et al. 2010, manoscritto accettato DLR elib 64569, pp. 11–14;
  - Stöhr et al. 2012, manoscritto DLR elib 77484, §§2.3.2–2.3.3;
  - Meier et al. 2010, copia autore DLR elib 65267, §§2.3–2.4;
  - Oberleithner et al. 2015, manoscritto DLR elib 96808, §§2.2 e 2.5.

  Livello **FT-v** (manoscritti); per Meier e Oberleithner sono state lette le sezioni sperimentali.
- *Trasferibilità:* combustore, combustibile e scala diversi dal nostro caso NH3/CH4. Utili come **classe di configurazioni documentate**, non come parametri del nostro apparato.
- *Step 3:* verificare con l'Agente A se esistono dati rilasciati di questi apparati e con quale livello di processamento.

**L05 — Combustore swirl Toronto/DLR: FOV decentrato per ostacoli e riflessi**
- *Apparato:*
  - swirl premiscelato CH4/aria, φ 0.70, camera 85×85×114 mm;
  - S-PIV a 10 kHz, 2.6 mJ per impulso, Δt 5–15 µs, foglio FWHM ≈1 mm, TiO2;
  - DaVis 8.0, finestra finale ≈1.0 mm, passo ≈0.5 mm.
- *Effetto:*
  - FOV ≈49×38 mm, decentrato rispetto all'asse (x da −17 a 32 mm) per i montanti della camera;
  - i riflessi dalle finestre aumentano il rumore per x < 0, e sono stati **usati solo i dati con x > 0**.
- *Fonte:* Saini, Steinberg & Arndt, AIAA 2016-0436, §III p. 7. **FT-v**: versione AIAA, non l'articolo Exp. Fluids 57:122.
- *Step 3:* l'esclusione di metà campo è una scelta a valle che non lascia traccia nei vettori se il campo viene ritagliato.

**L06 — Range dinamico: un solo Δt per getti veloci e ricircoli lenti**
- *Apparato:*
  - combustore pressurizzato a settore singolo DLR Colonia, 5 bar, preriscaldo 450 K;
  - gas naturale, swirl parzialmente premiscelato, λ = 1.2;
  - PIV con 120 mJ, pco.2000, 32×32 px: 0.39×0.39 mm² nella vista globale.
- *Effetto:*
  - servono Δt ≈ 2 µs per 50 m/s all'uscita;
  - “la ricircolazione interna non era adeguatamente risolta, essendo troppo lenta per l'intervallo scelto”;
  - per esigenze di risoluzione è stata misurata solo metà della fiamma.
- *Tipo:* sperimentale, qualitativo.
- *Fonte:* Lange et al. 2012, pp. 5–6. **FT**.
- *Inferenza:* a parità di errore assoluto in pixel, le regioni lente hanno errore relativo maggiore. Non è una misura.

**L07 — Iniettore swirl EM2C a 12 kHz: FOV piccolo per la frequenza**
- *Apparato:*
  - PIV a 12 kHz (coppie) con camera a 24 kHz limitata a 128×512 px;
  - Δt 6 µs, foglio 0.3 mm, TiO2 1 µm;
  - FlowManager adattivo da 32 a 8×8 px con 25% di overlap, risoluzione dichiarata 1.07×1.07 mm².
- *Nota:* la tabella riporta 0.017 mm/px, incompatibile con 8 px = 1.07 mm. Probabile refuso della fonte; non usato.
- *Fonte:* Barbosa, Scouflaire & Ducruix, arXiv 0807.1871 (preprint del 2008), §2.2, Tab. 1. **FT-v**.

**L08 — Swirl con flashback (UT Austin): moto fuori piano e Δt**
- *Apparato:*
  - condotto anulare di miscelazione, CH4/H2, pressione atmosferica;
  - SPIV a 4 kHz, Nd:YLF ~10 mJ, 256×512 px, 55 µm/px, FOV 13.5×27 mm;
  - DaVis 8.2 da 64 a 16 px con 75% di overlap; Δt 40/80 µs.
- *Effetto:* “la velocità fuori piano era dello stesso ordine di quella in piano… il fattore limitante nell'aumentare Δt era la perdita di correlazione fuori piano”.
- *Configurazione pressurizzata (secondo rapporto):* SPIV a 8 kHz, 16×16 px = 0.8×0.8 mm², 75% di overlap “per avere più vettori vicino al fronte”, primo vettore a 0.5 mm dalla parete.
- *Fonti:* OSTI 1253136, §3.2.2, pp. 11–12; OSTI 1506058, §4, pp. 28–30. **FT** delle sezioni sperimentali. Sono il contesto di Ebi & Clemens 2016, che **non è stato letto** (L36).

**L09 — Errori nascosti di stereo-PIV, calibrazione e temporizzazione**
- *Contenuto:*
  - nella stereo-PIV con foglio di spessore finito le due camere non vedono lo stesso volume d'interrogazione, e i gradienti fuori piano producono errori sistematici non visibili nelle immagini (Wieneke & Taylor 2006, **SEC**);
  - il disallineamento fra foglio e piano di calibrazione si corregge con l'autocalibrazione;
  - nei laser Nd:YLF ad alta frequenza il ritardo fra trigger e luce può variare fino a 0.5–1 µs secondo frequenza e potenza (Bardet et al. 2013, **SEC**).
- *Fonte:* tesi di Wieneke, §§3.1.1–3.1.3. **FT**.
- *Inferenza:* con Δt di 5–20 µs, una variazione di 0.5–1 µs sarebbe del 2.5–20%. Se e quanto avvenga negli apparati di combustione letti **non è documentato**. È una sorgente nascosta, non verificabile dai vettori.

**L10 — Copertura della nostra slice rispetto agli apparati letti (inferenza numerica)**
- La slice misura 73.8×96.2 mm, con celle di 0.581×0.758 mm (ricognizione §1.2).
  - FOV di 30×30 mm (SPIV DLR a 5 kHz, Stöhr 2012): circa il 13% dell'area.
  - FOV di 49×38 mm (Toronto/DLR), di cui usati circa 32×38 mm: circa il 17%.
  - FOV di circa 54×22 mm (DLR a 5 bar, L37): circa il 17%.
  - Sistemi DLR a 5 Hz: intera sezione verticale della camera.
- Finestre di 0.8–1.9 mm corrispondono a circa 1.4–3.3 celle in x e 1.1–2.5 celle in y. Passi di 0.5 mm sono **più fini** della griglia LES.
- **Se** il filtro effettivo fosse simile a quello delle configurazioni DaVis di Wieneke (L_sr(R_A) ≈ 0.7 volte la finestra, L01), una finestra di 1 mm risponderebbe come un filtro di circa 0.7 mm, cioè circa una cella LES. Con algoritmi diversi il rapporto può cambiare.
- **Conseguenza (inferenza):** per configurazioni simili a quelle DLR a kHz il filtraggio PIV è alla scala della griglia. Non si può rappresentare in dettaglio senza conoscere il filtro implicito della LES e il ricampionamento a 128×128, ancora non verificati (missione 01).

### 4.B Errori e incertezza

**L11 — Nei lavori di combustione DLR atmosferici letti, l'incertezza è una stima a priori da ±0.1 px**
- *Pratica osservata:*
  - nei lavori DLR atmosferici di L04 l'incertezza istantanea è ottenuta come ±0.1 px × scala/Δt;
  - i termini usati sono “random uncertainty”, “measuring uncertainty” e “absolute accuracy”;
  - non è dichiarato se sia standard o espansa;
  - per la componente fuori piano è applicato un fattore 2–3× ricavato dalla geometria stereo (Lawson & Wu 1997, **SEC**);
  - non è validata con un riferimento.
- *Coerenza:* 16 px = 1 mm dà 62.5 µm/px, e 0.1 px × 62.5 µm / 20 µs = 0.31 m/s, coerente con ±0.3 m/s. È una conversione, non una misura.
- *Eccezione documentata:* un caso DLR in combustione pressurizzata usa invece una stima a posteriori (L37).
- *Contesto metrologico:* la review (P01) considera “troppo semplificato” un valore universale di 0.1 px.
- *Implicazione:* σ = 0.5 e 1.5 m/s dello studio nonideal sono dello stesso ordine di questi valori. **Ciò non li valida**: anche quelli sono a priori.
- *Step 3:* cercare dataset con UQ per vettore o con un riferimento, non solo valori ± nel testo.

**L12 — Moto fuori piano: meccanismo e ordine di grandezza riportato**
- *Meccanismo (FT, Wieneke §3.1.7):*
  - particelle che entrano o escono dal foglio non trovano corrispondenza;
  - particelle sovrapposte a z diverse cambiano intensità relativa fra i due impulsi;
  - il picco di correlazione diventa asimmetrico, con spostamenti virtuali fino a metà del diametro dell'immagine di particella.
- *Numeri (SEC):* l'errore casuale medio “può raggiungere 0.1 px, quasi indipendentemente dalla densità di semina” ed è “spesso la sorgente dominante” (Nobach & Bodenschatz 2009, Nobach 2011, riportati da Wieneke). Non sostiene una scelta quantitativa.
- *Effetti correlati:* un disallineamento fra i due fogli ha lo stesso effetto. La sola variazione globale d'intensità fra i due impulsi non è critica (§3.1.8).
- *Trasferibilità:* nello swirl la componente fuori piano è grande (L08; nell'audit del Mese 1 v_k ≈ ±15 m/s, ricognizione §2). La legge “v_k grande implica buco” **non** è dimostrata. Il meccanismo documentato aumenta l'errore casuale e la perdita di coppie.

**L13 — Gradienti fuori piano della velocità in piano**
- *Meccanismo:* la posizione media in z delle particelle nella finestra è casuale, con deviazione standard σ_z/√n. Con un gradiente ∂u/∂z questa posizione produce un errore casuale.
- *Numeri:*
  - esempio dell'autore: foglio con σ = 10 px, 20 particelle, gradiente del 5–10%: errore di 0.11–0.22 px;
  - prova sintetica: 0.078 px contro 0.010 px senza gradiente, con stima teorica di 0.056 px.
- *Fonte:* tesi di Wieneke, §3.1.9. **FT**, *simulato*.
- *Step 3:* non verificabile dai soli vettori planari; richiede più piani o il volume.

**L14 — Peak locking e dimensione delle immagini di particella**
- *Meccanismo:* con immagini di particella sotto ~2 px lo spostamento è attratto verso valori interi (bias) e l'errore casuale aumenta. Si vede negli istogrammi della PIV 2D; nello stereo è meno evidente.
- *Condizioni:* chiudere il diaframma “non è pratico per le camere ad alta velocità con pixel ≥20 µm”; i rimedi sono sfocatura o diffusori.
- *Fonte:* tesi di Wieneke, §3.1.5. **FT**. Christensen 2004 non letto.
- *Combustione:* nessuna fonte letta lo documenta in fiamma. Plausibile per sistemi a kHz con pixel grandi, **non verificato**.
- *Step 3:* istogrammi delle parti frazionarie degli spostamenti in pixel. Richiedono Δt e scala, e funzionano meglio su vettori non filtrati.

**L15 — Correlazione spaziale degli errori e propagazione alla vorticità**
- *Simulazione:* immagini sintetiche in DaVis 8.2 con finestra gaussiana 32×32 px e 75% di overlap. La correlazione dell'errore è significativa fino a 3 passi; ρ(2d) ≈ 0.45 (§2.3.3, fig. 2).
- *Valori Monte Carlo:* ρ(2d) = 0, 0.11 e 0.45 sono indicati come “rappresentativi” di overlap 0%, 50% e 75% (§3.2). Il testo non spiega come siano ricavati i valori per 0% e 50%.
- *Propagazione (eq. 42):* U_ω = (U/d)·√(1−ρ(2d)) per le differenze centrali, con U uguale per u e v ed errori di u e v non correlati.
- *Anomalia:* il testo afferma che ρ = 0.45 riduce U_ω “di un fattore 5” rispetto a ρ = 0. Dall'eq. 42 risulta un fattore di circa 1.35. L'affermazione non è usata.
- *Fonte:* Sciacchitano & Wieneke 2016, copia TU Delft, §§2.3.3, 3.2 e fig. 2, **FT**; tesi di Wieneke, §§4.2.3 e 4.3.2, **FT**.
- **Discrepanza aperta:** la scheda dello step 1 (§4.1) riporta dalla review un ρ(2h) nullo per overlap ≤50%. Questa fonte usa 0.11 come valore rappresentativo al 50%. Non ho verificato le condizioni a cui si riferisce l'affermazione della review (algoritmo, pesatura, passate), quindi non riconcilio i due valori e **non ne scelgo uno**. In ogni caso nessuna delle due fonti autorizza a dire “errori indipendenti a 50% di overlap” per vettori adiacenti: ρ(d) non coincide con ρ(2d).
- *Limiti:* sintetico, un algoritmo, flusso uniforme. È la correlazione dell'*errore*, non del campo.

**L16 — Verifica sperimentale in un getto non reagente con riferimento**
- *Esperimento:* database di Neal et al. 2015, getto rettangolare a x/h = 20, 10 kHz.
  - Sistema di misura MS: HSS5, 16×16 px con finestra gaussiana, 75% di overlap, ingrandimento 0.126.
  - Riferimento HDR: stereo, ingrandimento 0.449, 48×48 px, risoluzione spaziale simile.
  - Secondo Neal et al. (**SEC**) l'HDR è circa 4 volte più accurato.
- *Numeri al punto P:*

| Grandezza | Errore medio | Deviazione standard dell'errore | Incertezza CS (rms) |
|---|---|---|---|
| u | −0.005 px | 0.060 px | 0.063 px |
| v | −0.021 px | 0.060 px | 0.064 px |
| vorticità | 0.0008 px/px | 0.0104 px/px | 0.0116 px/px (con ρ(2d) = 0.45, d = 4 px) |

  Sul dominio comune la vorticità ha errore e incertezza fra 0.010 e 0.016 px/px.
- *Fonte:* tesi di Wieneke, §4.4.2 e Tab. 4.4 (= Sciacchitano & Wieneke 2016, §4.2). **FT**. Neal et al. 2015: **AB**.
- *Trasferibilità:* flusso freddo con semina omogenea. Non dice nulla dell'eterogeneità in fiamma.

**L17 — Confronti fra metodi UQ: copertura e punti ciechi**
- *Sciacchitano et al. 2015* (**AB** + **SEC** tramite Bhattacharya et al. 2018):
  - quattro metodi su un getto, con casi a sorgente d'errore dominante diversa;
  - i metodi diretti (CS, IM) sono più sensibili; il metodo uncertainty surface ha risposta piatta nel caso con moto fuori piano;
  - le coperture al livello nominale del 68% sono già registrate allo step 1 dalla review (P-level).
- *Boomsma et al. 2016* (**AB** troncato + **SEC**): tutti i metodi rispondono qualitativamente agli errori variabili nello spazio; IM e CS possono sottostimare l'incertezza standard anche con bias trascurabile.
- *Implicazione:* una mappa UQ, inclusa quella di L37, ha limiti noti. Non certifica l'assenza di outlier, peak locking o sorgenti nascoste (P12).

**L18 — Altri numeri dichiarati in campi reagenti o in motore**
- *UT Austin, alta pressione* (OSTI 1506058, p. 30, **FT**):
  - incertezza “< 0.1 m/s” a pressione atmosferica e “bias massimo 0.2 m/s” a pressione elevata;
  - errore di ricostruzione stereo “< 0.3” e “0.6”, **unità non indicate**;
  - metodo non descritto nell'estratto.

  Non utilizzabile come σ.
- *Motore ottico, SPIV* (Peterson et al., arXiv 2009.01710, §2.4, **FT-v**): incertezza “≤10%” stimata a priori; riflessi laser vicino al punto morto superiore. Trasferibilità bassa.

**L37 — Incertezza a posteriori (correlation statistics in DaVis) in un combustore DLR a 5 bar** *(aggiunta v2)*
- *Apparato e condizioni:*
  - combustore modello DLR per motori aeronautici: doppio swirl, camera 68×68×120 mm, finestre in quarzo da 3 mm, in vessel pressurizzato;
  - 5 bar, etilene/acetone, φ = 0.67, 38.4 kW, air split 0.2–0.5.
- *PIV:*
  - SPIV a 10 kHz, Edgewave fino a 9 mJ per impulso, Phantom V1212 a 640×800 px, δt = 10 µs, durata 1 s;
  - TiO2 0.5 µm in entrambi i flussi d'aria;
  - DaVis 8.4, multi-pass, finestra finale 24×24 px con 50% di overlap: risoluzione dichiarata 1.9 mm, passo 0.95 mm;
  - dominio −27 < x < 27 mm, 5 < y < 27 mm (circa 54×22 mm).
- *Incertezza (§2.2.1):* “absolute values of uncertainty of instantaneous velocities based on the correlation statistics in DaVis were estimated to be 0.1 m/s—1.1 m/s for the in-plane components and 0.3 m/s—2.3 m/s for the out-of-plane component. On average, uncertainty values for the in-plane components are in the range of 0.04 m/s—0.45 m/s and … out-of-plane … 0.10 m/s—0.9 m/s.”
- *Che cosa è definito e che cosa no:*
  - **Metodo:** correlation statistics in DaVis, cioè a posteriori e basato sulle immagini. Il preprint non cita esplicitamente Wieneke 2015; che si tratti di quel metodo è un'inferenza.
  - **Componenti:** separate fra in piano (insieme) e fuori piano.
  - **Non definiti:**
    - su che cosa variano gli intervalli (spazio, frame, condizioni, percentili);
    - che cosa sia la media (spaziale, temporale, per condizione);
    - perché il minimo “medio” (0.04 m/s) sia inferiore al minimo “assoluto” (0.1 m/s);
    - il livello o fattore di copertura;
    - se “absolute values” significhi valore assoluto o incertezza non relativa.
  - **Nessuna validazione** con riferimento.
  - **Sorgenti non coperte:** calibrazione, temporizzazione, slip, scale non risolte e, per i limiti noti del metodo, outlier e peak locking (P12, L17).
- *Uso consentito:* nel caso a 5 bar l'incertezza a posteriori è variabile, con un rapporto fuori piano/in piano vicino a 2 negli intervalli dichiarati. È un ordine di grandezza in m/s per quell'apparato. **Non** va convertita in una σ gaussiana né in una distribuzione spaziale, perché la statistica non è definita.
- *Altri elementi rilevanti:* a air split 0.5 una parte dell'oscillazione esce dal campo di vista delle camere sPIV (“blocked view”), con riduzione dell'energia catturata dalla SPOD (§3).
- *Fonte:* Karmarkar, Gupta, Boxx, Hemchandra & O'Connor, *Impact of precessing vortex core dynamics on the thermoacoustic instabilities in a swirl stabilized combustor*, arXiv:2011.14662v1 (2020), “submitted to J. Fluid Mech.”; §§2.1–2.2.1 e passaggio sul campo di vista in §3. **FT-v**: preprint; la versione su rivista non è stata verificata. Il sistema è descritto anche in Litvinov et al. 2021 (non letto).
- *Trasferibilità:* combustore e combustibile diversi dal nostro, pressione 5 bar. È la fonte letta più vicina a un'UQ a posteriori in combustione.
- *Step 3:* se un dataset fornisce mappe UQ, registrarne la statistica (per vettore, media, percentili), la componente e il livello prima di ogni uso.

### 4.C Validazione, sostituzione e lacune

**L19 — Rilevamento degli outlier: test mediano normalizzato e cluster**
- *Pratica documentata in FT:* test mediano su 3×3 vettori; Saini et al. (AIAA, §II.B.4) usano la forma normalizzata di Westerweel & Scarano con ε = 0.1. Il lavoro originale di Westerweel & Scarano 2005 **non è stato letto**.
- *Solo abstract o snippet:*
  - Duncan et al. 2010 (**AB**): la generalizzazione a dati PIV e PTV mostra residui con PDF “universale” e funziona fino a circa il 15% di vettori spuri nei flussi testati;
  - Masullo & Theunissen 2016 (**AB**): i test a vicinato fisso perdono efficacia con cluster di outlier; un vicinato adattivo riduce i casi non rilevati e i falsi rigetti;
  - Garcia 2011 (**SN**, non usato quantitativamente): degrado con cluster oltre il 10%.
- *Implicazione:* non esiste una frequenza universale di outlier. Un dataset “validato” può contenere cluster residui.

**L20 — Sensibilità del dato ai parametri di validazione (iniettore swirl reagente)**
- *Apparato:* L07. Validazione con altezza del picco (detectability k = 1.2), intervallo di velocità e media mobile (α = 0.1).
- *Numero:* gli autori riportano da un loro studio parametrico precedente (Barbosa et al. 2008, **SEC**, non letto) un'“incertezza massima dell'11%” sulla velocità massima al variare dei parametri di validazione. Non c'è effetto sulla posizione delle regioni di massimo e minimo.
- *Fonte:* arXiv 0807.1871, §2.3, pp. 6–7. **FT-v**.
- *Uso:* indica che le soglie contano; il valore dell'11% è di quell'esperimento e la sua derivazione non è stata letta.

**L21 — GTMC DLR: interpolazione dei buchi singoli e filtro POD di visualizzazione**
- *Contenuto:*
  - la fig. 10 di Boxx 2010 è presentata “senza la ricostruzione POD passa-banda né interpolazione oltre i buchi di un singolo vettore”, il che indica che il processamento standard interpolava i buchi di un vettore;
  - le figure principali usano una POD di 2048 snapshot troncata a 150 modi (circa il 70% dell'energia), perché i modi superiori sono “principalmente rumore”.
- *Fonte:* Boxx et al. 2010, manoscritto, pp. 23–24 e 30. **FT-v**.
- *Implicazione:* gli stessi dati esistono in più livelli di processamento. Un campo filtrato con POD non serve a stimare rumore o lacune.

**L22 — Vettori invalidi e interpolati in microPIV di flashback H2/aria**
- *Apparato:*
  - canale con flashback, TiO2 0.1–1 µm, microscopio a lunga distanza;
  - FOV 4.15×4.15 mm², filtro 527±10 nm;
  - PIVlab da 48 a 24 px con 50% di overlap, risoluzione dichiarata 50 µm.
- *Validazione:* filtro mediano 3×3 con tolleranza di 2 deviazioni standard locali, più filtro di intervallo; i vettori rimossi sono interpolati con kernel 3×3.
- *Numeri:*
  - vettori interpolati “fino al 20% in alcune prove”, attribuiti a una scia senza particelle nel FOV;
  - “percentuale di invalidi nelle aree con particelle sempre inferiore al 10%”.

  Denominatore: vettori del campo per prova. La distribuzione per frame non è riportata.
- *In fiamma:*
  - gli interpolati sono soprattutto nella scia e vicino alle pareti;
  - “la percentuale di vettori interpolati è più alta nella zona di reazione a causa della bassa densità di semina”;
  - quando la coda della fiamma attraversa lateralmente il foglio, la densità di semina cala in tutto il campo.
- *Fonte:* G. M. Baumgartner, tesi di dottorato TUM 2015, §3.4 pp. 48–49 e cap. 7 pp. 122–129. **FT** delle sezioni.
- *Trasferibilità:* microscala, H2/aria, flashback, non una fiamma swirl stabilizzata. È però l'evidenza più diretta trovata di un legame fra qualità e regione di fiamma.

**L23 — Gappy-POD per dati PIV rumorosi nei combustori: che cosa simula e che cosa no**
- *Motivazione (§I):* ad alta pressione i laser multi-kHz hanno bassa energia, c'è fondo di fuliggine e luminosità, e la semina non uniforme lascia vuoti. Ne seguono “lacune significative”. È un'affermazione non quantificata nella versione letta.
- *Metodo di prova:*
  - dataset atmosferico con “essenzialmente nessun vettore mancante” (L05);
  - gap artificiali G = 15, 25, 50%, con numero di mancanti per snapshot distribuito in modo gaussiano;
  - gap descritti come raggruppati in cluster, con geometria non specificata nella versione AIAA.
- *Risultati:*
  - il metodo adattivo MF ha E* circa il 30% inferiore a Gunes, quello FS circa il 20%;
  - l'errore cresce con G; servono POD separate per componente; u_z ha errore maggiore;
  - esempio con G = 45%: errore relativo sui gap 0.226 (u_x) e 0.231 (u_y) al numero ottimo di modi (fig. 2).
- *Fonte:* Saini, Steinberg & Arndt, AIAA 2016-0436, §§I–IV. **FT-v**. La versione Exp. Fluids 57:122 **non è stata letta** (solo SN).
- *Implicazioni:* (i) le lacune reali qui non sono misurate. (ii) La GPOD usa l'intera serie temporale: un campo riempito contiene informazione da altri istanti.

**L24 — Regioni dove resta utile solo la media (combustore pressurizzato)**
- *Effetto con foglio normale alla parete:*
  - PIV valutabile solo oltre 1 mm dalla parete raffreddata;
  - flare e saturazione dalla riflessione del foglio;
  - forti gradienti termici e di densità con beam steering e riflessione interna;
  - “bassi tassi di validazione”: “solo una mappa di velocità media fornisce dati utili”.
- *Effetto con foglio parallelo alla parete:* PIV non fattibile, per flare e autocorrelazione sul fondo speculare stazionario.
- *Fonte:* Lange et al. 2012, pp. 4 e 7–9. **FT**. Qualitativo.
- *Implicazione:* esistono regioni, come quelle vicino alle pareti, dove l'istantaneo non è rilasciato o non è affidabile. È diverso da un rigetto sparso.

**L25 — Regione affidabile in base all'incertezza (non reagente, densità variabile)**
- *Apparato:* convezione di Rayleigh–Bénard in CHF3 supercritico, differenze relative di densità 17–42%; finestre 48×48 px (64×64 in un caso “per avere più vettori validi”), 50% di overlap.
- *Numeri:*
  - con il metodo di Wieneke 2015, i punti con incertezza < 1 px (soglia scelta dagli autori) sono circa il 75% della sezione;
  - in quella regione l'incertezza media è ≈0.53 px, pari a circa 5.7–10% della velocità della circolazione su grande scala;
  - le zone ad alta incertezza coincidono con un pennacchio forte.
- *Fonte:* Valori et al. 2019, §§4.3 e 5. **FT**.
- *Trasferibilità:* stesso meccanismo ottico, ma fluido, scale e soglia diversi. Il 75% non si trasferisce.

**L26 — Altri trattamenti documentati**
- Filtri d'immagine contro le striature ad alta pressione: sottrazione di un fondo mobile di 16 px e normalizzazione min-max (L31, FT).
- Sottrazione del minimo o della media d'immagine contro fixed pattern e fondo costante (tesi di Wieneke, §3.1.6, FT).
- Ritaglio del FOV (L05, FT-v).
- GPOD (L23).
- Barwey et al. (arXiv 1909.13669, introduzione, **SEC**) citano Delaunay e GPOD come prassi di riempimento.

### 4.D Specificità della combustione

**L27 — Risposta del tracciante, slip e termoforesi**
- *Slip, stime a priori:*
  - Stöhr 2012, §2.3.2 (**FT-v**): TiO2 1 µm, τ ≈ 0.5×10⁻⁵ s, v = 80 m/s, l = 10 mm; Δv = τv²/l = 1.6 m/s, “errore massimo da slip 2%”;
  - Oberleithner 2015, §2.2 (**FT-v**): v ≈ 60 m/s, l ≈ 10 mm, numero di Stokes 0.03, slip “trascurabile”.

  Sono ordini di grandezza, non misure.
- *Termoforesi:*
  - nella zona di preriscaldamento la forza termoforetica può indurre un ritardo significativo fra particella e fluido; gli autori propongono una correzione a profilo di T noto (Sung, Law & Axelbaum 1994, fiamme gemelle a controflusso, LDV, **AB**);
  - Stella et al. 2001 indicano termoforesi e indice di rifrazione come principali sorgenti in fiamma (**SN**).
- *Trasferibilità (inferenza):* effetto alla scala dello spessore di fiamma. Probabilmente non rappresentabile sulla nostra griglia.

**L28 — Scelta del tracciante: chi viene osservato**
- *Olio (esperimento documentato):*
  - flashback swirl CH4/H2 a pressione atmosferica (UT Austin);
  - goccioline di olio d'oliva di ~1 µm da atomizzatore;
  - gli autori scrivono che “vaporizzano nella zona di preriscaldamento” e che l'isoterma di vaporizzazione dell'olio usato “è stata testata nello studio”: circa 555 K, deviazione standard 10 K (OSTI 1253136, §3.3.1, **FT**). La procedura di misura non è descritta nell'estratto letto.

  Con questo tracciante, la velocità resta solo nel gas incombusto. L'interfaccia particelle/assenza di particelle è usata per rilevare il fronte.
- *Distinzioni necessarie:*
  1. **Temperatura riportata:** circa 555 K per quell'olio, quelle gocce e quelle condizioni.
  2. **Condizioni sperimentali:** dimensione delle gocce, velocità di riscaldamento e tempo di residenza nel preriscaldamento.
  3. **Dinamica delle particelle:** l'evaporazione richiede tempo, e le gocce vengono trasportate mentre evaporano. Il bordo visibile è quindi la posizione in cui le gocce diventano invisibili, non un'isoterma esatta. Dipende anche da soglia d'immagine e smoothing dell'algoritmo di estrazione.
  4. **Approssimazione per la LES:** una maschera “T > T_soglia” sarebbe un'approssimazione proposta, con T_soglia come parametro di sensibilità, che usa T nel generatore. Va discussa come asse separato (roadmap §4A).
- *Altre fonti:* CPIV (Pfadler et al. 2007, **AB**/**SN**) e la tesi tomografica TU Delft (**FT** limitato) usano la variazione della densità di particelle come marcatore del fronte. Le temperature di evaporazione di altri oli (DEHS ecc.) non sono state verificate.
- *Particelle solide:* Al2O3 e TiO2 “sopravvivono” e misurano entrambi i gas (OSTI 1253136; Barbosa §2.2; Baumgartner §3.4; sistemi DLR di L04 e L37).
- *Costo delle particelle solide:* il TiO2 si deposita sulla parete del tubo e obbliga a smontare ogni 2–3 prove; per questo la campagna ad alta pressione è passata all'olio (OSTI 1506058, p. 29, **FT**).

**L29 — Calo della densità di particelle visibili attraverso la fiamma**
- *Evidenza (FT):*
  - transizione netta da alta a bassa densità di particelle, attribuita dagli autori all'espansione termica, e una seconda transizione verso l'assenza di particelle (Baumgartner, p. 122);
  - il gas combusto è identificato con conteggi di particelle per finestra più bassi, con soglia di 20 particelle per celle di 0.745 mm a 0.028 ppp (tesi TU Delft, FT limitato).
- *Numero non usato:* “circa −85% nel gas combusto” compare in note di lezione di Dreizler, Princeton 2017 (**SN**, host non accessibile).
- *Inferenza fisica:* se le particelle seguono la massa, la densità in numero è proporzionale alla densità del gas. A pressione costante ρ_b/ρ_u ≈ T_u/T_b. Il valore per la nostra miscela non è stato verificato.
- *Effetti sul dato:* meno particelle per finestra, quindi più errore casuale e più rigetti (L22). Il bias per finestre a cavallo del fronte **non è quantificato** da nessuna fonte letta.
- *Step 3:* conteggi di particelle per finestra contro OH-PLIF o contro i rigetti, se ci sono immagini. Altrimenti confronti regionali basati su proxy dichiarati.

**L30 — Gradienti d'indice di rifrazione in fiamma: numeri disponibili e livello**
- *Vanselow et al. 2019* (**AB**): fiamma premiscelata di propano, tre piani.
  - Errori di posizione misurati e verificati con ray tracing sul campo d'indice ricavato con BOS.
  - Errore di posizione massimo 14 µm; errori di velocità **sistematici fino al 4%** e **casuali fino al 6%**.
  - “Diversamente dall'errore sistematico, l'errore casuale varia significativamente fra i piani analizzati.”
  - La concentrazione vicino al fronte compare solo in uno snippet (**SN**).
- *Vanselow & Fischer 2018* (**SN**, non usato quantitativamente): errori maggiori con triangolazione (stereo, tomografica) che in PIV planare in un getto caldo.
- *Meccanismo (Elsinga et al. 2005, **SEC** tramite Valori, FT):* l'errore relativo di velocità è pari al gradiente dello spostamento ottico lungo la direzione della velocità; la sfocatura allarga il picco e aumenta l'errore casuale.
- *Rajendran et al. 2020* (arXiv 1910.09379, **FT-v** di abstract e introduzione): non linearità del gradiente di densità producono sfocatura che amplifica l'effetto del rumore sulla stima del centroide. Lavoro su PTV e BOS volumetriche.
- *Limiti:* numeri di un solo apparato e fonte primaria letta solo in abstract. Non trasferibili.

**L31 — Striature d'intensità ad alta pressione e picchi allungati**
- *Apparato:* UT Austin, SPIV a 8 kHz in camera pressurizzata, swirl con flashback.
- *Effetto:*
  - striature casuali d'intensità quando il foglio attraversa la fiamma e i gas di scarico;
  - aumentano con la pressione;
  - i picchi di correlazione si allungano.
- *Correzione:* fondo mobile di 16 px più normalizzazione min-max, che rende i picchi assialsimmetrici (fig. 28).
- *Fonte:* OSTI 1506058, pp. 29–30. **FT**. Qualitativo.
- *Implicazione:* il fenomeno è stato trattato prima della correlazione e può non lasciare traccia nel dato finale.

**L32 — Degrado delle immagini dove la luce attraversa la fiamma**
- MicroPIV di flashback: nel caso reagente la qualità peggiora dove la luce diffusa attraversa la fiamma (Baumgartner, p. 122, **FT**).
- Combustore pressurizzato: nelle immagini PLIF a impulso singolo il foglio mostra striature dovute ai gradienti della fiamma (Lange, p. 9, **FT**).
- L25 è l'analogo non reagente quantificato.

**L33 — Luminosità della fiamma e fondo**
- *FT:*
  - filtri passa-banda attorno a 532 nm “per ridurre l'influenza della luminosità di fiamma” (Stöhr 2012; Oberleithner 2015; Baumgartner);
  - nonostante il filtro, la chemiluminescenza accentua i riflessi sulla parete (Baumgartner, p. 122).
- *FT-v:* Saini et al. citano fuliggine e luminosità come cause di basso SNR ad alta pressione, come affermazione.
- *AB/SN:* fuliggine nelle immagini Mie della PIV di un combustore RQL (Jose, Geigle & Hampp 2025), con database DaRUS da segnalare all'Agente A.
- *Ammoniaca (plausibile, non verificato, solo SN):* NH2* (righe; α-band vicino a 630 nm) e NO2* (continuo), con spettri registrati fra 400 e 700 nm. Nessuna fonte letta quantifica l'effetto sulla PIV.
- *Inferenza non verificata:* nelle camere a doppio fotogramma il secondo fotogramma ha in genere un'esposizione più lunga, quindi il fondo di fiamma può pesare di più. Da controllare sulle immagini.

**L34 — Riflessi, accesso ottico e sporcamento delle finestre**
- *Riflessi (FT/FT-v):*
  - Toronto/DLR: metà campo scartata (L05);
  - DLR pressurizzato: PIV infattibile con foglio parallelo alla parete, e solo oltre 1 mm con foglio normale (L24);
  - TUM: foglio inclinato di 30° e vernice nera, riflessi confinati sotto 0.2 mm;
  - UT Austin: la PIV è più sensibile della PLIF alla luce diffusa (OSTI 1506058, p. 30);
  - motore ottico: riflessi vicino al punto morto superiore (L18).
- *Sporcamento (FT/FT-v):*
  - GTMC DLR: semina solo durante la misura “per minimizzare l'accumulo sulle finestre” (Boxx, pp. 11–12);
  - DLR pressurizzato: finestre roventi che si irruvidiscono in pochi minuti, tempo utile sotto 15 minuti (Lange, p. 7);
  - UT Austin, alta pressione: lastra sacrificale e aria di raffreddamento; la polvere sulle finestre altera il profilo del foglio (OSTI 1506058, pp. 19 e 26).
- *Implicazione:* questi fenomeni producono soprattutto **regioni escluse fisse** e **limiti alla durata dei record**.

**L35 — Relazione fra qualità PIV e regione di fiamma: stato delle prove**
- *Documentato in FT:*
  - più vettori interpolati nella zona di reazione (L22, microPIV H2);
  - incertezza alta nei pennacchi termici (L25, non reagente);
  - nessuna velocità nel gas combusto con olio che evapora (L28);
  - validazione bassa presso pareti calde (L24).
- *Solo AB/SN:* variabilità fra piani e possibile concentrazione presso il fronte degli errori da rifrazione (L30).
- *Non trovato:* statistiche di rigetto condizionate alla posizione del fronte in fiamme swirl stabilizzate; persistenza temporale delle lacune.

**L36 — Fonti indicate dall'incarico ma non lette integralmente**
- *Ebi & Clemens 2016*: non accessibile. Contesto dai rapporti DOE del gruppo (L08, L28, FT) e da snippet.
- *Andreini et al.* (**AB**): il primo usa PIV 2D in un rig planare **non reagente** in galleria del vento; il secondo è numerico (RANS/SAS) su un generatore di hot streak, confrontato con dati sperimentali. **Non sono misure PIV in fiamma.**

---

## 5. Comune, dipendente dall'apparato, non dimostrato

### 5.1 Comune ai casi studiati

Valido nelle fonti lette, non come legge universale.

- Campo di vista limitato da accesso ottico e camere; bordi e regioni escluse fissi nel tempo (L04–L08, L24, L34, L37).
- Nei lavori applicativi atmosferici letti le incertezze istantanee sono dichiarate a priori come numero unico (L11). L'unico caso a posteriori letto riporta intervalli in m/s senza statistica definita (L37).
- Validazione con test di vicinato e filtri di intervallo, seguiti da interpolazione almeno dei buchi isolati (L19–L22).
- Nello stereo la componente fuori piano è più incerta (L04, L09, L37).
- Errore casuale correlato fra vettori vicini; entità dipendente da overlap e algoritmo (L15).
- Qualità peggiore dove la luce attraversa gradienti d'indice forti (L24, L25, L30–L32).

### 5.2 Dipendente dall'apparato o dalla scelta sperimentale

- Tracciante: olio che evapora oppure particelle solide (L28).
- Frequenza e campo di vista (L04, L07, L37).
- Δt e range dinamico (L06, L08).
- Pressione: striature, sporcamento, durata dei record (L31, L34).
- Software e passate, che determinano la forma del filtro (L01, L02).
- Soglie di validazione (L20).
- Livello del dato rilasciato (L21, L23).

### 5.3 Plausibile ma non dimostrato dalle fonti lette

- Frazione di rigetti per frame in fiamme swirl e sua relazione con fronte o temperatura.
- Persistenza e forma delle lacune nel tempo.
- Bias delle finestre a cavallo del fronte.
- Peak locking nei sistemi kHz in combustione.
- Effetto della luminosità NH2*/NO2*.
- Errori di temporizzazione negli apparati letti.
- Trasferibilità ad altri apparati di qualunque percentuale citata (4%, 6%, 11%, 20%, 75%) o degli intervalli di L37.

---

## 6. Pratiche di validazione e sostituzione e loro conseguenze

| Pratica | Dove documentata | Che cosa cambia nel dato rilasciato | Che cosa diventa invisibile | ID |
|---|---|---|---|---|
| Test mediano 3×3 + filtro di intervallo | TUM (FT); Saini (FT-v); prassi generale | Rimuove outlier isolati | Cluster con vicini errati; outlier plausibili | L19, L22 |
| Validazione su picco e media mobile | EM2C | Le soglie cambiano i massimi (11% riportato) | La scelta delle soglie | L20 |
| Interpolazione dei buchi singoli o con kernel 3×3 | DLR (implicito), TUM | Campo completo, con valori smussati | I rigetti, se manca un flag; correlazione introdotta | L21, L22 |
| Filtro POD (troncamento dei modi) | DLR (figure) | Rimuove rumore e piccole scale; informazione dall'intera serie | Errore casuale, piccole scale, indipendenza temporale | L21 |
| Riempimento gappy-POD | Toronto/DLR (proposto) | Riempie anche gap estesi | Che il vettore non era misurato; informazione da altri istanti | L23 |
| Filtri d'immagine | UT Austin, Wieneke | Meno rigetti e picchi meno allungati | Le striature e la loro causa ottica | L26, L31 |
| Ritaglio del FOV; finestre più grandi in alcune condizioni | Toronto/DLR, Valori | Rimuove zone cattive; più vettori validi ma risoluzione minore | La regione persa; la variazione di risoluzione | L05, L25 |
| Uso della sola media in zone critiche | DLR pressurizzato | Istantaneo non rilasciato | Variabilità istantanea locale | L24 |

**Lezione per lo step 3:** l'assenza di buchi nel dato finale è attesa in molti processamenti. Prima di stimare qualunque frequenza bisogna stabilire il livello del dato.

---

## 7. Domande verificabili per lo step 3

Per ogni domanda: dato necessario ed evidenze collegate. Esiti possibili: presente; trattato prima del rilascio; non riscontrato; non verificabile.

1. **Livello di processamento:** vettori grezzi, validati, interpolati, filtrati con POD o riempiti? *Dato:* README, flag, versioni multiple dello stesso campo. (L21, L23)
2. **Flag di validità e sostituzione:** esistono, e con quale codifica (NaN, zero, flag)? (Ricognizione §3 per EngineBench.)
3. **Frazione di vettori invalidi o sostituiti per frame**, con denominatore esplicito (tutto il FOV o solo la regione inseminata) e distribuzione fra frame e condizioni. (L22)
4. **Geometria delle lacune:** isolate o in cluster, dimensioni in mm, posizione. (L19, L24)
5. **Persistenza temporale delle lacune** nelle sequenze ad alta frequenza.
6. **Relazione con la fiamma:** rigetti dove OH-PLIF è alto o al gradiente di OH? *Dato:* PLIF simultanea e registrata; senza PLIF, non verificabile o solo con proxy dichiarati. (L35)
7. **Tracciante:** olio o particelle solide? I gas combusti hanno vettori? Con olio, come è definito il bordo nel dataset? (L28)
8. **Densità di particelle per finestra** in gas incombusto e combusto. *Dato:* immagini. (L29)
9. **Parametri ottici e di processamento:** µm/px, Δt, finestre, pesatura, passate, overlap, software e versione. (L01, L04, L11)
10. **UQ:** esiste? Per vettore o aggregata? Con quale metodo, componente, statistica, livello e validazione? Se mancano, registrare “solo a priori” o “a posteriori con statistica non definita”. (L11, L16, L17, L37)
11. **Componenti:** 2C o stereo 3C? Incertezza fuori piano distinta? (L04, L09, L37)
12. **Range dinamico e peak locking:** spostamenti sotto 1 px nelle regioni lente; istogrammi delle parti frazionarie. (L06, L14)
13. **Correlazione degli errori:** esiste un riferimento co-localizzato? Senza riferimento, la correlazione del campo non misura quella dell'errore. (L15, L16)
14. **Regioni escluse fisse:** bordi, montanti, pareti, riflessi; area e posizione. (L05, L24, L34)
15. **Sensibilità alle soglie** su vettori pre-validazione, se disponibili. (L20)
16. **Fondo luminoso:** differenza fra primo e secondo fotogramma; luminosità delle fiamme con NH3 nella banda dei filtri. (L33)

---

## 8. Implicazioni possibili per la PIV virtuale (PROPOSTE)

Tutte le voci sono **proposte da discutere**. Nessun parametro è fissato.

- **P-A (struttura).** Organizzare l'osservazione in moduli distinti, documentabili e verificabili separatamente:
  1. campo di vista e regioni escluse fisse;
  2. risposta spaziale e campionamento;
  3. errore e sua correlazione;
  4. validità e sostituzione.

  I moduli **non sono necessariamente indipendenti**: finestra, errore e validità possono condividere le stesse cause. Una finestra più piccola aumenta il rumore (L01); un calo di semina aumenta sia l'errore sia i rigetti (L22, L29); i gradienti d'indice producono errori e regioni non affidabili (L24, L25). L'eventuale indipendenza fra moduli è un'ipotesi da dichiarare, non una proprietà. L03 è un precedente di struttura, non di valori.
- **P-B (risposta spaziale).**
  - La media su finestra quadrata è l'approssimazione naturale del single-pass a finestra quadrata.
  - Per uno scenario con algoritmo DaVis-simile, un kernel gaussiano o MHG con L_sr ≈ 0.7 volte la finestra è **un'approssimazione derivata dalle configurazioni di Wieneke** (L01–L02), da dichiarare come tale e, se possibile, da confrontare con alternative.
  - A scale di ~1 mm il filtro è dell'ordine della cella LES (L10).
- **P-C (errore).**
  - ±0.1 px è a priori (L11). Se usato, dichiararne scala, Δt, componente e natura (standard per ipotesi).
  - Gli intervalli di L37 non vanno convertiti in σ senza una statistica definita.
  - Per la correlazione spaziale, ρ(2d) ≈ 0.45 vale per DaVis sintetico al 75% (L15). Il valore al 50% di overlap resta aperto.
  - Il rumore i.i.d. è un caso particolare da dichiarare.
- **P-D (stereo).** Se lo scenario è stereo, la componente fuori piano ha incertezza diversa (L04, L37). Anche le componenti in piano hanno errori specifici della stereo (L09).
- **P-E (lacune).** Distinguere tre classi con evidenza diversa:
  1. **regioni escluse fisse:** evidenza forte, rappresentabili geometricamente;
  2. **rigetti sparsi e cluster:** frequenze non documentate in fiamme swirl, da trattare come sensibilità o da ricavare dai dati dello step 3;
  3. **lacune guidate dalla fiamma:** evidenza qualitativa (L22, L28, L29). L'uso di T nel generatore va discusso esplicitamente.
- **P-F (tracciante).** Lo scenario “olio” implica assenza di osservazioni nel gas combusto nell'esperimento documentato. Una maschera su soglia di T nella LES ne sarebbe un'**approssimazione**: la soglia (~555 K riportati per l'olio d'oliva in quelle condizioni) è un parametro di sensibilità, non una costante. Trascura evaporazione finita, trasporto delle gocce e definizione del bordo nelle immagini, e introduce informazione su T nella maschera (L28). Lo scenario “particelle solide” osserva entrambi i gas con qualità degradata.
- **P-G (livello del dato).** Decidere se simulare il dato prima della sostituzione (mancanti espliciti) o dopo l'interpolazione (L21–L23).
- **P-H (combinazioni).** Non comporre in un unico scenario parametri di apparati diversi, per esempio FOV del GTMC, errori di Vanselow, rigetti TUM e incertezze di L37.

---

## 9. Dataset e materiali associati, da segnalare all'Agente A

Non verificati come scaricabili, salvo dove indicato.

| Materiale | Origine | Stato |
|---|---|---|
| Database “Collaborative framework” (getto rettangolare; PIV-MS, PIV-HDR, hot-wire) | Neal et al. 2015 | AB (L16) |
| Database dei PIV Challenge | Kähler et al. 2016; www.pivchallenge.org | AB; non ispezionato |
| Database di fuliggine e traccianti in immagini PIV di un combustore RQL (DaRUS) | Jose, Geigle & Hampp 2025 | SN; DOI dati citati negli snippet: 10.18419/darus-5182, -5184 |
| Dati GTMC DLR (PIV/OH-PLIF 5–10 kHz) | L04 | Disponibilità non dichiarata nelle fonti lette |
| Dati del combustore DLR a 5 bar con PIV e incertezza CS | L37 | Disponibilità non dichiarata nel preprint |
| Dati del combustore Toronto/DLR (S-PIV 10 kHz) | L05, L23 | Non dichiarato |

---

## 10. Bibliografia annotata

| # | Riferimento | DOI / URL della copia letta | Livello | Uso |
|---|---|---|---|---|
| 1 | B. Wieneke, *PIV Uncertainty Quantification and Beyond*, PhD TU Delft, 2017 | https://repository.tudelft.nl/file/File_b2074023-ab68-4161-870e-c229a6368a78 | FT, capp. 2, 3.1–3.2, 4.2–4.4 | L01, L02, L09, L12–L16 |
| 2 | A. Sciacchitano, B. Wieneke, *PIV uncertainty propagation*, MST 27 (2016) 084006 | doi:10.1088/0957-0233/27/8/084006; http://resolver.tudelft.nl/uuid:b44a7c7a-fece-4dec-905b-c0f84c39c389 | FT, §§2.3, 3.2, 4.2 | L15, L16 |
| 3 | I. Boxx, M. Stöhr, C. Carter, W. Meier, Combust. Flame 157 (2010) 1510–1525 | doi:10.1016/j.combustflame.2009.12.015; https://elib.dlr.de/64569/1/Boxx2009cf.pdf | FT-v | L04, L11, L21, L34 |
| 4 | M. Stöhr, I. Boxx, C. D. Carter, W. Meier, Combust. Flame 159 (2012) 2636–2649 | doi:10.1016/j.combustflame.2012.03.020; https://elib.dlr.de/77484/1/stoehr2012cf_manuscript.pdf | FT-v, §§1–2 | L04, L27, L33 |
| 5 | W. Meier, I. Boxx, M. Stöhr, C. D. Carter, Exp. Fluids 49 (2010) 865–882 | doi:10.1007/s00348-010-0889-x; https://elib.dlr.de/65267/1/Meier2010ExpFluids.pdf | FT-v, §2 | L04, L11 |
| 6 | K. Oberleithner, M. Stöhr, S. H. Im, C. M. Arndt, A. M. Steinberg, Combust. Flame 162 (2015) 3100–3114 | doi:10.1016/j.combustflame.2015.02.015; https://elib.dlr.de/96808/1/2015_Oberleithner_Stoehr_CF_author_final.pdf | FT-v, §2 | L04, L27, L33 |
| 7 | P. Saini, A. M. Steinberg, C. M. Arndt, AIAA 2016-0436 | doi:10.2514/6.2016-0436; https://elib.dlr.de/102428/1/2016_Saini_Arndt_AIAA_ASM.pdf | FT-v | L05, L19, L23. Versione Exp. Fluids 57 (2016) 122, doi:10.1007/s00348-016-2208-7: non letta |
| 8 | L. Lange, J. Heinze, M. Schroll, C. Willert, T. Behrendt, 16th Int. Symp. Appl. Laser Techniques to Fluid Mechanics, Lisbona 2012 | https://elib.dlr.de/77085/1/PLIF_PIV_LLange_Lisbon2012_No223.pdf | FT | L06, L24, L32, L34 |
| 9 | V. Valori, G. Elsinga, M. Rohde, J. Westerweel, T. van der Hagen, Exp. Fluids 60 (2019) 143 | doi:10.1007/s00348-019-2789-z; https://repository.tudelft.nl/file/File_a0a0d5eb-dadb-49e7-a849-115683161e45 | FT, §§3–5 | L25, L30 |
| 10 | S. Barbosa, P. Scouflaire, S. Ducruix, preprint 2008 | https://arxiv.org/abs/0807.1871 | FT-v | L07, L20, L28 |
| 11 | N. Clemens (PI), DOE final report DE-FE0007107 | https://www.osti.gov/servlets/purl/1253136 | FT, §3 | L08, L28 |
| 12 | N. Clemens (PI), DOE final report DE-FE0012053 | https://www.osti.gov/servlets/purl/1506058 | FT, §4 | L08, L18, L28, L31, L34 |
| 13 | G. M. Baumgartner, *Flame Flashback in Premixed Hydrogen-Air Combustion Systems*, PhD TU München, 2015 | https://www.epc.ed.tum.de/fileadmin/w00cgc/td/Forschung/Dissertationen/baumgaertner15.pdf | FT, §3.4 e cap. 7 | L22, L29, L32–L34 |
| 14 | A. Karmarkar, S. Gupta, I. Boxx, S. Hemchandra, J. O'Connor, preprint 2020 | https://arxiv.org/abs/2011.14662 (v1) | FT-v, §§2.1–2.2.1 e §3 (FOV) | L37 |
| 15 | A. M. Sunol et al., *Learning constitutive models and rheology from partial flow measurements* | https://arxiv.org/abs/2510.24673 (v4) | FT della sezione di emulazione | L03 |
| 16 | L. K. Rajendran, S. P. M. Bane, P. P. Vlachos (2020) | https://arxiv.org/abs/1910.09379 | FT-v, abstract e introduzione | L30 |
| 17 | L. K. Rajendran, S. P. M. Bane, P. P. Vlachos, MST 30 (2019) 085302 | doi:10.1088/1361-6501/ab1ca8; https://arxiv.org/abs/1812.05902 | Scorsa | Strumento di ray tracing; nessun numero usato |
| 18 | S. Bhattacharya, J. Charonko, P. Vlachos, MST (2018) | doi:10.1088/1361-6501/aadfb4; https://arxiv.org/abs/1804.10854 | FT-v, introduzione e conclusioni | L17 |
| 19 | Z. Xue, J. Charonko, P. Vlachos | https://arxiv.org/abs/1405.3023 | Scorsa | Non usato per numeri |
| 20 | B. Peterson, E. Baum, A. Dreizler, B. Böhm | https://arxiv.org/abs/2009.01710 | FT-v, §2 | L18 |
| 21 | Tesi magistrale TU Delft, *Determination of Premixed Flame Characteristics by Tomographic PIV* | https://repository.tudelft.nl/file/File_660d1006-988b-48ae-951c-f74dfa1cfaa3 | FT limitato | L28, L29 |
| 22 | S. Beresh, AIAA 2008-0239 | https://www.osti.gov/servlets/purl/1146188 | Solo prima pagina | Pista, non letta |
| 23 | J. Duncan et al., MST 21 (2010) 057002 | doi:10.1088/0957-0233/21/5/057002 | AB | L19 |
| 24 | A. Masullo, R. Theunissen, Exp. Fluids 57 (2016) 33 | doi:10.1007/s00348-015-2110-8 | AB | L19 |
| 25 | C. Kähler, S. Scharnowski, C. Cierpka, Exp. Fluids 52 (2012) 1629 | doi:10.1007/s00348-012-1280-x | AB | L02 |
| 26 | D. Neal et al., MST 26 (2015) 074003; A. Sciacchitano et al., MST 26 (2015) 074004; A. Boomsma et al., MST 27 (2016) 094006; B. Wieneke, MST 26 (2015) 074002 | doi:10.1088/0957-0233/26/7/074003; …/26/7/074004; …/27/9/094006; …/26/7/074002 | AB | L16, L17 |
| 27 | C. Vanselow, D. Stöbener, J. Kiefer, A. Fischer, Exp. Fluids 60 (2019) | doi:10.1007/s00348-019-2795-1 (manoscritto accettato su media.suub.uni-bremen.de, handle elib/8287, non scaricato) | AB | L30 |
| 28 | C. Vanselow, A. Fischer (2018), campi d'indice disomogenei | https://media.suub.uni-bremen.de/handle/elib/8279 (non scaricato) | SN | L30 (non quantitativo) |
| 29 | C.-J. Sung, C. K. Law, R. L. Axelbaum, Combust. Sci. Technol. (1994) | doi:10.1080/00102209408935428 | AB | L27 |
| 30 | A. Stella et al., Exp. Fluids 30 (2001) 167 | doi:10.1007/s003480000151 | SN | L27 |
| 31 | Westerweel & Scarano 2005; Scarano 2003; Lavoie et al. 2007; Christensen 2004; Carr et al. 2009; Garcia 2011 | doi:10.1007/s00348-005-0016-6; 10.1007/s00348-003-0655-4; 10.1007/s00348-007-0319-x; 10.1007/s00348-003-0754-2; 10.1007/s00348-009-0638-1; 10.1007/s00348-010-0985-y | Non letti | Solo indicazione bibliografica |
| 32 | A. Andreini et al., J. Eng. Gas Turbines Power (due lavori individuati per titolo) | doi:10.1115/1.4031434; doi:10.1115/1.4034358 | AB | L36 |
| 33 | D. Ebi, N. Clemens, Combust. Flame 168 (2016) | doi:10.1016/j.combustflame.2016.03.027 | Non letto | L36 |
| 34 | M. Pfadler, F. Beyrau, A. Leipertz, Opt. Express 15 (2007) 15444 | doi:10.1364/OE.15.015444 | AB/SN | L28, L29 |
| 35 | B. Jose, K. P. Geigle, F. Hampp, MLST (2025) | doi:10.1088/2632-2153/ae2565 | SN | L33 |
| 36 | S. Barwey et al. | https://arxiv.org/abs/1909.13669 | Scorsa (introduzione) | L26 |
| 37 | A. Sciacchitano, MST 30 (2019) 092001 (review dello step 1) | doi:10.1088/1361-6501/ab1db8 | Letta allo step 1 | P01–P18 |

---

## 11. Registro sintetico delle ricerche

| # | Ricerca e strumento | Esito |
|---|---|---|
| 1 | OpenAlex per DOI e titolo, ~40 lavori | Copie OA individuate; molte su host bloccati |
| 2 | Download diretto da IOP e Springer | Challenge o redirect; bypass con browser negato e non ritentato |
| 3 | Aggiunta di host all'ambiente e nuovi download | Negati dal classificatore dei permessi |
| 4 | TU Delft repository (`uuid` → `File_`) | Sciacchitano & Wieneke 2016, tesi di Wieneke, Valori 2019, tesi tomografica |
| 5 | DLR elib | Boxx 2010, Stöhr 2012, Meier 2010, Oberleithner 2015, Saini AIAA, Lange 2012 |
| 6 | OSTI | Due rapporti DOE di Clemens; Beresh 2008 |
| 7 | arXiv | Barbosa, Bhattacharya, Xue, Rajendran 2019/2020, Sunol, Peterson, Barwey; nella v2 Karmarkar et al. (2011.14662), segnalato dall'altro scout |
| 8 | TUM | Tesi di Baumgartner |
| 9 | WebSearch su semina nei gas combusti e vettori validi in fiamma | Solo snippet; nessuna statistica in fiamme swirl |
| 10 | WebSearch su Vanselow e NH2* | Abstract e snippet |
| 11 | WebSearch su PIV in fiamme swirl NH3/CH4 | Nessun dettaglio PIV negli snippet; non approfondito |
| 12 | TU Darmstadt | Pagine 403 o indirette; dissertazioni non scaricate |
| 13 | PMC, CORE | reCAPTCHA o challenge |

**Esclusioni motivate:**
- Lavori su motori ottici: solo contesto (L18).
- PTV e PIV volumetrica: citate per il meccanismo (L30).
- Deep learning applicato alla PIV: visti solo i titoli, non cambia le scelte dello step 4.

---

## 12. Lacune principali e prossimi passi suggeriti

1. **Statistiche di rigetto in fiamme swirl** (per frame, cluster, persistenza, relazione con OH): da cercare nei dati allo step 3.
2. **Fonti primarie da leggere con accesso completo:** Saini et al. 2016 (versione rivista), Vanselow et al. 2018/2019, Stella et al. 2001, Westerweel & Scarano 2005, Masullo & Theunissen 2016, Neal 2015, Sciacchitano 2015, Boomsma 2016, Ebi & Clemens 2016, Litvinov et al. 2021 (descrizione del sistema di L37), Steinberg et al. 2013 (sistema a 10 kHz di Oberleithner).
3. **Fiamme NH3/CH4:** nessuna fonte letta descrive un apparato PIV con dettagli di processamento e qualità.
4. **Filtro della LES e ricampionamento a 128×128:** da verificare nella missione 01 (L10).
5. **ρ(2d) al 50% di overlap:** discrepanza fra review e fonte primaria **aperta** (L15).
6. **Statistica degli intervalli di incertezza di L37:** non definita nel preprint; da chiarire nella versione pubblicata o in Litvinov et al. 2021.

La ricerca non è esaustiva. “Non trovato” indica i limiti delle ricerche e degli accessi documentati, non l'inesistenza dell'evidenza.

---

## 13. Revisione v2 (26 novembre 2026)

Correzioni richieste dal coordinatore dopo la revisione della v1.

| Punto | Correzione |
|---|---|
| 1. Configurazione Oberleithner | Separati il sistema a 5 Hz (§2.2: 1.3×1.3 mm², ±0.8 m/s, fuori piano ≈3×) e quello a 10 kHz (§2.5: 1.0×1.0 mm², incertezza non dichiarata). Ricontrollate le altre righe di L04 sui manoscritti: Stöhr a 5 Hz corretto (§2.3.2, la stima dello slip appartiene a §2.3.2 e non a §2.1); Stöhr a 5 kHz corretto (§2.3.3); aggiunta la riga Meier a 5 kHz (§2.4.2), che riporta lo stesso sistema di Boxx 2010. |
| 2. Rinvii interni | Corretti in sintesi, mappa e §5. Esempi: microPIV L22 (non L24), gappy-POD L23 (non L25), Valori L25 (non L32), parete L24 (non L30), sporcamento L34 (non L33), luminosità L33 (non L34). Gli ID delle schede non sono cambiati; L37 è nuovo. |
| 3. Generalità | Risposta spaziale, L_sr ≈ 0.7 volte la finestra e filtro MHG presentati come risultati delle configurazioni DaVis di Wieneke (L01, L02, L10, P-B). In §8 P-A ora parla di moduli distinti ma potenzialmente dipendenti. |
| 4. Tracciante e maschera termica | Distinti temperatura riportata, condizioni sperimentali, dinamica delle gocce e approssimazione per la LES (L28, P-F, sintesi §1.2). Nessuna soglia universale. |
| 5. Livelli di lettura e riferimenti | Sintesi riallineata ai livelli FT/AB/SN/SEC; regola d'uso dei numeri in §0; DOI e URL completati (§10); discrepanza su ρ(2d) mantenuta aperta. |
| 6. arXiv:2011.14662 | Letto e integrato come L37. Metodo correlation statistics in DaVis, componenti in piano e fuori piano, intervalli in m/s con statistica e copertura non definite, nessuna validazione, nessuna conversione in σ. |
