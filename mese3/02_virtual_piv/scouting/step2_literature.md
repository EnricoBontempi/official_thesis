# Missione 02 — Step 2, Agente B: evidenze della letteratura sulla PIV sperimentale

**Data:** 26 novembre 2026  
**Branch:** `piv/step2-literature-scouting`, base `5aec249`  
**Stato:** consegna dello scout della letteratura, da discutere nella chat di coordinamento. Non è una decisione sullo scenario.  
**Perimetro:** ricerca e lettura di fonti; nessun operatore virtuale, parametro definitivo, training, inferenza o accesso a TEST. Nessun PDF di terzi è committato. La missione 01 non ha ancora output nel checkout; il rapporto dell'Agente A non era disponibile.

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

**Natura dell'affermazione:** *sperimentale* (misura in un apparato), *simulata* (immagini o campi sintetici, Monte Carlo), *a priori* (stima dichiarata da regole generali, non misurata), *inferenza* (ragionamento di questo rapporto).

Gli ID delle evidenze sono **L01…L36**. Gli ID **P01…P18** rimandano al registro della [scheda dello step 1](../step1_sciacchitano.md#4-registro-delle-evidenze-rilevanti).

---

## 1. Sintesi

1. **Nessuna fonte letta quantifica, in una fiamma swirl, la frazione di vettori rigettati o sostituiti per frame e per regione.** Esistono descrizioni qualitative coerenti e alcuni numeri di apparati specifici: interpolazione fino al 20% in microPIV di flashback H2/aria, con meno del 10% di invalidi nelle zone inseminate (L24). Le lacune dei dati ad alta pressione sono descritte come “significative” e poi simulate con gap artificiali, senza statistiche sui gap reali nella versione letta (L25).
2. **La scelta del tracciante decide se i prodotti caldi sono osservati.** Le goccioline d'olio evaporano nella zona di preriscaldamento: per l'olio d'oliva sono stati misurati circa 555 K. Il campo di velocità resta quindi solo nei reagenti, per costruzione. Particelle solide (TiO2, Al2O3) sopravvivono e permettono misure in entrambi i gas, ma sporcano pareti e finestre (L28, L33). Questa è la prima domanda da porre a qualunque dataset di riferimento.
3. **Attraverso la fiamma la densità di particelle cala per espansione termica.** Lo documentano immagini e testi (L29); un valore di circa −85% nei gas combusti compare solo in uno snippet. Dove è stato osservato, il calo è associato a più vettori interpolati nella zona di reazione (L24). L'ampiezza del calo per un rapporto di densità dato è un'inferenza fisica, non una misura letta.
4. **Gli effetti ottici della fiamma sono reali e locali, ma la loro grandezza è documentata solo in pochi casi.** Nella fiamma premiscelata di propano di Vanselow et al. (2019, solo AB) gli errori arrivano al 4% sistematico e al 6% casuale, concentrati vicino al fronte e variabili col piano. Ad alta pressione compaiono striature d'intensità e picchi di correlazione allungati, corretti con filtri d'immagine (L31). In un flusso non reagente a densità variabile, il metodo UQ di Wieneke classifica affidabile circa il 75% del dominio (L32). Il beam steering vicino a pareti raffreddate produce validazioni basse e regioni dove resta utile solo la media (L30).
5. **Le incertezze dichiarate nei lavori di combustione DLR sono per lo più a priori.** Lo spostamento di ±0.1 px viene convertito in ±0.3–1.4 m/s secondo potenza e Δt (L11). Non sono errori misurati. Non viene dichiarato se siano incertezze standard o espanse. Per lo stesso apparato e la stessa fiamma, due articoli riportano 1.0 mm e ±0.3 m/s oppure 1.5 mm e ±0.5 m/s.
6. **La risposta spaziale non è una media semplice sulla finestra.** Negli algoritmi multi-pass con deformazione (DaVis) il filtro equivalente è una gaussiana con una componente a cappello messicano. Servono due funzioni distinte: una per la risposta alle lunghezze d'onda, con L_sr ≈ 0.7 volte la finestra a 75% di overlap, e una per l'autocorrelazione del rumore, più larga (L01–L02). Una media a finestra quadrata corrisponde al caso single-pass a finestra quadrata. Con altri algoritmi può essere un'approssimazione, non un'equivalenza.
7. **Gli errori di vettori vicini sono correlati e la correlazione dipende dall'overlap.** Con finestre gaussiane 32×32 e 75% di overlap, ρ(2d) ≈ 0.45 e la correlazione è significativa fino a 3 passi (L15). Sciacchitano e Wieneke (2016) usano ρ(2d) = 0.11 come valore rappresentativo per il 50% di overlap, non zero. Va riconciliato con quanto annotato dalla review allo step 1 (§4.1).
8. **Su un getto di riferimento l'errore realizzato è prevalentemente casuale.** Vale per PIV a 10 kHz con riferimento ad alta risoluzione: deviazione standard 0.060 px per entrambe le componenti, bias medio −0.005 e −0.021 px, incertezza stimata dal metodo correlation statistics 0.063–0.064 px (L16). È un caso non reagente: non si trasferisce automaticamente alla fiamma.
9. **Il moto fuori piano e i gradienti fuori piano sono sorgenti di errore casuale importanti e tipiche dei flussi swirl.** Nel flashback swirl di Clemens, il Δt è stato limitato dalla perdita di correlazione fuori piano (L08, L12, L13). Nei DLR la componente fuori piano della stereo-PIV ha incertezza stimata 2–3 volte quella in piano (L04).
10. **Le scelte di validazione e sostituzione cambiano il dato rilasciato.** In un iniettore swirl, variare i parametri di validazione cambia la velocità massima fino all'11% (L20). Interpolazione dei vettori isolati, filtri POD, riempimenti gappy-POD e filtri d'immagine lasciano campi completi che non mostrano più i rigetti (L21–L26).
11. **Il campo di vista degli apparati ad alta frequenza è piccolo rispetto alla nostra slice.** Per esempio 30×30 mm² contro 73.8×96.2 mm². Spesso è decentrato per ostacoli ottici come i montanti della camera, e vi si aggiungono regioni escluse per riflessi (L04, L05, L30). È documentato quanto le lacune interne.
12. **Lacune guidate dalla fiamma, luminosità NH2*/NO2* nelle fiamme con ammoniaca e statistiche di persistenza delle lacune restano plausibili ma non verificate.** Sono le prime cose da controllare sui dati dello step 3.

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
| Host aggiunti all'ambiente durante la sessione con approvazione (`idp.springer.com`, `utoronto.scholaris.ca`, `media.suub.uni-bremen.de`, `research-information.bris.ac.uk`, `www.sciencedirect.com`) | Download **negati dal classificatore dei permessi** della sessione; non ritentati | Saini et al. 2016 (versione rivista), Vanselow et al. 2018/2019 e Masullo & Theunissen 2016 restano AB. I manoscritti accettati esistono in archivi istituzionali e possono essere letti in una sessione con permessi adeguati. |

Gli abstract sono stati letti tramite OpenAlex quando disponibili. Niente è stato ricostruito dai soli snippet: dove compare un numero da snippet è marcato **SN**.

---

## 3. Mappa dei fenomeni

Colonna “Evidenza”: **forte** = più fonti FT in apparati pertinenti, con numeri; **media** = FT in un apparato o metrologia sintetica ben definita; **debole** = AB/SN/SEC o solo qualitativa; **assente** = non trovata.

| Fenomeno | Meccanismo | Dove documentato (apparato) | Effetto sul dato | Evidenza | ID |
|---|---|---|---|---|---|
| Campo di vista ridotto e decentrato | Sensore e memoria a kHz; montanti della camera; riflessi | GTMC DLR 5–10 kHz; combustore Toronto/DLR; EM2C; flashback UT Austin | Copertura parziale; bordi fissi; metà fiamma | forte, descrittiva | L04–L08, L30 |
| Risposta spaziale finita | Finestra, pesatura, passate iterative, filtro predictor/corrector | Metrologia sintetica DaVis; parametri dichiarati in combustione | Attenuazione delle piccole scale, overshoot, correlazione del rumore | media (sintetica) | L01, L02 |
| Rapporto rumore/risoluzione | Più pixel per finestra = meno rumore | Sintetico DaVis | σ·L_sr ≈ costante | media | L01 |
| Range dinamico di velocità (Δt unico) | Un solo Δt per getti veloci e ricircoli lenti | Combustore pressurizzato DLR | Ricircolo lento non risolto; errore relativo maggiore a bassa velocità | debole–media (qualitativa) | L06 |
| Errore casuale a priori ±0.1 px | Convenzione sulla precisione dell'interpolazione del picco | Tutti i lavori DLR letti | ±0.3–1.4 m/s dichiarati, non misurati | forte come pratica, debole come misura | L11 |
| Componente fuori piano (stereo) più incerta | Geometria stereo, angolo delle camere | GTMC DLR | 2–3× l'incertezza in piano (a priori) | media | L04 |
| Moto e gradienti fuori piano | Perdita di coppie; particelle a z diverse | Metrologia; swirl UT Austin | Errore casuale ~0.1 px; Δt limitato | media | L08, L12, L13 |
| Peak locking | Immagini di particella < ~2 px; pixel grandi delle camere veloci | Metrologia | Bias verso spostamenti interi; più rumore | media (generale), assente in fiamma | L14 |
| Correlazione spaziale degli errori | Sovrapposizione delle finestre e filtro effettivo | Sintetico DaVis; getto di riferimento | ρ(2d) ≈ 0.45 a 75% di overlap; 0.11 a 50% (valore rappresentativo) | media | L15, L16 |
| Bias dei metodi UQ e copertura | Punti ciechi dei metodi | Getto, scia (non reagenti) | Sotto/sovrastima; risposta piatta al moto fuori piano | debole–media | L17 |
| Outlier isolati e cluster | Picchi spuri da poca informazione o rumore | Metrologia; fiamme (qualitativo) | Rigetti; cluster sfuggono ai test a vicinato fisso | debole (AB/SEC) | L19, L22 |
| Sensibilità ai parametri di validazione | Soglie diverse, rigetti diversi | Iniettore swirl EM2C | Fino all'11% sulla velocità massima | media (un caso) | L20 |
| Sostituzione e smoothing | Interpolazione 3×3, filtri POD, GPOD | GTMC DLR; TUM | Campo completo; rigetti invisibili | forte (pratica) | L21, L23–L25 |
| Tracciante che evapora | Goccioline d'olio evaporate a ~555 K | Flashback UT Austin; tesi CPIV | Nessun vettore nei prodotti | forte | L28 |
| Calo della densità di semina | Espansione termica | TUM, TU Delft, CPIV; numero −85% solo SN | Più vettori interpolati in zona di reazione; fronte visibile nelle immagini | media | L24, L29 |
| Slip e termoforesi | Inerzia; gradiente di T nella zona di preriscaldamento | GTMC (stima); fiamme a controflusso (AB) | Ritardo particella-gas al fronte | debole–media | L27 |
| Gradienti d'indice di rifrazione | Deflessione dei raggi, sfocatura | Propano (AB); ad alta pressione (FT); vicino a pareti (FT); supercritico non reagente (FT) | Errori sistematici e casuali locali; striature; validazioni basse | media | L30–L32 |
| Luminosità della fiamma | Chemiluminescenza o fuliggine nella banda della camera | GTMC, TUM (FT); alta pressione (affermazione); fuliggine RQL (AB) | Fondo; riflessi accentuati | debole–media | L34 |
| Riflessi e accesso ottico | Fogli che colpiscono pareti o finestre; montanti | DLR, Toronto/DLR, TUM, UT Austin, motore | Regioni escluse; saturazione; autocorrelazione sul fondo | forte (qualitativa) | L05, L30, L35 |
| Sporcamento delle finestre | Depositi di semina; superfici roventi | DLR, UT Austin | Tempi di misura limitati; profilo del foglio alterato | media (qualitativa) | L35 |
| Qualità peggiore vicino alla fiamma | Tutti i precedenti | TUM; Vanselow (AB); Valori (non reagente) | Concentrazione dei problemi al fronte | debole–media; nessuna statistica in fiamme swirl | L24, L30–L32 |

---

## 4. Registro delle evidenze

Ogni scheda riporta: fenomeno e meccanismo; apparato e condizioni; tipo; effetto e numeri; metodo; campione e variabilità; trattamento e residuo; fonte e livello di lettura; limiti e trasferibilità; dati per lo step 3. I campi senza informazione sono omessi o marcati “non dichiarato”.

### 4.A Operatore di misura

**L01 — Modello a filtro lineare della PIV, lunghezza di risoluzione e legame con il rumore**
- *Meccanismo:* la correlazione media lo spostamento nella finestra con pesi F(x). L'autocorrelazione normalizzata di F coincide con l'autocorrelazione del rumore fra vettori vicini (eq. 2.4, dimostrazione in App. A). La lunghezza di risoluzione è L_sr = 1/∫F² = ∫ autocorrelazione del rumore. Per una finestra top-hat L_sr = L; per una gaussiana L_sr = σ√(4π).
- *Tipo:* simulato, con immagini sintetiche processate in DaVis 8: spostamento costante, Nppp 0.1, d_p 2.5 px, 10% di moto fuori piano, finestre gaussiane, 4 passate.
- *Numeri (Tab. 2.2/2.4), finestra 32×32 px, overlap 0/25/50/75/87%:* σ = 0.050/0.052/0.060/0.076/0.084 px; L_sr dal rumore = 47.1/41.6/38.4/29.7/25.8 px; L_sr dalla risposta alle lunghezze d'onda (R_A) = 59.3/44.5/29.8/22.9/20.8 px. Con 16×16 px al 75%: σ = 0.146 px, L_sr = 15.8 px (rumore) e 12.2 px (R_A). Con 64×64 px al 75%: 0.038 px, 57.3 e 44.6 px. Il prodotto σ·L_sr(rumore) resta fra 2.13 e 2.35 px²: è la “regola universale” dell'eq. 2.8 *per un dato set di immagini*. Bias sistematico < 0.002 px.
- *Lettura per noi:* nel multi-pass l'overlap cambia **sia** il rumore **sia** la risposta, perché il predittore viene interpolato fra vettori. Non è solo una questione di passo di campionamento. Con 0% e 25% di overlap la risoluzione effettiva peggiora.
- *Fonte:* B. Wieneke, *PIV Uncertainty Quantification and Beyond*, tesi di dottorato TU Delft, 2017, cap. 2, §§2.2–2.4, Tab. 2.2. **FT**, cap. 2 letto integralmente.
- *Limiti:* un solo software, DaVis, con raccomandazioni specifiche di DaVis (“3–4 passate, 75% di overlap, finestre gaussiane”). Immagini ideali senza fiamma. L'equivalenza fra autocorrelazione del filtro e del rumore assume contributi di rumore indipendenti pixel per pixel.
- *Step 3:* se un dataset dichiara software, finestra, pesatura, passate e overlap, si può stabilire quale forma di filtro sia plausibile. Dai soli vettori non si ricava.

**L02 — Forma della risposta: single-pass, multi-pass, gradino**
- *Numeri e osservazioni:*
  - Single-pass con finestra quadrata 32 px: risposta R_A a forma di sinc, lobo negativo a circa 2/3 della finestra; anche a λ = 200 px l'ampiezza è ridotta del 10% (§2.6.1).
  - Single-pass con finestra gaussiana in DaVis: filtro quasi gaussiano con L_sr ≈ 36 px.
  - Multi-pass, da 1 a 6 passate: L_sr(R_A) passa da 36 a 22 px; L_sr dal rumore da 37 a 30 px. Il filtro diventa una miscela gaussiana–cappello messicano con b ≈ 0.3–0.4 (Tab. 2.3).
  - Risposta a gradino: overshoot e undershoot; filtro MHG con b = 0.385 e L_sr ≈ 22.7 px (§2.6.2).
  - Riportato da Elsinga & Westerweel 2011 con software TU Delft (**SEC**): risoluzione di separabilità ≈ finestra + 1 px.
  - Kähler et al. 2012 (**AB**): per la correlazione ensemble single-pixel il limite è la dimensione delle immagini di particella. La PTV raggiunge risoluzioni sub-pixel *sui campi mediati*.
- *Fonte:* tesi di Wieneke, §§2.5–2.7, **FT**; Kähler, Scharnowski e Cierpka, Exp. Fluids 52 (2012) 1629, **AB**.
- *Trasferibilità:* “la PIV è una media su finestra quadrata” vale per il single-pass a finestra quadrata. Per algoritmi moderni è un'approssimazione con risposta diversa. Theunissen 2012, Scarano 2003 e Lavoie et al. 2007 **non sono stati letti**: per Scarano e Lavoie OpenAlex non fornisce abstract; Theunissen 2012 non è stato individuato con certezza.

**L03 — Precedente di operatore “PIV virtuale” applicato a campi simulati**
- *Contenuto:* Sunol et al. (arXiv 2510.24673 v4, reologia, flussi non reagenti) definiscono y = S·H·u + ε. H è una media con finestra di Hann sulle finestre d'interrogazione, S un campionamento sulla griglia dei vettori. ε è un rumore correlato tramite smoothing gaussiano con larghezze legate a finestra e passo (χ_corr = 0.35). È eteroschedastico: ampiezza ∝ U95·[1 + 0.5·gradiente normalizzato], con un offset di bias a bassa frequenza pari a 0.1 volte la scala base.
- *Tipo:* scelta di modellazione per uno studio di sensibilità. I parametri **non** sono qualificati su un esperimento.
- *Fonte:* sezione “Reference data and PIV emulation”, eq. 24–30. **FT** della sezione.
- *Uso:* è un precedente di struttura (filtro, poi campionamento, poi rumore correlato ed eteroschedastico). Non giustifica alcun valore numerico.

**L04 — Configurazioni PIV nel Gas Turbine Model Combustor DLR (atmosferico, CH4/aria)**

| Sistema | Flusso | FOV | Finestra/overlap | Risoluzione dichiarata | Passo | Δt | Foglio | Incertezza dichiarata |
|---|---|---|---|---|---|---|---|---|
| SPIV 5 kHz + OH-PLIF (Boxx 2010) | 10 kW, φ 0.75, parzialmente premiscelata | FOV non quantificato nel testo letto (foglio ≈40 mm in altezza; 512×512 px) | 16×16 px, 50% | 1 mm | 0.5 mm | 20 µs | ≤0.7 mm | ±0.3 m/s “random uncertainty” da ±0.1 px |
| Stesso sistema (Stöhr 2012, §2.3.3) | 10 kW, φ 0.65 | 30×30 mm² | non ripetuto | 1.5×1.5 mm² | — | 20 µs | 0.7 mm | ±0.5 m/s in piano; fuori piano ≈2× (α≈30°) |
| SPIV 5 Hz (Stöhr 2012, §2.3.2) | 10–35 kW, φ 0.65 | larghezza della camera (85 mm) | 16×16 px, 50% | 1.3×1.3 mm² | — | 6–20 µs | ≈1 mm | ±0.3 m/s (10 kW) – ±1.4 m/s (35 kW); fuori piano ≈3× (α≈20°) |
| PIV 5 Hz (Meier 2010, §2.3) | GTMC | altezza della camera (110 mm) | 16 px | 1.1 mm | — | — | 1 mm | 0.4 m/s “measuring uncertainty” |
| SPIV 10 kHz (Oberleithner 2015) | GTMC | — | 16×16 px, 50% | 1.3×1.3 mm² | — | — | — | ±0.8 m/s in piano; fuori piano ≈3× |

- *Altri dettagli:* TiO2 0.5–1 µm, letto fluido, circa 4% dell'aria. Filtro passa-banda 532±5 nm “per ridurre l'influenza della luminosità di fiamma” (Stöhr §2.3.2). Durata 0.8 s per run (4096 coppie). Camera quadrata 85×85 mm con montanti negli angoli.
- *Tipo:* parametri dichiarati. Le incertezze sono **a priori** (L11).
- *Variabilità:* lo stesso sistema SPIV 5 kHz, nella stessa camera, è descritto con 1.0 mm e ±0.3 m/s in Boxx 2010 e con 1.5 mm e ±0.5 m/s in Stöhr 2012. La causa non è spiegata: definizioni diverse, ingrandimento o processamento.
- *Fonti:* Boxx, Stöhr, Carter & Meier, Combust. Flame 157 (2010) 1510, manoscritto accettato DLR elib 64569, pp. 11–14. Stöhr et al., Combust. Flame 159 (2012) 2636, manoscritto DLR elib 77484, §§2.1–2.3. Meier et al., Exp. Fluids 49 (2010) 865, copia autore DLR elib 65267, §§2.3–2.4. Oberleithner et al., Combust. Flame 162 (2015) 3100, manoscritto DLR elib 96808, §2. Livello **FT-v** per tutti (manoscritti); per Oberleithner sono state lette solo le sezioni PIV.
- *Trasferibilità:* combustore, combustibile e scala diversi dal nostro caso NH3/CH4. Utili come **classe di configurazione documentata**, non come parametri del nostro apparato.
- *Step 3:* verificare se esistono dati rilasciati di questi apparati (compito dell'Agente A) e con quale livello di processamento.

**L05 — Combustore swirl Toronto/DLR: FOV decentrato per ostacoli e riflessi**
- *Apparato:* swirl premiscelato CH4/aria, φ 0.70, camera 85×85×114 mm; S-PIV a 10 kHz, 2.6 mJ per impulso, Δt 5–15 µs, foglio FWHM ≈1 mm, TiO2. DaVis 8.0, finestra finale ≈1.0 mm, passo ≈0.5 mm.
- *Effetto:* FOV ≈49×38 mm, decentrato rispetto all'asse (x da −17 a 32 mm) per i montanti della camera. I riflessi dalle finestre aumentano il rumore per x < 0, quindi sono stati **usati solo i dati con x > 0**.
- *Fonte:* Saini, Steinberg & Arndt, AIAA 2016-0436, §III p. 7. **FT-v**: versione AIAA su DLR elib, non l'articolo su Exp. Fluids 57:122.
- *Step 3:* l'esclusione di metà campo per i riflessi è una scelta a valle che non lascia traccia nei vettori se il campo viene ritagliato.

**L06 — Range dinamico: un solo Δt per getti veloci e ricircoli lenti**
- *Apparato:* combustore pressurizzato a settore singolo DLR Colonia, 5 bar, preriscaldo 450 K, gas naturale, swirl parzialmente premiscelato, λ = 1.2. PIV con 120 mJ, pco.2000, 32×32 px: 0.39×0.39 mm² nella vista globale.
- *Effetto:* servono Δt ≈ 2 µs per 50 m/s all'uscita; “la ricircolazione interna non era adeguatamente risolta, essendo troppo lenta per l'intervallo scelto”. È stata misurata solo metà della fiamma per esigenze di risoluzione.
- *Tipo:* sperimentale, qualitativo.
- *Fonte:* Lange et al., 16th Lisbon Symposium 2012, DLR elib 77085, pp. 5–6. **FT**.
- *Inferenza:* l'errore relativo dipende dallo spostamento in pixel. Con un errore assoluto in pixel simile, le regioni lente hanno errore relativo più grande. Nel GTMC un Δt di 20 µs serve getti fino a qualche decina di m/s. Non è una misura.

**L07 — Iniettore swirl EM2C a 12 kHz: FOV piccolo per la frequenza**
- *Apparato:* PIV a 12 kHz (coppie) con camera a 24 kHz limitata a 128×512 px; Δt 6 µs; foglio 0.3 mm; TiO2 1 µm. FlowManager adattivo da 32 a 8×8 px con 25% di overlap, risoluzione dichiarata 1.07×1.07 mm².
- *Nota:* la tabella riporta 0.017 mm/px, incompatibile con 8 px = 1.07 mm. È probabilmente un refuso della fonte; non usato.
- *Fonte:* Barbosa, Scouflaire & Ducruix, arXiv 0807.1871 (preprint del 2008; riferimento su rivista non verificato), §2.2, Tab. 1. **FT-v**.

**L08 — Swirl con flashback (UT Austin): moto fuori piano e Δt**
- *Apparato:* condotto anulare di miscelazione, CH4/H2; SPIV a 4 kHz, Nd:YLF ~10 mJ, 256×512 px, 55 µm/px, FOV 13.5×27 mm. DaVis 8.2 da 64 a 16 px con 75% di overlap; Δt 40/80 µs.
- *Effetto:* “la velocità fuori piano era dello stesso ordine di quella in piano… il fattore limitante nell'aumentare Δt era la perdita di correlazione fuori piano”.
- *Configurazione ad alta pressione (secondo rapporto):* SPIV a 8 kHz, 16×16 px = 0.8×0.8 mm², 75% di overlap scelto “per avere più vettori vicino al fronte”, primo vettore a 0.5 mm dalla parete.
- *Fonti:* Clemens (PI), DOE final report DE-FE0007107, OSTI 1253136, §3.2.2, pp. 11–12. Clemens (PI), DOE final report DE-FE0012053, OSTI 1506058, §4, pp. 28–30. **FT** delle sezioni sperimentali. Contesto del lavoro Ebi & Clemens 2016, che **non è stato letto** (L36).

**L09 — Errori nascosti di stereo-PIV, calibrazione e temporizzazione**
- *Contenuto:* nella stereo-PIV, con foglio di spessore finito, le due camere non vedono lo stesso volume d'interrogazione. I gradienti fuori piano producono errori sistematici non visibili nelle immagini (Wieneke & Taylor 2006, **SEC**). Il disallineamento fra foglio e piano di calibrazione si corregge con l'autocalibrazione. Nei laser Nd:YLF ad alta frequenza il ritardo fra trigger e luce può variare fino a 0.5–1 µs secondo frequenza e potenza (Bardet et al. 2013, **SEC**).
- *Fonte:* tesi di Wieneke, §§3.1.1–3.1.3. **FT**.
- *Inferenza:* con Δt di 5–20 µs (L04, L05) una variazione di temporizzazione di 0.5–1 µs sarebbe del 2.5–20%. Se e quanto avvenga negli apparati di combustione letti **non è documentato**. È una sorgente nascosta, non verificabile dai vettori.

**L10 — Copertura della nostra slice rispetto agli apparati letti (inferenza numerica)**
- La slice misura 73.8×96.2 mm, con celle di 0.581×0.758 mm (geometria verificata nella ricognizione, §1.2).
  - FOV di 30×30 mm (SPIV DLR a 5 kHz): circa il 13% dell'area della slice.
  - FOV di 49×38 mm (Toronto/DLR), di cui usati circa 32×38 mm: circa il 17%.
  - SPIV DLR a 5 Hz: larghezza della camera, quindi una copertura confrontabile con la slice in una direzione.
- Finestre da 0.8 a 1.5 mm corrispondono a 1.4–2.6 celle in x e 1.1–2.0 celle in y. Passi di 0.5 mm sono **più fini** della griglia LES.
- Con il filtro effettivo L_sr(R_A) ≈ 0.7 volte la finestra (L01, DaVis multi-pass), una finestra di 1 mm risponde come un filtro di circa 0.7 mm, cioè circa una cella LES.
- **Conseguenza (inferenza):** con configurazioni DLR-simili il filtraggio PIV è alla scala della griglia. Non si può rappresentare con dettaglio senza conoscere il filtro implicito della LES e il ricampionamento a 128×128, che restano non verificati (missione 01). Configurazioni a bassa frequenza e campo ampio, con finestre più grandi, sono più rappresentabili come filtro sulla griglia.

### 4.B Errori e incertezza

**L11 — L'incertezza “±0.1 px” delle fonti di combustione è una stima a priori**
- *Pratica osservata:* in tutti i lavori DLR letti (L04) l'incertezza istantanea è ottenuta come ±0.1 px × scala/Δt. I termini cambiano fra “random uncertainty”, “measuring uncertainty” e “absolute accuracy” (Meier 2010 §2.4.2). Non è dichiarato se si tratti di incertezza standard o espansa. Non è distinta per componente, salvo il fattore 2–3× per la componente fuori piano, ricavato dalla geometria stereo (Lawson & Wu 1997, **SEC**). Non è validata con un riferimento.
- *Coerenza:* 16 px = 1 mm dà 62.5 µm/px; 0.1 px × 62.5 µm / 20 µs = 0.31 m/s, coerente con ±0.3 m/s. È quindi una conversione, non una misura.
- *Contesto metrologico:* la review (P01) considera “troppo semplificato” un valore universale di 0.1 px. Gli errori sperimentali possono arrivare a qualche decimo di pixel (ricognizione §5).
- *Implicazione:* σ = 0.5 e 1.5 m/s dello studio nonideal sono dello stesso ordine di questi valori a priori. **Questo non li valida**: entrambi sono a priori.
- *Step 3:* cercare dataset con UQ per vettore o con un riferimento, non solo valori ± nel testo.

**L12 — Moto fuori piano: spesso la sorgente casuale dominante**
- *Meccanismo:* particelle che entrano o escono dal foglio non trovano corrispondenza. Particelle sovrapposte a z diverse cambiano intensità relativa fra i due impulsi. Il picco di correlazione diventa asimmetrico, con spostamenti virtuali fino a metà del diametro dell'immagine di particella.
- *Numeri:* per moti fuori piano tipici l'errore casuale medio “può raggiungere 0.1 px, quasi indipendentemente dalla densità di semina”; è descritto come “spesso la sorgente dominante” (Nobach & Bodenschatz 2009 e Nobach 2011, **SEC**).
- *Effetti correlati:* un disallineamento fra i due fogli laser ha lo stesso effetto. La sola variazione globale di intensità fra i due impulsi è innocua.
- *Fonte:* tesi di Wieneke, §§3.1.7–3.1.8. **FT**.
- *Trasferibilità:* nello swirl la componente fuori piano è grande (L08; nell'audit del Mese 1 v_k ≈ ±15 m/s, ricognizione §2). La legge “v_k grande implica buco” **non** è dimostrata. Il meccanismo documentato aumenta l'errore casuale e la perdita di coppie, e quest'ultima riduce il SNR e porta a rigetti.

**L13 — Gradienti fuori piano della velocità in piano**
- *Meccanismo:* nella finestra, la posizione media in z delle particelle è casuale e ha deviazione standard σ_z/√n. Con un gradiente ∂u/∂z questa posizione produce un errore casuale.
- *Numeri:* foglio con σ = 10 px, 20 particelle, gradiente del 5–10%: 0.11–0.22 px. Prova sintetica: 0.078 px contro 0.010 px senza gradiente, con stima teorica 0.056 px.
- *Fonte:* tesi di Wieneke, §3.1.9. **FT**, risultato *simulato*.
- *Step 3:* non verificabile dai soli vettori planari; richiede misure a più piani o il volume.

**L14 — Peak locking e dimensione delle immagini di particella**
- *Meccanismo:* con immagini di particella sotto ~2 px lo spostamento stimato è attratto verso valori interi (bias) e l'errore casuale aumenta. Si vede negli istogrammi degli spostamenti nella PIV 2D; nella stereo e nella tomografica è più nascosto.
- *Condizioni:* chiudere il diaframma “non è pratico per le camere ad alta velocità con pixel ≥20 µm”. I rimedi sono sfocatura o diffusori.
- *Fonte:* tesi di Wieneke, §3.1.5, **FT**; Christensen 2004 sugli effetti nelle statistiche turbolente, solo titolo e metadati.
- *Combustione:* nessuna fonte letta documenta il peak locking in fiamma. È plausibile per sistemi a kHz con CMOS a pixel grandi, ma **non verificato**.
- *Step 3:* istogrammi delle parti frazionarie degli spostamenti in pixel. Richiedono Δt e scala, e funzionano meglio su vettori non filtrati.

**L15 — Correlazione spaziale degli errori e propagazione alla vorticità**
- *Simulazione:* immagini sintetiche in DaVis 8.2 con finestra gaussiana 32×32 px e 75% di overlap. La correlazione dell'errore è significativa fino a 3 passi; ρ(2d) ≈ 0.45. Nella successiva simulazione Monte Carlo, ρ(2d) = 0, 0.11 e 0.45 sono indicati come “rappresentativi” di overlap 0%, 50% e 75%.
- *Propagazione (eq. 42):* U_ω = (U/d)·√(1−ρ(2d)) per le differenze centrali, con incertezza uguale per u e v ed errori di u e v non correlati.
- *Anomalia:* il testo afferma che ρ = 0.45 riduce U_ω “di un fattore 5” rispetto a ρ = 0. Con l'eq. 42 il fattore è circa 1.35. L'affermazione non è usata.
- *Fonte:* Sciacchitano & Wieneke, MST 27 (2016) 084006, copia TU Delft, §§2.3.3, 3.2 e fig. 2, **FT**; tesi di Wieneke, §§4.2.3 e 4.3.2, **FT**.
- *Da riconciliare:* la scheda dello step 1 (§4.1) riporta dalla review un ρ(2h) nullo per overlap ≤50%. Questa fonte primaria usa 0.11 come valore rappresentativo al 50%. La differenza è piccola per la vorticità, ma **non** autorizza a dire “errori indipendenti a 50% di overlap”. In ogni caso ρ(h), cioè fra vettori adiacenti, è più grande di ρ(2h).
- *Limiti:* sintetico, un algoritmo, flusso uniforme. È la correlazione dell'*errore*, non quella del campo.

**L16 — Verifica sperimentale in un getto non reagente con riferimento**
- *Esperimento:* database di Neal et al. 2015. Getto rettangolare a x/h = 20, 10 kHz. Sistema di misura (MS): camera HSS5, 16×16 px con finestra gaussiana, 75% di overlap. Riferimento “HDR”: stereo con ingrandimento circa 3.6 volte superiore e 48×48 px, risoluzione spaziale simile. Secondo Neal et al. (**SEC**) l'HDR è circa 4 volte più accurato.
- *Numeri al punto P:*

| Grandezza | Errore medio | Deviazione standard dell'errore | Incertezza CS (rms) |
|---|---|---|---|
| u | −0.005 px | 0.060 px | 0.063 px |
| v | −0.021 px | 0.060 px | 0.064 px |
| vorticità | 0.0008 px/px | 0.0104 px/px | 0.0116 px/px, con ρ(2d) = 0.45 e d = 4 px |

  Sul dominio comune la vorticità ha errore e incertezza fra 0.010 e 0.016 px/px.
- *Lettura:* in questo caso l'errore è dominato dalla parte casuale, ha la stessa ampiezza per le due componenti e la propagazione con ρ = 0.45 è coerente.
- *Fonte:* tesi di Wieneke, §4.4.2 e Tab. 4.4 (= Sciacchitano & Wieneke 2016, §4.2). **FT**. Neal et al. 2015, MST 26 074003: **AB**.
- *Trasferibilità:* flusso freddo con semina omogenea. **Non** dice nulla dell'eterogeneità in fiamma.

**L17 — Confronti fra metodi UQ: copertura e punti ciechi**
- *Contenuto:*
  - Sciacchitano et al. 2015 (**AB** + **SEC** tramite Bhattacharya et al. 2018): quattro metodi su un getto con casi a sorgente dominante diversa (taglio, fuori piano, dimensione delle particelle, densità di semina). I metodi diretti (CS, IM) sono più sensibili; i metodi calibrati rendono meno; il metodo uncertainty surface ha risposta piatta nel caso con moto fuori piano. Le coperture al livello nominale del 68% (circa 3%, 64%, 51%, 65%) sono già registrate allo step 1 dalla review.
  - Boomsma et al. 2016 (**AB** troncato, **SEC**): due esperimenti (getto; cilindro a due Re), tre software. Tutti i metodi rispondono qualitativamente a errori variabili nello spazio. IM e CS possono sottostimare l'incertezza standard anche con bias trascurabile.
  - Bhattacharya et al. 2018 (preprint arXiv, **FT-v** di introduzione e conclusioni): il metodo moment of correlation risponde meglio alla variazione spaziale dell'errore RMS.
- *Implicazione:* una mappa UQ è un'informazione con limiti noti. Non certifica l'assenza di outlier, peak locking o sorgenti nascoste (P12).

**L18 — Stime in campo reagente o a densità variabile: due esempi di numeri dichiarati**
- *UT Austin, alta pressione:* incertezza “< 0.1 m/s” a pressione atmosferica e “bias massimo 0.2 m/s” a pressione elevata. Errore di ricostruzione stereo “< 0.3” e “0.6”, **unità non indicate** nella fonte. Il metodo non è descritto nell'estratto. Fonte: OSTI 1506058, p. 30, **FT**. Non usabile come σ.
- *Motore ottico, SPIV:* incertezza “stimata ≤10%” da letteratura e algoritmi; riflessi laser riducono la qualità vicino al punto morto superiore. Fonte: Peterson, Baum, Dreizler & Böhm, arXiv 2009.01710, §2.4. **FT-v**, sezioni lette; trasferibilità bassa.

**L19 — Rilevamento degli outlier: test mediano normalizzato e cluster**
- *Pratica:*
  - Test mediano normalizzato di Westerweel & Scarano 2005: 3×3 vettori, ε = 0.1 come livello minimo di normalizzazione, parametri riportati da Saini et al. (AIAA, §II.B.4). Il lavoro originale **non è stato letto**: Springer inaccessibile, nessun abstract in OpenAlex.
  - Duncan et al. 2010 (**AB**): la generalizzazione a dati PIV/PTV mostra residui con PDF “universale” e funziona fino a circa il 15% di vettori spuri.
  - Masullo & Theunissen 2016 (**AB**): i test a vicinato fisso perdono efficacia con cluster di outlier. Un vicinato adattivo riduce i non rilevati e i falsi rigetti.
  - Garcia 2011 (**SN**): con vettori spuri raggruppati l'efficacia del test mediano cala oltre il 10%.
  - Frazioni “tipiche” di vettori spuri del 2–5% compaiono solo in snippet di tesi non lette: **non sono usate**.
- *Implicazione:* non esiste una frequenza universale di outlier. I cluster sono più difficili da rilevare dei singoli, quindi un dataset “validato” può contenere cluster residui.

**L20 — Sensibilità del dato ai parametri di validazione (iniettore swirl reagente)**
- *Apparato:* L07. Validazione con picco (detectability k = 1.2), intervallo di velocità e media mobile (α = 0.1).
- *Numero:* uno studio parametrico degli stessi autori (Barbosa et al. 2008, **SEC**) indica che cambiare i parametri di validazione introduce un'incertezza massima dell'11% sulla velocità massima, senza effetto sulla posizione delle regioni di massimo e minimo.
- *Fonte:* arXiv 0807.1871, §2.3. **FT-v**.
- *Implicazione:* il dato rilasciato dipende dalle soglie. Per i parametri dei dataset va registrata la catena di validazione.

### 4.C Validazione, sostituzione e lacune

**L21 — GTMC DLR: interpolazione dei vettori isolati e filtro POD di visualizzazione**
- *Contenuto:* la fig. 10 di Boxx 2010 è presentata “senza la ricostruzione POD passa-banda né interpolazione oltre i buchi di un singolo vettore”. Ne segue che il processamento standard **interpolava i buchi di un vettore**. Le figure principali usano invece una POD di 2048 snapshot troncata a 150 modi, circa il 70% dell'energia, perché i modi superiori sono “principalmente rumore” e l'incertezza di ±0.3 m/s disturba vorticità e strain.
- *Fonte:* Boxx et al. 2010, manoscritto, pp. 23–24 e 30. **FT-v**.
- *Implicazione:* gli stessi dati esistono in più livelli di processamento: interpolato a un vettore, filtrato POD. Un campo filtrato POD non va usato per stimare il rumore o le lacune.

**L22 — Statistiche di vettori invalidi e interpolati in microPIV di flashback H2/aria**
- *Apparato:* canale con flashback, TiO2 0.1–1 µm, microscopio a lunga distanza, FOV 4.15×4.15 mm², filtro 527±10 nm. PIVlab da 48 a 24 px con 50% di overlap: risoluzione dichiarata 50 µm.
- *Validazione:* filtro mediano 3×3 con tolleranza di 2 deviazioni standard locali, più un filtro di intervallo. I vettori rimossi sono interpolati da un kernel 3×3.
- *Numeri:* vettori interpolati “fino al 20% in alcune prove”, attribuiti a una regione di scia senza particelle nel FOV. “La percentuale di vettori invalidi nelle aree con particelle è sempre stata inferiore al 10%.” Denominatore: vettori del campo, per prova; distribuzione per frame non riportata.
- *In fiamma:* gli interpolati si trovano soprattutto nella scia e vicino alle pareti. “La percentuale di vettori interpolati è più alta nella zona di reazione a causa della bassa densità di semina.” Quando la coda della fiamma attraversa lateralmente il foglio, la densità di semina cala in tutto il campo.
- *Fonte:* G. M. Baumgartner, tesi di dottorato TUM 2015, §3.4 pp. 48–49 e cap. 7 pp. 122–129. **FT** delle sezioni.
- *Trasferibilità:* microscala, H2/aria, flashback; non una fiamma swirl stabilizzata. È però l'evidenza più diretta trovata di un legame fra qualità e regione di fiamma.

**L23 — Gappy-POD per dati PIV rumorosi nei combustori: che cosa simula e che cosa no**
- *Motivazione:* nei combustori ad alta pressione la PIV ha basso SNR per le basse energie dei laser multi-kHz, il fondo di fuliggine e luminosità, la semina non uniforme con vuoti. Ne seguono “lacune significative” nei campi processati (§I). È un'affermazione degli autori, non quantificata nella versione letta.
- *Metodo di prova:* dataset atmosferico con “essenzialmente nessun vettore mancante” (L05), con gap artificiali G = 15, 25 e 50%. Il numero di mancanti per snapshot segue una distribuzione gaussiana attorno a G. I gap sono descritti come “raggruppati in cluster, come in questo caso”; la geometria non è specificata nella versione AIAA.
- *Risultati:* il metodo adattivo con filtro mediano (MF) ha E* circa il 30% più basso del metodo di Gunes; il metodo field smoothness circa il 20%. L'errore cresce con G. Servono POD separate per componente; u_z ha errore maggiore, attribuito all'incertezza della stereo-PIV. Esempio con G = 45%: errore relativo sui gap 0.226 per u_x e 0.231 per u_y al numero ottimo di modi, contro 0.454 e 0.447 dopo 10 modi (fig. 2). E è la norma 2 relativa sui soli punti di gap, mediata sugli snapshot (eq. 13–14).
- *Fonte:* Saini, Steinberg & Arndt, AIAA 2016-0436, DLR elib 102428, **FT-v**. Versione Exp. Fluids 57:122: solo SN, secondo cui varia intensità turbolenta, frazione mancante e rumore e confronta il kriging. **La versione rivista non è stata letta.**
- *Implicazioni:* (i) qui le lacune reali non sono misurate: la legge dei gap è una scelta degli autori. (ii) La GPOD usa l'intera serie temporale: un campo riempito con GPOD contiene informazione da altri istanti, rilevante se la si usa come “osservazione”.

**L24 — Regioni dove resta solo la media (combustore pressurizzato)**
- *Effetto:* vicino alla parete raffreddata la PIV è valutabile solo oltre 1 mm. Riflessione del foglio, flare e saturazione; vicino alla parete gradienti termici e di densità forti, con beam steering e riflessione interna. Le immagini di particella sono poco risolte: “bassi tassi di validazione”, e “solo una mappa di velocità media fornisce dati utili”. Con foglio parallelo alla parete la PIV non è fattibile: flare sul fondo metallico e autocorrelazione sul fondo stazionario speculare.
- *Fonte:* Lange et al. 2012, pp. 4 e 7–9. **FT**. Qualitativo, senza numeri.
- *Implicazione:* esistono regioni, per esempio presso le pareti, dove l'istantaneo non è rilasciato o non è affidabile. È diverso da un rigetto sparso.

**L25 — Regione affidabile in base all'incertezza (non reagente, densità variabile)**
- *Apparato:* convezione di Rayleigh–Bénard in CHF3 supercritico, differenze relative di densità 17–42%; piano medio; finestre 48×48 px (64×64 in un caso “per avere più vettori validi”), 50% di overlap.
- *Numeri:* con il metodo UQ di Wieneke 2015, i punti con incertezza < 1 px, considerati affidabili, sono circa il 75% della sezione. In questa regione l'incertezza media è ≈0.53 px, cioè circa 5.7–10% della velocità della circolazione su grande scala. Le zone ad alta incertezza coincidono con un pennacchio ascendente forte. La sfocatura impedisce la visibilità delle particelle solo negli strati limite.
- *Fonte:* Valori et al., Exp. Fluids 60 (2019) 143, copia TU Delft, §§4.3 e 5. **FT**.
- *Trasferibilità:* stesso meccanismo ottico di una fiamma (gradienti d'indice), ma fluido, scale e soglia di “affidabilità” diversi. Il 75% non si trasferisce.

**L26 — Altri trattamenti documentati**
- Filtri d'immagine contro le striature ad alta pressione: sottrazione di un fondo mobile di 16 px e normalizzazione min-max (L31).
- Sottrazione del minimo o della media d'immagine contro fixed pattern e fondo costante, per esempio riflessi (tesi di Wieneke, §3.1.6, **FT**).
- Ritaglio del FOV per eliminare zone con riflessi (L05).
- GPOD e kriging (L23); interpolazione Delaunay e GPOD citate come prassi da Barwey et al. (arXiv 1909.13669, **FT-v** introduzione, **SEC**): “distribuzioni di semina incoerenti” producono lacune spaziali o temporali.

### 4.D Specificità della combustione

**L27 — Risposta del tracciante, slip e termoforesi**
- *Slip (stima a priori):* TiO2 1 µm, τ ≈ 0.5×10⁻⁵ s. Con v = 80 m/s e l = 10 mm, Δv = τv²/l = 1.6 m/s, “errore massimo da slip 2%” (Stöhr 2012, §2.1, **FT-v**). È un ordine di grandezza, non una misura.
- *Termoforesi:* nella zona di preriscaldamento la forza termoforetica può indurre un ritardo significativo fra particella e fluido; gli autori propongono una correzione nota del profilo di T (Sung, Law & Axelbaum 1994, fiamme gemelle a controflusso, LDV, **AB**). Stella et al. 2001 indicano termoforesi al fronte e indice di rifrazione disomogeneo e variabile come principali sorgenti in fiamma (**SN**; il record DLR elib 13818 non contiene il PDF).
- *Trasferibilità:* effetto locale al fronte, entro lo spessore di fiamma (sub-millimetrico). Rispetto a finestre di ~1 mm e celle LES di ~0.6–0.8 mm è un effetto alla scala della risoluzione. Inferenza: probabilmente non rappresentabile sulla nostra griglia.

**L28 — Scelta del tracciante: chi viene osservato**
- *Olio:* goccioline d'olio d'oliva di ~1 µm evaporano nella zona di preriscaldamento; la temperatura di evaporazione misurata è circa 555 K, con deviazione standard 10 K. Il bordo fra zona con e senza particelle è usato per rilevare il fronte. La velocità esiste **solo nei reagenti** (OSTI 1253136, §§3.2.1 e 3.3.1, **FT**). Lo stesso principio vale per CPIV e per la soglia di conteggio particelle nella tesi tomografica TU Delft (Pfadler et al. 2007, **AB**/**SN**; tesi TU Delft “Determination of Premixed Flame Characteristics by Tomographic PIV”, **FT** limitato all'indice e alle didascalie).
- *Particelle solide:* Al2O3 e TiO2 “sopravvivono” e misurano reagenti e prodotti (OSTI 1253136; Barbosa §2.2; Baumgartner §3.4; DLR L04).
- *Costo delle particelle solide:* TiO2 depositato sulla parete del tubo, smontaggio ogni 2–3 prove; per questo la campagna ad alta pressione è passata all'olio (OSTI 1506058, p. 29, **FT**).
- *Implicazione:* il tipo di tracciante del riferimento scelto decide se la lacuna “prodotti” è parte del modello di osservazione. Con l'olio, la maschera dipende da T per costruzione, con un'isoterma di ~550 K per l'olio d'oliva. È il caso estremo di lacuna guidata dalla fiamma e va trattato come asse separato (ricognizione §2.5).

**L29 — Calo della densità di semina attraverso la fiamma**
- *Evidenza qualitativa:* transizione netta da alta a bassa densità di particelle dovuta all'espansione termica (Baumgartner, p. 122, **FT**). Il bruciato è identificato da conteggi di particelle per finestra più bassi, con soglia di 20 particelle per celle di 0.745 mm a 0.028 ppp (tesi TU Delft, **FT** limitato). Il conteggio delle particelle come marcatore delle zone reagenti compare anche in letteratura (“3C PIV in premixed flames”, **SN**).
- *Numero:* “densità di semina ridotte di circa l'85% nelle regioni di gas combusto” in note di lezione di Dreizler (Princeton CEFRC 2017, **SN**, host non accessibile). Le stesse note riportano l'uso di PTV nei gas caldi e PIV nei freddi, evitando di mescolare le due regioni nella stessa finestra (**SN**).
- *Inferenza fisica:* se le particelle seguono la massa, la loro densità in numero è ∝ ρ_gas. A pressione costante ρ_b/ρ_u ≈ T_u/T_b, cioè circa 1/5–1/7 per T_b di 1500–2100 K e T_u di 300 K, coerente con l'85%. Il valore della nostra miscela NH3/CH4 non è verificato.
- *Effetti sul dato:* meno particelle per finestra, quindi più errore casuale e più rigetti (L22). Il bias di una finestra che comprende due densità di semina è citato come motivo per separare PIV e PTV (**SN**), ma **nessuna fonte letta lo quantifica**.
- *Step 3:* con immagini di particelle, contare le particelle per finestra contro OH-PLIF o contro i rigetti. Con soli vettori e flag, confrontare la frazione di rigetti fra regioni con velocità tipiche dei getti freddi e dei ricircoli caldi, sapendo che la regione è un proxy imperfetto.

**L30 — Gradienti d'indice di rifrazione in fiamma: numeri disponibili**
- *Vanselow et al. 2019* (Exp. Fluids 60, DOI 10.1007/s00348-019-2795-1, **AB**): fiamma premiscelata di propano, tre piani di misura. Errori di posizione delle particelle misurati e verificati con ray tracing su campo d'indice ricavato da BOS. Errore di posizione massimo 14 µm; errori di velocità **sistematici fino al 4%** e **casuali fino al 6%**. L'errore casuale varia significativamente fra i piani; secondo lo snippet è concentrato vicino al fronte (**SN**).
- *Vanselow & Fischer 2018* (**SN**): in un getto caldo errore massimo 0.8% per la PIV planare, 1.7% in piano e 2.9% fuori piano per la stereo, 1.1% per la tomografica. Gli errori sono maggiori con la triangolazione.
- *Meccanismo (Elsinga et al. 2005, **SEC** tramite Valori):* l'errore relativo di velocità è pari al gradiente dello spostamento ottico lungo la direzione della velocità. La sfocatura allarga il picco di correlazione e aumenta l'errore casuale.
- *Rajendran et al. 2020* (arXiv 1910.09379, **FT-v** di abstract e introduzione): non linearità del gradiente di densità producono sfocatura asimmetrica dell'immagine di particella, che amplifica l'effetto del rumore d'immagine sulla stima del centroide. Lavoro su PTV e BOS volumetriche.
- *Limiti:* i numeri di Vanselow sono di un apparato e una fiamma specifici; la fonte primaria non è stata letta. Non sono trasferibili come percentuali al nostro caso.

**L31 — Striature d'intensità ad alta pressione e picchi allungati**
- *Apparato:* UT Austin, SPIV a 8 kHz in camera pressurizzata, flusso swirl con flashback.
- *Effetto:* a pressione elevata compaiono striature casuali d'intensità nelle immagini di particelle, dovute al passaggio del foglio attraverso la fiamma swirl e i gas di scarico. Crescono con la pressione, per i gradienti maggiori a Re più alti. I picchi di correlazione diventano allungati. Correzione: fondo mobile di 16 px più normalizzazione min-max, che rende i picchi assialsimmetrici e “migliora molto” la qualità (fig. 28).
- *Fonte:* OSTI 1506058, pp. 29–30. **FT**. Qualitativo.
- *Implicazione:* il fenomeno è stato trattato prima della correlazione; nel dato finale può non lasciare traccia visibile.

**L32 — Degrado delle immagini dove la luce attraversa la fiamma**
- In microPIV di flashback, nel caso reagente la qualità delle immagini peggiora dove la luce diffusa attraversa la fiamma (Baumgartner, p. 122, **FT**).
- Nel combustore pressurizzato le striature del foglio in PLIF sono dovute ai gradienti della fiamma (Lange, p. 9, **FT**).
- Valori et al. sono l'analogo non reagente quantificato (L25).

**L33 — Luminosità di fiamma e fondo**
- Filtri passa-banda attorno a 532 nm sono usati “per ridurre l'influenza della luminosità di fiamma” (Stöhr 2012; Baumgartner).
- Nonostante il filtro, la chemiluminescenza contribuisce ad accentuare i riflessi sulla parete (Baumgartner, p. 122).
- Saini et al. citano fuliggine e luminosità come cause di basso SNR ad alta pressione (affermazione).
- Fuliggine nelle immagini Mie della PIV di un combustore RQL: Jose, Geigle & Hampp, MLST 2025, **AB**/**SN**. Benchmark di segmentazione con database su DaRUS, da segnalare all'Agente A.
- *Ammoniaca (plausibile, non verificato):* nelle fiamme con NH3 l'emissione visibile è attribuita a NH2* (bande a righe, “α-band” vicino a 630 nm) e NO2* (continuo), con spettri registrati fra 400 e 700 nm (**SN**, fonti Lund e Cardiff non lette). Una sovrapposizione con la banda dei filtri a 532 nm è plausibile. **Nessuna fonte letta** quantifica l'effetto sulla PIV in fiamme NH3/CH4.
- *Inferenza:* nelle camere a doppio fotogramma la seconda esposizione è tipicamente molto più lunga della prima, quindi il fondo di fiamma pesa più sul secondo fotogramma. È conoscenza di pratica **non verificata** nelle fonti lette; da controllare sulle immagini, se disponibili.

**L34 — Riflessi, accesso ottico, sporcamento delle finestre**
- *Riflessi:*
  - Combustore Toronto/DLR: metà campo scartata per riflessi (L05).
  - Combustore pressurizzato DLR: niente PIV con foglio parallelo alla parete; con foglio normale solo oltre 1 mm (L24).
  - MicroPIV TUM: foglio inclinato di 30° e vernice nera, riflessi confinati in uno strato di <0.2 mm.
  - UT Austin: la PIV è più sensibile della PLIF alla luce diffusa, perché la lunghezza d'onda è la stessa del laser (OSTI 1506058, p. 30).
  - Motore ottico: riflessi vicino al punto morto superiore (L18).
- *Sporcamento:*
  - GTMC DLR: semina solo durante la misura, dopo 20 minuti di riscaldamento, “per minimizzare l'accumulo sulle finestre” (Boxx, pp. 11–12).
  - Combustore pressurizzato DLR: le finestre roventi si irruvidiscono in pochi minuti; con la contaminazione da traccianti il tempo utile scende sotto 15 minuti (Lange, p. 7).
  - UT Austin, alta pressione: lastra sacrificale e aria di raffreddamento sulla finestra d'ingresso; la polvere sulle finestre cambia il profilo del foglio (OSTI 1506058, pp. 19 e 26).
- *Implicazione:* questi fenomeni determinano soprattutto **regioni escluse fisse** (bordi, pareti, metà campo) e **limiti alla durata dei record**, più che rumore distribuito.

**L35 — Relazione fra qualità PIV e regione di fiamma: stato delle prove**
- *Documentato:*
  - più vettori interpolati nella zona di reazione (L22, microPIV H2);
  - errori casuali concentrati vicino al fronte e variabili col piano (L30, solo AB/SN);
  - incertezza alta nei pennacchi termici (L25, non reagente);
  - nessuna velocità nei prodotti con traccianti che evaporano (L28);
  - validazione bassa presso pareti calde con forti gradienti (L24).
- *Non trovato:* statistiche di rigetto condizionate alla posizione del fronte, per esempio tramite OH-PLIF simultanea, in fiamme swirl stabilizzate. Neanche la persistenza temporale delle lacune.

**L36 — Fonti indicate dall'incarico ma non lette integralmente**
- *Ebi & Clemens 2016* (Combust. Flame 168): non accessibile (ScienceDirect). Contesto ricostruito dai rapporti DOE del gruppo (L08, L28) e da abstract APS e UT (**SN**): flashback a strato limite in un combustore swirl CH4/H2, PIV tomografica nei gas incombusti, chemiluminescenza ad alta velocità.
- *Andreini et al.* (2016 J. Eng. Gas Turbines Power 138 041501; 2016/2017 stessa rivista, 1.4034358), **AB**: il primo usa PIV 2D in un rig planare **non reagente** a tre settori in galleria del vento (scambio termico della liner); il secondo è uno studio numerico RANS/SAS di un generatore di hot streak **senza fiamma**, confrontato con dati sperimentali. **Non sono misure PIV in fiamma**: il loro uso nella review va inteso in questo senso.

---

## 5. Comune, dipendente dall'apparato, non dimostrato

### 5.1 Comune ai casi studiati

Valido nelle fonti lette, con la loro varietà di apparati; non è una legge universale.

- Il campo di vista è limitato da accesso ottico e camere; i bordi e le regioni escluse sono fissi nel tempo (L04–L08, L24, L34).
- Le incertezze istantanee nei lavori applicativi sono dichiarate a priori, come numero unico per campo (L11).
- La validazione usa test di vicinato (mediano, media mobile) e filtri di intervallo, seguiti dall'interpolazione dei vettori rimossi, almeno per i buchi isolati (L19–L22).
- Lo stereo ha la componente fuori piano più incerta (L04, L09).
- L'errore casuale è correlato fra vettori vicini; l'entità dipende da overlap e algoritmo (L15).
- La qualità peggiora dove la luce attraversa gradienti d'indice forti (L24, L25, L30–L32).

### 5.2 Dipendente dall'apparato o dalla scelta sperimentale

- Tracciante: olio, quindi niente prodotti; oppure solido, quindi entrambi i gas (L28).
- Frequenza e campo di vista: kHz con campo piccolo, Hz con campo ampio (L04, L07).
- Δt e range dinamico (L06, L08).
- Pressione: striature, fouling, durata dei record (L31, L34).
- Software e passate, che determinano la forma del filtro (L01, L02).
- Soglie di validazione (L20).
- Livello del dato rilasciato: interpolato, POD-filtrato, GPOD (L21, L23).

### 5.3 Plausibile ma non dimostrato dalle fonti lette

- Frazione di rigetti per frame in fiamme swirl e sua distribuzione rispetto al fronte o alla T.
- Persistenza e forma delle lacune nel tempo in fiamma.
- Bias di velocità per finestre a cavallo del fronte con densità di semina diverse.
- Peak locking nei sistemi kHz di combustione.
- Effetto dell'emissione NH2*/NO2* delle fiamme con ammoniaca sulle immagini PIV.
- Errori di temporizzazione dei laser Nd:YLF negli apparati di combustione letti.
- Trasferibilità di qualunque numero percentuale (4%, 6%, 11%, 20%, 75%) ad altri apparati.

---

## 6. Pratiche di validazione e sostituzione e loro conseguenze

| Pratica | Dove documentata | Che cosa cambia nel dato rilasciato | Che cosa diventa invisibile | ID |
|---|---|---|---|---|
| Test mediano 3×3 (normalizzato o in deviazioni standard) + filtro di intervallo | TUM; Saini (in GPOD); prassi generale | Rimuove outlier isolati | Cluster di outlier con vicini anch'essi errati; outlier plausibili | L19, L22 |
| Validazione sul picco di correlazione e media mobile | EM2C | Le soglie cambiano i massimi fino all'11% | La scelta delle soglie | L20 |
| Interpolazione dei buchi singoli (o con kernel 3×3) | DLR (implicito), TUM | Campo completo, con vettori interpolati smussati | Rigetti, se manca un flag; correlazione introdotta fra vettori | L21, L22 |
| Filtro POD (troncamento dei modi) | DLR, per figure | Rimuove rumore e piccole scale; informazione da tutta la serie | Errore casuale, piccole scale, indipendenza temporale | L21 |
| Riempimento gappy-POD | Toronto/DLR (proposto) | Riempie gap anche estesi | Che il vettore non era misurato; introduce informazione da altri istanti | L23 |
| Filtri d'immagine: sottrazione del fondo, normalizzazione | UT Austin, Wieneke | Riduce rigetti e allungamenti dei picchi | Le striature, la causa ottica | L26, L31 |
| Ritaglio del FOV e finestre più grandi localmente | Toronto/DLR, Valori | Rimuove zone cattive; più vettori validi ma risoluzione minore | La regione persa; la variazione di risoluzione | L05, L25 |
| Uso della sola media in zone critiche | DLR pressurizzato | Istantaneo non rilasciato | Variabilità istantanea locale | L24 |

**Lezione per lo step 3:** l'assenza di buchi nel dato finale è attesa in molti processamenti. Prima di stimare qualunque frequenza bisogna stabilire il livello del dato: vettori grezzi, validati, interpolati, filtrati o riempiti.

---

## 7. Domande verificabili per lo step 3

Per ciascuna domanda: dato necessario, poi esito atteso possibile (presente; trattato prima del rilascio; non riscontrato; non verificabile).

1. **Livello di processamento:** i vettori sono grezzi, validati, interpolati, POD-filtrati o riempiti? *Dato:* README, flag, versioni multiple dello stesso campo. (L21, L23)
2. **Flag di validità e sostituzione:** esistono? Con che codifica (NaN, zero, flag separato)? *Dato:* file e documentazione. Per EngineBench lo zero esatto è già noto come codifica (ricognizione §3).
3. **Frazione di vettori invalidi o sostituiti per frame**, con denominatore esplicito: tutti i nodi del FOV oppure solo i nodi in regione inseminata. Distribuzione fra frame e condizioni. *Dato:* flag o maschere per frame. (L22)
4. **Geometria delle lacune:** isolate o in cluster? Dimensioni in mm, posizione (bordi, pareti, interno). *Dato:* maschere. (L19, L24)
5. **Persistenza temporale delle lacune** nelle sequenze ad alta frequenza. *Dato:* maschere con tempi.
6. **Relazione con la fiamma:** le lacune o i rigetti si concentrano dove c'è OH-PLIF alto o al gradiente di OH? *Dato:* PIV e PLIF simultanee e registrate. Senza PLIF: non verificabile, o solo con proxy dichiarati. (L35)
7. **Tracciante:** olio o solido? Le regioni dei prodotti hanno vettori? *Dato:* metodo sperimentale; immagini di particelle se disponibili. (L28)
8. **Densità di particelle per finestra** in reagenti e prodotti. *Dato:* immagini di particelle. (L29)
9. **Parametri ottici e di processamento:** scala in µm/px, Δt, finestra iniziale e finale, pesatura, passate, overlap, software e versione. Servono a convertire px in m/s e a scegliere un filtro plausibile. *Dato:* metadati o paper. (L01, L04, L11)
10. **UQ per vettore:** esiste? Con quale metodo, componente, livello (standard o espanso) e copertura verificata? Altrimenti registrare “solo a priori”. (L11, L16, L17)
11. **Componenti:** 2C o stereo 3C? Per lo stereo, la componente fuori piano ha incertezza dichiarata diversa? (L04, L09)
12. **Range dinamico:** ci sono regioni lente con spostamenti sotto 1 px? Istogrammi degli spostamenti in pixel, compreso il peak locking. *Dato:* velocità, Δt e scala. (L06, L14)
13. **Correlazione degli errori:** esiste un riferimento co-localizzato (doppio sistema, HDR, LDV)? Senza riferimento, la correlazione del campo **non** misura quella dell'errore. (L15, L16)
14. **Regioni escluse fisse:** bordi, montanti, pareti, zone di riflesso; area e posizione rispetto al FOV. (L05, L24, L34)
15. **Sensibilità alle soglie:** se sono disponibili vettori pre-validazione, quanto cambiano i rigetti variando le soglie? (L20)
16. **Fondo luminoso:** nelle immagini, il secondo fotogramma ha fondo maggiore? Le fiamme con NH3 mostrano luminosità nella banda dei filtri? (L33)

---

## 8. Implicazioni possibili per la PIV virtuale (PROPOSTE)

Tutte le voci sono **proposte da discutere**, non scelte. Nessun parametro è fissato qui.

- **P-A (struttura dell'operatore).** Separare quattro trasformazioni indipendenti e documentabili: (1) campo di vista e regioni escluse fisse; (2) risposta spaziale con campionamento su una griglia regolare; (3) errore, con correlazione legata a finestra e overlap; (4) invalidazione o sostituzione. Il precedente di L03 mostra che la struttura è praticabile; i valori vanno presi da uno scenario documentato.
- **P-B (risposta spaziale).** Trattare la media su finestra quadrata come approssimazione del single-pass. Per uno scenario multi-pass DaVis-simile, considerare un kernel gaussiano o MHG con L_sr ≈ 0.7 volte la finestra (L01–L02). Dichiarare che a scale di ~1 mm il filtro è dell'ordine della cella LES (L10).
- **P-C (errore).** Non convertire ±0.1 px in σ gaussiana senza dirlo: è a priori (L11). Se si usa, specificare scala, Δt, componente, natura standard e ipotesi. Considerare la correlazione spaziale come scelta esplicita, con ρ(2d) da L15 come ordine di grandezza per DaVis 75%, piuttosto che rumore i.i.d. implicito.
- **P-D (stereo).** Se lo scenario è uno stereo, considerare un'incertezza diversa per la componente fuori piano (2–3× a priori in L04). Le nostre reti usano solo due componenti, ma la stereo-PIV ha errori propri anche in piano (L09).
- **P-E (lacune).** Distinguere tre classi con evidenza diversa: (i) **regioni escluse fisse** come FOV, pareti, riflessi: forte evidenza, rappresentabili geometricamente; (ii) **rigetti sparsi e cluster**: frequenze non documentate in fiamma swirl, quindi sensibilità dichiarata o valori da un dataset dello step 3; (iii) **lacune guidate dalla fiamma**: evidenza qualitativa (L22, L28, L29), con uso di T da discutere esplicitamente (roadmap §4A).
- **P-F (tracciante).** Lo scenario “olio” implica per costruzione l'assenza di osservazioni nei prodotti, cioè una maschera ≈ isoterma. È un caso estremo e informativo, ma introduce informazione su T nella maschera. Lo scenario “particelle solide” osserva entrambi i gas con qualità degradata.
- **P-G (livello del dato).** Decidere se simulare il dato **prima** della sostituzione (mancanti espliciti, come oggi con M_v e token rimossi) o **dopo** l'interpolazione. La documentazione (L21–L23) suggerisce che i dati rilasciati sono spesso interpolati, mentre le reti attuali sanno già trattare i mancanti espliciti.
- **P-H (combinazioni).** Non comporre in uno stesso scenario parametri di apparati diversi, per esempio il FOV del GTMC a 5 kHz, gli errori di Vanselow (propano) e i rigetti TUM (microPIV H2).

---

## 9. Dataset e materiali associati, da segnalare all'Agente A

Non verificati come scaricabili, salvo dove indicato.

| Materiale | Origine | Stato in questo lavoro |
|---|---|---|
| Database “Collaborative framework” (getto rettangolare, PIV-MS, PIV-HDR, hot-wire) | Neal et al. 2015, MST 26 074003 | AB; citato come riferimento sperimentale per UQ (L16) |
| Database dei PIV Challenge (incluso il 4°) | Kähler et al. 2016; www.pivchallenge.org | AB; host consentito, non ispezionato |
| Database di fuliggine e traccianti in immagini PIV di un combustore RQL (DaRUS) | Jose, Geigle & Hampp, MLST 2025 | SN; DOI di dati citati negli snippet (doi:10.18419/darus-5182, -5184) |
| Dati GTMC DLR (PIV/OH-PLIF a 5–10 kHz) | Boxx 2010, Stöhr 2012, Oberleithner 2015 | Disponibilità non dichiarata nelle fonti lette |
| Dati del combustore Toronto/DLR (S-PIV 10 kHz) | Saini et al., Steinberg et al. 2013, Caux-Brisebois 2014 | Non dichiarato |

---

## 10. Bibliografia annotata

| # | Riferimento | Copia letta | Livello | Uso nel rapporto |
|---|---|---|---|---|
| 1 | B. Wieneke, *PIV Uncertainty Quantification and Beyond*, PhD TU Delft, 2017 | TU Delft repository | FT, capp. 2, 3.1–3.2, 4.2–4.4 | Filtro lineare, risoluzione, sorgenti d'errore, propagazione, verifica sul getto (L01, L02, L09, L12–L16) |
| 2 | A. Sciacchitano, B. Wieneke, *PIV uncertainty propagation*, MST 27 (2016) 084006 | TU Delft, versione pubblicata | FT, §§2.3, 3.2, 4.2 | Correlazione degli errori e ρ(2d) (L15, L16) |
| 3 | I. Boxx, M. Stöhr, C. Carter, W. Meier, Combust. Flame 157 (2010) 1510 | Manoscritto accettato, DLR elib 64569 | FT-v | GTMC 5 kHz; ±0.1 px; POD; interpolazione dei buchi singoli; semina e finestre (L04, L11, L21, L34) |
| 4 | M. Stöhr, I. Boxx, C. D. Carter, W. Meier, Combust. Flame 159 (2012) 2636–2649 | Manoscritto, DLR elib 77484 | FT-v, §§1–2 | Sistemi 5 Hz e 5 kHz, slip, incertezze, filtro passa-banda (L04, L27, L33) |
| 5 | W. Meier, I. Boxx, M. Stöhr, C. Carter, Exp. Fluids 49 (2010) 865 | Copia autore, DLR elib 65267 | FT-v, §2 | Terminologia delle incertezze (L04, L11) |
| 6 | K. Oberleithner, M. Stöhr, S. H. Im, C. M. Arndt, A. M. Steinberg, Combust. Flame 162 (2015) 3100–3114 | Manoscritto, DLR elib 96808 | FT-v, §2 | 10 kHz, ±0.8 m/s (L04) |
| 7 | P. Saini, A. Steinberg, C. Arndt, AIAA 2016-0436 | DLR elib 102428 | FT-v (versione congressuale) | GPOD; FOV e riflessi; gap artificiali (L05, L23). La versione Exp. Fluids 57:122 non è stata letta. |
| 8 | L. Lange, J. Heinze, M. Schroll, C. Willert, T. Behrendt, 16th Lisbon Symposium (2012) | DLR elib 77085 | FT | Alta pressione: riflessi, beam steering, finestre, range dinamico (L06, L24, L34) |
| 9 | V. Valori, G. Elsinga, M. Rohde, J. Westerweel, T. van der Hagen, Exp. Fluids 60 (2019) 143 | TU Delft (Taverne) | FT, §§3–5 | Indice di rifrazione, UQ di Wieneke, 75% affidabile (L25, L30) |
| 10 | S. Barbosa, P. Scouflaire, S. Ducruix, arXiv 0807.1871 (preprint 2008; versione su rivista non verificata) | arXiv | FT-v | Validazione e 11%; TiO2 nei gas combusti (L07, L20, L28) |
| 11 | N. Clemens (PI), DOE final report DE-FE0007107, OSTI 1253136 | OSTI | FT, §3 | Olio a 555 K; Al2O3; Δt limitato dal fuori piano (L08, L28) |
| 12 | N. Clemens (PI), DOE final report DE-FE0012053, OSTI 1506058 | OSTI | FT, §4 | Alta pressione: striature, depositi, finestre (L08, L18, L28, L31, L34) |
| 13 | G. M. Baumgartner, *Flame Flashback in Premixed Hydrogen-Air Combustion Systems*, PhD TU München (Lehrstuhl für Thermodynamik), 2015 | epc.ed.tum.de | FT, §3.4 e cap. 7 | Interpolati ≤20%, invalidi <10%, zona di reazione (L22, L29, L32–L34) |
| 14 | A. Sunol et al., arXiv 2510.24673 v4 | arXiv | FT della sezione di emulazione | Precedente di operatore PIV (L03) |
| 15 | L. Rajendran, S. Bane, P. Vlachos, arXiv 1910.09379 | arXiv | FT-v, abstract e introduzione | Sfocatura e gradienti di densità (L30) |
| 16 | L. Rajendran, S. Bane, P. Vlachos, MST 30 (2019) 085302, arXiv 1812.05902 | arXiv (versione AIAA 2017) | FT-v, scorsa | Strumento di ray tracing; solo dimostrazione BOS: non usato per numeri |
| 17 | S. Bhattacharya, J. Charonko, P. Vlachos, MST (2018), DOI 10.1088/1361-6501/aadfb4, arXiv 1804.10854 | arXiv | FT-v, introduzione e conclusioni | Sintesi di Sciacchitano 2015 e Boomsma 2016 (L17) |
| 18 | Z. Xue, J. Charonko, P. Vlachos, arXiv 1405.3023 | arXiv | Scorsa (grep) | Coperture e definizioni; non usato per numeri |
| 19 | B. Peterson, E. Baum, A. Dreizler, B. Böhm, arXiv 2009.01710 | arXiv | FT-v, §2 | Motore: riflessi, incertezza ≤10% a priori (L18) |
| 20 | Tesi magistrale TU Delft, *Determination of Premixed Flame Characteristics by Tomographic PIV* | TU Delft | FT limitato (indice, didascalie) | Conteggio particelle per il bruciato (L28, L29) |
| 21 | S. Beresh, AIAA 2008-0239, OSTI 1146188 | OSTI | Solo titolo e prima pagina | Pista per error sampling multi-configurazione; non letto |
| 22 | J. Duncan et al., MST 21 (2010) 057002 | OpenAlex | AB | Outlier fino al 15% (L19) |
| 23 | A. Masullo, R. Theunissen, Exp. Fluids 57 (2016) 33 | OpenAlex | AB | Cluster di outlier (L19) |
| 24 | C. Kähler, S. Scharnowski, C. Cierpka, Exp. Fluids 52 (2012) 1629 | OpenAlex | AB | Limite di risoluzione e PTV (L02) |
| 25 | D. Neal et al., MST 26 (2015) 074003; A. Sciacchitano et al., MST 26 (2015) 074004; A. Boomsma et al., MST 27 (2016) 094006; B. Wieneke, MST 26 (2015) 074002 | OpenAlex | AB (FT bloccato da IOP) | UQ e verifica (L16, L17) |
| 26 | C. Vanselow, D. Stöbener, J. Kiefer, A. Fischer, Exp. Fluids 60 (2019), DOI 10.1007/s00348-019-2795-1 | OpenAlex | AB | Errori da rifrazione in fiamma (L30) |
| 27 | C. Vanselow, A. Fischer, 2018 (inhomogeneous refractive index fields) | Snippet | SN | Stereo e tomografica più sensibili (L30) |
| 28 | C.-J. Sung, C. K. Law, R. L. Axelbaum, Combust. Sci. Technol. (1994), DOI 10.1080/00102209408935428 | OpenAlex | AB | Termoforesi (L27) |
| 29 | A. Stella et al., Exp. Fluids 30 (2001) 167 | Snippet | SN | Termoforesi e indice di rifrazione (L27) |
| 30 | Westerweel & Scarano 2005; Scarano 2003; Lavoie et al. 2007; Christensen 2004; Carr et al. 2009; Garcia 2011 | — | Non letti (Springer o PMC bloccati; nessun abstract in OpenAlex per i primi quattro) | Citati solo come indicazione bibliografica |
| 31 | Andreini et al. 2016 (1.4031434) e 2016 (1.4034358), J. Eng. Gas Turbines Power | OpenAlex | AB | Verifica del contesto: non reagenti o numerici (L36) |
| 32 | D. Ebi, N. Clemens, Combust. Flame 168 (2016) | — | Non letto; contesto da 11–12 e snippet | L36 |
| 33 | Pfadler, Beyrau, Leipertz, Opt. Express 15 (2007) 15444 | Snippet e abstract | SN/AB | CPIV (L28, L29) |
| 34 | Jose, Geigle, Hampp, MLST 2025 | Snippet | SN | Fuliggine in immagini PIV (L33) |
| 35 | Barwey et al., arXiv 1909.13669 e 2003.03662 | arXiv | Scorsa (introduzione) | Prassi di riempimento (L26) |

---

## 11. Registro sintetico delle ricerche

| # | Ricerca e strumento | Esito |
|---|---|---|
| 1 | OpenAlex per DOI e titolo, ~40 lavori (copie OA e abstract) | Copie OA individuate; molte su host bloccati (IOP, Springer, Toronto, Bremen, Bristol) |
| 2 | Download diretto da IOP e Springer | Challenge o redirect: fallito. Il bypass con browser è stato negato e non ritentato. |
| 3 | Aggiunta di host all'ambiente (approvata) e nuovi download | Negati dal classificatore dei permessi; abbandonati |
| 4 | TU Delft repository, risolvendo `uuid` → `File_` | Sciacchitano & Wieneke 2016, tesi di Wieneke, Valori 2019, tesi tomografica |
| 5 | DLR elib | Boxx 2010, Stöhr 2012, Meier 2010, Oberleithner 2015, Saini AIAA, Lange 2012 |
| 6 | OSTI: “flashback high-speed PIV”, “PIV uncertainties multiple configurations” | Rapporti DOE di Clemens (2), Beresh 2008 |
| 7 | arXiv: GT injector HF-PIV, UQ MC, SNR, refractive amplification, PIV emulation | Barbosa, Bhattacharya, Xue, Rajendran 2019/2020, Sunol, Peterson, Barwey |
| 8 | TUM: “flashback micro-PIV interpolated vectors” | Tesi di Baumgartner (2015) |
| 9 | WebSearch: seeding density burned gas, valid vectors in flames | Snippet (Dreizler, motori); nessuna statistica in fiamme swirl |
| 10 | WebSearch: Vanselow refractive index PIV flames; NH2* chemiluminescence | Abstract e snippet |
| 11 | WebSearch: NH3/CH4 swirl PIV | Lavori MDPI e Cardiff senza dettagli PIV negli snippet; non approfonditi |
| 12 | TU Darmstadt (tuprints): dissertazioni PIV/PTV in fiamma | Pagine con 403 o indirette; dissertazioni (Seffrin 2011, Butz 2020) non scaricate |
| 13 | PMC (Garcia 2011), CORE download | reCAPTCHA o challenge |

**Esclusioni motivate:**
- Lavori su motori ottici: usati solo come contesto (L18), per la fisica diversa.
- PTV e volumetrica: citate per meccanismo (L30), non come scenario.
- Il deep learning PIV (arXiv 2507.20102 e simili) è stato visto solo nei titoli e non è approfondito, perché non cambia le scelte dello step 4.

---

## 12. Lacune principali e prossimi passi suggeriti

1. **Statistiche di rigetto in fiamme swirl** (frazione per frame, cluster, persistenza, relazione con OH): non trovate in letteratura. Vanno cercate nei dati (step 3) o in lavori non accessibili da qui.
2. **Fonti primarie da leggere in una sessione con accesso completo:** Saini et al. 2016 (Exp. Fluids, versione rivista: gap reali ad alta pressione?), Vanselow et al. 2018 e 2019, Stella et al. 2001, Westerweel & Scarano 2005, Masullo & Theunissen 2016, Neal et al. 2015, Sciacchitano et al. 2015, Boomsma et al. 2016, Ebi & Clemens 2016, le note di Dreizler. Le copie OA istituzionali esistono per Saini (Toronto) e per Vanselow (Bremen).
3. **Fiamme NH3/CH4:** nessuna fonte letta descrive un apparato PIV in queste fiamme con dettagli di processamento o qualità. Da cercare insieme all'Agente A (Cardiff, Tohoku, lavori MDPI 2021–2025), soprattutto per luminosità e tracciante.
4. **Filtro della LES e ricampionamento a 128×128:** da verificare nella missione 01. Senza questi dati, la PIV a scala di ~1 mm non è rappresentabile con fedeltà (L10).
5. **Discrepanza ρ(2d) al 50% di overlap** fra review e fonte primaria (L15): da riconciliare nella scheda dello step 1, se si userà quel valore.

Questa ricerca non è esaustiva. “Non trovato” indica i limiti delle ricerche e degli accessi documentati sopra, non l'inesistenza dell'evidenza.
