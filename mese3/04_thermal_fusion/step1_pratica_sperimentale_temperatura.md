# Missione 04 — Step 1: come si misura davvero la temperatura negli esperimenti di combustione con accesso ottico

**Versione:** v1, consegna dello scouting dello step 1.  
**Data:** 26 novembre 2026.  
**Stato:** **proposta da revisionare nella chat di coordinamento della missione 04. Non approvata.**  
**Branch:** `claude/thermal-measurement-scouting-rxj1l0`, imposto dall'ambiente. Parte dalla base indicata `a894257` (vedi §15).  
**Perimetro:** solo scouting bibliografico dello step 1 di [roadmap04.md](roadmap04.md). Nessun training, inferenza, download di dataset, implementazione di sensori virtuali, contatto con autori o scelta di dataset, architettura, numero o posizione dei sensori.  
**Documento collegato:** [step1_fonti_e_ricerche.md](step1_fonti_e_ricerche.md). Contiene bibliografia ragionata, registro delle ricerche, accessi falliti ed esclusioni.

---

## 0. Come leggere questo rapporto

**Etichette** (come nelle consegne delle missioni 01 e 02):

- **[D] Documentato.** Affermato nella fonte, con posizione (sezione, pagina, figura, tabella). Le citazioni tra virgolette sono testuali.
- **[I] Inferenza.** Una mia interpretazione motivata; non è un'affermazione degli autori.
- **[M] Mancante.** Informazione non riportata nelle fonti lette, o fonte non accessibile.

**Altre convenzioni:**

- **"Non riportato" non significa "non misurato".** Scrivo "non misurato" solo quando gli autori lo dicono esplicitamente.
- **"Nello stesso paper" non significa "simultaneo".** Distinguo sempre fra misure sincronizzate, campagne separate sullo stesso apparato e campagne in laboratori o condizioni diverse.
- **Ogni numero vale per l'apparato e la condizione indicati.** Non è una proprietà generale della tecnica né del combustore RealPDEBench.
- **Le quantità "±" sono riportate come le scrivono gli autori.** Aggiungo il significato statistico (1σ, 2σ, incertezza combinata) solo se dichiarato.
- **Identificativi dei casi:**
  - `GT-…`: combustori modello per turbine a gas e combustori confinati;
  - `NH-…`: rig ad ammoniaca;
  - `LF-…`: fiamme da laboratorio aperte;
  - `EN-…`: motori ottici;
  - `RP`: RealPDEBench.

  La tabella completa è al §10.
- **Profondità di lettura.** Ogni fonte è indicata come testo integrale (**TI**), parziale (**PZ**), abstract (**AB**) o secondaria/estratto di motore di ricerca (**SEC**). Le affermazioni decisive poggiano su TI. Le eccezioni sono segnalate.

**Come è stato prodotto.** Sette filoni di ricerca paralleli:

- combustori swirl per turbine a gas;
- swirl ad ammoniaca;
- fiamme da laboratorio;
- motori ottici;
- strumenti;
- scopi e assenze;
- RealPDEBench.

Ho integrato i risultati, verificato a campione sui PDF originali le affermazioni decisive e risolto le incongruenze. La verifica a campione è descritta in [fonti §2](step1_fonti_e_ricerche.md). Molti testi editoriali erano inaccessibili dall'ambiente; lacune e accessi falliti sono documentati.

---

## 1. Risposta diretta

### 1.1 In una frase

**Sì, negli esperimenti con accesso ottico e PIV si misura quasi sempre *qualche* temperatura. Raramente però è la temperatura del gas nella regione inquadrata dalla PIV, e quasi mai nello stesso istante della velocità.**

Le temperature misurate di routine sono di contorno:

- l'aria o la miscela all'ingresso;
- le superfici solide (pareti, finestre, bluff body, piano di dump, pistone);
- lo scarico;
- nei motori, refrigerante e olio.

La temperatura del gas in fiamma si misura con tecniche laser dedicate, in campagne separate dalla PIV.

### 1.2 Che cosa troveresti entrando in laboratorio

Parlo di un combustore swirl confinato da laboratorio con PIV. Fra parentesi il caso documentato.

1. **Una o poche termocoppie "di servizio", lente e puntuali, fuori dalla zona ottica.**
   - Dove: nel plenum dell'aria, nel condotto d'ingresso, a volte in un rastrello allo scarico.
   - Servono a fissare e verificare il punto di funzionamento e a capire quando il rig è in equilibrio termico. Per esempio, una termocoppia tipo K in una barra d'acciaio che regge le finestre segnala lo stato stazionario dopo circa 45 min (GT-EM2C).
   - Spesso il paper riporta solo "aria a temperatura ambiente", senza dire con che cosa e dove sia stata misurata (GT-DLR-SFB606, GT-PRECCINSTA, GT-EM2C, ETH).
2. **Misure di temperatura delle superfici, quando il gruppo lavora con chi fa LES.**
   - Strumento: soprattutto fosfori termografici (rivestimento di 10–20 µm, letto da un laser UV e da un fotomoltiplicatore o una camera veloce). A volte termocoppie tipo K incollate o inserite in parti metalliche.
   - Dove: finestre in quarzo (lato interno ed esterno), piano di dump, punta del bluff body o del centerbody, barre d'angolo.
   - Perché, esplicitamente: **condizioni al contorno termiche per le simulazioni** e perdite di calore. La forma della fiamma (V oppure M) e la sua stabilità dipendono dalla temperatura delle pareti (GT-EM2C, GT-PRECCINSTA, GT-DLR-SFB606, GT-KIT).
3. **Temperatura del gas in fiamma, se il laboratorio ha la diagnostica dedicata.** Tecniche:
   - Raman spontaneo a singolo colpo in un volume di circa 0,6 mm, spostato su 45–70 punti (DLR);
   - CARS a 5 kHz;
   - Rayleigh e Rayleigh filtrato (FRS) planari;
   - two-line OH-PLIF.

   Sono **campagne separate dalla PIV**, sulla stessa condizione operativa. Il DLR scrive esplicitamente: *«it is not possible to perform Raman and PIV measurements simultaneously»* (GT-DLR-SFB606). Al legame fra le due misure restano statistiche, eventualmente ordinate per fase su un segnale di pressione.
4. **Termocoppie nel gas di reazione: rare, e solo dove il gas è relativamente freddo.** Esempio: termocoppie tipo K da 1,5 mm inserite attraverso una barra d'angolo nella zona di ricircolo esterna, con T < 1200 K e nessuna correzione radiativa (GT-EM2C). Altri casi sono fuori dalla zona ottica:
   - un rastrello di 4 termocoppie allo scarico (NH-LSU);
   - una traversata con termocoppia tipo S in un combustore chiuso, con PIV solo a freddo (LF-Delft).

   **Non ho trovato alcun caso con una griglia di termocoppie nella zona di fiamma inquadrata dalla PIV.**
5. **Nei motori ottici:**
   - Temperature di set-point lente e puntuali (TCC-III, Darmstadt):
     - aria nel plenum e nel runner;
     - scarico;
     - refrigerante;
     - olio;
     - a volte la superficie esterna del cilindro in quarzo.
   - La temperatura del gas nel cilindro è quasi sempre **calcolata** dalla pressione con un modello.
   - La temperatura del gas fresco può essere misurata in un piano insieme alla PIV, con LIF di traccianti o con particelle fosforescenti, solo prima della combustione (EN-Darmstadt-LIF, EN-IFPEN).
6. **Le assenze sono frequenti e dichiarate.** Esempi:
   - *«The exact wall temperature was not measured in the experimental study»* (PRECCINSTA, Gövert 2018);
   - *«Wall temperature measurements are not available for the case considered»* (DLR GTMC, Kumar 2023);
   - *«the actual flame temperature or the burned gas was not measured in the present work»* (swirl aperto con PIV, Wang et al.).

   In molti paper PIV l'unica "temperatura di fiamma" è la **temperatura adiabatica calcolata**, non una misura.

### 1.3 La temperatura misurata contemporaneamente alla velocità, nello stesso posto

Esiste, ma in poche configurazioni, ciascuna con un limite forte:

| Via | Dove documentata | Limite principale |
|---|---|---|
| FRS planare + PIV a lunghezze d'onda diverse, sincronizzate | LF-OSU: getti pilotati non premiscelati, 10 Hz | Combustibile "su misura" (CH₄/H₂/Ar), così che il segnale dipenda solo da T. Foglio FRS alto ~8 mm contro 30 mm della PIV. A Lund lo stesso approccio è **fallito** per l'eccessiva semina PIV (LF-Lund) |
| Particelle fosforescenti che fanno da traccianti PIV e da termometro | LF-Yin (fiamma laminare), EN-IFPEN (motore) | Funziona solo sotto ~900–1000 K (BAM:Eu²⁺). Non vede gas combusto né fiamma |
| LIF di tracciante (toluene) + PIV ad alta velocità | EN-Darmstadt-LIF (motore trascinato) | Solo gas fresco o non bruciato. Calibrazione ancorata a una temperatura calcolata dalla pressione |
| "Isoterma gratuita" dalle immagini PIV: le goccioline d'olio evaporano a una temperatura nota | LF-isoterma | Dà un'informazione binaria (sopra/sotto ~500 K), non un valore di temperatura |

### 1.4 Che cosa significa per noi (inferenza, da discutere)

- [I] Lo scenario di Linzheng, *«slower, spatially limited thermal measurements»*, corrisponde bene alla pratica documentata. Ma le misure lente e sparse più realistiche sono:
  - **temperature di superficie o di contorno** (pareti, punta del bluff body, ingresso, scarico);
  - poche **termocoppie in zone di ricircolo relativamente fredde**;
  - **statistiche puntuali o lungo una linea** da Raman/CARS, acquisite in un'altra campagna.

  Non sono "termocoppie nel cuore della fiamma sincronizzate con la PIV".
- [I] La geometria di questi sensori potrebbe non essere rappresentabile nella sezione RealPDEBench: pareti e bordi non sono documentati nella slice (§11).
- **RealPDEBench:**
  - [D] non contiene alcuna temperatura misurata;
  - [D] per la combustione non ha PIV;
  - [M] le condizioni al contorno termiche della LES non sono dichiarate.

  Qualunque "osservazione termica realistica" che costruiremo sarà quindi un modello trasferito da altri apparati, da dichiarare come tale (§11, §13).

---

## 2. Mappa delle temperature: quale, dove, con che cosa, perché

| Temperatura | Dove (nei casi letti) | Strumenti documentati | Scopo dichiarato | Supporto e tempo tipici | Rapporto con la PIV |
|---|---|---|---|---|---|
| **Aria/miscela in ingresso, plenum** | Plenum aria (GT-KIT); runner di aspirazione vicino alla testa (EN-Darmstadt, Sandia HCCI); a monte della camera (NH-LSU) | Termocoppie tipo K [D NH-LSU, Sandia], tipo non dichiarato altrove [M]; spesso nessuno strumento riportato, solo "ambiente" o "293 K" [D GT-EM2C] | Definire e ripetere il punto di funzionamento; condizione al contorno di ingresso delle LES (GT-KIT; EN-Darmstadt: *«measured … average temperatures are imposed at the boundaries»*) | Punto; valore medio per prova o condizione | Stessa prova o condizione; non co-localizzata |
| **Preriscaldamento "involontario" dei reagenti** | Plenum dei combustori atmosferici non preriscaldati | Termocoppie nel plenum (GT-KIT: 450 K esterno, 350 K interno); in PRECCINSTA *«not precisely measured»* (320–380 K) | Spiegare la frequenza termoacustica e le discrepanze LES (GT-KIT: ≈635 Hz adiabatico, ≈725 Hz con scambio termico, ≈750 Hz misurato) | Punto, stazionario | Separata |
| **Superfici solide: finestre in quarzo, piano di dump, bluff body/centerbody, barre, pistone** | Strisce o 17–30 punti su **una** finestra, interpolati e specchiati; punta del bluff body; barre d'angolo; pistone nei motori | Fosfori termografici: tempo di decadimento o rapporto di intensità, calibrazione in forno contro una termocoppia. Termocoppie su parti metalliche (6 tipo K su una barra, GT-EM2C). IR attraverso il quarzo raro (solo AB) | **Condizioni al contorno termiche per LES/CHT**; perdite di calore (finestre ≈10 % della potenza a 1 bar, 15–16 % a 3 bar); forma V/M, PVC; interazione fiamma-parete | Punti, linee o mappe 2D parziali; quasi sempre medie di 150–500 colpi in stato stazionario; eccezione a 1 kHz sul bluff body (GT-PRECCINSTA) | Campagne separate. Simultaneità documentata solo con OH-PLIF, non con PIV (Yin 2017) |
| **Gas nella zona di ricircolo esterna (fredda)** | ORZ di un combustore confinato, attraverso una barra d'angolo | Termocoppie tipo K da 1,5 mm, non corrette per la radiazione (errore dichiarato ≤5 % sotto 1200 K) | Validare le simulazioni con pareti misurate | Punti, medie stazionarie | Stessa condizione, campagna separata [I] |
| **Gas in fiamma (reagente, combusto)** | Griglie di 45–70 punti fra 8 e 80 mm dal bruciatore (DLR); linee di 6 mm (Sandia); piani alti 8 mm (FRS) | Raman spontaneo (densità numerica totale → T con la legge dei gas ideali); CARS (spettro N₂); Rayleigh/FRS (intensità ∝ densità, con ipotesi sulla sezione d'urto); two-line OH-PLIF (solo dove c'è OH) | Struttura di fiamma, mescolamento, perdite di calore, validazione LES, modelli acustici (Helmholtz) | Singolo colpo (5–10 Hz) o serie a 5 kHz (CARS); statistiche e medie in fase | **Separata** nei combustori confinati letti. Simultanea solo FRS+PIV nei getti aperti (LF-OSU) |
| **Scarico** | Rastrello di 4 termocoppie ~30 cm sopra il piano di dump (NH-LSU); tubo di scarico (EN-Darmstadt) | Termocoppie (tipo e correzione non riportati [M]) | Stato di funzionamento; indicatore della fasatura nei motori | Punto, media | Separata. Nei motori con skip-firing è una media mista trascinato/acceso |
| **Gas nel cilindro (motori)** | — | **Calcolata** dalla pressione (T = pV/(mR), compressione adiabatica o politropica); LIF di traccianti calibrato su quella stima | Analisi del rilascio di calore; calibrazione della termometria LIF | Bulk (0D) oppure piano | Simultanea solo per gas fresco (EN-Darmstadt-LIF, EN-IFPEN) |
| **Particelle: soot** | Getti diesel | Pirometria a due colori: emissione integrata sulla linea di vista | Temperatura del soot | Proiezione, polarizzata verso il soot più caldo | Non combinabile in pratica con la PIV [I] |
| **Temperature calcolate, non misurate** | Tabelle delle condizioni dei paper PIV | T adiabatica (equilibrio). Usata anche per il numero di Stokes delle particelle PIV (GT-EM2C: 1881 K) | Caratterizzare la condizione; stime per la PIV | — | — |

---

## 3. A che cosa serve conoscere la temperatura (domanda 1)

Gli scopi documentati nei casi letti, con le fonti:

1. **Caratterizzare e ripetere la condizione operativa.**
   - [D] TCC-III: plenum, runner, scarico e refrigerante riscaldati elettricamente a 45 °C *«to minimize temperature gradients in the air flow»*. Il motore viene trascinato 15–20 min finché la superficie esterna del cilindro raggiunge 40 ± 4 °C (EN-TCC-motored, §1.1, p. 4; il significato di ±4 °C non è definito [M]).
   - [D] EM2C: si acquisisce solo dopo lo stato stazionario segnalato da una termocoppia, circa 45 min (GT-EM2C, §2.5).
   - [D] SJTU, swirl aperto con PIV: si attende *«~6 minutes»* senza monitoraggio riportato (LF-SJTU, §II).
2. **Condizioni al contorno termiche per LES e simulazioni.** È lo scopo più spesso **esplicito** per le temperature di parete.
   - [D] *«LES are then extremely sensitive to the prescription of thermal boundary conditions, which are here carefully measured»* (Mercier 2016, §2.5).
   - [D] *«the temperature distributions of the combustion chamber walls are often missing as important boundary conditions»* (Arndt, Nau, Meier 2021, Introduzione).
   - [D] *«These measurements are in particular valuable parameters for improved boundary conditions of numerical simulations»* (Nau 2019, p. 9).
   - [D] Nei motori: *«measured instantaneous static pressures and average temperatures are imposed at the boundaries»* (Chu 2023, pp. 7–8).
3. **Studio della combustione e della stabilizzazione.**
   - [D] Con pareti adiabatiche la LES prevede sempre una fiamma a M; con le pareti misurate ritrova sia la V sia la M (GT-EM2C, abstract). TNF12, p. 10: la LES adiabatica *«completely mispredicts the thermochemical conditions»*.
   - [D] *«As the formation of PVC is highly sensitive to the local density (temperature) field, knowledge of temperature on flame holders … is important»* (Yin 2017, Introduzione).
   - [D] Mescolamento e dissipazione su piccola scala: gradiente di T da FRS insieme al tasso di deformazione da PIV (LF-OSU, abstract).
4. **Scambio termico e perdite di calore.**
   - [D] Il gradiente di temperatura fra i due lati delle finestre dà il flusso termico attraverso di esse: ≈10 % della potenza a 1 bar (GT-DLR-SFB606, Arndt 2021, Tab. 2) e 15–16 % a 3 bar (GT-DLR-sooting, Nau 2017, abstract).
   - [D] Interazione fiamma-parete: CARS nel gas insieme ai fosfori sulla parete (LF-SWQ).
5. **Termoacustica.**
   - [D] La frequenza del modo instabile si sposta con il preriscaldamento e le perdite (GT-KIT, Kraus 2017, abstract).
   - [D] La condizione al contorno di parete decide se la LES prevede una fiamma stabile o instabile (Agostinelli 2021, abstract).
   - [D] Distribuzione di densità e velocità del suono per un modello di Helmholtz ricavata dal Raman (Arndt 2015, §2.6).
   - [D] Una sola termocoppia di parete come **variabile di stato** di una dinamica isteretica: transitorio di ~30 s fino a ~1250 °C (ETH, Bonciolini 2019, Fig. 1).
6. **Elaborazione e interpretazione della PIV.**
   - [D] Il numero di Stokes delle particelle è calcolato con la T adiabatica (Mercier 2016, §2.3).
   - [D, AB] Termoforesi e indice di rifrazione richiedono un profilo di T o di densità di riferimento (Stella 2001; Vanselow 2019).
   - [D] Le goccioline d'olio che evaporano marcano un'isoterma (LF-isoterma).
   - [D, metodo non letto] Esiste una densità stimata dall'intensità Mie delle particelle PIV (Oberleithner 2015, usata da Stöhr 2018).
7. **Emissioni** (rig ad ammoniaca).
   - [D] Temperature e perdite di calore servono a interpretare NO, N₂O e NH₃ residuo e ad ancorare reti di reattori. Ma spesso la perdita di calore è **regolata** anziché misurata: *«Heat loss is the only free variable in the reactor network simulations»* (NH-LSU, §4.2).
   - [D] Nel rig Cardiff ad alta pressione la *«flame zone temperature»* è elencata fra le condizioni al contorno incerte (NH-Cardiff-HPOC).
8. **Limiti dei materiali.**
   - [D] Ventilatori per *«avoid excessive over heating of the quartz windows»* (Sadanandan 2009, §2).
   - [D] Degrado delle finestre sopra ~1400 K (Nau 2017; Nau 2019).
   - [M] Nei casi letti non ho trovato una misura di temperatura del quarzo motivata esplicitamente dalla sicurezza.

---

## 4. Strumenti: che cosa si misura davvero e come si arriva alla temperatura (domanda 3)

### 4.1 Termocoppie

- **Segnale.** Una forza elettromotrice che dà la temperatura del **giunto** (bead), non del gas.
- **Dal segnale alla temperatura del gas.** Serve un bilancio sul giunto: convezione dal gas, irraggiamento verso pareti e finestre, conduzione lungo i fili.
  - [D] Correzione radiativa: T_g = T_tc + εσ(T_tc⁴ − T_w⁴)·d/(Nu·k) (Shaddix 2017, Eq. 1–3).
  - [D] Shaddix aggiunge che la formula è *«fundamentally incorrect»* quando il bead è più grande dei fili, perché la conduzione lo raffredda.
  - [D] La lettura è *«sensitive to variations in gas temperature over a spatial dimension that is much larger than the thermocouple bead»* (abstract; nell'esempio della Fig. 6, alcuni millimetri lungo i fili). Il supporto spaziale effettivo non è il solo bead.
  - [D] L'errore diventa importante sopra ~800 K (Shaddix 2017, Introduzione).
- **Risposta temporale.**
  - [D, AB] In fiamma turbolenta non esiste una costante di tempo unica, perché varia con T e velocità locali (Chandran 1985; Ballantyne & Moss 1977).
  - [D] Esempi numerici dello scouting della missione 01, apparato NIST:
    - τ ≈ 57 ms per una termocoppia da 50 µm in un pool fire;
    - la correzione inerziale cambia la deviazione standard da 237 K a 385 K.
  - [D] Termocoppie inguainate: τ dell'ordine di 0,5–5 s secondo il diametro (Yilmaz 2008, contesto incendi).
- **Nei casi con accesso ottico** stanno nei plenum, nelle parti metalliche, nella ORZ fredda o allo scarico, mai nella zona di fiamma inquadrata. Le termocoppie servono anche da **riferimento per calibrare i fosfori in forno** (Yin 2017 §3; Nau 2019).
- [D] DLR riassume pro e contro: *«widely used, well understood and can provide temperatures with high accuracy»* ma *«intrusive … only point measurements … the wiring can be difficult»* (Nau 2019, p. 2).

### 4.2 Fosfori termografici su superficie

- **Segnale.** Luminescenza di un rivestimento ceramico (10–20 µm con legante HPC) eccitato da un laser UV o a 532 nm. Si misura il **tempo di decadimento** (più robusto) o un rapporto di intensità (circa 10 volte meno sensibile, Nau 2019).
- **Temperatura.** Si ricava da una curva di calibrazione in forno contro una termocoppia di tipo B o K.
- **Accuratezza e intervalli** [D, nelle condizioni indicate]:
  - 0,5–1 % in laboratorio, fino a 2–3 % su quarzo caldo (Nau 2019; Arndt 2021);
  - ogni fosforo copre circa 300–400 K. Esempi: Mg₄FGeO₆:Mn fino a ~850–1050 K; YAG:Eu 1000–1500 K; YAG:Dy fino a 1800 K.
- **Supporto.**
  - Punti di 4×4 mm² letti da un fotomoltiplicatore, spostati con un motore passo-passo (17 punti, Nau 2019).
  - Strisce lette da una camera veloce (~80 mm a ~0,94 mm, Yin 2017).
  - Mappe con fibra ottica di 55×55 mm² a 1,7 mm/px (Nau 2024).
- **Tempo.** Di solito medie di 150–500 colpi. Risolto a 1 kHz sulla punta del bluff body, dove compare un picco a 475 Hz attribuito al PVC (Nau 2019, p. 8).
- **Limiti documentati:**
  - fluorescenza del quarzo sotto 266 nm;
  - finestre che diventano opache sopra ~1400 K;
  - fondo di radiazione termica intorno a 1300 K per l'emissione rossa;
  - deposito di soot;
  - nessun segnale per spot o fasci troppo grandi rispetto all'oggetto (1,8 mm su una punta da 6 mm, Mercier 2016).
- **Accorgimento d'installazione documentato.** Eccitazione e rivelazione dal lato esterno della finestra, *«therefore laser beam and phosphorescence signal did not have to pass through the flame»* (Nau 2019, p. 5).

### 4.3 Termografia infrarossa

- [D, solo AB] Attraverso liner in quarzo con filtro KG2 (Virginia Tech); vernici ad alta emissività a 800 nm (RSI 2006).
- [D] Limiti dichiarati: *«can suffer from flame emissions and reflected radiation in combustors»* (Nau 2019, p. 2). Il quarzo è *«opaque for infrared»* (Rodrigues 2018).
- [I] Nei casi principali letti non è lo strumento di riferimento.

### 4.4 Raman spontaneo (punto o linea) e Raman/Rayleigh

- **Segnale.** Intensità Raman delle specie principali (O₂, N₂, CH₄, H₂, CO, CO₂, H₂O).
- **Temperatura.** Si calcolano le densità numeriche con una calibrazione e poi *«the temperature was deduced from the total number density via the ideal gas law»* (Arndt 2015, §2.4).
  - Nella variante ibrida Raman/Rayleigh, T viene dal Rayleigh con una sezione d'urto calcolata dalla composizione Raman (LF-TNF, Sandia).
- **Supporto e tempo.**
  - DLR: volume di ~0,6 mm, singolo colpo a 5 Hz; 400–2500 colpi per punto; 45–70 punti spostati in sequenza.
  - Sandia: linea di 6 mm con risoluzione ottica di ~0,05 mm.
- **Incertezze** [D, Arndt 2015 §2.4]: sistematica ±3–4 % su T; statistica a singolo colpo 3–3,5 % (definizioni come nella fonte).
- **Limiti.**
  - Punti con h < 8 mm o r > 27 mm non accessibili per *«clipping of the solid angle of the detection optics»* (Arndt 2015).
  - Richiede un flusso privo di particelle: *«Elastic scattering from particles … can overwhelm the Rayleigh scattering signal»* (Frank 2021, §3).
  - Polvere → breakdown ottico (Lund).

### 4.5 CARS

- **Segnale.** Spettro coerente di N₂ (o rotazionale), confrontato con uno spettro teorico.
- **Supporto e tempo.** fs-CARS a 5 kHz in un volume di ~0,4–0,6 mm, 5000 colpi consecutivi per punto; precisione ±2 % e accuratezza ±3 % (Dennis 2016, GTMC). CARS 1D fino a ~30 µm da una parete (LF-SWQ).
- **Vantaggio.** *«CARS measurements can be performed in particle-laden flows and near surfaces»* (Frank 2021, §3).
- **Limiti.** Finestre che si muovono e diffondono il fascio di pompa: colpi scartati, 89–99 % validi (Dennis 2016, §3.1).

### 4.6 Rayleigh e Rayleigh filtrato (FRS)

- **Segnale.**
  - Rayleigh: intensità ∝ densità × sezione d'urto efficace; con la legge dei gas, T ∝ 1/intensità se la composizione è nota o ipotizzata.
  - FRS: un filtro molecolare o atomico (I₂ o Hg) assorbe la luce alla frequenza del laser, cioè la diffusione Mie e quella dalle pareti. Passa la parte allargata per effetto Doppler.
- **Limiti.**
  - Dipendenza dalla composizione: *«simultaneous 2D species concentration measurements are often not practical»* (Frank 2021, §4.1).
  - Si risolve con combustibili su misura (LF-OSU) o con l'ipotesi che la sezione d'urto dei gas combusti sia simile a quella dei freschi (< 1,3 %, LF-Lund).
  - FRS con modulazione di frequenza: solo medie temporali, *«requires long sampling times»* (DLR, AB).

### 4.7 Two-line OH-PLIF

- **Segnale.** Rapporto di due eccitazioni dell'OH.
- **Supporto.** Piano, ma **solo dove c'è OH**: *«no temperature information can be obtained from regions where OH radicals are missing»* (Giezendanner-Thoben 2005, AB).
- **Tempo.** Uso in fase nel GTMC; accuratezza 4–7 %.

### 4.8 LIF di traccianti (motori)

- **Segnale.** Fluorescenza del toluene in una o due bande; T dal rapporto.
- **Calibrazione.** Contro una temperatura **calcolata** dalla pressione con una compressione politropica o adiabatica. L'accuratezza assoluta è *«limited by the pressure-derived temperatures»* (Dec & Hwang 2009, p. 1).
- **Precisione (1σ)** [D]: 8,8 K al punto morto superiore e 3,3 K a 305° di manovella.
- **Limiti.** Solo gas fresco (trascinato, a volte in N₂); condensazione del tracciante (EN-Darmstadt-LIF, EN-Sandia-HCCI).

### 4.9 Particelle fosforescenti come traccianti PIV (PIV termografica)

- **Segnale.** Le stesse particelle danno diffusione Mie (velocità) e luminescenza (T, con il rapporto fra due colori).
- **Limite in fiamma.** *«The upper measurable temperature is found to be 900–1000 K»*, per spegnimento termico della luminescenza (Yin et al. 2017, abstract).
- **In motore.** ±1σ = 3,7 K a 317 K e 24,4 K a 575 K, con perdita di segnale vicino al punto morto superiore (Kopf 2023, AB).

### 4.10 Marcatori binari dalla PIV stessa

- [D] Olio d'oliva o DEHS evaporano al passaggio del fronte. Il contorno *«physically correspond[s] to the instantaneous location of the isotherm 500K»* (CORIA, arXiv 1004.0074, §3, olio d'oliva).
- [D] La PIV condizionata usa invece il salto di densità delle particelle (Pfadler 2007/2008, AB).
- **Informazione prodotta.** Variabile di progresso binaria, non T.

### 4.11 Pirometria del soot e assorbimento laser (TDLAS)

- **Pirometria a due colori.** Temperatura del soot integrata sulla linea di vista, *«bias toward high temperature soot»* (Reggeti 2022, p. 3).
- **TDLAS.** Media lungo il percorso, citata nello scouting della missione 01.
- [M] In questo scouting non ho trovato TDLAS insieme a PIV in un combustore swirl.

---

## 5. Dove si misura e perché lì (domanda 4)

### 5.1 Posizioni documentate

**Combustori confinati:**

- **Plenum e ingresso.** Termocoppie nei plenum dell'aria (GT-KIT, Kraus 2017 §2: *«The air temperatures in the plenums are monitored with thermocouples»*). Tipo K a monte della camera (NH-LSU, §2.2).
- **Barre o montanti d'angolo che reggono le finestre.**
  - 6 termocoppie tipo K lungo una barra d'acciaio, 593 K a z = 16 mm e 637 K a z = 56 mm (GT-EM2C, Fig. 3).
  - Una termocoppia in una barra per monitorare lo stato stazionario (GT-EM2C).
  - Un montante in Inconel con fosforo (GT-DLR-SFB606, Arndt 2021).
- **Piano di dump e punta del bluff body o del centerbody.**
  - Fosfori, profilo radiale: 435 K a r = 7 mm, 516 K a 25 mm. Punta dell'asta 500 ± 50 K (GT-EM2C, Fig. 4; ± come nella fonte).
  - Striscia di 4 mm sulla piastra di base e punta del bluff body (GT-PRECCINSTA, Yin 2017 §2.3).
- **Finestre in quarzo.**
  - 30 punti su una finestra (GT-EM2C, Fig. 5).
  - Tre strisce verticali da 5 mm a x = 0; 16,5; 33 mm su una finestra, lato interno ed esterno (GT-DLR-SFB606, Arndt 2021).
  - 17 spot di 4×4 mm² su segmenti di quarzo (GT-DLR-HBK, Nau 2019).
- **Gas nella ORZ.** Termocoppie tipo K da 1,5 mm *«through one of the four vertical steel bar holding the quartz windows»* (GT-EM2C, §2.5; mappa delle posizioni in Fig. 6).
- **Gas in fiamma (ottico).** Griglia Raman a h = 8–80 mm, r = −3…27 mm (GT-DLR-SFB606). Profili radiali a più altezze (Meier 2010).
- **Scarico.** Rastrello di 4 termocoppie a ~30 cm dal piano di dump, a raggi diversi (NH-LSU, §2.2). Sonda in allumina nello scarico, a x = 680 mm e 33 mm dalla parete (GT-DLR-HBK).

**Motori ottici:**

- Sonde di T e p distribuite sul banco. In Welch 2023, Fig. 1a:
  - T_in,1 a valle del plenum;
  - T_in,2 sul runner vicino alla testa;
  - T_ex,2 sul tubo di scarico accanto alla sonda lambda.
- Termocoppia nell'anello distanziale con il giunto ~2 mm sotto la superficie, 5 mm sotto il firedeck; il valore è poi estrapolato alla superficie (Sandia HCCI, p. 5).

### 5.2 Motivazioni del posizionamento: dichiarate e inferite

**Dichiarate dagli autori [D]:**

- Pareti e superfici: sono le condizioni al contorno che mancano alle LES (§3).
- Bluff body: legame con il PVC e con la forma V/M.
- Scarico (motori): indicatore della fasatura e dello stato termico.
- Plenum: preriscaldamento che sposta le frequenze termoacustiche.
- Eccitazione dei fosfori dal retro della finestra: evitare il percorso nella fiamma.
- Zone escluse:
  - Raman non accessibile vicino al bruciatore e alle pareti (angolo solido di raccolta);
  - PIV non eseguita dentro l'ugello *«due to difficulties from laser scattering inside the nozzle»* (Nau 2019, p. 8).

**Inferite [I], cioè non dichiarate:**

- Le termocoppie stanno in plenum, barre metalliche e scarico perché lì non bloccano l'accesso ottico, non perturbano la fiamma osservata e sopravvivono.
- Nella ORZ si possono usare perché il gas è sotto 1200 K e l'errore radiativo è piccolo (questa parte è dichiarata in GT-EM2C).
- **Nessun autore letto giustifica esplicitamente la scelta "periferica" con l'accesso ottico.**
- [I] La copertura delle pareti è parziale e simmetrizzata: una finestra misurata, poi interpolazione e specchiatura (Arndt 2021, Fig. 1 e 6; Mercier 2016). Il resto delle pareti è stimato. In GT-KIT, in parte, *«based on the annealing colors»* delle parti metalliche (Kraus 2017, §4).

---

## 6. Quale informazione produce ogni misura (domanda 5)

| Misura (casi) | Supporto spaziale | Istantaneo / filtrato / mediato / in fase | Acquisizione vs risposta vs esposizione vs durata |
|---|---|---|---|
| Termocoppia di servizio (plenum, scarico, refrigerante) — EN-TCC, EN-Darmstadt, NH-LSU | Punto | Valore medio per prova o condizione. In TCC-III la deviazione standard "in test" è 0 (Tab. A2), cioè un valore registrato per prova [I] | Frequenza [M]; risposta lenta [I]; durata = prova |
| Termocoppia nella ORZ — GT-EM2C | Punto (1,5 mm) più conduzione lungo i fili [I, per analogia con Shaddix 2017] | Media stazionaria | [M] |
| Termocoppia traversata nel gas (combustore chiuso) — LF-Delft | Punto, con traversata radiale a 0,62 mm/s | Media su un diametro di sonda (71 campioni) | Registrazione a 10 Hz; sonda inguainata da 3 mm, quindi risposta lenta [I] |
| Termocoppia di parete come variabile di stato — ETH | Punto | Transitorio risolto (decine di secondi) | [M] frequenza |
| Fosfori su superficie, a punti o strisce — GT-EM2C, GT-DLR, GT-HBK | Punti, linee, mappe parziali | Medie di 150–500 colpi | Laser a 10–15 Hz; decadimento registrato fino a 775 kHz; stato stazionario |
| Fosfori ad alta velocità — GT-PRECCINSTA | Linea di ~80 mm, ~0,94 mm | **Singolo colpo a 1 kHz**, 8,1 s | Rivestimento sottile: segue fluttuazioni a 475 Hz con ~7 K di ampiezza |
| Raman puntuale — GT-DLR | Volume di 0,6 mm, punti in sequenza | Singolo colpo; statistiche e medie in fase (8 classi da 45°) | 5 Hz; impulso di ~3 µs; 400–2500 colpi per punto |
| Raman/Rayleigh lungo una linea — LF-TNF | Linea di 6 mm | Singolo colpo | Gate Rayleigh 300 µs, Raman 3,9 µs |
| fs-CARS — GT-DLR (Dennis 2016) | Punto (~0,4–0,6 mm) | **Serie temporale a 5 kHz** | 5000 colpi per punto |
| FRS + PIV — LF-OSU | Piano 8 mm × larghezza [M], pixel di 128 µm | Singolo colpo simultaneo | 10 Hz; campioni indipendenti |
| PIV termografica — LF-Yin, EN-IFPEN | Piano, stesse particelle | Singolo colpo simultaneo | 5 Hz (fiamma laminare); risposta delle particelle ~10–100 µs [SEC] |
| LIF di toluene + PIV — EN-Darmstadt-LIF | Piano | Ad alta velocità (kHz), simultaneo | Solo gas fresco [AB] |
| T bulk nel cilindro | 0D | Risolta in angolo di manovella | **Modello**, non misura |
| Isoterma dalle goccioline d'olio | Contorno nel piano PIV | Istantaneo, con la PIV | Binaria |

**Una sintesi da non generalizzare oltre i casi [I].** Le misure dirette che risolvono la turbolenza (Raman, CARS, FRS) sono:

- puntuali, lineari o planari su piccoli campi;
- a 5–10 Hz, oppure 5 kHz solo per il CARS puntuale;
- in campagne dedicate.

Le misure sparse "di servizio" e di parete sono **medie stazionarie**. Fra i casi letti non ho trovato serie temporali di termocoppie nel gas registrate in sincronia con la PIV.

---

## 7. Rapporto con la misura di velocità (domanda 6)

Classifico il legame in cinque livelli. Uso "stesso istante" solo per misure sincronizzate.

| Livello | Definizione | Casi |
|---|---|---|
| **L1 — simultaneo e co-localizzato** | Stesso colpo, stesso piano | FRS + PIV (LF-OSU: *«The PIV laser pulses are timed to frame straddle the FRS laser pulse»*, §2); PIV termografica (LF-Yin, EN-IFPEN, < ~1000 K); LIF di toluene + PIV (EN-Darmstadt-LIF, gas fresco); isoterma binaria dalle goccioline (LF-isoterma) |
| **L2 — simultaneo, non co-localizzato o non con la PIV** | Stesso istante ma altra grandezza o regione | Fosfori sul bluff body simultanei con OH-PLIF, non con la PIV (Yin 2017). Termocoppie di servizio registrate durante le prove PIV [I] |
| **L3 — stesso apparato e condizione, campagne separate** | Legame solo statistico o in fase | Raman ↔ PIV (Arndt 2015: *«not possible … simultaneously»*); fs-CARS ↔ PIV (Dennis 2016); FRS/OH-PLIF ↔ PIV/OH-PLIF (LF-Lund: *«Simultaneous PIV and FRS were not successful … due to the high seeding densities required for PIV»*); pareti con fosfori ↔ PIV (GT-EM2C, GT-DLR, GT-HBK: le misure di parete *«complement existing measurements from PIV»*) |
| **L4 — stesso apparato, condizioni diverse** | Velocità e T a φ, potenza o regime diversi | Gövert 2018: LDV a φ = 0,75, Raman a φ = 0,83, altezze diverse. Kumar 2023: PIV solo a freddo (TP07), T in fiamma (TP09). Agostinelli 2021: pareti misurate a 20 kW, LES a 10 kW. LF-Delft: PIV a freddo, T in fiamma |
| **L5 — laboratori o geometrie diverse** | Scalari a Sandia, velocità nel laboratorio d'origine | Fiamme TNF Sandia D–F, CSSB, TSF (LF-TNF). Wang et al.: T inferita da dati di altri bruciatori (LF-SJTU) |

[D] Il workshop TNF13 chiede esplicitamente *«velocity measurements … at flow conditions matching those of the scalar experiments»* (TNF13, pp. 11–12). La separazione è una lacuna riconosciuta, non solo una scelta.

**PIV reagente o a freddo.**

- La PIV è in flusso reagente: GT-DLR, GT-PRECCINSTA, GT-EM2C, GT-KIT, LF-OSU, LF-Lund, LF-SJTU.
- È solo a freddo:
  - nel caso di validazione di Kumar 2023;
  - in LF-Delft;
  - nel motore TCC-III benchmark (trascinato).

**Cause documentate della non-simultaneità:**

- la diffusione Mie delle particelle copre Rayleigh e Raman (Frank 2021; Lund);
- polvere e particelle provocano breakdown con gli impulsi Raman (Lund; pulse stretching in LF-OSU);
- [I] laser e ripetizioni incompatibili e accesso ottico condiviso. Il motivo per Arndt 2015 non è dichiarato nella frase citata.

---

## 8. Quando la temperatura non si misura (domanda 7)

### 8.1 Assenze dichiarate esplicitamente

| Caso | Che cosa manca | Testo | Conseguenza o compenso |
|---|---|---|---|
| PRECCINSTA, Gövert et al. 2018 | Temperatura delle pareti | *«The exact wall temperature was not measured in the experimental study and is therefore unknown»* (§3.2, p. 516) | 1000 K imposti sul quarzo |
| DLR GTMC TP09, Kumar et al. 2023 | Pareti, per quel caso | *«Wall temperature measurements are not available for the case considered»* (p. 1000); bilancio delle perdite *«beyond the scope»* (§4.1) | Pareti adiabatiche, T sovrastimata a valle |
| Swirl aperto PIV + CH*, Wang et al. (SJTU) | Gas combusto e fiamma | *«the actual flame temperature or the burned gas was not measured in the present work»* (§IV.A) | T inferita da un modello calibrato su altri bruciatori; *«the accuracy of the temperature model … was not measured»* |
| PRECCINSTA, preriscaldamento | Ingresso | *«not precisely measured»* (Agostinelli 2021, Intro, citando Meier 2007) | 320 K imposti in LES |
| TCC-III acceso, Henrion 2022 | Temperatura del pistone e della finestra | Misura del pistone fra le *«challenges left unaddressed during this study»*. La trasmissione della finestra a caldo: *«This additional measurement was not performed»* (§2.2, pp. 8–9) | Trasmissione a temperatura ambiente |
| Darmstadt, PTV vicino al pistone (Ding 2019) | Pareti | *«Further measurements will include wall temperature measurements»* (conclusioni, p. 15) | — |
| Riportate da Mercier 2016 (SEC) | Pareti (Proch & Kempf; Tay-Wo-Chong; Nogenmyr) | *«were not measured»*, *«were not available»*, *«not realized on the same burner geometry»* | 1000 K imposti; confronti qualitativi |
| FRS+PIV in fiamme fuligginose (Sutton, rapporto DOE) | T del gas a valle | Soot troppo per l'FRS | Solo PIV + LII |

**Motivi generali dichiarati** (Kraus 2018 §1; Agostinelli 2021 Intro):

- le temperature di parete sono *«very difficult to determine»*, *«change for each operating condition»*, sono *«measured only at a few points»*;
- *«the combustion community does not consider the problem of heat conduction through walls as an interesting one»*.

### 8.2 Non riportate (nessuna conclusione sull'esistenza)

- RealPDEBench, combustione: nessuna temperatura sperimentale (§11).
- LSU, swirl non confinato con PIV (De, Zhu, Acharya): nessuna T.
- CECOST (Lund): LES con pareti in quarzo a temperatura ambiente, nessuna misura riportata.
- GE/Toronto a 1 MPa: preriscaldamento *«approximately 500 K»*, nessuno strumento.
- DLR GTMC classico: termocoppie, scarico e pareti non riportati nei paper PIV letti; Weigand 2006 non accessibile.
- SJTU PoF 2024 (candidato rig RealPDEBench): solo abstract.
- KAUST e Tohoku NH₃: solo abstract.

### 8.3 Il caso intermedio più frequente

La temperatura **misurata in un altro momento o in un'altra condizione** e poi riusata:

- pareti di Yin a 20 kW usate per LES a 10 kW o 25–30 kW (Agostinelli 2021; Pries 2026);
- pilot lip della TSF stimato a ~750 K nel 2014 e misurato a ~840 K solo dopo, con fosfori (TNF12 p. 3; TNF13 pp. 22–23).

[D] Le misure termiche spesso arrivano **anni dopo** il database di velocità, quando la LES non torna (TNF13).

[I] **Non deduco una pratica universale.** I casi sono sbilanciati verso rig con compagni LES (DLR, EM2C, TNF), dove le misure termiche sono più frequenti della media.

---

## 9. Vincoli che limitano le misure (domanda 8)

Solo quanto serve a capire la pratica. La caratterizzazione degli errori spetta allo step 2.

- **Intrusività.**
  - Una sonda perturba il flusso: in LF-Delft i valori sono mediati su un diametro di sonda.
  - [D] I campionamenti con sonda perturbano fino a 500 K (scouting missione 01, MBMS).
  - [I] In fiamma, la sonda blocca anche la vista e i fasci.
- **Accesso ottico.**
  - L'angolo solido di raccolta esclude h < 8 mm e r > 27 mm per il Raman.
  - Riflessioni nell'ugello impediscono la PIV a y < 0.
  - Montanti e barre tagliano la vista.
  - Le finestre si sporcano di soot, si muovono, fluorescono, diventano opache.
- **Materiali.**
  - Quarzo a 1000–1400 K nei rig misurati; degrado sopra ~1400 K.
  - Raffreddamento ad aria o con montanti raffreddati ad acqua. Il raffreddamento a sua volta altera la temperatura di parete (Nau 2024: calo dovuto al telaio raffreddato).
  - Limiti di servizio del quarzo fuso (~1100 °C) solo da fonti commerciali [SEC].
- **Risposta temporale.**
  - Termocoppie: decine di ms (fili fini) fino a secondi (inguainate), con τ variabile in fiamma.
  - Fosfori: µs–ms, con decadimenti che si accorciano a T alta.
  - Pareti: equilibrio in decine di secondi o minuti.
- **Risoluzione e supporto.**
  - Termocoppia: bead più conduzione lungo alcuni millimetri di filo (Shaddix 2017).
  - Raman: 0,6 mm.
  - FRS: foglio di 8 mm.
  - Fosfori: spot di 1,8 mm, poco adatti a parti piccole.
- **Calibrazione.**
  - Fosfori in forno con estrapolazione sopra il massimo del forno (Arndt 2021: forno fino a 1260 K, estrapolazione lineare).
  - LIF ancorato a una T calcolata.
  - Rayleigh dipendente dalla composizione.
- **Irraggiamento e conduzione.** Correzione delle termocoppie che dipende da Nu, emissività, velocità e **temperatura delle pareti**. [I] In un combustore con finestre la temperatura delle pareti è a sua volta poco nota.
- **Composizione.** Sezione d'urto Rayleigh; OH assente per la two-line PLIF.
- **Particelle.** Semina PIV incompatibile con Rayleigh e Raman; fosfori spenti oltre ~1000 K.
- **Operatività.**
  - Skip-firing nei motori: lo scarico è una media mista, 101,8 °C contro 386,0 °C in combustione continua (Chu 2023, Tab. 1).
  - Registrazioni brevi delle camere ad alta velocità.
  - Stato termico dipendente dalla storia (attese di 6–45 min).

---

## 10. Tabella dei casi decisivi

Le schede complete sono negli appunti dei filoni. Sintetizzo qui i campi richiesti. Le posizioni nelle fonti seguono il formato "§/p./Fig./Tab." della fonte indicata. Per le fonti lette tramite manoscritto o preprint (elib, HAL, OSTI, arXiv, Zenodo) la pagina è quella del manoscritto.

### 10a. Apparato, PIV, temperatura, strumento, posizione

| ID | Apparato, geometria, condizioni | PIV | Temperatura cercata e scopo | Strumento → grandezza misurata | Posizione e motivazione |
|---|---|---|---|---|---|
| **GT-DLR-SFB606** | DLR, doppio swirl, CH₄ parzialmente premiscelato, camera 89×89×112 mm³ in silice fusa, 1 bar. Fiamma B: 25 kW, φ = 0,7, oscillazione a 392 Hz; aria ambiente | Stereo-PIV a 5 Hz, 1200 colpi, reagente | (a) Gas in fiamma: struttura, perdite, modello acustico. (b) Pareti: BC per CFD, perdite attraverso le finestre | (a) Raman a singolo colpo → densità numeriche → T con la legge dei gas. (b) Fosfori (YAP:Eu, YAG:Eu), decadimento → T da calibrazione in forno | (a) 70 punti h = 8–80 mm; zone escluse per l'ottica [D]. (b) Tre strisce su una finestra (interno ed esterno) e un montante; *«often missing as important boundary conditions»* [D] |
| **GT-DLR-GTMC** | DLR "GTMC" classico, doppio swirl, 85×85×114 mm³, aria a temperatura ambiente; per esempio 10 kW, φ = 0,75, ~300 Hz | PIV/OH-PLIF a 5–10 kHz (Boxx, Stöhr) | Gas in fiamma; finestre ~1000 K (stima) | Raman; fs-CARS a 5 kHz (Dennis 2016) → spettro N₂ → T | Griglie di punti. Ventilatori sulle finestre per il surriscaldamento [D] |
| **GT-PRECCINSTA** | DLR/Turbomeca, swirl radiale + bluff body conico, 85×85×110–114 mm³, premiscelato CH₄. 20 kW φ = 0,7 bistabile; 25–30 kW | Stereo-PIV + OH-PLIF a 10 kHz (Stöhr 2018), campagna separata | Superficie del bluff body e della piastra di base: PVC, forma V/M, LES. Preriscaldamento 320–380 K *«not precisely measured»* | Fosfori (Mg₄FGeO₆:Mn) eccitati dal laser OH-PLIF, decadimento letto a 70 kHz | Punta del bluff body e striscia sulla piastra: legame con il PVC [D] |
| **GT-DLR-sooting** | DLR, 3 bar, C₂H₄, 68×68×120 mm³, finestre raffreddate ad aria | PIV in campagne precedenti | Lato interno ed esterno del quarzo → flusso termico (15–16 % della potenza) | Fosfori YAG:Eu/Dy, letti da un fotomoltiplicatore, eccitati dal retro | Striscia di 5 mm centrata. Le simulazioni prima *«had to rely on estimated wall temperatures»* [D] |
| **GT-DLR-HBK** | DLR HBK-S, FLOX a getti (non swirl), 8 bar, preriscaldamento 725 K | PIV + OH-LIF (Severin), separate | Pareti in quarzo 1050–1400 K; sonda allo scarico fino a 1700 K | Fosfori, 17 spot, fotomoltiplicatore | Spot su segmenti di quarzo; BC per le simulazioni [D] |
| **GT-EM2C** | EM2C, swirl confinato CH₄/H₂, 4 kW, φ = 0,7, X_H₂ = 0,6 o 0,9; finestre 250×92×12 mm; reagenti a 293 K | PIV planare a 10 Hz, reagente (ZrO₂) e a freddo (olio) | Pareti, piano di dump, punta dell'asta, barre, gas nella ORZ: BC della LES e validazione | Tipo K su una barra (6); fosfori (Mg₃.₅FGeO₅:Mn) con fotomoltiplicatore; tipo K da 1,5 mm nel gas | Barre, dump, punta, 30 punti su una finestra, ORZ attraverso una barra. Gas < 1200 K, quindi nessuna correzione radiativa [D] |
| **GT-KIT** | KIT/DLR, doppio swirl concentrico, 30 kW, φ = 0,85, senza preriscaldamento | Stereo-PIV a 5 Hz (DLR) | Plenum (preriscaldamento), pareti: frequenza termoacustica, BC | Termocoppie (plenum, pareti); colori di ricottura per il resto | Plenum aria (450/350 K) [D]. Altre posizioni [M] |
| **GT-ETH** | ETH, swirl premiscelato a gas naturale, pareti in quarzo e ceramica | Nessuna | Parete ceramica come variabile di stato dell'isteresi termoacustica | Una termocoppia tipo K | A valle della fiamma (Fig. 1) |
| **NH-SJTU** | SJTU (laboratorio Lu), swirl premiscelato NH₃/CH₄; candidato rig RealPDEBench [I] | OH* + PIV ad alta velocità **simultanee** (abstract) | *«temperature of combustion chamber»* legata alle emissioni (abstract) | [M] | [M] (solo abstract) |
| **NH-LSU** | LSU, swirl con micro-iniettori, NH₃/CH₄, 12–22 kW, preriscaldamento 20/50/100 °C, quarzo L = 22,8 cm | Nessuna | Ingresso; scarico | Tipo K all'ingresso; rastrello di 4 termocoppie (tipo [M]) | A monte della camera; ~30 cm sopra il dump, a raggi diversi. Valori dello scarico mai riportati [D] |
| **NH-Cardiff-HPOC** | Cardiff, alta pressione ottico, NH₃ crackizzata al 99 %, 1,1–6 bar, preriscaldamento ≈500 K | Nessuna; OH*/NH*/NH₂* a 10 Hz | Solo preriscaldamento e campione delle emissioni (433 K) | [M] | — |
| **LF-OSU** | Getti pilotati non premiscelati CH₄/H₂/Ar, Re 10–30k, aperti | Stereo-PIV a 568 nm, SiO₂, 10 Hz | Gas: dissipazione termica in funzione del tasso di deformazione | FRS (cella I₂), 532 nm; segnale ∝ T per il combustibile scelto | x/d = 20; foglio FRS di 8 mm |
| **LF-Lund** | Bruciatore low-swirl, CH₄ φ = 0,62, 27–40 kW, aperto | 2C e stereo-PIV a 8 Hz + OH-PLIF | Gas: database per la validazione LES | FRS (cella Hg, 254 nm) + OH-PLIF | Piano; FRS+PIV **fallito** [D] |
| **LF-Yin-TPIV** | Bruciatore di Gülder, fiamme laminari H₂/N₂/CH₄ | PIV con le stesse particelle BAM:Eu²⁺ | Gas lato combustibile; limite del metodo | Rapporto a due colori della luminescenza | Solo combustibile seminato; limite 900–1000 K [D] |
| **LF-SWQ** | Darmstadt, side-wall quenching, V-flame contro una parete raffreddata ad acqua | PIV/OH-LIF in campagne separate (Jainski, AB) | Gas vicino alla parete; parete → flusso termico, distanza di quenching | CARS 1D; fosfori sulla parete (Kosaka, SEC) | Fino a ~30 µm dalla parete [D] |
| **LF-SJTU** | SJTU, swirl aperto CH₄, Re 2000–7000 | PIV a 10 kHz + CH* | T del gas combusto per un criterio di Reynolds | **Nessuno**: T inferita da modello | — (assenza esplicita) |
| **LF-Delft** | TU Delft, jet-in-hot-coflow **chiuso**, 8 finestre | Stereo-PIV **solo a freddo** | Profili medi di T del gas | Tipo S inguainata da 3 mm, con correzione radiativa | 10 porte a 30 mm di passo su una piastra al posto di una finestra |
| **LF-TNF** | Sandia D–F, CSSB, TSF | LDV/PIV in altri laboratori | Scalari per la validazione | Raman/Rayleigh/CO-LIF lungo una linea; fosfori sul bluff body (CSSB, AB) | Laboratori diversi (L5) |
| **EN-TCC-motored** | GM/UMich TCC-III, cilindro in quarzo, trascinato, 800/1300 rpm | PIV ad alta velocità in 4 piani, olio | Set-point: aria al port, olio, refrigerante, esterno del cilindro | [M] tipo; *«available with the posted data»* | Port, uscite di olio e refrigerante, esterno del liner a metà corsa |
| **EN-TCC-fired** | TCC-III acceso, propano, finestra del pistone in zaffiro | Nessuna in quel paper | Pistone e finestra (trasmissione IR) | **Non misurata** | — |
| **EN-Darmstadt** | Darmstadt SG-DISI ottico, isoottano, 1500 rpm, skip-fire 1/7 | PIV ad alta velocità nel piano di tumble | Ingresso e scarico; BC della LES | Sonde di T/p (tipo [M]) | Fig. 1a di Welch 2023 |
| **EN-Darmstadt-LIF** | Darmstadt, trascinato | PIV ad alta velocità **simultanea** | Stratificazione di T del gas fresco | LIF di toluene a due colori, calibrato sulla compressione politropica | Piano (AB) |
| **EN-IFPEN** | Motore ottico IFPEN | PIV termografica (ZnO:Zn) | Gas fresco in compressione | Rapporto di intensità delle particelle | Piano (AB) |
| **EN-Sandia-HCCI** | Sandia HCCI ottico, in N₂ (trascinato) | Nessuna | Stratificazione termica | PLIF di toluene; termocoppie nei runner e nell'anello distanziale | Runner vicino alla testa; 2 mm sotto la superficie dell'anello |
| **RP** | RealPDEBench, swirl NH₃/CH₄/aria, 30 condizioni, 1 bar | **Nessuna per la combustione** | Nessuna T sperimentale | Solo chemiluminescenza OH* | — |

### 10b. Supporto e tempo, rapporto con la velocità, limiti, fonte, mancante

| ID | Supporto e tempo | Rapporto con la velocità | Limiti principali | Fonte (DOI/URL) e posizione | Informazioni mancanti |
|---|---|---|---|---|---|
| GT-DLR-SFB606 | (a) Punto 0,6 mm, 5 Hz, 400–2500 colpi, in fase. (b) Strisce, ~200 colpi, stazionario | L3: *«not possible to perform Raman and PIV measurements simultaneously»* | Zone escluse per l'ottica; finestre sporche a z = 30 mm; estrapolazione della calibrazione oltre 1260 K | Arndt 2015, [10.1007/s00348-015-1929-3](https://doi.org/10.1007/s00348-015-1929-3), manoscritto [elib 96436](https://elib.dlr.de/96436/) §2.3–2.4, §3. Arndt 2021, [10.1016/j.proci.2020.06.088](https://doi.org/10.1016/j.proci.2020.06.088), elib 141835, Intro, Tab. 2, Fig. 1/6 — TI | Ingresso, scarico, termocoppie [M] |
| GT-DLR-GTMC | Punti; CARS a 5 kHz × 5000 colpi | L3: PIV dalle campagne precedenti | Finestre che si muovono (scattering del fascio di pompa); T post-fiamma 30–100 K sotto l'adiabatica per le perdite | Dennis 2016, [10.1016/j.combustflame.2016.02.033](https://doi.org/10.1016/j.combustflame.2016.02.033), [OSTI 1533658](https://www.osti.gov/servlets/purl/1533658) §2–3 — TI parziale. Meier 2010, Exp. Fluids 49, elib 65267 §2 — TI. Sadanandan 2009, elib 60286 §2 | Weigand 2006 non letto; pareti [M] |
| GT-PRECCINSTA | Linea di ~80 mm a 0,94 mm, singolo colpo a 1 kHz, 8,1 s | L2 con OH-PLIF; L3/L4 con la PIV | Sensibilità bassa sotto 450 K; PIV assente nell'ugello | Yin 2017, [10.1016/j.expthermflusci.2016.11.004](https://doi.org/10.1016/j.expthermflusci.2016.11.004), [elib 112323](https://elib.dlr.de/112323/) §1–4 — TI. Nau 2019, [10.1115/1.4040716](https://doi.org/10.1115/1.4040716), elib 123633 pp. 4–8 — TI. Agostinelli 2021, [10.1016/j.combustflame.2021.111728](https://doi.org/10.1016/j.combustflame.2021.111728), [Zenodo 6450237](https://zenodo.org/records/6450237) Intro, §2–3.1 — TI | Strumento e posizione del preriscaldamento (Meier 2007 non letto) |
| GT-DLR-sooting | Punti traversati, 150 colpi a 15 Hz | L3 | Fondo del soot; degrado delle finestre; fluorescenza del quarzo | Nau 2017, [10.1007/s00340-017-6856-y](https://doi.org/10.1007/s00340-017-6856-y), [elib 115444](https://elib.dlr.de/115444/) §1–3 — TI | PIV e CARS di Geigle non letti |
| GT-DLR-HBK | 17 spot, interpolati | L3 (*«complement existing measurements from PIV»*) | Degrado > 1400 K; fondo radiativo | Nau 2019 (sopra), pp. 5–10 — TI | Termocoppie [M] |
| GT-EM2C | Punti; medie stazionarie (> 500 colpi sulla punta) | L3 [I: nessuna sincronizzazione dichiarata] | Spot troppo grande per la punta; copertura parziale delle pareti | Mercier 2016, [10.1016/j.combustflame.2016.05.006](https://doi.org/10.1016/j.combustflame.2016.05.006), [HAL hal-01405801](https://hal.science/hal-01405801) §2.3, §2.5, Fig. 2–6, §3.2 — TI. Koren 2018, [10.1007/s10494-018-9895-5](https://doi.org/10.1007/s10494-018-9895-5), hal-01744260 §3 — TI | Guiberti 2015 (originale) solo SEC; esterno delle finestre e scarico (Koren li chiede) |
| GT-KIT | Punti, stazionari | L3 | Pareti *«mostly based on estimated values»* | Kraus 2017, [10.1115/1.4035143](https://doi.org/10.1115/1.4035143), HAL hal-01688648 §2, §4, §5.1 — TI. Kraus 2018, [10.1016/j.combustflame.2018.01.007](https://doi.org/10.1016/j.combustflame.2018.01.007), hal-01730351 §1, §4 — TI | Numero e posizione delle termocoppie [M] |
| GT-ETH | Punto, transitorio di ~30 s | — | — | Bonciolini 2019, [arXiv 1807.02428](https://arxiv.org/abs/1807.02428) Fig. 1 — TI | Frequenza [M] |
| NH-SJTU | [M] | L1 per OH*–PIV; T [M] | — | Liu 2024, [10.1063/5.0196764](https://doi.org/10.1063/5.0196764) — **AB** | Tutto il setup |
| NH-LSU | Punti, media [I] | — | Valori allo scarico non riportati; perdita di calore regolata nel modello | Viswamithra 2022, [arXiv 2206.12015](https://arxiv.org/abs/2206.12015) §2.2, §4.2 — TI | Tipo, diametro, correzione |
| NH-Cardiff-HPOC | — | — | La *«flame zone temperature»* è una BC incerta del modello | PMC12926937 (Energy & Fuels), Materials and Methods, CRN — TI | Strumento del preriscaldamento |
| LF-OSU | Piano 8 mm, 10 Hz, singolo colpo | **L1** | Combustibile su misura; cella I₂; pulse stretching contro il breakdown | McManus & Sutton 2022, [10.1016/j.proci.2022.07.052](https://doi.org/10.1016/j.proci.2022.07.052), [OSTI 2420963](https://www.osti.gov/servlets/purl/2420963) §2 p. 3 — TI | Larghezza del campo, incertezza di T |
| LF-Lund | Piano, 10 Hz, statistiche | L3 | Semina PIV troppo densa per il filtro | Petersson et al. 2007, [10.1364/AO.46.003928](https://doi.org/10.1364/AO.46.003928) p. 3931, letto nella [tesi Lund 2014](https://lup.lub.lu.se/search/files/3031390/4255515.pdf) §6.3 — TI | Campo di vista dell'FRS |
| LF-Yin-TPIV | Piano, 5 Hz, risoluzione 600 µm | **L1** | Limite 900–1000 K; coflow non seminato | Yin et al. 2017 (CNF, comunicazione breve), [elib 113064](https://elib.dlr.de/113064/1/2017_Yin_CnFComm_author_final.pdf) §2–5 — TI | — |
| LF-SWQ | Linea, 500 colpi | L3 | Parete non misurata nel paper CARS | Bohlin 2017, PCI 36, [OSTI 1251595](https://www.osti.gov/servlets/purl/1251595) §2–3 — TI. Jainski 2017 — AB | Kosaka 2018 — SEC |
| LF-SJTU | — | — | T solo da modello | Wang et al., [arXiv 2210.04181](https://arxiv.org/abs/2210.04181) §II p. 7, §IV.A pp. 22–24 — TI | — |
| LF-Delft | Media su 71 campioni, 10 Hz | L4 | Sonda perturbante; velocità diverse fra le due campagne | van den Berg 2024, [10.1016/j.jaecs.2024.100298](https://doi.org/10.1016/j.jaecs.2024.100298) §2 — TI | — |
| LF-TNF | Linea a singolo colpo | L5 | — | Schneider 2019, OSTI 1497651 §2; TNF12 (OSTI 1242775) pp. 3–4, 10; TNF13 (OSTI 1505369) pp. 11–12, 19, 22–23 — TI/PZ | Dati non ispezionati |
| EN-TCC-motored | Valore per prova (deviazione standard in test = 0) | Stessa prova, non co-localizzata | Deriva dell'olio fino a 14 °C; esterno ≠ interno del liner | Schiffmann 2016, [10.2516/ogst/2015028](https://doi.org/10.2516/ogst/2015028), HAL hal-01707465 §1.1 p. 4, §2.1 p. 8, Tab. A2 — TI | Tipo e posizione dei sensori (Deep Blue inaccessibile) |
| EN-TCC-fired | — | — | La CHT-LES prevede 500 K di gradiente attraverso la finestra | Henrion 2022, [10.1016/j.combustflame.2022.112083](https://doi.org/10.1016/j.combustflame.2022.112083), [OSTI 1976995](https://www.osti.gov/servlets/purl/1976995) §2.2 pp. 8–9 — TI | — |
| EN-Darmstadt | Medie per condizione; incertezza standard combinata 0,5 K | Stesse prove | Lo skip-firing altera lo scarico; pareti della LES assunte | Welch 2023, [10.1007/s10494-022-00347-5](https://doi.org/10.1007/s10494-022-00347-5) §2.1, Tab. 1, Fig. 1a. Chu 2023, [10.1007/s10494-022-00353-7](https://doi.org/10.1007/s10494-022-00353-7) Tab. 1, pp. 7–8. Ding 2019, [arXiv 2009.01739](https://arxiv.org/abs/2009.01739) pp. 7, 15 — TI | Tipo di sonda |
| EN-Darmstadt-LIF | Piano, kHz | **L1** (gas fresco) | Precisione vicino al punto morto superiore; condensazione | Peterson et al. 2014, Appl. Phys. B 117 — **AB** | Testo integrale |
| EN-IFPEN | Piano | **L1** (< ~600 K) | Spegnimento termico | Kopf 2023, IJER, [10.1177/14680874211048559](https://doi.org/10.1177/14680874211048559) — **AB** | Testo integrale |
| EN-Sandia-HCCI | Piano; termocoppie | — | Accuratezza limitata dalla T calcolata dalla pressione | Dec & Hwang 2009, [10.4271/2009-01-0650](https://doi.org/10.4271/2009-01-0650), OSTI 1142409 pp. 1, 5, 7–9 — TI | — |
| RP | — | — | — | §11 | §11 |

---

## 11. Come si colloca RealPDEBench (domanda 9)

Fonti lette dal filone RealPDEBench:

- paper Hu et al., ICLR 2026, [arXiv 2601.01829](https://arxiv.org/abs/2601.01829) v2, **TI**;
- card e metadati Hugging Face `AI4Science-WestlakeU/RealPDEBench` (nessuno shard scaricato);
- sito del progetto;
- codice GitHub (clone superficiale, solo codice).

Le pagine si riferiscono al PDF v2.

1. **Apparato.**
   - [D] *«we construct a swirl combustor and use mass flow controllers to control the injection ratio of air, NH3, and CH4 (Figure 2d)»* (§3.2, p. 5).
   - [D] Condizioni: NH₃ al 0–80 % in volume, φ fra 0,75 e 1,3, contropressione 1 bar (App. B.2, pp. 26–27); 30 coppie (NH₃, φ) negli shard.
   - [M] Non dichiarati:
     - geometria;
     - swirler;
     - confinamento e materiale delle finestre;
     - potenza;
     - portate;
     - preriscaldamento;
     - laboratorio.
   - [I] Gli autori includono il gruppo SJTU di Xingcai Lu. L'unico lavoro sperimentale NH₃/CH₄ citato è Liu et al., Phys. Fluids 36 (2024) 034123, con **OH* e PIV simultanee** (abstract). Il rig è probabilmente lo stesso o affine, **non confermato**. Il testo integrale non è stato accessibile.
2. **Diagnostica dichiarata.**
   - [D] Solo chemiluminescenza **OH***: Photron Mini AX200 con intensificatore, filtro 310 ± 10 nm, *«2000 frames per second across a 70 × 90 mm² field of view»* (App. B.3, p. 30).
   - [D] **Nessuna PIV per la combustione.** La PIV del benchmark riguarda soltanto i dataset in galleria ad acqua: cilindro, cilindro controllato, FSI, profilo alare (App. B.3, pp. 27–30).
   - [D/M] **Nessuna temperatura sperimentale riportata:** né gas, né pareti, né ingresso, né scarico. Una ricerca di "thermocouple" e "wall temperature" nel testo integrale non dà risultati.
   - [I] La foto di Fig. 2d mostra fili e sonde non descritti vicino alla sezione ottica. Non si può dire che cosa siano.
3. **OH* non è OH.**
   - [I] L'OH* è un'**emissione di OH eccitato, integrata lungo la linea di vista** (proiezione).
   - L'OH-PLIF è fluorescenza dell'OH allo stato fondamentale su un foglio laser.
   - Il canale numerico è una **frazione molare di OH** su una sezione.
   - [D] Il paper usa la dicitura *«OH chemiluminescence signal»* (B.4, p. 30).
4. **Origine delle velocità e temperature che usiamo.**
   - [D] Sono **solo LES**: LES 3D implicita instazionaria, STAR-CCM+ 2022.1, EDC, meccanismo con 38 specie e 184 reazioni, CVODE. Ingressi in portata (aria e miscela di combustibile separate), uscita in pressione, pareti no-slip (B.2, p. 26).
   - [M] Non dichiarati:
     - modello SGS;
     - mesh;
     - passo temporale della LES;
     - **condizioni al contorno termiche di parete** (adiabatiche, isoterme o con perdite);
     - **temperatura di ingresso**;
     - piano e interpolazione della sezione 128 × 128.
   - [D] La sola frase sulle condizioni al contorno: *«These conditions correspond to our experimental measurement environment»* (B.2, p. 26).
   - Nessuna affermazione che una condizione termica derivi da misure.
5. **Simulazione ed esperimento.**
   - [D] Condividono le condizioni operative, non gli istanti.
   - [D] L'unico confronto quantitativo è spettrale: FFT del segnale OH* sperimentale contro l'integrale spaziale della concentrazione di OH della LES (B.4, pp. 30–31; Fig. 12).
   - [I] Lettura grafica della Fig. 12:
     - picco sperimentale ≈135–140 Hz;
     - picco simulato ≈170–175 Hz;
     - le condizioni dei pannelli non sono identificate [M].
   - [I] Uno scarto in frequenza è compatibile, fra le altre cause, con condizioni termiche o acustiche diverse, come nei casi GT-KIT e Agostinelli. Non è una diagnosi.
6. **`observed` negli shard numerici.**
   - [D] È una *«surrogate-mapped intensity»*: una rete addestrata a mappare i 15 canali LES più (NH₃ %, φ) sull'immagine OH* reale, accoppiando frame con lo stesso indice temporale (D.2, p. 34; codice).
   - [D] Il paper dice U-Net; lo script di generazione usa una FNO3d.
   - [I] È uno pseudo-OH* appreso statisticamente, non una sintesi radiativa.
7. **Incongruenze temporali non risolte.**
   - [D] *«The sampling frequency is 4000Hz and the duration is 1s»* (B.2, p. 27) contro *«2000 frames per second»* (B.3, p. 30).
   - [D] Tab. 7: 2001 frame, Δt = 2,5 × 10⁻⁴ s, cioè 0,5 s.
   - Le conseguenze sono già registrate nell'inventario della missione 01 (C2).

**Conseguenze per la missione [I]:**

- Nessuna misura termica di RealPDEBench può fare da modello di sensore o da riferimento sperimentale.
- La "verità" T della LES dipende da condizioni al contorno termiche ignote. Le evidenze dei §3 e §8 mostrano che, nei combustori confinati, queste condizioni cambiano livello termico della ORZ, forma della fiamma e termoacustica.
- Non sappiamo se la sezione 128 × 128 contenga pareti, piano di dump o ORZ: nella missione 01 l'orientamento è ancora aperto. Quindi non sappiamo se sensori "di parete" o "di ORZ" siano rappresentabili.

---

## 12. Disaccordi, incongruenze e punti irrisolti

1. **Quanto conta la temperatura di parete?**
   - Gövert 2018 imposta 1000 K citando *«only limited influence of the exact wall temperature»* da studi precedenti.
   - Mercier 2016, Agostinelli 2021 e Kraus 2017 trovano effetti forti su forma, stabilità e frequenza.
   - [I] Le condizioni sono diverse: fiamme CH₄/H₂ bistabili V/M e combustori instabili da una parte, una fiamma stabile e una quantità osservata diversa dall'altra. **Irrisolto** in generale; dipende dall'apparato.
2. **PIV con LES termicamente più realistica.** In GT-KIT l'accordo con la PIV in alcune zone **peggiora** con lo scambio termico, attribuito alle pareti *«mostly based on estimated values»* (Kraus 2017, §5.1). Misure parziali delle pareti non bastano necessariamente.
3. **Henrion 2022.**
   - Un appunto di filone attribuiva *«This additional measurement was not performed»* alla temperatura del pistone.
   - Verificato sul testo: la frase riguarda la misura di **trasmissione della finestra a caldo**.
   - La misura della temperatura del pistone è fra le *«challenges left unaddressed during this study»*.
   - Corretto in §8.1.
4. **Temperatura di parete nei motori.** Le LES di TCC-III impongono 400 K al cilindro (Wu et al. 2022). Il benchmark misura l'**esterno** del liner a 36,9–41,0 °C. Non sono grandezze confrontabili: interno ed esterno, motore trascinato. L'origine dei 400 K non è dichiarata. **Irrisolto.**
5. **Scarico nei combustori.**
   - Il filone GT non ha trovato temperature di scarico riportate nei paper letti.
   - Nau 2024 riporta T_ex = 1277–1354 K, ma senza metodo (rimanda a Petry 2023).
   - NH-LSU ha il rastrello ma non pubblica i valori.
   - Koren 2018 chiede misure allo scarico per una validazione completa.
   - [I] Lo scarico è strumentato più spesso di quanto sia pubblicato. Non verificabile.
6. **Arndt 2015.** La frase *«not possible … simultaneously»* non spiega il perché. La causa delle particelle è **documentata in altri apparati** (Frank 2021; Lund) e **inferita** qui.
7. **CARS vicino alle pareti contro Raman vicino alle pareti.** Il Raman è inaccessibile vicino alle pareti nel GTMC, mentre il CARS arriva a 30 µm in LF-SWQ. Non è un disaccordo: apparati e geometrie diverse (fasci passanti contro angolo solido di raccolta).
8. **Cardiff (rig NH₃ atmosferici).**
   - Tipi K e R, 1 Hz, medie su 120 s, correzione per radiazione, convezione ed eventualmente conduzione: tutto da estratti di motore di ricerca (SEC).
   - ORCA ha restituito una pagina anti-bot.
   - Non usati come evidenza decisiva.
9. **PoliMi.** Un bruciatore swirl non premiscelato con PIV, LDA e termocoppia Pt/Pt-13 %Rh da 0,3 mm, con correzione radiativa basata sulla velocità misurata, è noto solo da abstract ed estratti (IJETR 2017). Pertinente al contesto della tesi, **non verificato**.

---

## 13. Che cosa implica per la tesi (inferenze, non decisioni)

[I] Quanto segue serve a orientare la discussione. Non fissa sensori, numero, posizioni né dataset.

1. **"Termocoppie nella fiamma, lente e sincronizzate con la PIV" non è la configurazione documentata.**
   - Nei casi letti, le termocoppie nel gas stanno dove il gas è freddo (ORZ < 1200 K) o fuori dalla zona ottica.
   - Le misure in fiamma sono ottiche e separate dalla PIV.
   - Un esperimento "velocità sparse + poche T puntuali lente nella fiamma" sarebbe uno **scenario progettuale**. È legittimo, ma da dichiarare come tale.
2. **Configurazioni realistiche candidate** (da valutare nello step 2; non approvate):

   | Opzione | Che cosa rappresenta | Casi | Problema per i nostri dati |
   |---|---|---|---|
   | (a) T di superficie o contorno, medie stazionarie, poche posizioni | Pareti, punta del bluff body, piano di dump | GT-EM2C, GT-DLR, GT-PRECCINSTA | La sezione LES contiene pareti o bordi? Condizioni termiche della LES ignote |
   | (b) Poche termocoppie nella ORZ o allo scarico, medie o lente | Livello termico del gas combusto in zone fredde | GT-EM2C, NH-LSU, LF-Delft | Correzioni e τ dipendenti dal sito; serve una sequenza di frame |
   | (c) T puntuale o lineare a singolo colpo, **non simultanea** con la velocità | Statistiche (medie, PDF) condizionate alla condizione operativa | GT-DLR, LF-TNF | Informazione statistica, non istantanea: cambia il compito (livello termico o statistiche, coerente con Linzheng: *«sparse temperatures can help correct the overall thermal level»*) |
   | (d) T planare simultanea su un campo piccolo | Il caso L1 | LF-OSU | Raro, aperto, combustibili su misura; non trasferibile al combustore NH₃/CH₄ senza ipotesi |
   | (e) Isoterma binaria dalle particelle PIV | Informazione termica "gratuita" | LF-isoterma | Soglia (~500 K) e dipendenza dalla semina |
   | (f) T di gas fresco o freddo < ~1000 K da particelle fosforescenti | Regioni d'ingresso e mescolamento | LF-Yin-TPIV | Copre solo il fresco |

3. **La risposta temporale va trattata come proprietà del sito e del montaggio**, non della "termocoppia" in generale. Lo stesso vale per la correzione radiativa, che richiede la temperatura delle pareti.
4. **Validazione.** Nessun dataset letto offre T e velocità sperimentali simultanee in un combustore swirl confinato. La validazione resterà numerica (VN) o al più statistica (VST), come già registrato nella matrice della missione 01.

---

## 14. Indicazioni per lo step 2 (proposte motivate, da discutere)

### 14.1 Agente A — caratteristiche e problemi dei dati termici

In ordine di pertinenza [I]:

1. **Termocoppie a filo fine e inguainate in gas combusto relativamente freddo e allo scarico.**
   - Correzione radiativa e conduttiva.
   - Costante di tempo variabile e compensazione (due fili di diametro diverso).
   - Effetto della temperatura delle pareti.
   - Perturbazione della sonda.
   - Fonti da cui partire: Shaddix 2017 (TI); Heitor & Moreira 1993; Tagawa & Ohta 1997; Ballantyne & Moss 1977; Chandran 1985; Miles & Gouldin 1993 (questi ultimi solo AB, da recuperare).
   - **Motivo:** è il sensore lento e puntuale previsto dalla missione, ed è il meno documentato nei rig ottici.
2. **Fosfori termografici su superfici.**
   - Calibrazione ed estrapolazione.
   - Copertura parziale e interpolazione.
   - Dipendenza dalla condizione e dalla storia termica.
   - Degrado del quarzo.
   - **Motivo:** è la misura termica più sistematica nei rig PIV confinati.
3. **Raman/Rayleigh e CARS puntuali o lineari** come fonte di **statistiche** di T.
   - Incertezze sistematiche e statistiche.
   - Zone escluse.
   - Legame in fase con la pressione.
   - **Motivo:** è il riferimento termico realistico non simultaneo.
4. Solo se si vuole il caso simultaneo: **FRS + PIV** (McManus & Sutton; Most 2002) e **PIV termografica**, con i loro limiti.

### 14.2 Agente B — dataset

Archivi incontrati, da trattare come piste e **non ispezionati**:

- archivi TNF: Sandia D–F, CSSB con temperature del bluff body (Euler 2014), Darmstadt TSF/SWQ, LSB;
- dati di contorno DLR GTMC/PRECCINSTA, se esistono negli archivi TNF;
- ISF-3 (fiamma fuligginosa DLR a 3 bar, comprese le pareti);
- Deep Blue TCC-III: "Near-Wall Flow" con flusso termico in testa e PIV; "CFD Input Dataset"; serie accese;
- TUdatalib Darmstadt (contorno del motore; temperatura del pistone con H₂);
- Cardiff research data (rig NH₃);
- Adelaide JHC (Kildare et al. 2024, ricostruzione di T con U-Net da PLIF);
- tesi di McManus (FRS + PIV);
- dati Sydney sulla superficie del bluff body;
- EngineBench (solo velocità).

**Domanda da porre per ciascuno:** c'è T con velocità nella stessa condizione? Simultanea? Con quale supporto?

### 14.3 Da verificare prima o in parallelo, con la missione 01

- Piano, orientamento ed estensione della sezione RealPDEBench rispetto a pareti, ugello e ORZ.
- Se sia possibile ottenere dagli autori le condizioni al contorno termiche della LES e la geometria del rig (via Enrico; nessun contatto da parte mia).

---

## 15. Lacune, copertura e domande da discutere

### 15.1 Copertura raggiunta

**Letto in testo integrale** (dettagli nel [registro](step1_fonti_e_ricerche.md)):

- DLR: GTMC, SFB606, PRECCINSTA, fuligginoso, HBK;
- EM2C: Mercier, Koren;
- KIT: Kraus 2017/2018;
- ETH;
- fiamme da laboratorio: OSU, Lund (tesi), Yin TPIV, SWQ-CARS, Delft, TNF12/13;
- motori: TCC-III benchmark, Henrion, Darmstadt (3 paper FTC + Ding), Sandia HCCI;
- strumenti: Shaddix 2017, Frank 2021;
- LES con dichiarazioni sulle condizioni termiche: Gövert, Kumar, Agostinelli, Papafilippou, Poinsot 2017;
- RealPDEBench (paper e codice).

### 15.2 Lacune principali [M]

- **Rig sperimentali ad ammoniaca:** quasi tutto a livello di abstract. Il rig SJTU, probabile origine di RealPDEBench, non è accessibile. Cardiff ORCA ha restituito pagine anti-bot. KAUST, Tohoku e XJTU solo abstract.
- **Originali DLR non letti:** Weigand 2006 e Meier 2007. Non sappiamo se quelle campagne usassero termocoppie e dove.
- **Originali EM2C non letti:** Guiberti 2015 e Degenève 2019 (HAL anti-bot).
- **MIT (laboratorio Ghoniem):** DSpace ha restituito 429. **Non verificato** se e dove si usino termocoppie (per esempio sul flame-holder).
- **Review chiave non lette:** Hochgreb 2019 *Mind the gap*, Barlow 2007, Dreizler & Böhm 2015, Abram 2018, Brübach 2013, Heitor & Moreira 1993.
- **Nessun dato di forum o di pratica informale raccolto:** ResearchGate e Reddit non raggiungibili.
- **Copertura non esaustiva.** I casi sono sbilanciati verso gruppi con LES e archivi aperti (DLR elib, HAL, OSTI, arXiv). Gruppi cinesi, giapponesi e industriali sono sottorappresentati.

### 15.3 Domande per la discussione prima dello step 2

1. Il problema di fusione mira alla **T istantanea sulla sezione**, al **livello termico**, o a **statistiche**? Le configurazioni realistiche (§13, opzione c) favoriscono le ultime due. Linzheng: *«reconstruction targets do not necessarily need to be limited to the full instantaneous temperature field»*.
2. Accettiamo uno **scenario progettuale** (termocoppie lente in zone di gas non documentate sperimentalmente), purché dichiarato? Oppure vincoliamo i sensori a siti documentati (pareti, ORZ fredda, scarico, ingresso)?
3. Quali configurazioni affidare all'agente A dello step 2 (§14.1)? Propongo termocoppie in ORZ e scarico più fosfori di parete come priorità, Raman/CARS come riferimento statistico.
4. Serve chiedere agli autori RealPDEBench le condizioni al contorno termiche e il piano della sezione? Senza quelle informazioni, un sensore di parete o di ORZ non è collocabile con certezza.
5. Vale la pena recuperare tramite biblioteca i testi integrali bloccati? Priorità: rig SJTU (Liu 2024), Guiberti 2015, Weigand 2006/Meier 2007, Cardiff (tesi Davies 2025), MIT (Hong/Taamallah), PoliMi IJETR 2017.

### 15.4 Stato della consegna e branch

- **Branch.** L'ambiente impone `claude/thermal-measurement-scouting-rxj1l0`. All'avvio coincideva con la base `a894257497de1524566186c22b66e02bd7b7b8ee`, come anche `mission04/step1-thermal-scouting`. Le consegne sono committate sul branch imposto.
- **Non toccati:** codice, risultati storici, roadmap, altre missioni e `documents_thesis`.
- **Lo step 1 non è segnato come approvato.**
