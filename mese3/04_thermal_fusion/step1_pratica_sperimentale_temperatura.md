# Missione 04 — Step 1: come si misura la temperatura negli esperimenti di combustione con accesso ottico

**Versione:** v2, revisione mirata della consegna v1 (commit `5d566be`) dopo la lettura del coordinatore. Nota di revisione al §R.  
**Data:** 27 novembre 2026.  
**Stato:** **proposta da revisionare nella chat di coordinamento della missione 04. Non approvata.**  
**Branch:** `claude/thermal-measurement-scouting-rxj1l0` (imposto dall'ambiente; parte dalla base indicata `a894257`, vedi §15.4).  
**Perimetro:** solo scouting bibliografico dello step 1 di [roadmap04.md](roadmap04.md). Niente training, inferenza, download di dataset, implementazione di sensori virtuali o contatto con autori. Nessuna scelta di dataset, architettura, target, numero o posizione dei sensori, né di tipo di validazione.  
**Documento collegato:** [step1_fonti_e_ricerche.md](step1_fonti_e_ricerche.md): bibliografia ragionata con livello di lettura, registro delle ricerche, verifiche del coordinatore, accessi falliti ed esclusioni.

---

## R. Nota di revisione (v1 → v2)

**Generalizzazioni corrette.** La selezione bibliografica non è un censimento rappresentativo: è stata costruita in parte proprio cercando apparati con misure termiche. Per questo:

- ho rimosso, o riferito ai soli casi esaminati, le espressioni di frequenza: "si misura quasi sempre", "raramente", "quasi mai", termocoppie "solo" nelle regioni fredde o "mai" nella zona di fiamma, "soprattutto" fosfori;
- la frase di Arndt 2015 è attribuita alla configurazione descritta, senza aggiungere cause che gli autori non dichiarano;
- i limiti numerici (per esempio ~1000 K per le particelle fosforescenti, i limiti del quarzo) sono riferiti al materiale e all'apparato della fonte;
- ho separato "non trovato", "non riportato" e "non misurato";
- ho distinto le campagne separate *per impossibilità dichiarata* da quelle separate *senza motivazione*.

**Verifiche mirate effettuate.** Ricerca dei controesempi alla conclusione negativa sulle termocoppie nel gas caldo:

- pista PoliMi;
- velocimetria laser puntuale (LDV/LDA) con termocoppia;
- combustori moderni con PIV e termocoppie;
- rilettura del caso TU Delft.

Ho verificato personalmente sui testi i passaggi decisivi ([fonti §2](step1_fonti_e_ricerche.md)).

**Controesempi trovati.**

1. **Termocoppie nel gas caldo, dentro la regione studiata con PIV, sullo stesso apparato ma in campagne separate.** Documentati in testo integrale o nelle sezioni pertinenti:
   - **Virginia Tech (GT-VT):** sonda tipo B traversata nella zona primaria, 400–1600 K, griglia di 320 nodi; liner rimosso per la scansione.
   - **TU Delft, combustore a getti (GT-DelftJSC):** termocoppia tipo S inguainata traversata; ~1250 K nel ricircolo periferico (PRZ) e ~1500 K in quello centrale (CRZ). È usata una camera d'acciaio gemella *«used during intrusive measurements»*.
   - **TU Delft, jet-in-hot-coflow chiuso (LF-Delft):** termocoppia tipo S nel gas di reazione, coflow di riferimento a 1000 °C. PIV solo a freddo, usata per correggere la termocoppia.

   Solo a livello di abstract, con i dettagli decisivi mancanti:
   - PoliMi: Olivani et al. 2007, PIV + LDV + *«thermocouples for the temperature distribution»*;
   - Università del Maryland (UMD): Archer 2005, PIV + microtermocoppie compensate;
   - altri casi elencati al §5.3.
2. **Misure simultanee sonda + velocimetria laser nel gas caldo, con LDV e non con PIV:** Heitor, Taylor, Whitelaw 1985; Ferrão & Heitor 1998; Yanagi & Mimura 1981. Verificate solo a livello di abstract. Gli stessi autori documentano effetti della semina sulla termocoppia.

**Conclusioni cambiate.**

- "Non trovate termocoppie nel gas caldo nella regione PIV" era **errato**. Diventa: **termocoppie nel gas caldo trovate solo in campagne separate dalla PIV**. Non ho trovato, nei casi esaminati, termocoppie nel gas caldo **acquisite simultaneamente alla PIV**. Non concludo che la configurazione non esista.
- Il caso Delft-JHC era classificato "fuori dalla zona ottica": **era sbagliato** ed è stato corretto.
- "Termocoppie in fiamma associate alla velocità" non è più presentato come "scenario progettuale". La combinazione è documentata con LDV e, con PIV, in campagne separate. Non è documentata, nei casi letti, come acquisizione simultanea con PIV.
- "La validazione resterà numerica o al più statistica" è sostituito da una constatazione limitata alle condizioni esaminate (§13).
- Le statistiche Raman/CARS non sono più presentate come equivalenti a sensori lenti, né come implicanti un cambio di target.

**Lacune ancora aperte.**

- Testi integrali di PoliMi, UMD, Xi'an Jiaotong (XJTU) e degli altri candidati noti solo da abstract.
- Testo integrale del rig SJTU, probabile origine di RealPDEBench.
- Originali DLR del 2006–07 ed EM2C del 2015.
- Review metodologiche sulle sonde.
- Molti host editoriali restano bloccati dall'ambiente (§15.2).

---

## 0. Come leggere questo rapporto

**Etichette** (come nelle missioni 01 e 02):

- **[D] Documentato:** affermato nella fonte, con posizione (sezione, pagina, figura, tabella). Le citazioni tra virgolette sono testuali.
- **[I] Inferenza:** una mia interpretazione motivata.
- **[M] Mancante:** informazione non riportata nelle fonti lette o fonte non accessibile.

**Livello di lettura** (definizioni anche in [fonti](step1_fonti_e_ricerche.md)):

- **TI:** documento letto per intero.
- **TS:** testo integrale disponibile; lette per intero le sezioni pertinenti (setup, metodo, risultati usati); il resto solo scorso o cercato per parole chiave.
- **PZ:** lettura parziale o ricerca per parole chiave.
- **AB:** solo abstract o metadati.
- **SEC:** fonte secondaria o estratto del motore di ricerca.

Un abstract può segnalare un caso. Non può sostenere affermazioni che richiedono il metodo: posizioni, diametri, simultaneità, correzioni.

**Ambito delle conclusioni.** Tutte le affermazioni del tipo "si fa / non si fa" si riferiscono **ai casi esaminati** (tabella §10, controesempi §5.3), non alla pratica sperimentale in generale. La selezione privilegia gruppi che fanno anche simulazioni LES, archivi aperti (DLR elib, HAL, OSTI, arXiv, TU Delft) e apparati che documentano misure termiche. Non permette di stimare frequenze.

**Regole di linguaggio:**

- "Non riportato" ≠ "non misurato": scrivo "non misurato" solo se gli autori lo dicono.
- "Non trovato" ≠ "inesistente".
- "Nello stesso paper" ≠ "simultaneo".
- "Campagne separate" ≠ "impossibile insieme", salvo motivazione dichiarata dagli autori.

**Numeri.** Ogni numero vale per l'apparato e la condizione indicati. Le quantità "±" sono riportate come le scrivono gli autori.

**Più letture della stessa fonte non sono conferme indipendenti.**

**Identificativi dei casi:** `GT-…` combustori confinati; `NH-…` ammoniaca; `LF-…` fiamme da laboratorio; `EN-…` motori ottici; `LV-…` misure con LDV/LDA; `RP` RealPDEBench.

**Come è stato prodotto:**

- prima consegna: sette filoni paralleli (combustori GT, ammoniaca, fiamme da laboratorio, motori, strumenti, scopi e assenze, RealPDEBench);
- revisione: due ricerche mirate di controesempi (PoliMi e sonde + LDV; combustori moderni con PIV e termocoppie);
- integrazione e verifica a campione sui PDF originali da parte mia ([fonti §2](step1_fonti_e_ricerche.md)).

---

## 1. Risposta diretta

### 1.1 Sintesi

**Nella maggior parte dei casi esaminati compare almeno una temperatura, ma con significati molto diversi** (§1.3). Spesso è solo un parametro di funzionamento: "aria a temperatura ambiente" o un preriscaldamento nominale. La temperatura calcolata (adiabatica) è un'altra cosa. Fanno eccezione RealPDEBench, che non riporta alcuna temperatura sperimentale, e alcuni paper PIV che riportano solo condizioni ambiente (§8.2). Il conteggio non è una frequenza della pratica: la selezione privilegia apparati con misure termiche (§0).

**La temperatura del gas nella regione studiata con la PIV**, nei casi esaminati, è misurata in due modi:

1. **Con tecniche laser** (Raman, CARS, Rayleigh filtrato) in punti, linee o piccoli piani. Nei combustori confinati letti avviene in campagne separate dalla PIV. Simultaneamente alla PIV solo in poche configurazioni specifiche (§1.4).
2. **Con termocoppie traversate nel gas**, comprese zone calde (1200–1600 K documentati) dentro la regione inquadrata dalla PIV.
   - Nei casi verificati in testo integrale o sezioni sono sempre in **campagne separate** dalla PIV, spesso con hardware modificato (liner rimosso, camera d'acciaio gemella).
   - In due casi la velocità misurata serve proprio a **correggere** la termocoppia: LF-Delft (verificato) e PoliMi (solo estratto).

**Simultaneità.** Nei casi esaminati **non ho trovato** termocoppie nel gas caldo acquisite simultaneamente alla PIV. Ho invece trovato misure simultanee di termocoppie compensate e **LDV** (velocimetria puntuale, non PIV) in fiamme premiscelate. Queste ultime sono lette solo a livello di abstract.

### 1.2 Che cosa è documentato nei casi esaminati (combustore swirl confinato con PIV; fra parentesi i casi)

1. **Termocoppie o sensori di servizio.**
   - Dove: nel plenum, nel condotto d'ingresso, allo scarico, nelle barre metalliche che reggono le finestre (GT-EM2C, GT-KIT, NH-LSU; nei motori EN-TCC, EN-Darmstadt).
   - Scopo dichiarato: fissare e verificare il punto di funzionamento; monitorare l'equilibrio termico (circa 45 min in GT-EM2C); dare condizioni al contorno di ingresso alle simulazioni.
   - In vari paper è riportata solo la condizione nominale, senza strumento: GT-DLR-SFB606, GT-PRECCINSTA, GT-EM2C per l'ingresso, GE/Toronto.
2. **Temperature di superficie** (finestre lato interno ed esterno, piano di dump, punta del bluff body o del centerbody, barre).
   - Nei casi DLR ed EM2C con fosfori termografici.
   - In alcuni casi con termocoppie su parti metalliche (GT-EM2C, GT-ETH, GT-KIT).
   - Con IR attraverso il quarzo in un caso documentato da un rapporto tecnico (GT-VT).
   - Scopo dichiarato: condizioni al contorno termiche per LES, perdite di calore, forma della fiamma e PVC.
3. **Temperatura del gas con tecniche laser.**
   - DLR: Raman puntuale su 45–70 punti; fs-CARS a 5 kHz.
   - Fiamme aperte: FRS planare.
   - Nei combustori DLR in campagne distinte dalla PIV, alla stessa condizione operativa. Per GT-DLR-SFB606 gli autori scrivono che *«it is not possible to perform Raman and PIV measurements simultaneously»* in quella configurazione; il motivo non è dichiarato nella frase.
4. **Temperatura del gas con termocoppie dentro la camera:**
   - nella zona di ricircolo esterna relativamente fredda: < 1200 K, tipo K da 1,5 mm (GT-EM2C);
   - nella **zona primaria calda**: tipo B traversata, 400–1600 K, griglia di 320 nodi, liner rimosso (GT-VT);
   - nel **ricircolo caldo**: tipo S inguainata, PRZ ~1250 K e CRZ ~1500 K, camera d'acciaio gemella (GT-DelftJSC);
   - nel gas di reazione con coflow caldo, PIV solo a freddo (LF-Delft).

   In tutti i casi verificati le misure sono in campagne separate dalla PIV.
5. **Nei motori ottici** (EN-TCC, EN-Darmstadt):
   - temperature di set-point puntuali e mediate per prova;
   - gas nel cilindro calcolato dalla pressione;
   - gas fresco misurato in un piano insieme alla PIV, con LIF di toluene o particelle fosforescenti, prima della combustione (EN-Darmstadt-LIF, EN-IFPEN: solo abstract).
6. **Assenze dichiarate**, per esempio:
   - *«The exact wall temperature was not measured in the experimental study»* (PRECCINSTA, Gövert 2018);
   - *«Wall temperature measurements are not available for the case considered»* (Kumar 2023);
   - *«the actual flame temperature or the burned gas was not measured in the present work»* (Wang et al.).

   In vari paper PIV l'unica "temperatura di fiamma" riportata è quella adiabatica calcolata.

### 1.3 Quattro tipi di informazione termica da non confondere

| Tipo | Che cosa è | Esempi nei casi | Segue la stessa realizzazione del flusso della PIV? |
|---|---|---|---|
| **A. Misura locale dinamica durante la prova** | Segnale registrato nel tempo durante la stessa prova, in un punto o in un piano | FRS + PIV (LF-OSU); LIF di toluene + PIV (EN); termocoppie compensate + LDV (LV-…); fosfori a 1 kHz sul bluff body (GT-PRECCINSTA, con OH-PLIF) | Sì, se sincronizzata con la velocità. Nei casi trovati solo nelle configurazioni del §1.4 |
| **B. Temperatura media o parametro di funzionamento** | Valore medio di prova o di condizione, spesso nominale | Ingresso, plenum, scarico, refrigerante; T adiabatica (calcolata) | No: descrive la condizione |
| **C. Temperatura di superficie** | Pareti, finestre, bluff body, pistone; nei casi letti medie in stato stazionario, salvo GT-PRECCINSTA | GT-EM2C, GT-DLR, GT-HBK, EN-Darmstadt (pistone) | Di norma no (eccezione: GT-PRECCINSTA a 1 kHz) |
| **D. Statistiche termiche da un'altra campagna** | Medie, PDF e profili del gas da Raman/CARS/FRS o da termocoppie traversate, acquisiti in prove distinte | GT-DLR (Raman, CARS), LF-Lund, GT-VT, GT-DelftJSC, LF-Delft | No: stessa condizione nominale, altra realizzazione, a volte altro hardware |

[I] Un sensore lento registrato nella stessa prova della PIV (tipo A, filtrato dalla risposta del sensore) e le statistiche di tipo D portano **informazioni diverse**. Il primo segue, filtrata, la realizzazione osservata; le seconde descrivono la condizione. Come usarle nella ricostruzione è una questione metodologica aperta (§13).

### 1.4 Temperatura e velocità nello stesso istante e nello stesso posto

| Via | Dove documentata (lettura) | Limite nel caso documentato |
|---|---|---|
| FRS planare + PIV a lunghezze d'onda diverse, sincronizzate | LF-OSU (TI): getti pilotati non premiscelati, 10 Hz | Combustibile su misura (CH₄/H₂/Ar) perché il segnale dipenda solo da T; foglio FRS di ~8 mm contro 30 mm della PIV. A Lund PIV e FRS insieme *«were not successful … due to the high seeding densities required for PIV»* (LF-Lund) |
| Particelle fosforescenti usate sia come tracciante PIV sia come termometro | LF-Yin (TI, fiamma laminare); EN-IFPEN (AB) | Con BAM:Eu²⁺ in LF-Yin il limite superiore misurato è 900–1000 K. Con ZnO:Zn in EN-IFPEN il segnale cala vicino al punto morto superiore |
| LIF di toluene + PIV ad alta velocità | EN-Darmstadt-LIF (AB) | Solo gas fresco; calibrazione ancorata alla T calcolata dalla pressione |
| Termocoppie compensate + **LDV** (non PIV) | LV-Heitor 1985, LV-Ferrão 1998, Yanagi & Mimura 1981 (AB) | Velocimetria puntuale. L'esposizione alla semina può alterare la termocoppia; la termocoppia *«attenuates the measured velocity-temperature correlations»* rispetto al Rayleigh |
| Isoterma binaria dalle goccioline d'olio della PIV | LF-isoterma (PZ) | Informazione sopra/sotto una soglia (~500 K per l'olio d'oliva in quel caso) |

**Non trovato nei casi esaminati:** termocoppie nel gas caldo acquisite in sincronia con la PIV. È una questione aperta, non una conclusione di inesistenza (§5.3).

### 1.5 Che cosa significa per noi (inferenze da discutere, nessuna scelta)

- [I] Lo scenario di Linzheng, *«slower, spatially limited thermal measurements»*, ha nei casi esaminati diversi riscontri possibili, ciascuno con un tipo diverso di informazione (§1.3):
  - temperature di superficie o di contorno;
  - termocoppie traversate nel gas, anche caldo, in campagne separate;
  - termocoppie compensate in sincronia con una velocimetria puntuale (LDV).
- [I] Una termocoppia lenta nel gas registrata durante la stessa prova delle osservazioni PIV non è stata trovata documentata nei casi letti. Non è esclusa. Se la useremo, andrà dichiarato quali parametri (posizione, risposta, correzioni) provengono da quali casi.
- **RealPDEBench:**
  - [D] non rilascia temperature sperimentali e non dichiara PIV per la combustione;
  - [M] le condizioni al contorno termiche della LES non sono dichiarate;
  - [M] non sappiamo che cosa sia stato misurato sull'apparato senza essere pubblicato (§11).

---

## 2. Mappa delle temperature nei casi esaminati

| Temperatura | Dove (casi) | Strumenti documentati | Scopo dichiarato | Supporto e tempo nei casi | Rapporto con la PIV |
|---|---|---|---|---|---|
| **Aria o miscela in ingresso, plenum** | Plenum dell'aria (GT-KIT); runner di aspirazione (EN-Darmstadt, EN-Sandia-HCCI); a monte della camera (NH-LSU) | Termocoppie tipo K (NH-LSU, EN-Sandia-HCCI); tipo non dichiarato altrove [M]; spesso solo valore nominale | Definire e ripetere la condizione; condizioni al contorno di ingresso delle LES (GT-KIT; EN-Darmstadt: *«measured … average temperatures are imposed at the boundaries»*) | Punto; media per prova o per condizione | Stessa prova o condizione, non co-localizzata (tipo B) |
| **Preriscaldamento "involontario" dei reagenti** | Plenum di combustori non preriscaldati | Termocoppie (GT-KIT: 450/350 K); PRECCINSTA: *«not precisely measured»* (320–380 K) | Frequenza termoacustica, condizioni della LES | Punto, stazionario | Separata |
| **Superfici solide** | Strisce o 17–30 punti su una finestra, interpolati e specchiati; punta del bluff body; barre; pistone | Fosfori (decadimento o rapporto, calibrazione in forno); termocoppie su parti metalliche | Condizioni al contorno per LES/CHT; perdite (≈10 % della potenza a 1 bar in GT-DLR-SFB606, 15–16 % a 3 bar in GT-DLR-sooting); forma V/M e PVC | Punti, linee, mappe parziali; medie di 150–500 colpi; eccezione a 1 kHz (GT-PRECCINSTA) | Campagne separate. In GT-PRECCINSTA simultanea con OH-PLIF, non con la PIV |
| **Gas in zone di ricircolo fredde** | ORZ di GT-EM2C, attraverso una barra | Tipo K da 1,5 mm, senza correzione radiativa (errore dichiarato ≤ 5 % sotto 1200 K) | Validare simulazioni con pareti misurate | Punti, medie stazionarie | Stessa condizione; sincronizzazione non dichiarata |
| **Gas caldo con termocoppie traversate** | Zona primaria (GT-VT); PRZ e CRZ (GT-DelftJSC); coflow di reazione (LF-Delft); vicino all'ugello (PoliMi, UMD: AB) | Tipo B (GT-VT); tipo S inguainata (Delft); bead Pt-Rh da 0,3 mm (PoliMi, SEC) | Validazione CFD, struttura termica, emissioni | Punti traversati; valori di equilibrio o medie (2–10 Hz) | **Campagne separate**, spesso con hardware modificato. In LF-Delft la PIV serve a correggere la termocoppia |
| **Gas in fiamma, tecniche laser** | Griglie di 45–70 punti (DLR); linee di 6 mm (Sandia); piani di 8 mm (FRS) | Raman, CARS, Rayleigh/FRS, two-line OH-PLIF | Struttura di fiamma, mescolamento, perdite, LES, modelli acustici | Singolo colpo (5–10 Hz) o serie a 5 kHz (CARS); statistiche e medie in fase | Separata nei combustori confinati letti; simultanea in LF-OSU |
| **Scarico** | Rastrello di 4 termocoppie (NH-LSU); uscita (GT-VT); tubo di scarico (EN-Darmstadt) | Termocoppie | Stato di funzionamento; pattern factor (GT-VT); fasatura (motori) | Punto, media | Separata (tipo B) |
| **Gas nel cilindro (motori)** | — | **Calcolata** dalla pressione; LIF calibrato su quella stima | Rilascio di calore; calibrazione della LIF | Bulk (0D) o piano | Simultanea solo per gas fresco (EN-Darmstadt-LIF, EN-IFPEN: AB) |
| **Soot** | Getti diesel | Pirometria a due colori | Temperatura del soot | Proiezione | Non nei casi letti |
| **Calcolate** | Tabelle delle condizioni | T adiabatica | Caratterizzazione; numero di Stokes delle particelle PIV (GT-EM2C: 1881 K) | — | — |

---

## 3. A che cosa serve conoscere la temperatura (domanda 1)

Scopi **dichiarati** nei casi letti:

1. **Caratterizzare e ripetere la condizione.**
   - [D] TCC-III: *«to minimize temperature gradients in the air flow»*; motore trascinato 15–20 min prima delle prove (EN-TCC, §1.1, p. 4).
   - [D] EM2C: acquisizione dopo lo stato stazionario indicato da una termocoppia, circa 45 min (Mercier 2016, §2.5).
   - [D] SJTU: *«~6 minutes»* di attesa, senza monitoraggio riportato (Wang et al., §II).
2. **Condizioni al contorno termiche per le simulazioni.**
   - [D] *«LES are then extremely sensitive to the prescription of thermal boundary conditions, which are here carefully measured»* (Mercier 2016, §2.5).
   - [D] *«often missing as important boundary conditions»* (Arndt 2021, Intro).
   - [D] *«valuable parameters for improved boundary conditions of numerical simulations»* (Nau 2019, p. 9).
   - [D] *«measured … average temperatures are imposed at the boundaries»* (Chu 2023, pp. 7–8).
3. **Validazione delle simulazioni e struttura termica.**
   - [D] GT-VT: *«The experimental data can be used to validate CFD calculation, and also to understand flow structure and heat transfer in a swirl type combustor»* (rapporto DOE, introduzione al Cap. 6).
   - [D] GT-EM2C: le termocoppie della ORZ validano le simulazioni con pareti imposte.
4. **Studio della combustione.**
   - [D] Con pareti adiabatiche la LES prevede sempre una fiamma a M; con le pareti misurate anche la V (GT-EM2C).
   - [D] *«As the formation of PVC is highly sensitive to the local density (temperature) field…»* (Yin 2017).
   - [D] Mescolamento e dissipazione: gradiente di T dall'FRS insieme alla deformazione dalla PIV (LF-OSU).
   - [D, AB] Flusso turbolento di calore da correlazioni velocità–temperatura misurate con LDV + termocoppia (LV-Heitor).
5. **Scambio termico.**
   - [D] Flusso termico attraverso le finestre (GT-DLR-SFB606, GT-DLR-sooting).
   - [D] GT-VT: il profilo vicino al liner, 1000–1400 K, *«shows significant contribution of the flow temperature to the convection heat transfer to the liner wall»* (Cap. 6, discussione).
   - [D] Interazione fiamma–parete (LF-SWQ).
6. **Termoacustica.**
   - [D] Frequenze sensibili a preriscaldamento e perdite (GT-KIT).
   - [D] La condizione al contorno di parete decide se la LES prevede una fiamma stabile o instabile (Agostinelli 2021).
   - [D] Modello di Helmholtz costruito sulle temperature Raman (Arndt 2015, §2.6).
   - [D] Termocoppia di parete come variabile di stato (GT-ETH).
7. **Velocità usata per la termometria, e viceversa.**
   - [D] *«The non-reacting flow field measurements were used to correct the thermocouple measurements»* (LF-Delft, conclusioni).
   - [D, SEC] PoliMi: *«using the measured velocity values for the evaluation of convective heat transfer coefficient»*.
   - [D] La T adiabatica è usata per il numero di Stokes delle particelle (GT-EM2C).
   - [D, AB] Termoforesi e indice di rifrazione (Stella 2001; Vanselow 2019).
   - [D] Isoterma dalle goccioline (LF-isoterma).
8. **Emissioni** (rig ad ammoniaca e Delft).
   - [D] Perdite di calore e temperature servono a interpretare NO e ad ancorare reti di reattori. In NH-LSU la perdita è regolata come parametro libero (*«Heat loss is the only free variable»*, §4.2).
   - [D] La *«flame zone temperature»* è fra le condizioni incerte (NH-Cardiff-HPOC).
9. **Limiti dei materiali e delle sonde.**
   - [D] Ventilatori sulle finestre (Sadanandan 2009).
   - [D] Degrado del quarzo sopra ~1400 K nei rig DLR misurati (Nau 2017; Nau 2019).
   - [D] *«Thinner thermocouples could not be used during the flame experiments due to breakage at the high temperatures»* (GT-VT, Cap. 4, testo della Fig. 4-20).
   - [M] Nei casi letti non c'è una misura del quarzo motivata esplicitamente dalla sicurezza.

---

## 4. Strumenti: che cosa si misura e come si arriva alla temperatura (domanda 3)

### 4.1 Termocoppie

**Segnale.** Forza elettromotrice, cioè la temperatura del giunto.

**Dal giunto al gas.** Serve un bilancio con convezione, irraggiamento verso pareti e finestre e conduzione lungo i fili.

- [D] Formula classica: T_g = T_tc + εσ(T_tc⁴ − T_w⁴)·d/(Nu·k) (Shaddix 2017, Eq. 1–3). Per Shaddix è *«fundamentally incorrect»* quando il bead è più grande dei fili.
- [D] La lettura è *«sensitive to variations in gas temperature over a spatial dimension that is much larger than the thermocouple bead»* (abstract; Fig. 6).
- [D] L'errore radiativo diventa importante sopra ~800 K (Shaddix 2017, Intro).

**Correzioni nei casi letti.**

- Bilancio di energia con irraggiamento di gas e pareti più convezione:
  - LF-Delft: h dal campo di velocità a freddo;
  - GT-DelftJSC: parametri nell'appendice supplementare, non letta;
  - PoliMi (SEC): h dalla velocità misurata.
- Nessuna correzione, con stima dell'errore:
  - GT-EM2C: ≤ 5 % sotto 1200 K;
  - GT-VT: errore radiativo stimato in 11,1 K a 1600 K (tabella degli errori, Cap. 6).

**Risposta temporale nei casi letti.**

- [D] GT-VT, termocoppie di riferimento nel ricircolo e all'uscita:
  - bead da 0,762 mm, *«approximate time response of 1.5 seconds»* secondo il costruttore;
  - le guaine ceramiche possono allungarla.
- [D] GT-VT, scansione con tipo B:
  - tre campioni a 2 Hz, salvati quando la deviazione standard scende sotto lo 0,5 %;
  - isteresi secondo la direzione di scansione;
  - errore transitorio *«within 10%»* (Cap. 6).
- [D] LF-Delft: registrazione a 10 Hz, traversata a 0,62 mm/s, media su un diametro di sonda (71 campioni).
- [D, AB] Nelle misure LDV + termocoppia la risposta è compensata digitalmente, con costante di tempo funzione di velocità e temperatura (Heitor 1985).
- [D, AB] Chandran 1985, nella sua fiamma: *«there is no unique time constant in a turbulent flame with widely fluctuating temperature»*.
- [D] Missione 01 (pool fire NIST): τ ≈ 57 ms per un filo da 50 µm.

**Interazione con la semina (LDV)** [D, AB]:

- *«exposure to the solid particles required by the laser anemometer … able to cause large differences in the measured probability-density-distribution of the reaction progress variable»* (Heitor 1985, abstract OSTI);
- deposito di particelle compensato con una costante di tempo maggiorata (Yanagi & Mimura 1981);
- correlazioni attenuate rispetto al Rayleigh (Ferrão & Heitor 1998).

**Motivi dichiarati per evitarle in alcuni casi.**

- [D] *«thermocouples not only disturb the flow, but also their responses are not fast enough…»*: per questo un forno flameless di Delft usa LDA e CARS (X. Huang, tesi 2018, p. 44).
- [D] DLR: le termocoppie sono *«intrusive … only point measurements … the wiring can be difficult»*, ma anche *«widely used, well understood … high accuracy»* (Nau 2019, p. 2).

### 4.2 Fosfori termografici su superficie

- **Segnale.** Luminescenza di un rivestimento di 10–20 µm, letta come decadimento o come rapporto di intensità.
- **Temperatura.** Si ricava da una calibrazione in forno contro una termocoppia di tipo B o K.
- **Nei casi DLR/EM2C:**
  - accuratezza 0,5–1 %, fino a 2–3 % su quarzo caldo;
  - ogni fosforo copre circa 300–400 K;
  - punti, strisce o mappe;
  - medie di 150–500 colpi, oppure 1 kHz in GT-PRECCINSTA (picco a 475 Hz attribuito al PVC).
- **Limiti documentati:** fluorescenza del quarzo; opacizzazione sopra ~1400 K nei rig DLR; fondo radiativo; soot; spot grandi rispetto a parti piccole.
- **Montaggio:** eccitazione dal retro della finestra, *«therefore laser beam and phosphorescence signal did not have to pass through the flame»* (Nau 2019, p. 5).
- [I] Nei casi esaminati è la tecnica di superficie ricorrente nei gruppi DLR ed EM2C. Non ne stimo la prevalenza generale.

### 4.3 Termografia infrarossa (IR)

- [D] GT-VT: IR della parete attraverso il quarzo, calibrato con termocoppie di superficie (rapporto, Cap. 3–4).
- [D, AB] Pittura ad alta emissività sul liner (RSI 2006).
- [D] Limiti dichiarati: *«flame emissions and reflected radiation»* (Nau 2019); quarzo *«opaque for infrared»* (Rodrigues 2018).

### 4.4 Raman e Raman/Rayleigh

- **Segnale e temperatura.** Densità delle specie misurate; *«the temperature was deduced from the total number density via the ideal gas law»* (Arndt 2015, §2.4). Nella variante Raman/Rayleigh di Sandia, T viene dal Rayleigh con sezione d'urto ricavata dalla composizione Raman.
- **Supporto e tempo.**
  - DLR: volume di 0,6 mm, 5 Hz, 400–2500 colpi per punto, 45–70 punti.
  - Sandia: linea di 6 mm.
- **Incertezze** [D, Arndt 2015]: ±3–4 % sistematica; 3–3,5 % statistica a singolo colpo.
- **Limiti.**
  - In GT-DLR-SFB606 non sono accessibili h < 8 mm e r > 27 mm (*«clipping of the solid angle of the detection optics»*).
  - [D] *«Elastic scattering from particles … can overwhelm the Rayleigh scattering signal»* (Frank 2021, §3).
  - Polvere → breakdown (LF-Lund).
  - [I] Per Arndt 2015 queste sono cause plausibili della non-simultaneità con la PIV, ma gli autori non le dichiarano.

### 4.5 CARS

- **Segnale.** Spettro coerente di N₂.
- **Nei casi.**
  - fs-CARS a 5 kHz, volume di 0,4–0,6 mm, 5000 colpi, ±2 % di precisione e ±3 % di accuratezza (Dennis 2016).
  - CARS 1D fino a ~30 µm dalla parete (LF-SWQ).
- [D] *«CARS measurements can be performed in particle-laden flows and near surfaces»* (Frank 2021).
- [D] Finestre che si muovono → colpi scartati (Dennis 2016).
- [M] Un abstract Sandia (GRC 2017) annuncia CARS 2D + PIV simultanei; nessun paper trovato.

### 4.6 Rayleigh e Rayleigh filtrato (FRS)

- **Segnale.** Intensità ∝ densità × sezione d'urto. Nell'FRS una cella molecolare (I₂) o atomica (Hg) assorbe la luce alla frequenza del laser, cioè la diffusione Mie delle particelle e la luce diffusa dalle pareti.
- **Limiti.**
  - Dipendenza dalla composizione (Frank 2021, §4.1). Risolta con un combustibile su misura (LF-OSU) o con un'ipotesi sulla sezione d'urto (LF-Lund).
  - FRS modulato in frequenza: solo medie temporali (DLR, AB).

### 4.7 Two-line OH-PLIF

- Rapporto di due eccitazioni dell'OH. Nessuna temperatura dove manca l'OH (Giezendanner-Thoben 2005, AB). Accuratezza 4–7 %.

### 4.8 LIF di traccianti (motori)

- Rapporto di fluorescenza del toluene, calibrato su una T calcolata dalla pressione: accuratezza *«limited by the pressure-derived temperatures»*.
- Precisione 1σ di 8,8 K al punto morto superiore (Dec & Hwang 2009).
- Solo gas fresco.

### 4.9 Particelle fosforescenti come traccianti PIV

- Le stesse particelle danno la velocità (diffusione Mie) e la T (rapporto a due colori).
- [D] Con BAM:Eu²⁺, in fiamme laminari a H₂: *«The upper measurable temperature is found to be 900–1000 K»* (Yin et al. 2017).
- [D, AB] ZnO:Zn in motore: ±1σ = 3,7 K a 317 K e 24,4 K a 575 K (Kopf 2023).
- Sono limiti di quei fosfori in quelle prove.

### 4.10 Marcatori binari dalla PIV

- Evaporazione delle goccioline: isoterma di ~500 K per l'olio d'oliva (CORIA, arXiv 1004.0074, §3).
- PIV condizionata dal salto di densità delle particelle (Pfadler, AB).
- Informazione binaria.

### 4.11 Pirometria del soot e assorbimento laser (TDLAS)

- Pirometria a due colori: temperatura del soot sulla linea di vista, polarizzata verso il soot più caldo (Reggeti 2022).
- [M] Nessun caso TDLAS + PIV in un combustore swirl trovato.

---

## 5. Dove si misura e perché lì (domanda 4)

### 5.1 Posizioni documentate

- **Ingresso e plenum.**
  - [D] *«The air temperatures in the plenums are monitored with thermocouples»* (Kraus 2017, §2).
  - [D] Tipo K a monte della camera (NH-LSU, §2.2).
  - [D] Runner vicino alla testa (EN-Darmstadt, Welch 2023, Fig. 1a; Sandia HCCI).
- **Barre e montanti.**
  - [D] 6 tipo K su una barra, da 593 K (z = 16 mm) a 637 K (z = 56 mm) (GT-EM2C, Fig. 3).
  - [D] Montante in Inconel con fosforo (GT-DLR-SFB606).
- **Piano di dump e bluff body.**
  - [D] Fosfori, da 435 K (r = 7 mm) a 516 K (r = 25 mm); punta dell'asta 500 ± 50 K (GT-EM2C, Fig. 4).
  - [D] Striscia sulla piastra e punta del bluff body (GT-PRECCINSTA).
- **Finestre.**
  - [D] 30 punti (GT-EM2C, Fig. 5).
  - [D] Tre strisce, lato interno ed esterno (GT-DLR-SFB606).
  - [D] 17 spot (GT-DLR-HBK).
- **Gas nella ORZ fredda.** [D] Tipo K da 1,5 mm *«through one of the four vertical steel bar holding the quartz windows»* (GT-EM2C, §2.5, Fig. 6).
- **Gas caldo, termocoppie traversate.**
  - [D] **GT-VT.** Sonda *«inserted into the model combustor from exhaust side opening»*: filo tipo B in una canna ceramica di ~45 cm, montata su un tubo d'acciaio di 1 m e su un traslatore 2D. Griglia di 320 nodi, 0 < r/D_N < 1,36. *«the liner and transition piece were disconnected during thermocouple scanning in the primary zone»* (Cap. 6).
  - [D] **GT-DelftJSC.** Il combustore *«is either a quartz pipe or a stainless steel 304 pipe … whereas the steel chamber is used during intrusive measurements with probes such as thermocouple measurements»* (GT2022, §2). Termocoppia tipo S inguainata inserita dalle porte P1–P6 (Fuel 2025, §2.2).
  - [D] **LF-Delft.** *«One of the quartz windows is replaced by a stainless steel plate with equally spaced ports»*. Dieci porte a 30 mm di passo; traversata radiale alla porta 1, Y/D = 19,5 (§2).
  - [D, AB] **PoliMi.** *«in the close vicinity of the burner throat»* (Olivani 2007). Posizioni [M].
- **Gas in fiamma, ottico.** [D] Griglia Raman a h = 8–80 mm, r = −3…27 mm (GT-DLR-SFB606).
- **Scarico.**
  - [D] Rastrello di 4 termocoppie a ~30 cm dal dump (NH-LSU).
  - [D] Profili radiali nella transition piece e all'uscita; pattern factor 0,04 (GT-VT).
  - [D] Sonda in allumina (GT-DLR-HBK).

### 5.2 Motivazioni: dichiarate e inferite

**Dichiarate [D]:**

- Pareti: condizioni al contorno mancanti per le LES, perdite di calore, legame con la forma V/M e con il PVC.
- Eccitazione dei fosfori dal retro della finestra, per non attraversare la fiamma (Nau 2019).
- GT-VT: sonda inserita dallo scarico, con protezione termica e raffreddamento del connettore; *«made to avoid damage during the measurement»*. Termocoppie più sottili si rompevano.
- GT-DelftJSC: camera d'acciaio per le sonde, *«used during intrusive measurements»*; il quarzo è riservato all'ottica.
- Correzione delle termocoppie con la velocità misurata (LF-Delft; PoliMi SEC).
- Zone escluse dall'ottica: angolo solido del Raman; nessuna PIV dentro l'ugello a causa delle riflessioni (Nau 2019, p. 8).
- Tecniche laser scelte per non disturbare il flusso e per la risposta lenta delle termocoppie (X. Huang 2018, forno flameless).

**Inferite [I]:**

- Nei casi esaminati, durante le campagne PIV nel gas caldo la termocoppia è stata tolta, oppure montata su un hardware diverso, oppure usata in altre prove.
- Ragioni plausibili: interferenza della sonda e della guaina con il foglio laser e con la semina, perturbazione del flusso, sopravvivenza della sonda.
- Gli autori le dichiarano solo in parte: "intrusive" (GT-DelftJSC), rotture (GT-VT), disturbo e risposta lenta (X. Huang).

### 5.3 Esito della ricerca di controesempi (termocoppie nel gas caldo e PIV)

| Caso | Gas caldo documentato | Termocoppia | Posizione rispetto a fiamma e PIV | Simultaneità | Lettura |
|---|---|---|---|---|---|
| **GT-VT**: combustore a lattina ottico, swirl industriale premiscelato CH₄; Re 50 000, φ = 0,65, pilota 6–7 % | 400–1600 K nella zona primaria; 1000–1400 K vicino alla parete | Tipo B (PtRh/Pt) su sonda costruita in laboratorio. Tipo K con bead da 0,762 mm nel ricircolo e all'uscita durante le prove di scambio termico | Zona primaria, griglia 2D. PIV reagente nel Cap. 5 (Re 50 000 rispetto all'ugello; coincidenza di φ non verificata [M]) | **Separata.** Liner e transition piece scollegati per la scansione. Il confronto con la PIV è indicato come sviluppo futuro (*«Further comparison with PIV measured flow field will make more comprehensive data»*) | TS (Cap. 4 in parte, Cap. 5–6) |
| **GT-DelftJSC**: combustore a getti premiscelato CH₄/H₂; 60 kW nei casi di Fig. 2 | PRZ ~1250 K, CRZ ~1500 K; *«less than 1800 K»* nella maggior parte delle posizioni | Tipo S con guaina in platino (diametro [M] nel paper 2025) | Traversate radiali attraverso le interfacce getto/ricircolo, nel dominio studiato con PIV reagente | **Separata.** Camera d'acciaio gemella | TS (Fuel 2025 §2–3; GT2022 §2) |
| **LF-Delft**: jet-in-hot-coflow chiuso, coflow a 1000 °C di riferimento | Gas di reazione: valori nelle Fig. 9 e 17; nel caso senza getto, differenza di 700 K fra porta 1 e porta 10 | Tipo S con guaina in platino, 3 mm | Porte in una piastra al posto di una finestra | **Separata.** PIV solo a freddo, usata per correggere la termocoppia | TS (§2–3, conclusioni) |
| **GT-PoliMi**: swirl non premiscelato a gas naturale, confinato | *«thermal … field in the close vicinity of the burner throat»*; ~1370 K a 45 mm da un altro paper (SEC) | Bead Pt-Rh da 0,3 mm; correzione radiativa con la velocità misurata (SEC) | [M] | [M]; [I] probabilmente separata | AB / SEC |
| **GT-UMD**: Archer 2005, doppio swirl concentrico, 33 kW | Medie e rms della T in fiamma (abstract) | Microtermocoppie compensate | [M] | [M] | AB |
| **LV-Heitor**: fiamma premiscelata stabilizzata da baffle o disco | Attraverso il flame brush [I, forte] | Filo fine compensato digitalmente | Vicino al volume di misura LDA | **Simultanea con LDV** | AB |
| **LV-Ferrão**: fiamma a propano stabilizzata da disco | Fiamma ricircolante | Filo fine compensato; fino a 2000 Hz | Insieme all'LDV | **Simultanea con LDV**; confronto con Rayleigh | AB |
| Altri candidati segnalati: XJTU PCI 2023; Kim/Arghode/Gupta 2009; Li & Gutmark 2006; Kwak 2022; Guo GT2008; Hsu 2019; Valencia 2018 (fiamma a parete verticale) | Vari. Valencia 2018: massimo ~1220 K, fiamma a parete di tipo incendio | Vari | Valencia 2018: stesse altezze dei profili PIV, campagna separata [D] | Non stabilita | Valencia: TI; gli altri AB |

**Esito.**

1. La conclusione negativa della v1 è **smentita** nella forma "nessuna termocoppia nel gas caldo nella regione PIV". **Esistono** campagne di termocoppie nel gas caldo sugli stessi apparati e nella regione studiata con PIV: GT-VT, GT-DelftJSC, LF-Delft; PoliMi e UMD solo da abstract.
2. Nei casi verificati il legame con la PIV è di **campagna separata**: in due casi con hardware modificato o diverso, in uno con la PIV a freddo usata per la correzione.
3. Misure simultanee termocoppia + velocità nel gas caldo sono documentate con **LDV**, non con PIV (abstract).
4. **Non trovato:** una termocoppia nel gas caldo sincronizzata con la PIV. La ricerca è stata mirata ma non esaustiva: molti testi integrali erano bloccati.

---

## 6. Quale informazione produce ogni misura (domanda 5)

| Misura (casi) | Tipo (§1.3) | Supporto spaziale | Tempo: acquisizione, risposta, durata |
|---|---|---|---|
| Termocoppia di servizio (EN-TCC, EN-Darmstadt, NH-LSU) | B | Punto | Un valore per prova in TCC-III (deviazione standard in test = 0, Tab. A2); frequenza [M] |
| Termocoppia nella ORZ fredda (GT-EM2C) | D | Punto (1,5 mm) + conduzione lungo i fili [I, per analogia con Shaddix 2017] | Medie stazionarie |
| Termocoppia traversata nel gas caldo (GT-VT) | D | 320 punti su un piano | 2 Hz fino a deviazione standard < 0,5 %; isteresi; errore transitorio < 10 % |
| Termocoppia di riferimento nel ricircolo (GT-VT) | A/B, registrata durante le prove di scambio termico | Punto | ~1,5 s di risposta (costruttore) |
| Termocoppia traversata (LF-Delft, GT-DelftJSC) | D | Punti su profili radiali | 10 Hz; media su un diametro di sonda |
| Termocoppia compensata + LDV (LV-…) | A | Punto | Fino a 2000 Hz di campionamento combinato (LV-Ferrão, AB) |
| Termocoppia di parete come variabile di stato (GT-ETH) | A/C | Punto | Transitorio di decine di secondi |
| Fosfori su superficie (GT-EM2C, GT-DLR, GT-HBK) | C | Punti, linee, mappe parziali | 10–15 Hz; medie di 150–500 colpi |
| Fosfori ad alta velocità (GT-PRECCINSTA) | A (con OH-PLIF) / C | Linea di ~80 mm | 1 kHz per 8,1 s |
| Raman puntuale (GT-DLR) | D | 0,6 mm, punti in sequenza | 5 Hz; 400–2500 colpi; medie in fase |
| fs-CARS (GT-DLR) | D | Punto | 5 kHz per 5000 colpi |
| FRS + PIV (LF-OSU) | A | Piano alto 8 mm | 10 Hz, singolo colpo |
| PIV termografica (LF-Yin, EN-IFPEN) | A | Piano | 5 Hz (LF-Yin) |
| LIF di toluene + PIV (EN-Darmstadt-LIF) | A | Piano | kHz (AB) |
| T bulk nel cilindro | Modello | 0D | — |
| Isoterma dalle goccioline | A, binaria | Contorno | Come la PIV |

[I] Nei casi esaminati:

- le misure di tipo A sono poche e legate a configurazioni particolari;
- le termocoppie nel gas sono acquisite fino all'equilibrio della sonda, oppure mediate;
- termocoppie compensate per seguire le fluttuazioni compaiono solo nei lavori con LDV.

---

## 7. Rapporto con la misura di velocità (domanda 6)

| Livello | Definizione | Casi |
|---|---|---|
| **L1 — simultaneo e co-localizzato** | Stesso colpo o stessa finestra temporale, stesso punto o piano | FRS + PIV (LF-OSU: *«The PIV laser pulses are timed to frame straddle the FRS laser pulse»*); PIV termografica (LF-Yin; EN-IFPEN, AB); LIF di toluene + PIV (EN, AB); isoterma binaria; **termocoppia compensata + LDV** (LV-Heitor, LV-Ferrão, Yanagi & Mimura: AB; velocimetria puntuale, non PIV) |
| **L2 — simultaneo, ma non co-localizzato o non con la PIV** | Stessa prova, altra grandezza o regione | Fosfori + OH-PLIF (GT-PRECCINSTA); termocoppie di servizio o di riferimento durante le prove [I] |
| **L3 — stesso apparato e condizione, campagne separate** | Legame statistico o in fase | Raman ↔ PIV (GT-DLR-SFB606); fs-CARS ↔ PIV (GT-DLR-GTMC); FRS/OH ↔ PIV/OH (LF-Lund); fosfori ↔ PIV (GT-EM2C, GT-DLR, GT-HBK); **termocoppia nel gas caldo ↔ PIV reagente** (GT-DelftJSC con camera gemella; GT-VT con liner rimosso, φ della PIV da verificare) |
| **L4 — stesso apparato, condizioni diverse** | φ, potenza o regime diversi | Gövert 2018 (LDV a φ = 0,75, Raman a 0,83); Kumar 2023 (PIV a freddo); Agostinelli 2021; **LF-Delft** (PIV a freddo, termocoppia in fiamma) |
| **L5 — laboratori o geometrie diverse** | — | Fiamme TNF; Wang et al. (T inferita da dati di altri bruciatori) |

**Motivi della separazione, quando dichiarati.**

- LF-Lund: *«Simultaneous PIV and FRS were not successful, however, due to the high seeding densities required for PIV»*.
- GT-DelftJSC: camera d'acciaio *«used during intrusive measurements»*.
- GT-VT: liner scollegato per la scansione; rotture delle termocoppie sottili.
- GT-DLR-SFB606: *«not possible … simultaneously»* nella configurazione descritta, senza causa dichiarata.
- In molti altri casi nessun motivo è dichiarato: fosfori DLR ed EM2C, Raman e CARS, PoliMi, UMD.

[I] Campagne separate non implicano un'impossibilità tecnica. In molti casi riflettono l'organizzazione delle prove o la scelta degli strumenti.

[D] Il workshop TNF13 chiede misure di velocità *«at flow conditions matching those of the scalar experiments»* (pp. 11–12).

**PIV reagente o a freddo.**

- Reagente: GT-DLR, GT-PRECCINSTA, GT-EM2C, GT-KIT, GT-DelftJSC, GT-VT (Cap. 5), LF-OSU, LF-Lund, LF-SJTU.
- Solo a freddo: Kumar 2023, LF-Delft, EN-TCC benchmark (motore trascinato).

---

## 8. Quando la temperatura non si misura o non si riporta (domanda 7)

### 8.1 Assenze dichiarate esplicitamente

| Caso | Che cosa manca | Testo | Conseguenza |
|---|---|---|---|
| PRECCINSTA, Gövert 2018 | Pareti | *«The exact wall temperature was not measured in the experimental study and is therefore unknown»* (§3.2, p. 516) | 1000 K imposti |
| DLR GTMC TP09, Kumar 2023 | Pareti per quel caso | *«Wall temperature measurements are not available for the case considered»* (p. 1000) | Pareti adiabatiche |
| SJTU, swirl aperto, Wang et al. | Gas combusto | *«the actual flame temperature or the burned gas was not measured in the present work»* (§IV.A) | T da modello calibrato su altri bruciatori |
| PRECCINSTA, preriscaldamento | Ingresso | *«not precisely measured»* (Agostinelli 2021, che cita Meier 2007) | 320 K imposti in LES |
| TCC-III acceso, Henrion 2022 | Pistone; trasmissione della finestra a caldo | Pistone fra le *«challenges left unaddressed during this study»*; trasmissione: *«This additional measurement was not performed»* (§2.2, pp. 8–9) | Trasmissione a temperatura ambiente |
| Darmstadt PTV, Ding 2019 | Pareti | *«Further measurements will include wall temperature measurements»* (p. 15) | — |
| Casi riportati da Mercier 2016 (SEC) | Pareti | *«were not measured»*, *«were not available»*, *«not realized on the same burner geometry»* | 1000 K; confronti qualitativi |
| Sutton, rapporto DOE | T del gas a valle | Troppo soot per l'FRS | Solo PIV + LII |
| Forno flameless di Delft, X. Huang 2018 | Termocoppie nel gas, per scelta | Tecniche laser *«to avoid disturbing the internal flow»* (p. 44) | LDA + CARS |

Motivi generali dichiarati (Kraus 2018, §1; Agostinelli 2021): le temperature di parete sono *«very difficult to determine»*, *«change for each operating condition»*, sono *«measured only at a few points»*.

### 8.2 Non riportate (nessuna conclusione sull'esistenza)

- RealPDEBench (combustione).
- LSU, swirl aperto (De, Zhu, Acharya).
- CECOST.
- GE/Toronto: solo preriscaldamento nominale.
- Paper PIV del DLR GTMC classico: termocoppie, scarico e pareti.
- SJTU PoF 2024, KAUST, Tohoku: solo abstract.

### 8.3 Il caso intermedio

Temperature **misurate altrove o in un'altra condizione** e poi riusate:

- pareti di Yin misurate a 20 kW, usate per LES a 10 o 25–30 kW;
- pilot lip della TSF prima stimato (~750 K), poi misurato (~840 K) (TNF12/TNF13).

[D] In alcuni apparati TNF le misure termiche sono arrivate dopo il database di velocità (TNF13). Non generalizzo.

---

## 9. Vincoli che limitano le misure (domanda 8)

Riporto solo quanto serve a capire la pratica; la caratterizzazione degli errori spetta allo step 2. Ogni vincolo è riferito ai casi in cui è documentato.

- **Intrusività.**
  - La sonda perturba il flusso: media su un diametro di sonda (LF-Delft).
  - Disturbo dichiarato come motivo per usare tecniche laser (X. Huang 2018).
  - Modifica dell'hardware per inserire la sonda (GT-VT, GT-DelftJSC).
- **Sopravvivenza.**
  - Rotture delle termocoppie sottili (GT-VT).
  - Protezione del connettore; giunto di riferimento fino a 200 °C, con errore stimato di ~3 % (GT-VT, Cap. 6).
- **Accesso ottico.**
  - Angolo solido di raccolta del Raman (GT-DLR-SFB606).
  - Riflessioni nell'ugello (GT-PRECCINSTA).
  - Sporcamento delle finestre: *«Fouling of the windows is a recurrent issue for PIV measurements»* (GT-VT).
  - Finestre che si muovono o diventano opache (DLR).
- **Materiali.** Quarzo a 1000–1400 K nei rig DLR misurati, con degrado sopra ~1400 K; raffreddamento delle finestre.
- **Risposta temporale.** τ ≈ 57 ms per un filo da 50 µm (NIST); ~1,5 s per un bead da 0,762 mm (GT-VT); compensazione digitale (LDV).
- **Correzioni.** Dipendono da velocità, emissività e temperatura delle pareti (Shaddix 2017; Delft). [I] Nei combustori con finestre la temperatura delle pareti è a sua volta poco nota.
- **Particelle.**
  - Semina PIV contro FRS (LF-Lund).
  - Semina LDA sulla termocoppia (LV-Heitor, AB).
  - BAM:Eu²⁺ spenti oltre 900–1000 K (LF-Yin).
- **Calibrazione e composizione.**
  - Fosfori estrapolati oltre il campo del forno (Arndt 2021).
  - Rayleigh dipendente dalla composizione.
  - LIF ancorata alla pressione.
- **Operatività.**
  - Skip-firing nei motori.
  - Durate limitate (GT-VT: ~2 h per la fornitura di combustibile).
  - Attese per l'equilibrio termico (6–45 min).

---

## 10. Tabella dei casi decisivi

Livello di lettura: TI, TS, PZ, AB, SEC (§0). Per manoscritti e preprint (elib, HAL, OSTI, arXiv, Zenodo, TU Delft) la pagina è quella del manoscritto.

### 10a. Apparato, PIV, temperatura, strumento, posizione

| ID | Apparato e condizioni | PIV | Temperatura cercata e scopo | Strumento → grandezza | Posizione e motivazione |
|---|---|---|---|---|---|
| **GT-DLR-SFB606** | DLR, doppio swirl, CH₄ parzialmente premiscelato, 89×89×112 mm³, 1 bar; fiamma B: 25 kW, φ = 0,7, 392 Hz; aria ambiente | Stereo 5 Hz, reagente | (a) Gas: struttura, perdite, modello acustico. (b) Pareti: condizioni al contorno, flusso termico | (a) Raman → densità → T (gas ideali). (b) Fosfori, decadimento → T da calibrazione | (a) 70 punti, h = 8–80 mm; zone escluse. (b) Strisce su una finestra e su un montante; *«often missing as important boundary conditions»* |
| **GT-DLR-GTMC** | DLR GTMC classico, 85×85×114 mm³; per esempio 10 kW, φ = 0,75 | PIV/OH-PLIF 5–10 kHz | Gas; finestre ~1000 K (stima) | Raman; fs-CARS a 5 kHz | Griglie di punti; ventilatori sulle finestre |
| **GT-PRECCINSTA** | Swirl radiale + bluff body, CH₄ premiscelato; 20 kW bistabile; 25–30 kW | Stereo-PIV + OH-PLIF 10 kHz, campagna separata | Superfici (PVC, V/M, LES); preriscaldamento *«not precisely measured»* | Fosfori eccitati dal laser dell'OH-PLIF | Punta del bluff body, piastra di base |
| **GT-DLR-sooting** | 3 bar, C₂H₄, finestre raffreddate | Campagne precedenti | Quarzo interno ed esterno → flusso termico | Fosfori YAG:Eu/Dy, fotomoltiplicatore | Striscia centrata. *«had to rely on estimated wall temperatures»* |
| **GT-DLR-HBK** | FLOX a getti, 8 bar, 725 K | PIV + OH-LIF, separate | Pareti 1050–1400 K; scarico fino a 1700 K | Fosfori, 17 spot | Segmenti di quarzo; condizioni al contorno |
| **GT-EM2C** | Swirl CH₄/H₂, 4 kW, φ = 0,7, reagenti a 293 K | PIV 10 Hz, reagente e a freddo | Pareti, dump, asta, barre, ORZ: condizioni al contorno e validazione | Tipo K (barre; ORZ da 1,5 mm); fosfori | ORZ attraverso una barra; < 1200 K, nessuna correzione |
| **GT-KIT** | Doppio swirl concentrico, 30 kW, φ = 0,85 | Stereo 5 Hz | Plenum e pareti: termoacustica, condizioni al contorno | Termocoppie; colori di ricottura | Plenum (450/350 K) |
| **GT-ETH** | Swirl premiscelato a gas naturale | Nessuna | Parete come variabile di stato | Tipo K | Parete ceramica |
| **GT-VT** | Combustore a lattina ottico, swirl industriale premiscelato CH₄; Re 50 000, φ = 0,65 | Planare, reagente e a freddo (Cap. 5) | Gas nella zona primaria e all'uscita: validazione CFD, scambio termico | Tipo B scansionata; tipo K da 0,762 mm come riferimento; IR delle pareti | Dallo scarico, griglia di 320 nodi; liner rimosso |
| **GT-DelftJSC** | Combustore a getti premiscelato CH₄/H₂ (Sampat 2025; GT2022) | Planare reagente, camera in quarzo | Gas in PRZ e CRZ, emissioni | Tipo S inguainata; bilancio radiativo e convettivo | Porte P1–P6 della camera d'acciaio gemella |
| **GT-PoliMi** | Swirl non premiscelato a gas naturale, confinato | PIV + LDV (AB) | Campo termico vicino alla gola | Termocoppie (bead Pt-Rh da 0,3 mm: SEC) | [M] |
| **NH-SJTU** | SJTU, swirl NH₃/CH₄; candidato rig RealPDEBench [I] | OH* + PIV simultanee (AB) | *«temperature of combustion chamber»* (AB) | [M] | [M] |
| **NH-LSU** | Swirl NH₃/CH₄, 12–22 kW | Nessuna | Ingresso, scarico | Tipo K; rastrello di 4 termocoppie | Valori allo scarico non riportati |
| **NH-Cardiff-HPOC** | Alta pressione, NH₃ crackizzata al 99 % | Nessuna | Preriscaldamento; campione delle emissioni | [M] | — |
| **LF-OSU** | Getti pilotati CH₄/H₂/Ar, Re 10–30k | Stereo a 568 nm, 10 Hz | Dissipazione termica | FRS (cella I₂) | x/d = 20; foglio di 8 mm |
| **LF-Lund** | Low-swirl CH₄, φ = 0,62 | 2C/stereo 8 Hz | Database per LES | FRS (cella Hg) | PIV + FRS non riuscita |
| **LF-Yin-TPIV** | Fiamme laminari H₂/N₂/CH₄ | Stesse particelle | Lato combustibile; limiti del metodo | Rapporto a due colori | Limite 900–1000 K |
| **LF-SWQ** | Side-wall quenching, Darmstadt | Separata (AB) | Gas vicino alla parete; parete | CARS 1D; fosfori (SEC) | Fino a ~30 µm dalla parete |
| **LF-SJTU** | Swirl aperto CH₄ | PIV 10 kHz + CH* | T del gas combusto, per il numero di Reynolds | Nessuno: T da modello | — |
| **LF-Delft** | Jet-in-hot-coflow chiuso, coflow a 1000 °C di riferimento | Stereo **solo a freddo** | Profili di T del gas di reazione | Tipo S, guaina da 3 mm, correzione con la velocità PIV | Porte in una piastra al posto di una finestra |
| **LF-TNF** | Sandia D–F, CSSB, TSF | LDV/PIV in altri laboratori | Scalari | Raman/Rayleigh/CO-LIF | Laboratori diversi |
| **LV-Heitor / LV-Ferrão** | Fiamme premiscelate stabilizzate da disco o baffle | **LDV**, non PIV | Correlazioni velocità–T, flusso di calore | Filo fine compensato (Ferrão: anche Rayleigh) | Vicino al volume di misura LDA |
| **EN-TCC-motored** | TCC-III trascinato | PIV in 4 piani | Set-point | Tipo [M] | Port, olio, refrigerante, esterno del liner |
| **EN-TCC-fired** | TCC-III acceso, finestra del pistone in zaffiro | Nessuna | Pistone e finestra | Non misurata | — |
| **EN-Darmstadt** | SG-DISI ottico, skip-fire 1/7 | PIV ad alta velocità | Ingresso, scarico; condizioni al contorno della LES | Sonde di T e p | Fig. 1a di Welch 2023 |
| **EN-Darmstadt-LIF** | Darmstadt, trascinato | Simultanea | Gas fresco | LIF di toluene | Piano (AB) |
| **EN-IFPEN** | Motore IFPEN | PIV termografica | Gas fresco | Rapporto di intensità | Piano (AB) |
| **EN-Sandia-HCCI** | HCCI in N₂ | Nessuna | Stratificazione termica | PLIF di toluene; termocoppie | Runner; anello distanziale |
| **RP** | RealPDEBench, NH₃/CH₄/aria, 30 condizioni | Nessuna per la combustione | Nessuna T sperimentale rilasciata | OH* | — |

### 10b. Supporto e tempo, rapporto con la velocità, limiti, fonte, mancante

| ID | Supporto e tempo | Rapporto con la velocità | Limiti | Fonte (DOI/URL), posizione, lettura | Mancante |
|---|---|---|---|---|---|
| GT-DLR-SFB606 | Punto, 5 Hz, 400–2500 colpi; strisce, ~200 colpi | L3; *«not possible…»* senza causa dichiarata | Zone escluse; estrapolazione della calibrazione | Arndt 2015, [10.1007/s00348-015-1929-3](https://doi.org/10.1007/s00348-015-1929-3), [elib 96436](https://elib.dlr.de/96436/) §2.3–2.4, Results — **TS**. Arndt 2021, [10.1016/j.proci.2020.06.088](https://doi.org/10.1016/j.proci.2020.06.088), elib 141835 — **TI** | Ingresso, scarico |
| GT-DLR-GTMC | Punti; CARS a 5 kHz | L3 | Finestre che si muovono | Dennis 2016, [10.1016/j.combustflame.2016.02.033](https://doi.org/10.1016/j.combustflame.2016.02.033), OSTI 1533658 — **PZ**. Meier 2010, Exp. Fluids 49, elib 65267 — **TI** | Weigand 2006 |
| GT-PRECCINSTA | Linea, 1 kHz per 8,1 s | L2 con OH-PLIF; L3/L4 con la PIV | Bassa sensibilità sotto 450 K | Yin 2017, [10.1016/j.expthermflusci.2016.11.004](https://doi.org/10.1016/j.expthermflusci.2016.11.004) — **TI**. Nau 2019, [10.1115/1.4040716](https://doi.org/10.1115/1.4040716) — **TI**. Agostinelli 2021, [10.1016/j.combustflame.2021.111728](https://doi.org/10.1016/j.combustflame.2021.111728), Zenodo 6450237 — **TS** | Strumento del preriscaldamento |
| GT-DLR-sooting | Punti, 150 colpi | L3 | Soot; degrado delle finestre | Nau 2017, [10.1007/s00340-017-6856-y](https://doi.org/10.1007/s00340-017-6856-y) — **TS** (§1–3) | CARS e PIV di Geigle |
| GT-DLR-HBK | 17 spot | L3 | Degrado sopra 1400 K | Nau 2019, pp. 5–10 — **TI** | — |
| GT-EM2C | Punti, medie stazionarie | L3 [I] | Copertura parziale delle pareti | Mercier 2016, [10.1016/j.combustflame.2016.05.006](https://doi.org/10.1016/j.combustflame.2016.05.006), HAL hal-01405801 — **TS** (§1–3). Koren 2018, [10.1007/s10494-018-9895-5](https://doi.org/10.1007/s10494-018-9895-5) — **PZ** | Guiberti 2015 (SEC) |
| GT-KIT | Punti | L3 | Pareti *«mostly based on estimated values»* | Kraus 2017, [10.1115/1.4035143](https://doi.org/10.1115/1.4035143), hal-01688648 — **TI** (filone F). Kraus 2018, [10.1016/j.combustflame.2018.01.007](https://doi.org/10.1016/j.combustflame.2018.01.007) — **TS** | Posizioni |
| GT-ETH | Transitorio | — | — | Bonciolini 2019, [arXiv 1807.02428](https://arxiv.org/abs/1807.02428) — **TI** | Frequenza |
| GT-VT | 320 punti, 2 Hz fino all'equilibrio; riferimento ~1,5 s | L3 (liner rimosso); confronto con la PIV indicato come futuro | Isteresi; rotture; giunto di riferimento caldo | Ekkad et al., rapporto finale DOE/NETL DE-FE0011762 (Virginia Tech, 2018), [OSTI 1463254](https://www.osti.gov/biblio/1463254), Cap. 4 (testo della Fig. 4-20), Cap. 5, Cap. 6 (pp. 187–199) — **TS** | φ della PIV a Re 50 000; diametro del filo tipo B |
| GT-DelftJSC | Profili radiali | L3 (camera gemella) | Sonda ingombrante | Sampat et al., Fuel 395 (2025) 135059, [10.1016/j.fuel.2025.135059](https://doi.org/10.1016/j.fuel.2025.135059) §2.2, §3.1 — **TS**. Sampat et al., GT2022-81808, [10.1115/GT2022-81808](https://doi.org/10.1115/GT2022-81808) §2 — **TS**; entrambi dal repository TU Delft | Appendice B.1 (supplementare) |
| GT-PoliMi | [M] | [M] | [M] | Olivani et al. 2007, [10.1016/j.expthermflusci.2006.05.003](https://doi.org/10.1016/j.expthermflusci.2006.05.003), OSTI 20864947 — **AB**. Solero, IJETR 2017 — **AB/SEC** | Tutto il metodo |
| GT-UMD | [M] | [M] | [M] | Archer 2005, tesi UMD, hdl 1903/2949 — **AB** | Tutto il metodo |
| NH-SJTU | [M] | OH* e PIV simultanee (AB) | — | Liu 2024, [10.1063/5.0196764](https://doi.org/10.1063/5.0196764) — **AB** | Setup |
| NH-LSU | Media [I] | — | Valori allo scarico non riportati | [arXiv 2206.12015](https://arxiv.org/abs/2206.12015) §2–4 — **TS** | Tipo, correzione |
| NH-Cardiff-HPOC | — | — | — | PMC12926937 — **TS** | — |
| LF-OSU | Piano di 8 mm, 10 Hz | L1 | Combustibile su misura | McManus & Sutton 2022, [10.1016/j.proci.2022.07.052](https://doi.org/10.1016/j.proci.2022.07.052), OSTI 2420963 — **TI** | Larghezza del campo |
| LF-Lund | Piano, 10 Hz | L3 | Semina | Petersson et al. 2007, [10.1364/AO.46.003928](https://doi.org/10.1364/AO.46.003928), nella [tesi Lund 2014](https://lup.lub.lu.se/search/files/3031390/4255515.pdf) — **TS** | Campo di vista |
| LF-Yin-TPIV | Piano, 5 Hz | L1 | 900–1000 K | Yin et al. 2017 (CNF), [elib 113064](https://elib.dlr.de/113064/1/2017_Yin_CnFComm_author_final.pdf) — **TI** | — |
| LF-SWQ | Linea | L3 | — | Bohlin 2017, OSTI 1251595 — **TI** | Kosaka (SEC) |
| LF-SJTU | — | — | T da modello | [arXiv 2210.04181](https://arxiv.org/abs/2210.04181) — **TI** | — |
| LF-Delft | Profili, 10 Hz | L4 | Velocità diverse fra le due campagne | van den Berg 2024, [10.1016/j.jaecs.2024.100298](https://doi.org/10.1016/j.jaecs.2024.100298) §2–3 e conclusioni — **TS** | Valori numerici (solo nelle figure) |
| LF-TNF | Linea | L5 | — | Schneider 2019, OSTI 1497651 — **TS**; TNF12, OSTI 1242775 — **TI**; TNF13, OSTI 1505369 — **PZ** | Dati |
| LV-Heitor / LV-Ferrão | Punto; fino a 2000 Hz | L1 con LDV | Semina sulla termocoppia; correlazioni attenuate | Heitor, Taylor, Whitelaw 1985, Exp. Fluids 3, [10.1007/BF01830191](https://doi.org/10.1007/BF01830191), abstract OSTI ETDE [6435338](https://www.osti.gov/etdeweb/biblio/6435338) — **AB**. Ferrão & Heitor 1998, Exp. Fluids 24, abstract OSTI ETDE [669113](https://www.osti.gov/etdeweb/biblio/669113) — **AB** | Tutto il metodo |
| EN-TCC-motored | Un valore per prova | Stessa prova | Deriva dell'olio | Schiffmann 2016, [10.2516/ogst/2015028](https://doi.org/10.2516/ogst/2015028) — **TI** | Tipo e posizione dei sensori |
| EN-TCC-fired | — | — | — | Henrion 2022, [10.1016/j.combustflame.2022.112083](https://doi.org/10.1016/j.combustflame.2022.112083), OSTI 1976995 §2.2 — **TI** | — |
| EN-Darmstadt | Medie per condizione | Stesse prove | Skip-firing | Welch 2023, [10.1007/s10494-022-00347-5](https://doi.org/10.1007/s10494-022-00347-5); Chu 2023, [10.1007/s10494-022-00353-7](https://doi.org/10.1007/s10494-022-00353-7); Ding 2019, [arXiv 2009.01739](https://arxiv.org/abs/2009.01739) — **TI** | Tipo di sonda |
| EN-Darmstadt-LIF | Piano, kHz | L1 | Solo gas fresco | Peterson et al. 2014, Appl. Phys. B 117 — **AB** | Testo integrale |
| EN-IFPEN | Piano | L1 | Spegnimento termico | Kopf 2023, [10.1177/14680874211048559](https://doi.org/10.1177/14680874211048559) — **AB** | Testo integrale |
| EN-Sandia-HCCI | Piano | — | T calcolata dalla pressione | Dec & Hwang 2009, [10.4271/2009-01-0650](https://doi.org/10.4271/2009-01-0650) — **TI** | — |
| RP | — | — | — | §11 | §11 |

---

## 11. Come si colloca RealPDEBench (domanda 9)

**Fonti:**

- Hu et al., ICLR 2026, [arXiv 2601.01829](https://arxiv.org/abs/2601.01829) v2 (**TI**); le pagine si riferiscono a questo PDF;
- card e metadati Hugging Face `AI4Science-WestlakeU/RealPDEBench`, senza scaricare shard;
- sito del progetto;
- codice GitHub (solo codice).

### 11.1 Che cosa il benchmark rilascia

- [D] Per la combustione: 30 traiettorie con due rami.
  - Un ramo **sperimentale**, che contiene la sola immagine OH* (`observed`).
  - Un ramo **numerico LES**, con 15 canali fra cui T e tre componenti di velocità. Contiene anche un `observed` *«surrogate-mapped»*, prodotto da una rete che mappa i canali LES sull'OH* reale (App. D.2, p. 34; il codice usa una FNO3d, il paper parla di U-Net).
- [D] Non rilascia temperature misurate né velocità misurate per la combustione.

### 11.2 Che cosa gli autori dichiarano

- [D] Apparato: *«we construct a swirl combustor and use mass flow controllers…»* (§3.2, p. 5).
- [D] Condizioni: NH₃ al 0–80 % in volume, φ fra 0,75 e 1,3, 1 bar (App. B.2).
- [D] Diagnostica, *«primarily»*: chemiluminescenza OH* con Photron Mini AX200 e intensificatore, filtro 310 ± 10 nm, *«2000 frames per second across a 70 × 90 mm² field of view»* (App. B.3, p. 30).
- [D] La PIV del benchmark riguarda solo i dataset in galleria ad acqua (App. B.3).
- [D] LES: STAR-CCM+ 2022.1, EDC, 38 specie e 184 reazioni. Ingressi in portata, uscita in pressione, pareti no-slip; *«These conditions correspond to our experimental measurement environment»* (B.2, p. 26).
- [M] Non dichiarati: condizioni termiche di parete, temperatura d'ingresso, modello SGS, mesh, piano e interpolazione della sezione, geometria del rig, laboratorio.
- [D] Il confronto simulazione–esperimento è solo spettrale: FFT dell'OH* sperimentale contro l'OH integrato della LES (B.4; Fig. 12).
  - [I] Lettura grafica: picco sperimentale ≈135–140 Hz, simulato ≈170–175 Hz.
  - [M] Condizioni dei pannelli non indicate.
- [D] Incongruenze temporali: «4000 Hz … 1 s» (B.2), «2000 frames per second» (B.3), e 2001 frame con Δt = 0,25 ms (Tab. 7).

### 11.3 Che cosa potrebbe essere stato misurato sull'apparato senza essere pubblicato

- [M] Non sappiamo se il rig avesse termocoppie, misure di pressione, analisi dei fumi o PIV nelle campagne RealPDEBench.
- [I] Indizi, non prove:
  - (a) il paper del probabile rig d'origine (Liu et al., PoF 2024, SJTU; solo abstract) dichiara OH* e PIV simultanee, pressione ed emissioni. Cita la *«temperature of combustion chamber»*, senza che sappiamo come fosse ottenuta;
  - (b) la foto in Fig. 2d mostra fili e sonde non descritti.
- Nessun contatto con gli autori è stato fatto, né è autorizzato in questa attività.

### 11.4 Distinzioni da mantenere

- OH* (emissione dello stato eccitato, integrata sulla linea di vista) ≠ OH-PLIF (stato fondamentale, su un foglio laser) ≠ frazione molare di OH della LES (valore su una sezione).
- Accesso ottico ≠ PIV.
- Stessa condizione operativa ≠ stesso istante.
- Immagine ottica ≠ slice numerica.

### 11.5 Conseguenze [I]

- Nessuna misura termica rilasciata da RealPDEBench può fare da modello di sensore o da riferimento sperimentale.
- La T della LES dipende da condizioni al contorno termiche non dichiarate. Nei casi confinati letti (§3, §8) queste condizioni influiscono su livello termico, forma della fiamma e termoacustica.
- Non sappiamo se la sezione 128 × 128 contenga pareti, piano di dump o ORZ (questione aperta della missione 01). È quindi aperto se sensori di parete o di ricircolo siano rappresentabili.

---

## 12. Disaccordi, incongruenze e punti irrisolti

1. **Peso della temperatura di parete.** Gövert 2018 parla di *«only limited influence»*, citando studi precedenti; Mercier, Agostinelli e Kraus trovano effetti forti. Apparati e quantità sono diversi; irrisolto in generale.
2. **Accordo LES–PIV in GT-KIT.** Con lo scambio termico peggiora in alcune zone; le pareti erano in parte stimate (Kraus 2017, §5.1).
3. **Henrion 2022.** Attribuzione già corretta in v1: la frase *«not performed»* riguarda la trasmissione a caldo.
4. **TCC-III.** Parete della LES a 400 K contro la temperatura misurata all'esterno del liner, 36,9–41,0 °C. Grandezze non confrontabili; irrisolto.
5. **Scarico nei combustori.** Spesso strumentato ma non pubblicato: NH-LSU; Nau 2024 riporta valori senza metodo; Koren 2018 chiede la misura. Non verificabile.
6. **Arndt 2015.** La frase vale per la configurazione descritta, senza causa dichiarata. L'interferenza delle particelle è documentata in altri lavori (Frank 2021; LF-Lund), non in questo.
7. **CARS e Raman vicino alle pareti.** Apparati diversi; nessun conflitto.
8. **Fonti deboli.** Cardiff (NH₃): dettagli delle termocoppie solo da estratti di motore di ricerca. PoliMi: solo abstract o estratti.
9. **v2, caso Delft-JHC.** In v1 era classificato "fuori dalla zona ottica". In realtà le porte sono su una piastra che sostituisce una finestra, dentro la camera. Corretto.
10. **v2, conclusione negativa.** La v1 escludeva termocoppie nella fiamma inquadrata dalla PIV. È smentita dai casi GT-VT e Delft (§5.3). Resta non trovata la sola acquisizione simultanea con la PIV.
11. **v2, GT-VT.** Il filone di ricerca descriveva il confronto fra termocoppia e PIV come diretto. Il rapporto lo indica come sviluppo futuro (Cap. 6, discussione), e così è riportato qui.

---

## 13. Che cosa implica per la tesi (inferenze, non decisioni)

[I] Quanto segue orienta la discussione. Non fissa sensori, posizioni, target, dataset, architettura né tipo di validazione.

**1. Tipi di informazione termica che i casi esaminati rendono plausibili come "misure lente e sparse".**

| Opzione | Che cosa rappresenta | Tipo (§1.3) | Casi | Da chiarire per i nostri dati |
|---|---|---|---|---|
| (a) Temperature di superficie o di contorno in poche posizioni | Pareti, punta del bluff body, piano di dump | C | GT-EM2C, GT-DLR, GT-PRECCINSTA | La sezione LES contiene pareti o bordi? Condizioni termiche della LES ignote |
| (b) Termocoppie nel gas, in zone fredde o calde, mediate o lente | Livello termico locale | D, oppure A se registrate durante la prova | GT-EM2C, GT-VT, GT-DelftJSC, LF-Delft | Risposta e correzioni dipendono da sito e montaggio; per un uso dinamico servono sequenze di frame |
| (c) Termocoppie compensate, in sincronia con una velocimetria | Fluttuazioni filtrate della stessa realizzazione | A | LV-Heitor, LV-Ferrão (LDV, AB) | Trasferibilità alla PIV non documentata; effetti della semina |
| (d) Statistiche puntuali o lungo linee da Raman/CARS, in un'altra campagna | Medie e PDF della condizione | D | GT-DLR, LF-TNF | Informazione sulla condizione, non sulla realizzazione osservata |
| (e) T planare simultanea su un campo piccolo | Campo istantaneo locale | A | LF-OSU | Raro nei casi; combustibile su misura |
| (f) Isoterma binaria dalle particelle PIV | Sopra o sotto una soglia | A | LF-isoterma | Soglia e semina |
| (g) Gas fresco sotto ~1000 K da particelle fosforescenti | Ingresso e mescolamento | A | LF-Yin-TPIV | Nei casi copre solo il gas fresco |

**2. Informazione diversa, target aperto.**

- (b) e (d) descrivono la condizione o un'altra realizzazione.
- (a) descrive il contorno.
- (c), (e), (f) e (g) descrivono la stessa realizzazione osservata.

Usare statistiche non implica automaticamente un cambio del target della ricostruzione. Come impiegare queste informazioni (vincoli, correzioni del livello termico, valutazione) è una questione metodologica da discutere.

**3. Risposta temporale.** Va trattata come proprietà del sito, del montaggio e dell'eventuale compensazione, non della "termocoppia" in generale. Esempi: ~1,5 s in GT-VT; filo fine compensato negli studi con LDV.

**4. Riferimenti sperimentali.** Nel materiale esaminato **non ho identificato** un riferimento sperimentale che soddisfi insieme tutte queste condizioni:

- (i) temperatura del gas misurata;
- (ii) velocità misurata nella stessa realizzazione;
- (iii) in un combustore swirl confinato o in un apparato confrontabile con RealPDEBench;
- (iv) dati accessibili.

Questa ricerca non è stata sistematica sui dataset, che appartengono allo step 2; le piste sono al §14.2. Non traggo conclusioni sul tipo di validazione possibile.

---

## 14. Indicazioni per lo step 2 (proposte da discutere)

### 14.1 Agente A — caratteristiche e problemi dei dati termici

Candidati, in un ordine ancora da concordare:

1. **Termocoppie nel gas, in zone fredde e calde.** Temi:
   - correzioni radiativa e conduttiva;
   - coefficiente h ricavato dalla velocità misurata (Delft, PoliMi);
   - risposta e isteresi (GT-VT);
   - compensazione e interazione con la semina (Heitor; Ferrão & Heitor; Tagawa);
   - perturbazione dovuta alla sonda.

   Fonti: Shaddix 2017 (TI), rapporto VT (TS), Delft (TS). Da recuperare: Heitor & Moreira 1993 e i lavori LDV + termocoppia (AB).
2. **Fosfori su superficie.** Calibrazione, copertura parziale, dipendenza dalla condizione operativa.
3. **Raman, CARS e FRS** come statistiche o campi su piccola area. Incertezze, zone escluse.
4. **Configurazioni simultanee** (FRS + PIV, PIV termografica) e loro limiti.

### 14.2 Agente B — dataset

Piste incontrate, non ispezionate:

- archivi TNF: Sandia D–F, CSSB con temperature del bluff body, TSF/SWQ, LSB;
- dati di contorno DLR GTMC/PRECCINSTA;
- ISF-3;
- Deep Blue TCC-III;
- TUdatalib Darmstadt;
- Cardiff research data;
- Adelaide JHC;
- tesi McManus (FRS + PIV);
- dati Sydney sul bluff body;
- EngineBench.

**Nuove dalla revisione:** rapporto VT (OSTI 1463254; i dati non sono stati verificati); dataset TU Delft dei combustori a getti e JHC (disponibilità [M]); PoliMi.

Per ogni pista: T e velocità alla stessa condizione? Nella stessa realizzazione? Con quale supporto?

### 14.3 Con la missione 01

- Piano, orientamento ed estensione della sezione RealPDEBench.
- Utilità di chiedere condizioni termiche e geometria agli autori. Decide Enrico; nessun contatto da parte mia.

---

## 15. Lacune, copertura e domande

### 15.1 Copertura

Letto in TI o TS (dettagli e livelli nel [registro](step1_fonti_e_ricerche.md)):

- DLR: GTMC, SFB606, PRECCINSTA, combustore fuligginoso, HBK;
- EM2C, KIT, ETH;
- **VT e Delft (due apparati)**;
- OSU, Lund, Yin, SWQ, TNF;
- TCC-III, Darmstadt, Sandia;
- Shaddix, Frank;
- LES con dichiarazioni sulle condizioni termiche;
- RealPDEBench.

### 15.2 Lacune [M]

- **Controesempi noti solo da abstract:** PoliMi, UMD, XJTU, Gutmark, Kwak, Guo, Hsu; i lavori LDV + termocoppia (Heitor, Ferrão, Yanagi, Takagi, LaRue). Host bloccati: re.public.polimi.it, erpublication, cambridge.org, sciencedirect, escholarship, DRUM, Spiral, ResearchGate, CORE.
- **Rig ad ammoniaca e rig SJTU:** quasi solo abstract.
- **Originali mancanti:** DLR 2006–07 ed EM2C 2015.
- **MIT:** non verificato.
- **Review non lette:** Hochgreb 2019, Barlow 2007, Heitor & Moreira 1993, Brübach 2013, Abram 2018.
- **Pratica informale** (forum, discussioni fra sperimentatori): non raccolta.
- **Ricerca dei controesempi:** mirata, non esaustiva. Una termocoppia nel gas caldo sincronizzata con la PIV potrebbe esistere in letteratura non accessibile o non intercettata dalle query.

### 15.3 Domande per la discussione prima dello step 2

1. Quale tipo di informazione termica del §1.3 (A, B, C, D) vogliamo rappresentare, eventualmente più di uno? Il target resta aperto.
2. Vincoliamo i sensori alle configurazioni documentate (§13 con i relativi casi), oppure ammettiamo configurazioni non documentate dichiarandole come tali?
3. Quali configurazioni affidare all'agente A?
4. Chiedere agli autori di RealPDEBench le condizioni termiche e la geometria? Decide Enrico.
5. Recuperare tramite biblioteca i testi bloccati? Priorità per la questione centrale:
   - PoliMi: Olivani 2007; Solero 2017;
   - UMD: Archer 2005;
   - Heitor 1985 e Ferrão & Heitor 1998;
   - rig SJTU: Liu 2024.

### 15.4 Stato della consegna e branch

- **Branch:** `claude/thermal-measurement-scouting-rxj1l0`, imposto dall'ambiente, partito dalla base `a894257497de1524566186c22b66e02bd7b7b8ee` (la stessa di `mission04/step1-thermal-scouting`).
- **Commit:** v1 = `5d566be`; questa v2 è committata sullo stesso branch.
- **Non toccati:** codice, risultati storici, roadmap, altre missioni, `documents_thesis`.
- **Lo step 1 non è segnato come approvato.**
