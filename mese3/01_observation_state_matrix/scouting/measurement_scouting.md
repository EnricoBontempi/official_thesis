# Scouting delle misure in fiamma — prima mappa (F2)

**Data:** 26 novembre 2026  
**Missione:** [Observation–State Matrix](../README.md) · [roadmap01.md](../roadmap01.md) · [templates_abc.md](../templates_abc.md)  
**Incarico:** `measurements-scout` (ID `A-measurements-scout-NNN`, `C-measurements-scout-NNN`, fonti `S01…`)  
**Branch:** `matrix/f2-measurement-scouting`, derivato da `plan/month3-missions` @ `30d11e0`  
**Perimetro:** sola ricognizione bibliografica. Nessun esperimento, training, inferenza o download di dataset; nessun contatto esterno. Non contiene le schede dei due paper di Jason, assegnate a un altro incarico.

**Stato del lavoro.** È una **prima mappa**, non un'enciclopedia. Ho letto direttamente i testi integrali open access di 12 fonti, in tutto o in sezioni dichiarate. Le altre famiglie sono coperte da abstract o fonti secondarie e sono marcate come tali (§9). Riviste come IOP, Elsevier e SAE non erano leggibili in full text da questo ambiente: per quelle fonti nessun dettaglio metodologico viene ricostruito dall'abstract.

**Legenda (da [templates_abc.md](../templates_abc.md)).**
- **D:** documentato, con fonte e localizzatore. Il tipo di lettura (testo integrale, sezione, abstract o fonte secondaria) è nel registro §9.
- **I:** inferenza mia, con il ragionamento esplicito.
- **M:** mancante o non verificato, con la domanda aperta.

Ogni numero vale **per l'apparato e le condizioni indicate**. Non è una specifica generale della tecnica.

---

## 1. Sintesi comprensibile

1. **La grandezza raramente coincide con il segnale.** Il segnale va convertito in grandezza con ipotesi esplicite:
   - la PIV registra immagini di particelle, cioè la velocità delle particelle, non del gas;
   - la termocoppia registra la temperatura del giunto, non del gas;
   - il Rayleigh registra intensità diffusa, proporzionale a densità × sezione d'urto efficace;
   - la chemiluminescenza registra fotoni integrati lungo la linea di vista.

   Queste ipotesi sono la principale fonte di **bias dipendente dal contesto**, non il rumore del sensore (S01, S03, S04, S08).
2. **Quattro difetti distinti, che non si riducono a "rumore".**
   - **Errore sul valore.** Esempio: ±0.1 px sul picco di correlazione PIV, che diventa ±0.5–1.2 m/s a seconda della fiamma (S02).
   - **Filtraggio spaziale.** Esempi:
     - vettore PIV mediato su 0.94 mm e spaziato di 0.47 mm (S01);
     - tomo-PIV con interrogazione da 393 µm e spaziatura da 98.4 µm (S12).

     La spaziatura dei dati **non** è la risoluzione.
   - **Risposta temporale.** Esempio: costante di tempo di 57 ± 3 ms di una termocoppia da 50 µm in un pool fire. La correzione inerziale non sposta quasi la media ma cambia la deviazione standard da 237 K a 385 K (S04).
   - **Assenza di misura.**
     - Regioni senza OH non danno temperatura con il two-line OH-PLIF (S16).
     - Campo di vista PIV ridotto a 30 × 30 mm² su una camera da 85 × 85 mm² (S02).
     - Record limitati a circa 0.8 s dalla memoria delle camere (S01, S02).
3. **Supporto spaziale come asse primario.** Le misure di riferimento sono di cinque tipi:
   - **punti:** termocoppie, sonde;
   - **linee 1D:** Raman/Rayleigh, con risoluzione intorno a 0.1 mm (S03);
   - **piani:** PIV/PLIF;
   - **proiezioni:** chemiluminescenza, assorbimento, pirometria;
   - **volumi:** tomo-PIV e tomografie, in genere su piccoli volumi o con molte camere.

   Una sezione o una proiezione non è un volume. Una linea 1D dà il gradiente solo lungo la linea, salvo orientazione 3D misurata a parte (S03).
4. **Le misure più accurate di temperatura e composizione sono le meno disponibili.** Raman/Rayleigh line imaging fornisce T, specie maggiori ed equivalence ratio a singolo colpo:
   - in fiamma premiscelata a T = 2185 K, precisione σ = 0.75 % e accuratezza 2 % su T (S03, Tab. 2);
   - ma su una linea, con laser da circa 1.8 J/impulso e un laboratorio dedicato.

   PIV/OH-PLIF ad alta frequenza esiste in combustori modello (S01, S02), ma OH-PLIF vi è usato come **marcatore qualitativo** della zona di reazione, non come concentrazione.
5. **Grandezze derivate: la tecnica "naturale" è spesso inadeguata localmente.**
   - Il rilascio termico (HRR) da chemiluminescenza OH* è affidabile in forma **integrale**, non locale, nelle fiamme turbolente studiate da Lauer (S08). Altre fonti, lette solo in forma secondaria, lo ritengono localmente utile (S17): **contraddizione conservata** (§5, A-008).
   - L'HRR ricostruito da Raman/Rayleigh con un reattore omogeneo ha, nel solo test su DNS, errore relativo medio locale del 70.8 % senza OH e del 14.2 % con OH (S07).
6. **Le misure intrusive perturbano la grandezza.** Esempi:
   - campionamento MBMS: perturbazioni termiche fino a 500 K e spostamento del punto di prelievo di qualche diametro dell'orifizio (S09);
   - termocoppie: correzione radiativa che dipende dal modello di scambio adottato, con fonti in disaccordo sul ruolo della conduzione nei fili (S04 contro S05).
7. **La correzione di una misura può usare un modello o una simulazione.**
   - NIST usa velocità FDS simulate per correggere le termocoppie (S04).
   - La LAS è confrontata con misure **sintetiche** calcolate dalla CFD (S10).

   Per la matrice C significa che misura, modello e riferimento possono non essere indipendenti.
8. **Regime e trattamento numerico vanno annotati a parte**, come richiesto dalla roadmap. Esempi letti:
   - DNS ad alto Karlovitz per valutare un'inversione (S07);
   - LES con assimilazione EnKF di tomo-PIV (S12);
   - "CFD" non meglio specificata per misure sintetiche (S10).

   Nessuna fonte letta valida fluttuazioni istantanee con un riferimento RANS medio.
9. **Opportunità che emergono come problemi C** (§6):
   - ricostruzione delle fluttuazioni di T da sensori lenti;
   - HRR locale da osservabili indiretti;
   - inversione di misure line-of-sight con modelli di misura sintetici;
   - assimilazione statistica di misure sparse in simulazioni.

   Sono proposte, non scelte.

---

## 2. Distinzioni adottate

| Asse | Valori usati in questo rapporto |
|---|---|
| Grandezza (fisica) | Temperatura, velocità e componenti, pressione, composizione/specie, particolato, grandezze derivate (HRR, gradienti, dissipazione, fronte di fiamma, frequenze) |
| Tecnica | Strumento più elaborazione. La **PIV è una tecnica**, la velocità è la grandezza |
| Segnale registrato | Ciò che il sensore restituisce prima della conversione (immagini, carica, tensione, spettri, trasmissione) |
| Supporto | Punto / linea / piano / proiezione (integrale di linea) / volume; componenti osservate |
| Imperfezioni | **E** errore sul valore (casuale, bias, correlato); **F** filtraggio spaziale (risoluzione finita); **R** risposta e integrazione temporale; **L** lacune (assenza di osservazione, rigetto, campo di vista, durata) |
| Regime fisico | Laminare/turbolento; premiscelato/non premiscelato/parzialmente premiscelato/stratificato; pressione |
| Trattamento numerico (se c'è una simulazione) | RANS/URANS/LES/DNS/modello ridotto; dichiarato o **M** |

---

## 3. Tabella sintetica di famiglie e tecniche

Visione d'insieme. Le colonne "Tempo" e "Limite" riportano solo quanto documentato nella fonte citata o, se marcato, inferito. "–" significa non verificato in questo giro. L'ultima colonna rimanda alla scheda A, quando esiste.

| Famiglia | Grandezza | Tecnica | Segnale registrato | Supporto | Tempo (esempio documentato) | Limite principale (tipo) | Fonte / stato | Scheda |
|---|---|---|---|---|---|---|---|---|
| Temperatura | T gas | Termocoppia fine-wire | Tensione (T del giunto) | Punto | 60 Hz; τ ≈ 57 ms nel pool fire NIST | Radiazione e inerzia (E, R); intrusiva | S04, S05 D | 001 |
| Temperatura | T, specie maggiori, φ | Raman/Rayleigh (+CO-LIF) line imaging | Spettri Raman, intensità Rayleigh, LIF | Linea 1D (~0.1 mm) | Singolo colpo; frequenza M | Solo 1D; apparato pesante; interferenze (E, F) | S03 D | 002 |
| Temperatura | T | Rayleigh planare / filtered Rayleigh | Intensità diffusa (∝ n·σ_eff) | Piano | – | Sezione d'urto dipendente dalla composizione (E-bias) | S25 sec. | 003 (breve) |
| Temperatura | T | Two-line OH-PLIF (phase-locked) | Rapporto di due PLIF OH | Piano, solo dove c'è OH | Fase-mediato | Nessuna T senza OH (L); 4–7 % | S16 sec. | 004 (breve) |
| Temperatura | T | CARS (ns, fs/ps) | Spettro coerente | Punto/linea | kHz per fs/ps (secondario) | Punto; apparato complesso | S29 sec. | – |
| Temperatura | T, H2O, CO, CO2, P | Assorbimento laser (TDLAS/LAS) | Trasmissione spettrale | Linea di vista (integrale) | 500 kHz–1 MHz (fireball) | Media lungo il percorso (F, E-bias) | S10 D; S11 metadati | 005 |
| Temperatura | T gas + velocità | Fosfori termografici in aerosol | Fosforescenza + scattering Mie | Piano | 3 kHz (abstract) | Precisione < 5 % a 500 K; regime non combustivo nell'esempio | S19 sec. | – |
| Temperatura (soot) | T soot, KL | Pirometria a due colori / DBI-TI | Emissione o estinzione integrate | Proiezione | – | Bias verso il soot caldo lungo la linea di vista | S14 abstract; S15 sec. | 011 |
| Temperatura/densità | n, T (dedotta) | BOS tomografico | Spostamento di uno sfondo (deflessione) | Volume da proiezioni | Istantaneo (23 camere) | T solo via ipotesi indice–composizione | S21 sec. | – |
| Velocità | u (2C/3C) | PIV planare / stereo ad alta frequenza | Coppie di immagini di particelle | Piano, 3C nello stereo | 5–10 kHz; record ≈ 0.8 s | Filtro di finestra; seeding; FOV (E, F, L) | S01, S02 D | 006 |
| Velocità | u (3C) | Tomo-PIV | Proiezioni di particelle (4 camere) | Volume piccolo | 10 kHz | Profondità limitata; ricostruzione (F, L) | S12 D (jet inerte); S20 sec. | 007 |
| Velocità | u (punto) | LDV | Burst Doppler | Punto | – | Statistiche puntuali, non campi | Rapporto preesistente S27 | – |
| Velocità | u in motore | PIV in motore ottico (TCC-III) | Immagini di particelle | Piano, fasi in CAD | Per ciclo o fase | Distorsione del cilindro; lacune | S27 (rapp. preesist.), S26 sec. | 012 |
| Pressione | p cilindro | Trasduttore piezoelettrico | Carica (relativa) | Punto a parete | Per grado di manovella | Thermal shock, pegging, fasatura (E) | S23 sec. | 009 |
| Pressione | p' acustica | Microfoni / tubo semi-infinito | Tensione | Punti a parete | kHz | Funzione di trasferimento della linea; ipotesi di propagazione | S24 sec. | 009 |
| Composizione | Specie stabili e radicali | Campionamento con sonda + MBMS | Correnti ioniche per m/z | Punto (profilo 1D in fiamma laminare) | Stazionario | Perturbazione della sonda (E-bias, posizione) | S09 D | 010 |
| Composizione | Specie stabili | Sonda + analizzatori dei fumi | Concentrazioni dopo il prelievo | Punto o uscita | Lento | Fuori dalla zona di reazione; perturbazione | Non letto | – |
| Composizione | OH (o CH2O, CH…) | PLIF | Fluorescenza (dipende da quenching e Boltzmann) | Piano | 5–10 kHz | Qualitativa senza correzioni; energia laser a kHz | S01, S02, S03 D | 008 |
| Ottica | OH*, CH*, CO2* | Chemiluminescenza | Intensità integrata lungo la vista | Proiezione (o deconvoluzione assisimmetrica) | Esposizione 40 µs (S02); kHz | Non locale; fondo CO2* (F, E-bias) | S08 D; S17 sec. | 013 |
| Ottica | Indice di rifrazione 3D | Tomografia della chemiluminescenza (CTC) | Molte proiezioni | Volume | Istantaneo | Numero e disposizione delle camere; validazione su fantasmi | S22 sec. | – |
| Particolato | f_v, dimensione primaria | LII (+ estinzione) | Incandescenza dopo un impulso laser | Piano | Impulso singolo | Calibrazione dipendente da E(m) | S13 D (§3.2.2) | 011 |
| Derivate | HRR locale | OH·CH2O PLIF; Raman/Rayleigh(+OH) + reattore | Più segnali combinati | Piano o linea | Singolo colpo | Indicatore, non misura (I, E) | S07 D | 014 |
| Derivate | Gradienti, orientazione del fronte | Raman/Rayleigh + OH-PLIF incrociati | Linea + due piani | Linea | Singolo colpo | Proiezione 1D del gradiente | S03 D | 002 |
| Derivate | Frequenze (PVC, instabilità), Da | PIV/PLIF time-resolved, pressione | Serie temporali | Piano/punto | 5–10 kHz | Da con ±20 % (S02) | S02 D | 006 |
| Derivate | HRR globale, frazione radiativa | Calorimetria O2, flussimetri | Composizione dei fumi, flusso radiante | Integrale | Medio | Solo globale | S04 D (contesto pool fire) | – |

---

## 4. Righe sintetiche A (formato template)

| ID | Grandezza e contesto | Tecnica e segnale | Spazio e copertura | Tempo | Errori e limiti | Disponibilità e fonti |
|---|---|---|---|---|---|---|
| A-measurements-scout-001 | T gas in pool fire di metanolo da 1 m (non premiscelato, con puffing) | Termocoppia di tipo S da 50 µm, giunto da circa 153 µm; T giunto → T gas con correzioni di radiazione e inerzia | Punto; 46 posizioni sull'asse e radiali | 60 Hz × 120 s; τ ≈ 57 ± 3 ms (a z = 30 cm) | Correzione radiativa fino a 55 K; correzione inerziale con media 0.2 K e std 293 K; incertezza combinata 5 % sulla media e 26 % sulla std | NIST TN 2083r1 (S04); Shaddix 2017 (S05) |
| A-measurements-scout-002 | T, specie maggiori, φ in fiamme CH4/aria turbolente premiscelate e stratificate (slot burner) | Raman/Rayleigh line imaging + CO-LIF a due fotoni; OH-PLIF su piani incrociati per l'orientazione | Linea 1D; superpixel di 0.104 mm (Raman); 3D dei gradienti solo con piani incrociati | Singolo colpo; gate Raman di 3.9 µs | A T = 2185 K: σ_T = 0.75 %, accuratezza 2 %; CO2 3.2 %/4 %; φ 2.2 %/5 %; bias di φ nei prodotti di circa +0.02 | Sandia TCL (S03) |
| A-measurements-scout-003 | T planare (Rayleigh, FRS) | Intensità di diffusione → densità → T con σ_eff | Piano | – | Bias dalla σ_eff dipendente dalla composizione; valori solo da fonti secondarie | S25 (sec.); M |
| A-measurements-scout-004 | T planare in combustore swirl con instabilità | Two-line OH-PLIF phase-locked | Piano, solo dove c'è OH | Media di fase | Accuratezza 4–7 % (abstract); nessuna T senza OH | S16 (sec.); M |
| A-measurements-scout-005 | T, P, frazioni molari (column density) nel fireball post-detonazione | LAS (QCL/ICL), fit spettrale | Una linea di vista, 83 mm sopra la carica | 1 MHz (CO), 500 kHz (CO2) | Media lungo il percorso; T(CO) usata per CO2; P "only an estimate"; bande = range su 4 prove | S10; review S11 (non letta) |
| A-measurements-scout-006 | Velocità 3C + fronte OH in fiamme swirl CH4/aria a iniezione separata, «largely premixed» (DLR GTMC, 10–35 kW) | Stereo-PIV (TiO2) + OH-PLIF a 5 e 10 kHz | Piano; FOV PIV 30 × 30 mm² (5 kHz) o 40 × 40 mm² (10 kHz); risoluzione da 0.94 a 3 mm secondo la configurazione | 5 o 10 kHz; 0.8 s per record | Casuale ±0.5/0.8/1.2 m/s, solo da ±0.1 px; bias non quantificati | S01, S02 |
| A-measurements-scout-007 | Velocità 3C volumetrica (jet inerte Re = 13 500; in fiamma solo secondario) | Tomo-PIV a 4 camere, MART | Volume 12.3 × 2.5 × 16.5 mm³; interrogazione da 393 µm, spaziatura da 98.4 µm | 10 kHz; sequenza di 50 ms usata | Vettori spuri sostituiti e filtro spaziale (dati già regolarizzati) | S12 (jet); S20 (fiamma, sec.) |
| A-measurements-scout-008 | OH come marcatore della zona di reazione o del gas combusto | OH-PLIF (A-X (1,0), Q1(7)) a kHz | Piano; foglio da circa 0.4 mm; altezza del foglio circa 40 mm | 5 kHz; gate di 500 ns | Qualitativa; senza correzione di quenching e Boltzmann non è una concentrazione | S01, S02, S07 |
| A-measurements-scout-009 | Pressione nel cilindro e pressione dinamica in combustore | Piezoelettrico + pegging; microfoni / linea semi-infinita | Punti a parete | Per grado di manovella; kHz | Thermal shock (IMEP > 10 % in un caso), pegging, funzione di trasferimento della linea | S23, S24 (sec.); M |
| A-measurements-scout-010 | Specie stabili e radicali in fiamme laminari premiscelate a bassa pressione | Sonda di quarzo + MBMS (EI o fotoionizzazione) | Profilo 1D lungo l'asse | Stazionario | Perturbazione termica fino a 500 K; prelievo qualche diametro a monte; sezioni d'urto di ionizzazione incerte | S09 |
| A-measurements-scout-011 | Soot: f_v, dimensioni, T soot | LII (+ estinzione); pirometria 2C; DBI-TI | Piano (LII); proiezione (pirometria, DBI) | Impulso singolo; per ciclo motore | Calibrazione dipendente da E(m); bias LOS verso il soot caldo | S13 (sez.); S14 (abs.); S15 (sec.) |
| A-measurements-scout-012 | Velocità in motore ottico (TCC-III) con pressione | PIV in motore + pressione nel cilindro + immagini OH* | Piani; griglia di vettori; lacune | Per fase o ciclo | Lacune nel dato rilasciato codificate come 0; errore 1.5–8 % (secondario) | S27 (rapp. preesistente), S26 (sec.) |
| A-measurements-scout-013 | HRR (integrale/locale) da OH* | Chemiluminescenza OH* (+ CH*), correzione con modello di strain | Proiezione; deconvoluzione solo per medie assisimmetriche | Medie su 2048 coppie di immagini | Integrale sì, locale no (S08); contraddizione con S17 | S08; S17 (sec.) |
| A-measurements-scout-014 | HRR locale da Raman/Rayleigh (+OH) via reattore omogeneo | Inversione fisica: T + 7 specie (+OH) → stato completo → HRR | Linea 1D (in pratica) | Singolo colpo | Errore locale medio 70.8 % senza OH e 14.2 % con OH (DNS); validazione solo su DNS | S07 |

---

## 5. Schede A

### A-measurements-scout-001 — Termocoppia fine-wire in un pool fire di metanolo (NIST)

**Sintesi.** Misura puntuale, economica e diffusissima, ma la grandezza registrata è la temperatura del giunto. Nel pool fire NIST la correzione radiativa sposta la media di decine di K. La correzione inerziale lascia la media quasi invariata ma **cambia molto la statistica delle fluttuazioni**. Esempio concreto di come la **risposta temporale** sia un asse separato dall'errore sul valore.

| Campo | Contenuto |
|---|---|
| Grandezza e contesto | T del gas (K) in un pool fire di metanolo da 1 m, ben ventilato, in ambiente quiescente. Massima T media 1371 K a 30 cm sull'asse (S04, Abstract). Puffing a circa 1.37 Hz (S04, Abstract). **D**. Regime: fuoco non premiscelato, con fluttuazioni ampie (std 385 K alla posizione analizzata); classificazione turbolenta **I** |
| Tecnica e segnale | Termocoppia di tipo S (Pt 10 % Rh/Pt), filo nudo da 50 µm, giunto quasi sferico da circa 153.3 µm (S04, §2.1, pp. 3–4). Segnale: f.e.m., cioè T del giunto T_b. T_gas = T_b + T_rad + T_the: bilancio energetico sul giunto con convezione e radiazione; conduzione nei fili trascurata per l'analisi in App. G (S04, §2.1.1, p. 5). dT_b/dt stimata con un fit polinomiale di 2° ordine su tre punti consecutivi (S04, §2.1.1). **D** |
| Spazio | Punto. 46 posizioni, ogni 10 cm in direzione radiale (da −10 a 50 cm) e ogni 5–10 cm in assiale (da 5 a 210 cm) (S04, §2.1, p. 3). Supporto fisico del giunto circa 0.15 mm. Shaddix (S05, Fig. 6 e conclusioni) mostra però che, tramite la conduzione nei fili, la lettura è sensibile a variazioni di T del gas «over a spatial dimension that is much larger than the thermocouple bead» (alcuni mm). **D** |
| Copertura e validità | Misure ripetute 2–3 volte per posizione (S04, §2.1). Nessuna lacuna dichiarata; intrusività e durabilità sono il compromesso esplicito: «The finer the wire … the faster the measurement time response, but the more fragile» (S04, §2.1). **D** |
| Tempo | Acquisizione a 60 Hz per 120 s (720 campioni, circa 170 cicli di puffing) (S04, §2.1, p. 4). Costante di tempo media **57 ± 3 ms** a (z, r) = (30 cm, 0) (S04, §3.7, Fig. 16c, p. 20); circa inversamente proporzionale alla T media (S04, Fig. 17, p. 21). Gli estremi corretti sono anticipati nel tempo rispetto a T_b (S04, p. 19). **D** |
| Errori e limiti | Alla posizione (30 cm, 0), test 3 (S04, Tab. 5, p. 21):<br>• media: T_g 1337 K, T_b 1315 K, T_rad 22 K, T_the 0.2 K;<br>• deviazione standard: T_g 385 K, T_b 237 K, T_the 293 K.<br>Correzione radiativa massima 55 K a T_b = 1694 K (S04, p. 19). Incertezza combinata media: **5 % sulla media e 26 % sulla deviazione standard** (S04, Abstract). u_c(T_the) domina per T_g < 700 K o > 2000 K (S04, p. 21). La correzione usa la **velocità del gas da simulazione FDS** come ingresso (S04, Abstract e §2.1.1): la misura corretta dipende da un modello. **D** |
| Qualificazione | Propagazione degli errori (App. E.1); codice verificato su una sinusoide e validato su FDS (App. D). Nessun confronto con un termometro ottico simultaneo nelle parti lette. **D/M** |
| Disponibilità pratica | Molto ampia (laboratori, forni, caldaie: S05, Introduzione). Limitata dove la sonda perturba o non sopravvive. **D (generale), I per i casi specifici** |
| Fonti e aperti | **Contraddizione:** S04 trascura la conduzione nei fili (App. G); S05 sostiene che la correzione radiativa corretta è «substantially larger» di quella che ignora la conduzione. S05 lo mostra con calcoli su un filo da 101.6 µm in prodotti stechiometrici a 1 m/s, emissività 0.25 (S05, Fig. 3–5); i valori numerici delle figure non sono stati estratti. Condizioni e fili diversi: va conservato. **M:** effetto in fiamme premiscelate swirl ad alta velocità; ruolo della catalisi superficiale (citata nella letteratura, non verificata qui). |
| Collegamenti | C-measurements-scout-001; missione [04_thermal_fusion](../../04_thermal_fusion/README.md) |

### A-measurements-scout-002 — Raman/Rayleigh/CO-LIF line imaging con OH-PLIF incrociato (Sandia, slot burner)

**Sintesi.** Lo stato dell'arte per T, specie maggiori ed equivalence ratio **simultanei a singolo colpo**, con risoluzione intorno a 0.1 mm. Il prezzo:
- il supporto è una **linea**;
- l'apparato è un laboratorio dedicato;
- i gradienti 3D richiedono due piani OH aggiuntivi.

È il tipo di misura che fa da riferimento in fiamme da laboratorio (archivio TNF), non da sensore in un combustore applicativo.

| Campo | Contenuto |
|---|---|
| Grandezza e contesto | T (K), frazioni molari di CO2, O2, CO, N2, CH4, H2O e H2, equivalence ratio locale; fiamme CH4/aria turbolente premiscelate e stratificate su uno slot burner di Cambridge (S03, §2.1 e §2.2). **D** |
| Tecnica e segnale | Quattro Nd:YAG duplicati, 1.8 J/impulso totali nel volume di misura, impulsi allungati per evitare il breakdown; fuoco di circa 0.22 mm (1/e²). CO-LIF a due fotoni a 230.1 nm (S03, §2.2, p. 5). Segnali: spettro Raman (~550–700 nm), Rayleigh (532 nm), fluorescenza CO (~480–488 nm). Gate con chopper: 300 µs e 3.9 µs FWHM per il Raman (S03, p. 6). T dal Rayleigh con la composizione dal Raman: **I** sulla catena esatta, non descritta nelle parti lette. **D/I** |
| Spazio | Linea 1D. Superpixel lungo il laser di 0.104 mm (Raman), 0.020 mm (Rayleigh) e 0.101 mm (CO-LIF); quasi tutti i raggi entro un raggio di 0.075 mm dopo ingrandimento 2:1 (S03, p. 6). Orientazione 3D del fronte da due piani OH incrociati sulla linea, area di 10 × 8 mm per piano, pixel di 0.150 mm (S03, pp. 6–7). Il gradiente lungo la linea è una proiezione: un caso con normale a 9° dalla linea dà un «cosine error of 1%» (S03, §3.2). **D** |
| Copertura e validità | Profili mediati a intervalli di 3 mm mostrano artefatti dovuti a throughput ottico non completamente corretto (S03, §3.1). Criteri di rigetto: **M** (non trovati nelle parti lette). |
| Tempo | Singolo colpo, impulsi di ns allungati. Frequenza di ripetizione **M** nelle parti lette (I: tipicamente 10 Hz per Nd:YAG flash-lamp, da verificare). Nessuna sequenza temporale. |
| Errori e limiti | Tab. 2 (S03, p. 16), fiamma premiscelata φ = 0.97, T = 2185 K:<br>• T: precisione σ 0.75 %, accuratezza 2 %;<br>• N2: 0.7 %, 2 %;<br>• CO2: 3.2 %, 4 %;<br>• H2O: 2.4 %, 3 %;<br>• φ: 2.2 %, 5 %.<br>Precisione = deviazione standard; accuratezza stimata da incertezza delle condizioni di calibrazione e ripetibilità (S03, p. 7). In turbolenza: φ nei prodotti troppo alta di circa 0.02 (0.73 nominale); fluttuazioni di φ < 2 % nel bruciato e circa 3 % nel flame brush (S03, §3.1). H2 troppo alto, calibrazione H2 incerta di circa 10 % (S03, §3.2). Possibili errori da media spaziale su gradienti ripidi (> 2000 K/mm nel premiscelato), trattati nella §3.2. **D** |
| Qualificazione | Calibrazione su flussi freddi e riscaldati, fiamme laminari di diffusione e fiamme piatte a composizione nota; mass flow controller entro l'1 % (S03, p. 7). Profilo T e gradiente confrontati con un calcolo di fiamma laminare (S03, §3.2, Fig. 4). **D** |
| Disponibilità pratica | Sandia Turbulent Combustion Laboratory; dati di questo tipo alimentano l'archivio TNF (S27, rapporto preesistente). Non trasferibile a combustori chiusi senza ampio accesso ottico (I). |
| Fonti e aperti | **S03 è la versione sottomessa** (SAND2007-8048C, con "Note to reviewers"); la versione pubblicata (PCI 32, 2009) può differire. **M:** frequenza di ripetizione; criteri di rigetto; incertezza in fiamme sootanti o ad alta pressione. |
| Collegamenti | A-014 (uso derivato per HRR); C-measurements-scout-002 |

### A-measurements-scout-003 — Rayleigh planare / filtered Rayleigh (scheda breve, solo fonti secondarie)

- **D (secondario, S25):** la T si ricava da un'intensità proporzionale a densità × sezione d'urto efficace, media delle sezioni dei componenti pesata con le frazioni molari. Le variazioni spaziali di σ_eff producono **bias**.
- **D (secondario, S25):** in un caso CH4/N2/O2 a φ = 0.78, a 355 nm, è riportata un'incertezza del 2–8 % da σ_eff contro l'1.4 % da energia laser. In un altro studio su fiamme laminari ad alta pressione è dichiarata un'incertezza relativa < 10 % a singolo colpo con correzione via Cantera.
- **Non** verificati sulle fonti primarie. Doll et al. 2012 (DLR, FRS a scansione di frequenza in combustori; elib.dlr.de/77091) è stato scaricato ma non letto: **M**.
- **I:** in fiamme non premiscelate il bias da composizione è più critico che nel premiscelato, dove σ_eff varia meno. È il motivo per cui S03 combina Rayleigh e Raman.

### A-measurements-scout-004 — Two-line OH-PLIF thermometry phase-locked (scheda breve)

- **D (secondario, abstract via motore di ricerca, S16):** applicata a una fiamma swirl CH4/aria in un GT model combustor con instabilità autoeccitata. Accuratezza «4%-7% depending on the location within the flame and on the temperature level». «No temperature information can be obtained from regions where OH radicals are missing or present in insufficient concentration.»
- **Lezione per la matrice:** **lacuna correlata con lo stato.** T è disponibile solo nel gas caldo dove c'è OH, quindi la maschera di validità porta informazione sul target (vedi rischio analogo in [00_ricognizione_osservazioni.md](../../../docs/nonideal/00_ricognizione_osservazioni.md), §2 punto 5).
- **M:** risoluzione, numero di immagini per fase, effetti di assorbimento laser; full text non letto.

### A-measurements-scout-005 — Assorbimento laser (LAS) line-of-sight ad alta frequenza, con misure sintetiche

**Sintesi.** Misure di T, P e specie fino a 1 MHz con un solo fascio: è l'esempio più rapido di questo giro. Il dato è però **integrale lungo il percorso**. Gli autori lo confrontano con misure **sintetiche** calcolate integrando la CFD lungo la stessa linea di vista: un modello di osservazione esplicito.

| Campo | Contenuto |
|---|---|
| Grandezza e contesto | T, P, prodotti colonna p_CO·L, p_CO2·L e X·L nel fireball post-detonazione di cariche emisferiche di PETN da 12 mm (afterburning turbolento in aria) (S10, Abstract). **Non è una fiamma da bruciatore:** analogo metodologico. **D** |
| Tecnica e segnale | QCL scansionato su transizioni CO vicino a 2008.5 cm⁻¹ a 1 MHz; ICL su CO2 vicino a 2394.8 cm⁻¹ a 500 kHz; fasci combinati su un solo percorso (S10, Abstract). Segnale: intensità trasmessa, cioè spettro di assorbanza, fittato con un modello spettroscopico. La T dal CO è usata nel fit del CO2 «since temperature cannot be reliably inferred from the measured CO2 spectra» (S10, p. 074701-8). **D** |
| Spazio | Una linea di vista circa 83 mm sopra la carica (S10, Abstract). Nessuna risoluzione lungo il percorso: si ottengono T «path-averaged» e grandezze colonna. **D** |
| Copertura e validità | Un solo percorso. Per CO e CO2 la T media sul percorso può differire, ma secondo la CFD non oltre 200 K (S10, p. 074701-8). La pressione inferita è «only an estimate», perché il modello assume che ogni specie assorbente sia allargata solo da collisioni con sé stessa o con la pseudo-specie aria (S10, p. 074701-8). **D** |
| Tempo | 1 MHz (CO) e 500 kHz (CO2); schlieren simultaneo a 500 kHz (S10, Abstract). Incertezza di circa 2.5 µs nell'allineamento dello zero dei tempi tra esperimento e CFD (S10, p. 074701-10). **D** |
| Errori e limiti | Bande sperimentali = range su quattro prove. Bande sintetiche = variazione della posizione azimutale della linea di vista (struttura turbolenta) e incertezza delle linestrength, in Tab. II come deviazione standard normalizzata (S10, p. 074701-10/11). Valori della Tab. II: **M** (non estratti). **D** |
| Qualificazione | Confronto con misure sintetiche da CFD; accordo «reasonable» a tempi brevi e disaccordo attribuito a shock riflesso e getto della cavità (S10, Abstract). Qualifica **congiuntamente** misura e modello. **D** |
| Disponibilità pratica | La review S11 (Goldenstein et al., PECS 2017) copre applicazioni in motori, combustori e altre geometrie: **non letta**, solo metadati. Il vantaggio pratico (piccoli accessi ottici) è **I**, da verificare su S11. |
| Fonti e aperti | Trattamento numerico della CFD di S10: **M**. Estensione tomografica (molti fasci): review Cai & Kaminski 2017 (DOI 10.1016/j.pecs.2016.11.002), non letta. |
| Collegamenti | C-measurements-scout-003 |

### A-measurements-scout-006 — Stereo-PIV + OH-PLIF a 5–10 kHz nel DLR Gas Turbine Model Combustor

**Sintesi.** Il riferimento più vicino a un "campo di velocità sperimentale in fiamma swirl" letto in questo giro. Fornisce tre componenti in un piano, ad alta frequenza, sincronizzate con un marcatore OH. Le sole incertezze dichiarate sono casuali e derivano dalla precisione del picco di correlazione. Seeding, moto fuori piano, effetti della fiamma e bias **non** sono quantificati nelle parti lette.

| Campo | Contenuto |
|---|---|
| Grandezza e contesto | Velocità 3C (m/s) nel piano centrale di un combustore swirl CH4/aria a pressione atmosferica, camera da 85 × 85 × 114 mm con pareti in quarzo:<br>• S01: 10 kW, φ = 0.75, 281 g/min di aria e 12.3 g/min di CH4 (S01, «Burners»);<br>• S02: tre condizioni con P = 10/20/35 kW, Re = 15 000–52 500 e f_PVC = 500/1000/1750 Hz (S02, §2, Tab. 1).<br>Regime: combustibile e aria iniettati separatamente, ma con fiamme «largely premixed» per il forte mescolamento prima della base (S02, §3.1, da misure Raman citate); in letteratura il caso è spesso detto parzialmente premiscelato (**I**, titolo di Boxx et al. 2010 visto solo nei risultati di ricerca). **D/I** |
| Tecnica e segnale | PIV con Nd:YAG a doppia cavità, 2.6 mJ/impulso, Δt = 20–30 µs, particelle di TiO2, due CMOS in stereo con Scheimpflug, integrazione di 100 µs. Elaborazione con DaVis 7.2 multi-pass, finestra finale 16 × 16 px e overlap 50 % (S01, «PIV System»). **Solo l'aria è inseminata** (S01, «Burners»). OH-PLIF: dye laser a 283.2 nm, linea Q1(7) di A-X (1,0), circa 100 µJ/impulso a 5 kHz; intensificatore con gate di 500 ns (S01, «PLIF System»). **D** |
| Spazio | Piano 3C. Foglio PIV ≤ 0.7 mm, foglio PLIF circa 0.4 mm (S01; S02, §2). Risoluzione e campo di vista:<br>• S01: «spatial resolution of 0.94 mm and vector spacing of 0.47 mm» con finestra da 16 px;<br>• S02: «in-plane spatial resolution … 1.5 mm for 5 kHz PIV and 3 mm for 10 kHz PIV», FOV 30 × 30 mm² (5 kHz) e 40 × 40 mm² (10 kHz).<br>Il PLIF copre tutta la larghezza fino a y = 45 mm (S02, §2). **D** |
| Copertura e validità | PIV limitata a una regione vicino all'ugello, molto più piccola della camera. Frazione di vettori rigettati e regole di validazione: **M**. Lacune per seeding non uniforme: **I** (solo l'aria è inseminata; il gas combusto è meno denso). |
| Tempo | 5 kHz: memoria per 4096 doppie immagini, cioè 0.8 s (S01; S02); PLIF 1.09 s a 5 kHz (S01). 10 kHz: 8192 registrazioni (S02, §2). Impulso PLIF interlacciato tra i due impulsi PIV (S01). Chemiluminescenza OH registrata con esposizione di 40 µs (S02, §2). **D** |
| Errori e limiti | «Based on the ±0.1 pixel uncertainty of the cross-correlation peak-finding algorithm, the random uncertainty … is estimated as ±0.5 m/s, ±0.8 m/s and ±1.2 m/s for flames 1, 2 and 3» (S02, §2). Il Da derivato ha incertezza di ±20 % (S02, §4). Il beam steering in avanti può differire tra le due camere (S01). **D** |
| Qualificazione | Nessuna validazione indipendente della velocità nelle parti lette; mappatura PIV↔PLIF con un target 3D a doppio piano (S01). **D** |
| Disponibilità pratica | Apparato DLR documentato; accesso ai dati **M**. Il combustore è un caso molto studiato (S01 cita Weigand et al. 2006, Combust. Flame 144: LDV, Raman, PLIF; letto solo nei metadati). |
| Fonti e aperti | **Due valori di risoluzione a 5 kHz:** 0.94 mm (S01) e 1.5 mm (S02), da sistemi e configurazioni simili ma non identici. Non riconciliati; non scegliere un valore "tipico". **M:** frazione di vettori rigettati; effetto della dilatazione sul seeding; incertezza sistematica. |
| Collegamenti | Missione [02_virtual_piv](../../02_virtual_piv/README.md); rapporto preesistente §2 e §5 (S27); C-measurements-scout-004 |

### A-measurements-scout-007 — Tomo-PIV ad alta frequenza (jet inerte Sandia; esempio di dato già regolarizzato)

**Sintesi.** Tre componenti in un volume e ad alta frequenza, ma in un volume piccolo e in un jet **non reattivo**. È utile soprattutto per due punti:
- la **spaziatura dei vettori è sovracampionata** rispetto al volume di interrogazione;
- il dato rilasciato ha già i vettori spuri **sostituiti** e un **filtro spaziale** applicato.

Le lacune e il rumore originari sono quindi già stati trasformati.

| Campo | Contenuto |
|---|---|
| Grandezza e contesto | Velocità 3C in un jet d'aria (bruciatore della serie Sandia, D = 7.45 mm, U_b = 27.5 m/s, Re = 13 500, coflow a 0.9 m/s) (S12, §5.1). **Non reattivo.** **D** |
| Tecnica e segnale | Nd:YAG a doppia testa ad alta frequenza, 5 mJ/impulso, Δt = 10 µs; 4 CMOS a 20 kHz in frame straddling; particelle di Al2O3 da 0.3 µm. Elaborazione in quattro passi (S12, §5.1, pp. 14–15):<br>1. filtro gaussiano 3 × 3 px;<br>2. ricostruzione MART;<br>3. correlazione volumetrica 24³ voxel con overlap 75 %;<br>4. rimozione o sostituzione degli spuri e filtro penalized least squares.<br>**D** |
| Spazio | Volume 12.3 × 2.5 × 16.5 mm³ con 126 × 26 × 169 vettori a 98.4 µm, finestra (393 µm)³: «an over-sampled representation» (S12, p. 15). **D** |
| Tempo | 10 kHz; usata una sequenza di 50 ms (S12, pp. 14–15). **D** |
| Errori e limiti | Incertezze nelle ref. [20, 35] di S12: **M**. La matrice di covarianza degli errori è «assumed to be known» (S12, §5.2). **D/M** |
| Disponibilità e in fiamma | In fiamma laminare premiscelata (S20, secondario):<br>• profondità 1:1 con 8 CCD e laser da 200 mJ;<br>• ostacoli elencati: illuminazione non uniforme, seeding non uniforme, distorsioni ottiche, errori di calibrazione 3D, luminosità della fiamma;<br>• il rapporto profondità/campo di vista nelle misure in fiamma è tipicamente 1:10.<br>**D sec.** |
| Collegamenti | C-measurements-scout-004 (uso nell'assimilazione) |

### A-measurements-scout-008 — OH-PLIF come marcatore (non come concentrazione)

- **D (S01, Risultati):** nelle immagini del GTMC, le regioni senza OH rappresentano «gas at low and medium temperatures (T < 1500K)». OH in superequilibrio marca la zona di reazione. Uso **qualitativo e topologico**.
- **D (S01, «PLIF System»):** a 5 kHz l'energia di eccitazione è circa 100 µJ/impulso. A 10 Hz, nello stesso bruciatore, si usavano circa 2.5 mJ/impulso: oltre un ordine di grandezza in più.
- **D (S01):** il fondo di luminosità e il rumore della camera sono corretti con medie a laser spenti; la non uniformità del foglio con un'immagine media di acetone-LIF.
- **D (S07, §1):** combinando OH-LIF con Raman/Rayleigh è possibile correggere il segnale OH colpo per colpo per frazione di Boltzmann e quenching. Senza questa correzione, le immagini PLIF di OH·CH2O non sono una misura quantitativa di HRR.
- **Implicazione:** l'intensità OH-PLIF non va trattata come frazione molare di OH. Né OH numerico né OH-PLIF vanno equiparati a OH* (cfr. [proposal.md](../../../docs/proposal.md), §3).

### A-measurements-scout-009 — Pressione: cilindro motore e pressione dinamica in combustore (scheda breve, solo secondario)

- **D (secondario, abstract SAE via motore di ricerca, S23):**
  - il trasduttore piezoelettrico misura una pressione **relativa**: serve il *pegging*, cioè il riferimento assoluto;
  - il **thermal shock** è una deriva intraciclo dovuta al flusso termico sull'elemento sensibile; in uno studio SI può alterare l'IMEP di oltre il 10 %, con effetto massimo a basso regime, alto carico e anticipo elevato;
  - l'errore di pegging non cambia l'IMEP né la posizione del picco, ma altera durata della combustione ed esponenti politropici;
  - la variabilità termica persiste in tutto il ciclo, qualunque procedura di pegging si scelga.

  Numeri per motore e condizioni specifiche, **non trasferibili**.
- **D (secondario, S24):**
  - nei combustori, la pressione dinamica si misura spesso con microfoni a distanza su una linea semi-infinita ("infinite tube probe"), che richiede una funzione di trasferimento tra pressione interna e lettura;
  - il metodo multi-microfono per funzioni di trasferimento di fiamma assume propagazione senza perdite a velocità del suono costante tra microfoni e fiamma;
  - la stima da serie temporali finite induce incertezza su guadagno e fase.
- **I:** la pressione è l'osservabile più disponibile nei motori reali. Le grandezze derivate (HRR apparente, CA50, IMEP) sono **integrali sul volume**, non locali. Questo la rende il collegamento naturale con i dati RANS/URANS motore citati da Jason.
- **M:** full text non letti; nessun valore di incertezza primaria in questo giro.

### A-measurements-scout-010 — Composizione per campionamento con sonda + MBMS in fiamme laminari a bassa pressione

**Sintesi.** È l'unica famiglia che misura molti intermedi e radicali contemporaneamente. Funziona in fiamme **laminari, premiscelate, stabilizzate su bruciatore e a bassa pressione**, dove la sonda è tollerata meglio. Il prezzo è una perturbazione documentata di temperatura e posizione.

| Campo | Contenuto |
|---|---|
| Contesto | Fiamme premiscelate laminari a 10–100 Torr; zona di reazione di circa 3–10 mm, inversamente proporzionale alla pressione; profilo quasi 1D (S09, §2.1, pp. 7–8). **D** |
| Tecnica e segnale | Sonda di quarzo con orifizio da 250–500 µm; espansione a circa 10⁻⁴ Torr, poi skimmer a circa 10⁻⁶ Torr. Ionizzazione con elettroni, REMPI o fotoni (sincrotrone, VUV), poi correnti ioniche per m/z. Le frazioni molari richiedono sezioni d'urto di ionizzazione, spesso poco note per i radicali (S09, §2.2–2.3, pp. 9–17). **D** |
| Errori e limiti | Perturbazioni termiche indotte dalla sonda fino a 500 K, 150 K e circa 200 K in diverse fiamme (S09, p. 10). La sonda campiona «a few orifice diameters upstream of its physical position», con lo spostamento dei profili come correzione accettata. «A quantitative description of the nozzle effects on the flame structure remains currently unfeasible» (S09, pp. 10–11). **D** |
| Disponibilità | Laboratori di cinetica, spesso con luce di sincrotrone (S09). Non applicabile in turbolenza nello stesso modo (I). |
| Lezione | **Bias di posizione:** il valore è corretto, ma attribuito al punto sbagliato. È un tipo di errore che non si modella come rumore additivo sul valore. |

### A-measurements-scout-011 — Particolato: LII e pirometria/estinzione

- **D (S13, §3.2.2, p. 41):**
  - la calibrazione LII con l'estinzione «is only valid if the value of E(m) is the same for the calibration (extinction) and LII-measurement conditions»;
  - E(m) dipende da lunghezza d'onda, composizione e temperatura della particella;
  - nella LII a due colori, l'incertezza sulla dipendenza spettrale di E(m) si propaga a T della particella e alle dimensioni.
- **D (secondario, S15, Musculus et al. 2008, DOI 10.1016/j.combustflame.2007.10.023):** in un motore diesel heavy-duty, con gradienti di T soot di 600 K lungo la linea di vista:
  - la T a due colori risulta 200 K più alta della media pesata sulla massa di soot;
  - f_v risulta più bassa del 50 %.

  Bias verso il soot caldo.
- **D (S14, abstract):**
  - DBI-TI estende la diffuse back-illumination extinction imaging a mappe 2D di T soot **integrate sulla linea di vista** in un motore diesel ottico;
  - è più vicina alla T media pesata sulla massa rispetto ad altre tecniche;
  - l'incertezza è stimata con Monte Carlo.
- **Lezione:** nelle proiezioni, l'operatore di misura non è una media uniforme. Pesa le regioni con più segnale, ed è quindi **non lineare e dipendente dallo stato**.

### A-measurements-scout-012 — PIV in motore ottico TCC-III (rimando)

Già trattata nel rapporto preesistente [00_ricognizione_osservazioni.md](../../../docs/nonideal/00_ricognizione_osservazioni.md), §3 (S27). Punti principali:
- EngineBench è un motore **trascinato** (motored), senza combustione;
- vettori mancanti codificati come zero;
- errore 1.5–8 % citato da Schiffmann et al. 2016.

Novità di questo giro, **solo da fonte secondaria** (S26, pagine Deep Blue via motore di ricerca): esistono deposizioni UMich TCC-III **con combustione** (*fired*). Una di queste, "Spark Plug Region", contiene:
- campi di velocità;
- pressione nel cilindro;
- immagini OH* della fiamma;
- risoluzione PIV dichiarata di 1 mm, 1300 rpm, 40 kPa MAP, metano e propano.

L'articolo associato è Schiffmann, Reuss & Sick, *Int. J. Engine Res.* 19 (2018) 491–508, DOI 10.1177/1468087417720558. **M:** contenuto, licenza, sincronizzazione PIV–OH*–pressione; nulla ispezionato.

### A-measurements-scout-013 — Chemiluminescenza OH* per il rilascio termico

**Sintesi.** È il segnale ottico più semplice e diffuso, ed è il tipo presente in RealPDEBench. Nella tesi di Lauer:
- l'intensità integrata segue l'HRR integrale;
- la distribuzione locale non segue l'HRR in turbolenza, se non con una correzione basata su un modello di strain e su misure di velocità.

| Campo | Contenuto |
|---|---|
| Grandezza e contesto | HRR (W/m³ locale o W integrale) in fiamme turbolente magre, premiscelate e parzialmente premiscelate (S08, Abstract). **D** |
| Tecnica e segnale | Camere intensificate con filtri per OH* (circa 310 nm) e un canale per il fondo CO2* a larga banda (circa 456 nm), acquisiti simultaneamente (S08, §4.2 e nota 5). La camera applica in firmware una gamma di 0.7, corretta con 1.3 (S08, §4.2): un esempio di trasformazione del segnale da conoscere. **D** |
| Spazio | **Proiezione**: «no spatial information normal to the camera detection plane». Per fiamme assisimmetriche **mediate nel tempo** la sezione mediana si ottiene per deconvoluzione; l'autoassorbimento è trascurabile a pressione atmosferica (S08, §4.2.2, p. 60). La deconvoluzione di immagini istantanee lascia un errore «that cannot be avoided» (S08, nota 5 del cap. 4). **D** |
| Tempo | Medie su 2048 coppie di immagini (S08, cap. 4); PIV a 1 kHz con Δt_PIV = 20 µs (S08, Fig. 4.14). **D** |
| Errori e limiti | «although the integral heat release rate can be obtained from the integral chemiluminescence intensity, no information about the local heat release rate is provided by chemiluminescence due to turbulence-chemistry interactions» (S08, Abstract). Proporzionalità «commonly assumed, but was never proven» (ibid.). **D** |
| Qualificazione | Tecnica di riferimento costruita con chemiluminescenza, PIV e OH-PLIF e un bilancio di entalpia. È **basata su grandezze mediate nel tempo**, perché gli ingressi non sono simultanei, e «hardly applicable to experimental setups close to technical application» (S08, inizio cap. 5). Validazione quindi in media, non istantanea. **D** |
| Contraddizione | Secondo fonti secondarie (S17), Hardalupas & Orain (2004, Combust. Flame 139:188–207) e lavori successivi ritengono OH*/CH* locali utili per HRR, e il rapporto OH*/CH* per φ, in fiamme counterflow premiscelate e con il sensore FLAMESEEK. Diverse fonti secondarie riportano conclusioni negative per fiamme turbolente o molto curve (Ayoola et al. 2006, S17; Najm et al. 1998, S18, solo metadati). **Conservare:** contesti diversi (laminare o counterflow contro turbolento; locale con ottica raccolta puntuale contro immagine proiettata). |
| Collegamenti | Missione [05_optical_diagnostics](../../05_optical_diagnostics/README.md); C-measurements-scout-002 |

### A-measurements-scout-014 — HRR locale da Raman/Rayleigh (+OH) con reattore omogeneo (grandezza derivata)

**Sintesi.** Esempio di **inversione fisica classica**. Dalle quantità misurabili (T e 7 specie maggiori, eventualmente OH) si ricostruisce uno stato termochimico completo con un calcolo di reattore vincolato, e da questo l'HRR. La valutazione è fatta **solo su DNS**, con incertezze sperimentali imposte.

| Campo | Contenuto |
|---|---|
| Osservazioni (input) | T e N2, O2, CH4, CO2, H2O, CO, H2 (accessibili al Raman/Rayleigh), più OH opzionale (S07, §1). **D** |
| Metodo | Reattore omogeneo a P e T costanti; le specie maggiori sono vincolate ai valori iniziali o alle bande di incertezza, i radicali crescono da zero. Calcolo di HRR e del chemical explosive mode (S07, §1). **D** |
| Riferimento | Due DNS di fiamme CH4/aria premiscelate magre (φ = 0.7) con Karlovitz fino a 253 (Bunsen e round jet) (S07, §1). **Regime:** turbolento, premiscelato o stratificato; **trattamento:** DNS. **D** |
| Risultati | Bunsen, 225 campioni, agli zero-crossing del CM, errore relativo medio (S07, §4):<br>• T: 2.31 % senza OH e 0.15 % con OH;<br>• **HRR: 70.83 % senza OH e 14.24 % con OH**;<br>• HRR integrato sui profili: entro l'1 %.<br>Round jet:<br>• HRR locale: 62.59 % e 12.91 %;<br>• HRR integrato: 10.89 % e 0.51 %.<br>**D** |
| Limiti | Validazione numerica, non sperimentale. Gli errori integrati piccoli nascondono «offsetting errors» locali (S07, §4). I marcatori convenzionali non catturano zone di rilascio debole a valle, dove la ricombinazione CO + OH può pesare il 20–30 % (S07, §1, citando la ref. [4]). Effetto delle incertezze sperimentali nella §5 (analisi PCE): risultati numerici non estratti, **M**. **D/M** |
| Collegamenti | A-002, A-008; C-measurements-scout-002 |

---

## 6. Proposte C (brevi)

| ID | Bisogno | Input disponibili | Stato desiderato | Approcci esistenti | Validazione | Fattibilità e aperti |
|---|---|---|---|---|---|---|
| C-measurements-scout-001 | Le termocoppie sono il sensore termico più disponibile, ma la risposta lenta altera le fluttuazioni (A-001) | Serie di T_b da 1 o pochi giunti; geometria del giunto; velocità locale stimata | T_gas puntuale con le sue fluttuazioni, o statistica (std, PDF) | Correzione inerziale da bilancio energetico (S04); compensazione a due fili o transiente (citata, non letta); possibile stima statistica o appresa (I) | Termometria ottica simultanea e co-localizzata (Rayleigh, CARS) o DNS/LES con sensore sintetico; distinguere media e std | **I:** collega il filone [04_thermal_fusion](../../04_thermal_fusion/README.md) a un modello di sensore documentato (τ, errori). **M:** coppie sperimentali simultanee termocoppia–ottica disponibili |
| C-measurements-scout-002 | L'HRR locale non si misura direttamente; i marcatori sono qualitativi o validi solo come integrale (A-013, A-014) | Uno a scelta, secondo l'apparato: proiezione OH*; OH-PLIF (+CH2O); linea Raman/Rayleigh (+OH); velocità PIV | HRR locale istantaneo o medio, oppure HRR integrale | Correzione con strain (S08); reattore vincolato (S07); prodotti di marcatori; approcci appresi (non cercati in questo giro) | DNS, con incertezze imposte (S07); riferimento sperimentale a bilancio di entalpia **solo in media** (S08) | **I:** il target può non essere T. Coerente con Linzheng: «targets do not necessarily need to be limited to the full instantaneous temperature field». **M:** dati con OH* e campo di riferimento co-registrati |
| C-measurements-scout-003 | Le misure line-of-sight (LAS, chemiluminescenza, pirometria) sono rapide e praticabili ma integrali e pesate dallo stato (A-005, A-011, A-013) | Una o poche proiezioni, spettri o intensità; geometria dei fasci; tempi | Distribuzione spaziale (profilo o campo) di T o specie, o statistiche lungo il percorso | Misure sintetiche dalla CFD per un confronto nello spazio delle misure (S10); tomografia con molte viste (S21, S22, sec.) | Confronto sintetico contro sperimentale (valida misura e modello insieme); fantasmi numerici (S22, sec.) | **I:** il modello di osservazione (integrale pesato, non lineare) è la parte riutilizzabile per costruire osservazioni sintetiche dai campi numerici. **M:** pochi accessi ottici contro molte viste |
| C-measurements-scout-004 | Le misure ad alta frequenza sono sparse e limitate a piani o piccoli volumi; le simulazioni non riproducono lo stesso evento istantaneo | Velocità tomo-PIV o PIV, OH-PLIF, con covarianza dell'errore stimata | Stato completo stimato (LES) coerente con le misure, con incertezza dell'ensemble | EnKF con localizzazione (S12); studio di localizzazione, rumore, frequenza di assimilazione, sparsità e dimensione dell'ensemble. Applicazione reattiva con tomo-PIV e OH-PLIF su un getto DME (S28, sec.) | Errore rispetto a dati non assimilati; limiti dati dall'incertezza sperimentale (S12, Abstract) | **I:** alternativa statistica e fisica ai ricostruttori ML; utile come baseline o confronto concettuale per la missione [03_dmf_gen](../../03_dmf_gen/README.md) sul tema dell'incertezza. **M:** costo computazionale; caso reattivo letto solo in forma secondaria |

---

## 7. Dataset segnalati (non ispezionati)

Nessun dataset è stato scaricato o aperto. Lo stato è quello dichiarato dalle fonti. Non sono schede B.

| Dataset / fonte | Che cosa dichiara la fonte | Stato | Collegamento |
|---|---|---|---|
| Archivio TNF (Sandia flames, Sydney swirl, TECFLAM) | Raman/Rayleigh/LIF, LDV; campagne non simultanee per velocità e composizione | Rapporto preesistente (S27, §4) | A-002 |
| UMich Deep Blue TCC-III, *fired* "Spark Plug Region" e *full-view* | PIV (1 mm), pressione nel cilindro, OH* (solo spark plug region) | Snippet del motore di ricerca (S26) | A-012 |
| EngineBench (TCC-III, motored) | PIV, nessuna combustione | Rapporto preesistente (S27, §3) | A-012 |
| DLR GTMC (PIV/OH-PLIF 5–10 kHz; Raman e LDV in Weigand 2006) | Campi e statistiche nei paper | Accesso **M** | A-006 |
| Tomo-PIV del jet Sandia (ref. [20, 35] di S12) | 3C volumetrica a 10 kHz, non reattiva | Accesso **M** | A-007 |
| RealPDEBench (combustione) | Campi LES e OH* sperimentale | Assegnato all'incarico Dati | – |

---

## 8. Implicazioni per il nostro lavoro (senza forzarle su RealPDEBench)

- **D + I:**
  - Una "velocità sparsa ideale" corrisponde a una PIV solo dopo aver aggiunto filtro di finestra (circa 1 mm in S01 e S02), campo di vista ridotto, durata dei record e lacune.
  - L'errore casuale è l'unico dichiarato in S02.
  - Il trasferimento di questi parametri al nostro combustore resta **I**.
- **I:** per sensori termici "lenti" esiste ora un ordine di grandezza **documentato in un contesto preciso** (τ circa 57 ms con un filo da 50 µm in un pool fire, S04). Non è un parametro del nostro combustore: τ dipende da diametro, velocità e T.
- **I:** l'OH* di RealPDEBench è una proiezione con relazione non locale con l'HRR (S08). Un target come l'HRR integrale o una sua distribuzione mediata è più vicino a ciò che la letteratura letta sa validare. Per l'HRR locale istantaneo la validazione documentata è solo su DNS.
- **I:** la roadmap chiede di confrontare problemi su campi medi e istantanei. Le fonti lette validano spesso grandezze **mediate** (S08, cap. 5; S04, media e std): un target medio è sperimentalmente più validabile, ma non valida fluttuazioni.

---

## 9. Fonti e stato della lettura

| ID | Citazione e DOI/URL | Accesso e parti lette | Localizzatori usati | Schede |
|---|---|---|---|---|
| S01 | Boxx, Stöhr, Carter, Meier, *Appl. Phys. B* 95 (2009) 23–29, DOI 10.1007/s00340-009-3420-4; manoscritto https://elib.dlr.de/58698/1/Boxx2009APB_manuscript.pdf | **Testo integrale** del manoscritto (figure non viste) | «PLIF System», «PIV System», «Burners», «Results» | A-006, A-008 |
| S02 | Stöhr, Arndt, Meier, *Proc. Combust. Inst.* 34 (2013) 3107–3115, DOI 10.1016/j.proci.2012.06.086; manoscritto https://elib.dlr.de/77873/1/Stoehr2013cs_manuscript.pdf | Abstract, §2 (setup) per intero, Tab. 1, §4 (stima di Da); resto scorso | §2, §4 | A-006 |
| S03 | Barlow, Wang, Anselmo-Filho, Sweeney, Hochgreb, *Proc. Combust. Inst.* 32 (2009) 945–953, DOI 10.1016/j.proci.2008.06.070; **versione sottomessa** SAND2007-8048C, https://www.osti.gov/servlets/purl/1146262 | Testo integrale della versione sottomessa (figure non viste) | §2.2, pp. 5–7; §3.1–3.2; Tab. 2, p. 16 | A-002, A-008 |
| S04 | Sung, Chen, Bundy, Fernandez, Hamins, *The Thermal Character of a 1 m Methanol Pool Fire*, NIST TN 2083r1 (2021), DOI 10.6028/NIST.TN.2083r1 | Abstract, prefazione, §2.1–2.1.1, §3.7 (pp. 19–21, Tab. 5); appendici non lette | pp. i–ii, 3–5, 19–21 | A-001 |
| S05 | Shaddix, *A new method to compute the radiant correction of bare-wire thermocouples*, MCS 10, Napoli 2017, SAND2017-4404C, https://www.osti.gov/servlets/purl/1456684 | Testo integrale (valori delle figure non estratti) | Abstract, Introduzione, Fig. 3–6, Conclusioni | A-001 |
| S06 | Shaddix, *Correcting thermocouple measurements for radiation loss: a critical review* (1999), OSTI 20002672 | Solo snippet del motore di ricerca | – | A-001 (contesto) |
| S07 | Hartl, Geyer, Hasse, Zhao, Wang, Barlow, *Combust. Flame* 209 (2019) 214–224, DOI 10.1016/j.combustflame.2019.07.038; manoscritto R3 SAND2019-8792J, https://www.osti.gov/servlets/purl/1559499 | Abstract, §1, §4 (risultati principali), inizio §5 | §1, §4, §5 | A-008, A-014 |
| S08 | Lauer, *Determination of the Heat Release Distribution in Turbulent Flames by Chemiluminescence Imaging*, tesi di dottorato TU München (2011), https://www.epc.ed.tum.de/fileadmin/w00cgc/td/Forschung/Dissertationen/lauer11.pdf | Abstract, §4.2 (passaggi su line-of-sight, gamma, acquisizione), nota 5, inizio cap. 5; resto non letto | p. vii, p. 60, cap. 4–5 | A-013 |
| S09 | Hansen, Cool, Westmoreland, Kohse-Höinghaus, *Prog. Energy Combust. Sci.* 35 (2009) 168–191, DOI 10.1016/j.pecs.2008.10.001; manoscritto SAND2008-3883C, https://www.osti.gov/servlets/purl/1143217 | §2.1–2.3 (pp. 7–17 del manoscritto) | pp. 7–11, 17 | A-010 |
| S10 | Schwartz et al., *J. Appl. Phys.* 139 (2026) 074701, DOI 10.1063/5.0312755; https://www.osti.gov/servlets/purl/3019976 | Abstract, introduzione parziale, §II.B (misure sintetiche), passaggi su fit e incertezza | Abstract; pp. 074701-8, -10, -11 | A-005 |
| S11 | Goldenstein, Spearrin, Jeffries, Hanson, *Prog. Energy Combust. Sci.* 60 (2017) 132–176, DOI 10.1016/j.pecs.2016.12.002 | **Non letto** (solo metadati e snippet) | – | A-005 |
| S12 | Labahn, Wu, Harris, Coriton, Frank, Ihme, *Flow Turbul. Combust.* 104 (2020) 861–893, DOI 10.1007/s10494-019-00093-1; manoscritto SAND2021-2782J, https://www.osti.gov/pages/servlets/purl/1772036 | Abstract, §1, §4.1 (localizzazione), §5.1 (dati sperimentali), inizio §5.2 | pp. 1–3, 14–16 | A-007; C-004 |
| S13 | Michelsen, Schulz, Smallwood, Will, *Prog. Energy Combust. Sci.* 51 (2015) 2–48, DOI 10.1016/j.pecs.2015.07.001; manoscritto SAND2015-6103J, https://www.osti.gov/pages/servlets/purl/1265196 | Solo §3.2.2 (funzione di assorbimento) | p. 41 del manoscritto | A-011 |
| S14 | Lind et al., *Combust. Flame* 240 (2022) 111949, DOI 10.1016/j.combustflame.2021.111949; https://www.osti.gov/pages/servlets/purl/1841997 | Solo abstract | Abstract | A-011 |
| S15 | Musculus, Singh, Reitz, *Combust. Flame* 153 (2008) 216–227, DOI 10.1016/j.combustflame.2007.10.023 | Snippet secondario | – | A-011 |
| S16 | Giezendanner-Thoben et al., *Appl. Opt.* 44 (2005) 6565–6577, DOI 10.1364/AO.44.006565 | Frasi dell'abstract via motore di ricerca | – | A-004 |
| S17 | Hardalupas & Orain, *Combust. Flame* 139 (2004) 188–207; e fonti che li citano (Comptes Rendus Mécanique 2010; Ayoola et al. 2006, DOI 10.1016/j.combustflame.2005.06.005) | Solo fonti secondarie e metadati | – | A-013 |
| S18 | Najm, Paul, Mueller, Wyckoff, *Combust. Flame* 113 (1998) 312–332, DOI 10.1016/S0010-2180(97)00209-5 | Solo metadati | – | A-013 |
| S19 | Abram, Fond, Heyes, Beyrau, *Appl. Phys. B* 111 (2013) 155–160, DOI 10.1007/s00340-013-5411-8 | Snippet dell'abstract | – | Tabella §3 |
| S20 | Tokarev et al., *Meas. Sci. Technol.* 26 (2015) 064001, DOI 10.1088/0957-0233/26/6/064001 | Frammenti dell'abstract (IOP inaccessibile) | – | A-007 |
| S21 | Grauer et al., *Combust. Flame* (2018), DOI 10.1016/j.combustflame.2018.06.022 | Solo fonti secondarie | – | Tabella §3; C-003 |
| S22 | Floyd & Kempf, *Proc. Combust. Inst.* 33 (2011); Unterberger et al. (CTC industriale) | Solo fonti secondarie | – | Tabella §3; C-003 |
| S23 | SAE 2006-01-1346, 1999-01-1329, 1999-01-0935, 900170/900171 | Snippet degli abstract | – | A-009 |
| S24 | NASA NTRS 20190025903 (ITP); lavori su FTF multi-microfono (TUM, ETH) | Snippet | – | A-009 |
| S25 | Fonti su Rayleigh/FRS (risultati del motore di ricerca, NIST firedoc, IOP CPB); Doll et al. 2012, elib.dlr.de/77091 (scaricato, **non letto**) | Snippet | – | A-003 |
| S26 | UMich Deep Blue, deposizioni TCC-III (*fired* e *motored*) | Snippet | – | A-012 |
| S27 | Rapporto preesistente [docs/nonideal/00_ricognizione_osservazioni.md](../../../docs/nonideal/00_ricognizione_osservazioni.md) (EngineBench, TNF, Sciacchitano 2015/2019, Wieneke 2015) | Letto integralmente; le sue marcature [P]/[S] restano valide | §2–§5 | A-006, A-012 |
| S28 | Applicazione EnKF reattiva (tomo-PIV + OH-PLIF in LES di un getto turbolento DME), pagina pubblicazioni Sandia «Data assimilation using high-speed measurements and LES to examine local extinction events in turbulent flames» | Snippet | – | C-004 |
| S29 | CARS: Applied Optics 56 (2017) 8797 (fs-CARS chirped-probe) e slide Sandia su fs/ps CARS (risultati del motore di ricerca) | Snippet, non verificati | – | Tabella §3 |

Riferimenti di contesto non letti ma segnalati per il prossimo giro:
- Sciacchitano 2019, DOI 10.1088/1361-6501/ab1db8 (IOP bloccato da questo ambiente; sintesi nel rapporto preesistente);
- Cai & Kaminski 2017, *PECS*, DOI 10.1016/j.pecs.2016.11.002 (tomografia ad assorbimento);
- Aldén et al. 2011, *PCI* 33, DOI 10.1016/j.proci.2010.09.004;
- Sick 2013, *PCI* 34, DOI 10.1016/j.proci.2012.08.012;
- Weigand et al. 2006, *Combust. Flame* 144:205–224;
- Hassel & Linow 2000, *MST* (review su velocità, concentrazioni e T).

---

## 10. Registro delle ricerche, esclusioni e lacune

**Percorso seguito.**
1. Ricerca web per review e fonti autorevoli per famiglia, ~20 query. Temi: review generali, Raman/Rayleigh TNF, termocoppie, CARS, assorbimento, PIV/PLIF DLR, chemiluminescenza/HRR, pressione nel cilindro, PIV in fiamma e termoforesi, BOS, fosfori, LII, tomo-PIV, CTC, pressione dinamica/FTF, Hardalupas & Orain, assimilazione, motore ottico TCC-III, Rayleigh planare, pirometria soot.
2. Ricerca di versioni open access tramite OpenAlex e l'API OSTI.
3. Lettura diretta dei PDF scaricati (testo estratto e letto per sezioni).

**Accesso.** All'inizio la rete dell'ambiente bloccava tutte le fonti, quindi è stato creato un ambiente personale con domini accademici consentiti, approvato dall'utente. Restano inaccessibili:
- IOP (challenge anti-bot): Sciacchitano 2019, Tokarev 2015;
- core.ac.uk (challenge): tesi di Windsor sulla pressione nel cilindro;
- ScienceDirect, SAE, Cambridge repository: fuori dai domini consentiti.

**Esclusioni rilevanti.**
- Paper di Jason (Nie 2024, Liu 2025): assegnati a un altro incarico.
- Approfondimento metrologico PIV oltre S01, S02, S12: lasciato alla missione 02 e al rapporto preesistente.
- Lavori ML di ricostruzione: non cercati di proposito in questo giro, per non sbilanciare la mappa verso il ML.
- CARS: solo panorama secondario, nessuna scheda.

**Lacune che possono cambiare una valutazione.**
1. **PIV in fiamma, bias specifici** (termoforesi, calo di seeding nei prodotti, beam steering, luminosità): solo fonti secondarie. È la lacuna più rilevante per la missione 02.
2. **Pressione:** nessuna fonte primaria letta. Serve per valutare i dati motore (RANS/URANS) indicati da Linzheng e Jason.
3. **Assorbimento laser e tomografia ad assorbimento:** review S11 e Cai & Kaminski non lette.
4. **Temperatura planare quantitativa in combustori** (two-line OH, Rayleigh/FRS): solo abstract o secondario.
5. **Gas analyzers / emissioni in combustori e motori:** non coperto.
6. **CARS:** non coperto oltre il panorama.
7. **Frequenza di ripetizione e criteri di rigetto Raman/Rayleigh:** M nella versione letta di S03.

---

## 11. Suggerimenti per i prossimi approfondimenti

1. **Pressione (motore e combustore).** Leggere una fonte primaria su pegging e thermal shock (per esempio il libro di Rogers o gli SAE citati, via accesso istituzionale) e una su funzioni di trasferimento delle linee. **Motivo:** è la misura più disponibile per i dati motore e il ponte con RANS/URANS.
2. **PIV in fiamma.** Picano et al. 2011 (DOI 10.1007/s00348-010-0896-y) su seeding e termoforesi, più una fonte sulla frazione di vettori rigettati in fiamma swirl. **Motivo:** decide se le lacune della PIV virtuale devono essere correlate alla fiamma (già indicato come rischio nel rapporto preesistente).
3. **Goldenstein 2017 (S11) e Cai & Kaminski 2017.** **Motivo:** completare A-005 e C-003 con limiti pratici di accesso, numero di fasci e incertezza.
4. **Dataset TCC-III *fired* (S26).** Verificare, con una sola lettura documentale e senza download, se PIV, OH* e pressione sono sincronizzati per ciclo. **Motivo:** sarebbe un raro caso sperimentale con velocità, osservabile ottico e pressione nello stesso motore. Da passare all'incarico Dati.
5. **Hardalupas & Orain 2004 e Ayoola 2006 in full text.** **Motivo:** risolvere o documentare meglio la contraddizione su OH* locale (A-013), centrale per la missione ottica.
6. **Termocoppie compensate** (due fili, compensazione dinamica in fiamme turbolente). **Motivo:** dare a C-001 approcci classici documentati prima di proporne di appresi.
7. **Revisione trasversale:** ricontrollare in particolare i numeri di Tab. 2 di S03 sulla versione pubblicata e i risultati della §5 di S07.
