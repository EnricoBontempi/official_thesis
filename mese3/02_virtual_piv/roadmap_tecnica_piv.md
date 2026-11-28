# Missione 02 — Roadmap tecnica: osservazioni PIV sulla LES e campagna U-Net/GLU

**Data:** 27 novembre 2026 · **Branch:** `piv/experiment-proposal` · **Destinatario dell'esecuzione:** Sonnet IDE, dopo l'approvazione di Enrico
**Stato:** impostazione approvata da Enrico il 27/11/2026; revisione v2 con le correzioni richieste per il passaggio a Sonnet (supporto e discretizzazione del filtro, confronti sugli stessi nodi, significato dei campi completi, download autorizzati: §10). Questo documento non implementa nulla e non autorizza download di dataset, job, training, inferenze o accesso a TEST. Ogni fase parte solo dopo il punto di revisione che la precede (§6).

**§0bis — Decisioni R2 (28/11/2026).** La revisione R2 è chiusa con [decisione_R2.md](decisione_R2.md), che prevale su questa roadmap dove le due divergono:
- modello principale invariato;
- correlazione dell'errore: SENS-ρ0 (SENS-RHO0) diventa **obbligatoria** accanto al principale in ogni confronto su A e B-in;
- specifica **v1.1 additiva**: + SENS-TAIL (code pesanti, ν = 7) su A e B-in; + sostituzione 5×5 (`sub5`) per SENS-REJ iso e clu;
- prima della Fase 3: F2.8 (v1.1), F2.9 (regressione, test nuovi, qualifica supplementare) e controllo dell'hash fra macchine;
- F2.7 autorizzata subito.

**§0 — Decisioni R1 (27/11/2026).** La revisione R1 è chiusa con [decisione_R1.md](decisione_R1.md), che prevale su questa roadmap dove le due divergono. Le sezioni interessate sono state aggiornate:
- componenti (vi, vj), con vi trasversale e vj assiale (esito *probabile*); SENS-OOP disattivata;
- righe ↔ `x` (passo 0,581 mm), colonne ↔ `y` (passo 0,758 mm) per generatore e metriche in mm; la convenzione storica diventa SENS-SWAP;
- kernel tarati sempre all'1 %; top-hat solo come sensibilità, tarato a L_sr = W; il criterio D-kern è sostituito;
- v_ref sull'intero record di ogni condizione;
- B allenato nella posizione adiacente al bordo d'ingresso, centrata sull'asse (B-in);
- definizioni diagnostiche fissate;
- attività autorizzate: F2.0–F2.6.

**Problema invariato.** Dataset RealPDEBench combustion, parte LES numerica. Compito: ricostruzione istantanea della T dalle velocità osservate, con le stesse condizioni, gli stessi split e la stessa T come riferimento. Cambiano soltanto le osservazioni di velocità. Non ci sono nuove architetture, né varianti con tracciante a olio, né dataset sperimentali usati come dati di training: Neal, Cardiff e gli articoli servono solo a qualificare il modello di misura.

**Base documentale.**
- [sintesi_evidenze_modello_piv.md](sintesi_evidenze_modello_piv.md) (in seguito **SINT**), con le sigle delle fonti primarie lette (STO12, OBE15, BOX10, SAI16, LAN12, KAR20, UTA1, UTA2, BAU15, WIE17).
- Scouting letteratura @ `2cbf5ab` (**PIVL:Lnn**) e scouting dataset @ `ee9948e` (**PIVD:S2D-nn**).
- [Scheda Sciacchitano](step1_sciacchitano.md) (**P01–P18**) e [roadmap 02](roadmap02.md).
- Missione 01: [inventario](../01_observation_state_matrix/data_inventory.md) (**INV**) e [matrice](../01_observation_state_matrix/observation_state_matrix.md).
- Missione 00: `nonideal/specialist-results` @ `cb77e6e` (`docs/nonideal/05_specialist_results.md`).

**Stato dei documenti precedenti.** Nessuno è eliminato.

| Documento | Stato rispetto a questa roadmap |
|---|---|
| [proposta_esperimenti_v1.md](proposta_esperimenti_v1.md) | Proposta alternativa conservata; non è il piano della missione |
| [proposta_missione02.md](proposta_missione02.md) | **Superata** per la definizione delle osservazioni e per la campagna. Resta valida la scelta del GTMC come ancoraggio dei parametri. Sono abbandonati il trasferimento dei rigetti da Cardiff e i gruppi di training lì elencati |
| [sintesi_evidenze_modello_piv.md](sintesi_evidenze_modello_piv.md) | **Base** di questa roadmap. Non adottati: la variante olio e la sensibilità dipendente da T sulla densità di particelle. La perdita di coppie fuori piano diventa una sensibilità condizionata. Le ambiguità corrette sono elencate al §2.7 |
| [roadmap02.md](roadmap02.md) | Roadmap di missione, invariata. Questo documento ne esegue gli step 3–5 |

---

## 1. Per Enrico: che cosa cambia, perché, quale campagna

**Come cambiano i dati.** Oggi le reti ricevono 819 valori di velocità esatti, presi in punti casuali della slice. Con la roadmap riceveranno ciò che restituirebbe una PIV planare in un combustore swirl:

- un vettore per ogni nodo dentro il campo di vista;
- ogni vettore è una media pesata su circa 1,3 mm, come la finestra di interrogazione;
- un errore casuale di circa 0,45–0,87 m/s secondo la condizione (media ≈0,6 m/s, dalla Fase 1), correlato fra vettori vicini, con ampiezza fissata dal modo in cui si progetta la misura per quella condizione;
- nessun vettore dove la finestra esce dal campo.

Le configurazioni principali sono due, entrambe tratte dallo stesso articolo sul combustore modello GTMC del DLR (Stöhr et al. 2012):

- **A — pieno campo, basso rateo:** tutta la slice;
- **B — campo ridotto, alto rateo:** una finestra di 30×30 mm adiacente al bordo d'ingresso e centrata sull'asse, come il campo GTMC "above the nozzle" (decisione R1).

I rigetti dei vettori interni non hanno una base quantitativa in fiamma swirl atmosferica. Il modello principale rappresenta il dato **rilasciato e completo**, ma un campo sperimentale completo dopo interpolazione non dimostra che rigetti e sostituzioni siano senza effetto. Il modello principale li approssima ignorandoli. Le **sensibilità dichiarate**, con e senza sostituzione, ne esplorano alcune conseguenze, non la loro entità reale.

**Perché è più rappresentativo della missione 00.** La 00 aggiungeva rumore indipendente e dischi senza dati a un layout che nessuna PIV produce. Qui ogni elemento segue una proprietà documentata della misura:

- layout e passo dalla finestra;
- media spaziale dall'algoritmo;
- ampiezza dell'errore dal progetto di Δt;
- correlazione degli errori dalla metrologia;
- copertura dall'accesso ottico e dalla frequenza.

Le intensità non sono scelte in base al danno sulle reti. Se filtro ed errore risultassero piccoli sulla nostra griglia, lo riporteremmo come risultato.

**Quale campagna propongo.**
1. **Verifiche e implementazione:** orientamento delle componenti dalla provenienza del dataset, supporto della griglia alle scale PIV, generatore delle osservazioni. Poi una **revisione delle osservazioni generate**, con figure e statistiche, prima di qualunque rete.
2. **Test dei checkpoint esistenti** (12 originali e 2 U-Net dense del Mese 2), costruiti per separare il cambio di layout dagli effetti di filtro, errore e copertura, con un ponte diretto ai numeri della 00.
3. **Training da zero su A e su B**, ciascuno con il suo controllo ideale **sugli stessi nodi** e con la stessa ricetta:
   - U-Net: 4 celle × 5 seed;
   - GLU: 4 celle × 3 seed;
   - un ponte GLU di 3 run che chiude il confondente di batching della 00.

   In tutto 35 run necessari, contro 30 nella 00. Gruppi aggiuntivi (rigetti, training su più configurazioni, separazione filtro/errore) solo se i risultati li motivano.

Costo indicativo del necessario: circa 180–850 GPU-h, contro 210 nella 00. L'incertezza viene quasi tutta dal GLU con migliaia di token, da misurare con una profilazione prima di decidere.

---

## 2. Definizione delle osservazioni PIV (contratto v1 da congelare alla revisione R1)

### 2.1 Vincoli dal dataset

Dalla missione 01 (INV B-data-audit-001, §7):
- griglia 128×128; coordinate x ∈ [−36,9; 36,9] mm e y ∈ [62,9; 159,1] mm. Assegnazione agli assi **decisa a R1**: righe ↔ x (passo 0,581 mm, laterale), colonne ↔ y (passo 0,758 mm, assiale), esito *probabile*. La convenzione storica della missione 01 (colonne ↔ x) è SENS-SWAP;
- celle 0,581 mm (lungo le righe) × 0,758 mm (lungo le colonne);
- velocità in m/s e T in K, per inferenza dagli intervalli;
- **definizione di i/j/k, segno e piano della slice non dichiarati**. Fase 1: slice per l'asse del bruciatore, con l'asse sulla riga 63,5; vj assiale, vi trasversale, vk fuori piano (*probabile*; [orientation_note.md](orientation_note.md), [decisione_R1.md](decisione_R1.md) §2);
- vk presente nei raw ma non nelle cache;
- filtro e ricampionamento della LES ignoti.

La cache densa C149 contiene vi e vj normalizzati su tutti i nodi: è la sorgente dei campi da osservare. Le statistiche di normalizzazione restano quelle fit-only congelate, sha256 `678263fb…`.

### 2.2 Configurazioni

| ID | Ruolo | Descrizione | Fonte |
|---|---|---|---|
| **PIV-A** | Principale | Stereo-PIV a basso rateo, pieno campo, particelle solide; finestra W = 1,3 mm; overlap 50 % | STO12 §2.3.2; OBE15 §2.2 |
| **PIV-B** | Principale | Stereo-PIV ad alto rateo, campo 30×30 mm, particelle solide; W = 1,5 mm; overlap 50 % | STO12 §2.3.3 (stesso articolo e combustore di A) |
| **IDEAL-full** | Riferimento di layout (solo valutazione) | Valore puntuale LES su tutti i 16 384 nodi | Mese 2, U4/U6 |
| **IDEAL-A** | Controllo | Valore puntuale LES sugli **stessi nodi validi** di PIV-A | — |
| **IDEAL-B** | Controllo | Valore puntuale LES sugli **stessi nodi validi** di PIV-B, nella stessa posizione | — |
| **SENS-…** | Sensibilità | Varianti non calibrate o alternative documentate (§2.6) | — |

A e B **non** mescolano apparati: sono i due sistemi descritti in STO12. Le altre fonti mostrano che le proprietà ricorrono (SINT §2.2), ma non forniscono parametri da sommare.

Entrambi i sistemi GTMC sono stereo e misurano tre componenti. Ne usiamo due (vi, vj). R1 ha chiuso questo punto con esito *probabile*: vi trasversale e vj assiale nel piano, vk fuori piano. Il limite residuo (vi potrebbe essere fuori piano) è dichiarato in decisione_R1 §2.

### 2.3 Elemento per elemento

Formato di ogni elemento: fonte → motivazione → parametri → assunzioni e incertezze.

**(1) Layout regolare e passo**
- *Fonte:* nelle configurazioni DLR il passo è metà della finestra con overlap 50 %: 1,0/0,5 mm (BOX10 p. 15), 1,0/0,5 mm (SAI16 p. 7), 1,9/0,95 mm (KAR20 p. 5). Per W = 1,3 mm il passo è ≈0,65 mm [inferenza].
- *Rappresentazione:* **un vettore per ogni nodo LES valido**, passo 0,581×0,758 mm. Nessuna interpolazione fuori nodo. U-Net e GLU ricevono così gli stessi valori nelle stesse posizioni.
- *Assunzioni:* il passo è anisotropo invece di quadrato; per B il passo dedotto (0,75 mm) coincide quasi con la cella lungo le colonne (0,758 mm). Il numero di vettori supera l'informazione indipendente (P10). Si dichiara, non si corregge.

**(2) Campo di vista e copertura**
- *A:* tutta la slice ("entire vertical section", STO12 §2.3.2), meno la fascia di bordo del punto (3).
- *B:* finestra di 30×30 mm (STO12 §2.3.3: "region above the nozzle"), cioè 52 righe × 40 colonne (30,2 × 30,3 mm) e il 13 % dell'area.
  - **Posizioni (decisione R1, regola geometrica e di velocità, indipendente dai risultati):**
    - **B-in**, training e valutazione: righe 38–89, colonne 88–127; adiacente al bordo d'ingresso, centrata sull'asse;
    - **B-down**, solo valutazione: righe 38–89, colonne 48–87; 30 mm a valle sulla stessa linea d'asse;
    - **B-off**, solo valutazione: righe 64–115, colonne 88–127; adiacente all'ingresso, con l'asse al bordo della finestra.
- *Motivazione:* il campo ridotto ad alto rateo è documentato in più apparati (Toronto/DLR, DLR 5 bar, UT Austin; SINT §2.2). Il decentramento dipende dalla geometria del combustore, che non conosciamo (INV C1).
- *Assunzioni:* l'ugello potrebbe essere fuori dalla slice (y parte da 62,9 mm): B-in è l'approssimazione più vicina a "above the nozzle". Risultati riportati per posizione. Nessuna esclusione da riflessi o pareti nel modello di base, perché non sappiamo se la slice contenga pareti (SINT G10).

**(3) Risposta spaziale**
- *Fonte:* il vettore è una media pesata dello spostamento. Per il multi-pass DaVis su immagini sintetiche la risposta alle lunghezze d'onda è un MHG (gaussiana con 30–40 % di cappello messicano) con L_sr(R_A) = 29,8 px su una finestra di 32 px al 50 %, cioè ≈0,93 W (WIE17 §2.6, Tab. 2.4, PDF 39–40). Per il single-pass a finestra quadrata la risposta è un top-hat di lato W.
- *Kernel continuo:* separabile, F(x, y) = F(x) F(y), con per ciascun asse la forma MHG di WIE17 (Tab. 2.1, p. 15 stampata, PDF 26):
  F(s) = (1 − b s²/σ_r²) · exp(−s²/(2σ_r²)) / ((1 − b) σ_r √(2π)), con ∫F ds = 1.
  - Lunghezza di filtro: L_sr = 1/∫F² ds = σ_r √(4π) (1 − b)² / (1 − b + 3b²/4).
  - Con b = 0,35 e target L_sr = 0,93 W si ottiene il valore **nominale** σ_r = 0,4607 W (0,599 mm in A, 0,691 mm in B). I valori usati sono quelli **tarati** per asse (sotto e decisione_R1 §4), per esempio 0,545 mm lungo le righe e 0,531 mm lungo le colonne con W = 1,3. Il lobo negativo inizia a |s| = σ_r/√b ≈ 0,78 W. Calcolo verificato numericamente in questa revisione.
- *Supporto e troncamento:* il supporto numerico è |s| ≤ R_r = 4 σ_r ≈ 1,84 W, cioè **2,40 mm in A e 2,76 mm in B**. **Non coincide con ±W/2**: la risposta equivalente del multi-pass si estende oltre la finestra. La massa |F| oltre 4 σ_r è circa 5·10⁻⁴. Il kernel è troncato a R_r e i pesi sono rinormalizzati a somma 1.
- *Discretizzazione sulla griglia LES:* il campo LES è trattato come costante a tratti nella cella (assunzione dichiarata; nessun sovracampionamento, che introdurrebbe un'interpolazione in più).
  - Peso per asse: w_x(i) = ∫ F(s) ds sulla cella i, cioè sull'intervallo [s_i − Δ/2, s_i + Δ/2] ∩ [−R_r, R_r], con s_i offset del nodo e Δ il passo dell'asse (0,581 mm lungo le righe, 0,758 mm lungo le colonne).
  - Pesi 2D = prodotto; normalizzazione Σ w = 1.
  - Con W = 1,3 mm il kernel tarato copre ±4 nodi lungo le righe (passo 0,581 mm) e ±3 lungo le colonne (passo 0,758 mm).
- *Verifica e taratura della risposta discretizzata:* per ciascun asse si calcolano la lunghezza discreta L_sr,disc = Δ / Σ_i w(i)² e la risposta discreta H(k) = Σ_i w(i) cos(k s_i).
  - **Decisione R1:** σ_r si tara **sempre** per asse al target 0,93 W, con tolleranza dell'1 % (non più "solo oltre il 5 %").
  - Nella specifica si registrano σ_r nominale e tarato, L_sr,disc e H(k) alle lunghezze d'onda 2–10 mm.
  - La risposta dichiarata è quella **discreta realizzata**, non quella continua.
- *Parametri:* A: W = 1,3 mm. B: W = 1,5 mm.
- *Campo e bordi:* il campo è un rettangolo in mm.
  - **A:** rettangolo dei nodi estremi della slice, sotto l'assunzione dichiarata "slice = sezione osservata". Margine: 2 nodi lungo le righe e 1 lungo le colonne (W = 1,3 e 1,5).
  - **B:** [x_c ± 15 mm] × [y_c ± 15 mm].
  - **Esistenza del vettore (regola PIV):** il vettore esiste se la sua **finestra di interrogazione** [±W/2]² è interamente nel campo. Margine atteso: 2 nodi in x e 1 nodo in y sia per W = 1,3 sia per W = 1,5. Vettori validi: A = 124 × 126 = 15 624 (righe × colonne); B = 48 × 38 = 1 824 per posizione. I conteggi esatti si fissano nella specifica.
  - **Vettori vicini al bordo** (finestra dentro, supporto R_r fuori): kernel troncato ai soli nodi del campo e rinormalizzato a somma 1. È un'approssimazione [ipotesi] della risposta modificata di una PIV reale vicino al bordo dell'immagine, dove contribuiscono solo le particelle visibili. Questi vettori sono marcati nei metadati e la loro risposta realizzata si misura a parte (F2.6).
  - In B la LES esiste anche fuori dalla finestra ma **non si usa**: la camera non la vede.
  - Niente estrapolazione né riflessione.
  - **SENS-EDGE** (solo valutazione): esistono solo i vettori il cui supporto R_r è interamente nel campo, con margine di circa 4–5 nodi lungo le righe e 3–4 lungo le colonne in A.
- *Top-hat (SENS-TH):* F = 1/W_eff su |s| ≤ W_eff/2, stessa integrazione per cella con peso proporzionale alla sovrapposizione, stessa rinormalizzazione e stesse regole di bordo. **Decisione R1:** W_eff si tara per asse perché L_sr,disc = W, cioè la lunghezza del single-pass; senza taratura la discretizzazione lo allarga di 1,2–1,5 volte. Solo sensibilità di valutazione: l'apparato di riferimento usa una correlazione multi-pass.
- *Assunzioni e incertezze:*
  - pesatura e numero di passate GTMC non dichiarati (DaVis 7.2 e 8.0, "multi-scale");
  - la finestra copre circa 2 celle e il supporto del kernel ±4 nodi lungo le righe e ±3 lungo le colonne: l'effetto dipende dal contenuto della LES a quelle scale (verifica F1.2) e dalla discretizzazione, verificata come sopra;
  - **la media attraverso lo spessore del foglio (~1 mm) non è rappresentabile** su una slice.

**(4) Ampiezza dell'errore casuale**
- *Fonte:*
  - il GTMC dichiara "±0,1 pixel uncertainty of the peak-finding algorithm" → ±0,3–1,4 m/s, con Δt da 20 µs a 6 µs secondo la potenza (STO12 §2.3.2);
  - con 16 px = 1,3 mm e v ≈ 80 m/s a 6 µs lo spostamento massimo è ≈6 px [inferenza da STO12];
  - deviazione standard dell'errore **misurata** contro un riferimento in un getto non reagente, 16×16 px al 75 %: 0,060 px (WIE17 §4.4.2, PIVL:L16);
  - **sintetico** DaVis 16×16 px al 75 %: 0,146 px (WIE17 Tab. 2.4).
- *Regola di rappresentazione:* σ_u = σ_px · v_ref,c / d_max per ciascuna componente in piano, uniforme nel frame. È proporzionale alla velocità di riferimento della condizione c, perché lo sperimentatore sceglie Δt condizione per condizione.
  - **Decisione R1:** v_ref,c = p99,9 di |u_planare| su tutti i nodi dei frame **0:2001:10 dell'intero record** della condizione (201 frame, dal raw), **solo velocità**; IC95 con bootstrap a blocchi. Motivo: STO12 ricava la scala da campi istantanei della condizione, e Δt vale per l'intera prova (decisione_R1 §6);
  - per TEST si calcola soltanto al momento dell'apertura, autorizzata separatamente.
- *Parametri:*
  - σ_px = **0,10 px** come valore principale, scelto perché sta all'interno dell'intervallo coperto da due evidenze indipendenti (0,06 px misurato; 0,15 px sintetico) e coincide con l'ordine dichiarato dalla fonte;
  - sensibilità **SENS-σlo** a 0,06 px e **SENS-σhi** a 0,15 px;
  - d_max = 6 px [inferenza da STO12].

  Ordine di grandezza dalla Fase 1 (pool C149): σ_u fra 0,45 e 0,87 m/s secondo la condizione, media ≈0,59 m/s. Il valore 0,8 m/s della versione precedente veniva dal p99,9 raggruppato su tutte le condizioni ed era una sovrastima.
- *Distribuzione:* gaussiana, **come ipotesi di modello**.
- *Assunzioni e incertezze:*
  - **il "±0,1 px" pubblicato non è trattato come σ**: la fonte non dichiara se sia standard o espansa, né la copertura;
  - l'errore reale in fiamma non è uniforme: cresce dove le particelle sono rade e con il moto fuori piano (SINT §2.2, §3.1). Il modello principale non lo rappresenta.

**(5) Correlazione spaziale dell'errore**
- *Fonte:* in DaVis multi-pass la risposta e l'autocorrelazione del rumore sono **due funzioni diverse** (WIE17 §2.5, PDF 39). Con W = 32 px al 50 %, la lunghezza integrale dell'autocorrelazione del rumore è L_sr(rumore) = 38,4 px ≈ 1,2 W, contro 0,93 W della risposta. Monte Carlo: ρ(2d) = 0,11 "rappresentativo" al 50 % (WIE17 §4.3, PDF 90). La review riporta ρ(2h) = 0 per overlap ≤50 % in quel caso (P16).
- *Rappresentazione principale:* **kernel di rumore g distinto dal kernel di risposta**, con la stessa forma MHG (b = 0,35) e L_sr,n = 1,2 W, quindi σ_n = 0,5944 W: **0,773 mm in A** e **0,892 mm in B**.
  - g è troncato a 4 σ_n, discretizzato per cella e **tarato sempre** per asse a L_sr,disc = 1,2 W (tolleranza 1 %), come nel punto (3). Valori indicativi con W = 1,3: 0,731 mm lungo le righe e 0,702 mm lungo le colonne.
  - Per ogni componente: rumore bianco N(0, 1) indipendente per nodo, su una griglia estesa di R_n = 4 σ_n oltre il campo, così il rumore è stazionario fino ai bordi e non serve troncarlo; convoluzione con g; divisione per √(Σ g²) per avere varianza unitaria; moltiplicazione per σ_u,c; campionamento sui nodi validi.
  - Nel modello lineare di Wieneke l'integrale dell'autocorrelazione del rumore vale 1/∫g², quindi ≈1,2 W per costruzione. La stessa taratura discreta del punto (3) vale per σ_n, con tolleranza del 5 %.
  - Valori attesi (continuo, per asse): ρ(0,5 W) ≈ 0,68, ρ(W) ≈ 0,15, ρ(1,5 W) ≈ −0,07. Con passo d = W/2 (overlap 50 %) si ha ρ(2d) ≈ 0,15, dello stesso ordine del valore "rappresentativo" 0,11. È una coerenza, non una verifica. Sulla griglia anisotropa si riportano ρ ai ritardi effettivi dei nodi (0,581 e 0,758 mm e multipli).
  - Le componenti u e v sono indipendenti fra loro (ipotesi; WIE17 §4.3 la adotta).
- *Sensibilità SENS-ρ0:* rumore indipendente per nodo. **È un caso limite di modellazione, non l'equivalente del risultato della review**: ρ = 0 a una distanza (2h) non implica indipendenza a tutte le distanze.
- *Incertezze:* la divergenza fra 0 e 0,11 al 50 % non è risolta. La procedura per chiuderla è al §5, D-ρ.

**(6) Bias**
Non incluso. Slip ≤2 % stimato a priori (STO12) o trascurabile (OBE15); nessun bias trasferibile documentato in fiamma atmosferica (SINT §2.2). L'esclusione è dichiarata come scelta.

**(7) Regioni mancanti, rigetti e sostituzioni**
- *Modello principale:* in A e B le uniche regioni senza vettori sono **fuori dal campo e nella fascia di bordo**. Il modello **approssima** il dato GTMC come rilasciato: campi completi, con buchi di un solo vettore interpolati (BOX10 p. 31).
- *Che cosa questa approssimazione non dimostra:* un campo sperimentale completo dopo interpolazione **non dimostra** che rigetti e sostituzioni siano senza effetto.
  - I valori sostituiti sono stime locali, non misure, e lo diventano in modo invisibile se manca un flag (SINT §2.1).
  - Possono concentrarsi dove la misura è più difficile: più vettori interpolati nella zona di reazione (BAU15 p. 127 stampata).
  - Il modello principale **trascura** questi effetti perché frequenza e distribuzione in fiamma swirl atmosferica non sono documentate.
  - Le sensibilità qui sotto ne esplorano **alcune conseguenze** (assenza esplicita contro sostituzione non segnalata; rigetti isolati contro a cluster), **non** la loro entità reale.
- *Rigetti interni:* esistono certamente prima della validazione, ma **non sono quantificati** in fiamma swirl atmosferica (SINT §3.2). Per questo non entrano nel modello principale.
- *Sensibilità non calibrate SENS-REJ*, applicate a PIV-A con le stesse posizioni nei due livelli di dato:
  - **REJ-iso:** rigetti isolati, uniformi sui vettori validi;
  - **REJ-clu:** rigetti a cluster, cioè componenti connesse di una soglia su un campo gaussiano correlato su scala W, alla stessa frazione;
  - frazione unica di **10 %** dei vettori nel campo. È il limite superiore dichiarato nell'unica fonte in fiamma con un numero ("always less than 10 %" nelle zone inseminate, microPIV H₂, BAU15 p. 49 stampata). È etichettato **non calibrato e non trasferito**: un'altra configurazione, citata come scenario, non come frequenza della nostra fiamma;
  - due livelli di dato:
    - **flag:** mancanti espliciti (U-Net: Mv = 0; GLU: token rimosso);
    - **sub:** valori sostituiti con media iterativa 3×3 dei vicini validi (pratica TUM e GTMC) e presentati come misurati (Mv = 1; token presente).
- **Nessuna percentuale isoterma** (Cardiff) è trasferita. Cardiff serve solo, se autorizzato, a verificare che la regola di sostituzione si comporti come una catena reale (F2.5).
- *Rigetti condizionati (SENS-OOP):* perdita di coppie per moto fuori piano, guidata da |vk|. Solo se F1.1 stabilisce dalla provenienza che k è normale alla slice. Legge da fissare dopo la lettura di Keane & Adrian (SINT G7). Altrimenti non si usa.

**Ordine delle operazioni:**
1. velocità LES in m/s, ricostruita dalla cache con le statistiche congelate;
2. convoluzione di risposta con il kernel discreto tarato, limitata ai nodi del campo e rinormalizzata vicino ai bordi;
3. selezione dei nodi validi (regola della finestra);
4. errore correlato;
5. rigetti (solo SENS-REJ);
6. sostituzione (solo `sub`);
7. normalizzazione congelata;
8. pacchetto U-Net/GLU.

**Nessun accesso a T** in tutte le configurazioni di questa roadmap: la firma del generatore non contiene T.

### 2.4 Seed delle realizzazioni

Nuovo namespace, distinto da quelli storici (20261114/16/18) e della 00 (20261125, 20261201, 20261202):
- tre seed separati per TRAIN, SELEZIONE e VALUTAZIONE;
- chiave (namespace, seed, componente, sim_id, frame, replica o epoca);
- **senza** famiglia né seed del modello, così U-Net e GLU ricevono realizzazioni identiche;
- in training una nuova realizzazione dell'errore per (frame, epoca);
- in valutazione 3 repliche.

### 2.5 Che cosa cambia rispetto alla missione 00

| Aspetto | Missione 00 | Questa roadmap |
|---|---|---|
| Layout | 819 siti casuali (5 %) | Nodi validi nel campo (A = 15 624; B = 1 824 per posizione) |
| Valore | Puntuale esatto | Media pesata su W (1,3 / 1,5 mm) |
| Errore | i.i.d. 0,5 / 1,5 m/s, livelli di scenario | Correlato, con lunghezza ≈1,2 W; σ dal progetto di Δt, per condizione |
| Lacune | Dischi al 25 % dell'area | Fuori dal campo e bordo; rigetti solo come sensibilità, perché un dato rilasciato completo non ne dimostra l'assenza |
| Sostituzione | Mai | Sensibilità `sub` |
| Selezione | VAL pulita | VAL della stessa osservazione |
| Ricette | GLU originali batch 8 contro specialisti microbatch 1 | Stessa ricetta per tutte le celle; ponte con gli originali |

### 2.6 Elenco delle sensibilità

| ID | Che cosa varia | Uso |
|---|---|---|
| SENS-TH | Top-hat tarato a L_sr,disc = W invece di MHG | Valutazione; training T10 solo con la regola R4 (decisione_R1 §5) |
| SENS-SWAP | Passi di cella scambiati (convenzione storica colonne ↔ x) | Qualifica F2.6 e valutazione in Fase 3 |
| SENS-EDGE | Esclusione stretta: solo vettori con supporto R_r interamente nel campo | Valutazione |
| SENS-σlo / SENS-σhi | σ_px 0,06 / 0,15 | Valutazione |
| SENS-ρ0 | Rumore indipendente | Valutazione; **obbligatoria** accanto al principale su A e B-in (R2) |
| SENS-TAIL | Code pesanti: trasformazione marginale a t di Student, ν = 7 (eccesso di curtosi 2), stessa σ e stessa realizzazione (v1.1, R2 §4) | Valutazione, A e B-in |
| SENS-REJ-{iso,clu}-{flag,sub} | Rigetti non calibrati al 10 % | Valutazione; training condizionato (gruppo T8) |
| SENS-REJ-{iso,clu}-sub5 | Stessi rigetti, sostituzione con media iterativa 5×5 (area dichiarata nell'XML Cardiff, non validata; v1.1, R2 §5) | Valutazione |
| SENS-B-pos | Posizioni B-down e B-off | Valutazione |
| SENS-W | W = 1,0 mm (OBE15 10 kHz; BOX10) e W = 1,9 mm (KAR20, configurazione non GTMC) | Solo valutazione: test di configurazioni non viste |
| SENS-OOP | Perdita di coppie da vk | Condizionata a F1.1 |

### 2.7 Ambiguità corrette rispetto a SINT

1. "±" pubblicato ≠ σ: la σ principale è scelta dall'intervallo delle evidenze (0,06–0,15 px), non letta da "±0,1 px" (§2.3, punto 4).
2. ρ = 0 a una distanza ≠ indipendenza: SENS-ρ0 è un caso limite (§2.3, punto 5).
3. Risposta e autocorrelazione dell'errore sono **due kernel separati**, con lunghezze diverse: 0,93 W e 1,2 W.
4. Un'eventuale verifica su Neal (F2.4) qualifica la forma dell'errore **in un getto non reagente** con il processamento scelto. Non certifica una PIV reagente.
5. Nessuna percentuale di rigetto isoterma è trasferita alla combustione. Il 10 % di SENS-REJ è uno scenario dichiarato, non una frequenza.
6. Il supporto del kernel non è ±W/2: è 4 σ_r ≈ 1,84 W. L'esistenza del vettore segue invece la finestra ±W/2 (§2.3, punto 3).
7. Un campo rilasciato completo non dimostra l'assenza di effetti da rigetto e sostituzione (§2.3, punto 7).

---

## 3. Fase 1–2: verifiche preliminari, implementazione, qualifica

Ogni attività riporta ID, dipendenze e criterio di completamento. Tutto si esegue **solo** su TRAIN e sui metadati, salvo dove indicato; VAL non è necessaria in queste fasi.

### Fase 1 — Verifiche sui dati (**completata**: [fase1_report_R1.md](fase1_report_R1.md); R1 chiusa con [decisione_R1.md](decisione_R1.md))

**F1.1 Orientamento e componenti dalla provenienza** (coordinamento con la missione 01).
1. Raccogliere dalla provenienza:
   - nomi dei canali `Velocity[i/j/k]` in `combustion__channels.json`;
   - convenzione STAR-CCM+ per le componenti vettoriali nel sistema di riferimento del modello, dalla documentazione ufficiale del software (verificare la fonte, non assumerla);
   - paper RealPDEBench (§3.2, App. B.1–B.2) e card;
   - codice di conversione pubblico di RealPDEBench, se esiste, per piano di taglio, sistema di coordinate e ricampionamento;
   - significato fisico degli array `x`, `y` (le y partono da 62,9 mm: offset rispetto a quale origine?).
2. Controllo **ausiliario**, mai prova: confronto fra lo spostamento apparente delle strutture scalari tra frame consecutivi (0,25 ms) e (vi, vj, vk), sui getti freddi di 2–3 condizioni TRAIN. T e specie si usano solo per questa qualifica, mai come input.
3. Se la provenienza non basta, preparare per Enrico una domanda per gli autori (non inviarla).

*Completamento:* nota `orientation_note.md`, con esito classificato come **confermato dalla provenienza**, **probabile** (provenienza parziale e controllo ausiliario coerente) o **ignoto**, insieme alle conseguenze per 2C e SENS-OOP. Aggiornamento proposto per la missione 01, non applicato senza il suo coordinamento.

**F1.2 Supporto della griglia alle scale PIV.**
- Su TRAIN: spettri 1D (x e y) e funzioni di struttura di vi e vj, per condizione.
- Per W ∈ {1,0; 1,3; 1,5; 1,9} mm e per i due kernel (MHG, top-hat): RMS e p95 di |u − u_filtrato| in m/s, globale e per regione. Le regioni si definiscono **solo su velocità**: alto gradiente di velocità, getti ad alta velocità. Una ripartizione per regioni di T è ammessa solo come diagnostica dichiarata.
- Indicatori di risoluzione effettiva del campo esportato, per esempio la caduta spettrale vicino a Nyquist.

*Completamento:* rapporto con tabelle e figure; confronto dell'effetto del filtro con σ_u (§2.3, punto 4); raccomandazione per D-kern (§5).

**F1.3 Statistiche per condizione.** v_ref,c per tutte le condizioni TRAIN e VAL (p99,9 di |u_planare|) e σ_u risultante.
*Completamento:* tabella versionata.

### Fase 2 — Implementazione e qualifica

**F2.0 Aggiornamento dei preliminari dopo R1** (CPU; solo TRAIN e sola velocità di VAL; nessun TEST).
- v_ref,c sull'intero record (frame 0:2001:10), TRAIN e VAL, con IC95 a blocchi, e σ_u per σ_px 0,06/0,10/0,15.
- Registro dei kernel (MHG di risposta, rumore, top-hat) con passi righe 0,581 / colonne 0,758 mm, taratura all'1 %, per W ∈ {1,0; 1,3; 1,5; 1,9}; più il registro SENS-SWAP.
- Contenuto delle finestre B-in, B-down e B-off (decisione_R1 §8).
- Indice che renda le tabelle `f12_swapcells` il riferimento F1.2.

*Completamento:* tabelle versionate in `results/piv/phase2/f20/` con provenienza.

**F2.1 Specifica versionata** `piv_observation_spec_v1` (YAML con sha256).
- Contenuto: configurazioni A, B, IDEAL-*, SENS-*; parametri e fonti per ogni parametro; ordine delle operazioni; namespace e seed; regola delle posizioni di B.
- Le categorie per ogni parametro: *documentato*, *trasferimento*, *sensibilità*.

*Dipende da:* F2.0 e decisione_R1. Contiene esplicitamente: assegnazione righe/colonne e passi; componenti; σ tarati; v_ref,c; posizioni B; definizioni diagnostiche.

**F2.2 Generatore** (modulo nuovo, per esempio `src/combustion_recon/data/piv_observations.py`).
- Riusare `CanonicalSparseObservation`, il pacchetto U-Net a 7 canali e i token GLU senza modificare il comportamento di `nonideal_observations.py`.
- Firma **senza T**.
- Output identico per le due famiglie: stessi nodi, stessi valori, stesso hash.
- Pacchetti `flag` e `sub` distinti, con metadati (nodi rigettati, nodi sostituiti, campo, configurazione).

*Completamento:* test CPU che passano (F2.3).

**F2.3 Test unitari.**
- Identità: IDEAL-full riproduce bit a bit la cache densa; IDEAL-A e IDEAL-B riproducono la cache sui nodi validi di PIV-A e PIV-B.
- Nodi validi: PIV-A e IDEAL-A hanno lo stesso insieme di nodi (hash uguale); lo stesso per PIV-B e IDEAL-B in ogni posizione.
- Pesi del kernel: somma 1 dopo troncamento e dopo la rinormalizzazione di bordo; L_sr,disc entro il 5 % del target per asse.
- Trasferimento del filtro su campi sinusoidali in mm, contro H(k) discreta attesa, separatamente per vettori interni e di bordo.
- σ e autocorrelazione empiriche del rumore contro i target (L_sr, ρ(d), ρ(2d), ρ(3d) lungo righe e colonne), su molte realizzazioni.
- Regola della finestra e margini; SENS-EDGE; geometria delle tre posizioni di B.
- Stazionarietà del rumore fino ai bordi (griglia estesa).
- Frazione e connettività dei rigetti in SENS-REJ.
- Regola di sostituzione.
- Determinismo e separazione dei namespace.
- Equivalenza U-Net/GLU.
- Assenza di accesso a T: test di firma più test di regressione con T perturbata.

**F2.4 (opzionale, download autorizzato: §10) Verifica metrologica su Neal.**
- Elaborare le immagini MS con un software aperto documentato (finestre 16 e 32 px; overlap 50 % e 75 %).
- Allineare all'HDR (kernel di 5 coppie).
- Stimare distribuzione e ρ(d), ρ(2d) dei residui, per componente e direzione.
- Confrontare la lunghezza integrale dei residui con la relazione rumore–finestra usata in §2.3, punto 5.

*Esito:* qualifica la forma dell'errore e la relazione con l'overlap **in un getto freddo, con il processamento scelto e con i limiti del riferimento HDR** (kernel di 5 coppie, outlier interpolati non marcati, ricampionamento a 1 px). Non certifica una PIV reagente. Serve solo a chiudere D-ρ. I dati non diventano dati di training.

**F2.5 (opzionale, download autorizzato: §10) Cardiff, processamento e sostituzione.**
- Validare il decoder non ufficiale: media dei 514 campi decodificati contro la statistica `V.*` salvata da DynamicStudio; regione NaN del primo stadio contro la maschera `D.4xd8dily`; riproduzione degli sha256 dei 12 file già estratti dallo scout.
- Descrivere sui 514 frame come la catena reale trasforma i vettori fra i tre stadi (transizioni dei codici, cambiamenti dei valori, distanza dei sostituiti dalla media dei vicini).
- Confrontare la regola di sostituzione di F2.2 (media iterativa 3×3) con quel comportamento.

*Esito:* qualifica del processamento e della regola di sostituzione **in una PIV isoterma**. Frazioni e morfologia dei codici si riportano come descrizione di quel dataset. **Nessuna frequenza è trasferita** alla combustione né a SENS-REJ. I dati non diventano dati di training.

**F2.6 Qualifica delle osservazioni generate** (su `train_fit`, 760 frame, e su 3 condizioni TRAIN rappresentative).

*Statistiche:*
- conteggi e copertura;
- risposta realizzata sulla griglia discreta (L_sr,disc e H(k)), **separata per vettori interni e di bordo**, confrontata con il target;
- σ_u per condizione;
- autocorrelazione dell'errore realizzata;
- differenza RMS fra campo ideale e osservato, per componente e per regione di velocità;
- per SENS-REJ: frazioni e distribuzione delle dimensioni dei cluster;
- SENS-TH (tarato): differenza dei campi rispetto al MHG, per regione (descrittiva; il criterio D-kern non si applica più);
- SENS-SWAP: stesse statistiche con i passi scambiati;
- contenuto delle tre posizioni B.

*Figure:* per 3 condizioni × 2 frame, pannelli vi e vj ideale | filtrato | osservato A | osservato B (B-in, B-down, B-off) | errore | maschere; istogrammi; autocorrelazioni.

*Tabella di confronto:* proprietà sostenuta dalle fonti (SINT §2.2) → valore prodotto → esito.

*Completamento:* rapporto `piv_observation_qualification.md` con figure. Revisione **R2** superata.

**F2.7 Profilazione GPU** (smoke non scientifico, ≈2–4 GPU-h, dopo R2).
- U-Net e GLU su A, B e IDEAL-A: tempo per epoca, VRAM, tempo di valutazione.
- GLU con **tutti** i nodi validi di A (≈15 600 token) a microbatch 1 con accumulo.

*Completamento:* tabella dei costi e proposta per D-GLU (§5).

---

## 4. Fase 3 — Test dei checkpoint esistenti

### 4.1 Modelli

- 12 originali della 00: U-Net e GLU, p_on e p_off, seed 0/1/2. Checkpoint `best` già identificati in `degradation_spec_v1_12models.yaml`.
- U-Net densa del Mese 2: U4 (p_on, seed 0) e U6 (p_off, seed 0). Non hanno il canale di maschera: si valutano solo su osservazioni definite su tutti i nodi.
- Specialisti della 00: non inclusi.

### 4.2 Osservazioni di test e contrasti

**Regola:** un contrasto che dichiara di isolare filtro o errore usa **gli stessi nodi validi** e le stesse realizzazioni su entrambi i lati. Solo i contrasti di layout e di copertura cambiano i nodi, e lo fanno per definizione.

| ID | Nodi | Valori | Errore | Modelli |
|---|---|---|---|---|
| E0 | 819 siti storici | Ideali | — | 12 originali. Replay R0: gate ≤0,05 K macro contro la 00 |
| E0v | Siti storici ∩ nodi validi di A | Ideali | — | 12 originali |
| E1 | Come E0v | Filtrati (A, W = 1,3) | — | 12 originali |
| E2 | Come E0v | Filtrati | Errore A | 12 originali |
| E3 | Tutti i 16 384 nodi | Ideali | — | 12 originali, U4, U6 |
| E3a | Nodi validi di A | Ideali | — | 12 originali |
| E4 | Nodi validi di A | Filtrati | — | 12 originali |
| E5 | Nodi validi di A | Filtrati | Errore A | 12 originali (**PIV-A**) |
| E6 | Nodi validi di B (B-in, B-down, B-off) | Ideali | — | 12 originali |
| E7 | Come E6, stessa posizione | Filtrati (W = 1,5) | Errore B | 12 originali (**PIV-B**) |
| E4*, E5* | Tutti i 16 384 nodi | Filtrati con kernel troncato e rinormalizzato anche entro W/2 dal bordo; E5* con errore | Come E4, E5 | Solo U4, U6 (diagnostica: non hanno canale di maschera) |

**Contrasti** (stessi frame e stesse realizzazioni):

| Contrasto | Nodi | Che cosa isola |
|---|---|---|
| E0v − E0 | Diversi (siti di bordo rimossi) | Effetto della sola rimozione dei siti di bordo; atteso piccolo, riportato |
| E1 − E0v | Uguali | Filtro **a layout storico fisso**: ponte diretto con la 00 |
| E2 − E1 | Uguali | Errore PIV correlato a layout storico fisso; confrontabile con N1/N2 della 00 |
| E3 − E0 | Diversi per definizione | **Cambio di layout** (densità), valori ideali |
| E3a − E3 | Diversi (fascia di bordo) | Effetto della fascia di bordo esclusa |
| E4 − E3a | Uguali | Filtro su layout denso |
| E5 − E4 | Uguali | Errore su layout denso |
| E6 − E3a | Diversi per definizione | **Copertura** con valori ideali |
| E7 − E6 | Uguali (per posizione) | Misura dentro il campo ridotto |
| E4* − E3, E5* − E4* | Uguali | Filtro ed errore per le U-Net dense (diagnostica) |

Sensibilità in sola valutazione, dopo E0–E7: **SENS-ρ0 (obbligatoria, sempre accanto a E5 ed E7)**, SENS-TH, SENS-SWAP, SENS-EDGE, SENS-σlo/hi, SENS-TAIL, SENS-REJ (4 combinazioni più 2 `sub5`), SENS-B-pos, SENS-W. Gli effetti del filtro (E1 − E0v, E4 − E3a) si leggono con la cautela di decisione_R2 §2: una parte rimuove artefatti di esportazione della LES. Ognuna è confrontata con la configurazione principale **sugli stessi nodi**. Per SENS-EDGE il confronto si fa sull'intersezione dei nodi validi.

**Interpretazione.** I modelli al 5 % su E3–E7 sono **fuori distribuzione**: U-Net con maschera quasi piena, GLU con migliaia di token. Questi test misurano la robustezza di modelli costruiti per un'altra osservazione, non l'informazione contenuta nella PIV. Il GLU originale a 819 token potrebbe richiedere molta memoria con input densi: la profilazione F2.7 decide se valutarlo a microbatch 1. Nessuna riduzione silenziosa dei token.

**Costo stimato:** ≈5–15 GPU-h, scalando dal Blocco II (1,56 GPU-h per 12 modelli × 6 celle × 3 repliche), con un margine per il GLU denso.

**Completamento:** pacchetto con tabelle per frame, condizione e macro, gate di replay superato, figure appaiate. Revisione **R3**.

---

## 5. Decisioni aperte e procedura di chiusura

| ID | Decisione | Procedura | Chiusura |
|---|---|---|---|
| D-ori | 2C in piano? SENS-OOP sì o no? | F1.1 | **Chiusa a R1:** esito *probabile*, 2C = (vi, vj), SENS-OOP disattivata (decisione_R1 §2). Regola originale: con esito **confermato** o **probabile** si procede con 2C. Con esito **ignoto** si procede con 2C come ipotesi dichiarata e SENS-OOP viene escluso |
| D-kern | Kernel principale MHG o top-hat | F1.2 | **Chiusa a R1:** MHG principale; top-hat tarato solo come sensibilità di valutazione; T10 deciso a R4 se i modelli T2 su SENS-TH perdono più della deviazione standard fra seed di T2 (decisione_R1 §5). Regola originale, sostituita: se la differenza fra i due filtrati è piccola rispetto a σ_u (RMS < 0,25 σ_u nelle regioni ad alto gradiente), MHG resta principale e il top-hat si valuta solo nella Fase 3. Altrimenti il top-hat entra anche in training (gruppo T10) |
| D-ρ | Correlazione principale | F2.4 se autorizzata | **Chiusa a R2:** Neal non conferma il modello al primo ritardo, ma con un processamento diverso (single-pass) e un residuo che include il riferimento; principale invariato, SENS-ρ0 obbligatoria (decisione_R2 §3). Regola originale: R2: se i residui Neal al 50 % mostrano una correlazione incompatibile con il kernel di rumore, si rivede il kernel e lo si registra (resta qualifica non reagente). Senza F2.4: principale invariato, SENS-ρ0 riportato sempre |
| D-σ | σ_px | Nessuna nuova misura possibile sulla nostra fiamma | Congelato a 0,10 px con sensibilità 0,06 e 0,15 |
| D-REJ | Training con rigetti | Fase 3 e risultati di T1/T2 | R4: si allena (T8) solo se la valutazione dei modelli T2 su SENS-REJ mostra una perdita superiore alla dispersione fra seed della cella T2 |
| D-GLU | Layout del GLU | F2.7 | R3: con costo stimato per run GLU su A ≤ 10× la 00 (≤ 90 GPU-h) si usano **tutti** i nodi. Altrimenti si propone un layout ridotto dichiarato (per esempio passo di 2 nodi, che corrisponde a un overlap più basso), applicato **a entrambe le famiglie** come configurazione separata A2 con cella U-Net corrispondente. Mai solo al GLU |
| D-mix | Training su più configurazioni | Valutazioni incrociate T1–T4 | R4: si allena (T9) solo se il costo di trasferimento A→B o B→A supera la dispersione fra seed |

---

## 6. Punti di revisione

| ID | Quando | Che cosa si guarda | Chi decide |
|---|---|---|---|
| **R0** | Ora | Questa roadmap | Enrico |
| **R1** | Dopo F1.1–F1.3 | Orientamento, supporto della griglia, v_ref; chiusura di D-ori e D-kern; congelamento della specifica v1 | **Chiusa** il 27/11/2026: [decisione_R1.md](decisione_R1.md) (mandato tecnico delegato da Enrico) |
| **R2** | Dopo F2.6 (ed eventualmente F2.4–F2.5) | **Osservazioni generate**: figure, statistiche, confronto con le proprietà delle fonti. Nessuna rete prima di R2 | **Chiusa** il 28/11/2026: [decisione_R2.md](decisione_R2.md) (mandato tecnico delegato da Enrico) |
| **R3** | Dopo la Fase 3 e F2.7 | Risultati dei checkpoint esistenti; D-GLU; conferma della matrice di training | Enrico |
| **R4** | Dopo T1–T7 e le valutazioni | Risultati; D-REJ, D-mix, D-kern-train; preregistrazione per TEST | Enrico |
| **R5** | Eventuale | Apertura di TEST, una sola volta | Enrico |

---

## 7. Fase 4 — Campagna di training e valutazione

### 7.1 Lezioni della missione 00 applicate

1. **Ricette confrontabili.** Tutte le celle usano la ricetta S7: AdamW 1e-3, wd 1e-4, clip 1, batch effettivo 8, 760 presentazioni/epoca, warm-up 950, ReduceLROnPlateau, fp32. Early stop: dopo 200 epoche, pazienza 15 valutazioni, miglioramento relativo 10⁻³. Budget: 1000 epoche più un'unica continuazione fino a 1200, con `budget_exhausted` etichettato. Pool di training: C149 (2980 frame).
2. **Batching GLU uniforme.** Microbatch 1 con accumulo a 8 per **tutte** le celle GLU, ideali comprese. Ponte T7 con gli originali.
3. **Selezione coerente.** `best` = primo minimo della RMSE macro su VAL **della stessa osservazione** (namespace SELEZIONE, 3 repliche). Si riportano sempre anche `last` e la curva.
4. **Variabilità.** Nella 00 la dispersione fra seed delle baseline sulla VAL era di circa 4 K. Quindi 5 seed (0–4) per la U-Net e 3 seed (0–2) per il GLU, più costoso. Stessa inizializzazione e stesso stream di presentazioni per indice di seed fra le celle.
5. **Contesto:** solo p_on (NH₃ e φ), come gli specialisti. p_off non fa parte della campagna.
6. **Nessun campione ideale** nelle celle PIV: il compito è la PIV. Il controllo ideale è una cella separata.

### 7.2 Matrice

**Necessari**

| Gruppo | Cella | Osservazione di training | Famiglia × seed | Run | Perché |
|---|---|---|---|---|---|
| T1 | IDEAL-A | **Stessi nodi validi di PIV-A**, valori ideali | U-Net × 5 | 5 | Controllo di T2 con la stessa ricetta e gli stessi nodi. Ancoraggio a U4 tramite la valutazione su E3 |
| T2 | PIV-A | §2.3, configurazione A | U-Net × 5 | 5 | **Cuore della missione**: ricostruzione da PIV a pieno campo |
| T3 | IDEAL-B | **Stessi nodi validi di PIV-B** (posizione B-in), ideali | U-Net × 5 | 5 | Controllo di T4; con T1 separa la copertura dalla qualità dei vettori |
| T4 | PIV-B | Configurazione B, posizione B-in | U-Net × 5 | 5 | PIV ad alto rateo, campo ridotto |
| T5 | IDEAL-A, PIV-A | Come T1 e T2 | GLU × 3 + 3 | 6 | Le conclusioni valgono anche per la seconda architettura? Stesse osservazioni fisiche della U-Net (D-GLU) |
| T6 | IDEAL-B, PIV-B | Come T3 e T4 | GLU × 3 + 3 | 6 | Come T5; ≈1 800–1 900 token |
| T7 | Ponte 5 % ideale | Siti casuali al 5 %, come gli originali | GLU × 3 | 3 | Separa la ricetta (batching) dall'osservazione; collega la campagna alla 00 |
| | | | **Totale** | **35** | |

**Conseguenze della correzione sui nodi.** Il controllo ideale di A passa da IDEAL-full (tutti i nodi) a **IDEAL-A** (stessi nodi validi di PIV-A); quello di B usa gli stessi nodi validi di PIV-B. Il numero di run **non cambia**: 35 necessari, come nella versione approvata. IDEAL-full non si allena:
- differisce da IDEAL-A solo per la fascia di bordo (2 nodi lungo le righe e 1 lungo le colonne, per lato);
- il suo effetto si misura in valutazione (E3a − E3);
- l'ancoraggio al Mese 2 resta affidato a U4 e U6 e alla valutazione dei modelli T1 su E3.

Nessun training è stato aggiunto.

**Condizionati**

| Gruppo | Cella | Condizione (§5) | Run |
|---|---|---|---|
| T8 | PIV-A + REJ-clu, nei livelli `flag` e `sub` | D-REJ | U-Net 5 + 5 |
| T9 | Più configurazioni: A e B con posizioni campionate | D-mix | U-Net 5 (+ GLU 3 solo se la U-Net mostra un beneficio) |
| T10 | PIV-A con top-hat tarato, oppure PIV-A senza errore (solo filtro), sugli stessi nodi validi | Regola R4 su SENS-TH (decisione_R1 §5), o se la Fase 3 mostra filtro ed errore entrambi rilevanti e non separabili | U-Net 5 |
| T11 | A2 (layout ridotto) | Solo se D-GLU lo richiede | U-Net 5 + GLU 3 |

### 7.3 Valutazione

Ogni modello T1–T11 viene valutato con `best` e `last` su seen220 e VAL123, con 3 repliche, su tutte le osservazioni E3–E7 e sulle sensibilità del §2.6. Per B si usano le tre posizioni.

Ne risulta una **matrice incrociata**, in cui righe e colonne hanno i ruoli seguenti:
- *diagonale* (training = test): l'informazione utilizzabile con quell'osservazione;
- *fuori diagonale*: il costo di un operatore sbagliato e il trasferimento fra configurazioni;
- *T1 testato su E5, contro T2 su E5* (stessi nodi): il valore del training coerente con la misura;
- *T1 su E3a contro T2 su E5*: la perdita d'informazione attribuibile a filtro ed errore con modelli allenati per ciascuna osservazione.

**Metriche** (in K, sul campo 128×128, con l'aggregazione della 00: frame → condizione → media uniforme → repliche):
- RMSE, bias, RMSE centrata, MAE;
- zona calda (T ≥ 2100 K): RMSE e IoU; getti (T < 1800 K): RMSE; errore su picco e p95;
- distanza media fra le isoterme vere e predette a 1800 K e 2100 K, definita e congelata prima delle valutazioni;
- per B: dentro e fuori dal campo, ed errore in funzione della distanza dal vettore valido più vicino (classi di distanza congelate prima);
- per SENS-REJ: errore nei nodi rigettati o sostituiti.

Tutto è riportato per condizione e per seed.

**Unità di replica e lettura.**
- Le affermazioni sui training si basano sui seed: media, valori per seed, deviazione standard.
- Una differenza fra celle minore della deviazione standard fra seed del controllo corrispondente non si interpreta come ordinamento.
- Condizioni (3 in VAL) e frame correlati: bootstrap gerarchico descrittivo.
- Le repliche di rumore non sono campioni indipendenti.
- Nessun test di significatività fissato a priori.

**VAL e TEST.** VAL è già usata per la selezione: i risultati di questa fase sono **di sviluppo**. Dopo R4 si scrive una preregistrazione (celle, contrasti, metriche primarie) e TEST si apre una sola volta, su decisione R5.

### 7.4 Costi stimati

Ipotesi:
- U-Net: 1–5 GPU-h per run. Estremo basso dalla profilazione densa del Mese 2 (28,5 min per 730 epoche); estremo alto dagli specialisti della 00 (4,5 GPU-h, con valutazioni ogni 10 epoche e preemption). Gli input densi non cambiano il costo della U-Net.
- GLU: 8,9 GPU-h per run a 819 token (00). A ≈15 600 token da ×1 a ×10, da misurare in F2.7; per B, ≈1 800–1 900 token, da ×1 a ×2.
- I tempi di valutazione includono generazione delle osservazioni, I/O e metriche: non sono latenze d'inferenza.

| Voce | Run | GPU-h |
|---|---|---|
| F2.7 profilazione | smoke | 2–4 |
| Fase 3 | — | 5–15 |
| T1–T4 U-Net | 20 | 20–100 |
| T5 GLU (A, denso) | 6 | 53–534 |
| T6 GLU (B) | 6 | 53–107 |
| T7 ponte GLU | 3 | ≈27 |
| Valutazioni T1–T7 (best + last, matrice incrociata) | — | 20–60 |
| **Necessario** | **35** | **≈180–850** |
| T8 / T9 / T10 / T11 | 10 / 5–8 / 5 / 8 | 10–50 / 5–295 / 5–25 / 30–80 |

Riferimento: la 00 è costata 210,1 GPU-h per 30 run (95 % training), in circa 53 h di calendario con fino a 8 GPU.

---

## 8. Istruzioni operative per Sonnet IDE

1. **Branch:** lavorare su un branch dedicato da `piv/experiment-proposal` (nome da concordare); niente merge in `main`. Nessuna modifica a `nonideal_observations.py`, alle specifiche della 00, ai risultati storici o ai documenti delle missioni 00 e 01: si propongono gli aggiornamenti alla 01.
2. **Ordine vincolante:** Fase 1 → R1 (chiusa) → Fase 2 (F2.0 → F2.1 → F2.2–F2.3 → F2.4–F2.5 opzionali → F2.6) → R2 (chiusa) → F2.7 (subito) e F2.8–F2.9 con il controllo dell'hash fra macchine (decisione_R2 §8) → Fase 3 → R3 → Fase 4 (T1–T7) → R4 → condizionati. Non anticipare training o valutazioni neurali prima di R2.
3. **Divieti:** TEST mai letto (rifiutare i ruoli `test*` come nella 00); nessun uso di T nel generatore; nessun parametro scelto guardando metriche delle reti; nessuna riduzione di token o di nodi non dichiarata nella specifica.
4. **Download:** F2.4 e F2.5 sono **già autorizzati** (§10): non chiedere di nuovo l'autorizzazione. Le verifiche restano opzionali e motivate. Seguire esattamente §10: file, destinazione fuori dalla repository, checksum e manifest. Nessun altro download di dataset.
5. **Provenienza:** ogni artefatto registra SHA del codice, sha256 della specifica, della cache e delle statistiche, namespace e seed, configurazione, posizione di B, livello `flag` o `sub`.
6. **Riuso:** pipeline di valutazione, gate di replay, aggregazione, metriche e accounting della 00. L'accounting Slurm va prodotto con lo stesso metodo (`sacct -D`).
7. **Consegne per ogni fase:** un documento leggibile nella cartella della missione, più tabelle e figure in `results/piv/…`, con un indice.

---

## 9. Che cosa questa roadmap non dimostra

- La T di riferimento è numerica: **nessuna validazione sperimentale della T**.
- Le osservazioni rappresentano una classe di PIV documentata (GTMC), con trasferimenti dichiarati. Non sono una riproduzione del nostro combustore, la cui geometria è ignota.
- Fenomeni **non rappresentabili** con una slice: spessore del foglio, errori propri della stereo, rifrazione, luminosità di fiamma, traiettorie delle particelle.
- Fenomeni **documentati ma non quantificati**, quindi solo sensibilità o esclusi: rigetti in fiamma, densità di particelle nel gas combusto.
- Una qualifica su Neal o Cardiff riguarda flussi non reagenti o isotermi e non certifica una misura reagente.
- Il modello principale rappresenta il dato rilasciato completo: **non** dimostra che rigetti e sostituzioni in fiamma siano senza effetto. Le sensibilità ne esplorano alcune conseguenze, non l'entità reale.

---

## 10. Download autorizzati per le verifiche F2.4 e F2.5

**Autorizzazione.** Enrico, 27 novembre 2026, autorizza Sonnet a scaricare durante l'esecuzione locale o sul cluster i due insiemi di dati sotto, **solo** per le verifiche F2.4 e F2.5. Sonnet **non deve chiedere di nuovo** l'autorizzazione.
- Le verifiche restano **opzionali** e motivate dalle decisioni D-ρ (Neal) e dalla qualifica della regola di sostituzione (Cardiff).
- I dati **non** diventano dataset di training o di valutazione delle reti.
- I dati **non** certificano una misura reagente.
- Nessun altro download di dataset è autorizzato da questa voce.

Identificativi verificati il 27/11/2026 tramite le API pubbliche dei repository (solo metadati; nessun dato scaricato in questa revisione).

### 10.1 Neal et al., "unsteady inviscid core" (F2.4)

| Campo | Valore |
|---|---|
| Titolo | *PIV uncertainty - Database of the unsteady inviscid core of a transitional jet* |
| DOI e versione | 10.4121/13663376.v1 (4TU.ResearchData), UUID del record `309145fd-dcd3-47c4-8b88-8349d9e70cdb` |
| Licenza | CC0 |
| File | `Unsteady Inviscid core.zip`, **271 827 177 byte**, MD5 dichiarato dal repository `eb6d86228f534d6285859ccc0f16c369` |
| URL | `https://data.4tu.nl/file/309145fd-dcd3-47c4-8b88-8349d9e70cdb/426acdf9-5762-4fad-8d99-f8604bc69c86` |
| Che cosa serve | L'intero zip: immagini MS, campi HDR, `Database description.pdf`, `data description.txt` |
| Metodo | Download intero. Verificare dimensione e MD5 contro il repository; calcolare sha256 dello zip e dei membri estratti |

### 10.2 Cardiff, Runyon et al. (F2.5)

| Campo | Valore |
|---|---|
| Titolo | *Lean methane flame stability in a premixed generic swirl burner: isothermal flow and atmospheric combustion characterization* |
| DOI e versione | 10.17035/d.2017.0011500240; figshare article 27051964, versione 1 |
| Licenza | CC BY 4.0 (citare il dataset) |
| File sorgente | `d.2017.0011500240.zip`, file figshare **49264399**, 7 301 821 358 byte, MD5 dichiarato `a0dc9d9574a1ffdf14cc9634b97ddd52` |
| URL | `https://ndownloader.figshare.com/files/49264399` |
| Che cosa serve | **Non** l'intero archivio da 7,3 GB, ma i membri seguenti, estratti via HTTP Range (≈0,8 GB) |
| Membri | Sotto `d.2017.0011500240/PIVImages/20170303_143734/HiSense Mk II #1.4wewffbs/I.4xd8gr7l/`: `0h6c/data/AdaptivePIV#k.vectors`, `0h6c/bled/data/ValRange#k.vectors`, `0h6c/bled/h4h3/data/ValMovAvg#k.vectors` per k = 0…513 (514 × 3 file da 512 422 byte ≈ 790 MB); gli `AnalysisEnsemble.xml` dei tre stadi. Inoltre: `PIVImages/20170303_143734/Run.xml`; `HiSense Mk II #1.4wewffbs/AcquiredImageEnsemble.xml`; `README.txt` alla radice |
| Membri da individuare dal listing | Definizione della maschera (`D.4xd8dily`) e statistica DynamicStudio (`V.*`) del ramo principale. Il percorso esatto va letto dalla directory centrale dello zip e registrato nel manifest |
| Strumenti | `step2_datasets_remote_zip.py` (`list`, `extract`) e `step2_datasets_dynamicstudio_vectors.py` (decoder **non ufficiale**) da `piv/step2-dataset-scouting` @ `ee9948e227f0d9272ae064cc8601c5e5399c28b4`, cartella `mese3/02_virtual_piv/scouting/`. Copiarli con `git show` in una cartella di strumenti separata dai dati |
| Controllo di riproduzione | I 12 file già estratti dallo scout (frame 0, 1, 257, 513 ai tre stadi) devono riprodurre gli sha256 di `step2_datasets_samples_manifest.tsv` sullo stesso commit |
| Metodo | Per ogni membro: CRC-32 dalla directory centrale dello zip verificato dopo l'estrazione; sha256 calcolato |

### 10.3 Destinazione, sicurezza e provenienza

- **Destinazione fuori dalla repository**, per esempio `$HOME/piv_qualification_data/neal_unsteady_core/` e `$HOME/piv_qualification_data/cardiff_d2017/` (sul cluster o in locale). Mai committare dati.
- **Dati non fidati:** script in una cartella diversa dai dati; Python con `-I`; nessun eseguibile o script degli archivi viene eseguito.
- **Manifest di provenienza committato**, per esempio `results/piv/qualification/download_manifest.tsv`, con una riga per file o membro:
  - dataset, DOI e versione, URL;
  - data e ora UTC del download;
  - metodo (intero / HTTP Range);
  - percorso interno all'archivio;
  - byte;
  - MD5 dichiarato contro calcolato (dove esiste);
  - CRC-32 del membro;
  - sha256;
  - commit degli strumenti usati;
  - esito delle verifiche.
- Una discrepanza di dimensione o checksum blocca l'uso del file e va riportata, senza nuovi tentativi indiscriminati.
