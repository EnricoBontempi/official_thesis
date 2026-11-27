# Missione 02 — Decisione R1

**Data:** 27 novembre 2026 · **Branch:** `mission02/piv-execution` · **Consegna esaminata:** `a1cb3a07ade3df624549b4d261ab815a6121f13d`

**Letture:**
- [fase1_report_R1.md](fase1_report_R1.md) e [orientation_note.md](orientation_note.md);
- [results/piv/phase1/INDEX.md](../../results/piv/phase1/INDEX.md), con le tabelle di `f12/`, `f12_swapcells/` e `f13/` ricontrollate direttamente sui CSV;
- `scripts/piv_phase1/piv_kernels.py` e `f12_grid_support.py`;
- Stöhr et al. 2012 (manoscritto DLR, §§2.3.2–2.3.3), riletto per la posizione di B e per la scala di velocità.

**Stato:** R1 **chiusa**. Le scelte sotto sono vincolanti per la specifica v1 e per la Fase 2. La [roadmap tecnica](roadmap_tecnica_piv.md) è aggiornata di conseguenza (§0 "Decisioni R1" e sezioni interessate). Nessuna implementazione, training, inferenza o accesso a TEST in questa revisione.

---

## 1. In breve

| Tema | Decisione | Forza dell'evidenza |
|---|---|---|
| Componenti | Osserviamo **(vi, vj)**, interpretate come **trasversale nel piano (vi) e assiale (vj)**; vk fuori piano, non usata. SENS-OOP resta disattivata | Probabile (simmetria più coerenza con le etichette) |
| Coordinate e assi | **Righe ↔ coordinata `x`** (laterale, centrata sull'asse, passo 0,581 mm); **colonne ↔ coordinata `y`** (assiale, passo 0,758 mm). Il generatore usa questi passi; la convenzione precedente diventa la sensibilità SENS-SWAP | Probabile; scelta motivata, non provata |
| Kernel | MHG principale, tarato **sempre** sul target discreto (tolleranza 1 %), per asse, con i passi della riga sopra. Kernel di rumore tarato allo stesso modo | Evidenza (WIE17) più taratura numerica |
| Top-hat | Solo sensibilità di valutazione SENS-TH, **tarato** a L_sr,disc = W. Il criterio D-kern della roadmap è sostituito; T10 resta condizionato, deciso a R4 | — |
| Velocità di riferimento | p99,9 di \|u_planare\| sull'**intero record** di ogni condizione (frame 0:2001:10, 201 frame, tutti i nodi), solo velocità; σ_px = 0,10 px, d_max = 6 px | STO12: Δt scelto sulla scala di velocità dei campi istantanei |
| Posizione di B | **Finestra di training adiacente al bordo d'ingresso** (bordo destro), centrata sull'asse, come il campo GTMC "above the nozzle". Due posizioni solo di valutazione: a valle e decentrata | Evidenza (STO12 §2.3.3) più orientamento probabile |
| Definizioni diagnostiche | Fissate (§8) | — |
| Blocchi | **Nessuno.** Sonnet può eseguire F2.0–F2.6 (§9) | — |

---

## 2. Componenti utilizzabili

**Che cosa è stabilito dalla Fase 1.**
- I campi medi sono simmetrici rispetto alla riga centrale (riga 63,5): T e vj simmetrici, vi e vk antisimmetrici, con frazioni di energia 0,98–1,00 (orientation_note §3.1).
- Ne segue che j è assiale lungo le colonne e che {i, k} = {trasversale nel piano, fuori piano}.
- La simmetria non dice quale delle due sia quale. Il controllo sugli spostamenti non separa vi da vk (sono anticorrelati). La decorrelazione con |vk| (tracciante NH₃) è un indizio debole a favore di k fuori piano.

**Argomento aggiuntivo (coerenza, non prova).**
- Il vettore `x` è centrato sullo zero (−36,9…+36,9 mm), mentre `y` va da 62,9 a 159,1 mm. In un taglio per l'asse, la coordinata centrata sullo zero è quella **laterale**, e la riga centrale è proprio l'asse di simmetria.
- Se, come è naturale per un'esportazione da CFD, `Velocity[i]` e `Velocity[j]` sono le componenti lungo `x` e `y` (convenzione **non verificata** per STAR-CCM+), allora vi è laterale e vj assiale. È esattamente il ruolo che la simmetria assegna loro. Resterebbe k = z fuori piano.
- L'alternativa "vi fuori piano" richiede invece una permutazione non motivata dei nomi.
- Resta **un'incoerenza di segno**: i getti entrano dal bordo destro e scorrono verso le colonne decrescenti con vj > 0, cioè +j punta verso le colonne decrescenti (orientation_note §3.1–3.2). Segno e verso degli assi non sono quindi risolti, ma non influenzano il generatore né le reti.

**Decisione.**
- Osservazione **2C = (vi, vj)**, ipotesi "componenti nel piano" con esito *probabile*.
- vk non entra nelle osservazioni; aggiungerla cambierebbe il numero di canali, escluso dalla missione.
- **SENS-OOP non si attiva**: richiederebbe una provenienza che non c'è e una legge di perdita di coppie non letta.
- **Incertezza residua:** se vi fosse fuori piano, una delle due componenti osservate non sarebbe misurabile con una PIV planare 2C. Va dichiarato come limite nel report finale.
- La domanda agli autori (orientation_note §6) è **utile ma non bloccante**: si raccomanda a Enrico di inoltrarla. Se la risposta arriva prima di R2 e contraddice queste scelte, la specifica v1 si corregge prima della Fase 3. Se arriva dopo, si dichiara e si valuta l'impatto con SENS-SWAP.

## 3. Corrispondenza fra coordinate e assi degli array

**Decisione.** Per il generatore e per tutte le metriche in millimetri:

| Asse dell'array | Coordinata | Passo | Ruolo fisico (probabile) |
|---|---|---|---|
| **Righe** (indice 0) | `x`, −36,9…+36,9 mm | **0,581 mm** | Laterale; asse a x ≈ 0 (riga 63,5) |
| **Colonne** (indice 1) | `y`, 62,9…159,1 mm | **0,758 mm** | Assiale; ingresso dal lato delle colonne alte |

**Motivazione.** È l'unica assegnazione in cui coincidono la linea di simmetria osservata e lo zero della coordinata centrata, e in cui i ruoli di vi e vj coincidono con i nomi i/j letti come componenti cartesiane x/y (§2). La convenzione precedente (colonne = x), registrata dalla missione 01, era basata sull'ordine dei vettori nel loader, non su una prova fisica (orientation_note §5).

**Conseguenze.**
- Coincide con l'insieme di calcoli `f12_swapcells` già prodotto da Sonnet: le tabelle F1.2 di riferimento diventano quelle. Le conclusioni F1.2 non cambiano.
  - A W = 1,3 mm, RMS del filtro MHG globale: 0,135 m/s (vi) e 0,142 m/s (vj); in alto gradiente: 0,309 e 0,328 m/s, ricalcolati sui CSV `f12_swapcells`.
  - A W = 1,5 mm: 0,176 e 0,187 m/s globali; 0,399 e 0,433 m/s in alto gradiente.
- **Non si cambiano** le coordinate in ingresso alle reti (`x_scaled`, `y_scaled` e la geometria del GLU): fanno parte della pipeline e dei checkpoint esistenti, e usarle invariate mantiene la confrontabilità con la missione 00. È un limite dichiarato: la metrica dei vicini del GLU resta quella della convenzione storica.
- Le metriche in mm della valutazione (distanza dal vettore valido più vicino, distanza fra isoterme) usano i passi della tabella sopra.
- **SENS-SWAP** (sola qualifica F2.6 e valutazione in Fase 3): passi scambiati, cioè la convenzione storica.

**Proposta per la missione 01**, non applicata: annotare in INV C1/Q1 l'assegnazione righe ↔ `x` come *probabile*, con le prove di orientation_note §3.1 e di questo §2–3.

## 4. Kernel: definizione, discretizzazione, taratura

**Confermato:**
- forma MHG di WIE17 con b = 0,35;
- supporto 4σ con troncamento e rinormalizzazione;
- pesi integrati per cella (campo costante a tratti);
- regola della finestra per l'esistenza del vettore;
- kernel troncato al campo vicino ai bordi.

Le formule implementate da Sonnet in `piv_kernels.py` sono corrette (verifica numerica in `kernel_formula_verification.json`).

**Cambiato:**
1. **Taratura sempre, tolleranza 1 %.** La regola "taratura solo oltre il 5 %" lasciava alcuni assi esatti e altri al +2–4 % (per esempio W = 1,5 lungo il passo di 0,581 mm, W = 1,9). Tutti i kernel si tarano al target discreto entro l'1 %:
   - risposta L_sr,disc = 0,93 W;
   - rumore L_sr,disc = 1,2 W.
2. **Passi per asse** dalla §3: 0,581 mm lungo le righe, 0,758 mm lungo le colonne.

Valori indicativi della taratura già calcolata (`f12_swapcells/kernel_registry.csv`), da rigenerare con la tolleranza dell'1 %:

| W (mm) | Asse | Passo (mm) | σ_r tarato (mm) | σ_n tarato (mm) |
|---|---|---|---|---|
| 1,3 (A) | righe | 0,581 | 0,545 | 0,731 |
| 1,3 (A) | colonne | 0,758 | 0,531 | 0,702 |
| 1,5 (B) | righe | 0,581 | 0,644 → da ritarare | 0,892 → da ritarare |
| 1,5 (B) | colonne | 0,758 | 0,617 | 0,830 |

**Margini e conteggi.** Con la nuova assegnazione:
- W = 1,3 e 1,5 mm: margine di 2 nodi lungo le righe e 1 lungo le colonne;
- A: 124 × 126 = 15 624 vettori (righe × colonne);
- B: finestra di 52 righe × 40 colonne (30,2 × 30,3 mm), 48 × 38 = 1 824 vettori.

## 5. Ruolo del top-hat

**Che cosa ha mostrato F1.2.** Il top-hat "di lato W" discretizzato per cella ha L_sr,disc 1,2–1,5 volte quella del MHG tarato. Il criterio D-kern (RMS(MHG − top-hat) < 0,25 σ_u in alto gradiente) confrontava quindi **due larghezze diverse**:
- come applicato: mediane 0,51–0,68, criterio non soddisfatto (stessi esiti con i passi della §3);
- a larghezza pareggiata: 0,1–0,4 σ_u.

**Decisione.**
- **MHG resta il kernel principale.** Il sistema di riferimento GTMC usa una correlazione DaVis multi-pass/multi-scala, che WIE17 descrive con il MHG. Il top-hat rappresenta il single-pass, che non è l'algoritmo dell'apparato.
- **SENS-TH** è una sensibilità **di valutazione** sull'algoritmo. Il top-hat si **tara** a L_sr,disc = W (la sua lunghezza continua per definizione), con la stessa procedura della §4. Così il confronto riflette la forma e la differenza fisica fra single-pass (L_sr = W) e multi-pass (0,93 W), non un allargamento numerico.
- **Il criterio D-kern della roadmap è sostituito.** Una differenza fra campi filtrati non dice se la scelta del kernel conti per lo scopo della missione.
  - F2.6 ricalcola e riporta la differenza con il top-hat tarato, come descrizione.
  - **T10 resta condizionato** e si decide a R4: si allena solo se i modelli T2, valutati sulle osservazioni SENS-TH, perdono più della deviazione standard fra seed di T2 (RMSE macro VAL, `best`).
  - È un criterio sulla necessità di un training aggiuntivo, non sull'intensità del degrado.

## 6. Velocità di riferimento e ampiezza dell'errore

**Base.** STO12 §2.3.2 ricava lo slip da una "velocity scale v = 80 m/s… cf. Fig. 6b", e la Fig. 6 mostra **campi istantanei** (verificato). Il Δt (6–20 µs) è scelto per la condizione. La scala è quindi un massimo dei campi istantanei della condizione, che lo sperimentatore conosce per l'intera prova. Il p99,9 di |u| è un'approssimazione ragionevole di questo massimo.

**Problema rilevato.** Il p99,9 calcolato su pool diversi varia fino al 12–20 % per 0NH3_1.2 e 0NH3_1.25. Sono code pesanti (massimi fino a 58 m/s) ed eventi intermittenti distribuiti nel tempo. I pool storici coprono parti diverse del record: C149 i frame 0–1480, seen i frame 1600–2000.

**Decisione.**
- v_ref,c = p99,9 di \|u_planare\| = √(vi² + vj²) su **tutti i nodi dei frame 0:2001:10** (201 frame) della condizione, letti dal raw, **solo velocità**.
- Stessa regola e stessa dimensione del campione per TRAIN e VAL. Per TEST solo all'apertura (R5).
- IC95 con bootstrap a blocchi temporali (blocchi di 10 frame consecutivi del campione), riportato.
- Motivo: lo sperimentatore sceglie Δt per l'intera prova; usare tutto il record media la non stazionarietà invece di dipendere dal pool di training.
- Le due condizioni instabili non ricevono un trattamento speciale: la loro incertezza si dichiara ed è coperta dall'intervallo di σ_px.
- σ_u,c = σ_px · v_ref,c / d_max, con σ_px = **0,10 px** (principale), 0,06 e 0,15 (SENS-σlo, SENS-σhi) e d_max = 6 px, come prima.
- Ordine di grandezza atteso, dalla Fase 1: σ_u medio **≈0,6 m/s** (0,45–0,87 in TRAIN), non 0,8 m/s.

## 7. Posizione del campo B

**Evidenza.** Nel GTMC il campo a 5 kHz era "limited to a 30×30 mm² region **above the nozzle**" (STO12 §2.3.3, verificato). Il campo ridotto si mette dove stanno il getto d'ingresso, gli strati di taglio e la radice della fiamma. La regola geometrica precedente (y_mid ± 30 mm) era stata fissata quando l'orientamento era ignoto. Con la lettura della §2–3 sposta la finestra lungo la direzione laterale, e le due posizioni laterali sono immagini speculari.

**Decisione** (passi della §3; finestra di 52 righe × 40 colonne):

| Posizione | Righe | Colonne | Uso | Regola |
|---|---|---|---|---|
| **B-in** | 38–89 | 88–127 | **Training e valutazione** | Adiacente al bordo d'ingresso (colonne alte), centrata sull'asse (riga 63,5): equivalente di "above the nozzle" |
| B-down | 38–89 | 48–87 | Solo valutazione | Stessa linea d'asse, 30 mm a valle |
| B-off | 64–115 | 88–127 | Solo valutazione | Adiacente all'ingresso, decentrata: l'asse al bordo della finestra, come i campi decentrati per l'accesso ottico (Toronto/DLR) |

- La regola usa **geometria e velocità**, non T e non le reti: l'asse dalla simmetria di vj, il lato d'ingresso dal getto ad alta velocità.
- La frazione di gas freddo citata da Sonnet resta una diagnostica dichiarata.
- **Limite:** non sappiamo se l'ugello sia dentro o fuori dalla slice (y parte da 62,9 mm). "Adiacente al bordo d'ingresso" è l'approssimazione più vicina a "above the nozzle" consentita dai dati.
- La finestra tocca il bordo del dominio: lì valgono la regola della finestra e il kernel troncato, come in A.

## 8. Definizioni diagnostiche fissate

| Quantità | Definizione |
|---|---|
| \|∇u\| | √[(∂vi/∂r)² + (∂vi/∂c)² + (∂vj/∂r)² + (∂vj/∂c)²] in (m/s)/mm, differenze centrate con i passi della §3 (r = righe, c = colonne) |
| Regione HG | Nodi con \|∇u\| nel 10 % più alto della condizione (soglia sul pool della condizione). Il 5 % è solo diagnostico |
| Regione JET | Nodi con \|u_planare\| nel 10 % più alto della condizione |
| REST | Il resto |
| Regioni da T (T < 1800 K ecc.) | Solo diagnostica dichiarata, mai per decisioni sul generatore |
| Autocorrelazione dell'errore | Per asse, ai ritardi di 1–4 nodi, riportata anche in mm |
| Contenuto delle finestre B | Frazione di \|u\| > 20 m/s, velocità media, RMS di vj (velocità); frazione T < 1800 K solo come diagnostica |
| Distanza dal vettore valido più vicino | Trasformata di distanza euclidea in mm con i passi della §3. Classi: 0; (0, 2]; (2, 5]; (5, 10]; (10, 20]; > 20 mm |
| Distanza fra isoterme (1800 e 2100 K) | Distanza media simmetrica fra i bordi delle maschere {T ≥ soglia} vera e predetta, in mm, con trasformata di distanza. Definita per frame; se una maschera è vuota il frame si conta a parte |
| Interno/bordo per la risposta | "Bordo" = vettori il cui supporto R_r (4σ_r) esce dal campo |

## 9. Che cosa Sonnet può eseguire ora

**Autorizzato subito, in quest'ordine:**

1. **F2.0 — Aggiornamento dei preliminari** (nuovo, CPU, solo TRAIN e solo velocità di VAL):
   - v_ref,c sull'intero record (§6), per TRAIN e VAL, con IC95;
   - registro dei kernel con i passi della §3 e taratura all'1 %, per MHG, rumore e top-hat (§4–5);
   - contenuto delle finestre B-in, B-down e B-off (§7);
   - aggiornamento delle tabelle F1.2 di riferimento alla convenzione della §3, come indice ai file `f12_swapcells` esistenti, senza ricalcolo se i dati coincidono.
2. **F2.1 — Specifica `piv_observation_spec_v1`**, congelata con sha256, contenente le decisioni di questo documento.
3. **F2.2–F2.3 — Generatore e test CPU**, secondo la roadmap aggiornata.
4. **F2.4 e F2.5 — Verifiche opzionali su Neal e Cardiff**, download già autorizzati (roadmap §10).
5. **F2.6 — Qualifica delle osservazioni e rapporto per R2.** Include: SENS-SWAP, SENS-TH tarato, risposta interna e di bordo, le tre posizioni B.

**Non ancora:** F2.7 (profilazione GPU) e Fase 3, solo dopo R2. Nessun training.

## 10. Incertezze residue dichiarate

- **Orientamento:** *probabile*. Restano non verificati il segno delle componenti, l'origine delle coordinate, quale fra vi e vk sia nel piano e l'assegnazione righe ↔ `x`. Sensibilità: SENS-SWAP. Chiusura possibile: risposta degli autori.
- **Contenuto della LES a scala di finestra:** il campo esportato è liscio a 2–3 mm. L'effetto del filtro PIV sulla LES è un limite inferiore rispetto a una fiamma reale.
- **Ampiezza dell'errore:** trasferita dal progetto GTMC (σ_px, d_max); "±0,1 px" non è trattato come σ.
- **Rigetti e sostituzioni:** non quantificati in fiamma; solo sensibilità (roadmap §2.3, punto 7).
- **Ugello:** posizione rispetto alla slice ignota; B-in è l'approssimazione.
