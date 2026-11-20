# Sintesi scientifica Astra–Opus e proposta decisionale per O6, S7 e chiusura sparse

**Data:** 2026-11-20 (America/New_York)  
**Ruolo:** Revisione scientifica e sintesi decisionale (proposta per Enrico Bontempi e ChatGPT/Sol)  
**Repository:** `EnricoBontempi/official_thesis` — MIT / Deng Lab  
**Stato del documento:** **PROPOSTA DOCUMENTALE — non ancora approvata operativamente.** Non costituisce una roadmap esecutiva né autorizza training, inferenze, submission Slurm, aperture di `TEST` o merge fra branch.  
**Perimetro:** Integrazione di S5/S6, Astra primaria (A0–A3), Astra `median9` second pass e Opus (O0–O5) per proporre la decisione O6, definire S7, esplicitare gli impegni necessari a chiudere `roadmapsparse.md` e separare gli aperti residui del Mese 2 (Steps 2.7–2.9). **HONF è escluso.**

---

## 0. Provenienza e versioni esatte delle fonti consultate

I documenti rilevanti risiedono su branch differenti e sono stati ispezionati alle versioni esatte seguenti **senza eseguire alcun merge**:

| Artefatto / Documento | Branch | Commit / SHA di riferimento | Stato operativo |
|---|---|---|---|
| Report 06 — Review scientifica Opus | `review/opus-06-report` | `4f0e2637758de0766d0bb0891fe8ff97eee978c9` | Review favorevole su O0–O5; O6 aperto |
| Consegna tecnica Opus O5 (`o5/INDEX_FOR_NEON.md`, tabelle, figure, ledger) | `opus/o5-eval-attempt2` | `5373aa3c7633585edcee3d879b3c92bbb4748c11` | Base del presente worktree; eval `91828680` / `29b94345`, post-proc `258106bc` |
| Report 07 — Astra primaria (revisione coordinatore) | `review/astra-07-report` | `82bff35726fb70af9987081802ece3ef718e42bd` | Conclusa nel perimetro S5 (A0–A3); copia identica anche in `04e3b68` |
| Consegna tecnica Astra primaria (A0–A3) | `mission/astra-level` | `86a925af61accdef0b235ac084827cf639c1040c` | Inferenza `eb264bdd637f216f51267c4990f583d0ed165a32` |
| Report 08 — Review Astra `median9` second pass | `review/astra-08-median9` | `7cb7bfdc6430a2731c924a31d90897a3aa52a89f` | Conclusa nel perimetro S5 (diagnostica esplorativa) |
| Consegna tecnica Astra `median9` full set | `astra/median9-second-pass` | `04e3b68addc891034f0840c2cbaa141d5ef8c4ac` | Inferenza `44e5e5e0bd0907560d1736a53d357933680b090a` |
| S5/S6 Sparse & STOP E (`05_sparse_reconstruction.md`) | `mission/dense-unet` | S5 `38ac7fe40178982ac1bd33c5d5602a5e7d928d3b`; STOP E `a6b2bad7445170e6d821a14de9f7af9a1905ab2c` | Chiuso allo STOP E; S7 sospeso |
| Analisi tripla revisori STOP E (`analisi tripla revisore s7.md`) | `mission/dense-unet` | `6e2d176fa74d097aa8128c472103cf76c5a8a7a6` | Proposte Opus/Astra/Neon conservate |
| Diagnostico U-Net denso U0–U7 (`03_dense_velocity_diagnostic.md`) | `mission/dense-unet` | `87d1e3f40826a9343c3c2ce19d140d481b911619` | Concluso nel perimetro dichiarato (seed 0) |
| Consolidamento baseline Step 2.3 | `mission/baseline-consolidation` | `3cf7bf0` | Eseguito; integrazione in `main` pendente |
| Roadmap & diario (`roadmapsparse.md`, `roadmapopus.md`, `roadmapastra.md`, `roadmap_month2.md`, `docs/proposal.md`, `docs/RESEARCH_LOG.md`) | `mission/dense-unet` / `opus/o5-eval-attempt2` | `402bb87` / `5373aa3` (decisione diario `925f5568`) | Letti integralmente senza modifiche |

---

## 1. Sintesi critica integrata: cosa sappiamo e cosa NON sappiamo

Tutti i risultati discussi riguardano la ricostruzione **single-frame** del campo istantaneo di temperatura $T$ ($128 \times 128$, in K) da osservazioni puntuali ideali di **due componenti di velocità co-locate** ($v_i, v_j$) al **5%** (819 siti) in training, senza storia temporale, con eventuale condizionamento sui parametri operativi $p = (\mathrm{NH_3}, \phi)$. Le temperature osservate ($K = 9$) entrano **solo** nella correzione a valle (mai nel forward neurale). `TEST` resta completamente chiuso.

### 1.1 Quadro numerico consolidato (5% velocità, media macro per condizione e 3 maschere eval)

| Filone / Esperimento | Modello / Regime | Seed | VAL `best` RMSE (K) | VAL `best` C-RMSE (K) | VAL `last` RMSE (K) | `seen` `best` / `best_seen` / `last` (K) | Note strutturali e livello (`40NH3_0.9`) |
|---|---|---:|---:|---:|---:|---:|---|
| **Baseline congelate** (`5%`, `K=0`) | `GlobalMean` (`p_off`) / `ParamMean` (`p_on`) | — | `139.70` / `88.01` | `—` | `—` | `—` | Solo livello medio |
| **Baseline congelate** (`5%`, `K=0`) | `GappyPOD R=32` `p_on` / `p_off` | — | `70.60` / `94.80` | `—` | `—` | `—` | `1%/20%`: `71.37/70.46` (`on`), `98.44/93.81` (`off`) |
| **Denso diagnostico** (`100%`, separato) | `U-Net` densa `p_on` / `p_off` | 0 | `26.93` / `39.67` | `24.44` / `28.94` (`last`) | `28.06` / `44.10` | — | Più informazione, fuori classifica sparse |
| **S5/S6 storico** (`A38`, 760 frame, `DataLoader`) | `U-Net` `p_on` / `p_off` | 0 | `42.96` / `43.53` | `38.36` / `38.00` | `49.50` / `47.67` | `last 56.2` / `61.6` | `U-Net on` su `40NH3_0.9`: bias `−29.35 K`, IoU `0.310` |
| **S5/S6 storico** (`A38`, 760 frame, `DataLoader`) | `GLU-det` `p_on` / `p_off` | 0 | `45.80` / `47.54` | `44.61` / `42.72` | `46.84` / `47.87` | `last 55.0` / `59.9` | `GLU on` su `40NH3_0.9`: bias `−3.25 K`, IoU `0.925` |
| **Astra `median9`** (sui 4 `best` S5, `K=9` ideali) | `U-Net` `p_on` / `p_off` + `median9` | 0 | `38.58` / `38.49` | `38.36` / `38.00` (invariata) | — | Migliora `7/8` modelli | Entro `+0.21..+0.49 K` dall'`oracle`; IoU VAL `0.972`; Jet RMSE ancora `127–146 K` |
| **Astra `median9`** (sui 4 `best` S5, `K=9` ideali) | `GLU-det` `p_on` / `p_off` + `median9` | 0 | `44.88` / `43.16` | `44.61` / `42.72` (invariata) | — | `GLU on +0.05 K` vs free | Reti `1.8×–2.2×` meglio di `GappyPOD+median9` (`68.6–72.8 K`) |
| **Opus runner** (`B19`, 380 frame) | `U-Net` `p_on` / `p_off` | 0 | `43.93` / `46.74` | — | — | `best` `p_on−off = −8.29 K` | Normalizzatore 760 congelato; 1 solo seed |
| **Opus runner** (`A38`, 760 frame) | `U-Net` `p_on` / `p_off` | 0 | `42.39` / `42.92` | `38.16` / `—` | `45.50` / `49.84` | `51.33` (`best_seen` on), `57.99` (off) | `A38-on-s0` `40NH3_0.9` bias `−24.92 K`, IoU `0.499` |
| **Opus runner** (`A38`, 760 frame) | `U-Net` `p_on` / `p_off` | 1 | `48.22` / `43.06` | `37.43` / `—` | `51.33` / `49.53` | `s1−s0` on: `+0.73 K` | `A38-on-s1` `40NH3_0.9` bias `−48.51 K`, IoU `0.075` (`+5.83 K` VAL) |
| **Opus runner** (`C149`, 2980 frame) | `U-Net` `p_on` / `p_off` | 0 | **`38.71`** / **`39.82`** | `−2.84` a `−7.36 K` vs `A38` | `39.90` / `41.92` | `best_seen`: `39.79` / `42.68`; `last`: `37.91` / `40.03` | Jet RMSE VAL `111.9–114.8 K`; **ma `C149-on` su `40NH3_0.9` conserva bias `−21.6 / −25.9 K` (IoU `0.787 / 0.474 / 0.562`)!** |

> **Nota di comparabilità:** `S5 storico` (`DataLoader`) e `Opus` (`ScientificPresentationStream`, 95 update/epoca scientifica) generano sequenze di mini-batch differenti anche a seed nominale `0`. Il confronto sui dati è quindi valido **all'interno del runner Opus** (`B19 / A38 / C149` seed 0), mantenendo S5 storico come riferimento separato.

---

### 1.2 La scoperta centrale integrata: complementarità strutturale fra Opus e Astra

Mettere fianco a fianco i risultati di Opus e di Astra rivela un meccanismo netto e immediatamente operativo:

- **Opus (`C149`) attacca il collo di bottiglia strutturale.** Passare da 760 a 2980 istantanee supervisionate riduce in modo marcato la `centered-RMSE` (`−2.84` a `−7.36 K` su VAL, `−5.79` a `−16.55 K` su `seen`) e l'errore nei getti freddi (`Jet RMSE` VAL da `124.5–127.5 K` a `111.9–114.8 K` al `best`, e `−21.6 / −22.2 K` al `last`; `seen −29` a `−42 K`), attenuando fortemente il degrado tardivo (`last − best` su VAL cala da `+3.1..+6.9 K` a `+1.19 / +2.10 K`). **Tuttavia `C149` NON risolve l'errore di livello su `40NH3_0.9`:** su quella condizione `O-C149-on-s0` ottiene una `centered-RMSE` eccellente (`19.4–20.2 K`), ma mantiene un bias negativo di `−21.6` / `−25.9 K` che trascina la `IoU` della regione calda fino a `0.474` (`best_seen`) e `0.562` (`last`).
- **Astra (`median9`) attacca il collo di bottiglia del livello termico.** Una traslazione scalare stimata con la mediana dei residui su 9 ancore ideali abbatte il bias (`|bias| 3.11–5.09 K` su VAL S5), porta tutte e 4 le reti S5 entro `+0.21..+0.49 K` dall'`oracle-offset` e risana la regione calda su `40NH3_0.9` (`IoU` U-Net `p_on` da `0.310` a `0.949`). **Tuttavia una traslazione scalare lascia rigorosamente invariata la `centered-RMSE` per frame** e non può toccare l'errore strutturale nei getti freddi (`Jet RMSE 127–146 K` in S5).

In sintesi, **Opus e Astra agiscono su due componenti quasi ortogonali dell'errore**. La prima grande domanda rimasta aperta dalla combinazione dei due filoni è se i due guadagni **si compongano** effettivamente sugli stessi checkpoint (`O-C149` + `median9`) e se `median9` neutralizzi l'oscillazione di livello fra seed vista su `O-A38-on-s1`.

---

### 1.3 Bilancio per dimensione: noto, ignoto e limiti da non irrigidire

#### 1. Generalizzazione a nuovi frame `seen` (`seen_train_eval`, 20×11 = 220 frame)
- **Cosa sappiamo:** Nel regime `A38` (760 frame), la U-Net manifesta forte memorizzazione dei frame di fit con errore elevato su istantanee future della stessa traiettoria (`seen last 53–62 K`). In `C149` (2980 frame, stessa durata in aggiornamenti/epoca scientifica), l'errore su `seen` scende su **20/20 condizioni** (`−11.54 K` `best_seen`, `−15.12 K` `last` per `p_on`; `−15.00 / −15.31 / −19.95 K` per `p_off`), con guadagno trainato dalla struttura (`centered-RMSE` e `Jet RMSE`).
- **Cosa NON sappiamo:** `C149` migliora `seen` nella ricetta valutata, **ma non dimostra che l'overfitting sia eliminato** (resta un divario netto tra `ref760` e `seen`, e i frame `1600–2000` sono una porzione della stessa traiettoria LES, non repliche indipendenti). Inoltre `C149` è stato eseguito con un solo seed (`seed 0`).

#### 2. Trasferimento alle tre condizioni `VAL` (`0NH3_1.05`, `20NH3_1.1`, `40NH3_0.9`)
- **Cosa sappiamo:** Al 5% le ricette neurali restano largamente superiori alle baseline congelate (`38.7–47.5 K` contro `GappyPOD p_on 70.6 K` e `ParamMean 88.0 K`). Nel runner Opus, `C149` migliora `VAL` su **3/3 condizioni** rispetto ad `A38` seed 0 (`p_on best 42.39 → 38.71 K`, `p_off 42.92 → 39.82 K`; `last −5.60 / −7.56 K`).
- **Cosa NON sappiamo:** `VAL` comprende **solo 3 condizioni**, molto eterogenee per variabilità intrinseca (`std` temporale oracolo ~119 / 37 / 29 K), ed **è stata usata per selezione checkpoint, `ReduceLROnPlateau` e arresto** in tutti i run. Non abbiamo quindi una stima non distorta del trasferimento fuori da `VAL` (`TEST` resta chiuso), né sappiamo se il guadagno `C149` su VAL regga al variare del seed.

#### 3. Livello termico e struttura
- **Cosa sappiamo:** Su `VAL` S5, la U-Net ha struttura (`centered-RMSE 38.00–38.36 K`) nettamente migliore del GLU adattato (`42.72–44.61 K`), ma può subire forti bias di livello su condizioni nuove (`40NH3_0.9`), che sotto la soglia fissa `2100 K` provocano un crollo della `IoU` pur con buona geometria relativa. Una traslazione uniforme lascia invariata la `centered-RMSE`, ma **può modificare sensibilmente gli errori regionali e la `IoU`** perché sposta i pixel attraverso le soglie fisse (`<1800`, `≥2100`, `≥2200 K`).
- **Cosa NON sappiamo:** Se `C149` con altri seed continui a generare bias di livello su `40NH3_0.9`, e quale livello di RMSE/IoU raggiunga `C149` quando il bias viene corretto da `oracle` o `median9`.

#### 4. Quantità di dati e variabilità init/shuffle (e selezione `R0`)
- **Cosa sappiamo:** Nel runner Opus a seed 0, la curva `B19 (380) → A38 (760) → C149 (2980)` è monotona favorevole su `seen` e `VAL`. Sulla variabilità di training (misurata solo su `A38` con `seed 0` vs `seed 1`), `p_off` è molto stabile su `VAL` (`+0.14 K` best, `−0.31 K` last), mentre `p_on` mostra uno scarto di **`+5.83 K`** su `VAL` (`48.22` vs `42.39 K`) interamente attribuibile al bias su `40NH3_0.9` (`−48.51` vs `−24.92 K`), con `centered-RMSE` VAL quasi invariata (`37.43` vs `38.16 K`).
- **Cosa NON sappiamo (limiti critici):**
  - **Solo `A38` ha due seed; `B19` e `C149` hanno un solo seed.**
  - **La differenza fra due seed non stabilisce come la variabilità si distribuisca in generale**, né autorizza test di significatività o soglie rigide come "2× la differenza s0–s1".
  - **`R0` (rianalisi `causal_ma5`, `seen_select`, `LOCO-VAL` su S5) è una misura di sensibilità della selezione**, non una cross-validation indipendente né una stima non distorta dell'ottimismo (le traiettorie erano già condizionate da VAL via scheduler/early-stop e non esistono pesi salvati alle epoche intermedie).
  - I frame `C149` (passo 10) sono temporalmente più correlati di `A38` (passo 40): più esempi supervisionati aiutano la ricetta, ma non equivalgono a $3.9\times$ campioni indipendenti. Inoltre il normalizzatore 760 era congelato (non rifittato su `B19` né su `C149`).

#### 5. Utilità dei parametri operativi $p = (\mathrm{NH_3}, \phi)$
- **Cosa sappiamo:** Su `seen`, l'utilità di `p_on` decresce fortemente all'aumentare dei dati (`−8.29 K` in `B19`, `−6.14 / −9.46 K` in `A38`, fino a `−1.97 / −2.12 K` in `C149 best_seen/last`). Su `VAL` al 5%, il vantaggio di `p_on` è piccolo e **instabile fra seed** (`−2.81, −0.53, +5.16, −1.11 K` al `best` nei quattro confronti Opus), poiché `p_on` può estrapolare male il livello termico su una condizione esclusa. Per contro, `p_on` protegge fortemente la stabilità fuori densità (`20%` su VAL).
- **Cosa NON sappiamo:** Se su `C149` multi-seed `p_on` resti mediamente superiore o più volatile di `p_off` al 5%, e se nel regime `assisted` (`median9`) il vantaggio/rischio di `p` sul livello scompaia quasi del tutto lasciando a `p_on` solo il beneficio fuori densità o un modesto guadagno strutturale.

#### 6. Copertura delle velocità (`1% / 5% / 20%`)
- **Cosa sappiamo:** **Tutte** le valutazioni a `1%` e `20%` provengono da reti addestrate esclusivamente al `5%`. Al `1%`, gli errori salgono a `67–79 K` su VAL. Al `20%`, nessuna rete allenata al `5%` migliora rispetto al `5%`; tutti i modelli `p_off` degradano sensibilmente (`+2.8` a `+10.2 K`, incluso `O-C149-off`: `39.82 → 48.87 K`), mentre i `p_on` degradano poco (`O-C149-on`: `38.71 → 39.21 K`).
- **Cosa NON sappiamo:** Quanto questo comportamento al `1%` e `20%` dipenda esclusivamente dal domain shift di densità (risolvibile con training multi-densità) rispetto a un limite della rappresentazione input (`zero + M_v` nella U-Net o banda fissa `sigma=0.05` / `top_k=16` nel GLU).

#### 7. Ricostruzione `free` vs `assisted` (`AO9`, `median9`, `oracle`, baseline)
- **Cosa sappiamo:** Nel perimetro S5 (`K=9` ideali su griglia `3×3`, rumore 0), `AO9` (media con $\tau = 1$) peggiora 3/4 reti su VAL e `seen` perché due ancore al bordo dei getti freddi (`(32,95)`, `(95,95)`) concentrano residui di `72–131 K`, sovra-rappresentando il 6.8% dell'area VAL con il 22.2% delle ancore. La mediana (`median9`) elimina questo trascinamento da outlier e batte sia `AO9` sia `free` su **8/8 modelli** su VAL (e **7/8** su `seen`), avvicinando tutte le reti S5 all'`oracle` (`+0.21..+0.49 K`). Rimuovere solo lo shrink (`mean9`) peggiora 8/8 modelli: il merito è della robustezza dello stimatore. Le reti restano `1.8×–2.2×` meglio di `GappyPOD+median9` (`68.6–72.8 K`) e `~8×` meglio di `T_only_const` (`311.8 K`).
- **Cosa NON sappiamo (limiti rigorosi):**
  - **`median9` è una diagnostica esplorativa motivata da `VAL` già osservata** (non preregistrata prima di vedere il fallimento di `AO9`).
  - **Le nove $T$ sono osservazioni numeriche ideali** (rumore 0, puntuali, sincronizzate, non intrusive): **non costituiscono una validazione strumentale di termocoppie reali**.
  - Su condizioni `seen` ad alto NH3 (`80NH3_1.15/1.2`), 4/9 ancore entrano in regioni ad alto errore, avvicinando il punto di rottura della mediana.
  - Mancano ancora: (a) la verifica sui checkpoint `Opus` (`C149`, `A38 s0/s1`), (b) la curva su $K \in \{0, 1, 4, 9, 16, 32\}$ richiesta dallo Step 2.7, e (c) la sensibilità a layout non a griglia e a rumore realistico.

---

## 2. Decisione proposta per O6 (chiusura della sotto-missione Opus)

**Raccomandazione:** **Chiudere O6 positivamente** nel perimetro di `roadmapopus.md`, senza aprire nuovi sweep dentro Opus, con le seguenti quattro determinazioni formali:

1. **Esito tecnico O0–O5 accettato:** Il pacchetto `5373aa3` e la review `4f0e263` confermano che l'esperimento Opus è tecnicamente pulito, tracciabile e scientificamente informativo; non sono richieste ri-esecuzioni di O0–O5.
2. **Adozione del regime `C149` + `ScientificPresentationStream` per i prossimi training `SparseUNet`:** A parità di budget per epoca scientifica (760 presentazioni, 95 update) e normalizzatore fit-only 760 congelato, `C149` (2980 frame) offre un miglioramento coerente su `seen` (20/20), `VAL` (3/3), `centered-RMSE`, `Jet RMSE` e stabilità tardiva. Proseguire S7 sul vecchio regime `A38` (760 frame) sprecherebbe 4 training su un regime già dimostrato strutturalmente inferiore.
3. **Separazione storica rigorosa:** I risultati `S5` storici (`A38`, `DataLoader`) e i risultati `Opus/S7` (`ScientificPresentationStream`) restano in tabelle chiaramente etichettate e mai mescolate come se avessero identica pipeline di campionamento.
4. **Trasferimento dell'incertezza residua in S7 (non in un O6 infinito):** L'unico vero nodo lasciato aperto da Opus — il fatto che `C149` abbia un solo seed (`seed 0`), mentre `A38 p_on` ha oscillato di `+5.83 K` fra `s0` e `s1` — viene risolto direttamente nella proposta **S7** (§4), anziché duplicare campagne fra `roadmapopus.md` e `roadmapsparse.md`. I limiti di Opus restano registrati esplicitamente (1 seed su `B19/C149`, normalizzatore 760 congelato, correlazione temporale passo 10, `C149` non elimina l'overfitting né risolve da solo il bias su `40NH3_0.9`, nessuna evidenza nuova su `GLU`).

---

## 3. Riesame esplicito delle proposte Opus / Astra / Neon (`analisi tripla revisore s7.md`)

Alla luce di S5/S6, Astra A0–A3, `median9` e Opus O0–O5, riesaminiamo ciascuna direzione documentata in `analisi tripla revisore s7.md` (`6e2d176`).

| Direzione / Proposta (`analisi tripla revisore s7.md`) | Classificazione attuale | Giustificazione ed evidenza |
|---|---|---|
| **Curva dati `B19/A38/C149` + `A38 seed 1` + rianalisi `R0`** (Opus §4.3–4.4) | **Già affrontata** | Eseguita e revisionata in Opus O0–O5 (`5373aa3`, report `06`). Mostra guadagno strutturale consistente con `C149` a seed 0, variabilità livello su `A38 p_on` (`+5.83 K`), e sensibilità selezione `R0` su S5. Resta aperta la replica multi-seed di `C149`. |
| **Decomposizione livello/struttura `free/oracle/AO9` + `median9` su S5** (Astra §5.3–5.5) | **Già affrontata (perimetro S5)** | Eseguita e revisionata in Astra A0–A3 (`86a925a`, report `07`) e second pass `median9` (`04e3b68`, report `08`). Dimostra che il fallimento `40NH3_0.9` di U-Net è bias di livello, che `AO9` fallisce per outlier nei getti freddi e che `median9` recupera quasi tutto il gap fino all'`oracle`. |
| **Preparazione tecnica comune §8** (YAML, SHA distinti, hash payload, gate Gappy, pannelli tutte le VAL, tabelle regionali, curve) | **In larga parte affrontata; 1 coda minore da chiudere** | Correzioni strutturali implementate e verificate nei runner Opus e Astra/median9. Resta aperto il solo **`GappyPOD` snapshot-swap @5% rep 0** (1–2 min CPU, incluso in S7-A) oltre all'integrazione finale dei branch. |
| **Eventuale A4 sui checkpoint Opus (`C149` + `A38 s0/s1`)** (Astra §5.5 / `roadmapastra.md`) | **Ancora necessaria adesso** | **Zero training (~30–35 min GPU).** Risponde subito a due domande che cambiano il resto del Mese 2: (1) `C149` (struttura) e `median9` (livello) si sommano davvero? (2) `median9` abbatte lo scarto di `+5.83 K` fra `O-A38-on-s0` e `s1`? |
| **Conferma `C149` su altri seed (`seed 1, 2`, `U-Net p_on/p_off`)** (Opus §4.5 / `roadmapsparse.md` S7) | **Ancora necessaria adesso** | **Cuore di S7.** Oggi `C149` ha un solo seed (`s0`), mentre `A38 p_on` ha oscillato di `+5.83 K` fra due seed. Senza repliche `s1, s2` su `C149` non possiamo sostenere né che il miglioramento `C149` su VAL sia riproducibile né stabilire il comportamento reale di `p_on` vs `p_off` nel regime ricco. |
| **Confronto `U-Net` vs `GLU` a dati e ricetta comparabili (`C149`)** (Opus §4.6, Neon §6.4 R2) | **Utile ma rinviabile** *(o Alternativa S7-Alt1 solo se prioritaria per scelta architetturale)* | **Attenzione metodologica critica:** non si deve mai confrontare `U-Net C149` (`38.71 K`) con `GLU A38` (`45.80 K`) attribuendo il divario all'architettura! Il solo confronto matched resta `A38 seed 0`, dove U-Net ha già `centered-RMSE` migliore di `4.4–6.2 K` a ~`2.7×` minor costo di training. Replicare GLU su `C149` conviene solo se il laboratorio vuole ancora investire in GLU nel Mese 2; altrimenti va dichiarato onestamente che il confronto GLU è chiuso allo screening `A38 seed 0`. Il rilievo Neon su `d_model=32 / FFN=2048` riguarda un cambio architetturale (Mese 3), non un controllo della ricetta S5. |
| **Training multidensità `1/5/20%`** (Neon §6.4 R3/R4, Opus §4.6) | **Utile ma rinviabile** *(a Mese 3 o Alternativa S7-Alt2)* | Cambia simultaneamente esposizione TRAIN e criterio di selezione (media `1/5/20%` vs `5%`), creando un nuovo protocollo prima che `C149` sia confermato multi-seed. Inoltre per `GLU` al `20%` (3277 token, self-attention $O(N^2)$) serve profiling memoria dedicato. Utile quando la variabilità di copertura PIV diventerà priorità primaria (Mese 3). |
| **Conferma `median9`, sweep $K \in \{0,1,4,9,16,32\}$ e sensibilità layout/rumore** (Neon §6.3, Step 2.7) | **Sdoppiata:** sweep $K$ **necessaria subito dopo S7 (Step 2.7)**; layout/rumore **utile ma rinviabile (Mese 3–4)** | Lo sweep $K$ annidato (`0/1/4/9/16/32`) con `AO_K`, `median_K` congelato e `GappyPOD+T-block` è richiesto da Step 2.7 ed è a sola valutazione offline sui checkpoint congelati S7 (costo minimo). Invece sensibilità a layout realistici (rake/non-grid), rumore di misura, risposta dinamica e intrusività delle termocoppie sono esplicitamente lavoro dei Mesi 3–4 (`roadmap_month2.md`); fino ad allora va sempre ribadito che $T_9$ sono ideali. |
| **Pseudo-estrapolazione (fold interno TRAIN) / altri controlli trasferimento** (Neon §6.5, Opus §4.6, Step 2.8) | **Utile ma rinviabile** (a Step 2.8 / Mese 3) | Richiede ri-fit completo di normalizzatore, POD e `ParamMean` escludendo la condizione trattenuta. **Cautela obbligatoria:** un nuovo split esplorativo costruito dopo aver esaminato S5/Opus/Astra **non è un `TEST` indipendente**, perché scelte di dati, architettura e stimatore sono già state guidate dai risultati complessivi. Serve come stress-test diagnostico in Step 2.8 (se deciso) o Mese 3, non per sostituire S7. |
| **Stimatore lineare supervisionato (ridge su POD a rango alto)** (Opus §4.6) | **Non giustificata adesso** | Ha senso solo se la tesi vuole sostenere formalmente che il guadagno neurale dipenda dalla *non linearità* rispetto a qualsiasi stimatore lineare supervisionato. Non cambia nessuna scelta ingegneristica attuale; conservare la cautela nel testo ("vantaggio della ricetta, non prova universale della non linearità") costa zero. |
| **Testa residua appresa `anchor_residual.py` / modifiche architetturali (Voronoi, conv normalizzate, banda adattiva, importance appresa, Beta/incertezza)** (Neon §6.5, Step 2.7/2.9) | **Non giustificata adesso** | In `roadmap_month2.md` la testa residua è esplicitamente dietro un **gate** (solo dopo aver visto lo sweep $K$ di Step 2.7). Le modifiche architetturali e l'incertezza appartengono al Mese 3, dopo la scelta di una singola direzione in Step 2.9. |

---

## 4. Proposta S7: UNA strada raccomandata (e due alternative solo per obiettivi diversi)

### 4.1 Razionale della strada raccomandata

La strada raccomandata unisce massimo potere chiarificatore, disciplina sperimentale e costo trasparente, articolandosi in **due blocchi sequenziali senza ridondanze**:

1. **Blocco S7-A (solo valutazione offline, 0 training, ~35 min GPU + ~2 min CPU):** estensione **A4** sui 6 run Opus esistenti (`C149 s0`, `A38 s0/s1`, `p_on/p_off`) + completamento `GappyPOD` snapshot-swap mancante da S6.
2. **Blocco S7-B (4 nuovi training GPU `SparseUNet C149` + eval completa):** repliche `seed 1` e `seed 2` di `U-Net C149 p_on` e `U-Net C149 p_off`, riutilizzando `O-C149-on/off-s0` già esistenti come `seed 0`, così da ottenere **3 seed completi (`0, 1, 2`)** sul contrasto prioritario nel regime ricco `C149`, più valutazione O5 (`1/5/20%`), swap e Astra/median9 sui nuovi checkpoint.

> **Perché 3 seed (`0, 1, 2`) — e cosa può davvero sostenere questa scelta:**  
> Non scegliamo 3 seed fingendo una precisione statistica o intervalli di confidenza affidabili che con $n = 3$ non esistono (e mantenendo `train_mask_seed = 20261118` fisso, non misuriamo nemmeno la variabilità della sequenza di maschere TRAIN). Con 3 seed completi per entrambi i rami `p_on` e `p_off` otteniamo però **un controllo descrittivo onesto e diretto** allineato a `roadmapsparse.md` §6/§8: (a) verifichiamo se il salto strutturale di `C149` su `seen` e `VAL` si ripete oltre il singolo `seed 0`, (b) verifichiamo se l'instabilità di livello su `40NH3_0.9` in `p_on` (osservata su `A38 s1`) ricompare o si attenua con 2980 frame, e (c) verifichiamo se `median9` stabilizza tutti e 3 i seed. Se emergesse un outlier estremo fra i 3 seed, lo documentiamo apertamente anziché nasconderlo in una media.

---

### 4.2 Specifica dettagliata — Blocco S7-A (Ponte diagnostico A4 + completamento S6)

- **Domanda e conclusione che potrebbe sostenere:**
  1. I guadagni di **Opus (`C149`, struttura)** e **Astra (`median9`, livello)** si compongono sullo stesso checkpoint, portando `U-Net C149 + median9` chiaramente sotto il miglior S5+median9 (`38.5 K` VAL) e risanando la `IoU` su `40NH3_0.9` anche al `last` / `best_seen`?
  2. La forte oscillazione fra seed di `O-A38-on` (`+5.83 K` VAL tra `s0` e `s1`) viene quasi annullata da `median9`, confermando che è variabilità di livello e non di struttura?
  3. Nel completamento S6, `GappyPOD` al 5% reagisce allo snapshot-swap in modo coerente rispetto alle reti neurali?
- **Perché i risultati attuali non bastano:** Astra e `median9` sono stati eseguiti **solo sui 4 `best` S5 (`A38 seed 0`)**. Non esiste alcuna prova diretta su `O-C149` né su `O-A38 s0/s1`, e lo swap `GappyPOD` richiesto da `roadmapsparse.md` §6/§7 non è mai stato consegnato.
- **Matrice concreta:**
  - *Parte A4 (GPU):* 6 run Opus esistenti (`O-A38-on-s0`, `O-A38-off-s0`, `O-A38-on-s1`, `O-A38-off-s1`, `O-C149-on-s0`, `O-C149-off-s0`) × 3 checkpoint (`best`, `best_seen`, `last` — oppure `best` e `last` come primari + `best_seen` diagnostico) × copertura `5%` × 3 repliche maschera (`0, 1, 2`) × 2 ruoli (`seen_train_eval` 220, `val_eval` 123) × 5 trattamenti (`free`, `oracle`, `AO9 τ=1`, `mean9`, `median9`) sulle 9 ancore griglia `3×3` `[32,64,95]×[32,64,95]`, rumore 0.
  - *Parte swap GappyPOD (CPU):* `GappyPOD R=32 p_on` e `p_off` congelati (`lambda` storici), copertura `5%`, replica `0`, `seen_train_eval` e `val_eval`, pairing U5 identico a S6.
- **Risultati esistenti riutilizzabili e nuove esecuzioni necessarie:**
  - *Riutilizzabili:* tutti i 6 checkpoint Opus (`5373aa3`), le 9 ancore persistite e lo script di inferenza/aggregazione Astra `median9` (`44e5e5e0` / `04e3b68`), il `GappyPOD` fit-only consolidato e il pairing swap S6 (`a6b2bad`).
  - *Nuove esecuzioni:* 1 job GPU (inferenza + aggregazione A4) + 1 job CPU (`GappyPOD` swap).
- **Training oppure sola valutazione:** **Sola valutazione offline** (0 training).
- **Prerequisiti e dipendenze:** Accesso ai checkpoint Opus su Engaging pool + ambiente/script Astra `median9` con controllo rigoroso del replay `K=0` ($\Delta = 0.000$ K rispetto ai CSV O5 `5373aa3`).
- **Metriche e regole di selezione da congelare:** Nessuna nuova selezione di pesi. Replay `K=0` esatto contro O5; metriche macro frame→condizione→maschera e per-condizione (RMSE, MAE, bias, `|bias|`, `centered-RMSE`, `Jet RMSE T<1800`, `T≥2100`/`≥2200` RMSE/IoU/Dice/area, off-anchor e alle ancore separate, invarianza `centered-RMSE` per frame verificata). Etichettare `median9` come **diagnostico confermativo su pesi nuovi ma stimatore originato in modo esplorativo su VAL**.
- **Esiti possibili e decisione conseguente:**
  - *Esito A (atteso):* `C149 + median9` scende sensibilmente sotto `38.5 K` VAL (attorno al livello `centered-RMSE` di `C149`, es. `~34–36 K`), ripristina `IoU > 0.94` su `40NH3_0.9`, e `median9` riduce drasticamente lo scarto `s0–s1` di `O-A38-on` → conferma piena della complementarità Opus+Astra; si procede con S7-B e poi Step 2.7 (`K` sweep) senza riaprire ricerche architetturali.
  - *Esito B:* `median9` degrada `C149` o non riduce lo scarto `s0–s1` su `A38` (es. per spostamento dei residui sulle ancore) → analizzare i residui per ancora su `C149` prima di congelare lo stimatore per Step 2.7; S7-B resta comunque necessario per la stabilità `free` di `C149`.
- **Criterio di completamento:** Replay `K=0` verificato entro tolleranza numerica su tutti i checkpoint, invarianza `centered-RMSE` verificata, tabelle per-condizione/macro/ancore + mappe frame 1000 delle 3 VAL consegnate, `GappyPOD` swap consegnato con `RMSE(T_donor, T_dest)` di riferimento.
- **Costo stimato (da tempi misurati):**
  - *Calcolo:* `~2 min`/checkpoint su 1 L40S (misurato in Astra `median9`) × 18 valutazioni (`6 run × 3 ckpt`) = **`~35–40 min` GPU** (o `~25 min` con soli `best/last`); `GappyPOD` swap **`~1–2 min` CPU**.
  - *Coda (separata):* tipicamente pochi minuti – `~30 min` per 1 singola GPU L40S su `mit_normal_gpu` (`walltime` richiesto `1h–1h30m`) e pochi minuti su `mit_normal` CPU.

---

### 4.3 Specifica dettagliata — Blocco S7-B (Conferma multi-seed `SparseUNet C149 p_on/p_off`)

- **Domanda e conclusione che potrebbe sostenere:**
  1. Il miglioramento di `SparseUNet` nel regime `C149` (2980 frame) su `seen` e `VAL` è riproducibile su 3 inizializzazioni indipendenti (`seed 0, 1, 2`), oppure `O-C149-s0` era un'estrazione favorevole?
  2. Nel regime `C149`, qual è la differenza reale e la dispersione fra `p_on` e `p_off` su `VAL` al `5%`, fuori densità (`1%`, `20%`), su `seen`, sulle regioni calde (`40NH3_0.9`) e con `median9`?
  3. Quale coppia (`C149 p_on` vs `p_off`, e con quali cautele sul livello) congelare come finalista `SparseUNet` per chiudere `roadmapsparse.md` e passare a Step 2.7?
- **Perché i risultati attuali non bastano:** `C149` oggi ha **1 solo seed (`seed 0`)**, quindi qualsiasi conclusione su `C149` o sul confronto `p_on vs p_off` in `C149` è vulnerabile a una singola inizializzazione (come dimostrato dallo scarto `+5.83 K` di `O-A38-on-s1`). Inoltre `roadmapsparse.md` §6/§8 richiede esplicitamente `seed 1` e `seed 2` per il contrasto prioritario prima di chiudere S7 (STOP F).
- **Matrice concreta:**

| Run ID proposto | Modello | Pool dati | `p` | `model_seed` | `shuffle_seed` | `train_mask_seed` | Copertura TRAIN | Eval post-training | Stato |
|---|---|---|---:|---:|---:|---:|---|---|---|
| `O-C149-on-s0` | SparseUNet b32 | C149 (2980) | on | 0 | 0 | `20261118` | 5% | O5 già fatto; A4 in S7-A | **Già esistente (riuso)** |
| `O-C149-off-s0` | SparseUNet b32 | C149 (2980) | off | 0 | 0 | `20261118` | 5% | O5 già fatto; A4 in S7-A | **Già esistente (riuso)** |
| `S7-C149-on-s1` | SparseUNet b32 | C149 (2980) | on | 1 | 1 | `20261118` | 5% | O5 (`1/5/20%`, 3 rep) + swap + Astra/median9 | **Nuovo training** |
| `S7-C149-off-s1` | SparseUNet b32 | C149 (2980) | off | 1 | 1 | `20261118` | 5% | O5 (`1/5/20%`, 3 rep) + swap + Astra/median9 | **Nuovo training** |
| `S7-C149-on-s2` | SparseUNet b32 | C149 (2980) | on | 2 | 2 | `20261118` | 5% | O5 (`1/5/20%`, 3 rep) + swap + Astra/median9 | **Nuovo training** |
| `S7-C149-off-s2` | SparseUNet b32 | C149 (2980) | off | 2 | 2 | `20261118` | 5% | O5 (`1/5/20%`, 3 rep) + swap + Astra/median9 | **Nuovo training** |

- **Risultati esistenti riutilizzabili e nuove esecuzioni necessarie:**
  - *Riutilizzabili:* cache estesa `C149`, normalizzatore `train_norm_stats_month2_fit.json` (760 congelato, identico a `O-C149-s0`), manifest maschere annidate `1/5/20%`, runner Opus (`ScientificPresentationStream`), evaluator O5, script Astra `median9`, e i 2 run completi `O-C149-on/off-s0` (`5373aa3`).
  - *Nuove esecuzioni:* **4 training GPU** (`s1, s2 × on/off`) + **4 job di valutazione GPU** (O5 + swap + Astra/median9 sui 4 nuovi run) + aggregazione multi-seed `0/1/2`.
- **Training oppure sola valutazione:** **4 training completi + valutazione completa.**
- **Prerequisiti e dipendenze:** Approvazione esplicita di Enrico/Sol su O6 e S7; consegna S7-A completata (o almeno verifica che il runner Opus e la cache `C149` siano integri); congelamento preliminare del config S7 (identico a `O-C149-s0` tranne `model_seed`/`shuffle_seed ∈ {1, 2}` e output directory dedicate). **Nessuna modifica silenziosa** di normalizzatore, optimizer, warmup, scheduler o criteri di stop rispetto a `O-C149-s0`, altrimenti `s0` non è più accoppiabile a `s1/s2`.
- **Metriche e regole di selezione da congelare:**
  - *Selezione checkpoint:* identica a `roadmapsparse.md` §6.1 e Opus — `best` = primo minimo assoluto su `VAL 5%` (media macro frame→condizione→3 repliche maschera); registrare anche `best_seen` e `last` come diagnostici.
  - *Early stop / tranche:* identici a `O-C149-s0` (tranche 1000 epoche scientifiche da 95 update, `min_epochs=200`, patience 15 eval a soglia rel `0.1%`, reset al calo LR; se ancora in miglioramento a 1000 epoche, fermarsi e discutere continuazione senza truncamento cieco).
  - *Reporting:* **tutti i singoli seed `0, 1, 2` riportati individualmente**, più media e range (min–max / deviazione descrittiva), **tenendo rigorosamente separata la dispersione fra seed di training dalla dispersione fra le 3 maschere di valutazione**. Metriche: RMSE, MAE, NRMSE, bias, `|bias|`, `centered-RMSE`, `Jet RMSE`, `T≥2100/2200` IoU/Dice/area, curve `1/5/20%`, swap @5% rep 0, e confronti `free / oracle / AO9 / median9`.
- **Esiti possibili e decisione conseguente:**
  - *Esito 1 (stabilità buona su 3 seed):* `C149` batte coerentemente `A38` su `seen` e `VAL` su tutti e 3 i seed (`s0, s1, s2`), con dispersione moderata → **`SparseUNet C149` diventa il backbone congelato** per chiudere `roadmapsparse.md` (STOP F) e passare a Step 2.7 (`K` sweep). Si mantiene `p_on` come candidato primario se regge su 5% e protegge 20%, conservando `p_off` come ramo unconditioned richiesto da Step 2.6.
  - *Esito 2 (oscillazione di livello di `p_on` anche in `C149`, ma struttura stabile):* `p_on` mostra ancora salti di bias su `40NH3_0.9` in `s1` o `s2` mentre `centered-RMSE` resta bassa e `median9` riallinea tutti i seed → dimostrazione netta che il regime `free` con `p_on` resta esposto a errori di livello su condizioni nuove, rafforzando la motivazione scientifica di Step 2.7 (`assisted`) e segnalando `p_off` (o `p_on + medianK`) come ramo più regolare sul livello.
  - *Esito 3 (`C149 s1/s2` regrediscono vicino ad `A38` su VAL pur restando migliori su `seen`):* l'effetto dati su VAL è piccolo rispetto alla variabilità di training → ridimensionare il claim sul trasferimento VAL, documentare onestamente la varianza di training, e valutare in Step 2.8/Mese 3 controlli su più condizioni escluse.
- **Criterio di completamento:** Tutti e 4 i training terminati per plateau (o tranche completata e discussa), zero errori di integrità/leakage, replay `best`/`last` verificati, tabelle individuali `seed 0/1/2` per `p_on/p_off` (con dispersione seed separata da maschere), figure comparative su tutte le 3 VAL, costi reali registrati, report S7 pronto per **STOP F**.
- **Costo stimato (da tempi misurati su L40S, calcolo vs coda separati):**
  - *Training (calcolo):* dai run `O-A38` / `O-C149` (630–710 epoche scientifiche, stesso budget 95 update/epoca, con eval periodica ogni 10 epoche: `3h 42m – 3h 56m` misurati su L40S), stimiamo **`~3.5–4.5 h`/run** → **`~14–18 GPU-h` totali** per i 4 run. Con 4 GPU in parallelo (cap QoS `mit_normal_gpu` = 4 GPU): **`~4–4.5 h` di muro di calcolo** (richiedere walltime Slurm `8h` con checkpoint di resume per sicurezza su eventuali run più lenti fino a 1000 epoche).
  - *Valutazione post-training (calcolo):* da `setting_costs.csv` O5 (`51–77 min`/run) + Astra `median9` (`~6 min`/run) + swap (`~2 min`/run) = **`~1.1–1.4 h`/run** → **`~4.5–5.6 GPU-h` totali** (`~1.2–1.5 h` di muro con 4 GPU in parallelo).
  - *Coda (separata dal calcolo):* dipende dal carico su `mit_normal_gpu` (account `mit_amf_advanced_gpu`, max 4 GPU simultanee); storicamente da pochi minuti a `~1–2 h` per onda completa da 4 GPU.
  - **Totale strada raccomandata (S7-A + S7-B):** **4 training GPU + 5 job eval GPU + 1 job eval CPU**, circa **`~19–24 GPU-h` di calcolo complessivo** (~`6–7 h` di calcolo parallelo su 4 L40S + coda).

---

### 4.4 Alternative esplicite (SOLO se corrisponde a un obiettivo scientifico diverso)

Queste due opzioni **non** fanno parte della strada raccomandata di default; vanno aggiunte solo se Enrico e il coordinatore scelgono esplicitamente l'obiettivo scientifico corrispondente.

#### Alternativa S7-Alt1 — Confronto architetturale `GLU adattato` su `C149` (2 training aggiuntivi)
- **Obiettivo diverso che la giustifica:** Voler pronunciare un verdetto comparativo `U-Net` vs `GLU adattato` anche nel regime ricco `C149`, anziché chiudere il confronto `GLU` allo screening storico `A38 seed 0`.
- **Domanda e conclusione:** Con 2980 istantanee supervisionate e lo stesso `ScientificPresentationStream`, `GLU adattato` riduce il divario strutturale (`centered-RMSE`) rispetto a `U-Net C149` o resta strutturalmente dietro a costo superiore?
- **Perché i risultati attuali non bastano:** Oggi disponiamo solo di `GLU A38 seed 0` (S5 storico). Confrontarlo con `U-Net C149` sarebbe metodologicamente scorretto (confonderebbe architettura, quantità di dati e runner).
- **Matrice concreta:** 2 nuovi training `GLU adattato` (`d_model=32`, stessa architettura S5 congelata) su `C149`, `seed 0`, `p_on` e `p_off`, 5%, `train_mask_seed=20261118` + eval O5 (`1/5/20%`) + swap + Astra/median9.
- **Riutilizzo vs nuovo:** Riutilizza wrapper GLU S3/S5, cache `C149`, normalizzatore 760, `O-C149-on/off-s0` come comparatore `U-Net` matched a seed 0. Nuovi: 2 training GPU + 2 eval GPU.
- **Training/valutazione:** 2 training + valutazione completa.
- **Prerequisiti:** Integrazione del wrapper GLU nel runner `ScientificPresentationStream` + smoke parity su 1 epoca prima del lancio (nessuna modifica architetturale, quindi *non* cambiare `d_model` o `FFN=2048` in questo braccio).
- **Metriche e selezione:** Identiche a S7-B (`VAL 5% best`, report `best`/`best_seen`/`last`, `centered-RMSE`, `Jet RMSE`, `1/5/20%`, `median9`, costo e latenza batch=1).
- **Esiti e decisione:** Se `GLU C149` resta peggiore di `U-Net C149` su `centered-RMSE` (e più costoso), si archivia definitivamente `GLU` S5 come baseline comparata anche a 2980 frame; se `GLU C149` supera nettamente `U-Net C149` sulla struttura o sul livello, diventa necessario replicarlo prima di congelare il backbone di Step 2.7.
- **Completamento:** 2 run a plateau + eval O5/Astra/swap + tabella matched `U-Net C149 s0` vs `GLU C149 s0`.
- **Costo stimato (calcolo vs coda):** In S5 `GLU` (890–980 ep) impiegò `2h 20m – 2h 35m` contro `~53 min` di U-Net (`~2.7×` a parità di runner, ma con eval periodica Opus più leggera del fattore `2.7×` sul solo training); stimiamo **`~4–6 h`/run** → **`~8–12 GPU-h` training** + **`~3–4 GPU-h` eval** (per `20%` GLU con 3277 token è più lento in inferenza) = **`~11–16 GPU-h` calcolo** (`~6–8 h` muro su 2 GPU + coda `mit_normal_gpu`).

#### Alternativa S7-Alt2 — Verifica esplorativa multi-densità `1/5/20%` su `U-Net C149` (2 training aggiuntivi)
- **Obiettivo diverso che la giustifica:** Se la tenuta a copertura variabile (`20%` di `p_off` e `1%`) è ritenuta prioritaria già nel perimetro S7 anziché nel Mese 3.
- **Domanda e conclusione:** Esporre `U-Net C149` in training a batch omogenei campionati fra `1%`, `5%` e `20%` elimina il degrado di `p_off` al `20%` e migliora il `1%` senza penalizzare il `5%`?
- **Perché i risultati attuali non bastano:** Tutti i modelli esistenti sono allenati solo al `5%`, quindi la curva `1/5/20%` attuale misura insieme contenuto informativo e shift di densità.
- **Matrice concreta:** 2 nuovi training `U-Net C149` `p_on` e `p_off`, `seed 0`, copertura campionata uniformemente per batch fra `{1%, 5%, 20%}` (batch omogenei), + eval O5 (`1/5/20%`) + Astra `median9` @5%.
- **Riutilizzo vs nuovo:** Riutilizza cache `C149`, normalizzatore 760 e `O-C149-s0` come controllo `5%-only`. Nuovi: 2 training GPU + 2 eval GPU.
- **Training/valutazione:** 2 training + valutazione completa.
- **Prerequisiti:** Nuova specifica esplicita di protocollo (`month2_multidensity_protocol`, distinta da `roadmapsparse.md` §3.3); decisione preventiva sulla regola di selezione (es. **mantenere selezione `VAL 5%`** se si vuole isolare la sola esposizione multi-densità, oppure dichiarare esplicitamente un **nuovo protocollo con selezione media `1/5/20%`** sapendo che cambiano due fattori insieme).
- **Metriche e selezione:** Stessa batteria O5, confrontando ciascuna densità `1% / 5% / 20%` contro `O-C149-on/off-s0`.
- **Esiti e decisione:** Se la curva diventa monotona (`20% < 5% < 1%`) e `p_off` smette di degradare al `20%`, il difetto era puro shift di densità e non serve modificare l'architettura U-Net; se persiste, motiva nel Mese 3 una modifica della rappresentazione input (es. convoluzioni normalizzate o canali distanza).
- **Completamento:** 2 run + confronto appaiato `1/5/20%` contro `O-C149-s0` + analisi bias/struttura per densità.
- **Costo stimato (calcolo vs coda):** **`~7–9 GPU-h` training** (`~3.5–4.5 h` muro su 2 GPU) + **`~2.5 GPU-h` eval** = **`~9.5–11.5 GPU-h` calcolo** + coda.

---

## 5. Chiusura di `roadmapsparse.md` e aperti residui del Mese 2 (Steps 2.7–2.9)

**Chiarimento fondamentale:** completare S7 (o anche un'eventuale futura roadmap Neon) **chiude `roadmapsparse.md` (STOP F)**, ma **NON chiude automaticamente il Mese 2 (`roadmap_month2.md`)**, che mantiene aperti gli Step **2.7, 2.8 e 2.9**, l'integrazione tecnica dei branch e la ratifica scientifica dei supervisori.

### 5.1 Tabella dei requisiti effettivi: Sparse e Mese 2

| Ambito / Requisito (fonte) | Evidenza disponibile | Lacuna residua | Azione necessaria o rinvio da concordare |
|---|---|---|---|
| **Sparse S0–S4** (`roadmapsparse.md` §8): protocollo, manifest `1⊂5⊂20%` (3309/3309 join), U-Net/GLU test, GappyPOD nested gate-pass (`29601906`) | Chiusi e verificati (`38ac7fe`, `a6b2bad`). | Nessuna lacuna sperimentale; solo consolidamento git dei branch. | Nessun nuovo esperimento; consolidare nel merge finale dedicato. |
| **Sparse S5** (`roadmapsparse.md` §8): 4 screening `seed 0` (`U-Net/GLU × p_on/off`) | Chiuso (`38ac7fe`, STOP D); integrato da Opus O0–O5 su `U-Net` (`5373aa3`). | `GLU` presente solo su `A38 seed 0` (non su `C149`). | **Decisione da concordare:** dichiarare il confronto `GLU` chiuso al perimetro screening `A38 seed 0` (raccomandato, permesso da `roadmapsparse.md` §8 S7 *"confronto GLU presente oppure apertamente incompleto"*) **oppure** attivare S7-Alt1. |
| **Sparse S6** (`roadmapsparse.md` §6–§8): eval `best/last` `1/5/20%`, 3 ruoli × 3 maschere, swap neurali + **swap GappyPOD** | Consegnato in STOP E (`a6b2bad`, report `05`), ma **swap `GappyPOD` mancante**. | Manca il solo controllo snapshot-swap su `GappyPOD` @5% rep 0. | **Azione necessaria (S7-A):** eseguire `GappyPOD` swap CPU (~1–2 min) e allegare il riferimento `RMSE(T_donor, T_dest)`. |
| **Sparse S7** (`roadmapsparse.md` §6, §8): repliche `seed 1, 2` del contrasto prioritario, stesso `train_mask_seed`, risultati individuali `0/1/2` con dispersione training separata da maschere, aggiornamento report & stato 2.5–2.6 | Solo `seed 0` su `C149`; solo `seed 0/1` su `A38` (Opus). | Mancano `seed 1, 2` su `C149 p_on/p_off` e report integrato di chiusura STOP F. | **Azione necessaria (S7-B):** 4 training `U-Net C149` (`s1, s2 × on/off`) + eval completa + report conclusivo S7 (STOP F). |
| **Mese 2 — Step 2.7** (`roadmap_month2.md`): confronto `free` vs `assisted`, **sweep $K \in \{0,1,4,9,16,32\}$** annidati su finalista congelato, `ParamMean+AO`, `GappyPOD+AO/T-block`, off-anchor separato, **gate opzionale** sulla testa residua appresa | Astra primaria + `median9` coprono **`K=9` su S5 seed 0** (`86a925a`, `04e3b68`, report `07`/`08`). S7-A estende `K=9` ai checkpoint Opus. | Mancano: (a) sweep $K \in \{0,1,4,9,16,32\}$ annidati sul finalista congelato `C149` (3 seed) con `AO_K` e `median_K` congelato + `GappyPOD+T-block` matched; (b) decisione formale sul gate della testa residua appresa (`anchor_residual.py`). | **Azione necessaria post-S7 (solo eval offline, 0 training per lo sweep $K$):** eseguire lo sweep $K$ sul finalista `C149` congelato + baseline matched, poi applicare il gate (raccomandazione attesa: se `C149 + medianK` basta, rinviare la testa appresa al Mese 3 come esplicitamente previsto da Step 2.7). |
| **Mese 2 — Step 2.8** (`roadmap_month2.md`): matrice essenziale completa con variabilità/costi; eventuale controllo su 2 fold interni TRAIN (con ri-fit); eventuale **`month2_selection_freeze.json` + pilot `TEST_INTERP/TEST_EXTRAP`** | Costi S5/O5 e curve S6/O5 disponibili; variabilità multi-seed in arrivo da S7-B. `TEST` rigorosamente chiuso. | Mancano: (a) decisione esplicita se eseguire o rinviare i 2 fold pseudo-extrap TRAIN; (b) congelamento formale `month2_selection_freeze.json` + decisione se/quando eseguire il pilot controllato su `TEST_INTERP/TEST_EXTRAP`. | **Decisione separata post-2.7:** non mischiare in S7. Decidere con Enrico e supervisori se chiudere il Mese 2 con un pilot `TEST` congelato (dopo `month2_selection_freeze.json`) e/o con un controllo fold diagnostico (dichiarando che un fold post-hoc non è un `TEST` indipendente). |
| **Mese 2 — Step 2.9** (`roadmap_month2.md`): report breve conclusivo Mese 2, scelta singola direzione Mese 3, aggiornamento diario/README/stato, allineamento con i supervisori | Report `00–08` (e questa proposta `09`) esistenti ma distribuiti su branch diversi; nessuna chiusura finale Mese 2. | Mancano sintesi finale Mese 2, scelta formale direzione Mese 3 e **ratifica scientifica dei supervisori (Deng Lab)**. | **Azione documentale + confronto Lab post-2.8:** scrivere report finale Mese 2, registrare scelta Mese 3 e presentare a Prof.ssa Deng/supervisori. Finché manca la ratifica, dichiarare *"completato operativamente, revisione supervisori aperta"*. |
| **Integrazione tecnica e pulizia Git** (`roadmap_month2.md` §Step 2.1–2.9) | Lavoro consolidato distribuito su `mission/baseline-consolidation` (`3cf7bf0`), `mission/dense-unet`, `mission/astra-level`, `astra/median9-second-pass`, `opus/o5-eval-attempt2`, `review/*`. | Nessun merge su `main` o ramo integrato unico; `roadmapsparse.md` e `roadmap_month2.md` vanno aggiornati solo dopo approvazione/chiusura. | **Incarico dedicato di integrazione** dopo approvazione/chiusura S7 (non durante questa proposta), mantenendo intatti gli artefatti storici e senza riscrivere `docs/proposal.md`. |

---

### 5.2 Distinzione chiara dei quattro livelli di chiusura

Per evitare di confondere codice scritto, esperimenti eseguiti e approvazioni scientifiche, separiamo i quattro livelli che portano da oggi alla chiusura del Mese 2:

1. **Completamento sperimentale:**
   - *Per chiudere `roadmapsparse.md` (STOP F):* **S7-A + S7-B** (0 + 4 training GPU, ~19–24 GPU-h calcolo) — ed eventualmente S7-Alt1 o S7-Alt2 solo se scelte esplicitamente.
   - *Per completare gli esperimenti residui del Mese 2:* **Step 2.7** (sweep $K \in \{0,1,4,9,16,32\}$ offline + gate testa residua) e **Step 2.8** (eventuali fold diagnostici e/o pilot `TEST` dopo `month2_selection_freeze.json`).
2. **Sintesi, report e integrazione tecnica:**
   - Redazione del report di chiusura S7 (STOP F) e, dopo 2.7–2.8, del report conclusivo Mese 2 (`Step 2.9`).
   - Unificazione controllata (merge) dei branch tecnici e dei report `06`/`07`/`08`/`09` su una linea coerente, con aggiornamento finale di `docs/RESEARCH_LOG.md`, `roadmapsparse.md`, `roadmap_month2.md` e `README`, preservando tutti i manifest e gli hash storici.
3. **Decisioni e ratifica dei supervisori (Deng Lab):**
   - Ancora aperta dallo Step 1.6 / 2.1. Le autorizzazioni operative di Enrico consentono di procedere fino alla preparazione e agli esperimenti di sviluppo (`VAL`), ma la **ratifica scientifica** (in particolare prima di aprire `TEST` in Step 2.8 e per fissare la direzione del Mese 3 in Step 2.9: peso relativo fra `free` e `assisted`, eventuale passaggio a multi-densità/rumore/termocoppie realistiche o finestre temporali) resta un passaggio esplicito da svolgere con i supervisori.
4. **Estensioni fuori dal minimo del Mese 2 (da rinviare al Mese 3–4):**
   - Sensibilità strumentale realistica delle ancore $T$ (rumore, risposta temporale, layout rake/intrusività), training multi-densità esteso / nuovi layout di occlusione PIV, eventuali modifiche architetturali GLU (`d_model`, banda adattiva, importance) o U-Net (conv normalizzate/Voronoi), 3C, finestre temporali / forecasting, incertezza calibrata.

---

## 6. Decisioni richieste ora a Enrico e ChatGPT/Sol

Prima di scrivere qualsiasi roadmap operativa o dare un nuovo incarico a Cursor/Sonnet, si sottopongono **cinque decisioni puntuali**:

1. **Decisione O6 (Opus):** Approvare la chiusura di Opus (O0–O5) nel suo perimetro e adottare il regime **`C149` (2980 frame) + `ScientificPresentationStream`** (normalizzatore 760 congelato) come base dei prossimi training `SparseUNet` in S7, tenendo separato lo storico S5 (`A38`)?
2. **Blocco S7-A (zero training, ~35–40 min GPU + ~2 min CPU):** Autorizzare subito il ponte diagnostico **A4 sui 6 run Opus esistenti** (`free/oracle/AO9/mean9/median9`) insieme al completamento **`GappyPOD` snapshot-swap** mancante da S6?
3. **Blocco S7-B (4 training GPU, ~19–24 GPU-h calcolo totali inclusa eval):** Approvare le repliche **`seed 1` e `seed 2` di `SparseUNet C149 p_on` e `p_off`** (accoppiate ai `seed 0` già pronti `O-C149-on/off-s0` per avere 3 seed completi) come strada principale di S7 per chiudere `roadmapsparse.md` (STOP F)?
4. **Alternative opzionali (S7-Alt1 / S7-Alt2):** Confermare il **rinvio** sia di `GLU C149` (Alt1: chiudendo il confronto GLU allo screening `A38 seed 0`) sia del training multidensità `1/5/20%` (Alt2: rinviato a Mese 3), oppure promuovere una delle due per uno specifico obiettivo di laboratorio?
5. **Confine S7 → Mese 2 (2.7–2.9):** Confermare la sequenza **S7 (STOP F) → Step 2.7 offline sweep $K \in \{0,1,4,9,16,32\}$ (+ decisione gate testa residua) → allineamento supervisori prima di `month2_selection_freeze.json` / eventuale pilot `TEST` (2.8) e chiusura Mese 2 (2.9)**?

---

*Fine della proposta. Nessun training, inferenza, submission Slurm, merge o apertura di `TEST` è stato avviato.*
