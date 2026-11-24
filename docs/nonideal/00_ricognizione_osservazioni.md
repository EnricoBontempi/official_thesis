# Ricognizione — dati, osservazioni reali e degradazioni trasferibili

**Data:** 24 novembre 2026 · America/New_York
**Scopo:** base documentale del Blocco I di [roadmaptests.md](../../roadmaptests.md). Raccoglie ciò che è stato verificato oggi da codice, manifest e fonti primarie. Non contiene risultati di modelli.
**Perimetro:** sola lettura di codice e documenti, controlli leggeri su metadati e su frame **TRAIN** già in cache locale. Nessuna inferenza, nessun job, nessun accesso a TEST, VAL letto solo nei manifest.

Legenda di affidabilità usata sotto:
- **[V]** verificato direttamente (codice, file, calcolo locale riproducibile);
- **[P]** letto in fonte primaria (testo verbatim);
- **[S]** riportato da una sintesi della fonte, non letto verbatim: da ricontrollare prima di citarlo;
- **[NV]** non verificato / non trovato.

---

## 1. RealPDEBench, combustione: che cosa contiene

### 1.1 Origine dei campi

| Aspetto | Stato | Contenuto |
|---|---|---|
| Natura | [P] | LES 3D implicita e instazionaria di fiamme swirl NH3/CH4/aria; interazione turbolenza-chimica con Eddy Dissipation Concept. arXiv:2601.01829 §3.2: «a three-dimensional, implicit, unsteady LES»; dati «cross-sectional». |
| Solver, meccanismo, BC | [S] | STAR-CCM+ 2022.1, meccanismo ridotto 38 specie / 184 reazioni, due inlet a portata, outlet in pressione (Appendice B, solo sintesi). |
| Mesh, modello SGS, geometria del bruciatore | [NV] | Non trovati nel testo letto. |
| Come si passa dalla LES alla griglia 128×128 | [NV] | Piano, interpolazione dalla mesh nativa, filtri, crop: **non dichiarati** nelle parti lette. Il campo "vero" è quindi già un campo LES interpolato a una scala di filtro ignota. |
| Modalità reale del dataset | [P] | Solo intensità OH* (chemiluminescenza, 1 canale). Nessuna PIV e nessuna temperatura misurata per la combustione; la PIV del paper riguarda i sistemi fluidi. |
| Allineamento simulazione/esperimento | [S] | Stesse condizioni operative; confronto del segnale OH* integrato e in frequenza, non registrazione campo per campo. |
| Licenza | [P] | CC BY-NC 4.0 (dataset card HF). |

Incoerenze documentate dalle fonti: docs «4000 Hz for 1 s» con 2001 frame a Δt=2.5×10⁻⁴ s (=0.5 s); camera 2000 fps nell'appendice [S]. Non incidono sul nostro uso (frame singoli), ma vanno citate correttamente.

### 1.2 Struttura e coordinate (dal repository)

| Aspetto | Stato | Contenuto |
|---|---|---|
| Canali | [V][P] | 15 numerici; T=ch10 (K), Velocity[i/j/k]=ch11/12/13 (m/s), ch14 magnitudo esportata non euclidea (`docs/month1/02_pilot_data_audit_report.md`). |
| Griglia | [V] | 128×128, x∈[−0.0369, 0.0369] m (colonne), y∈[0.0629, 0.1591] m (righe); Δx≈0.581 mm, Δy≈0.758 mm: **celle anisotrope** (`geometry.npz` della cache C149). |
| Tempo | [V] | Δt=0.25 ms; C149 usa frame ogni 10 indici (2.5 ms), 149 per condizione. |
| Maschera di dominio | [V] | Nessuna: `mask_all_true: true` in tutti i record di `frames.json`; ogni pixel è un valore numerico valido. |
| Componenti usate in input | [V] | Solo vi, vj (planari, `velocity_components: planar_2C`); vk (fuori piano) **non** è nella cache, ma è nel dato grezzo locale. |
| Normalizzazione | [V] | z-score globale fit-only su 760 frame TRAIN: vi μ=−0.060, σ=5.674 m/s; vj μ=4.272, σ=11.910 m/s (`train_norm_stats_month2_fit.json`, sha `678263fb…`). Lo zero normalizzato corrisponde alla velocità media, non a 0 m/s. |
| Orientamento | [V parziale] | Su 298 frame TRAIN: getti freddi (T<1200 K, 5.1% dei pixel) nelle colonne ~90–127, con vj media ≈ +18.4 m/s e vi ≈ −0.2 m/s; vi media antisimmetrica rispetto alla riga 64. Quindi **vj è la componente lungo l'asse delle colonne (direzione dei getti)** e vi quella lungo le righe. La nota pilot del Mese 1 («y assiale, bruciatore sotto») è incoerente con questo e con l'audit full30; il segno di vj rispetto a +x resta da verificare. Non incide su operatori isotropi; incide su qualunque operatore direzionale. |

### 1.3 Scale utili per tradurre le degradazioni (solo frame TRAIN)

Calcolo leggero, riproducibile, su 298 frame TRAIN (ogni 10° frame della cache C149), velocità ricostruita da valori normalizzati con le statistiche congelate:

- |u| planare: mediana 9.0 m/s, p99 35.9 m/s, p99.9 46.9 m/s, max 57.9 m/s.
- Effetto di una media a finestra quadrata centrata applicata al campo denso (20 frame TRAIN, uno per condizione):

| Finestra (celle) | ≈ mm (x × y) | RMS |Δu| | p95 |Δu| | Frazione pixel con |Δu|>1 m/s |
|---|---|---:|---:|---:|
| 3×3 | 1.7 × 2.3 | 0.63 m/s | 1.16 m/s | 6.7% |
| 5×5 | 2.9 × 3.8 | 1.12 m/s | 2.19 m/s | 21.6% |
| 7×7 | 4.1 × 5.3 | 1.56 m/s | 3.10 m/s | 37.4% |

Lettura: una media spaziale alla scala tipica di una finestra PIV cambia la velocità di un ordine di grandezza paragonabile a un rumore di misura plausibile. È quindi **un asse distinto**, da non confondere con il rumore. (Le finestre pari sono state scartate perché introducono uno spostamento di mezza cella.)

Conversione del rumore in unità del modello: σ=0.5 m/s ⇒ 0.088 (vi) e 0.042 (vj) normalizzati; σ=1.5 m/s ⇒ 0.264 e 0.126. Lo stesso rumore fisico pesa il doppio su vi: il rumore va sempre definito in m/s e normalizzato dopo.

## 2. Che cosa misurerebbe una PIV in questo combustore (differenza con la slice numerica)

La slice numerica **non** è una PIV. Una PIV planare 2C nello stesso piano fornirebbe, a confronto:

1. **Media spaziale**: ogni vettore è lo spostamento medio delle particelle in una finestra di interrogazione (e nello spessore del foglio laser), non il valore puntuale del campo LES.
2. **Campionamento regolare e denso** in un campo di vista, non siti casuali al 5% su tutto il dominio.
3. **Errore casuale** e, in parte, **correlato** tra vettori vicini (finestre sovrapposte).
4. **Vettori mancanti o spuri**: riflessi su pareti/finestre, ombre, seeding scarso, perdita di coppie per moto fuori piano; gli spuri vengono in genere rimossi dalla validazione e diventano mancanti.
5. **Effetti specifici della fiamma** (motivati fisicamente, non quantificati da fonti lette): per gas ideale a pressione costante la densità del gas combusto è circa T_u/T_b ≈ 1/6–1/7 di quella dei reagenti, quindi una densità di particelle trasportate dal flusso cala in proporzione; inoltre il forte moto fuori piano dello swirl (vk dell'ordine di ±15 m/s nell'audit del Mese 1) aumenta la perdita di coppie. Entrambi i fenomeni rendono le lacune **correlate con la fiamma, quindi con T**.

Il punto 5 crea un rischio metodologico: una maschera di validità che dipende da T **porta informazione su T** nel modello. È realistico, ma va trattato come asse separato e controllato (vedi roadmap, D5).

Che cosa resta da un domain expert di diagnostica laser: spessore del foglio, finestra e sovrapposizione realistiche per questo bruciatore, tipo di seeding, regioni ottiche inaccessibili, frazioni tipiche di vettori rigettati in fiamma. Il Prof. D'Errico, secondo quanto riferito da Enrico, non si occupa direttamente di queste diagnostiche.

## 3. EngineBench (Oxford TPSRG)

Fonti: arXiv:2406.03325 (HTML, in parte [S]); repository `sambkr/EngineBench` letto via API GitHub [V]; pagina Oxford non raggiungibile (HTTP 500); pagine Kaggle non leggibili senza rendering.

| Domanda | Stato | Risposta |
|---|---|---|
| Dati | [P] | PIV in motore ottico TCC-III, motorizzato (nessuna combustione), fasi in gradi di manovella; subset LSP_small 5205 snapshot, piano di swirl inferiore, 1300 rpm, 40 kPa. |
| Griglia rilasciata | [P] | 50×49 vettori, padding a zero fino a 128×128. |
| Spaziatura vettori | [S] | 1.25–1.4 mm (≈ 2.2×1.7 celle della nostra griglia). Finestra, sovrapposizione, validazione: [NV]. |
| Vettori mancanti nel dato rilasciato | [V] | Codificati come **zero esatto**. Il loader (`michigan_piv.py`) costruisce la maschera iniziale con `(u == 0) | (v == 0)`; quindi validità implicita, nessun flag dedicato. Non è distinguibile nel codice un vettore mancante da uno nullo vero; tipo e frazione di lacune interne al campo di vista [NV]. |
| Lacune del benchmark | [V][P] | **Artificiali**: `SheetGapHandler` (poligono da 4 punti casuali sui bordi, rimozione ≤25% dei vettori validi; configurazioni 10% e 25%, in media ~7% al livello 10%) e `DefaultHandler` (quadrato fisso). In test, due maschere fisse verticale/orizzontale. Motivazione dichiarata: campo di vista ridotto da finestra o orifizio. |
| Ordine operazioni | [V] | Standardizzazione per fase (media/deviazione per fase) → lacune → zeri nelle lacune → padding. |
| Incertezza | [P] | «average velocity error of 1.5−8%» (90–270 CAD, a seconda del piano), da Schiffmann et al. 2016. Nessun indicatore di qualità per vettore, nessuna correlazione degli errori. |
| Temperatura o scalari | [P] | Nessuno. |
| Licenza | [P]/[S] | Testo: CC BY 4.0; appendici [S]: CC BY-NC-SA 4.0. Da chiarire su Kaggle prima di qualsiasi uso dei dati. |

Trasferibilità al combustore: geometria, fisica (motore motorizzato, nessuna fiamma), scale e ottica diverse. Utile come **precedente metodologico** (validità esplicita, lacune per regioni contigue anziché punti sparsi, maschere fisse di test) e come ordine di grandezza dell'errore relativo; **non** come maschera realistica del nostro bruciatore.

## 4. Archivi TNF/Sandia (scouting mirato)

- Pagina swirl dell'archivio TNF: solo **Sydney swirl** e **TECFLAM**.
- **Sydney swirl** [P]: velocità LDV (puntuale, 2–3 componenti), composizione con Raman/Rayleigh/LIF istantanei misurati a Sandia; campagne e file separati, con l'avvertenza che profili di velocità e composizione «may not overlap». Regime non premiscelato. Nessuna PIV.
- **TECFLAM** [P]: dati su richiesta al gruppo di Darmstadt; diagnostiche non dichiarate sulla pagina.
- Fiamme pilotate e bluff-body: non premiscelate, senza swirl; velocità su richiesta per alcune.
- **Nessun dataset letto** fornisce velocità e temperatura simultanee e co-localizzate a singolo colpo. Il combustore modello DLR (PIV/OH-PLIF/Raman, Meier, Stöhr, Boxx) **non** compare nell'archivio TNF letto; non è stato verificato altrove.

Conclusione: lo scouting non fornisce oggi un riferimento sperimentale per validare la ricostruzione di T. Non motiva un cambio di dataset.

## 5. Incertezza PIV nella letteratura metrologica

| Fonte | Stato | Contenuto |
|---|---|---|
| Sciacchitano et al. 2015, MST 26 074004 | [P] | «Most studies converge to results in the range 0.03 to 0.1 pixels»; errori casuali da rumore camera, sfocatura, «out-of-plane particle motion and inhomogeneous seeding density». Monte Carlo sottostima l'errore reale. |
| Wieneke 2015, MST 26 074002 | [P] | Stima grezza spesso citata «0.05 to 0.1 pixel». [S]: livelli rilevanti 0.02–0.3 px; il moto fuori piano spesso domina; autocorrelazione dell'errore ~nulla oltre ~una finestra. |
| Sciacchitano 2019, MST 30 092001 | [P] | «A typical figure … 0.1 pixel units»; la costante universale è «overly simplistic». [S]: errori sperimentali fino a «several tenths of a pixel»; correlazione tra misure adiacenti. |
| Westerweel & Scarano 2005 (validazione) | [NV] | Non letto. Nessuna frazione tipica di vettori spuri verificata. |
| PIV in fiamme swirl (DLR GTMC, NH3/CH4) | [NV] | Parametri (finestra, spaziatura, seeding, frazione rigettata) non verificati. |

Conversione in m/s (ipotesi dichiarata, classe B): σ_u ≈ σ_px · U_fs / d_max. Con σ_px=0.1 px, spostamento massimo di progetto d_max≈8 px (ipotesi di progetto, non dato misurato) e U_fs = p99–p99.9 di |u| TRAIN (36–47 m/s): **σ_u ≈ 0.45–0.6 m/s**. Con σ_px=0.3 px: **≈1.35–1.75 m/s**. Il risultato dipende linearmente da d_max e U_fs, che dipendono dal progetto dell'esperimento.

## 6. Supporto reale del codice per osservazioni mancanti

| Elemento | Stato | Contenuto |
|---|---|---|
| U-Net masked | [V] | Input a 7 canali `[vi, vj, Mv, x, y, nh3, phi]` (`training/sparse_view.py`); siti non osservati = 0 normalizzato **con Mv=0**. Un vettore invalidato si rappresenta esattamente come un sito non osservato: validità esplicita nativa, nessuno zero spacciato per misura. |
| U-Net e densità | [P report] | Il report Mese 2 segnala che U-Net peggiora anche passando a 20% (ipotesi non testata: sensibilità alla statistica del canale Mv). Rimuovere siti cambia la densità: il controllo a pari numero di siti è necessario. |
| GLU adattato | [V] | Token `[qx, qy, vi_norm, vj_norm]`; `padding_mask` e N eterogeneo nel batch **vietati** (`glu_reconstructor.py`, `glu_minimal.yaml`); decoder top-k=min(16, N). N variabile tra frame è supportato solo con batch omogenei (es. batch=1); N=0 non è supportato (cdist su insieme vuoto, encoder senza token). |
| GLU e batch | [P INDEX S7] | Diagnostico S7: nessuna dipendenza dagli altri esempi, ma sensibilità numerica dei pareggi top-k alla forma del batch (fino a 0.04 K per frame nello smoke a sottoinsieme). Una valutazione con batch diversi richiede un proprio replay ideale. |
| Ricostruzione con velocità nascoste | [V] | Nessun percorso di riempimento: i siti invalidati devono semplicemente sparire (U-Net: Mv=0; GLU: token rimosso). |

## 7. Identità dei sei checkpoint di riferimento

Da `configs/experiments/month2_s7_c149_eval_matrix.json` (checkpoint `best`, criterio `first_absolute_min_val_mean_condition_rmse_K`):

| Run | Famiglia | Epoca best | file_sha256 (prefisso) | Training SHA |
|---|---|---:|---|---|
| O-C149-on-s0 | U-Net | 90 | `bed10229` | `d633200` (riuso) |
| S7-C149-unet-on-s1 | U-Net | 240 | `008b1913` | `f792b73` |
| S7-C149-unet-on-s2 | U-Net | 180 | `f11be53f` | `f792b73` |
| S7-C149-glu-on-s0 | GLU | 290 | `e1aad601` | `f792b73` |
| S7-C149-glu-on-s1 | GLU | 630 | `e33673b5` | `501d41a` (continuazione) |
| S7-C149-glu-on-s2 | GLU | 400 | `d5034d49` | `f792b73` |

Input condivisi: cache C149 `fa1c9358…`, statistiche `678263fb…`, maschere eval seed 20261116 `random_sparse_primary`, manifest maschere `2e2d8ff6…`. Percorsi sul cluster nella matrice; **disponibilità fisica sul cluster non verificata oggi** (nessun accesso eseguito): è il primo controllo del Blocco II.

## 8. Che cosa non è stato fatto

- Appendice B del paper RealPDEBench non letta verbatim.
- Codice di EngineBench letto solo per loader e lacune; dati Kaggle non scaricati.
- Parametri PIV in fiamme swirl non verificati da fonte primaria.
- Nessuna lettura di vk dal dato grezzo, nessun controllo su VAL o TEST, nessuna inferenza.
