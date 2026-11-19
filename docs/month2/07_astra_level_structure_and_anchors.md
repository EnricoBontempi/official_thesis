# Report ASTRA — Livello termico, struttura spaziale e nove temperature osservate

**Stato:** Blocco primario S5 A0–A3 concluso nel perimetro dichiarato; estensione A4 rinviata alla consegna Opus e a decisione esplicita. Riesame delle proposte Neon e decisione S7 successivi alla sintesi Astra/Opus; sparse e Mese 2 restano aperti.  
**Repository:** `EnricoBontempi/official_thesis`  
**Commit tecnico analizzato:** `86a925af61accdef0b235ac084827cf639c1040c` (branch `mission/astra-level`)  
**Codice di esecuzione A2:** `eb264bdd637f216f51267c4990f583d0ed165a32` (job Engaging `29612403–29612407`, tutti `COMPLETED 0:0`)  
**Fonti primarie:** `results/month2/astra_level_structure/` (tabelle CSV, `verification/`, `provenance/`, 6 figure PNG), `docs/month2/astra_analysis_handoff.md`, `configs/experiments/month2_astra_level_structure.yaml`, `roadmapastra.md`.

> **Avvertenza sul perimetro:** L'analisi è **esplorativa**. Le tre condizioni `VAL` hanno guidato selezione del checkpoint, scheduler e arresto nei training S5; `oracle-offset` legge l'intero campo target `T` ed è un riferimento puramente diagnostico; le nove temperature di `AO9` sono campioni numerici ideali (rumore 0, nessuna dinamica o intrusività strumentale) e non validano termocoppie reali. Le tre repliche sono **tre maschere di velocità**, non tre seed di training (unico `seed 0`).

---

## Sintesi esecutiva

1. **L'ipotesi sul livello termico è confermata dall'oracolo.** Nel caso motivante `40NH3_0.9`, il crollo dell'IoU calda di U-Net `p_on` (`0.310`, area predetta `24.4%` contro `78.5%` vera) era in larga parte recuperabile correggendo il bias medio di `-29.35 K`. Applicando la sola traslazione scalare perfetta (`oracle`), U-Net `p_on` passa a RMSE `26.80 K` e IoU calda `0.950`, superando GLU `p_on` (`28.55 K`, IoU `0.936`). Su tutta `VAL`, U-Net ha errore strutturale (`centered-RMSE`) nettamente inferiore a GLU (`38.00–38.36 K` contro `42.72–44.61 K`).
2. **Tuttavia, `AO9` nella ricetta congelata (griglia fissa $3\times 3$ `[32, 64, 95]`, media con $\tau=1$) fallisce come correttore generale.** Su `VAL`, `AO9` peggiora o lascia invariata la RMSE macro di 3 reti su 4 (`U-Net p_on`: `+0.17 K`, `GLU p_on`: `+5.31 K`, `GLU p_off`: `+0.67 K`; unico lieve miglioramento `U-Net p_off`: `-0.67 K`). Su `seen`, peggiora 3 reti su 4 (`GLU p_on` peggiora in **20/20** condizioni).
3. **Una possibile spiegazione del fallimento è suggerita dalle mappe esemplificative:** i due siti a destra della griglia $3\times 3$, `(row=32, col=95)` e `(row=95, col=95)`, cadono sul bordo dei getti freddi (V-jet) dove le reti presentano errori locali di centinaia di kelvin. Una media di 9 residui può trasferire grandi errori locali in un offset globale poco rappresentativo. Nelle tabelle aggregate si osserva un bias negativo dopo AO9 (`-22.07 K` su `20NH3_1.1` per U-Net `p_on`, `-25.81 K` per GLU `p_on`).
4. **Le reti restano nettamente superiori alle baseline anche a parità di T osservate.** Con `AO9`, U-Net (`42.86–43.14 K` su VAL; `53.78–53.81 K` su seen) quasi dimezza l'errore di `GappyPOD+AO9` (`76.24–79.95 K` su VAL; `111.88–115.97 K` su seen), stacca `ParamMean+AO9` (`94.30 K` VAL) e `GlobalMean+AO9` (`122.01 K` VAL), e supera di oltre 7× `T_only_const` (`311.79 K` VAL, dove i getti hanno RMSE `> 1100 K`).
5. **Il collo di bottiglia strutturale residuo è nei getti freddi (`T < 1800 K`, RMSE `124–147 K` su VAL),** Una traslazione uniforme lascia invariata la centered-RMSE sullo stesso insieme di siti; può invece migliorare o peggiorare la RMSE nei getti.

---

## 1. Verifica tecnica: contratti, aggregazioni, replay e tracciabilità

Sono state verificate direttamente le tabelle in `results/month2/astra_level_structure/tables/`, i file di verifica in `verification/` e il codice sorgente (`src/combustion_recon/evaluation/astra_level.py`, `scripts/18_astra_level_structure_eval.py`, `scripts/19_astra_replay_k0_verify.py`).

### 1.1 Coerenza matematica e aggregazioni
- **Definizione di macro:** media per frame $\to$ media sui frame della condizione (`sim_id`) $\to$ media uniforme sulle condizioni (20 su `seen`, 3 su `VAL`) per singola replica; `replica=mean` è la media aritmetica delle tre repliche. Ricalcolando `per_condition_metrics.csv -> macro_headline_wide.csv` si ottiene scarto massimo `5.03e-7 K` su `seen` e `5.00e-7 K` su `VAL`.
- **Differenze appaiate (`paired_diff_macro_long.csv`):** coerenti con le differenze tra macro di `macro_headline_wide.csv` entro `8.91e-7 K`.
- **Invarianza strutturale e oracolo:** per tutti gli 8 modelli con trattamenti `{free, oracle, ao9}` su `seen` e `VAL`, `all_centered_rmse_K` e `off_centered_rmse_K` sono identiche tra `free`, `oracle` e `ao9`, e `oracle all_rmse_K == all_centered_rmse_K` entro `1e-6 K`.
- **Convenzioni di segno e shrink AO9:** in `astra_level.py`, `bias = mean(T_pred - T_true)`, `delta_oracle = mean(T_true - T_pred)`, `delta_ao9 = sum_k(T_obs_k - T_pred_k) / (9 + 1)`. Verificato su tutte le combinazioni ruolo × modello (scarto `< 1e-4 K`):
  - `all_bias_K(free) == -delta_oracle_K`,
  - `all_abs_bias_K(free) == abs_delta_oracle_K`,
  - `all_bias_K(ao9) == delta_ao9_minus_oracle_K`,
  - `all_abs_bias_K(ao9) == abs_ao9_minus_oracle_K`,
  - `delta_ao9_K == 0.9 * anchor_mean_resid_K`.
- **Distinzione tra metriche (avvertenza):** `all_bias_K` è il bias firmato medio (può cancellarsi tra frame/condizioni di segno opposto), `all_abs_bias_K` è la media di `|bias|` per frame (non il valore assoluto del bias macro), `all_rmse_K` è la media delle RMSE per frame. L'identità $\text{RMSE}^2 = \text{bias}^2 + \text{centered}^2$ vale **per singolo frame**, non sulle medie macro; nessuna decomposizione impropria delle macro viene usata in questo report.
- **Completezza e NaN:** `completeness_rows.csv` e `region_valid_frame_counts.csv` confermano 220/220 frame validi su `seen` e 123/123 su `VAL` in tutte le regioni (`hot`, `peak`, `jet`, `nonhot`, `hot_iou`), senza regioni vuote. Gli unici NaN (`1029` righe = `(220+123)×3`) sono nelle colonne offset di `T_only_const`, dove sono legittimi per definizione.
- **Nota su `mse_recovery_vs_oracle`:** nei file `offsets_macro_long.csv` compaiono valori macro negativi molto grandi (es. `-721,299` su GLU `p_on` VAL rep 0). Non è un errore di codice: segue esattamente la formula congelata `(MSE_free - MSE_AO9)/(MSE_free - MSE_oracle)` per frame con soglia `den > 1e-6 K^2` senza troncamento. Quando un frame ha bias `free` quasi nullo (denominatore $\sim 10^{-4}\text{ K}^2$), un errore di offset `AO9` di `10–20 K` genera rapporti negativi enormi che dominano la media aritmetica. La metrica va quindi letta solo come segnale qualitativo di instabilità, usando i delta in Kelvin (`RMSE`, `|bias|`, `bias`) come misura principale.

### 1.2 Audit del replay K=0 e delle maschere baseline
- **Replay neurale sostitutivo (`verification/replay_k0_verification.json`, `scripts/19_astra_replay_k0_verify.py`):**
  - I `replay_k0.json` originali erano rimasti non eseguiti (`ok=null`) perché `scripts/18` cercava `checkpoint=="best"` in `summary_all_settings.csv` di S6, che contiene solo `last` e `best_swapped`.
  - La verifica sostitutiva confronta Astra `free` contro tre fonti storiche distinte (SHA-256 registrati; hash diversi non dimostrano indipendenza della produzione degli artefatti):
    - **Fonte A** (`S6 per_frame_best_cov05_*`): **24/24 PASS** (tutti i 4 run × 2 ruoli × 3 repliche, join su `(sim_id, frame_index)`), scarto macro massimo `8.66e-4 K`, scarto per-frame massimo `0.0335 K` (entro `0.05 K`).
    - **Fonti B/C** (`S5 eval_history.json` e `S5 evals/epoch_<best>/*`): **16/16 PASS** ciascuna sulle celle disponibili (VAL rep 0/1/2 e seen rep 0), coincidenza esatta entro `1.42e-14 K` (B) e `0.000 K` (C).
    - **Epoche `best`:** confermate identiche a `summary.json` di S5 (`60, 140, 440, 650`).
  - **Valutazione critica:** la coincidenza esatta (`≤1.4e-14 K`) con S5 su VAL (tutte e 3 le repliche) e seen rep 0 sostiene la coerenza numerica del replay; l'identità dei checkpoint e la policy fp32 sono documentate separatamente dalla provenienza. Lo scarto per-frame fino a `0.033 K` rispetto a S6 (fonte A) rientra nella tolleranza congelata. Una differenza di esecuzione numerica è una possibile spiegazione, non una causa accertata da questa analisi. Le 16 celle `MISSING` in B/C (`seen rep 1/2`) sono coperte dalla fonte A: il gate previsto è coperto, con minore ridondanza dei riferimenti.
- **Maschere baseline (`verification/baseline_mask_crosscheck.json`):**
  - In `scripts/18` (righe 351–359), GappyPOD legge le maschere persistite `sites_5pct` da `nested_sites.npz` senza fallback da seed (`KeyError` se mancante) e `baseline_mask_crosscheck.json` conferma `n_velocity_sites=819` e `n_t9_on_velocity_sites` identici ai neurali su tutte le 660 chiavi `seen` e 369 chiavi `VAL`. Il replay `GappyPOD` K=0 coincide con la tabella storica entro `~1e-8 K`.
  - **Limite di tracciabilità residuo:** le tabelle baseline non registrano `mask_hash` per frame, e `baselines/replay_k0.json` confronta la media sulle repliche senza registrare l'hash del file storico di riferimento. Non è una prova crittografica completa dei siti consumati, ma codice + conteggi + replay esatto escludono con ragionevole certezza errori di allineamento.
- **Esito audit:** **nessun errore bloccante** nelle tabelle o nelle figure. Il pacchetto tecnico `86a925a` è utilizzabile integralmente per l'analisi scientifica.

---

## 2. Risultati globali: free, oracle, AO9 e comportamento off-anchor

Fonti: `tables/macro_headline_wide.csv`, `tables/offsets_macro_long.csv`, `tables/paired_diff_macro_long.csv`. Tutti i valori sono medie sulle 3 repliche (in K).

### 2.1 Macro risultati su `val_eval` (3 condizioni, 123 frame)

| Modello | `p` | RMSE free | Cent. (oracle) | RMSE AO9 | $\Delta$(or−free) | $\Delta$(AO9−free) | $\Delta$(AO9−or) | Bias free | `|Bias|` free | Bias AO9 | `|Bias|` AO9 | `off−all` RMSE (AO9) |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **U-Net `p_on`** | sì | 42.96 | 38.36 | 43.14 | −4.60 | **+0.17** | +4.77 | −7.50 | 12.88 | −12.35 | 14.83 | −0.012 |
| **U-Net `p_off`** | no | 43.53 | 38.00 | 42.86 | −5.52 | **−0.67** | +4.85 | −5.85 | 17.52 | −6.96 | 14.66 | −0.012 |
| **GLU `p_on`** | sì | 45.80 | 44.61 | 51.12 | −1.19 | **+5.31** | +6.51 | +2.48 | 8.77 | −12.00 | 19.14 | −0.015 |
| **GLU `p_off`** | no | 47.54 | 42.72 | 48.20 | −4.82 | **+0.67** | +5.49 | −0.61 | 18.58 | −6.52 | 16.83 | −0.011 |
| **GappyPOD `p_on`** | sì | 70.60 | 68.19 | 76.24 | −2.40 | +5.64 | +8.04 | −15.33 | 15.38 | +4.27 | 30.47 | −0.018 |
| **GappyPOD `p_off`** | no | 94.80 | 71.69 | 79.95 | −23.11 | **−14.85** | +8.26 | −46.63 | 57.58 | +0.36 | 27.84 | −0.017 |
| **ParamMean `p_on`** | sì | 88.01 | 84.82 | 94.30 | −3.19 | +6.29 | +9.48 | −16.85 | 18.25 | +8.01 | 34.18 | −0.016 |
| **GlobalMean** | no | 139.70 | 114.84 | 122.01 | −24.86 | **−17.69** | +7.17 | −61.37 | 69.36 | +10.49 | 35.19 | −0.002 |
| **T-only costante** | no | — | 308.06 | 311.79 | — | — | — | — | — | +25.03 | 45.25 | +0.062 |

### 2.2 Macro risultati su `seen_train_eval` (20 condizioni, 220 frame)

| Modello | `p` | RMSE free | Cent. (oracle) | RMSE AO9 | $\Delta$(or−free) | $\Delta$(AO9−free) | $\Delta$(AO9−or) | Bias free | `|Bias|` free | Bias AO9 | `|Bias|` AO9 | Cond. migliorate da AO9 |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **U-Net `p_on`** | sì | 51.67 | 50.61 | 53.81 | −1.06 | **+2.14** | +3.20 | +5.62 | 8.02 | −3.27 | 13.66 | 5 / 20 |
| **U-Net `p_off`** | no | 55.44 | 51.17 | 53.78 | −4.27 | **−1.67** | +2.61 | +3.23 | 16.00 | −0.19 | 12.76 | 12 / 20 |
| **GLU `p_on`** | sì | 55.51 | 55.09 | 58.48 | −0.42 | **+2.96** | +3.39 | −0.79 | 5.59 | −3.30 | 14.82 | **0 / 20** |
| **GLU `p_off`** | no | 60.31 | 58.51 | 61.83 | −1.80 | **+1.52** | +3.32 | +0.83 | 11.31 | −1.37 | 15.65 | 3 / 20 |
| **GappyPOD `p_on`** | sì | 108.36 | 106.59 | 115.97 | −1.78 | +7.61 | +9.39 | +0.03 | 14.95 | −3.43 | 38.13 | 1 / 20 |
| **GappyPOD `p_off`** | no | 113.08 | 103.46 | 111.88 | −9.62 | −1.19 | +8.43 | −1.09 | 35.47 | −0.39 | 35.79 | 12 / 20 |
| **ParamMean `p_on`** | sì | 129.35 | 127.17 | 139.54 | −2.17 | +10.20 | +12.37 | +0.28 | 19.08 | −4.99 | 47.75 | 4 / 20 |
| **GlobalMean** | no | 173.68 | 151.38 | 163.21 | −22.30 | −10.47 | +11.83 | +0.28 | 71.08 | −4.99 | 49.19 | 11 / 20 |
| **T-only costante** | no | — | 315.85 | 323.84 | — | — | — | — | — | +0.97 | 60.74 | — |

### 2.3 Lettura principale dei risultati globali e off-anchor
- **Andamento descrittivo nelle macro esaminate, non soglia generale:** i miglioramenti di `AO9` si osservano quando il modello `free` ha un errore di livello già elevato (`|Bias| free ≳ 16–20 K`), come `GlobalMean` (`-17.69 K` su VAL, `-10.47 K` su seen), `GappyPOD p_off` (`-14.85 K` su VAL) e `U-Net p_off` (`-0.67 K` su VAL, `-1.67 K` su seen). Quando il modello `free` ha già un livello discreto (`|Bias| free ≈ 6–15 K`, cioè tutte le varianti `p_on` e `GLU p_off`), l'errore di stima dell'offset di `AO9` (`|dAO9 − dOr| = 12.8–19.1 K` sui neurali, `27.8–38.1 K` su GappyPOD) **supera il bias iniziale** e degrada sistematicamente la ricostruzione.
- **Distanza residua `AO9 − oracle`:** è sempre positiva in 100% delle celle (60/60 su `seen`, 9/9 su `VAL`), pari a `+2.61`–`+3.39 K` sui neurali su `seen` e `+4.77`–`+6.51 K` su `VAL`.
- **Comportamento `off-anchor` (16375 pixel):** gli scarti tra `off_rmse_K` e `all_rmse_K` sono compresi tra `-0.036 K` e `+0.000 K` su tutti i neurali e baseline (e `+0.037`/`+0.062 K` su `T_only_const`), e l'IoU calda coincide fino alla terza cifra decimale. Poiché `AO9` è una traslazione uniforme dell'intero campo e 9 pixel rappresentano lo `0.055%` della griglia, il comportamento `off-anchor` è praticamente identico al campo completo: eventuali guadagni (come su `40NH3_0.9`) e peggioramenti (come su `20NH3_1.1`) sono reali fuori dalle ancore, non artefatti locali sui 9 siti.

---

## 3. Struttura, regioni calde e lettura delle tre VAL e delle sei figure

Fonti: `tables/per_condition_metrics.csv`, `tables/macro_regional_long.csv`, `tables/figure_panel_check.csv`, `figures/val_*_frame1000_rep0_{fields,signed_error}.png`.

### 3.1 Dettaglio per condizione su `val_eval` (media delle 3 repliche)

#### A) `40NH3_0.9.h5` (caso motivante: area calda vera `78.5%`, media di campo vicina a `2100 K`)

| Modello | Tratt. | RMSE | Cent. | Bias | `|Bias|` | Hot IoU | Area pred | Hot RMSE | Peak RMSE | Jet RMSE | `dOr` | `dAO9` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net `p_on` | free | 39.92 | 26.80 | −29.35 | 29.35 | **0.310** | 24.4% | 33.8 | 57.7 | 74.2 | +29.35 | +24.89 |
| U-Net `p_on` | oracle | **26.80** | 26.80 | 0.00 | 0.00 | **0.950** | 80.3% | **11.6** | 41.6 | 73.4 | +29.35 | +24.89 |
| U-Net `p_on` | AO9 | **29.15** | 26.80 | −4.46 | 8.90 | **0.889** | 74.4% | 16.5 | 44.1 | 73.3 | +29.35 | +24.89 |
| U-Net `p_off` | free | 36.93 | 26.87 | −1.81 | 23.06 | 0.713 | 60.3% | 28.9 | 40.7 | 73.4 | +1.81 | +5.23 |
| U-Net `p_off` | oracle | 26.87 | 26.87 | 0.00 | 0.00 | **0.952** | 78.1% | 11.6 | 32.0 | 77.1 | +1.81 | +5.23 |
| U-Net `p_off` | AO9 | 29.37 | 26.87 | +3.42 | 9.21 | **0.900** | 75.2% | 15.6 | 32.3 | 78.0 | +1.81 | +5.23 |
| GLU `p_on` | free | 29.24 | 28.55 | −3.25 | 4.69 | 0.925 | 76.3% | 16.7 | 53.2 | 70.8 | +3.25 | +7.80 |
| GLU `p_on` | oracle | 28.55 | 28.55 | 0.00 | 0.00 | 0.936 | 77.8% | 15.6 | 50.9 | 70.6 | +3.25 | +7.80 |
| GLU `p_on` | AO9 | 32.43 | 28.55 | +4.55 | 12.21 | 0.891 | 75.4% | 21.8 | 49.7 | 72.4 | +3.25 | +7.80 |
| GLU `p_off` | free | 39.31 | 32.23 | +3.10 | 20.77 | 0.748 | 64.8% | 26.9 | 60.0 | 91.9 | −3.10 | +5.54 |
| GLU `p_off` | oracle | 32.23 | 32.23 | 0.00 | 0.00 | 0.938 | 78.3% | 15.6 | 61.8 | 90.5 | −3.10 | +5.54 |
| GLU `p_off` | AO9 | 35.50 | 32.23 | +8.65 | 13.18 | 0.904 | 78.1% | 19.1 | 57.2 | 94.9 | −3.10 | +5.54 |
| GappyPOD `p_on` | free / AO9 | 48.06 / 55.83 | 44.80 | −16.99 / +25.09 | 16.99 / 31.56 | 0.888 / 0.883 | 71.8 / 78.1% | 26.1 / 43.0 | 57.7 / 70.6 | 116.3 / 114.6 | +16.99 | +42.08 |
| ParamMean `p_on` | free / AO9 | 49.77 / 60.84 | 45.78 | −19.37 / +35.45 | 19.37 / 38.80 | 0.818 / 0.886 | 65.3 / 80.1% | 32.4 / 45.4 | 96.4 / 56.6 | 108.0 / 132.4 | +19.37 | +54.82 |
| T-only costante | — | 327.88 | 323.08 | +51.51 | 54.05 | 0.019 | 2.4% | 55.9 | 202.0 | 1112.1 | — | — |

#### B) `20NH3_1.1.h5` (area calda vera `88.1%`: qui AO9 degrada tutte le reti)

| Modello | Tratt. | RMSE | Cent. | Bias | `|Bias|` | Hot IoU | Area pred | Hot RMSE | Jet RMSE | `dOr` | `dAO9` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net `p_on` | free / oracle / AO9 | 44.05 / 43.52 / **52.45** | 43.52 | +6.32 / 0 / **−22.07** | 6.37 / 0 / 23.26 | 0.984 / 0.985 / **0.948** | 89.0 / 88.7 / 84.6% | 12.8 / 12.8 / 31.0 | 142.7 / 141.1 / 138.8 | −6.32 | **−28.39** |
| U-Net `p_off` | free / oracle / AO9 | 42.29 / 40.29 / **49.73** | 40.29 | +3.14 / 0 / **−16.91** | 10.60 / 0 / 23.77 | 0.983 / 0.984 / **0.954** | 87.5 / 87.4 / 84.6% | 18.2 / 13.9 / 30.3 | 125.2 / 124.9 / 125.8 | −3.14 | **−20.05** |
| GLU `p_on` | free / oracle / AO9 | 54.66 / 52.23 / **63.54** | 52.23 | +15.79 / 0 / **−25.81** | 15.79 / 0 / 28.24 | 0.982 / 0.985 / **0.879** | 89.0 / 88.2 / 78.3% | 17.5 / 14.6 / 37.9 | 179.7 / 174.7 / 170.5 | −15.79 | **−41.60** |
| GLU `p_off` | free / oracle / AO9 | 48.97 / 46.07 / **57.32** | 46.07 | +14.69 / 0 / **−21.75** | 15.33 / 0 / 27.45 | 0.982 / 0.982 / **0.897** | 88.5 / 87.7 / 79.9% | 21.1 / 15.9 / 36.8 | 148.9 / 145.5 / 144.0 | −14.69 | **−36.44** |
| GappyPOD `p_on` | free / oracle / AO9 | 69.60 / 68.84 / 76.46 | 68.84 | −9.31 / 0 / −2.04 | 9.31 / 0 / 30.63 | 0.949 / 0.953 / 0.940 | 84.4 / 85.1 / 84.1% | 39.5 / 36.4 / 48.9 | 199.6 / 204.5 / 204.8 | +9.31 | +7.27 |
| GappyPOD `p_off` | free / oracle / AO9 | 98.34 / 66.91 / 76.27 | 66.91 | −70.68 / 0 / −21.26 | 70.68 / 0 / 28.85 | 0.774 / 0.932 / 0.898 | 68.3 / 82.6 / 79.6% | 83.5 / 39.5 / 54.3 | 157.7 / 173.3 / 167.4 | +70.68 | +49.42 |

#### C) `0NH3_1.05.h5` (area calda vera `91.2%`: getto più intenso, `p_off` migliora con AO9 ma `p_on` peggiora)

| Modello | Tratt. | RMSE | Cent. | Bias | `|Bias|` | Hot IoU | Area pred | Hot RMSE | Jet RMSE | `dOr` | `dAO9` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net `p_on` | free / oracle / AO9 | 44.91 / 44.77 / 47.82 | 44.77 | +0.54 / 0 / −10.52 | 2.93 / 0 / 12.34 | 0.983 / 0.983 / 0.984 | 92.0 / 91.9 / 91.5% | 18.2 / 18.6 / 24.2 | 168.6 / 167.7 / 168.1 | −0.54 | −11.06 |
| U-Net `p_off` | free / oracle / AO9 | 51.36 / 46.85 / 49.46 | 46.85 | −18.89 / 0 / −7.38 | 18.89 / 0 / 11.01 | 0.983 / 0.984 / 0.982 | 90.9 / 91.7 / 91.3% | 28.1 / 18.5 / 23.5 | 173.2 / 176.2 / 176.4 | +18.89 | +11.51 |
| GLU `p_on` | free / oracle / AO9 | 53.50 / 53.04 / 57.37 | 53.04 | −5.09 / 0 / −14.75 | 5.84 / 0 / 16.96 | 0.980 / 0.980 / 0.978 | 91.5 / 91.6 / 90.9% | 22.4 / 22.0 / 30.3 | 189.8 / 189.2 / 188.1 | +5.09 | −9.65 |
| GLU `p_off` | free / oracle / AO9 | 54.32 / 49.85 / 51.79 | 49.85 | −19.62 / 0 / −6.46 | 19.62 / 0 / 9.87 | 0.978 / 0.979 / 0.979 | 90.5 / 91.2 / 91.0% | 31.4 / 22.6 / 26.3 | 173.3 / 176.0 / 175.5 | +19.62 | +13.16 |
| GappyPOD `p_off` | free / oracle / AO9 | 124.60 / 91.54 / 95.86 | 91.54 | −81.65 / 0 / +5.94 | 81.65 / 0 / 22.71 | 0.882 / 0.938 / 0.939 | 80.9 / 86.8 / 86.9% | 103.5 / 62.6 / 68.6 | 215.0 / 230.0 / 234.6 | +81.65 | +87.59 |
| GlobalMean | free / oracle / AO9 | 214.50 / 155.07 / 159.54 | 155.07 | −142.75 / 0 / −8.47 | 142.75 / 0 / 29.37 | 0.774 / 0.924 / 0.915 | 71.0 / 87.0 / 85.8% | 167.8 / 84.9 / 94.8 | 454.3 / 439.3 / 436.2 | +142.75 | +134.28 |

### 3.2 Macro regionale (Hot IoU, Hot/Peak/Jet/NonHot RMSE)

| Ruolo | Modello | Free IoU | Oracle IoU | AO9 IoU | Hot RMSE (F/O/A) | Peak RMSE (F/O/A) | Jet RMSE (F/O/A) | NonHot RMSE (F/O/A) |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| **VAL** | U-Net `p_on` | 0.759 | **0.973** | 0.940 | 21.6 / **14.3** / 23.9 | 29.4 / 25.7 / 35.8 | 128.5 / 127.4 / 126.7 | 106.0 / 104.2 / 104.2 |
| **VAL** | U-Net `p_off` | 0.893 | **0.973** | **0.945** | 25.0 / 14.7 / 23.1 | 28.9 / **21.5** / 29.6 | 123.9 / 126.1 / 126.7 | 103.3 / 103.3 / 105.0 |
| **VAL** | GLU `p_on` | **0.962** | 0.967 | 0.916 | **18.9** / 17.4 / 30.0 | 29.8 / 30.2 / 41.1 | 146.8 / 144.8 / 143.7 | 122.7 / 120.7 / 120.9 |
| **VAL** | GLU `p_off` | 0.903 | 0.967 | 0.927 | 26.5 / 18.0 / 27.4 | 37.6 / 34.8 / 41.6 | 138.0 / 137.3 / 138.1 | 114.6 / 112.8 / 114.6 |
| **VAL** | GappyPOD `p_on` | 0.933 | 0.949 | 0.929 | 37.6 / 34.2 / 48.2 | 54.7 / 51.4 / 62.0 | 204.5 / 205.1 / 206.4 | 177.5 / 175.1 / 177.0 |
| **VAL** | T-only costante | — | — | 0.482 | — / — / 71.3 | — / — / 129.8 | — / — / 1167.6 | — / — / 838.1 |
| **seen** | U-Net `p_on` | **0.766** | 0.739 | 0.707 | **23.5** / 25.1 / 31.4 | 57.1 / 59.9 / 63.6 | 132.6 / 130.7 / 131.7 | 89.9 / 88.1 / 90.2 |
| **seen** | U-Net `p_off` | 0.707 | **0.753** | 0.733 | 26.4 / **24.4** / 28.6 | 55.0 / **53.9** / 55.3 | 130.4 / **127.5** / 128.9 | 90.5 / **87.0** / 88.9 |
| **seen** | GLU `p_on` | 0.759 | 0.746 | 0.703 | 28.5 / 28.9 / 35.2 | 70.8 / 70.6 / 74.3 | 137.1 / 136.3 / 137.4 | 94.9 / 94.3 / 96.7 |
| **seen** | GLU `p_off` | 0.721 | 0.725 | 0.693 | 30.6 / 31.6 / 36.9 | 78.4 / 79.8 / 81.6 | 145.8 / 143.3 / 144.4 | 101.0 / 99.2 / 101.6 |

### 3.3 Lettura critica delle sei figure (frame 1000 vs intero insieme)
Le sei figure (`figures/val_*_frame1000_rep0_{fields,signed_error}.png`) usano scale comuni (`50–2750 K` per i campi, `±700 K` per `T_pred − T_true`), contorno `T = 2100 K` e cerchi sui 9 siti `T9` (tutti i 36 pannelli verificati contro il CSV entro `5e-7 K` in `figure_panel_check.csv`). L'ispezione visiva suggerisce un possibile meccanismo; i pannelli di un solo frame non ne quantificano il contributo sull'intera traiettoria:

1. **Posizione critica dei siti `(row=32, col=95)` e `(row=95, col=95)`:** in tutte e tre le condizioni VAL, i due siti a destra della griglia $3\times 3$ cadono **sul bordo interno o dentro i due rami del getto freddo (V-jet)**, esattamente nella fascia ad alto gradiente dove i modelli neurali accumulano errori locali di `±200–600 K` (macchie rosse/blu intense nei pannelli `signed_error`). Gli altri 7 siti giacciono invece nella regione calda di prodotti dove l'errore locale è piccolo e uniforme (`~10–30 K`).
2. **`40NH3_0.9` frame 1000 — conferma della soglia e dimostrazione dell'outlier AO9:**
   - Su `U-Net p_on` (riga 1), in `free` il campo caldo è spostato sotto `2100 K` (`bias -25.5 K`), frammentando il contorno bianco in tutta la camera (`RMSE 37.8 K`). Con `oracle` e `AO9` (`+25.5 K` e `+23.2 K`), il contorno `2100 K` torna quasi identico a `truth` (`RMSE 27.9 / 28.0 K`), coerente col recupero sull'intera condizione (`IoU 0.310 -> 0.950 oracle / 0.889 AO9`).
   - Su `U-Net p_off` (riga 2), lo stesso frame 1000 mostra però la sensibilità della media semplice a residui locali grandi: il sito in basso a destra `(95, 95)` colpisce una macchia blu (`T_pred - T_true ≈ -400 K`) sul bordo del getto inferiore. La mappa suggerisce un contributo rilevante di quel sito all'offset complessivo; non è stata pubblicata una decomposizione per ancora. Nel frame `delta_ao9` a `+55.2 K` contro `delta_oracle = +15.9 K`, facendo esplodere il bias a `+39.4 K` e la RMSE da `31.0 K` a `47.5 K` (sfondo rosa uniforme su tutta la camera!).
3. **`20NH3_1.1` — divario netto tra frame 1000 e media dei 41 frame:**
   - Al frame 1000, `U-Net p_on` ha `dOr = -6.22 K` e `dAO9 = -5.97 K` (`RMSE 32.2 -> 31.6 K`), e `GLU p_on` passa da `49.0` a `46.0 K`. Presa da sola, la figura del frame 1000 farebbe apparire `AO9` efficace su `20NH3_1.1`.
   - Sull'intero insieme di 41 frame, invece, le tabelle mostrano una sovracorrezione media negativa. Un contributo dei residui nei siti prossimi ai getti è plausibile, ma richiede una verifica per ancora sull'intero insieme. I valori aggregati di `dAO9` sono `-28.39 K` (contro `dOr = -6.32 K`) per U-Net `p_on` e `-41.60 K` (contro `-15.79 K`) per GLU `p_on`, degradando la RMSE macro di `+8.40` e `+8.88 K` e facendo crollare l'IoU calda di GLU `p_on` da `0.982` a `0.879`.
4. **`0NH3_1.05` — errore concentrato nei getti:** le patch rosse fino a `+500–600 K` nel getto superiore e blu/rosse nel getto inferiore mantengono errori spaziali localizzati dopo le traslazioni (`Jet RMSE ≈ 168–190 K`, con variazioni tra trattamenti). La centered-RMSE resta invariata; la RMSE regionale può cambiare.
5. **Nota su `seen` (oracle può ridurre leggermente Hot IoU):** su `seen`, `oracle` riduce la RMSE globale ma abbassa leggermente l'IoU calda di `U-Net p_on` (`0.766 -> 0.739`) e `GLU p_on` (`0.759 -> 0.746`). Questo accade perché `oracle` minimizza la MSE sull'intero campo (inclusi getti freddi), non la soglia binaria `2100 K`: un piccolo bias positivo nei prodotti poteva compensare localmente una sottostima vicino alla soglia. È un promemoria importante che RMSE globale e metrica di soglia non sono sempre monotone tra loro.

---

## 4. Contributo incrementale delle reti rispetto alle baseline (a informazione comparabile)

Tenendo `oracle` rigorosamente separato dai metodi utilizzabili con sole osservazioni:

- **Senza `p` (`p_off`), solo velocità 5% (`free`):**
  - VAL: U-Net `43.53 K` (IoU `0.893`), GLU `47.54 K` (`0.903`) vs `GappyPOD p_off` `94.80 K` (`0.848`) e `GlobalMean` `139.70 K` (`0.808`).
  - Seen: U-Net `55.44 K`, GLU `60.31 K` vs `GappyPOD p_off` `113.08 K` e `GlobalMean` `173.68 K`.
- **Senza `p` (`p_off`) + 9 T ideali (`AO9`):**
  - VAL: U-Net `42.86 K` (`0.945`), GLU `48.20 K` (`0.927`) vs `GappyPOD p_off + AO9` `79.95 K` (`0.877`), `GlobalMean + AO9` `122.01 K` (`0.890`) e `T_only_const` `311.79 K` (`0.482`).
  - Seen: U-Net `53.78 K`, GLU `61.83 K` vs `GappyPOD p_off + AO9` `111.88 K`, `GlobalMean + AO9` `163.21 K`, `T_only_const` `323.84 K`.
- **Con `p` (`p_on`) + 9 T ideali (`AO9`):** *(Nota: `ParamMean` riceve `p`, quindi appartiene a questo gruppo e NON va confrontato col divieto `p` di `p_off`.)*
  - VAL: U-Net `43.14 K` (`0.940`), GLU `51.12 K` (`0.916`) vs `GappyPOD p_on + AO9` `76.24 K` (`0.929`) e `ParamMean + AO9` `94.30 K` (`0.925`).
  - Seen: U-Net `53.81 K`, GLU `58.48 K` vs `GappyPOD p_on + AO9` `115.97 K` e `ParamMean + AO9` `139.54 K`.
- **Riferimento diagnostico `oracle` (non deployable):** anche eliminando completamente l'errore di livello da tutti i metodi, il divario strutturale delle reti resta enorme: su VAL `38.0–44.6 K` contro `68.2–71.7 K` (`GappyPOD`), `84.8 K` (`ParamMean`) e `114.8 K` (`GlobalMean`); su `seen` `50.6–58.5 K` contro `103.5–106.6 K` (`GappyPOD`).

**Conclusione sul contributo incrementale:** l'aggiunta delle 9 T non riduce il valore della ricostruzione neurale da velocità. `T_only_const` ha errori molto maggiori in questo confronto (`Jet RMSE > 1040 K`), e anche `ParamMean+AO9` / `GappyPOD+AO9` restano circa **1.8×–2.6× peggiori** delle reti sulla RMSE globale e **1.5×–2.5× peggiori** nei getti e nelle regioni di picco.

---

## 5. Effetto su U-Net vs GLU, `p_on` vs `p_off` e differenze tra condizioni

Fonti: `tables/paired_diff_macro_long.csv`, `tables/per_condition_metrics.csv`.

### 5.1 U-Net contro GLU
- **In `free`:** su VAL, U-Net aveva un vantaggio modesto in RMSE (`-2.84 K` `p_on`, 6/9 celle; `-4.01 K` `p_off`, 9/9 celle), ma soffriva un'IoU calda macro nettamente peggiore con `p_on` (`0.759` vs `0.962`) a causa del solo `40NH3_0.9`.
- **Dopo `oracle`:** il vantaggio apparente di GLU nelle regioni calde **scompare del tutto**. U-Net batte GLU in **9/9 celle VAL** (`-6.24 K` `p_on`, `-4.72 K` `p_off`), ottiene `Hot RMSE` migliore (`14.3` vs `17.4 K` `p_on`), `Jet RMSE` migliore (`127.4` vs `144.8 K`), e `Hot IoU` pari o superiore (`0.973` vs `0.967` macro; su `40NH3_0.9`: `0.950` vs `0.936`). Anche su `seen`, U-Net batte GLU in `43/60` celle (`p_on`, `-4.48 K`) e `58/60` celle (`p_off`, `-7.34 K`).
- **Dopo `AO9`:** il divario a favore di U-Net si amplia ulteriormente su VAL (`-7.98 K` `p_on`, `-5.35 K` `p_off`, 9/9 celle; IoU `0.940` vs `0.916` con `p_on`), perché GLU soffre errori locali più grandi nei getti e quindi un bias `AO9` peggiore.
- **Cautela:** questa è evidenza su **un solo seed (`seed 0`)**, una sola densità di training (5%) e un adattamento deterministico specifico di GLU; non autorizza a proclamare U-Net vincitore universale prima di verificare repliche di seed e comportamento multi-densità.

### 5.2 `p_on` contro `p_off`
- **In `free`:** fornire `p` (NH₃, $\phi$) aiuta soprattutto il **livello termico**, riducendo `|Bias|` su VAL da `17.52` a `12.88 K` (U-Net), da `18.58` a `8.77 K` (GLU) e da `57.58` a `15.38 K` (GappyPOD).
- **Dopo `oracle`:** su VAL il vantaggio di `p` **svanisce o si inverte leggermente** per le reti: `U-Net p_on − p_off = +0.36 K` (`centered`: `38.36` vs `38.00 K`), `GLU p_on − p_off = +1.89 K` (`44.61` vs `42.72 K`). Su `seen`, il guadagno strutturale di `p` è quasi nullo per U-Net (`-0.56 K`) e moderato per GLU (`-3.42 K`).
- **Dopo `AO9`:** su VAL, `p_off + AO9` eguaglia o supera `p_on + AO9` (`U-Net`: `42.86` vs `43.14 K`, IoU `0.945` vs `0.940`; `GLU`: `48.20` vs `51.12 K`).
- **Lettura:** nei checkpoint S5 esaminati, il vantaggio di p è più evidente nel livello che nella centered-RMSE, soprattutto su VAL. Questo confronto non identifica da solo quali informazioni determinino la geometria né dimostra un ruolo generale di p.

### 5.3 Eterogeneità tra condizioni
Le tre condizioni VAL si comportano in modo molto diverso:
- `40NH3_0.9` è **dominata dal livello** (`centered 26.8–32.2 K`, area calda vicino a `2100 K`): qui un offset di `25–30 K` ribalta completamente l'IoU calda.
- `0NH3_1.05` e `20NH3_1.1` sono **dominate dalla struttura dei getti** (`centered 40.3–53.0 K`, `Jet RMSE 125–190 K`): qui l'offset oracolo recupera poco (`0.14–4.51 K`), mentre `AO9` con media semplice peggiora nettamente la ricostruzione con un comportamento compatibile con una stima dell'offset influenzata da grandi residui locali.

---

## 6. Risposte alle cinque domande di `roadmapastra.md` §2

### Domanda 1 — Quanto errore dipende da una traslazione uniforme del livello?
- **Evidenza quantitativa:** `oracle − free` riduce la RMSE macro su VAL di `-4.60 K` (`U-Net p_on`), `-5.52 K` (`U-Net p_off`), `-1.19 K` (`GLU p_on`), `-4.82 K` (`GLU p_off`), contro `-23.11 K` (`GappyPOD p_off`) e `-24.86 K` (`GlobalMean`). Su `seen`, il recupero neurale è `-0.42`–`-4.27 K`. Su `40NH3_0.9`, invece, per `U-Net p_on` l'offset vale `-13.12 K` di RMSE e `+0.640` di IoU calda (`0.310 -> 0.950`).
- **Interpretazione sostenuta:** la centered-RMSE resta vicina alla RMSE free nelle macro neurali, e i getti presentano errori regionali elevati. Il rapporto tra queste macro non è una decomposizione percentuale dell'errore. Tuttavia, vicino alla soglia calda (`2100 K`), anche un errore di livello di `20–30 K` produce un fallimento catastrofico delle regioni calde pur con struttura spaziale molto buona.
- **Ciò che resta incerto:** quanto il bias di livello su `40NH3_0.9` sia sistematico rispetto a fluttuazioni del singolo `seed 0`, e quanto pesi in condizioni di vera estrapolazione fuori inviluppo (`TEST` o fold di estrapolazione).

### Domanda 2 — Nove T stimano abbastanza bene quella traslazione?
- **Evidenza quantitativa:** **No, con la ricetta congelata (`[32,64,95]`, media con $\tau=1$).** Su VAL, `AO9 − oracle` resta `+4.77`–`+6.51 K` sui neurali (`|dAO9 − dOr| = 14.7–19.1 K`), e rispetto a `free` `AO9` peggiora 3 reti su 4 su VAL e 3 reti su 4 su `seen` (`0/20` miglioramenti su `GLU p_on`). Funziona bene solo dove il bias `free` è grande (`40NH3_0.9` per U-Net `p_on`: `-10.77 K`, IoU `0.310 -> 0.889`; `GappyPOD p_off`: `-14.85 K`).
- **Interpretazione sostenuta:** il fallimento non indica che 9 misure termiche siano intrinsecamente inutili, ma che la stima da nove residui può essere poco rappresentativa dell'offset globale. Le mappe suggeriscono un possibile ruolo dei siti prossimi ai getti; manca una quantificazione per ancora su tutti i frame.
- **Ciò che resta incerto:** quanto migliorerebbe la stima dell'offset usando uno stimatore robusto (es. mediana/Huber) sugli stessi 9 siti, oppure evitando la fascia di ingresso dei getti, oppure aggiungendo rumore realistico di misura.

### Domanda 3 — Il vantaggio GLU nelle regioni calde sopravvive alla correzione?
- **Evidenza quantitativa:** **No.** Dopo `oracle`, `U-Net` supera `GLU` sia in RMSE (9/9 celle VAL, `-6.24 K` `p_on`, `-4.72 K` `p_off`) sia in `Hot RMSE` (`14.3` vs `17.4 K`), sia in `Hot IoU` (`0.973` vs `0.967` macro; `0.950` vs `0.936` su `40NH3_0.9`). Lo stesso avviene dopo `AO9` (`IoU 0.940–0.945` vs `0.916–0.927`).
- **Interpretazione sostenuta:** il vantaggio di GLU `p_on` su `40NH3_0.9` in `free` (`IoU 0.925` vs `0.310`) derivava dal miglior livello medio del checkpoint `seed 0` su quella condizione, non da una migliore ricostruzione spaziale della fiamma.
- **Ciò che resta incerto:** se questo vantaggio strutturale di U-Net si mantenga cambiando `seed`, cambiando densità delle velocità (1% / 20% / training multidensità), o con un GLU riadattato nella capacità latente (`d_model`).

### Domanda 4 — `p` serve ancora con poche T?
- **Evidenza quantitativa:** con `oracle` su VAL, `p_on` non migliora `p_off` (`U-Net +0.36 K`, `GLU +1.89 K`, IoU identica). Con `AO9` su VAL, `p_off` eguaglia o batte `p_on` (`U-Net 42.86` vs `43.14 K`; `GLU 48.20` vs `51.12 K`). Su `seen` con `oracle`, `p` dà `-0.56 K` su U-Net e `-3.42 K` su GLU.
- **Interpretazione sostenuta:** in questo setup, nei checkpoint S5 esaminati il vantaggio di p si riduce o si inverte su VAL dopo oracle, mentre persiste un vantaggio centered su seen, soprattutto per GLU. È una descrizione di questi run, non una conclusione generale sul ruolo di p.
- **Ciò che resta incerto:** se `p` torni decisivo in estrapolazione severa (miscele con $\text{NH}_3$ o $\phi$ fuori intervallo) o a coperture di velocità molto basse (es. 1%), dove le sole velocità potrebbero non identificare il regime di fiamma.

### Domanda 5 — La rete resta utile quando aggiungiamo T?
- **Evidenza quantitativa:** **Sì, con ampio margine.** A informazione comparabile con `AO9` su VAL, `U-Net` (`42.86–43.14 K`) supera `GappyPOD` (`76.24–79.95 K`), `ParamMean` (`94.30 K`, solo con `p`), `GlobalMean` (`122.01 K`) e `T_only_const` (`311.79 K`). Su `seen` il divario è ancora maggiore (`53.8 K` vs `111.9–116.0 K` GappyPOD vs `323.8 K` T-only).
- **Interpretazione sostenuta:** con la correzione scalare studiata, le T cambiano il livello ma non la centered-RMSE. Le reti da velocità sparse superano le baseline considerate; non è dimostrata l'indispensabilità delle reti o della non linearità, né esclusa l'utilità di altri metodi T-only.
- **Ciò che resta incerto:** il confronto contro un eventuale interpolatore spaziale T-only (non nel perimetro A2) e la resa delle reti quando le velocità hanno rumore o geometria PIV realistica.

---

## 7. Limiti che incidono sulle decisioni

1. **Un solo seed (`seed 0`) e 3 condizioni `VAL` usate per selezione/early stop S5:** tutte le cifre VAL sono evidenze di sviluppo, non validazione indipendente. Le differenze di `2–6 K` tra modelli o tra `p_on`/`p_off` non hanno ancora stima di variabilità tra training.
2. **Un solo layout `T9` e un solo stimatore (`AO9` media con $\tau=1$):** una possibile spiegazione del fallimento medio di `AO9` riguarda l'intersezione tra la griglia `[32,64,95]` e i getti freddi combinata con una media non robusta. Non va letto come impossibilità fisica dell'approccio assisted.
3. **Osservazioni numeriche ideali (rumore 0):** sia i guadagni su `40NH3_0.9` sia i limiti rilevati valgono per misure puntuali ideali, non per termocoppie reali.
4. **Tracciabilità secondaria (`seen rep 1/2` solo fonte A; `mask_hash` assente nelle baseline):** documentata in §1.2; non impedisce l'analisi nel perimetro dichiarato; resta un limite della verifica degli input storici.

---

## 8. Prossimo passo scientifico minimo e proposta per A4

### 8.1 Cosa può procedere subito (zero training, basso costo)
Non è necessaria alcuna rilavorazione da parte di Sonnet sui risultati A3 appena consegnati. Prima di decidere se investire in metodi assisted o scartare `AO9`, un possibile controllo diagnostico successivo (previa verifica della disponibilità dei nove residui per tutti i frame e autorizzazione del coordinatore) è verificare **se uno stimatore robusto dell'offset (mediana o media troncata / Huber) sugli stessi 9 siti** elimina il bias negativo sistematico su `20NH3_1.1`, `0NH3_1.05` e `seen`. Sarebbe un'estensione esplorativa suggerita dai risultati, da congelare prima dell'esecuzione; potrebbe aiutare a distinguere *"9 T sono insufficienti"* e *"la media semplice è corrotta dai 2 siti nel getto"*.

### 8.2 Cosa deve attendere Opus C149 — Proposta motivata per A4 (non eseguita)
L'estensione **A4** va decisa dopo la chiusura dei due run Opus `C149` (attualmente in corso su `mission/dense-unet`). La domanda di collegamento è precisa: *aumentare le istantanee TRAIN da 38 a 149 riduce soprattutto l'errore strutturale nei getti (`centered-RMSE`), riduce anche il bias di livello (come `40NH3_0.9`), oppure i due effetti sono indipendenti?*

**Matrice A4 proposta (solo valutazione offline, zero training):**
- **Blocco primario dati:** `O-C149-on-s0` e `O-C149-off-s0` (`best.pt` @5%), confrontati in modo appaiato con `REF38` (`s5_sparse_unet_pon/poff_seed0`), sugli stessi ruoli (`seen220`, `VAL123`), 3 repliche velocità 5% e stessi trattamenti (`free`, `oracle`, `AO9`).
- **Blocco sensibilità seed:** `O-A38-on-s1` e `O-A38-off-s1` (`best.pt` @5%), per verificare se il bias `-29.35 K` su `40NH3_0.9` è sistematico o dipendente dal `seed 0`.
- **Vincolo:** nessuna selezione post-hoc tra checkpoint Opus; tabelle A4 tenute distinte dal blocco primario S5.

### 8.3 Eventuale incarico tecnico opzionale per Sonnet (solo se/quando A4 verrà autorizzata)
Il pacchetto `86a925a` **non richiede correzioni bloccanti**. Se e quando A4 sarà autorizzata, due piccoli miglioramenti igienici in `scripts/18_astra_level_structure_eval.py` (non retroattivi) miglioreranno la tracciabilità futura: (i) registrare `mask_hash` anche nelle righe baseline; (ii) puntare il replay inline neurale ai file `per_frame_best_cov05_*` invece di `summary_all_settings.csv`.


## 9. Decisione del coordinatore — 19 novembre 2026

Il blocco primario S5 A0–A3 è accettato come concluso nel perimetro dichiarato. A4 e gli stimatori robusti restano estensioni non eseguite e non necessarie alla chiusura del blocco primario. Si attende il completamento e l'analisi Opus; seguiranno sintesi con Enrico, riesame delle proposte Neon già documentate e decisione su S7. Solo dopo gli impegni concordati si chiuderanno sparse e gli aperti del Mese 2. Questa revisione del coordinatore riguarda interpretazione e stato, senza ricalcolo delle tabelle o nuova verifica dei job.
