# Report ASTRA (second pass) — Diagnostica `median9` e `mean9` sull'insieme completo S5 + baseline

**Stato:** Estensione esplorativa **ASTRA `median9` second pass** conclusa nel perimetro dichiarato. Non sostituisce il report primario A0–A3 ([07_astra_level_structure_and_anchors.md](07_astra_level_structure_and_anchors.md)), non è una conferma indipendente, non avvia A4, le proposte Neon né la decisione S7 (che restano rinviate alla sintesi Astra–Opus dopo la chiusura di Opus).  
**Repository:** `EnricoBontempi/official_thesis`  
**Branch esecutore / commit aggregazione analizzato:** `astra/median9-second-pass`, SHA `04e3b68addc891034f0840c2cbaa141d5ef8c4ac`  
**Commit codice inferenza:** `44e5e5e0bd0907560d1736a53d357933680b090a` (job Engaging riferiti: `29686178_0..3`, `29686179`)  
**Verifica report primario:** confermato che `82bff35726fb70af9987081802ece3ef718e42bd` (`review/astra-07-report`) è antenato di `04e3b68` e che `docs/month2/07_astra_level_structure_and_anchors.md` coincide byte per byte con la copia revisionata dal coordinatore (`git diff` = 0 righe).  
**Fonti ispezionate direttamente in Git:** `docs/month2/astra_median9_full_set_handoff.md`, `docs/month2/astra_median9_second_pass_handoff.md`, `results/month2/astra_median9_second_pass/full_set/INDEX.md`, tutte le 15 tabelle CSV e `aggregation_meta.json` in `results/month2/astra_median9_second_pass/full_set/tables/`, `results/month2/astra_level_structure/tables/`, `configs/experiments/month2_astra_median9_second_pass.yaml`, `src/combustion_recon/evaluation/astra_level.py`, `scripts/20_astra_a3_aggregate.py`, `scripts/24_astra_median9_full_eval.py`, `scripts/25_astra_median9_aggregate.py`, `tests/test_astra_median9_aggregate.py`.

> **Avvertenza sul perimetro e sui dati verificati:**  
> 1. **Natura esplorativa:** questa diagnostica nasce dai risultati `VAL` già osservati nel blocco primario A0–A3; gli stessi tre casi `VAL` avevano guidato selezione del checkpoint, scheduler e arresto in S5. Le tre repliche sono **maschere di velocità (5%)**, non tre seed di training (unico `seed 0`). Le nove temperature `T9` restano campioni numerici ideali (rumore 0, nessun ritardo dinamico o intrusività di termocoppie reali).  
> 2. **Separazione tra verificato e riferito:** tutte le verifiche sulle tabelle aggregate, sui ricalcoli macro/appaiati, sul replay contro le tabelle primarie in Git e sul codice sorgente sono state eseguite direttamente su `04e3b68`. I CSV grezzi remoti su Engaging (`/home/bontempi/official_thesis_astra_results/median9_second_pass/`), i log Slurm (`29686178_0..3`, `29686179`), i file `GATE.json`/`COMPLETE.json`/`provenance.json` e i controlli per-riga sui `mask_hash` (1029 righe) non sono presenti in Git e restano quindi **documentati dall'esecutore (Cursor)** e riassunti in `aggregation_meta.json`.  
> 3. **Separazione da `example_frame1000_rep0`:** i 12 campi esemplificativi (`results/month2/astra_median9_second_pass/example_frame1000_rep0/`) restano un oggetto separato e non sono usati per trarre conclusioni sull'insieme completo.

---

## Sintesi esecutiva

1. **Il cambio di stimatore da media (`AO9`/`mean9`) a mediana (`median9`) ribalta il fallimento della correzione di livello osservato in A0–A3.** Sugli stessi pesi, frame, maschere e 9 siti prefissati (`[32,64,95] × [32,64,95]`), `median9` abbatte l'errore assoluto di stima dell'offset (`|bias| = |delta_applied − delta_oracle|`) su **tutti gli 8 modelli** sia su `VAL` (neurali: da `14.66–19.14 K` con `AO9` a **`3.11–5.09 K`** con `median9`) sia su `seen` (neurali: da `12.76–15.65 K` a **`5.51–7.50 K`**).
2. **`median9` non si limita a riparare il danno di `AO9`: migliora nettamente anche rispetto a `free` e si porta a pochi decimi di kelvin dall'`oracle`.** Su `VAL`, migliora la RMSE rispetto a `free` su **8 modelli su 8** (neurali: `-0.92` a `-5.03 K`, distanza residua da `oracle` scesa da `+4.77`–`+6.51 K` ad appena **`+0.21`–`+0.49 K`**; su `40NH3_0.9`, `U-Net p_on` passa da `39.92 K` / IoU `0.310` a **`26.93 K` / IoU `0.949`**, contro `26.80 K` / `0.950` dell'`oracle`). Su `seen`, migliora `free` su **7 modelli su 8** (`-0.58` a `-3.66 K` sui neurali; `-0.55` a `-19.21 K` sulle baseline); l'unica eccezione è `GLU p_on` (`+0.05 K`, `55.51 -> 55.56 K`), dove `free` aveva già `|bias| = 5.59 K` e margine oracolo minimo (`-0.42 K`).
3. **Il guadagno dipende interamente dallo stimatore (mediana vs media), NON dalla rimozione dello shrink.** Togliere solo lo shrink (`mean9 − AO9`) **peggiora sistematicamente 8 modelli su 8** su entrambi i ruoli (`+0.94` a `+1.46 K` neurali VAL, `0/9` celle migliorate; `+0.59` a `+0.71 K` neurali seen, `0–1/60` celle migliorate; `+1.55` a `+3.48 K` sulle baseline). Di conseguenza, `median9 − mean9` è più favorevole di `median9 − AO9` ovunque (`-5.31` a `-7.69 K` neurali VAL).
4. **I 9 residui per ancora dimostrano il meccanismo locale e chiariscono il compromesso fisico.** Su `VAL`, i due siti sui bordi del getto freddo — `(32,95)` e `(95,95)` — presentano residui medi assoluti di `72–131 K` sui neurali (`151–323 K` sulle baseline) contro `9–22 K` sugli altri 7 siti (incluso `(64,95)`), generando il forte bias negativo di `AO9` su `20NH3_1.1` (dove `(95,95)` ha residuo firmato medio tra `-92` e `-229 K`). Tuttavia quei residui non sono «rumore di sensore»: sono errori reali del modello nei getti freddi (`6.8%` dell'area VAL, ma `2/9 = 22.2%` delle ancore). La mediana ignora i siti nei getti e stima l'offset della regione calda maggioritaria (`~79–91%` dell'area VAL): ciò spiega perché `median9` ottenga persino **MAE leggermente migliore dell'`oracle`** su tutti gli 8 modelli, ma lasci invariato o lievemente peggiore `Jet RMSE` (`127–146 K` VAL).
5. **Il contributo delle reti rispetto alle baseline resta invariato anche dopo `median9`.** Sebbene le baseline guadagnino molto in termini assoluti da `median9` (avvicinandosi a `+0.43`–`+1.25 K` dal rispettivo `oracle` su VAL), il divario strutturale resta invariato: `U-Net + median9` (`38.49–38.58 K` VAL; `51.10–51.79 K` seen) rimane circa **1.8×–2.2× migliore** di `GappyPOD + median9` (`68.63–72.77 K` VAL; `105.50–107.81 K` seen) e `ParamMean + median9` (`85.53 K` VAL; `128.70 K` seen), e ~8× migliore di `T_only_const` (`311.79 K` VAL).
6. **Audit tecnico: nessun errore bloccante; rilevata un'imprecisione non bloccante nel tag `col95`.** Tutti i controlli su aggregazione, segni, invarianze e replay contro A0–A3 passano a precisione numerica (`0.000 K` su `free`/`oracle`/`ao9`). Si segnala un'imprecisione di etichettatura/filtro in `scripts/25_astra_median9_aggregate.py` (riga 130): `anchor_col == 95` aggrega **tre** ancore (`(32,95)`, `(64,95)`, `(95,95)`), mentre commento, `INDEX.md` e handoff dichiarano «le **due** ancore di colonna 95». L'anomalia riguarda solo `anchor_col95_share_macro_long.csv` e non altera nessuna metrica, offset o tabella per-ancora; l'estensione può essere chiusa nel perimetro svolto.

---

## 1. Audit tecnico: contratto, aggregazioni, replay e due rilievi minori

Sono stati verificati direttamente codice e tabelle in `04e3b68addc891034f0840c2cbaa141d5ef8c4ac` (`src/combustion_recon/evaluation/astra_level.py`, `scripts/24_astra_median9_full_eval.py`, `scripts/25_astra_median9_aggregate.py`, `results/month2/astra_median9_second_pass/full_set/tables/`).

### 1.1 Verifiche eseguite direttamente sugli artefatti in Git
- **Formule implementate (`astra_level.py`):**
  - `resid = T_obs - T_pred` sui 9 siti prefissati `r*128 + c` con `r, c ∈ {32, 64, 95}` (indici ordinati `anchor_id = 0..8`).
  - `delta_oracle = mean_grid(T_true - T_pred)`, `delta_ao9 = sum(resid)/(9+1)`, `delta_mean9 = sum(resid)/9`, `delta_median9 = sorted(resid)[4]` (quinto valore ordinato, senza shrink né clipping). Tutti i 5 trattamenti derivano dalla stessa predizione base in kelvin (`level_recipe = same_base_prediction`). `T_only_const` resta campo costante uguale alla media delle 9 `T`, non riceve `mean9`/`median9`, ha tutti gli offset a `NaN` ed è isolato in `t_only_*.csv`.
- **Coerenza dell'aggregazione e completezza:**
  - Definizione macro confermata identica ad A3 (`scripts/20`): media per frame $\to$ media sui frame della condizione (`sim_id`: `11` frame su tutte le `20` condizioni `seen`, `41` frame su tutte le `3` condizioni `VAL`) $\to$ media uniforme tra condizioni per ciascuna replica (`0, 1, 2`) $\to$ media aritmetica delle 3 repliche (`replica='mean'`, con `rep_min`/`rep_max`). `seen` e `VAL` non sono mai mescolati.
  - Ricalcolo indipendente da `per_condition_metrics.csv` verso `macro_headline_wide.csv`: scarto massimo assoluto `5.03e-7 K` (arrotondamento di stampa CSV a 6 decimali).
  - Coerenza delle differenze appaiate (`paired_diff_macro_long.csv` vs `wide(A) − wide(B)` su 1600 celle): scarto massimo `9.80e-7 K`.
  - Completezza regionale: `0` regioni vuote; `__n_valid_frames` vale esattamente `11` su tutte le righe `seen` e `41` su tutte le righe `VAL` per tutte le metriche regionali (`hot`, `peak`, `jet`, `nonhot`, `hot_iou`).
  - Conteggi di segno (`paired_frame_signs.csv`): `n_negative + n_positive + n_zero == n_frames` (`660` su `seen`, `369` su `VAL` per modello sommando le 3 repliche) verificato ovunque senza eccezioni. Coincidono i totali aggregati citati nell'handoff (`3804-/1476+` su `seen`, `2299-/653+` su `VAL` per `median9−ao9`), che non vanno però letti come classifica perché sommano metodi eterogenei.
- **Replay contro il pacchetto primario A0–A3 (`results/month2/astra_level_structure/tables/`):**
  - Confronto diretto su tutte le `48` righe macro headline e tutte le `1656` righe `per_condition_metrics.csv` dei trattamenti primari `{free, oracle, ao9}`: scarto massimo assoluto **`0.000e+00 K`**.
  - Confronto diretto di `T_only_const` su `20` celle headline (`seen` e `VAL`): scarto massimo `4.80e-7 K` (formato stampa).
- **Identità matematiche e convenzioni di segno verificate:**
  - `all_centered_rmse_K` e `off_centered_rmse_K` invariate tra tutti e 5 i trattamenti (`0.000e+00 K` nelle tabelle headline; scarto per-frame `4.07e-15 K` registrato in `aggregation_meta.json`).
  - Per `oracle`: `|all_rmse_K − all_centered_rmse_K| = 0`, `|all_bias_K| = 0`, `|all_abs_bias_K| = 0`.
  - `all_bias_K == delta_applied_minus_oracle_K` (scarto max `5.0e-7 K`), `all_abs_bias_K == abs_offset_error_K` (scarto max `4.9e-7 K`), `delta_ao9_K == 0.9 * delta_mean9_K` (scarto max `6.0e-9 K`).
  - Da `anchor_residual_per_condition.csv`: `mean(9 resid_K) == delta_mean9_K` (scarto max `4.4e-8 K`) e `sum(9 ao9_contribution_K) == delta_ao9_K` (scarto max `4.7e-8 K`).
  - Verifica per-frame dell'equivalenza di segno: per due traslazioni pure dello stesso frame (`RMSE² = centered² + bias²`), `all_rmse_K(A) < all_rmse_K(B)` equivale esattamente a `all_abs_bias_K(A) < all_abs_bias_K(B)`; infatti i conteggi `n_negative`/`n_positive` in `paired_frame_signs.csv` coincidono identicamente fra `all_rmse_K` e `all_abs_bias_K` su tutti i modelli.

### 1.2 Informazioni riferite dall'esecutore (non verificate direttamente sui raw remoti)
Restano documentati dall'esecutore (`astra_median9_full_set_handoff.md`, `astra_median9_second_pass_handoff.md`, `aggregation_meta.json`) senza accesso diretto al filesystem Engaging: esiti Slurm `COMPLETED 0:0` dei job `29686178_0..3` (1 min 58 s – 2 min 07 s) e `29686179` (29 s), stderr vuoto, presenza dei file `GATE.json`/`COMPLETE.json`/`provenance.json` con `scientific_validated=true`, assenza di chiavi duplicate e presenza di `mask_hash` verificato su tutte le 1029 istanze dei CSV remoti (`3300/1845` righe per run neurale, `13860/7749` per baseline), nonché l'assenza di nuove mappe complete salvate. Il codice in `scripts/24_astra_median9_full_eval.py` impone tuttavia a runtime i controlli SHA256 di checkpoint, cache, stats, artefatto Gappy e tutte le 1029 righe di maschere prima di scrivere `COMPLETE.json`, e il match esatto `0.000 K` delle tabelle in Git con il pacchetto primario A0–A3 conferma la piena riproduzione dei tre trattamenti base.

### 1.3 Rilievi tecnici minori (non bloccanti)
1. **Imprecisione nel filtro/etichetta `col95` (`scripts/25_astra_median9_aggregate.py` L130 e L135; `INDEX.md` L20; `astra_median9_full_set_handoff.md` L38):**  
   Il codice definisce `site_tag = np.where(resid["anchor_col"] == 95, "col95", "other")`. Nella griglia $3\times 3$ `[32,64,95] × [32,64,95]` esistono **tre** ancore con colonna 95: `(32,95)`, `(64,95)` e `(95,95)`, e tutte e tre ricevono `site_tag="col95"`. Tuttavia il commento del codice, `INDEX.md` e l'handoff descrivono `anchor_col95_share_macro_long.csv` come la quota proveniente dalle *«due ancore di colonna 95»* (alludendo ai soli due siti sui bordi del getto `(32,95)` e `(95,95)`). Come verificato in §4 da `anchor_residual_per_condition.csv`, l'ancora centrale `(64,95)` ha residui piccoli come i 6 siti interni (`~2–6%` della somma macro dei `|resid|`), quindi la quota per-frame in `anchor_col95_share_macro_long.csv` (`0.55–0.80`) include anche quella piccola terza quota. **Non bloccante:** non tocca nessuna tabella di metriche, offset, differenze appaiate o residui per ancora.
2. **Colonne parzialmente `NaN` in `per_condition_metrics.csv`:**  
   `delta_mean9_minus_oracle_K` e `delta_median9_minus_oracle_K` presentano `1656` `NaN` (esattamente sulle righe `free`, `oracle`, `ao9`, ovvero `3/5` di `2760`), perché `prediction_tables()` in `astra_level.py` le valorizza solo sulle righe `mean9`/`median9`. **Non bloccante:** le tabelle `offset_error_per_condition.csv` e `offset_error_macro_long.csv` usano `delta_applied_minus_oracle_K` e `abs_offset_error_K`, che hanno `0` `NaN`.

---

## 2. Tabelle essenziali dei risultati (`VAL` e `seen` separati)

Fonti: `macro_headline_wide.csv`, `macro_regional_long.csv`, `paired_diff_macro_long.csv`, `paired_frame_signs.csv`, `t_only_macro_long.csv`. Tutti i valori sono medie sulle 3 repliche di maschera; tutte le unità di temperatura/errore sono in **K**.

> **Nota sui segni:** per `RMSE`, `MAE` e `|bias|` (che coincide con `|delta_applied − delta_oracle|`), una differenza `A − B < 0` indica che `A` è più vicino al vero di `B` (miglioramento). Per `Hot IoU`, un valore maggiore è migliore (`A − B > 0`). Per il bias firmato `Bias`, il segno indica sovrastima (`+`) o sottostima (`−`) e può cancellarsi tra frame/condizioni: per valutare la qualità dell'offset si usano `|bias|` e la distanza dall'`oracle`.

### 2.1 `val_eval` (3 condizioni, 123 frame × 3 repliche = 369 istanze/modello)

#### A) Livelli assoluti per metodo e trattamento (`VAL`)

| Modello | `p` | Tratt. | RMSE | Off-anc RMSE | `|Bias|` (= err. offset) | Bias | Cent. | MAE | Hot IoU | Hot RMSE | Peak RMSE | Jet RMSE |
| --- | :---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **U-Net `p_on`** | sì | `free` | 42.96 | 42.94 | 12.88 | −7.50 | 38.36 | 23.21 | 0.759 | 21.6 | 29.4 | 128.5 |
| | | `oracle` | 38.36 | 38.34 | 0.00 | 0.00 | 38.36 | 16.54 | 0.973 | 14.3 | 25.7 | 127.4 |
| | | `ao9` | 43.14 | 43.12 | 14.83 | −12.35 | 38.36 | 26.23 | 0.940 | 23.9 | 35.8 | 126.7 |
| | | `mean9` | 44.10 | 44.09 | 16.44 | −12.89 | 38.36 | 27.61 | 0.930 | 25.2 | 37.0 | 127.0 |
| | | **`median9`** | **38.58** | **38.55** | **3.11** | +1.65 | 38.36 | **16.14** | **0.972** | **14.2** | **25.3** | 128.1 |
| **U-Net `p_off`** | no | `free` | 43.53 | 43.50 | 17.52 | −5.85 | 38.00 | 26.94 | 0.893 | 25.0 | 28.9 | 123.9 |
| | | `oracle` | 38.00 | 37.98 | 0.00 | 0.00 | 38.00 | 16.68 | 0.973 | 14.7 | 21.5 | 126.1 |
| | | `ao9` | 42.86 | 42.84 | 14.66 | −6.96 | 38.00 | 25.40 | 0.945 | 23.1 | 29.6 | 126.7 |
| | | `mean9` | 43.80 | 43.79 | 16.21 | −7.08 | 38.00 | 26.49 | 0.936 | 24.2 | 30.7 | 127.5 |
| | | **`median9`** | **38.49** | **38.47** | **5.09** | −0.08 | 38.00 | **16.27** | **0.972** | **14.8** | **22.3** | 127.4 |
| **GLU `p_on`** | sì | `free` | 45.80 | 45.77 | 8.77 | +2.48 | 44.61 | 21.31 | 0.962 | 18.9 | 29.8 | 146.8 |
| | | `oracle` | 44.61 | 44.58 | 0.00 | 0.00 | 44.61 | 19.13 | 0.967 | 17.4 | 30.2 | 144.8 |
| | | `ao9` | 51.12 | 51.10 | 19.14 | −12.00 | 44.61 | 32.02 | 0.916 | 30.0 | 41.1 | 143.7 |
| | | `mean9` | 52.58 | 52.56 | 21.64 | −13.61 | 44.61 | 34.10 | 0.895 | 32.2 | 43.0 | 143.8 |
| | | **`median9`** | **44.88** | **44.85** | **3.85** | −0.78 | 44.61 | **18.80** | **0.964** | **17.3** | 30.7 | 145.5 |
| **GLU `p_off`** | no | `free` | 47.54 | 47.51 | 18.58 | −0.61 | 42.72 | 28.46 | 0.903 | 26.5 | 37.6 | 138.0 |
| | | `oracle` | 42.72 | 42.69 | 0.00 | 0.00 | 42.72 | 19.02 | 0.967 | 18.0 | 34.8 | 137.3 |
| | | `ao9` | 48.20 | 48.19 | 16.83 | −6.52 | 42.72 | 29.21 | 0.927 | 27.4 | 41.6 | 138.1 |
| | | `mean9` | 49.39 | 49.38 | 18.74 | −7.18 | 42.72 | 30.70 | 0.910 | 28.9 | 43.0 | 138.6 |
| | | **`median9`** | **43.16** | **43.14** | **4.60** | +1.71 | 42.72 | **18.85** | **0.967** | **17.9** | **34.1** | 138.8 |
| **GappyPOD `p_on`** | sì | `free` / `oracle` / `ao9` / `mean9` / **`median9`** | 70.60 / 68.19 / 76.24 / 78.19 / **68.63** | **68.59** | 15.38 / 0 / 30.47 / 34.35 / **6.19** | −0.47 (`md9`) | 68.19 | **30.16** (`md9`) | **0.947** (`md9`) | **33.5** (`md9`) | **50.3** (`md9`) | 207.1 (`md9`) |
| **GappyPOD `p_off`** | no | `free` / `oracle` / `ao9` / `mean9` / **`median9`** | 94.80 / 71.69 / 79.95 / 81.50 / **72.77** | **72.73** | 57.58 / 0 / 27.84 / 31.58 / **10.45** | −5.67 (`md9`) | 71.69 | **37.49** (`md9`) | **0.916** (`md9`) | **46.0** (`md9`) | **61.6** (`md9`) | 178.6 (`md9`) |
| **ParamMean `p_on`** | sì | `free` / `oracle` / `ao9` / `mean9` / **`median9`** | 88.01 / 84.82 / 94.30 / 96.71 / **85.53** | **85.49** | 18.25 / 0 / 34.18 / 38.70 / **9.16** | +0.55 (`md9`) | 84.82 | **34.11** (`md9`) | **0.932** (`md9`) | **38.1** (`md9`) | **64.6** (`md9`) | 266.4 (`md9`) |
| **GlobalMean** | no | `free` / `oracle` / `ao9` / `mean9` / **`median9`** | 139.70 / 114.84 / 122.01 / 123.61 / **116.09** | **116.07** | 69.36 / 0 / 35.19 / 41.59 / **12.78** | −9.98 (`md9`) | 114.84 | **51.59** (`md9`) | **0.872** (`md9`) | **63.5** (`md9`) | **144.3** (`md9`) | 330.5 (`md9`) |
| **`T_only_const`** *(a parte)* | no | storico | 311.79 | 311.85 | 45.25 | +25.03 | 308.06 | 137.81 | 0.482 | 71.3 | 129.8 | 1167.6 |

#### B) Contrasti chiave e distanza residua dall'oracle (`VAL`, `all_rmse_K` e `|bias|`)

| Modello | `med9−ao9` RMSE (frame `−/+`, celle) | `med9−mean9` RMSE | `mean9−ao9` RMSE (solo shrink) | `med9−free` RMSE (frame `−/+`, celle) | `med9−or` vs `ao9−or` RMSE | `med9−ao9` `|Bias|` | `med9−free` `|Bias|` | `med9−ao9` IoU |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| **U-Net `p_on`** | **−4.56** (285/84; 9/0) | −5.53 | +0.96 (0/9 celle) | **−4.39** (272/97; 6/3) | **+0.21** vs +4.77 | −11.72 | −9.77 | +0.032 |
| **U-Net `p_off`** | **−4.37** (254/115; 9/0) | −5.31 | +0.94 (0/9) | **−5.03** (323/46; 9/0) | **+0.49** vs +4.85 | −9.57 | −12.43 | +0.026 |
| **GLU `p_on`** | **−6.23** (289/80; 9/0) | −7.69 | +1.46 (0/9) | **−0.92** (260/109; 9/0) | **+0.27** vs +6.51 | −15.29 | −4.92 | +0.048 |
| **GLU `p_off`** | **−5.04** (270/99; 9/0) | −6.23 | +1.19 (0/9) | **−4.37** (337/32; 9/0) | **+0.45** vs +5.49 | −12.24 | −13.98 | +0.041 |
| **GappyPOD `p_on`** | **−7.61** (349/20; 9/0) | −9.56 | +1.95 (0/9) | **−1.97** (331/38; 9/0) | **+0.43** vs +8.04 | −24.27 | −9.19 | +0.018 |
| **GappyPOD `p_off`** | **−7.18** (264/105; 9/0) | −8.73 | +1.55 (0/9) | **−22.03** (347/22; 9/0) | **+1.07** vs +8.26 | −17.39 | −47.13 | +0.040 |
| **ParamMean `p_on`** | **−8.78** (291/78; 9/0) | −11.18 | +2.41 (0/9) | **−2.48** (306/63; 9/0) | **+0.71** vs +9.48 | −25.02 | −9.09 | +0.007 |
| **GlobalMean** | **−5.92** (297/72; 9/0) | −7.52 | +1.60 (0/9) | **−23.61** (360/9; 9/0) | **+1.25** vs +7.17 | −22.41 | −56.58 | −0.017 |

*Nota sulla dispersione fra repliche di maschera su VAL:* gli intervalli `[rep_min, rep_max]` sono molto stretti rispetto agli effetti: per `median9 − ao9` sui neurali la variazione tra le 3 repliche è `≤ 0.95 K` (es. `U-Net p_on`: `[−4.73, −4.45] K`; `GLU p_on`: `[−6.30, −6.14] K`), e `median9` ha range RMSE di appena `0.10–0.68 K` tra repliche.

---

### 2.2 `seen_train_eval` (20 condizioni, 220 frame × 3 repliche = 660 istanze/modello)

#### A) Livelli assoluti per metodo e trattamento (`seen`)

| Modello | `p` | Tratt. | RMSE | Off-anc RMSE | `|Bias|` | Bias | Cent. | MAE | Hot IoU | Hot RMSE | Peak RMSE | Jet RMSE |
| --- | :---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **U-Net `p_on`** | sì | `free` | 51.67 | 51.66 | 8.02 | +5.62 | 50.61 | 24.57 | **0.766** | **23.5** | **57.1** | 132.6 |
| | | `oracle` | 50.61 | 50.60 | 0.00 | 0.00 | 50.61 | 24.10 | 0.739 | 25.1 | 59.9 | 130.7 |
| | | `ao9` | 53.81 | 53.81 | 13.66 | −3.27 | 50.61 | 30.39 | 0.707 | 31.4 | 63.6 | 131.7 |
| | | `mean9` | 54.52 | 54.52 | 15.27 | −4.25 | 50.61 | 31.61 | 0.699 | 32.7 | 64.6 | 131.9 |
| | | **`median9`** | **51.10** | **51.09** | **5.51** | +1.46 | 50.61 | **23.46** | 0.754 | 24.5 | 58.8 | 132.1 |
| **U-Net `p_off`** | no | `free` | 55.44 | 55.44 | 16.00 | +3.23 | 51.17 | 30.28 | 0.707 | 26.4 | 55.0 | 130.4 |
| | | `oracle` | 51.17 | 51.17 | 0.00 | 0.00 | 51.17 | 24.66 | 0.753 | 24.4 | 53.9 | 127.5 |
| | | `ao9` | 53.78 | 53.78 | 12.76 | −0.19 | 51.17 | 29.44 | 0.733 | 28.6 | 55.3 | 128.9 |
| | | `mean9` | 54.37 | 54.37 | 14.09 | −0.57 | 51.17 | 30.52 | 0.728 | 29.8 | 56.1 | 129.1 |
| | | **`median9`** | **51.79** | **51.78** | **6.45** | +1.16 | 51.17 | **23.68** | **0.766** | **23.2** | **53.1** | 129.4 |
| **GLU `p_on`** | sì | `free` | **55.51** | **55.50** | **5.59** | −0.79 | 55.09 | **25.73** | **0.759** | **28.5** | 70.8 | 137.1 |
| | | `oracle` | 55.09 | 55.08 | 0.00 | 0.00 | 55.09 | 26.11 | 0.746 | 28.9 | 70.6 | 136.3 |
| | | `ao9` | 58.48 | 58.48 | 14.82 | −3.30 | 55.09 | 32.91 | 0.703 | 35.2 | 74.3 | 137.4 |
| | | `mean9` | 59.17 | 59.17 | 16.40 | −3.58 | 55.09 | 34.11 | 0.697 | 36.3 | 74.9 | 137.7 |
| | | **`median9`** | 55.56 | 55.55 | 5.74 | −0.48 | 55.09 | 25.76 | 0.750 | 28.7 | **70.5** | 137.2 |
| **GLU `p_off`** | no | `free` | 60.31 | 60.31 | 11.31 | +0.83 | 58.51 | 29.68 | 0.721 | 30.6 | 78.4 | 145.8 |
| | | `oracle` | 58.51 | 58.51 | 0.00 | 0.00 | 58.51 | 28.43 | 0.725 | 31.6 | 79.8 | 143.3 |
| | | `ao9` | 61.83 | 61.83 | 15.65 | −1.37 | 58.51 | 34.71 | 0.693 | 36.9 | 81.6 | 144.4 |
| | | `mean9` | 62.49 | 62.49 | 17.13 | −1.61 | 58.51 | 35.96 | 0.689 | 38.2 | 82.2 | 144.5 |
| | | **`median9`** | **59.25** | **59.24** | **7.50** | +1.96 | 58.51 | **27.40** | **0.742** | **29.9** | **77.7** | 145.5 |
| **GappyPOD `p_on`** | sì | `free` / `oracle` / `ao9` / `mean9` / **`median9`** | 108.36 / 106.59 / 115.97 / 118.08 / **107.81** | **107.79** | 14.95 / 0 / 38.13 / 42.64 / **12.47** | +3.70 (`md9`) | 106.59 | **51.61** (`md9`) | **0.670** (`md9`) | **64.1** (`md9`) | **160.7** (`md9`) | 271.6 (`md9`) |
| **GappyPOD `p_off`** | no | `free` / `oracle` / `ao9` / `mean9` / **`median9`** | 113.08 / 103.46 / 111.88 / 113.88 / **105.50** | **105.48** | 35.47 / 0 / 35.79 / 40.27 / **17.08** | +2.65 (`md9`) | 103.46 | **54.97** (`md9`) | **0.655** (`md9`) | **70.0** (`md9`) | **146.0** (`md9`) | 237.5 (`md9`) |
| **ParamMean `p_on`** | sì | `free` / `oracle` / `ao9` / `mean9` / **`median9`** | 129.35 / 127.17 / 139.54 / 142.35 / **128.70** | **128.67** | 19.08 / 0 / 47.75 / 53.69 / **15.30** | +4.83 (`md9`) | 127.17 | **59.04** (`md9`) | **0.657** (`md9`) | **73.6** (`md9`) | **193.4** (`md9`) | 336.1 (`md9`) |
| **GlobalMean** | no | `free` / `oracle` / `ao9` / `mean9` / **`median9`** | 173.68 / 151.38 / 163.21 / 166.69 / **154.47** | **154.44** | 71.08 / 0 / 49.19 / 58.90 / **25.97** | +6.10 (`md9`) | 151.38 | **71.82** (`md9`) | **0.601** (`md9`) | **94.2** (`md9`) | **232.8** (`md9`) | 393.7 (`md9`) |
| **`T_only_const`** *(a parte)* | no | storico | 323.84 | 323.88 | 60.74 | +0.97 | 315.85 | 175.34 | 0.360 | 126.7 | 206.8 | 1042.9 |

#### B) Contrasti chiave e distanza residua dall'oracle (`seen`, `all_rmse_K` e `|bias|`)

| Modello | `med9−ao9` RMSE (frame `−/+`; cond. `−/+`) | `med9−mean9` RMSE | `mean9−ao9` RMSE (solo shrink) | `med9−free` RMSE (frame `−/+`; cond. `−/+`) | `med9−or` vs `ao9−or` RMSE | `med9−ao9` `|Bias|` | `med9−free` `|Bias|` |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| **U-Net `p_on`** | **−2.71** (441/219; 18/2) | −3.42 | +0.71 (0/60 celle) | **−0.58** (410/250; 13/7) | **+0.48** vs +3.20 | −8.14 | −2.51 |
| **U-Net `p_off`** | **−1.99** (453/207; 20/0) | −2.58 | +0.59 (1/59) | **−3.66** (484/176; 18/2) | **+0.61** vs +2.61 | −6.31 | −9.55 |
| **GLU `p_on`** | **−2.91** (485/175; 20/0) | −3.61 | +0.69 (0/60) | **+0.05** (306/354; 8/12) | **+0.47** vs +3.39 | −9.08 | +0.15 |
| **GLU `p_off`** | **−2.58** (462/198; 19/1) | −3.24 | +0.65 (1/59) | **−1.06** (408/252; 15/5) | **+0.74** vs +3.32 | −8.15 | −3.81 |
| **GappyPOD `p_on`** | **−8.16** (525/135; 19/1) | −10.27 | +2.11 (0/60) | **−0.55** (387/273; 12/8) | **+1.23** vs +9.39 | −25.66 | −2.48 |
| **GappyPOD `p_off`** | **−6.38** (493/167; 20/0) | −8.38 | +2.00 (0/60) | **−7.57** (494/166; 19/1) | **+2.04** vs +8.43 | −18.71 | −18.40 |
| **ParamMean `p_on`** | **−10.85** (522/138; 19/1) | −13.65 | +2.81 (3/57) | **−0.65** (420/240; 12/8) | **+1.53** vs +12.37 | −32.46 | −3.78 |
| **GlobalMean** | **−8.74** (423/237; 16/4) | −12.22 | +3.48 (0/60) | **−19.21** (585/75; 19/1) | **+3.09** vs +11.83 | −23.22 | −45.11 |

---

## 3. Risposte dirette alle cinque domande diagnostiche

### 3.1 La mediana migliora la stima dell'offset?
**Sì, in modo netto e generalizzato.** Poiché `all_abs_bias_K` dopo una traslazione coincide con l'errore assoluto medio dell'offset `|delta_applied − delta_oracle|`, i dati mostrano un crollo dell'errore di stima dell'offset di un fattore **2.5×–5×** rispetto ad `AO9` e **3×–5.6×** rispetto a `mean9`:
- Su `VAL`, `|delta_applied − delta_oracle|` scende da `14.66–19.14 K` (`AO9`) a **`3.11–5.09 K`** (`median9`) sui quattro modelli neurali, e da `27.84–35.19 K` a **`6.19–12.78 K`** sulle baseline.
- Su `seen`, scende da `12.76–15.65 K` (`AO9`) a **`5.51–7.50 K`** (`median9`) sui neurali, e da `35.79–49.19 K` a **`12.47–25.97 K`** sulle baseline.
- Anche il bias firmato macro negativo che affliggeva `AO9` su `VAL` (`−6.52` a `−12.35 K` sui neurali) scompare quasi del tutto (`−0.78` a `+1.71 K`).

### 3.2 Migliora anche rispetto a `free`, oppure ripara soltanto parte del danno di `AO9`?
**Supera nettamente la semplice riparazione di `AO9`:**
- Su **`VAL`**, `median9` batte `free` su **8 modelli su 8** (e in `9/9` celle condizione×replica per 7 modelli su 8). Sui neurali, il guadagno `median9 − free` è **`−4.39 K`** (`U-Net p_on`), **`−5.03 K`** (`U-Net p_off`), **`−0.92 K`** (`GLU p_on`) e **`−4.37 K`** (`GLU p_off`), colmando quasi tutto lo spazio verso l'`oracle` (`+0.21`–`+0.49 K` residui).
- L'unica condizione `VAL` su cui `median9` peggiora leggermente rispetto a `free` è `0NH3_1.05` per `U-Net p_on` (`44.91 -> 45.18 K`, **`+0.27 K`**; `|bias|` da `2.93` a `5.08 K`), esattamente dove `free` aveva già bias quasi nullo (`+0.54 K`). Su tutte le altre 11 coppie condizione×rete di `VAL`, `median9` migliora `free` (`−0.20` a `−13.00 K`).
- Su **`seen`**, `median9` migliora `free` su **7 modelli su 8**, con guadagni ampi sulle varianti `p_off` (`U-Net p_off −3.66 K`, `18/20` condizioni, fino a `−17.49 K` su `40NH3_1.h5`; `GLU p_off −1.06 K`, `15/20`; `GappyPOD p_off −7.57 K`; `GlobalMean −19.21 K`) e modesti sulle `p_on` (`U-Net p_on −0.58 K`, `13/20`, peggioramento massimo `+0.26 K` su `0NH3_1.h5`).
- **Eccezione esplicita (`GLU p_on` su `seen`):** qui `free` aveva già `|bias| = 5.59 K` (`oracle − free = −0.42 K`); `median9` produce un esito neutro/leggermente peggiorativo (**`+0.05 K`**, `8/20` condizioni migliorate e `12/20` peggiorate; frame `306−/354+`; peggioramento massimo `+0.35 K` su `20NH3_1.2.h5`). Questo mostra il limite intrinseco: quando il modello `free` ha già un errore di livello intorno a `3–6 K`, la mediana su 9 siti non può garantire un guadagno ulteriore, ma evita le degradazioni gravi di `AO9` (`+2.96 K` macro, `0/20` condizioni migliorate).

### 3.3 Coerenza fra modelli, `p_on/p_off`, `seen/VAL` e condizioni critiche
Il dettaglio per le tre condizioni `VAL` (media sulle 3 repliche, 4 reti neurali) chiarisce l'eterogeneità dei regimi:

| Condizione VAL | Rete | `free` RMSE / IoU | `oracle` RMSE / IoU | `ao9` RMSE / IoU | `median9` RMSE / IoU | `med9−ao9` | `med9−free` | `med9−or` |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **`40NH3_0.9`** *(livello critico)* | U-Net `p_on` | 39.92 / 0.310 | 26.80 / 0.950 | 29.15 / 0.889 | **26.93 / 0.949** | −2.22 | **−13.00** | +0.13 |
| | U-Net `p_off` | 36.93 / 0.713 | 26.87 / 0.952 | 29.37 / 0.900 | **27.35 / 0.950** | −2.02 | **−9.57** | +0.49 |
| | GLU `p_on` | 29.24 / 0.925 | 28.55 / 0.936 | 32.43 / 0.891 | **29.02 / 0.929** | −3.41 | −0.22 | +0.47 |
| | GLU `p_off` | 39.31 / 0.748 | 32.23 / 0.938 | 35.50 / 0.904 | **33.02 / 0.941** | −2.48 | **−6.30** | +0.79 |
| **`20NH3_1.1`** *(dove AO9 crollava)* | U-Net `p_on` | 44.05 / 0.984 | 43.52 / 0.985 | 52.45 / 0.948 | **43.62 / 0.985** | **−8.83** | −0.44 | +0.09 |
| | U-Net `p_off` | 42.29 / 0.983 | 40.29 / 0.984 | 49.73 / 0.954 | **40.80 / 0.982** | **−8.93** | −1.49 | +0.51 |
| | GLU `p_on` | 54.66 / 0.982 | 52.23 / 0.985 | 63.54 / 0.879 | **52.33 / 0.984** | **−11.22** | −2.33 | +0.10 |
| | GLU `p_off` | 48.97 / 0.982 | 46.07 / 0.982 | 57.32 / 0.897 | **46.25 / 0.982** | **−11.07** | −2.72 | +0.19 |
| **`0NH3_1.05`** *(getto forte)* | U-Net `p_on` | 44.91 / 0.983 | 44.77 / 0.983 | 47.82 / 0.984 | 45.18 / 0.982 | −2.63 | **+0.27** | +0.42 |
| | U-Net `p_off` | 51.36 / 0.983 | 46.85 / 0.984 | 49.46 / 0.982 | **47.32 / 0.983** | −2.14 | −4.05 | +0.47 |
| | GLU `p_on` | 53.50 / 0.980 | 53.04 / 0.980 | 57.37 / 0.978 | **53.29 / 0.980** | −4.07 | −0.20 | +0.25 |
| | GLU `p_off` | 54.32 / 0.978 | 49.85 / 0.979 | 51.79 / 0.979 | **50.22 / 0.979** | −1.57 | −4.10 | +0.37 |

- **Coerenza vs `AO9`:** `median9` batte `AO9` su `9/9` celle VAL per tutti gli 8 modelli e su `18–20/20` condizioni `seen` sui neurali (unici lievi peggioramenti `seen`: `+0.52 K` per `U-Net p_on` su `0NH3_0.9`, `+0.29 K` su `0NH3_1.h5`, `+0.06 K` per `GLU p_off` su `0NH3_0.9`). Su `20NH3_1.1`, dove `AO9` degradava tutte le reti di `+7.4`–`+8.9 K`, `median9` elimina completamente il danno e migliora tutte e quattro le reti anche rispetto a `free`.
- **Peggioramenti residui rispetto a `free` su `seen`:** sulle reti neurali restano contenuti (`+0.26 K` peggior caso `U-Net p_on`, `+0.35 K` `GLU p_on`, `+0.82 K` `U-Net p_off` e `+0.85 K` `GLU p_off` entrambi su `0NH3_0.85.h5`). Sulle baseline, il peggior caso rispetto a `free` è sempre `0NH3_0.85.h5` (`+7.21 K` `GappyPOD p_on`, `+5.07 K` `ParamMean`, `+2.99 K` `GappyPOD p_off`), condizione in cui i residui delle baseline presentano forte eterogeneità spaziale tra ancore anche fuori dai getti (vedi §4).
- **Effetto `p_on` vs `p_off`:** coerentemente con A0–A3, il beneficio assoluto di `median9` rispetto a `free` è molto più marcato sui modelli `p_off` (che in `free` hanno bias più grande). Dopo `median9` su `VAL`, **`p_off` eguaglia o supera `p_on`** (`U-Net`: `38.49` vs `38.58 K`, differenza `p_on − p_off = +0.09 K`; `GLU`: `43.16` vs `44.88 K`, `+1.72 K`), ricalcando da vicino il comportamento `oracle` (`+0.36 K` e `+1.89 K`). Su `seen`, `p_on` conserva un piccolo margine strutturale (`−0.69 K` su U-Net, `−3.69 K` su GLU), coerente con `oracle` (`−0.56 K` e `−3.42 K`).
- **Effetto U-Net vs GLU:** dopo `median9`, U-Net mantiene integralmente il vantaggio strutturale già emerso con `oracle`: su `VAL`, `U-Net − GLU = −6.31 K` (`p_on`) e `−4.67 K` (`p_off`), con `Hot IoU` pari o superiore (`0.972` vs `0.964–0.967`); su `seen`, `−4.47 K` (`p_on`) e `−7.46 K` (`p_off`). Resta un risultato a singolo `seed 0`.

### 3.4 Migliorano regioni calde, getti e `off-anchor`?
- **Regioni calde (`T ≥ 2100 K`) e picchi (`T ≥ 2200 K`):**
  - Su `VAL`, `median9` recupera quasi interamente il livello `oracle`: `U-Net p_on` sale da `0.759` (`free`) / `0.940` (`AO9`) a **`0.972`** (`oracle 0.973`), `Hot RMSE` scende da `21.6` / `23.9` a **`14.2 K`** (`oracle 14.3`), `Peak RMSE` da `29.4` / `35.8` a **`25.3 K`** (`oracle 25.7`). Su `GLU p_on`, `median9` ripara il crollo di `AO9` (`0.916 -> 0.964`, `Hot RMSE 30.0 -> 17.3 K`).
  - Su `seen`, `median9` migliora `Hot RMSE`, `Peak RMSE` e `Hot IoU` su tutti gli 8 modelli sia rispetto ad `AO9` sia — fatto notevole spiegato in §4.2 — rispetto allo stesso `oracle` (`+0.004` a `+0.016` di IoU sui neurali). Rispetto a `free`, `Hot IoU` migliora nettamente sui `p_off` (`U-Net +0.059`, `GLU +0.020`) e scende leggermente sui `p_on` (`U-Net 0.766 -> 0.754`, `GLU 0.759 -> 0.750`), replicando in forma attenuata la non-monotonicità già osservata per `oracle` su `seen` (`0.739` e `0.746`).
- **Getti freddi (`T < 1800 K`):** **nessun miglioramento.** Su `VAL`, `Jet RMSE` con `median9` resta `127.4–145.5 K` sui neurali (`129.4–145.5 K` su `seen`), leggermente peggiore di `oracle` (`+0.62` a `+1.51 K` su VAL; `+0.93` a `+2.18 K` su `seen`) e su `U-Net p_off` peggiore di `free` (`123.9 -> 127.4 K`). Come previsto matematicamente, nessuna traslazione scalare modifica la `centered-RMSE` né risolve l'errore strutturale nei getti.
- **Comportamento `off-anchor` (16375 siti):** per tutti e quattro i neurali, `off_rmse_K − all_rmse_K` con `median9` è compreso tra `−0.034 K` e `−0.005 K`, `off_hot_iou − all_hot_iou = −0.00001`, `off_hot_rmse_K − all_hot_rmse_K ≤ +0.003 K`, `off_jet_rmse_K − all_jet_rmse_K` tra `−0.12` e `−0.02 K`. I conteggi di segno frame-by-frame su `off_rmse_K` coincidono con `all_rmse_K` (a meno di 1 singolo frame su 660 per `GLU p_off` `seen`). I guadagni di `median9` si trasferiscono interamente fuori dalle 9 ancore.

### 3.5 Quanto del cambiamento dipende dallo shrink anziché dallo stimatore?
La decomposizione algebrica esatta è:
$$\text{median9} - \text{AO9} = \underbrace{(\text{median9} - \text{mean9})}_{\text{cambio di stimatore (senza shrink)}} + \underbrace{(\text{mean9} - \text{AO9})}_{\text{rimozione dello shrink } (\tau=1 \to 0)}$$

I risultati sono **univoci su 8 modelli su 8 e su entrambi i ruoli**:
- **`mean9 − AO9` è sempre positivo (peggiorativo):** `+0.94` a `+1.46 K` sui neurali `VAL` (`0/9` celle migliorate; `62–143` frame migliorati contro `226–307` peggiorati); `+0.59` a `+0.71 K` sui neurali `seen` (`0–1/60` celle migliorate); `+1.55` a `+3.48 K` sulle baseline. Lo stesso vale per `|bias|` (`+1.33` a `+2.50 K` sui neurali; `+3.75` a `+9.71 K` sulle baseline) e `Hot IoU` (peggiora su tutti i neurali).
- **`median9 − mean9` è più negativo di `median9 − AO9`:** `−5.31` a `−7.69 K` sui neurali `VAL`, `−2.58` a `−3.61 K` sui neurali `seen`, `−7.52` a `−13.65 K` sulle baseline.
- **Conclusione:** lo shrink $\tau=1$ in `AO9` (`delta_ao9 = 0.9 * delta_mean9`) stava in realtà **mitigando del 10%** la sovracorrezione causata dalla media non robusta. Rimuovere lo shrink da solo peggiora le prestazioni; **l'intero beneficio (oltre il 100% del guadagno netto `median9 − AO9`) proviene dalla sostituzione della media con la mediana.**

---

## 4. Analisi dei nove residui per ancora: ruolo dei siti nei getti e cautela statistica

Fonte: `anchor_residual_per_condition.csv`, `anchor_col95_share_macro_long.csv`, `offset_error_per_condition.csv`, `macro_valid_counts_long.csv`.

### 4.1 Cosa dimostrano i dati per-ancora
La tabella seguente riporta, per i quattro modelli neurali, la media macro del residuo firmato `T_obs − T_pred` e del residuo assoluto `|T_obs − T_pred|` su ciascuna delle 9 ancore (ordine `anchor_id = 0..8`):

#### A) `val_eval` (macro sulle 3 condizioni e 3 repliche)

| Rete | Metrica | `(32,32)` | `(32,64)` | **`(32,95)`** | `(64,32)` | `(64,64)` | `(64,95)` | `(95,32)` | `(95,64)` | **`(95,95)`** | Quota macro `|r|` di `{(32,95),(95,95)}` vs `(64,95)` | Quota tabella `col95` *(3 ancore)* |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| **U-Net `p_on`** | firmato | +10.2 | +8.8 | **−36.4** | +9.9 | +11.0 | +4.3 | +10.2 | +11.0 | **−77.5** | 66.9% vs 4.8% | 66.4% |
| | `|resid|` | 15.0 | 13.6 | **91.1** | 12.9 | 14.9 | 14.5 | 14.2 | 15.0 | **111.1** | | |
| **U-Net `p_off`** | firmato | +5.8 | +6.0 | **−30.2** | +6.1 | +4.9 | +4.7 | +6.7 | +5.3 | **−20.3** | 56.2% vs 6.0% | 54.9% |
| | `|resid|` | 21.9 | 20.4 | **82.4** | 22.1 | 21.0 | 20.3 | 21.8 | 21.7 | **108.8** | | |
| **GLU `p_on`** | firmato | −6.2 | −1.8 | **−37.5** | −4.8 | −3.6 | −1.9 | −2.1 | −1.1 | **−85.8** | 77.1% vs 3.4% | 70.5% |
| | `|resid|` | 10.4 | 10.8 | **107.0** | 10.9 | 8.7 | 10.3 | 10.5 | 8.7 | **129.1** | | |
| **GLU `p_off`** | firmato | −2.5 | +1.6 | **−13.1** | +3.6 | +3.5 | +0.0 | +4.8 | +3.7 | **−60.7** | 57.8% vs 6.2% | 55.0% |
| | `|resid|` | 21.2 | 20.6 | **72.1** | 22.2 | 22.0 | 21.7 | 19.5 | 21.0 | **130.7** | | |

*(Sulle baseline `VAL` la concentrazione è analoga o maggiore: per `GappyPOD p_on` e `ParamMean p_on`, `(32,95)` e `(95,95)` hanno `|resid| = 154–251 K` contro `11–21 K` negli altri 7 siti, coprendo il `79.7%` della somma macro `|resid|`, mentre `(64,95)` pesa appena l'`1.9–2.2%`.)*

#### B) `seen_train_eval` (macro sulle 20 condizioni e 3 repliche)

| Rete | Metrica | `(32,32)` | **`(32,64)`** | **`(32,95)`** | `(64,32)` | `(64,64)` | `(64,95)` | `(95,32)` | **`(95,64)`** | **`(95,95)`** | Quota macro `|r|` di `{(32,95),(95,95)}` vs `(64,95)` | Quota tabella `col95` *(3 ancore)* |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| **U-Net `p_on`** | firmato | −1.4 | −15.6 | +5.1 | −3.8 | −4.0 | −6.0 | −3.9 | −8.9 | **−50.5** | 63.9% vs 3.6% | 64.4% |
| | `|resid|` | 8.8 | **23.8** | **65.0** | 8.4 | 8.5 | 9.6 | 10.5 | **24.9** | **102.5** | | |
| **U-Net `p_off`** | firmato | −0.4 | −12.0 | +0.2 | −1.7 | −1.5 | −3.7 | −1.4 | −5.2 | −8.5 | 50.7% vs 5.7% | 55.0% |
| | `|resid|` | 16.5 | **31.1** | **66.1** | 16.1 | 17.0 | 17.1 | 18.5 | **30.3** | **84.9** | | |
| **GLU `p_on`** | firmato | +2.4 | −5.5 | +6.4 | +1.0 | −0.4 | −1.0 | −2.7 | −3.9 | **−21.4** | 64.7% vs 3.0% | 65.2% |
| | `|resid|` | 10.1 | **25.5** | **73.8** | 7.5 | 5.8 | 7.7 | 10.3 | **25.1** | **94.8** | | |
| **GLU `p_off`** | firmato | −3.3 | −9.1 | −5.1 | +4.5 | +1.9 | +0.4 | +3.0 | −7.2 | −7.0 | 60.7% vs 3.8% | 61.5% |
| | `|resid|` | 12.7 | **29.6** | **84.8** | 11.4 | 10.6 | 11.4 | 14.0 | **26.8** | **95.2** | | |

**Tre evidenze quantitative dimostrate dai dati per-ancora:**
1. **Conferma dell'ipotesi primaria sui due siti di bordo getto su `VAL`:** in tutte e tre le condizioni `VAL`, `(32,95)` e `(95,95)` hanno residui assoluti `4×–15×` più grandi degli altri sette siti — incluso `(64,95)`, che si comporta esattamente come i 6 siti interni. Su `20NH3_1.1`, il residuo firmato medio sui 7 siti interni è `−5.1 K` (`U-Net p_on`, contro `dOr = −6.3 K`) e `−13.5 K` (`GLU p_on`, contro `dOr = −15.8 K`), mentre `(32,95)` vale `−59.2 / −92.0 K` e `(95,95)` vale **`−189.0 / −229.1 K`**: sono questi due siti a trascinare `dAO9` a `−28.4 / −41.6 K` (`dMean9 = −31.5 / −46.2 K`), mentre `median9` (`−6.0 / −14.1 K`) resta ancorata ai 7 siti concordi. Analogamente, su `0NH3_1.05`, `(32,95)` vale `−109 K` per `U-Net p_on` e `GLU p_on`, trascinando `AO9` in negativo (`−11.1 / −9.7 K`).
2. **Su `seen` il fenomeno non è limitato alla sola colonna 95:** nelle miscele ad alta ammoniaca (`80NH3_1.15.h5` e `80NH3_1.2.h5`), dove cambia la geometria della fiamma, anche **`(32,64)` e `(95,64)`** sviluppano residui molto elevati (`|resid| = 160.3 K` e `134.9 K` su `80NH3_1.15`, `70.9 K` e `86.3 K` su `80NH3_1.2` per `U-Net p_on`). In questi casi **4 ancore su 9** cadono in zone ad alto gradiente/errore, avvicinandosi al punto di rottura della mediana di 9 elementi: infatti le massime distanze `median9 − oracle` sui neurali `seen` (`+1.13` a `+2.16 K`) si verificano proprio su `80NH3_1.15` e `80NH3_1.2`.
3. **Sulle baseline, errori spaziali diffusi possono colpire anche fuori dai getti:** su `0NH3_0.85.h5` (`seen`), `GappyPOD p_on` presenta residui firmati tra `−56.7 K` e `+122.8 K` distribuiti su quasi tutte le 9 ancore, motivo per cui lì `median9` dista `+8.44 K` dall'`oracle` e peggiora `free` di `+7.21 K`.

### 4.2 Cosa suggeriscono i dati: perché `median9` batte `oracle` in `MAE`, ma non è uno stimatore «giusto» della media spaziale
Un risultato apparentemente paradossale delle tabelle (§2) offre una lezione statistica e fisica molto chiara: **su tutti gli 8 modelli e su entrambi i ruoli (`VAL` e `seen`), `median9` ha `MAE` inferiore all'`oracle`** (`−0.17` a `−0.42 K` neurali VAL; `−0.34` a `−1.03 K` neurali `seen`; `−1.56` a `−9.74 K` sulle baseline) e su `seen` ottiene anche `Hot RMSE`, `Peak RMSE` e `Hot IoU` migliori dell'`oracle`, mentre ha `RMSE` leggermente peggiore (`+0.21`–`+0.74 K` neurali) e `Jet RMSE` leggermente peggiore (`+0.62`–`+2.18 K` neurali).

La spiegazione è strutturale:
- I getti freddi (`T < 1800 K`) coprono solo il **`6.8%`** dei pixel su `VAL` (`4.7%–8.4%`) e il **`9.2%`** su `seen`, mentre la regione calda (`T ≥ 2100 K`) copre l'**`85.9%`** su `VAL` (`78.5%–91.2%`) e il **`60.3%`** su `seen`.
- I residui locali grandi su `(32,95)` e `(95,95)` **non sono rumore di misura**: sono veri errori del modello in una regione fisica rilevante (i getti freddi), spesso asimmetrici (sovrastima della temperatura nel getto $\implies T_{\text{true}} - T_{\text{pred}} < 0$).
- Nella griglia fissa $3\times 3$, quei due siti pesano `2/9 = 22.2%` — oltre **3×** l'area reale dei getti su `VAL` — per cui `mean9` e `AO9` sovrarappresentano fortemente i getti e distorcono l'offset globale.
- Al contrario, `oracle` è la **media spaziale** su tutti i `16384` pixel (minimizza la MSE/RMSE globale e viene leggermente trascinato verso i grandi residui negativi dei getti), mentre `median9` **esclude i siti nei getti** e stima l'offset del **bulk caldo maggioritario**. Ciò riduce `MAE` (sensibile alla mediana spaziale degli errori) e allinea meglio i prodotti caldi, lasciando un piccolo bias residuo positivo rispetto alla media globale (`+1.2`–`+2.0 K` su `seen`, `+4.2`–`+4.3 K` su `0NH3_1.05` U-Net) e peggiorando lievemente `Jet RMSE`.
- **Cautela concettuale:** la mediana **non** è automaticamente uno stimatore consistente né ottimale della media spaziale del campo. Qui funziona bene rispetto ad `AO9` perché la frazione di ancore colpite da grandi errori localizzati (`2/9` su VAL, `2/9`–`4/9` su seen) resta al di sotto del punto di rottura (`5/9`) e perché la maggioranza della griglia è occupata dal bulk caldo. In una geometria o layout in cui ≥5 ancore cadessero in zone ad alto gradiente (o con distribuzione d'errore bimodale), la mediana potrebbe fallire bruscamente.

### 4.3 Cosa non è dimostrato da queste tabelle e richiederebbe altri dati/esperimenti
- **Distribuzione per-frame dei residui e quota esatta a 2 ancore:** `anchor_residual_per_condition.csv` contiene medie per condizione, non i residui per singolo frame (che restano nei `.csv.gz` remoti su Engaging). Non è quindi possibile da Git ricalcolare la quota esatta per-frame delle sole 2 ancore `{(32,95),(95,95)}` né contare in quanti singoli frame la mediana coincida con una specifica ancora o subisca rottura su `80NH3_1.15/1.2`.
- **Generalità oltre questa griglia e rumore zero:** questi dati non dimostrano robustezza ad altri layout `T`, a `K` più piccolo (es. `K=3` o `K=5`), a rumore/deriva/risposta dinamica di termocoppie reali, né a condizioni di vera estrapolazione fuori inviluppo (`TEST` o fold di estrapolazione).

---

## 5. Confronto reti vs baseline a informazione dichiarata

Tutte le baseline traggono grande beneficio da `median9` rispetto ad `AO9` (`−5.92` a `−10.85 K`) e rispetto a `free` (`−1.97` a `−23.61 K` su VAL; `−0.55` a `−19.21 K` su seen), avvicinandosi molto al proprio `oracle` (`+0.43`–`+1.25 K` VAL). Nonostante questo recupero di livello, **la conclusione primaria sul contributo incrementale delle reti rispetto alle baseline considerate non cambia:**

| Informazione disponibile | Modelli a confronto | RMSE `VAL` (`median9` / `oracle`) | RMSE `seen` (`median9` / `oracle`) | Hot IoU `VAL` (`median9`) | Jet RMSE `VAL` (`median9`) |
| --- | --- | --- | --- | ---: | ---: |
| **Senza `p` (`p_off`) + vel. 5% + 9 T ideali** | **U-Net `p_off`** | **38.49** / 38.00 | **51.79** / 51.17 | **0.972** | **127.4** |
| | **GLU `p_off`** | **43.16** / 42.72 | **59.25** / 58.51 | **0.967** | **138.8** |
| | GappyPOD `p_off` | 72.77 / 71.69 | 105.50 / 103.46 | 0.916 | 178.6 |
| **Senza `p`, senza vel. + 9 T ideali** | GlobalMean + `median9` | 116.09 / 114.84 | 154.47 / 151.38 | 0.872 | 330.5 |
| | `T_only_const` *(media 9 T)* | 311.79 / 308.06 | 323.84 / 315.85 | 0.482 | 1167.6 |
| **Con `p` (`p_on`) + vel. 5% + 9 T ideali** | **U-Net `p_on`** | **38.58** / 38.36 | **51.10** / 50.61 | **0.972** | **128.1** |
| | **GLU `p_on`** | **44.88** / 44.61 | **55.56** / 55.09 | **0.964** | **145.5** |
| | GappyPOD `p_on` | 68.63 / 68.19 | 107.81 / 106.59 | 0.947 | 207.1 |
| **Solo `p` (no vel.) + 9 T ideali** | ParamMean `p_on` *(non matched a `p_off`)* | 85.53 / 84.82 | 128.70 / 127.17 | 0.932 | 266.4 |

- **Lettura:** `median9` porta sia le reti sia le baseline quasi al loro pavimento di `centered-RMSE`. Poiché una traslazione scalare non riduce la `centered-RMSE`, il divario di `~30–47 K` su `VAL` e `~54–78 K` su `seen` tra `U-Net` e `GappyPOD`/`ParamMean` rimane intatto, e le regioni dei getti restano `1.4×–2.1×` più accurate nelle reti (e `~9×` rispetto a `T_only_const`).
- **Limite dell'affermazione:** questo confronto conferma che, nel perimetro valutato, le reti da velocità sparse superano nettamente le baseline congelate anche quando l'offset dalle 9 T è stimato in modo robusto; **non dimostra** in senso generale l'indispensabilità delle reti neurali o della non linearità (né esclude che un diverso interpolatore spaziale T-only o una baseline lineare/kernel non congelata qui possa fare meglio di `T_only_const` e `GappyPOD` R=32).

---

## 6. Giudizio sulla chiusura della diagnostica e implicazioni per la sintesi Astra–Opus

### 6.1 Giudizio di chiusura
**L'estensione esplorativa `ASTRA median9` può essere chiusa nel perimetro svolto.**
- La domanda diagnostica che l'aveva motivata ha ricevuto una risposta quantitativa completa, coerente e non ambigua su tutto l'insieme `seen220` + `VAL123` (3 repliche, 4 reti S5 + 4 baseline + `T_only_const` separato).
- Non è presente alcun errore bloccante. Non occorre alcuna nuova inferenza, nessuno sweep su altri stimatori robusti (Huber, trimmed mean), nessun cambio di layout/K e nessuna modifica al report primario `07`.
- **Nota opzionale non bloccante per Cursor (solo se si desidera igiene documentale su `astra/median9-second-pass`, non richiesta per chiudere):** in `scripts/25_astra_median9_aggregate.py` L130–138 e nei testi di accompagnamento (`INDEX.md` L20, `astra_median9_full_set_handoff.md` L38), chiarire che `anchor_col == 95` include 3 ancore (`(32,95)`, `(64,95)`, `(95,95)`), oppure — se mai si riaggregassero i CSV remoti — separare esplicitamente le 2 ancore di bordo getto `{(32,95),(95,95)}` da `(64,95)`.

### 6.2 Punti da portare alla sintesi Astra–Opus (senza decidere S7)
Dopo la chiusura e la revisione dei risultati Opus (`C149` / `A38` seed), le evidenze Astra (blocco primario `07` + second pass `08`) da integrare nella discussione con Enrico e il coordinatore sono:
1. **Il fallimento di `AO9` in A0–A3 era dovuto allo stimatore (media su 9 residui con due siti sui bordi del getto), non all'impossibilità di stimare il livello da 9 T ideali.** Con la semplice mediana ordinaria (senza shrink, tuning o esclusione manuale di siti), 9 T ideali portano tutte e 4 le reti S5 a `0.2–0.5 K` dall'`oracle` su `VAL` (`0.5–0.7 K` su `seen`), risolvendo sia il crollo di `40NH3_0.9` sia la sovracorrezione di `20NH3_1.1`.
2. **Qualsiasi eventuale pista *assisted* futura resta però ancora diagnostica ed esplorativa:** questo risultato è stato motivato da `VAL`, vale a `seed 0`, su un solo layout `3×3`, con rumore 0 (non termocoppie reali), e mostra già in `80NH3_1.15/1.2` la vicinanza al limite di rottura della mediana quando 4 ancore su 9 cadono in zone ad alto errore.
3. **Il vero collo di bottiglia strutturale resta invariato nei getti freddi (`Jet RMSE ≈ 127–146 K` VAL, `129–146 K` seen):** nessuna correzione scalare dell'offset (neppure `oracle`) risolve la struttura dei getti. La sintesi con Opus dovrà chiarire quanto più dati di training (`C149` vs `A38`) riducano la `centered-RMSE` nei getti rispetto al solo bias di livello.
4. **Confronto architetturale e ruolo di `p` (a `seed 0`):** dopo correzione robusta del livello (`median9`), `U-Net` conserva un vantaggio strutturale netto su `GLU` (`−4.7` a `−6.3 K` su `VAL`, `−4.5` a `−7.5 K` su `seen`), mentre su `VAL` `p_off + median9` eguaglia o supera `p_on + median9` (al contrario del regime `free`). Prima di trarre conclusioni generali o scegliere S7 serviranno la dispersione tra seed da Opus e il riesame congiunto delle proposte Neon (inclusi eventuale estrapolazione, multidensità, layout/rumore T).
