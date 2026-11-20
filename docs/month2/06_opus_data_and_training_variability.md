# Report OPUS — Quantità di istantanee supervisionate, variabilità del training e selezione del checkpoint

**Stato:** Sotto-missione **OPUS (O0–O5)** revisionata e valutata come **chiudibile nel perimetro concordato** (con limiti espliciti dichiarati in §6). Questo report riguarda la sola sotto-missione Opus: la sintesi generale Astra–Opus, il riesame delle proposte Neon e la decisione su S7 / A4 / O6 appartengono all'incarico successivo e non vengono anticipati qui.  
**Repository:** `EnricoBontempi/official_thesis`  
**Branch esecutore / commit consegna analizzato:** `opus/o5-eval-attempt2`, SHA `5373aa3c7633585edcee3d879b3c92bbb4748c11`  
**SHA evaluator O5 dichiarati e verificati:** `918286809d3cbb33d155b70a6e03a9b898bcab9e` (7 run: array `29728465` task `0, 2–7`) e `29b94345f0cac3f725662ad687a15ff841c7c730` (`O-B19-off-s0`, sostitutivo `29732630_1`; diff verificato: evaluator Python identico, modificati solo preflight/launcher)  
**SHA post-processing pacchetto finale (job `29771334`):** `258106bc6e0e6dc7a721be2645b5ba66472acc8b` (preceduto dal fix NaN/forward condivisi `e23f51bfac45f83e90d451fd1bfd178b2f85c540`)  
**Fonti ispezionate direttamente in Git:** `results/month2/opus_data_study/o5/{INDEX_FOR_NEON.md,FINAL_SUMMARY.md,LEDGER_O0_O6.md}`, `o5/package/{TABLES.md,package_manifest.json}`, tutte le 11 tabelle in `o5/package/tables/`, tutte le 12 figure PNG in `o5/package/figures/`, `o5_prep/loop_curves.csv`, `r0_selection/*`, `bridge_s5/*`, `o0_checks/S6_DELIVERY_CHECK.md`, `configs/experiments/opus_o5_matrix.json`, `roadmapopus.md`, `docs/month2/opus_overnight_handoff.md`, `docs/month2/05_sparse_reconstruction.md`, `docs/RESEARCH_LOG.md`, `src/combustion_recon/evaluation/{opus_o5_package.py,metrics.py}`.

> **Avvertenza sul perimetro e sulla natura delle evidenze:**  
> 1. **Sviluppo, non holdout indipendente:** le tre condizioni `VAL123` (`0NH3_1.05`, `20NH3_1.1`, `40NH3_0.9`) hanno guidato selezione `best`, scheduler `ReduceLROnPlateau` ed early-stop in tutti gli otto training. Nessun split `TEST` è stato caricato o usato.  
> 2. **Disegno dei seed e delle repliche:** **solo `A38` ha due seed di training (`seed 0` e `seed 1`)**; `B19` e `C149` hanno un solo seed (`seed 0`). Le tre repliche di valutazione (`0/1/2`) sono **tre maschere di velocità**, non tre training indipendenti. Due seed forniscono un metro descrittivo della variabilità, non una distribuzione stimata robustamente; non viene usato alcun test di significatività da pixel/frame/maschere né alcuna soglia automatica (es. «2× gap seed»).  
> 3. **Confrontabilità S5 vs Opus:** i due riferimenti `O-A38-on/off-s0` nel runner Opus costituiscono il braccio `A38 seed 0` del confronto dati; S5 storico resta separato per la diversa sequenza di campionamento (`DataLoader(shuffle=True)` vs `ScientificPresentationStream`).  
> 4. **Controlli diretti vs artefatti remoti:** ricalcoli sulle tabelle aggregate, verifiche SHA del manifest, audit codice/NaN e ispezione visiva di tutte le 12 figure sono stati eseguiti direttamente su `5373aa3`. I CSV per-frame remoti, i `COMPLETE.json`/`provenance.json`, gli `.npz` dei campi e i log Slurm su Engaging restano documentati dall'esecutore (Sonnet).

---

## Sintesi esecutiva

1. **Aumentare le istantanee supervisionate (`B19 → A38 → C149`, a parità di ricetta U-Net sparse e budget di 95 update/epoca scientifica) migliora nettamente sia `seen220` sia `VAL123`, riducendo soprattutto l'errore strutturale e l'overfitting.** Sul runner Opus (`seed 0`, 5%):
   - Su **`seen220`**, `C149` abbatte la RMSE rispetto ad `A38` in **20/20 condizioni** a `best_seen_diagnostic` (`p_on`: `50.85 → 39.31 K`, **`−11.54 K`**; `p_off`: `56.59 → 41.28 K`, **`−15.31 K`**) e a `last` (`p_on`: **`−15.12 K`**; `p_off`: **`−19.95 K`**). Al checkpoint `best VAL` il guadagno `C149 − A38` è **`−15.00 K`** (`20/20`) su `p_off` e **`−5.43 K`** (`11/20`) su `p_on`, dove le 9 condizioni più deboli dipendono dalla selezione precoce del `best VAL` di `O-C149-on-s0` all'epoca `90` (appena `23` passaggi del pool).
   - Su **`VAL123`**, `C149` migliora rispetto ad `A38` in **3/3 condizioni VAL** su entrambi i `p` e su tutti e tre i checkpoint (`best`: `p_on` `42.39 → 38.71 K` **`−3.68 K`**, `p_off` `42.92 → 39.82 K` **`−3.09 K`**; `last`: **`−5.60 K`** e **`−7.56 K`**). La riduzione riguarda in larga misura la componente strutturale (`centered-RMSE`: `−2.84` a `−7.36 K` su VAL; `−5.79` a `−16.55 K` su seen) e i **getti freddi (`Jet RMSE` VAL: `−12.5` a `−12.7 K` a `best`, `−21.6` a `−22.2 K` a `last`; `seen`: `−29` a `−42 K`)**.
   - **Tuttavia, più frame NON risolvono il bias di livello su `40NH3_0.9`:** su quella condizione VAL (area calda vicino alla soglia `2100 K`), `O-C149-on-s0` mantiene un bias medio di `−21.6` a `−25.9 K` (analogo a `−24.9 K` di `A38`), per cui l'IoU calda oscilla tra `0.787` (`best`) e `0.474`/`0.562` (`best_seen`/`last`), pur scendendo nella `centered-RMSE` a `19.4–20.2 K`.
2. **La variabilità fra `seed 0` e `seed 1` su `A38` è piccola nella struttura ma può essere molto grande nel livello termico su `VAL`.** Su `A38 p_off`, `seed 1 − seed 0` su `VAL` è quasi nullo (`+0.14 K` `best`, `−0.21 K` `best_seen`, `−0.31 K` `last`). Su `A38 p_on`, invece, lo scarto `VAL` raggiunge **`+5.83 K`** sia a `best` (`48.22` vs `42.39 K`) sia a `last` (`51.33` vs `45.50 K`) — ma nasce interamente sulla sola `40NH3_0.9` (`+20.79 K`, `Bias = −48.51 K`, `Hot IoU = 0.075`), mentre la `centered-RMSE` VAL di `seed 1` (`37.43 K`) è addirittura leggermente migliore di `seed 0` (`38.16 K`) e su `seen` i due seed differiscono solo di `+0.73 K`. Poiché `B19` e `C149` hanno un solo seed, questa variabilità di livello invita a non sovrainterpretare piccole differenze di RMSE `VAL` su un solo seed.
3. **La selezione del checkpoint pesa molto su `B19`/`A38` (overfitting severo dopo il minimo) e molto meno su `C149`.** Su `B19` e `A38`, `last − best` degrada `VAL` di `+3.11` a `+6.86 K` e `seen` di `+5.02` a `+8.76 K`; su `C149`, `last − best` su `VAL` scende a `+1.19` / `+2.10 K` e su `seen` la curva resta quasi piatta (`39.3 → 40.7 K` `p_on`, `41.3 → 42.9 K` `p_off`). La rianalisi `R0` su S5 conferma che il minimo assoluto sulle 3 `VAL` è ottimistico rispetto al leave-one-VAL-condition-out (`+1.19` a `+3.91 K`), pur lasciando invariato il vantaggio S5 di U-Net su GLU sotto `LOO` e `min_seen`.
4. **I parametri operativi `p` NON diventano più utili con più dati — anzi il loro margine su `seen` si riduce nettamente in `C149`.** Su `seen220`, il vantaggio `p_on − p_off` passa da `−8.29 K` (`B19 best`) e `−6.14 / −9.46 K` (`A38 best`) a soli `−1.97 / −2.12 K` su `C149` (`best_seen`/`last`; addirittura `+3.43 K` a `best` per selezione precoce di `C149-on`). Su `VAL123`, `p_on − p_off` è modesto e instabile di segno fra seed (`−2.81 K` `B19`, `−0.53 K` `A38 s0`, `+5.16 K` `A38 s1`, `−1.11 K` `C149 s0` a `best`). Con più dati, `p_off` recupera gran parte del gap di livello su `seen`, ma continua a degradare nettamente nella generalizzazione fuori densità al `20%` su `VAL` (`+9.05 K` `C149-off`, `+10.17 K` `A38-off s0`, contro stabilità/miglioramento di `p_on`).
5. **Audit tecnico superato, nessun errore bloccante; sotto-missione chiudibile.** Tutti i ricalcoli, i controlli di manifest (`21/21`), i forward condivisi, il confronto col loop (`72/72` entro `0.0011 K`), i NaN regionali (solo 3 frame di `train_fit` con regione `peak` vuota) e l'ispezione visiva di tutte le 12 figure (incluse le 2 `best_seen_diagnostic` non aperte da Sonnet) confermano la validità del pacchetto. Opus può essere chiuso nel perimetro concordato senza ulteriori run.

---

## 1. Contratto sperimentale, fonti e confrontabilità

### 1.1 Disegno congelato (otto run nel runner Opus)
Tutti gli otto training usano `SparseUNet` `b=32` (`4,903,041` parametri), inizializzati da zero, su griglia `128×128`, 2 componenti di velocità (`vi, vj`) al **5%** (`819` siti), `K=0` (nessuna `T` in ingresso), single-frame, precisione `fp32` con `AMP`/`TF32`/`compile` esplicitamente disabilitati, AdamW (`lr=1e-3`, `wd=1e-4`, clip `1.0`), batch effettivo `8`, warmup `950` update, `ReduceLROnPlateau` su `VAL` 5% (media 3 maschere persistite, eval ogni `10` epoche scientifiche), early-stop dopo minimo `200` epoche e `15` eval senza miglioramento relativo `0.1%` (reset pazienza al calo LR).

- **Unità di ottimizzazione comune:** 1 epoca scientifica = `760` presentazioni = `95` optimizer update in tutti i pool, tramite `ScientificPresentationStream` (permutazioni deterministiche senza rimpiazzo con stato persistito).
- **Pool nidificati (`B19 ⊂ A38 ⊂ C149`, 20 condizioni TRAIN):**
  - `B19`: 19 frame/condizione (`0…1440` passo 80) = `380` frame (`seed 0`, `p_on`/`p_off`) $\implies$ 2 permutazioni per epoca scientifica (stessa maschera TRAIN sulle due occorrenze dello stesso frame nella stessa epoca).
  - `A38`: 38 frame/condizione (`0…1480` passo 40) = `760` frame (`seed 0` e `seed 1`, `p_on`/`p_off`) $\implies$ 1 permutazione per epoca scientifica.
  - `C149`: 149 frame/condizione (`0…1480` passo 10) = `2980` frame (`seed 0`, `p_on`/`p_off`) $\implies$ 1 permutazione attraversa `3.92` epoche scientifiche. I frame passo 10 sono temporalmente correlati (autocorrelazione del fronte di fiamma `50–100` frame, audit Mese 1): aumentare di `3.92×` i frame non equivale a `3.92×` campioni statisticamente indipendenti.
- **Preprocessing congelato e ruoli di valutazione:** normalizzatore fit-only storico calcolato sui `760` frame `A38` anche per `B19` (esperimento sul numero di esempi supervisionati a preprocessing congelato, non pipeline fittata su soli `380` frame). Valutazione O5 su tre ruoli (`train_fit = reference760`, sempre gli indici `A38` anche per `B19` e `C149`; `seen_train_eval = seen220`, indici `1600…2000` passo 40; `val_eval = VAL123`, 3 condizioni escluse `0…2000` passo 50), tre coperture annidate (`1% / 5% / 20%`, dove `1%` e `20%` sono generalizzazione dalla densità TRAIN `5%`), tre repliche di maschera persistite (`0/1/2`, seed `20261116`), tre checkpoint (`best` VAL, `best_seen_diagnostic` scelto su `seen` replica 0, `last` all'early-stop).
- **Nota critica su `reference760` per `B19`:** per `B19`, `reference760` contiene 380 frame ottimizzati e 380 frame *mai visti in training* (gli indici dispari di `A38`), perciò non va mai letto come «errore sui soli frame di training B19».

### 1.2 Confrontabilità con S5 storico (Bridge O2)
Il bridge (`results/month2/opus_data_study/bridge_s5/LEDGER.md`) ha dimostrato che S5 aveva eseguito `DataLoader(shuffle=True)` (che estrae un intero `_base_seed` dal generatore prima della permutazione), mentre il runner Opus usa `ScientificPresentationStream` (`randperm` diretto): a parità di seed nominale, tutti i 95 batch di ogni epoca differiscono, cambiando la traiettoria SGD. Il replay di inferenza dei due `best.pt` U-Net S5 (`job 29670888`) è passato con scarto `0.0 K`, ma verifica la sola inferenza (`training_equivalence = not_tested`). Correttamente, la matrice O5 esclude S5 e introduce `O-A38-on/off-s0` nel runner Opus. A titolo puramente descrittivo separato, i due `A38 seed 0` Opus su `VAL best @5%` (`42.39` / `42.92 K`) risultano comunque molto vicini ai corrispondenti S5 storici (`42.96` / `43.53 K`, scarto `~0.6 K`).

---

## 2. Verifica critica della consegna tecnica O5 (e stato O0–O5)

### 2.1 Controlli indipendenti eseguiti direttamente in Git (`5373aa3`)
- **Integrità del pacchetto (`package_manifest.json`):** tutti i `21/21` file elencati nel manifest coincidono al bit via SHA-256 (`0` mismatch), incluso `compare_by_condition.csv.gz` verificato sul flusso decompresso, `configs/experiments/opus_o5_matrix.json` (`1f7fbb1505b5…`) e `o5_prep/loop_curves.csv` (`5c8ad6aaa300…`).
- **Identità dell'evaluator fra i due SHA di launcher:** il `git diff 918286809d3cbb33d155b70a6e03a9b898bcab9e 29b94345f0cac3f725662ad687a15ff841c7c730 --stat` mostra modifiche **esclusivamente** in `scripts/engaging/o5_preflight_paths.py`, `opus_o5_{eval,smoke}.sbatch`, `tests/test_opus_o5_preflight_paths.py` e `docs/month2/opus_overnight_handoff.md`: `scripts/26_opus_o5_eval.py`, `src/` e `configs/` sono identici al bit.
- **Ricalcoli indipendenti delle tabelle aggregate:**
  - `global_by_replica.csv → global_by_setting.csv` (14 metriche × 216 setting): scarto massimo assoluto **`5.68e-14 K`**.
  - `per_condition.csv → global_by_setting.csv`: scarto massimo assoluto **`5.68e-14 K`**.
  - `compare_summary.csv` vs `global_by_setting(A) − global_by_setting(B)` (`7560` righe): scarto massimo assoluto **`7.46e-14 K`**.
  - `loop_vs_o5_5pct.csv` ricalcolato contro `loop_curves.csv` e `global_by_replica.csv`: scarto di ricalcolo **`0.000e+00 K`**; `72/72` confronti entro `0.05 K` (scarto massimo `0.0003 K` su `train_fit` rep 0, `0.0005 K` su `seen` rep 0, `0.001146 K` su `VAL` media 3 repliche).
- **Forward condivisi (identità preservate):** nei tre run in cui `best` e `best_seen_diagnostic` coincidono alla stessa epoca (`O-B19-on-s0` ep. 50, `O-A38-on-s0` ep. 120, `O-A38-on-s1` ep. 60), `opus_o5_matrix.json` documenta `weights_sha256` identici ma `file_sha256` distinti, e in `global_by_setting.csv` tutte le 14 metriche su tutti i ruoli e coperture coincidono con scarto **`0.000e+00 K`**. In `setting_costs.csv`, questi 3 run eseguono `54` forward e `27` condivisioni (~`51–54 min` ciascuno), gli altri 5 eseguono `81` forward (`75–77 min`), tutti con memoria di picco `0.17 GB`.
- **Audit dei NaN regionali (punto critico 1):**
  - `undefined_regional_metric_frames.csv` contiene esattamente `72` righe (`8 run × 3 checkpoint × 3 coperture`), **tutte e sole** con `role = train_fit`, `metric = peak_rmse_K` e **`n_nan_frames = 9`** (cioè esattamente `3` frame × `3` repliche su `760 × 3 = 2280` righe per setting).
  - La verifica di `src/combustion_recon/evaluation/metrics.py` (righe 82–90, 119, 124) conferma che `peak_rmse_K` restituisce `NaN` se e solo se nessun pixel del frame vero soddisfa `T >= 2200 K` (regione `peak` vuota). Il numero identico (`9`) su tutti gli 8 run, 3 checkpoint e 3 coperture dimostra che l'indefinitezza è una proprietà pura dei **3 frame target** in `reference760`, non un difetto del modello o della valutazione.
  - **Zero `NaN`** su `seen_train_eval`, **zero `NaN`** su `val_eval`, **zero `NaN`** in tutte le metriche globali (`rmse_K`, `mae_K`, `mean_signed_error_K`, `abs_bias_K`, `centered_rmse_K`), e **zero `NaN`** in `per_condition.csv`, `regional_by_condition.csv`, `global_by_setting.csv` e `compare_summary.csv`: i 3 frame vuoti in `train_fit` appartengono a condizioni che possiedono comunque altri frame con `T >= 2200 K`, quindi le medie per condizione sono tutte definite (`skipna` sui soli 3 frame vuoti) e i denominatori di condizione sono esatti (`20/20` su `train_fit` e `seen`, `3/3` su `VAL`).
- **Ispezione visiva diretta delle figure (punto critico 3):**
  - Sono state aperte e ispezionate **tutte le 12 figure PNG** in `o5/package/figures/`, **incluse le due figure `best_seen_diagnostic` non aperte singolarmente da Sonnet** (`o5_regional_5pct_best_seen_diagnostic.png` e `val_frame1000_best_seen_diagnostic.png`): entrambe sono complete, leggibili, con legende/barre colore corrette e pienamente coerenti con i CSV (le etichette `T (K)` / `signed error (K)` in basso restano vicine al bordo ma leggibili; nessuna rigenerazione richiesta).
  - **Cautela confermata sulle scale di errore nei campi:** le scale di temperatura sono comuni fra tutti i pannelli (`[208, 2652] K`), mentre le scale dell'errore firmato sono comuni *dentro ciascuna figura* ma diverse fra checkpoint: **`±707 K`** per `val_frame1000_best.png`, **`±711 K`** per `val_frame1000_best_seen_diagnostic.png`, **`±853 K`** per `val_frame1000_last.png`. I colori d'errore non vanno quindi confrontati a occhio tra `best` e `last` senza tener conto del range diverso.
- **Rilievo tecnico minore (non bloccante):** le 3 figure dei campi (`figures/val_frame1000_{best,best_seen_diagnostic,last}.png`) sono committate in `o5/package/figures/` ma **assenti da `package_manifest.json`** (che elenca `21` file perché viene scritto da `scripts/29_opus_o5_package.py` prima che `scripts/28_opus_o5_figures.py` produca i 3 pannelli in `scripts/30_opus_o5_finalize.py`). Nessun impatto sui dati; da annotare come limite minore di tracciabilità del manifest.

### 2.2 Elementi documentati dall'esecutore (non ispezionabili direttamente in Git)
Restano verificati dall'esecutore sul cluster Engaging (non presenti in Git): i `8 × 81 = 648` CSV per-frame grezzi e i `24` file `.npz` in `/home/bontempi/official_thesis_opus_o5_attempt2_runs/`, i file `COMPLETE.json` e `provenance.json`, la verifica riga-per-riga dei `mask_hash` (3309 istanze) eseguita da `scripts/30_opus_o5_finalize.py` (job `29771334`), nonché i log Slurm e gli output grezzi degli 8 training (`29612010–13`, `29645053–54`, `29671719/21`) e del replay `29670888`. La lettura diretta di `opus_o5_package.py` conferma comunque che il gate finale `scripts/30` rifiuta l'aggregazione (`BLOCK_REPORT.md`, exit 3) in caso di qualsiasi difformità su `COMPLETE.json`, hash di checkpoint/cache/stats/maschere, set di frame o uguaglianza dei forward condivisi (`equal_nan=True`).

---

## 3. Tabelle essenziali dei risultati O5 (5%, per condizione e coperture 1/5/20%)

Fonti: `global_by_setting.csv`, `per_condition.csv`, `compare_summary.csv`, `o5_prep/loop_curves.csv`. Tutti i valori di errore sono in **K** (media sulle condizioni e sulle 3 repliche di maschera).

### 3.1 Prestazioni globali e regionali al 5% (`VAL123`, `seen220`, `reference760`)

#### A) `val_eval` (`VAL123`, 3 condizioni × 41 frame, 3 repliche di maschera)

| Run | Pool / `p` / seed | Ckpt | Ep. | RMSE `[min,max]` | MAE | Bias | `\|Bias\|` | Cent. | Hot IoU | Hot RMSE | Peak RMSE | Jet RMSE | NonHot RMSE |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `O-B19-on-s0` | B19 / on / s0 | `best` (`=b_seen`) | 50 | **43.93** `[43.89,44.01]` | 24.08 | −6.49 | 15.50 | 38.14 | 0.746 | 22.4 | 28.7 | 129.7 | 107.2 |
| | | `last` | 620 | 50.25 `[50.10,50.36]` | 24.63 | −7.13 | 14.62 | 45.07 | 0.757 | 22.2 | 24.6 | 152.9 | 128.4 |
| `O-B19-off-s0` | B19 / off / s0 | `best` | 60 | **46.74** `[46.44,47.19]` | 29.47 | −13.59 | 19.14 | 40.98 | 0.872 | 28.6 | 38.4 | 130.7 | 108.6 |
| | | `best_seen` | 40 | 48.20 `[47.80,48.82]` | 30.92 | −16.22 | 21.53 | 41.16 | 0.891 | 29.4 | 43.9 | 130.6 | 110.9 |
| | | `last` | 630 | 53.60 `[53.45,53.69]` | 31.63 | −8.81 | 21.46 | 46.98 | 0.888 | 29.2 | 32.5 | 153.3 | 129.4 |
| `O-A38-on-s0` | A38 / on / s0 | `best` (`=b_seen`) | 120 | **42.39** `[42.29,42.53]` | 22.20 | −4.07 | 13.13 | 38.16 | 0.823 | 20.2 | 22.6 | 127.5 | 106.7 |
| | | `last` | 630 | 45.50 `[45.48,45.52]` | 21.64 | −3.20 | 12.22 | 41.96 | 0.892 | 19.0 | 21.9 | 143.7 | 119.3 |
| `O-A38-off-s0` | A38 / off / s0 | `best` | 90 | **42.92** `[42.73,43.11]` | 27.33 | +1.08 | 17.87 | 37.24 | **0.931** | 24.9 | 28.2 | 124.5 | 101.1 |
| | | `best_seen` | 130 | 44.71 `[44.57,44.82]` | 28.55 | −8.53 | 19.46 | 38.49 | 0.895 | 26.8 | 31.2 | 125.0 | 104.5 |
| | | `last` | 660 | 49.48 `[49.34,49.66]` | 30.66 | −9.76 | 21.18 | 43.06 | 0.873 | 28.1 | 31.8 | 141.9 | 118.5 |
| `O-A38-on-s1` | A38 / on / s1 | `best` (`=b_seen`) | 60 | **48.22** `[48.00,48.39]` | 31.43 | −11.16 | 22.72 | 37.43 | 0.681 | 31.0 | 38.1 | 122.1 | 104.9 |
| | | `last` | 630 | 51.33 `[51.21,51.39]` | 27.28 | −8.01 | 18.71 | 43.69 | 0.708 | 25.0 | 25.2 | 149.5 | 126.2 |
| `O-A38-off-s1` | A38 / off / s1 | `best` | 40 | **43.06** `[42.59,43.70]` | 25.75 | −1.86 | 12.89 | 39.92 | 0.926 | 24.1 | 37.9 | 126.3 | 104.0 |
| | | `best_seen` | 110 | 44.50 `[44.23,44.91]` | 28.73 | −12.09 | 18.89 | 38.12 | 0.825 | 27.1 | 33.8 | 122.5 | 102.8 |
| | | `last` | 610 | 49.17 `[48.90,49.68]` | 30.04 | −11.41 | 20.59 | 42.63 | 0.828 | 27.5 | 31.8 | 140.0 | 117.4 |
| `O-C149-on-s0` | C149 / on / s0 | `best` | 90 | **38.71** `[38.30,39.01]` | 21.78 | −6.24 | 13.02 | 35.32 | 0.919 | 20.3 | 27.7 | 114.8 | 95.7 |
| | | `best_seen` | 220 | 38.97 `[38.62,39.16]` | 21.51 | −3.93 | 13.81 | **33.86** | 0.816 | 20.1 | 22.6 | 116.9 | 96.3 |
| | | `last` | 660 | 39.90 `[39.71,40.03]` | **21.00** | −3.91 | 13.08 | 35.21 | 0.845 | **19.4** | **21.1** | 122.1 | 100.9 |
| `O-C149-off-s0` | C149 / off / s0 | `best` | 260 | **39.82** `[39.03,40.38]` | 25.61 | −7.42 | 16.72 | **33.91** | 0.873 | 24.3 | 27.9 | **111.9** | **92.1** |
| | | `best_seen` | 240 | 40.52 `[39.77,41.08]` | 26.35 | −6.13 | 17.58 | 34.26 | 0.855 | 24.9 | 28.5 | 114.4 | 93.1 |
| | | `last` | 710 | 41.92 `[41.27,42.26]` | 26.54 | −4.84 | 17.88 | 35.70 | 0.866 | 25.0 | 27.7 | 119.7 | 98.2 |

#### B) `seen_train_eval` (`seen220`, 20 condizioni × 11 frame) e `reference760` (`train_fit`)

| Run | Ckpt | Ep. | `seen` RMSE | `seen` Bias / `\|Bias\|` | `seen` Cent. | `seen` Hot IoU | `seen` Hot / Peak RMSE | `seen` Jet RMSE | `seen` RelAreaErr | `ref760` RMSE *(nota B19)* |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `O-B19-on-s0` | `best` (`=b_seen`) | 50 | 57.56 | +3.27 / 7.11 | 56.90 | 0.771 | 26.2 / 61.6 | 146.3 | 0.150 | 41.43 *(380 opt + 380 non-opt)* |
| | `last` | 620 | 66.32 | +1.99 / 6.51 | 65.84 | 0.772 | 27.9 / 63.8 | 170.0 | 0.114 | 35.13 |
| `O-B19-off-s0` | `best` | 60 | 65.85 | +3.17 / 20.57 | 60.35 | 0.691 | 32.9 / 69.5 | 154.3 | 2.552 | 44.78 |
| | `best_seen` / `last` | 40 / 630 | 65.37 / 73.95 | +1.63 / 20.01 (`bs`) | 60.08 / 68.58 | 0.698 / 0.689 | 32.1 / 72.7 (`bs`) | 156.7 / 174.7 | 3.078 / 1.830 | 48.29 / 39.02 |
| `O-A38-on-s0` | `best` (`=b_seen`) | 120 | 50.85 | +2.02 / 5.62 | 50.38 | 0.792 | 22.4 / 52.0 | 127.8 | 0.119 | 16.26 |
| | `last` | 630 | 55.87 | +0.98 / 5.46 | 55.46 | 0.783 | 24.0 / 58.0 | 141.6 | 0.097 | 7.16 |
| `O-A38-off-s0` | `best` | 90 | 56.99 | +6.96 / 17.50 | 52.24 | 0.701 | 27.4 / 58.1 | 136.5 | 1.994 | 26.46 |
| | `best_seen` / `last` | 130 / 660 | 56.59 / 62.83 | +2.12 / 16.87 (`bs`) | 51.97 / 57.90 | 0.696 / 0.683 | 28.3 / 58.8 (`bs`) | 132.3 / 146.6 | 2.004 / 1.689 | 19.67 / 7.84 |
| `O-A38-on-s1` | `best` (`=b_seen`) | 60 | 51.58 | −1.98 / 6.66 | 50.82 | 0.765 | 28.7 / 68.8 | 125.3 | 0.179 | 30.59 |
| | `last` | 630 | 56.84 | +0.97 / 5.67 | 56.40 | 0.787 | 24.9 / 57.5 | 143.9 | 0.111 | 7.07 |
| `O-A38-off-s1` | `best` | 40 | 61.04 | +5.59 / 18.12 | 56.59 | 0.671 | 33.2 / 74.5 | 144.5 | 1.433 | 44.81 |
| | `best_seen` / `last` | 110 / 610 | 54.83 / 60.98 | −1.58 / 15.10 (`bs`) | 50.92 / 56.89 | 0.708 / 0.707 | 27.2 / 53.8 (`bs`) | 127.7 / 145.4 | 0.756 / 1.044 | 20.03 / 8.79 |
| `O-C149-on-s0` | `best` | 90 | 45.42 | +1.67 / 6.74 | 44.59 | 0.770 | 23.1 / 56.4 | 114.5 | 0.169 | 38.24 |
| | `best_seen` | 220 | **39.31** | +1.29 / 4.09 | **38.96** | **0.804** | **19.1 / 44.5** | **98.7** | 0.114 | 20.43 |
| | `last` | 660 | 40.75 | +0.30 / **3.78** | 40.46 | 0.803 | 19.4 / **44.3** | 102.2 | 0.111 | 15.07 |
| `O-C149-off-s0` | `best` | 260 | 41.99 | −0.99 / 8.90 | 40.39 | 0.756 | 21.2 / 46.2 | 102.7 | 0.554 | 20.26 |
| | `best_seen` | 240 | 41.28 | +1.24 / 8.42 | 39.79 | 0.762 | 20.5 / 44.8 | 100.7 | 0.633 | 20.61 |
| | `last` | 710 | 42.88 | +1.09 / 8.64 | 41.34 | 0.760 | 20.4 / 44.8 | 104.4 | 0.629 | 14.95 |

### 3.2 Delta appaiati al 5% (`A − B`, negativi = `A` migliore in errore; IoU: positivi = `A` migliore)

| Famiglia | Contrasto (`A − B`) | Ckpt | `VAL` ΔRMSE (`n_low/hi`) | `VAL` ΔCent. | `VAL` Δ`\|Bias\|` | `VAL` ΔJet | `VAL` ΔIoU | `seen` ΔRMSE (`n_low/hi`) | `seen` ΔCent. | `seen` ΔJet |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| **Dati** | `B19 − A38` (`s0, p_on`) | `best` / `last` | +1.54 (2/1) / +4.75 (0/3) | −0.02 / +3.11 | +2.38 / +2.40 | +2.1 / +9.2 | −0.077 / −0.135 | +6.71 (1/19) / +10.45 (0/20) | +6.52 / +10.38 | +18.5 / +28.4 |
| | `B19 − A38` (`s0, p_off`) | `best` / `last` | +3.82 (1/2) / +4.12 (0/3) | +3.74 / +3.92 | +1.27 / +0.29 | +6.2 / +11.4 | −0.059 / +0.016 | +8.86 (2/18) / +11.12 (1/19) | +8.10 / +10.68 | +17.9 / +28.1 |
| | **`C149 − A38` (`s0, p_on`)** | `best` | **−3.68 (3/0)** | −2.84 | −0.11 | −12.7 | +0.096 | −5.43 (11/9) | −5.79 | −13.3 |
| | | `best_seen` | **−3.42 (3/0)** | −4.30 | +0.68 | −10.6 | −0.007 | **−11.54 (20/0)** | −11.43 | −29.1 |
| | | `last` | **−5.60 (3/0)** | −6.76 | +0.85 | −21.6 | −0.047 | **−15.12 (20/0)** | −15.00 | −39.4 |
| | **`C149 − A38` (`s0, p_off`)** | `best` | **−3.09 (3/0)** | −3.33 | −1.15 | −12.5 | −0.058 | **−15.00 (20/0)** | −11.86 | −33.8 |
| | | `best_seen` | **−4.19 (3/0)** | −4.23 | −1.87 | −10.6 | −0.040 | **−15.31 (20/0)** | −12.18 | −31.6 |
| | | `last` | **−7.56 (3/0)** | −7.36 | −3.29 | −22.2 | −0.006 | **−19.95 (20/0)** | −16.55 | −42.2 |
| | `C149 − B19` (`s0, p_on`) | `best` / `last` | −5.22 (3/0) / −10.35 (3/0) | −2.82 / −9.87 | −2.48 / −1.54 | −14.9 / −30.8 | +0.173 / +0.089 | −12.14 (16/4) / −25.57 (20/0) | −12.31 / −25.38 | −31.8 / −67.7 |
| | `C149 − B19` (`s0, p_off`) | `best` / `last` | −6.91 (3/0) / −11.68 (3/0) | −7.07 / −11.28 | −2.42 / −3.58 | −18.8 / −33.6 | +0.001 / −0.022 | −23.86 (20/0) / −31.07 (20/0) | −19.96 / −27.24 | −51.7 / −70.3 |
| **Seed** | `A38 s1 − s0` (`p_on`) | `best` / `last` | **+5.83 (2/1)** / +5.83 (0/3) | **−0.73** / +1.72 | **+9.59** / +6.48 | −5.4 / +5.9 | −0.142 / −0.184 | +0.73 (6/14) / +0.97 (4/16) | +0.43 / +0.93 | −2.5 / +2.3 |
| | `A38 s1 − s0` (`p_off`) | `best` / `b_seen` / `last` | +0.14 / −0.21 / −0.31 | +2.68 / −0.36 / −0.43 | −4.98 / −0.57 / −0.59 | +1.8 / −2.6 / −1.9 | −0.004 / −0.069 / −0.045 | +4.05 / −1.76 / −1.85 | +4.34 / −1.05 / −1.01 | +8.0 / −4.6 / −1.2 |
| **Param. `p`** | `B19 s0` (`on − off`) | `best` / `last` | −2.81 (1/2) / −3.35 (2/1) | −2.84 / −1.91 | −3.64 / −6.84 | −1.1 / −0.4 | −0.126 / −0.132 | −8.29 (20/0) / −7.63 (17/3) | −3.45 / −2.74 | −8.0 / −4.7 |
| | `A38 s0` (`on − off`) | `best` / `last` | −0.53 (2/1) / −3.98 (2/1) | +0.92 / −1.09 | −4.75 / −8.95 | +3.1 / +1.8 | −0.107 / +0.019 | −6.14 (17/3) / −6.96 (19/1) | −1.86 / −2.44 | −8.7 / −5.0 |
| | `A38 s1` (`on − off`) | `best` / `last` | **+5.16 (1/2)** / +2.16 (1/2) | −2.49 / +1.06 | +9.83 / −1.88 | −4.2 / +9.5 | −0.245 / −0.119 | −9.46 (20/0) / −4.14 (17/3) | −5.77 / −0.49 | −19.2 / −1.5 |
| | `C149 s0` (`on − off`) | `best` / `b_seen` / `last` | −1.11 / −1.55 / −2.02 (2/1) | +1.41 / −0.40 / −0.49 | −3.70 / −3.78 / −4.81 | +2.8 / +2.5 / +2.4 | +0.046 / −0.039 / −0.021 | +3.43 (5/15) / **−1.97 (17/3)** / **−2.12 (16/4)** | +4.20 / −0.83 / −0.88 | +11.9 / −1.9 / −2.1 |

### 3.3 Dettaglio per condizione su `val_eval` al 5% (`best` e `last`)

| Run | `0NH3_1.05` (`best` RMSE / Bias / Cent / Jet) | `20NH3_1.1` (`best` RMSE / Bias / Cent / Jet) | `40NH3_0.9` (`best` RMSE / Bias / Cent / IoU / AreaP vs 0.785) | `40NH3_0.9` (`last` RMSE / Bias / Cent / IoU) |
| --- | --- | --- | --- | --- |
| `O-B19-on-s0` | 45.07 / +2.29 / 44.84 / 172.0 | 45.01 / +10.62 / 43.67 / 146.1 | 41.71 / **−32.40** / 25.91 / **0.269** / 0.212 | 40.52 / −30.54 / 26.13 / 0.305 |
| `O-B19-off-s0` | 57.51 / −23.49 / 51.41 / 184.4 | 44.02 / −7.08 / 40.46 / 123.5 | 38.68 / −10.19 / 31.08 / 0.662 / 0.543 | 42.38 / +1.01 / 32.77 / 0.707 |
| `O-A38-on-s0` | 45.37 / +0.58 / 45.28 / 169.7 | 47.16 / +12.15 / 45.50 / 154.2 | 34.64 / **−24.92** / 23.70 / **0.499** / 0.392 | 32.82 / −22.22 / 23.73 / 0.706 |
| `O-A38-off-s0` | 47.71 / −17.24 / 43.62 / 161.7 | 41.84 / +9.02 / 39.52 / 124.9 | 39.20 / +11.47 / 28.59 / 0.824 / 0.711 | 38.33 / −5.65 / 29.03 / 0.657 |
| `O-A38-on-s1` | 44.24 / −1.71 / 44.09 / 161.3 | 44.99 / +16.74 / 41.67 / 138.8 | **55.43** / **−48.51** / 26.53 / **0.075** / 0.059 | 45.09 / −37.52 / 24.42 / 0.159 |
| `O-A38-off-s1` | 47.90 / −12.83 / 45.37 / 165.9 | 42.92 / +3.79 / 41.92 / 123.4 | 38.34 / +3.47 / 32.47 / 0.814 / 0.691 | 40.43 / −16.94 / 29.16 / 0.518 |
| **`O-C149-on-s0`** | **43.59** / −7.20 / 42.73 / **159.2** | **38.48** / +10.09 / 37.07 / 119.6 | **34.07** / **−21.62** / 26.15 / **0.787** / 0.620 | **31.37** / −24.31 / **19.42** / **0.562** *(b_seen 0.474)* |
| **`O-C149-off-s0`** | 47.09 / −19.10 / **42.42** / 162.7 | **37.52** / +6.45 / **36.47** / **112.2** | 34.86 / −9.62 / **22.85** / 0.648 / 0.541 | 36.04 / −7.57 / 23.15 / 0.629 |

### 3.4 Esposizione al pool e confronto fino all'orizzonte comune (`epoch ≤ 610`, da `loop_curves.csv`)
Ogni run visita tutti i propri frame distinti (`380`, `760`, `2980`) entro le prime `1–4` epoche scientifiche. Gli arresti variano fra epoca `610` (`57,950` update, `O-A38-off-s1`) e epoca `710` (`67,450` update, `O-C149-off-s0`). Non esistono checkpoint intermedi salvati all'epoca comune `610`, ma `loop_curves.csv` permette di confrontare le curve esattamente fino a `epoch = 610` senza inventare pesi:

| Run | Ep. arresto (update / passaggi pool) | Ep. `best` (passaggi) | Min VAL `ep ≤ 610` | Min `seen` `ep ≤ 610` | `@ep 610`: `VAL` / `seen` / `ref760` | `@~100 passaggi`: ep. / `VAL` / `seen` |
| --- | --- | --- | ---: | ---: | --- | --- |
| `O-B19-on-s0` | 620 (58.9k / 1240×) | 50 (100×) | 43.93 (ep 50) | 57.52 (ep 50) | 50.22 / 66.39 / 35.08 | ep 50 / 43.93 / 57.52 |
| `O-B19-off-s0` | 630 (59.9k / 1260×) | 60 (120×) | 46.74 (ep 60) | 65.47 (ep 40) | 53.34 / 73.89 / 39.01 | ep 50 / 47.32 / 66.47 |
| `O-A38-on-s0` | 630 (59.9k / 630×) | 120 (120×) | 42.39 (ep 120) | 50.88 (ep 120) | 45.52 / 55.97 / 7.19 | ep 100 / 44.62 / 51.26 |
| `O-A38-off-s0` | 660 (62.7k / 660×) | 90 (90×) | 42.91 (ep 90) | 56.75 (ep 130) | 49.46 / 62.89 / 7.88 | ep 100 / 45.58 / 57.52 |
| `O-A38-on-s1` | 630 (59.9k / 630×) | 60 (60×) | 48.22 (ep 60) | 51.36 (ep 60) | 51.28 / 56.87 / 7.09 | — |
| `O-A38-off-s1` | 610 (58.0k / 610×) | 40 (40×) | 43.06 (ep 40) | 54.88 (ep 110) | 49.17 / 60.84 / 8.79 | — |
| `O-C149-on-s0` | 660 (62.7k / 168×) | 90 (23×) | **38.71** (ep 90) | **39.17** (ep 220) | **39.89 / 40.58** / 15.13 | ep 390 / **40.23 / 40.29** |
| `O-C149-off-s0` | 710 (67.5k / 181×) | 260 (66×) | **39.82** (ep 260) | **41.21** (ep 240) | **41.84 / 42.69** / 15.08 | ep 390 / **41.63 / 42.25** |

Tutti i minimi (`best` e `best_seen`) avvengono ampiamente prima dell'epoca comune `610`, e `C149` supera `A38` e `B19` **sia a parità di update** (es. `ep 610`), **sia a parità di passaggi sul pool** (es. ~`50`, `100`, `168` passaggi).

### 3.5 Generalizzazione fuori densità di training: coperture `1% / 5% / 20%` (modelli allenati solo al `5%`)

| Run | `VAL123` `best`: `1% / 5% / 20%` (RMSE K) | `VAL123` `last`: `1% / 5% / 20%` | `seen220` `best`: `1% / 5% / 20%` | `seen220` `last`: `1% / 5% / 20%` |
| --- | --- | --- | --- | --- |
| `O-B19-on-s0` | 74.47 / 43.93 / 41.00 | 64.08 / 50.25 / 45.06 | 94.91 / 57.56 / 63.70 | 81.52 / 66.32 / 68.54 |
| `O-B19-off-s0` | 76.21 / 46.74 / **49.53** | 71.29 / 53.60 / **54.31** | 106.75 / 65.85 / 69.77 | 99.72 / 73.95 / 75.10 |
| `O-A38-on-s0` | 64.67 / 42.39 / 41.20 | 59.52 / 45.50 / 42.11 | 80.98 / 50.85 / 55.10 | 75.15 / 55.87 / 58.30 |
| `O-A38-off-s0` | 78.90 / 42.92 / **53.09** | 72.44 / 49.48 / **55.63** | 106.91 / 56.99 / 62.66 | 95.21 / 62.83 / 65.42 |
| `O-A38-on-s1` | 80.82 / 48.22 / 39.82 | 62.80 / 51.33 / 46.40 | 96.09 / 51.58 / 56.77 | 78.11 / 56.84 / 59.59 |
| `O-A38-off-s1` | 89.47 / 43.06 / **48.95** | 68.92 / 49.17 / **51.22** | 122.23 / 61.04 / 64.72 | 95.00 / 60.98 / 63.46 |
| `O-C149-on-s0` | 73.09 / **38.71** / 40.01 *(b_seen: 61.57 / 38.97 / 37.93)* | 61.28 / **39.90 / 38.28** | 94.27 / 45.42 / 52.04 *(b_seen: 78.30 / 39.31 / 45.78)* | 75.50 / **40.75 / 46.21** |
| `O-C149-off-s0` | 74.09 / **39.82 / 48.87** | 69.61 / **41.92 / 48.77** | 93.72 / **41.99 / 45.14** | 90.33 / **42.88 / 45.76** |

*Lettura:* (1) Al `1%` la RMSE sale a `60–90 K` su `VAL` e `75–122 K` su `seen`; i checkpoint più allenati (`last`/`best_seen`) reggono paradossalmente meglio del `best` precoce al `1%`, e `p_on` supera sistematicamente `p_off` al `1%`. (2) Al `20%` su `VAL`, tutti e quattro i run `p_off` **peggiorano nettamente** rispetto al `5%` (`+2.8` a `+10.2 K` al `best`), confermando la fragilità del modello `p_off` allenato a singola densità `5%`, mentre i `p_on` restano stabili o migliorano leggermente (recuperando persino `Hot IoU = 0.966–0.976` su `VAL`). Su `seen`, invece, tutti gli 8 run peggiorano al `20%` rispetto al `5%` (`+3` a `+7 K`).

---

## 4. Risposte esplicite alle cinque domande di `roadmapopus.md` §2

### Domanda 1 — Più frame aiutano a ricostruire nuove istantanee (`seen` e `VAL`)?
- **Su `seen220` (nuove istantanee delle 20 condizioni TRAIN):** **sì, in modo netto e monotono** (`B19 → A38 → C149`). Passare da `38` a `149` frame/condizione riduce la RMSE di **`−11.54` a `−15.12 K`** su `p_on` (`best_seen`/`last`, `20/20` condizioni) e di **`−15.00` a `−19.95 K`** su `p_off` (`20/20` condizioni su tutti i checkpoint), e sopprime quasi del tutto l'overfitting temporale osservato su `B19` e `A38` (dove `reference760` crolla a `7–8 K` mentre `seen` risale di `+5`–`+9 K`). Nel solo caso `C149 − A38 p_on best` (`−5.43 K`, `11/20`), le 9 condizioni più deboli (`+0.6` a `+7.4 K`, tutte fiamme stabili/a basso errore come `0NH3_1`, `0NH3_1.15`) derivano dal fatto che `best VAL` arresta la scelta di `O-C149-on-s0` all'epoca `90` (`23` passaggi), mentre le condizioni ricche/ad alta ammoniaca (`20NH3_1.25 −22.5 K`, `80NH3_1.15 −16.7 K`, `40NH3_1.2 −14.8 K`) migliorano già nettamente; a epoca `220` (`best_seen`) il guadagno diventa `20/20`.
- **Su `VAL123` (tre condizioni escluse dal fit):** **sì, ma con guadagno più moderato e ancora condizionato a 1 seed su C149/B19 e 3 sole VAL.** `C149` supera `A38` in **3/3 condizioni VAL** su entrambi i `p` e tutti i checkpoint (`−3.68 K` `p_on best`, `−3.09 K` `p_off best`; `−5.60` / `−7.56 K` `last`), con riduzione genuina della `centered-RMSE` (`−2.84` a `−7.36 K`) e dei getti (`−12.5` a `−22.2 K`). Restano tuttavia aperte la generalizzazione a vero `TEST`/estrapolazione e la sensibilità a un secondo seed su `C149`.

### Domanda 2 — Il ranking attuale dipende dalla variabilità del training (`A38 seed 0 vs seed 1`)?
- **Sì, in modo forte per il livello termico di `p_on` su `VAL`, mentre è molto più contenuto per la struttura e per `p_off`.**
- Su `A38 p_off`, i due seed sono vicinissimi su `VAL` (`+0.14 K` `best`, `−0.21 K` `best_seen`, `−0.31 K` `last`); su `seen` lo scarto a `best` (`+4.05 K`) dipende dal `best` precoce di `seed 1` (ep. 40 vs 90) e si riduce a `−1.76 / −1.85 K` a `best_seen`/`last`.
- Su `A38 p_on`, lo scarto `seed 1 − seed 0` su `VAL` raggiunge **`+5.83 K`** (`48.22` vs `42.39 K` `best`), ribaltando completamente il confronto `p_on vs p_off` su `VAL` (`−0.53 K` a `seed 0` vs **`+5.16 K`** a `seed 1`!). Ma la decomposizione per condizione mostra che questo salto nasce **interamente da un forte bias negativo di livello su `40NH3_0.9`** (`−48.51 K` vs `−24.92 K`, `RMSE 55.43` vs `34.64 K`, `Hot IoU 0.075` vs `0.499`), mentre la `centered-RMSE` `VAL` (`37.43` vs `38.16 K`) e `seen` (`51.58` vs `50.85 K`) sono quasi sovrapponibili tra i due seed.
- **Implicazione:** un singolo seed su `3` condizioni VAL può subire oscillazioni di livello di `~5–6 K` macro (e `>20 K` su una singola condizione). Ciò non invalida il segnale strutturale di `C149` (che migliora `centered-RMSE` e `Jet RMSE` in `3/3` VAL e abbatte `seen` di `11–20 K`, molto sopra il gap seed `0.7–1.9 K` su `seen`), ma impone prudenza nel proclamare gerarchie sottili su `VAL` da un solo seed.

### Domanda 3 — Quanto conta il minimo scelto su `VAL` (`best`, `best_seen`, `last`, rianalisi `R0`)?
- **Molto su `B19`/`A38`, sensibilmente meno su `C149`.**
  - Su `B19` e `A38`, `last` peggiora `VAL` di `+3.11`–`+6.86 K` e `seen` di `+5.02`–`+8.76 K` rispetto a `best` (overfitting marcato). Su `C149`, l'overfitting si attenua drasticamente (`last − best` su `VAL`: `+1.19 K` `p_on`, `+2.10 K` `p_off`; su `seen`, `O-C149-on-s0` addirittura migliora di `−4.67 K` rispetto al `best` precoce di ep. 90).
  - La selezione diagnostica `best_seen_diagnostic` coincide con `best` in 3 run (`B19-on-s0`, `A38-on-s0`, `A38-on-s1`); nei 5 run in cui differisce, migliora `seen` (fino a `−6.11 K` su `C149-on`, `−6.21 K` su `A38-off-s1`) pagando `+0.26`–`+1.79 K` su `VAL`.
- **Rianalisi `R0` sui quattro S5 (`r0_selection/r0_selections.csv`):**

| Run S5 storico | `best VAL` (ep.) | `LOO` media held-out (epoche 3 fold) | `min_seen` VAL (ep.) | `causal_ma5` VAL (ep.) | `last` VAL (ep.) |
| --- | --- | --- | --- | --- | --- |
| `s5_sparse_unet_pon_seed0` | 42.96 (60) | **44.15** `[60, 60, 70]` (+1.19 K) | 43.20 (70) | 46.72 (90) | 49.50 (630) |
| `s5_sparse_unet_poff_seed0` | 43.53 (140) | **46.22** `[150, 140, 70]` (+2.69 K) | 45.11 (120) | 43.53 (140) | 47.67 (650) |
| `s5_adapted_glu_pon_seed0` | 45.80 (440) | **49.71** `[480, 440, 360]` (+3.91 K) | 46.94 (660) | 45.97 (480) | 46.84 (890) |
| `s5_adapted_glu_poff_seed0` | 47.54 (650) | **48.99** `[510, 850, 470]` (+1.45 K) | 48.29 (930) | 48.04 (870) | 47.87 (980) |

  - `R0` quantifica l'ottimismo della selezione diretta sul minimo assoluto di 3 sole `VAL` (`+1.2` a `+3.9 K` di divario verso il `LOO` held-out, pur contaminato da scheduler/early-stop). Sotto `causal_ma5`, `U-Net p_on` scivola da epoca `60` a epoca `90` e perde `3.76 K` (da `42.96` a `46.72 K`, guidati dal salto `39.92 → 49.75 K` su `40NH3_0.9`), invertendo l'ordine rispetto a `U-Net p_off` e `GLU p_on` — ulteriore prova che il minimo `best VAL` di `U-Net p_on` sui dataset ridotti (`A38`) è un punto stretto e sensibile al rumore della traiettoria.

### Domanda 4 — `p` diventa più utile con più dati?
- **No, nel perimetro misurato l'evidenza va in direzione opposta su `seen` e resta modesta/non conclusiva su `VAL`.**
- Su `seen220`, il vantaggio di `p_on` rispetto a `p_off` si **restringe** all'aumentare dei dati: da `−8.29 K` (`B19 best`) e `−6.14 / −9.46 K` (`A38 best`) a **`−1.97 K`** (`C149 best_seen`) e **`−2.12 K`** (`C149 last`) — e a `best` vale persino `+3.43 K` per il mismatch d'epoca (`90` vs `260`). Con 149 istantanee per condizione, `p_off` impara molto meglio il regime termico dalle sole velocità (`|bias|` `seen` dimezzato da `17.5–20.6 K` a `8.4–8.9 K`).
- Su `VAL123`, `p_on − p_off` resta piccolo e dipendente dal seed (`−2.81 K` `B19`, `−0.53 K` `A38 s0`, `+5.16 K` `A38 s1`, `−1.11 K` `C149 s0` a `best`). Con un solo seed su `C149` non si può affermare che `p` guadagni forza in `C149`; dove `p_on` conserva invece un vantaggio chiaro anche in `C149` è nella **tenuta fuori densità** (`1%` e soprattutto `20%` su `VAL`, dove `p_off` peggiora di `+9.05 K`).

### Domanda 5 — Il miglioramento riguarda anche regioni e fallimenti rilevanti per la combustione?
- **Sì per i getti freddi (`T < 1800 K`) e le regioni non calde — guadagno strutturale rilevante:** `C149` riduce `Jet RMSE` su `VAL` da `124.5–130.7 K` (`A38`/`B19`) a **`111.9–114.8 K`** (`best`) e a `last` contiene la deriva (`119.7–122.1 K` contro `140–153 K` di `A38`/`B19`); su `seen` scende fino a **`98.7–102.7 K`** (`−29` a `−42 K` rispetto ad `A38`). I pannelli di errore firmato (`val_frame1000_*.png`) confermano visivamente l'attenuazione delle macchie rosse/blu lungo i rami del V-jet su `0NH3_1.05` e `20NH3_1.1`.
- **Sì sulle aree calde critiche di `seen` per `p_off`:** le sovrastime estreme di area calda (`high_rel_area_error` medio `2.55` in `B19-off` e `1.99` in `A38-off`, dovute a miscele con area calda vera `<2%` come `40NH3_0.85` e `80NH3_1.2`) si riducono fortemente in `C149-off` (`0.55`).
- **NO sul fallimento di livello di `40NH3_0.9` su `VAL`:** sebbene `C149` migliori la `centered-RMSE` su `40NH3_0.9` (fino a `19.4–20.2 K` su `p_on` `last`/`best_seen`, `22.9 K` su `p_off`), il bias negativo di `p_on` resta tra `−21.6 K` e `−25.9 K`, mantenendo l'IoU calda instabile (`0.787` a `best`, ma **`0.474`** a `best_seen` e **`0.562`** a `last`). Più frame migliorano dunque la **struttura** della fiamma e dei getti, ma **non eliminano** il problema del livello termico sulle condizioni escluse vicino a `2100 K`.

---

## 5. Classificazione delle lacune, limiti e giudizio di chiusura

### 5.1 Classificazione trasparente delle lacune
1. **Errori bloccanti per uno specifico confronto:** **nessuno** nei confronti interni al runner Opus (`B19`/`A38`/`C149` seed 0 e `A38` seed 0/1). *(Il divario storico S5 `DataLoader` vs stream Opus era bloccante per usare S5 come `A38 seed 0`, ed è stato chiuso operativamente eseguendo `O-A38-on/off-s0` e tenendo S5 fuori da O5.)*
2. **Limiti accettabili da dichiarare esplicitamente:**
   - Un solo training seed (`seed 0`) su `B19` e `C149`; solo 2 seed su `A38` (con salto di `+5.83 K` sul livello `VAL` di `p_on` da `40NH3_0.9`).
   - Tre sole condizioni `VAL` già coinvolte in selezione, scheduler e arresto (evidenza di sviluppo, non validazione indipendente); `R0` resta contaminata da scheduler/early-stop e priva di checkpoint intermedi.
   - Normalizzatore congelato sui 760 frame `A38` anche per `B19`; `reference760` per `B19` mescola 380 frame ottimizzati e 380 non visti in training; i frame passo 10 di `C149` sono temporalmente correlati.
   - Coperture `1%` e `20%` sono generalizzazione dalla densità TRAIN `5%`, non prestazioni ottimali a ogni densità.
   - Raw remoti per-frame, `COMPLETE.json`, log Slurm e `mask_hash` per-riga non presenti in Git (documentati dall'esecutore e validati dal gate `scripts/30`); i 3 PNG dei campi `val_frame1000_*.png` sono committati e verificati visivamente ma omessi da `package_manifest.json`; le figure storiche S6 (loss lineare, swap per-pannello) non sono state rigenerate (`S6_DELIVERY_CHECK.md`).
3. **Estensioni opzionali (NON necessarie per chiudere Opus):**
   - Replica `seed 1` di `C149` on/off (eventuale passo `O6`, da valutare solo dopo la sintesi congiunta Astra–Opus e non da avviare automaticamente qui).
   - Aggiornamento igienico di `package_manifest.json` per includere gli SHA dei 3 `val_frame1000_*.png` (nessun impatto numerico o scientifico).
   - Suddivisione diagnostica di `reference760` per `B19` nei 380 frame ottimizzati vs 380 non ottimizzati.

### 5.2 Giudizio esplicito sulla chiusura di OPUS
**La sotto-missione OPUS (O0–O5) può essere chiusa scientificamente nel perimetro concordato.**
- Tutti gli otto training nel runner uniforme e tutte le valutazioni O5 (`8 × 81` CSV + `8 × 3` file campi) sono completi, coerenti e verificati senza errori bloccanti.
- Tutte le cinque domande di `roadmapopus.md` §2 hanno ricevuto risposta quantitativa chiara, separando `seen220` da `VAL123`, livello da struttura, regioni calde da getti freddi e coperture `1/5/20%`.
- Nessuna correzione tecnica è richiesta a Sonnet per la chiusura di Opus. La decisione `O6`, la sintesi generale Astra–Opus, il riesame delle proposte Neon e la scelta di S7 restano all'incarico successivo con Enrico e il coordinatore.
