# Astra — handoff tecnico per l'analisi (Neon)

Stato: consegna **tecnica**, nessuna conclusione scientifica. Branch `mission/astra-level`. Pacchetto:
`results/month2/astra_level_structure/` (indice in `README.md` lì).

## 1. Domanda e contratto
Quanto errore di ricostruzione dipende dal livello termico (offset uniforme), quanto ne recupera una traslazione
oracolo e quanto nove temperature puntuali ideali (AO9). Le nove T sono valori ideali da simulazione, senza rumore né
dinamica strumentale: non sono una validazione di termocoppie reali.
- Quattro best S5 originali seed0 (U-Net/GLU × p_on/p_off), nessun training, nessuna nuova selezione.
- Seen220 e VAL123 frame; velocità 5%, maschere storiche persistite, repliche 0/1/2 (tre maschere, non tre seed).
- Un forward fp32 per frame/replica; free, oracle (`delta=mean(T_true−T_pred)`), AO9 (`delta=Σresidui/(9+1)`, tau 1) dalla stessa predizione; correzioni in kelvin dopo denormalizzazione, senza clipping.
- T9: griglia 3×3 righe/colonne [32,64,95], zero-based, indice `r*128+c`; rumore 0; siti fissati a priori. Off-anchor = 16375 siti.
- Baseline: GappyPOD p_off/p_on, ParamMean (riceve p, quindi non matched al divieto p), GlobalMean, T-only costante (media delle nove T). Nessun refit.
- Soglie: caldo ≥2100 K, peak ≥2200 K, jet <1800 K. Regioni vuote → NaN con conteggi.
- Esplorativa: VAL ha influenzato selezione, scheduler e arresto S5; oracle usa tutta la T vera (solo diagnostica); AO9 cambia l'informazione disponibile rispetto al free.
- Codice eseguito: SHA `eb264bdd637f216f51267c4990f583d0ed165a32`; job Engaging 29612403–29612407 (COMPLETED 0:0, vedi `provenance/sacct_astra_jobs.psv`).

## 2. Inventario
Tabelle, figure e loro significato: `results/month2/astra_level_structure/README.md`. Macro: frame → condizione → media sulle
condizioni, per replica; `replica=mean` è la media dei tre valori di replica (min/max riportati). Seen e VAL separati.

## 3. Errore del replay originale e verifica sostitutiva
- I `replay_k0.json` neurali (4 run) hanno `all_ok=false`, `ok=null`: `scripts/18_*.py` filtra il riferimento S6 con `checkpoint=="best"`, ma `summary_all_settings.csv` di S6 contiene solo `last` e `best_swapped`; il riferimento è rimasto vuoto. I JSON originali sono preservati, non modificati.
- Verifica sostitutiva (`scripts/19_astra_replay_k0_verify.py`; output `verification/replay_k0_verification{.json,_cells.csv}`), tolleranza 0.05 K, nessuna inferenza. Fonti storiche:
  - **A** S6 `per_frame_best_cov05_<role>_rep<r>.csv.gz` (ruolo, replica, variante `best`, copertura 5%, run_id; 24 celle run×ruolo×replica; confronto anche per frame su `(sim_id, frame_index)`): 24/24 PASS; scarto macro max 8.7e-4 K; scarto per-frame max 0.033 K.
  - **B** S5 `eval_history.json` all'epoca del best e **C** S5 `evals/epoch_<best>/<role>_per_frame.csv.gz`: VAL repliche 0/1/2 e seen replica 0: 16+16 celle PASS, scarto max 1.4e-14 K.
  - Epoca: `checkpoint_epoch` Astra = `best_val_epoch` di S5 `summary.json` per tutti e quattro (60, 140, 440, 650).
- **Riferimento mancante (non PASS implicito):** per seen replica 1 e 2 non esiste figura in B e C (`eval_history` ha una sola cifra seen; il CSV per-frame ha solo `mask_replica_id=0`). Quelle 16 celle (4 run × 2 repliche × 2 fonti) sono `MISSING`; sono coperte solo dalla fonte A.
- Limite: i file S6 `per_frame_best_*` non hanno colonna epoca; l'identità del best poggia su (i) epoca provenance = S5 best, (ii) coincidenza esatta con B/C. Lo scarto fino a 0.033 K per frame rispetto ad A (esatto con B/C) è coerente con una diversa esecuzione numerica di S6; causa non indagata.
- Esito: replay completo PASS entro 0.05 K su tutte le 24 celle con riferimento; per seen rep1/2 la copertura è a fonte singola.
- Baseline: i replay in `baselines/replay_k0.json` (8 controlli, `all_ok=true`) confrontano la media sulle repliche con la tabella storica K=0 5%; non per replica. Il JSON non registra file/hash di riferimento. T-only non ha riferimento storico.

## 4. Baseline e maschere persistite
- **Percorso del codice (letto, `scripts/18` righe ~357–372 e `scripts/15` righe 151–182):** le baseline Gappy leggono `<mask_root>/<role>_nested_sites.npz`, colonna `sites_5pct`, indicizzata da `(sim_id, frame, replica)`; chiave mancante → `KeyError`; nessun fallback da seed. Runtime: hash SHA256 di `manifest.json` registrato in provenance (`2e2d8ff6…`), conteggio frame atteso (220/123), hash artefatto Gappy confrontato con la config.
- **Cosa il percorso baseline NON controlla:** `hash_5pct` per frame non è confrontato (lo fa solo il percorso neurale, `scripts/18` riga ~222), né il manifest della cache, né `aggregate_payload_sha256` delle maschere.
- **Prova sugli input dei job storici (indiretta):** `verification/baseline_mask_crosscheck.json`: per tutte le chiavi, `n_velocity_sites` (819) e `n_t9_on_velocity_sites` coincidono fra baseline e neurali (maschere verificate per hash); i replay Gappy con Δ≈1e-8 K contro la tabella storica sono coerenti con le stesse maschere. Non è una prova di identità degli insiemi di siti: le tabelle baseline non hanno `mask_hash`. ParamMean, GlobalMean e T-only non usano le maschere (valori identici fra repliche).

## 5. Controlli eseguiti (letture di artefatti, nessuna inferenza)
Slurm (sacct), log, COMPLETE/provenance/timing, hash checkpoint vs ledger STOP E, config e albero frozen; righe attese (neurali 1980/1107 per run, baseline 8580/4797), nessun duplicato né inf; NaN solo nelle colonne offset di T-only; nessun NaN nelle metriche regionali; oracle RMSE = centered (≤3e-14 K); centered invariato tra free/oracle/AO9; `all_bias=−delta_oracle`; `delta_ao9=0.9×residuo medio ancore`; chiavi di frame e `mask_hash` identici nei quattro run; i quattro npz aperti, finiti e coerenti con t9_sites; 36 pannelli di figura confrontati con l'RMSE del CSV (≤5e-7 K).

## 6. Limiti residui
- Seen rep1/2: riferimento storico a fonte singola (A). Scarti per-frame fino a 0.033 K rispetto a S6.
- Maschere baseline: prova indiretta (§4). Baseline senza campi salvati: nessuna figura per le baseline.
- `mse_recovery_vs_oracle` è per frame, NaN se denominatore ≤1e-6 K² (mai NaN per i neurali/Gappy/ParamMean/GlobalMean in questo run; sempre NaN per T-only). La media per frame di un rapporto non coincide con il rapporto fra MSE macro.
- Tre maschere non sono tre seed; seen = 20 condizioni, VAL = 3. Nessun test di significatività. Compatibilità con le patch comuni Opus non verificata (A4 non iniziata).

## 7. Riproduzione
```
# dati Astra: ~/official_thesis_astra_results/level_structure_v1 (Engaging); copia locale ASTRA=...
python scripts/19_astra_replay_k0_verify.py --astra-root $ASTRA --s6-root <official_thesis_s6_results> \
    --s5-runs-root <official_thesis_s5_runs> --out results/month2/astra_level_structure/verification
python scripts/20_astra_a3_aggregate.py --astra-root $ASTRA --out results/month2/astra_level_structure
python scripts/21_astra_a3_figures.py   --astra-root $ASTRA --out results/month2/astra_level_structure
python scripts/22_astra_baseline_mask_crosscheck.py --astra-root $ASTRA \
    --out results/month2/astra_level_structure/verification/baseline_mask_crosscheck.json
```
Fonti storiche: S6 `~/official_thesis_s6_results/<run>/<run>/per_frame_best_cov05_*`; S5 `~/official_thesis_s5_runs/<run>/{eval_history.json,summary.json,evals/epoch_<best>/}`;
hash di tutte le fonti nel JSON di verifica (`source_sha256`). Python con pandas/numpy/matplotlib (`.venv` del repo).
