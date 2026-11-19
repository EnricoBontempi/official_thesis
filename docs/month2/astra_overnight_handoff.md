# Astra — consegna notturna (A0–A2, submission)

Stato: **SUBMITTED, non completato.** Nessun risultato scientifico ancora disponibile.

## Sorgenti
- Branch dedicato: `mission/astra-level` (worktree `/home/enric/projects/official_thesis_astra`, creato da `85c720b`; pushato su origin).
- **SHA eseguito sui job: `eb264bdd637f216f51267c4990f583d0ed165a32`** (clone congelato `~/projects/official_thesis_astra_frozen`, controllato in sbatch: SHA esatto e albero pulito).
- Checkout S5 `~/projects/official_thesis_s5` usato solo in lettura (cache, maschere, stats, tabella baseline). Nessun file Opus/Cursor toccato; il working tree di `mission/dense-unet` non è stato modificato.

## File nuovi (tutti Astra)
- `src/combustion_recon/evaluation/astra_level.py` (T9, oracle, AO9 via `apply_anchor_offset` esistente, metriche all-field/off-anchor/regioni)
- `scripts/18_astra_level_structure_eval.py` (importa per riuso `17_*` e `15_*`; nessun secondo framework)
- `scripts/engaging/astra_level_{neural,baselines,smoke}.sbatch`
- `configs/experiments/month2_astra_level_structure.yaml` (freeze A0)
- `tests/test_astra_level.py`

## Freeze A0
- 4 `best.pt` verificati per SHA256 contro ledger STOP E: U-Net p_on `b03b896e…`, U-Net p_off `0d9d93a1…`, GLU p_on `bfd126e5…`, GLU p_off `fbae86d5…`; git_sha di training `38ac7fe…` verificato da `run_identity.json`; hash cache/maschere verificati a runtime; maschere persistite (hash per frame confrontato, nessun fallback da seed).
- Nessun campo predetto persistito riusabile (solo CSV) → nuova inferenza fp32 (no AMP/TF32/compile), un forward per frame/replica, free/oracle/AO9 dalla stessa predizione.
- T9: griglia 3×3 righe/colonne `[32,64,95]`, indice `r*128+c`; non esiste un layout K=9 storico univoco (le ancore storiche sono random per frame). tau=1, rumore 0, dominio 128×128 interamente valido. Soglie: caldo ≥2100 K, peak ≥2200, jet <1800; regioni vuote → NaN con conteggi.
- Baseline: GappyPOD p_off/p_on e ParamMean (coef dall'artefatto Gappy `a6fbdfd2…`, nessun refit), GlobalMean (media train_fit), T-only costante (media delle nove T). ParamMean è p_on.
- Rilievi O0 comuni: **non** importate patch Cursor in corso (`sparse_dataset.py`, `sparse_protocol.py`, trainer…); uso i sorgenti congelati a `85c720b`. Dipendenza registrata: se O0 cambia l'evaluator/sampler comuni, l'estensione A4 sui checkpoint Opus dovrà ripetere il replay K=0.

## Test eseguiti
- Locale: `tests/test_astra_level.py` (12 + 2 test: formula/segno/tau, K=0, permutazione ancore, oracle RMSE=centered, centered invariante a traslazione, mutazione T fuori ancore, mutazione ancora, off-anchor 16375, regioni vuote, p_off GLU) + isolamento p/baseline esistenti: 42 passed, 3 skipped.
- Cluster, job smoke **29612142** (COMPLETED, 47 s, CPU): stessa suite 42 passed/3 skipped; smoke 4 frame su U-Net p_on e GLU p_off (CPU) e baseline; identità oracle RMSE=centered a 3e-14.
- **Non eseguito**: il replay K=0 completo (0.05 K) avviene dentro ogni job (`replay_k0.json`); verifica solo a job finiti.

## Job sottomessi (tutti PENDING alla submission)
| Job ID | Scopo | Account/QoS | Risorse |
|---|---|---|---|
| 29612403 | U-Net p_on | mit_amf_advanced_gpu / mit_amf_advanced_gpu, `mit_normal_gpu` | 1 L40S, 4 CPU, 32G, 4h |
| 29612404 | U-Net p_off | idem | idem |
| 29612405 | GLU p_on | idem | idem |
| 29612406 | GLU p_off | idem | idem |
| 29612407 | Baseline matched | mit_amf_advanced_cpu / mit_amf_advanced_cpu, `mit_normal` | 4 CPU, 64G, 4h |

Walltime 4h è un vincolo operativo (lo smoke 4-frame fa pochi secondi). Nessun preemptable, nessun mit_general, nessun job Opus/HONF toccato.

## Percorsi (HOME Engaging, `/home/bontempi`)
- Output: `~/official_thesis_astra_results/level_structure_v1/<run_id>/` e `.../baselines/` (directory non sovrascrivibili: lo script fallisce se esistono). Contenuto: `per_frame_{seen_train_eval,val_eval}.csv.gz`, `provenance.json`, `replay_k0.json`, `timing.csv`, `val_frame1000_rep0_fields.npz`, `COMPLETE.json`.
- Log: `~/official_thesis_astra_results/slurm_logs/{neural_<run>_<jobid>.out,baselines_<jobid>.out,smoke_29612142.out}`
- Smoke: `~/official_thesis_astra_results/smoke_29612142/` (non scientifico).

## Controllo manuale
```
ssh engaging 'squeue -u bontempi -o "%i %j %T %r"; sacct -j 29612403,29612404,29612405,29612406,29612407 -o JobID,State,Elapsed,ExitCode'
ssh engaging 'ls ~/official_thesis_astra_results/level_structure_v1/*/COMPLETE.json; cat ~/official_thesis_astra_results/level_structure_v1/*/replay_k0.json | grep -E "all_ok"'
```

## Non fatto / da fare
- Nessuna aggregazione A3, report, figure PNG (salvati i campi frame 1000 VAL rep0 per le tre condizioni), né estensione A4 Opus, né sweep K/layout/rumore.
- Il replay baseline confronta con la tabella storica (ParamMean+AO storico usa ancore random: non confrontabile con T9, per design).
- Il tip `mission/dense-unet` e diario/roadmap condivisi non sono stati modificati; integrazione futura: merge/cherry-pick di `mission/astra-level` (7 file nuovi, nessun conflitto atteso) dopo che Cursor chiude O0.
