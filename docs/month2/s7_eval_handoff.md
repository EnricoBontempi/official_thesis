# S7 — continuazione glu-on-s1 e preparazione evaluator: stato al 21 novembre 2026, 13:12 EST

**Stato:** preparazione dell'evaluator INTERROTTA su richiesta di Enrico. Nessuna valutazione finale sottomessa. Nessuna dichiarazione di S7 concluso. Metriche citate = metriche DEL LOOP, non valutazione finale.

## 1. Continuazione di S7-C149-glu-on-s1 (autorizzata da Enrico, sottomessa, in esecuzione)
- Job **29966737** (singolo, L40S, `mit_normal_gpu`, account/QoS `mit_amf_advanced_gpu`, 8 CPU, 64G, 6h, cap interno 5.5h, `--requeue`), partito 12:38:53 su node4201.
- SHA eseguito: **501d41a0be44709c69cf320737bd8d263fdb5216** (checkout `/home/bontempi/projects/official_thesis_s7_ext`, detached). Differenza dal training originale (f792b73): solo `--extend-n-epochs`, `plan_extension`/`preserve_tranche`, branch `extend` del runner e script `s7_extend*`. Ricetta, scheduler, regola early-stop invariati.
- Ripresa da `last.pt` completo di epoca 1000 (update 95000, stato `running`), `n_epochs` 1100, regola early-stop invariata (min 200 epoche, pazienza 15 eval, 0.1% relativo). Optimizer, scheduler, RNG, sampler, storia e contatori ripristinati dal checkpoint.
- Prima tranche preservata byte-per-byte con manifest SHA256 in `S7-C149-glu-on-s1/tranche1_epoch1000/` (summary, progress, eval_history, curves, stream_trace, run_identity, config, 3 checkpoint). Provenienza in `MANIFEST.json` e nel ledger (`extension_start`, `attempt_start action=extend`); `summary_before_resume_2.json`, `run_identity_before_resume_2.json`, `effective_config_resume_2.json` scritti dal trainer.
- Ultima lettura (13:12): epoca 1079, best invariato 41.501 K @630, early_stop_bad_evals 14/15, LR 1e-5. Se non compare un nuovo best, l'early-stop scatta all'epoca 1080 (o 1090); altrimenti arriva a 1100. Esito finale NON verificato.
- Se il job termina `stopped_wall_time`, il runner fa requeue automatico dello stesso job con gli stessi parametri di estensione.

## 2. Best e soglie
`best.pt` segue il primo minimo assoluto stretto (`score < best`); la soglia relativa 0.1% governa solo il contatore early-stop (`relative_improved`) e, con soglia propria, `ReduceLROnPlateau`. La frase "best cambia sotto 41.46 K" era imprecisa: il best cambia per qualsiasi valore < 41.50100804 K; 41.4595 K è la soglia che azzera il contatore early-stop. Riscontro nelle storie: in glu-off-s0 le eval 370 e 520 hanno fissato un nuovo best con guadagno 0.039% e 0.080% (contatore NON azzerato). In tutti i run `best_val_epoch` = primo minimo assoluto e il best registrato coincide con il minimo corrente a ogni eval.

## 3. Evaluator S7 (stato di preparazione)
Commit locali sul branch `s7/c149-free-campaign` (NON pushati; i checkout cluster li hanno via bundle):
- `64a772a`/`ade4bc4`: `scripts/38_s7_eval.py` (modi full/smoke/parity, U-Net e GLU, query grid GLU, p_on/off, fp32 senza TF32, gate 0.05 K sul 5% per run/checkpoint/ruolo/replica, assenza di riferimento = FAIL, scarti per-frame come diagnostica), `scripts/39_s7_write_eval_matrix.py` (matrice 12 run con checkpoint, SHA file/pesi, epoche, forward condivisi), `src/combustion_recon/evaluation/s7_eval_plan.py` + test, `scripts/engaging/s7_eval.sbatch` e `s7_eval_submit.sh` (ceiling 4 GPU), preflight O5 riusato.
- Checkout cluster di valutazione: `/home/bontempi/projects/official_thesis_s7_eval` @ **ade4bc43358bfa62ea8041c6b2be8df251bb087e**.
- **Smoke reale** (array 29967397, 4 run: unet-on-s1, glu-on-s0, glu-off-s0, O-C149-on-s0; 6 frame/ruolo, tutte le coperture 1/5/20%, 81 impostazioni): SMOKE_OK per tutti. Caricamento e query grid (16384×2) ok; p_on GLU reagisce a p (max diff 1.53 norm.), p_off GLU bit-identico (0.0); forward condiviso registrato (27 impostazioni su glu-off-s0, last←best_seen). Scarti per-frame vs CSV del loop sul sottoinsieme: U-Net ≤ 9e-6 K; GLU train/seen ~1e-5 K, GLU VAL fino a 0.025–0.040 K su singoli frame (media 0.001–0.003 K): da qualificare col parity completo. Il gate macro non è applicabile al sottoinsieme.
- **Parity completo al 5%** (array 29968643, 11 run finiti, modo `parity`, out `/home/bontempi/official_thesis_s7_eval_smoke_parity_v1`, concurrency 3): sottomesso PRIMA dell'interruzione e NON cancellato. Alla lettura delle 13:12 erano in corso i task 3–5, in coda 6–10; i task 0–2 avevano già scritto output. **Esiti non letti.** Per ogni run: `PARITY_OK.json` oppure `PARITY_FAILED.json` + `gate.json`, `gate_macro.csv`, `gate_per_frame_diagnostic.csv`.
- Matrici di prova (fuori repo): `/home/bontempi/official_thesis_s7_eval_smoke_cfg/matrix_smoke_v1.json` (7 run), `matrix_parity_v1.json` (11 run).

## 4. Da fare alla ripresa
1. Leggere gli esiti del parity (29968643) e del job 29966737; verificare il run glu-on-s1 esteso (summary, storia 1000→fine, best, checkpoint, ledger).
2. Se serve: qualificare gli scarti GLU VAL per-frame (ripetere un parity per misurare il rumore run-to-run GPU), senza cambiare la soglia 0.05 K.
3. Generare la matrice finale a 12 run (`39_s7_write_eval_matrix.py`, dopo la fine di glu-on-s1) e committarla; stimare il tempo (smoke → circa 1.5 h/run, richiedere ~4 h).
4. Solo dopo continuazione verificata e PASS: commit/push e sottomissione delle valutazioni finali con `s7_eval_submit.sh` (output nuovi, normal + Advanced L40S, max 4 GPU).
- Comando di sottomissione previsto: `scripts/engaging/s7_eval_submit.sh --tip /home/bontempi/projects/official_thesis_s7_eval --sha <SHA> --matrix configs/experiments/month2_s7_c149_eval_matrix.json --out-root /home/bontempi/official_thesis_s7_c149_eval --mode full --partition mit_normal_gpu --account mit_amf_advanced_gpu --qos mit_amf_advanced_gpu --time 04:00:00 --max-gpus 4`

## Comandi manuali
```
ssh engaging
squeue -u bontempi -r -o "%i %j %T %M %r %N"
sacct -D -X -j 29966737,29968643 -o JobID%18,State,ExitCode,Elapsed,NodeList
R=/home/bontempi/official_thesis_s7_c149_runs
tail -f $R/slurm/slurm-ext-29966737.out
cat $R/S7-C149-glu-on-s1/progress.json; cat $R/_ledger/S7-C149-glu-on-s1.jsonl
ls /home/bontempi/official_thesis_s7_eval_smoke_parity_v1/*/{PARITY_OK,PARITY_FAILED}.json
```
Nessuno sweep K né training con T. STOP.

---
## AGGIORNAMENTO 21 novembre 2026, ~15:15 EST — valutazioni finali sottomesse
Supera le sezioni «interrotta» sopra. Piano, controlli e stime: `docs/month2/s7_eval_plan.md`.
- **SHA eseguito (valutazione):** `acc8e2b4071578a584accfbde8c3c1a2cb5e2bf1`, pushato su `origin/s7/c149-free-campaign` (fast-forward da f1e3947, verificato con `git ls-remote`). Checkout remoto congelato (detached, pulito): `/home/bontempi/projects/official_thesis_s7_final_eval`. Matrice: `configs/experiments/month2_s7_c149_eval_matrix.json`, SHA256 `a76bac9b88e9a8be3a363302c1b171f77a0c7de0c232c5056add762ff3cd8da3`.
- **glu-on-s1 esteso:** early-stop a 1080, best 630 invariato, tranche 1 preservata e riverificata; training SHA 501d41a.
- **Parity 11 run finiti (29968643):** 11/11 PARITY_OK, scarto macro e per-frame massimo 0.0.
- **Job:**
  - 29984795 — valutazione completa, 11 run (array 0–10 %3), 3 h/task. Mapping indice→run: 0 unet-on-s1, 1 unet-off-s1, 2 unet-on-s2, 3 unet-off-s2, 4 glu-on-s0, 5 glu-off-s0, 6 glu-off-s1, 7 glu-on-s2, 8 glu-off-s2, 9 O-C149-on-s0, 10 O-C149-off-s0.
  - 29984793 — parity dei checkpoint finali di glu-on-s1 (1 GPU, 1 h), output `/home/bontempi/official_thesis_s7_eval_smoke_parity_s1_v1`.
  - 29984812 — valutazione completa di glu-on-s1, 3 h, `--dependency=afterok:29984793` (parte solo se il parity passa; se fallisce resta non avviata, `--kill-on-invalid-dep=yes`).
- **Risorse:** `mit_normal_gpu`, account/QoS `mit_amf_advanced_gpu`, 1×L40S, 8 CPU, 64G; max 4 GPU complessive (3 + 1 del run esteso, mai insieme al suo parity oltre il ceiling).
- **Output:** `/home/bontempi/official_thesis_s7_c149_eval/<run_id>/` (log `slurm/`, `submission_*.json`); non sovrascrivibili. Per run: `per_frame/<ckpt>/cov*/<role>/rep*.csv` (81), `fields/`, `provenance.json`, `setting_costs.csv`, `gate.json`, `gate_macro.csv`, `gate_per_frame_diagnostic.csv`, e `COMPLETE.json` solo se readback e gate passano (altrimenti `GATE_FAILED.json`).
- **Stima:** ~85–95 min per run (56–67 min con forward condiviso: glu-off-s0, glu-on-s2); 3 ondate; ~4–5 h dopo la coda.
- **Non lanciato:** swap best@5%, residuo GappyPOD p_on/off S6, baseline C149, aggregazione/report/protocollo (vedi piano).
- **Comandi manuali:** 
```
ssh engaging
squeue -u bontempi -r -o "%i %j %T %M %r %N"
sacct -D -X -j 29984793,29984795,29984812 -o JobID%16,State,ExitCode,Elapsed,NodeList
O=/home/bontempi/official_thesis_s7_c149_eval
ls $O/*/COMPLETE.json $O/*/GATE_FAILED.json 2>/dev/null
tail -f $O/slurm/slurm-29984795_0.out
```

---
## AGGIORNAMENTO 21 novembre 2026, ~16:00 EST — verifica esiti e aggregazione preparata
- **Stato job (15:47):** 29984793 (parity glu-on-s1) COMPLETED, **PARITY_OK** sui checkpoint finali. 29984795 task 0–2 e 29984812 in esecuzione (~36 min su ~90; 29/81 CSV per run), task 3–10 in coda per il limite %3. Nessun fallimento. Nessun run ancora COMPLETE: matrice incompleta, nulla aggregato.
- **Aggregazione (non eseguita, pronta):** `scripts/40_s7_aggregate.py` + `src/combustion_recon/evaluation/s7_package.py` (controlli: COMPLETE 81/81 e gate 18/18 per run, provenance e matrice, chiavi uniche/complete, `mask_hash` per riga contro il payload verificato 1/5/20%, identità dalla matrice congelata, forward condivisi bit-identici; poi tabelle per seed/condizione/macro, dispersione sui seed, contrasti appaiati famiglia e p, degradi per copertura/ruolo/selezione, peggior condizione VAL, regionali con NaN espliciti, figure, `INDEX.md`), `scripts/41_s7_costs.py` (calcolo vs attesa da sacct, tutti gli attempt), test `tests/test_s7_package.py` (10 test OK sul cluster). Il checker non è ancora stato esercitato su output reali completi (esce prima se manca COMPLETE.json).
- **Job di aggregazione CPU 29988745** (`mit_normal`, `mit_amf_advanced_cpu`, 8 CPU, 64G, 1 h), `--dependency=afterok:29984795:29984812 --kill-on-invalid-dep=yes`: parte da solo solo se tutte le valutazioni finiscono con exit 0; rifiuta comunque una matrice non valida. Checkout `/home/bontempi/projects/official_thesis_s7_agg` @ `aadf05cc91dab49c11d59fae147f83b5e374e4d4`. Output: `/home/bontempi/official_thesis_s7_c149_package/` (+ `_costs/`), log `/home/bontempi/official_thesis_s7_c149_package_slurm/`.
- **Da fare dopo:** leggere l'esito di 29988745 (se rifiuta, elencare i problemi), copiare tabelle/figure in `results/month2/s7_c149_comparison/package/`, commit e push. Se un task di valutazione fallisce: riportare causa e recupero prima di rilanciare.
- **Impegni S7 ancora mancanti (nessun STOP F):** swap neurali best@5%, swap GappyPOD p_on/off S6, baseline C149, protocollo con regola di seguito da approvare, report 10.

---
## AGGIORNAMENTO 21 novembre 2026, 22:15 EST — valutazioni concluse e pacchetto consegnato
- **Esiti:** 29984795 (11 run) tutti COMPLETED 0:0 (53–82 min ciascuno), 29984793 parity glu-on-s1 PARITY_OK, 29984812 glu-on-s1 esteso COMPLETED (81 min), 29988745 aggregazione CPU COMPLETED (48 s). **12/12 run COMPLETE, gate 18/18 per run, nessun GATE_FAILED, nessun problema tecnico.**
- **Pacchetto:** `results/month2/s7_c149_comparison/package/` (INDEX.md, `tables/`, `figures/`, `package_manifest.json`); 357372 righe, 972 CSV. Verifiche indipendenti (ricalcolo di 4 macro dai CSV grezzi: scarto 0), figure ispezionate. INDEX rivisto a mano (provenienza dichiarata).
- **Costi** (da sacct): training compute 3.7–4.2 h U-Net, 5.5–7.8 h GLU (glu-on-s1: 3 attempt, 7.76 h); evaluation 53–82 min a run, GLU e U-Net equivalenti. L'attesa include il limite %4/%3 degli array.
- **Mancano per STOP F:** swap neurali, swap GappyPOD, baseline C149, protocollo con regola di scelta (da approvare), report 10. STOP F non dichiarato.

---
## AGGIORNAMENTO 21 novembre 2026, ~22:45 EST — residui S7 in corso (sessione SSH scaduta)
Incarico: completare swap neurali, swap GappyPOD A38, baseline C149, diagnostico batch GLU, figure dei campi, report 10, stato roadmap. **Sessione SSH a Engaging scaduta alle ~22:45** (richiede password+Duo): lavoro cluster interrotto qui; i job sotto girano da soli. Nessun training, nessuno sweep K, nessun TEST.
Codice pushato su `s7/c149-free-campaign`: scripts 42 (swap neurale), 43 (baseline C149), 44 (swap Gappy A38), 45 (diagnostico batch), 46 (figure dei campi + etichetta corretta centered-vs-bias), 48 (pacchetto residui), `configs/experiments/month2_s7_c149_baselines.yaml`, sbatch relativi.
| Job | Cosa | SHA eseguito | Stato all'ultima lettura | Output |
|---|---|---|---|---|
| 30027323 / 30027560 / 30027620 | smoke swap neurale / baseline / swap Gappy | a6d7592 / 9e7ca68 / e5783bc | tutti SMOKE_OK | `official_thesis_s7_swap_smoke_v1`, `..._baselines_smoke_v2`, `..._gappy_swap_smoke_v1` |
| **30027642** | baseline C149 (control A38-subset, fit 2980 frame, 108 CSV) | 9e7ca68 | **COMPLETE**; control 12/12, scarto max 2.5e-8 K vs storico | `/home/bontempi/official_thesis_s7_c149_baselines` |
| **30027643** | swap GappyPOD A38 congelato p_on/off | e5783bc | **COMPLETE**; replay aligned = righe S6 (delta 0.0) | `/home/bontempi/official_thesis_s7_gappy_swap_a38` |
| **30027646** | swap neurale 12 best, array 0–11 %3 | e5783bc | 3 COMPLETE su 12 alle 22:43, resto in corso (~1 min/task) | `/home/bontempi/official_thesis_s7_c149_swap/<run>/` |
| **30027668** | diagnostico batch GLU (p_on, p_off) + U-Net | 158dfa3 | PENDING (slot GPU) | `/home/bontempi/official_thesis_s7_batch_diag_v1` |
Primi numeri (non interpretati): baseline C149 VAL 5%: GlobalTrainMean 139.59, ParamConditionedMean 87.73, GappyPOD p_off 93.79, p_on 70.56 K (A38 storico: 139.70 / 88.01 / 94.80 / 70.60). Swap Gappy A38 VAL: p_on aligned 70.5 → swapped 98.4, p_off 95.1 → 129.0, riferimento donatore 97.36 K (come S6).
**Da fare alla ripresa (in ordine):** (1) leggere esiti 30027646 e 30027668 (`sacct`, marker COMPLETE/DIAG_OK, `SWAP_FAILED.json`); se falliscono, causa prima di rilanciare; (2) `scripts/46_s7_field_figures.py --matrix configs/experiments/month2_s7_c149_eval_matrix.json --eval-root /home/bontempi/official_thesis_s7_c149_eval --package-tables /home/bontempi/official_thesis_s7_c149_package/tables --dest /home/bontempi/official_thesis_s7_c149_fields` (CPU, no inferenza) e ispezionare le figure; (3) `scripts/48_s7_residuals_package.py` con `--swap-root`, `--gappy-swap-root`, `--baselines-root`, `--diag-root`, `--package-tables`, `--eval-root`, `--matrix`, `--dest /home/bontempi/official_thesis_s7_c149_residuals`; (4) scaricare in `results/month2/s7_c149_comparison/{residuals,fields}/`, correggere la figura centered_vs_bias nel pacchetto (nuova etichetta «mean over frames of per-frame |bias|»); (5) scrivere `docs/month2/10_s7_c149_model_comparison.md` e aggiornare in modo circoscritto roadmapsparse/roadmap_month2/RESEARCH_LOG/README; STOP F solo se tutto verificato; (6) commit/push.

---
## AGGIORNAMENTO 21 novembre 2026, ~23:30 EST — residui conclusi, STOP F registrato
Swap neurali 30027646 (12/12 COMPLETE), diagnostico 30027668 (DIAG_OK), baseline 30027642, swap Gappy 30027643: tutti verificati e aggregati (`results/month2/s7_c149_comparison/{residuals,fields}`; code `b2b6e79`/`951748b`). Report `docs/month2/10_s7_c149_model_comparison.md`; indice unico `results/month2/s7_c149_comparison/package/INDEX.md`. STOP F concluso nel perimetro temperature-free; Mese 2 aperto; prossimo blocco sweep K su entrambe le famiglie e p_on/off.
