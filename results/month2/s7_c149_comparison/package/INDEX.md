# S7 C149 — pacchetto di valutazione finale (12 run)

**Provenienza.** Tabelle, figure e `package_manifest.json` sono output grezzi del job CPU 29988745 (21 novembre 2026, 48 s; codice `aadf05c`, evaluator `acc8e2b`) sulle valutazioni complete 29984795 (11 run) e 29984812 (glu-on-s1 esteso). I costi vengono da `sacct`. Questo `INDEX.md` è stato rivisto a mano dopo la generazione (sezioni «validati» e «mancanti», verifiche del revisore tecnico): tabelle e figure non sono state toccate. Output completi sul cluster: `/home/bontempi/official_thesis_s7_c149_eval/` (per-frame, campi, provenance) e `/home/bontempi/official_thesis_s7_c149_package/`.

Pacchetto tecnico per il revisore. Descrittivo: nessuna classifica, nessun vincitore, nessuna regola di selezione applicata (la proposta Opus non è approvata).
Metriche = valutazione finale S7, non metriche del loop. VAL123 già usata nella selezione del best: non è conferma indipendente.

## Completezza

- Run validi: 12/12; CSV per-frame: 972 (81 per run); righe totali: 357372.
- Gate loop-parity al 5% (macro 0.05 K): tutti i run 18/18; massimo scarto macro registrato: 0 K.
- Matrice congelata SHA256 `a76bac9b88e9a8be3a363302c1b171f77a0c7de0c232c5056add762ff3cd8da3`; evaluator SHA acc8e2b4071578a584accfbde8c3c1a2cb5e2bf1.
- Controlli superati: chiavi uniche e complete; `mask_hash` per riga = payload verificato (1/5/20%); famiglia/p/seed/epoca/criterio/hash file e pesi = matrice; forward condivisi bit-identici alla sorgente; provenienza e hash input.

## Risultati già validati e disponibili per il confronto U-Net/GLU

Utilizzabili ora per l'analisi descrittiva (nessun vincitore indicato qui):
- Dodici run su dodici: U-Net masked e GLU adattato deterministico, C149, p_on/p_off, seed 0/1/2 (i due U-Net seed 0 sono i riusati O-C149, rivalutati con lo stesso evaluator).
- Tutti chiusi per early-stop con la stessa regola (glu-on-s1 dopo la continuazione autorizzata fino a epoca 1080; best invariato a epoca 630). Best primario = primo minimo assoluto VAL 5% del loop; last e best_seen_diagnostic restano separati e non selezionano nulla.
- Valutazione finale completa: reference760, seen220, VAL123; coperture 1/5/20%; repliche di maschera 0/1/2; 3 checkpoint; 81 impostazioni per run.
- Gate scientifico: macro RMSE al 5% entro 0.05 K dal loop per run/checkpoint/ruolo/replica, 18/18 per ogni run; scarto massimo registrato 0 K (valutazione bit-identica al loop), anche per-frame.
- Verifiche del revisore tecnico oltre i controlli automatici: (i) ricalcolo indipendente dai CSV grezzi di quattro macro (glu-off-s1 last VAL 20%, O-C149-on-s0 best reference760 1%, unet-off-s2 best_seen seen220 5%, glu-on-s1 best VAL 5%): scarto 0 rispetto a `global_by_setting.csv`; (ii) 324 righe macro = 12 × 3 × 3 × 3; (iii) il best VAL 5% in `headline_best_5pct_by_run.csv` coincide con il valore del loop per tutti i run; (iv) figure principali ispezionate visivamente (leggibili, scale e legende coerenti).

**Limiti da tenere presenti (non risolti):**
- VAL123 è stata usata per la selezione del best: non è conferma indipendente (vale per tutti i best; last e best_seen non sono scelti su VAL ma VAL resta la stessa).
- `peak_rmse_K` è non definita (regione di picco vuota) su 972 frame di reference760, dichiarati in `undefined_regional_metric_frames.csv` ed esclusi dalle medie regionali; nessun NaN nelle metriche globali.
- Coperture 1% e 20% generalizzano la densità di TRAIN del 5%: non sono una curva ottima del numero di sensori.
- I contrasti U-Net/GLU a pari indice di seed non hanno init appaiata (architetture diverse); i contrasti on/off sì entro la stessa famiglia.
- La figura dei costi separa calcolo e coda, ma l'attesa dei training GLU include il limite `%4` dell'array, non solo la coda Slurm.
- Lo smoke a sottoinsieme di frame mostrava differenze fino a 0.04 K per frame sul GLU VAL, attribuite (ipotesi non verificata con test dedicato) alla composizione dei batch; con i batch completi il parity è identico.

**Stato dell'analisi.** I dati sono completi e validi per iniziare l'analisi descrittiva. NON esiste ancora la specifica `configs/experiments/month2_s7_c149_protocol.yaml` con la regola di scelta del seguito, che la roadmap voleva fissata PRIMA di leggere gli esiti: i risultati sono ora visibili, quindi la regola va approvata esplicitamente da Enrico (le regole suggerite nelle review Opus/Neon non sono approvate e non sono state applicate).

## Tabelle (`tables/`)

- `global_by_setting.csv` / `global_by_replica.csv`: macro per run × checkpoint (best, last, best_seen separati) × ruolo (reference760/seen220/VAL123) × copertura (1/5/20%); RMSE, MAE, bias, |bias|, centered, regioni.
- `per_condition.csv`: stessa griglia per condizione (VAL: 3 condizioni), con min/max sulle repliche di maschera.
- `regional_by_setting.csv`, `undefined_regional_metric_frames.csv`: metriche regionali e conteggio dei frame con regione vuota (NaN esplicito, esclusi dalle medie regionali).
- `family_p_seed_dispersion.csv`: media/min/max/deviazione (ddof=1, n=3) sui seed per famiglia × p — dispersione descrittiva dei training, separata dalle maschere.
- `paired_summary.csv`, `paired_by_condition.csv.gz`: differenze appaiate GLU−U-Net (stesso p, stesso indice di seed) e p_on−p_off (stessa famiglia, stesso seed), per condizione e riassunto (media/min/max, n condizioni migliori/peggiori). Nessun test, nessun vincitore; i seed di famiglie diverse hanno init diverse.
- `coverage_degradation.csv` (1%/20% vs 5%), `role_gap.csv` (VAL−seen, VAL−reference760), `selection_shift.csv` (last−best, best_seen−best), `worst_val_condition.csv`: peggioramenti.
- `headline_best_5pct_by_run.csv`, `run_matrix.csv`, `completeness.csv`.
- Costi: training_costs_by_*.csv, eval_costs_by_task.csv (calcolo e attesa separati; per i requeue l'attesa è dal requeue; l'attesa dell'array include il limite %4).

## Figure (`figures/`)

- `centered_vs_bias_5pct_best.png`
- `costs_compute_vs_queue.png`
- `paired_differences_val5.png`
- `rmse_points_5pct.png`
- `rmse_vs_coverage_best.png`
- `rmse_vs_coverage_best_seen_diagnostic.png`
- `rmse_vs_coverage_last.png`
- `val_by_condition_5pct_best.png`

## Attività ancora MANCANTI per STOP F (non eseguite, non dichiarate complete)

| # | Attività | Stato |
|---|---|---|
| 1 | Snapshot-swap neurali, best@5% replica 0, seen e VAL, stesso pairing/osservazioni/riferimento donatore, 12 run | NON eseguito (funzioni S6 `evaluate_swap`/`build_within_condition_swap_pairs` family-aware; manca il wrapper S7 guidato dalla matrice, con readback e gate) |
| 2 | Residuo swap GappyPOD p_on/off di S6 (CPU) | NON eseguito né verificato |
| 3 | Baseline C149 con pool e ricetta di fit versionati (nessun refit A38; baseline A38 = riferimenti storici) | NON eseguite |
| 4 | Specifica `month2_s7_c149_protocol.yaml` con regola di scelta del seguito (da approvare), report `docs/month2/10_s7_c149_model_comparison.md`, decisione sul seguito | NON creati |
| 5 | Altri controlli: `own_pool_train` (opzionale, separato) non calcolato; analisi interpretativa, pannelli di campo su tutte le VAL, tabelle di regioni con conteggi validi commentate | NON eseguiti (il pacchetto è tecnico, non interpretativo) |

**STOP F non è dichiarato.** Mese 2 e assisted restano aperti. Nessuno sweep K, nessun training con T, nessun TEST.
