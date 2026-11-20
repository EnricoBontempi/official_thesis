# Indice tecnico O5 per Neon

Pacchetto tecnico completo per la revisione. Non contiene il report scientifico né la decisione O6.

## Dove

- Pacchetto: `/home/bontempi/official_thesis_opus_o5_final/package_29771334/package` (`tables/`, `figures/`, `TABLES.md`, `package_manifest.json` con SHA-256 di ogni file).
- Output grezzi per frame, directory per run: `/home/bontempi/official_thesis_opus_o5_attempt2_runs/<run_id>/`.
- Ledger O0-O6: `/home/bontempi/official_thesis_opus_o5_final/package_29771334/LEDGER_O0_O6.md`. Riepilogo: `/home/bontempi/official_thesis_opus_o5_final/package_29771334/FINAL_SUMMARY.md`.
- R0 (già prodotto, non rifatto): `results/month2/opus_data_study/r0_selection/`. Limite: rianalisi sulle curve dei quattro S5, non una validazione indipendente; scheduler e early stop hanno usato VAL.
- Bridge S5 (già prodotto): `results/month2/opus_data_study/bridge_s5/LEDGER.md`. Limite: l'ordine dei frame di S5 (`DataLoader(shuffle=True)`) differisce dallo stream Opus (`randperm`); il replay di inferenza ha delta 0.0 K entro 0.05 K ma `training_equivalence` resta `not_tested`. Gli S5 non sono righe di O5; i due riferimenti A38 seed 0 usano il runner nuovo.
- Curve del loop (`training_loop_not_o5`): `results/month2/opus_data_study/o5_prep/loop_curves.csv`.

## Run

| run | pool | p | seed | training job | epoche best / best_seen / last | eval SHA |
| --- | --- | --- | ---: | ---: | --- | --- |
| O-B19-on-s0 | B19 | on | 0 | 29612012 | 50 / 50 / 620 | `918286809d` |
| O-B19-off-s0 | B19 | off | 0 | 29612013 | 60 / 40 / 630 | `29b94345f0` |
| O-A38-on-s0 | A38 | on | 0 | 29671719 | 120 / 120 / 630 | `918286809d` |
| O-A38-off-s0 | A38 | off | 0 | 29671721 | 90 / 130 / 660 | `918286809d` |
| O-A38-on-s1 | A38 | on | 1 | 29612010 | 60 / 60 / 630 | `918286809d` |
| O-A38-off-s1 | A38 | off | 1 | 29612011 | 40 / 110 / 610 | `918286809d` |
| O-C149-on-s0 | C149 | on | 0 | 29645053 | 90 / 220 / 660 | `918286809d` |
| O-C149-off-s0 | C149 | off | 0 | 29645054 | 260 / 240 / 710 | `918286809d` |

Il run O-B19-off-s0 è stato valutato dal job sostitutivo 29732630 (launcher `29b94345…`); gli altri sette dall'array 29728465 (launcher `91828680…`). Evaluator identico nei due; il pacchetto non descrive la campagna come una singola esecuzione. Il task 29728465_1 è fallito prima dell'evaluator per un controllo di accesso e non ha scritto output.

## Tabelle (`package/tables/`)

- `global_by_setting.csv`: run × checkpoint × ruolo × copertura, media sulle condizioni e sulle repliche, con min/max delle repliche per la RMSE.
- `per_condition.csv`: stessa griglia per condizione.
- `regional_by_setting.csv`, `regional_by_condition.csv`: metriche delle regioni calde, picco, getto, esterno; IoU e area relativa.
- `compare_summary.csv`, `compare_by_condition.csv`: delta appaiati dati (B19/A38/C149, seed 0, stessa p), seed (A38 1 meno 0), p (on meno off) e selezione (last/best_seen meno best). Descrittivi, nessun test, nessun vincitore.
- `loop_vs_o5_5pct.csv`: O5 al 5% contro il loop alla stessa epoca; train e seen replica 0, VAL media delle tre maschere, tolleranza 0.05 K.
- `setting_costs.csv`, `completeness.csv`.

## Figure (`package/figures/`)

Curve del loop contro update e passaggi del pool con i punti O5; RMSE al 5% per checkpoint; curve di copertura 1/5/20%; VAL per condizione; metriche regionali per checkpoint; delta appaiati; campi VAL frame 1000 con scala comune (un frame, non l'insieme VAL). Scale comuni dichiarate nei titoli. Le figure vanno guardate dal revisore prima dell'uso.

## Limiti dichiarati

- 1% e 20% sono generalizzazioni della densità TRAIN 5%, non una curva ottimale del numero di sensori.
- B19 usa il normalizzatore fit-only dei 760 frame; reference760 è la stessa diagnostica per tutti i pool.
- Tre run hanno pesi identici fra `best` e `best_seen_diagnostic`: forward condiviso, identità e righe distinte e verificate uguali.
- Due seed e tre maschere sono un metro descrittivo, non una distribuzione stimata. VAL ha influenzato selezione, scheduler e arresto.
- Nessun TEST, refit, A4 o nuovo training.

## Aggiunte dopo il controllo manuale (2026-11-20)

**Pacchetto di riferimento:** generato dal job finale **29771334** (checkout `258106bc6e0e6dc7a721be2645b5ba66472acc8b`), copia in questa cartella. Il file `package/tables/compare_by_condition.csv` è committato compresso (`.csv.gz`, 19 MB non compressi); lo SHA-256 in `package_manifest.json` si riferisce al CSV non compresso. Sul cluster: `/home/bontempi/official_thesis_opus_o5_final/package_29771334/`.

**Tentativi e SHA, in ordine.**

| Passo | Job | Esito | Eseguito a |
| --- | --- | --- | --- |
| Array O5 attempt1 | 29716019 | tutti FAILED prima di scrivere righe: import di un modulo inesistente | `c705659…` (directory vuote conservate) |
| Smoke B19 e C149 | 29717341 | task 0 passato; task 6 TIMEOUT in attesa di un automount su node4108 | `a4aceafe…` |
| Smoke C149, retry | 29725163 | passato | `91828680…` |
| Campagna O5 attempt2 | 29728465 task 0, 2-7 | COMPLETED, evaluator identico | `91828680…` (checkout `official_thesis_opus_o5_v5`) |
| Task 1, fallito | 29728465_1 | FAILED 6:0 al preflight su node4204, nessun output | `91828680…` |
| O-B19-off-s0 sostitutivo | 29732630_1 | COMPLETED | `29b94345…` (checkout `official_thesis_opus_o5_v6`) |
| Job finale, bloccato | 29735692 | FAILED 3:0: il post-processing rifiutava otto run validi (`peak_rmse_K` NaN su 3 frame; NaN ≠ NaN nel confronto dei forward condivisi). Nessuna valutazione cambiata | `6cdb25d1…` |
| Job finale, corretto | 29771069, poi 29771334 | pacchetto costruito; il secondo include le correzioni di legenda e barre colore | `e23f51bf…`, `258106bc…` |

I due SHA di launcher dell'evaluazione (`91828680…`, `29b94345…`) hanno lo stesso evaluator. Il post-processing ha SHA propri e non influenza i numeri.

**Metriche regionali non definite.** `peak_rmse_K` è NaN su alcuni frame (per esempio tre frame di `train_fit` di O-B19-on-s0 con `best`, copertura 0.01 replica 0). Le medie li saltano; l'elenco è `package/tables/undefined_regional_metric_frames.csv`. RMSE, MAE, bias e centered RMSE sono finite ovunque.

**Controllo O5/loop al 5%.** 72 confronti su 72 entro 0.05 K; scarti massimi 0.0003 K (train, replica 0), 0.0005 K (seen, replica 0), 0.0011 K (VAL, media delle tre maschere).

**Controllo visivo delle figure (fatto a mano).** Aperte e guardate: curve del loop contro update e contro passaggi del pool, RMSE al 5%, curve di copertura, VAL per condizione, delta appaiati, metriche regionali per `best` e `last`, campi VAL per `best` e `last`. Non aperte singolarmente: metriche regionali e campi per `best_seen_diagnostic`, che hanno lo stesso formato. Difetti trovati e corretti prima della consegna: legenda senza p off nelle curve, titoli sovrapposti e assenza di barre colore nei campi. Limiti rimasti: la scala comune degli errori dei campi vale dentro ogni figura (±707 K per `best`, ±853 K per `last`), non fra checkpoint; l'etichetta delle barre colore dei campi è vicina al bordo; i grafici a punti con tre sole condizioni VAL non hanno una banda di incertezza, solo il minimo e il massimo sulle repliche.

**O0.** Controllo delle tabelle S6 in `results/month2/opus_data_study/o0_checks/S6_DELIVERY_CHECK.md` (tabelle coerenti; figure di loss su scala lineare e figure swap con barre colore per pannello, non rigenerate). `roadmapsparse.md` e `roadmapopus.md` hanno lo stato corrente aggiunto; lo stato del 18 novembre è conservato come storico. `roadmapastra.md` e i report non sono stati toccati.

**Nota di lettura per Neon (lettura di tabelle, nessuna conclusione).** Al 5%, a seed 0 e stessa p: C149 ha RMSE VAL media minore di A38 in tutte e tre le condizioni per tutti e tre i checkpoint e per p on e off; su seen la RMSE è minore in tutte le 20 condizioni salvo C149−A38 p on con `best` (11 su 20). B19 è peggiore di A38 sul VAL in media (+1.5 K p on, +3.8 K p off con `best`). La differenza fra i due seed A38 sul VAL è fino a +5.8 K (p on) e +0.1 K (p off, `best`); l'effetto di p (on meno off) cambia segno fra regimi sul VAL e resta negativo su seen per tutti e quattro. Sono differenze descrittive con due seed, tre sole condizioni VAL e checkpoint scelti su VAL: attribuzione e decisione O6 spettano alla revisione.

**Non fatto, per mandato.** Report `06_opus_data_and_training_variability.md`, decisione O6, S7, A4, TEST, nuovi training. Opus non è chiuso scientificamente.
