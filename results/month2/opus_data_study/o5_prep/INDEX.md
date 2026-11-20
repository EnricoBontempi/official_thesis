# Indice tecnico O5 per Neon

Pacchetto preparato il 2026-11-19. O5 non è stata sottomessa. Questo indice non sceglie un run e non chiude Opus.

## Già nel repository

- Curve del loop, non valutazione O5: `loop_curves.csv` in questa cartella. 515 eval, una riga per epoca registrata degli otto run. `train_rep0_rmse_K` e `seen_rep0_rmse_K` sono la replica 0 del loop. `val_mean3_rmse_K` è la media delle tre maschere usata per selezione, scheduler e arresto. `metric_source` è `training_loop_not_o5`. L'update è `epoca × 95` e le presentazioni sono `epoca × 760`. `pool_passages` è presentazioni diviso la dimensione del pool (B19 380, A38 760, C149 2980).
- Copertura del pool, dal contratto dello stream e dai conteggi registrati, non da una lista di indici salvata. Lo stream è senza reinserimento sul pool. `samples_seen = epoche × 760` supera il pool in tutti gli otto run. B19 e A38 coprono l'intero pool dopo la prima epoca scientifica. C149 lo copre dopo quattro epoche (3040 presentazioni, pool 2980). Un'epoca C149 che attraversa il confine di un ciclo può ripetere un indice dentro le 760 presentazioni: il conteggio distinto per epoca non è stato persistito e non viene inventato.
- R0 sulle quattro history S5, già prodotta e non rifatta: `../r0_selection/` con `PROVENANCE.md`. Non è una validazione indipendente.
- Bridge e replay: `../bridge_s5/LEDGER.md`. Il replay 29670888 è passato a delta 0.0 K entro 0.05 K sui due best U-Net S5 al 5%. `training_equivalence` resta `not_tested`. Non cancella la differenza di ordine fra DataLoader S5 e stream Opus.
- Matrice dei soli otto run del runner Opus: `configs/experiments/opus_o5_matrix.json`. Gli S5 storici non sono righe.

## Cosa faranno i job, quando autorizzati

Per ogni run, tre checkpoint (`best`, `best_seen_diagnostic`, `last`), tre ruoli, coperture 0.01/0.05/0.20, repliche 0/1/2.

- `train_fit` è reference760: indici A38, anche su B19 e C149. Seen è 220 e VAL è 123.
- Il criterio di `best` è il primo minimo assoluto della VAL media sulle tre maschere. `best_seen_diagnostic` è il minimo seen della sola replica 0. `last` è l'epoca di early-stop. O5 non ricalcola questi criteri.
- Le metriche di riga sono quelle storiche del valutatore: RMSE, MAE, bias firmato e assoluto, centered RMSE, regioni high/peak/jet/non-high, IoU e area relativa della zona calda. Le tabelle restano separate per checkpoint, ruolo, copertura, condizione e replica.
- L'1% e il 20% sono generalizzazioni della densità TRAIN 5%, non una curva ottimale del numero di sensori.
- Tre run hanno pesi identici fra `best` e `best_seen_diagnostic` (`O-B19-on-s0` epoca 50, `O-A38-on-s0` epoca 120, `O-A38-on-s1` epoca 60). Il forward può essere condiviso. I file, gli hash e le righe restano distinti. Gli altri cinque `best_seen_diagnostic` sono epoche diverse.
- Nessun TEST, refit, resume o nuova selezione. Cache storica per B19/A38, cache C149 solo per i due attempt2. Il payload delle maschere è obbligatorio: le repliche 0/1/2 sono presenti su tutti e tre i ruoli (760, 220 e 123 frame per replica).

Output previsto, directory nuova e non sovrascrivibile:

`/home/bontempi/official_thesis_opus_o5_runs/<run_id>/`

Dentro: `per_frame/`, `fields/val_frame1000_<checkpoint>_cov05_rep0.npz` (un frame per le tre condizioni VAL, non l'insieme VAL), `setting_costs.csv`, `provenance.json`, `COMPLETE.json` solo se i 81 setting sono stati scritti.

## Dopo i job, senza rifare il training

1. `scripts/27_opus_o5_aggregate.py` scrive medie per condizione, medie sulle condizioni e medie sulle repliche. Non aggiunge un rango.
2. `scripts/28_opus_o5_figures.py` disegna il frame VAL 1000 al 5%, replica 0, con una scala di temperatura comune presa dalle verità e una scala simmetrica comune dell'errore firmato. Va lanciato una volta per `best`, `last` e `best_seen_diagnostic`.
3. Il report scientifico `docs/month2/06_opus_data_and_training_variability.md` resta di Neon. Non è in questo pacchetto.

## Costo proposto, non eseguito

Dal replay 29670888, un checkpoint al 5%: reference760 replica 0 circa 156–158 s (760 frame), seen replica 0 circa 13.4 s (220), VAL una replica circa 4.4 s (123). Applicando queste durate alle tre coperture e alle tre repliche, un checkpoint distinto costa circa 26 minuti. I pesi condivisi tolgono tre forward su ventiquattro: circa 9.2 ore-GPU. Se ogni frame costasse quanto reference760 (0.21 s), si arriva a circa 12 ore-GPU. Il wall di 4 ore per job copre la stima lenta e il ricalcolo degli hash di cache, che non è incluso nei 26 minuti.

Otto task, array `0-7%4`, al massimo quattro L40S insieme. Partizione `mit_normal_gpu`, account e QoS `mit_amf_advanced_gpu`, `gpu:l40s:1`, 8 CPU, 64G. Il launcher rifiuta l'avvio senza `AUTHORIZE_OPUS_O5=1` e senza `EXPECT_SHA`. Checkout proposto, da creare solo alla submission: `/home/bontempi/projects/official_thesis_opus_o5`.

> **Aggiornamento 2026-11-20.** Questo indice descrive la preparazione. O5 è stata eseguita sugli otto run e il pacchetto tecnico è in `../o5/` (indice di lettura: `../o5/INDEX_FOR_NEON.md`).
