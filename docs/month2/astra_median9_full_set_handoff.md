# Astra median9 — handoff per l’analisi dell’insieme completo

Second pass esplorativa. I checkpoint S5 erano già stati scelti con la VAL. Queste tabelle non sono una conferma indipendente, non sostituiscono A0–A3 e non scelgono uno stimatore. Non è A4 e non è una decisione S7.

## Fonti

Inferenza: commit `44e5e5e0bd0907560d1736a53d357933680b090a`, checkout `/home/bontempi/projects/official_thesis_astra_median9`.
Output grezzi, non copiati in git: `/home/bontempi/official_thesis_astra_results/median9_second_pass/`.
Riferimento primario, non modificato: `/home/bontempi/official_thesis_astra_results/level_structure_v1/`.

| Task | Modello | Stato | Exit | Durata | Nodo |
| --- | --- | --- | --- | --- | --- |
| `29686178_0` | U-Net p_on | COMPLETED | 0:0 | 2 min 00 s | node2804 |
| `29686178_1` | U-Net p_off | COMPLETED | 0:0 | 2 min 07 s | node3512 |
| `29686178_2` | GLU p_on | COMPLETED | 0:0 | 1 min 58 s | node2804 |
| `29686178_3` | GLU p_off | COMPLETED | 0:0 | 2 min 02 s | node3006 |
| `29686179` | baseline | COMPLETED | 0:0 | 29 s | node2705 |

Log vuoti in stderr. Ogni run ha `GATE.json` e `COMPLETE.json` con `scientific_validated` vero. Il replay free/oracle/AO9 copre 18 celle per run neurale e 72 celle per le baseline, più 6 celle T-only. Riferimenti presenti, accoppiamento completo, scarto macro, scarto per-frame e offset applicato tutti a 0 K nel JSON di replay, dentro la tolleranza 0.05 K. I CSV primari non sono stati riscritti.

Checkpoint, cache `710bab90…`, stats `678263fb…` e manifest maschere `2e2d8ff6…` coincidono con il contratto. Il payload delle maschere è stato verificato: 1029 righe, cioè tutte le istanze seen (220×3) e VAL (123×3). Ogni riga di metrica porta `mask_hash`. L’artefatto Gappy è `a6fbdfd2…`. Non sono state salvate le mappe complete.

Righe attese e trovate: 3300/1845 per run neurale, 13860/7749 per le baseline. Nessun duplicato di chiave. Le metriche di errore e gli offset dei cinque trattamenti sono finiti. I NaN regionali restano la convenzione storica delle regioni vuote. T-only ha gli offset a NaN, compresi mean9 e median9.

I nove residui ci sono, con `anchor_id` 0…8, riga, colonna e indice piatto. Le posizioni sono la griglia `(32, 64, 95) × (32, 64, 95)`.

## Tabelle

Indice: `results/month2/astra_median9_second_pass/full_set/INDEX.md`.
Script: `scripts/25_astra_median9_aggregate.py`, stessa macro di `scripts/20_astra_a3_aggregate.py`.

Seen ha 20 condizioni, VAL ne ha 3. Le otto metodi con predizione sono i quattro neurali e Gappy p_off, Gappy p_on, ParamMean, GlobalMean. T-only è solo in `t_only_*.csv`.

La differenza è sempre A−B. Per RMSE, MAE e bias assoluto il segno negativo è un avvicinamento al vero e il segno positivo è un peggioramento. `paired_frame_signs.csv` conta entrambi. Sommando gli otto metodi, i casi frame×replica in cui la RMSE di campo di median9−AO9 è negativa o positiva sono 3804 e 1476 su seen, 2299 e 653 su VAL. Nello stesso conteggio mean9−AO9 è positiva più spesso: togliere solo lo shrink non riproduce il confronto con la mediana. Questi totali non sono una classifica.

`offset_error_macro_long.csv` riporta, per trattamento, la distanza dell’offset applicato dall’oracle. L’oracle ha distanza zero per costruzione.

`anchor_col95_share_macro_long.csv` è la quota della somma dei contributi assoluti ad AO9 che viene dalle due ancore di colonna 95. Sulle macro di replica la quota sta fra 0.55 e 0.80, seen e VAL. È una descrizione della grandezza assoluta, non la prova che quei due siti spieghino lo scarto firmato rispetto all’oracle. I residui firmati, per ancora e per condizione, sono in `anchor_residual_per_condition.csv`.

La centered-RMSE, dopo le traslazioni, resta invariata a meno di 1e-14 K.

## Limiti

Il pacchetto `example_frame1000_rep0` resta un altro oggetto: tre condizioni VAL, frame 1000, replica 0, quattro neurali. Non va letto come l’insieme.

median9 contro AO9 cambia insieme lo stimatore e lo shrink. mean9, la media senza tau, è il confronto che isola lo shrink. median9 contro mean9 isola lo stimatore. Non c’è Huber, trimming, esclusione di ancore, altro K o rumore.

La VAL è già entrata nella scelta dei best S5. Le repliche sono maschere, non semi indipendenti. Nessuna inferenza nuova è stata lanciata per questa aggregazione.
