# Studio con temperatura — valutazione finale dei 12 training assisted

Pacchetto tecnico, descrittivo: nessuna classifica, nessun vincitore, nessuna regola di scelta applicata. Metriche = valutazione finale, non del loop.


## Completezza

- Run validi: 12/12; righe per-frame: 714744; gate di parità con il loop (macro RMSE di dominio comune entro 0.05 K per run/checkpoint/ruolo/replica e per K su VAL): scarto massimo 0 K.
- Matrice congelata SHA256 `3663d33820aac446b9621bf35ae49dec26ee4cc6a8379ad5363b0a81e8d922f7`; manifest T e protocollo come da matrice.
- Controlli: chiavi uniche e complete, `mask_hash` per riga = payload verificato, identità/hash dei checkpoint = matrice, forward condivisi identici alla sorgente.

## Contenuto

- `tables/global_by_setting.csv`, `global_by_replica.csv`, `per_condition.csv`: macro per run × checkpoint (best, last, best_seen) × ruolo × K, con full-field, dominio comune (griglia senza i 32 siti del layout), off-anchor(K) e alle ancore; RMSE, MAE, bias, centered, regionali.
- `gain_vs_K0_by_run.csv`, `family_p_seed_dispersion.csv`, `paired_contrasts.csv`, `free_S7_reference_vs_assisted_K0.csv` (riferimento free: modelli diversi, nessuna identità promessa), costi (calcolo e coda separati).

## Limiti

- Il criterio di selezione (media dei sei K, dominio comune, VAL 5%) è nuovo rispetto a S7; VAL123 resta usata nella selezione.
- Le tre coppie layout-T/maschera-velocità misurano variabilità congiunta: non attribuibile ai soli sensori T.
- Assisted(K=0) è un altro modello: non è il free originale. T ideali, nessun rumore, nessun TEST.

## Figure

- `costs_compute_vs_queue.png`
- `domains_vs_K_val_best.png`
- `gain_vs_K0_val_best.png`
- `rmse_vs_K_common_best.png`
- `rmse_vs_K_common_best_seen_diagnostic.png`
- `rmse_vs_K_common_last.png`
