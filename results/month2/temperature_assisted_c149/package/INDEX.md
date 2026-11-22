# Studio con temperatura — valutazione finale dei 12 training assisted

**Revisione del 22 novembre:** tabelle e figure sono output grezzi dei job 30060516/30060517 (SHA `b7e3b1f`); questo INDEX è stato esteso a mano con provenienza e stato. Report: `docs/month2/12_temperature_input_study.md`. Verifiche indipendenti: gate 69/69 per run con scarto massimo 0.0 K, ricalcolo di tre macro dai CSV grezzi (≤ 1.4e-14), figure ispezionate; 12/12 training conclusi (9 early-stop, 3 a fine tranche senza plateau accertato: glu-on-s1, glu-off-s1, glu-off-s2), preemption dei GLU gestite con resume da `last.pt`. File aggiuntivi in `..`: `matrix_frozen.json` (SHA256 `1c91ce25…`), `chain_submitted.json`.

**Già validato e utilizzabile:** valutazione completa dei 12 run sui sei K, tre checkpoint, tre ruoli, tre repliche. **Residuo:** continuazione eventuale dei tre run a fine tranche (non autorizzata), diagnostici di sensibilità a T, confronto con lo sweep di Cursor.

Pacchetto tecnico, descrittivo: nessuna classifica, nessun vincitore, nessuna regola di scelta applicata. Metriche = valutazione finale, non del loop.


## Completezza

- Run validi: 12/12; righe per-frame: 714744; gate di parità con il loop (macro RMSE di dominio comune entro 0.05 K per run/checkpoint/ruolo/replica e per K su VAL): scarto massimo 0 K.
- Matrice congelata SHA256 `1c91ce257e0ae93e6cb909bc23741f08ba7339ea9cb0b3e83677b3b15dff5042`; manifest T e protocollo come da matrice.
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
