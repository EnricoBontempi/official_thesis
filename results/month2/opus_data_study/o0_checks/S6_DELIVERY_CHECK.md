# Controllo O0 della consegna S6 (STOP E)

Lettura sola, `scripts/31_o0_s6_delivery_check.py`.

- Righe per condizione: 3096, run: s5_adapted_glu_poff_seed0, s5_adapted_glu_pon_seed0, s5_sparse_unet_poff_seed0, s5_sparse_unet_pon_seed0.
  - s5_adapted_glu_poff_seed0: condizioni {'train_fit': 20, 'seen_train_eval': 20, 'val_eval': 3}, coperture ['0.01', '0.05', '0.2'], checkpoint ['best', 'last'], repliche ['0', '1', '2'].
  - s5_adapted_glu_pon_seed0: condizioni {'train_fit': 20, 'seen_train_eval': 20, 'val_eval': 3}, coperture ['0.01', '0.05', '0.2'], checkpoint ['best', 'last'], repliche ['0', '1', '2'].
  - s5_sparse_unet_poff_seed0: condizioni {'train_fit': 20, 'seen_train_eval': 20, 'val_eval': 3}, coperture ['0.01', '0.05', '0.2'], checkpoint ['best', 'last'], repliche ['0', '1', '2'].
  - s5_sparse_unet_pon_seed0: condizioni {'train_fit': 20, 'seen_train_eval': 20, 'val_eval': 3}, coperture ['0.01', '0.05', '0.2'], checkpoint ['best', 'last'], repliche ['0', '1', '2'].
- Colonne regionali presenti: 23 (high_rmse_K, high_mae_K, non_high_rmse_K, non_high_mae_K, peak_rmse_K, peak_mae_K, jet_rmse_K, jet_mae_K, peak_abs_error_K, true_peak_K, pred_peak_K). Valori mancanti nella tabella per condizione: nessuno.
- VAL: 24 righe di copertura, `n_frames` = ['369'] (123 frame × 3 repliche).
- Figure presenti: 12 (tre per run).

## Esito

Tabelle: conteggi, condizioni, repliche e colonne regionali coerenti con il contratto: **sì**.

## Limiti delle figure (osservati guardandole, non rigenerati)

- `*_train_loss_curves.png`: loss di training su scala lineare, una sola curva, senza VAL/seen né best/last; dopo circa 100 epoche la curva è indistinguibile dallo zero. Non permette di leggere il plateau.
- `*_val_frame1000_aligned_vs_swapped.png`: ogni pannello ha la propria barra colore, e l'errore assoluto allineato e scambiato usa scale diverse (circa 600 e 1400 K). Serve a vedere il contrasto dello swap, non a confrontare le ampiezze fra pannelli.
- Le figure di O5 (`results/month2/opus_data_study/o5/package/figures/`) usano scale comuni dichiarate. Le figure S6 non sono state rigenerate: lo script S6 e i dati storici non sono stati toccati.
