# Median9 second pass — disponibilità e soli frame esemplificativi

Questo non è il risultato su seen220/VAL123. I CSV per-frame di Astra salvano `anchor_mean_resid_K`, non i nove residui. La mediana non si ricostruisce da quella media.

`example_frame1000_rep0/` usa i quattro `val_frame1000_rep0_fields.npz` neurali (tre condizioni VAL, frame 1000, replica 0). Free, oracle e AO9 rigenerati da quei campi coincidono con il CSV primario entro 4.1e-5 K, sotto la tolleranza 0.05 K già usata per il replay. I riferimenti non sono stati modificati.

Le baseline non hanno campi salvati. `T_only_const` non riceve mean9 né median9.

Il confronto mediano–AO9 cambia insieme stimatore e shrink. `mean9` è la media dei nove residui senza tau. Dettaglio e limiti: `docs/month2/astra_median9_second_pass_handoff.md`.
