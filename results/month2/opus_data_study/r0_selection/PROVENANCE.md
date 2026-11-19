# Provenienza R0

Lettura sola del 2026-11-19 da /home/bontempi/official_thesis_s5_runs su Engaging. eval_history.json omette l'eval di arresto; quell'epoca è nel CSV evals/epoch_XXXX/val_eval_per_frame.csv.gz e in progress.json. Gli hash dei checkpoint coincidono con results/month2/s6_sparse_eval/STOP_E_delivery/provenance/runs_provenance.json. Nessun file storico è stato modificato.

Sorgente committata: `s5_curves_source.json`.
Script: `scripts/19_write_opus_r0_tables.py`.
Nessun checkpoint è stato ricaricato per questa tabella. I pesi best/last restano solo sugli originali.

| Run | git | best.pt | last.pt | best epoca | last epoca | stato |
| --- | --- | --- | --- | ---: | ---: | --- |
| s5_sparse_unet_pon_seed0 | `38ac7fe40178982ac1bd33c5d5602a5e7d928d3b` | `b03b896eb4a9f9da3f14969772e832547a09987b19915f9db25a539b07464b7a` | `fad71e8f7ea2618c741046e7fddd9a6aa6147c84c331f7c3535f1180b2761552` | 60 | 630 | stopped_early_plateau |
| s5_sparse_unet_poff_seed0 | `38ac7fe40178982ac1bd33c5d5602a5e7d928d3b` | `0d9d93a164bcdcaf19b1fd8e1eb98a8cf95567b95e6e4d291f864ca2407fb9fc` | `a29eed9d5ca0af06857116f593790fd9c1f76fc35d78ac60340cbd067f3cd7b7` | 140 | 650 | stopped_early_plateau |
| s5_adapted_glu_pon_seed0 | `38ac7fe40178982ac1bd33c5d5602a5e7d928d3b` | `bfd126e5a4289b16c5104fba70f9af8fa3376929df503589ca0a8b792f4639c1` | `a405bffbaf7db0533094a8a657cc4e09b03784d3f4667c85478c59700fbb0f4b` | 440 | 890 | stopped_early_plateau |
| s5_adapted_glu_poff_seed0 | `38ac7fe40178982ac1bd33c5d5602a5e7d928d3b` | `fbae86d5ac1850f99dce69843ea612df984abc1e4dc6fadfd535211db552d1f4` | `a2d9a08db1571fd6de13022cbf498fbd0a847695ebc79389fe57fbf6b461ba8c` | 650 | 980 | stopped_early_plateau |
