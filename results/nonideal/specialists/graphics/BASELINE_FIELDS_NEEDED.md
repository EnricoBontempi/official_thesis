# Campi mancanti per la stessa raccolta dei 12 modelli originali

La raccolta dei 30 specialisti chiede, per il checkpoint best, replica 0, frame 1000, le tre condizioni VAL, i sei scenari:
`pred_T_K`, `true_T_K`, `error_signed_K` (predizione − verità), `original_sites`, `surviving_sites`, `removed_sites`, `g1_region_mask`,
nel file `fields_val_frame1000_best.npz` con `fields_manifest.json` (formato `nonideal_specialist_fields_v1`).

Quei file non esistono per i 12 modelli del Blocco II. Non vanno ricostruiti dal modulo dell'errore delle figure già pubblicate.
Le figure pubblicate `fig5_fields_*.png` coprono solo quattro configurazioni a seed 0 (U-Net/GLU × con e senza φ/NH3) e mostrano il modulo dell'errore, non la temperatura predetta.

Sul cluster, per tutti i 12 run, esiste invece `fields_figure_frames.npz` (formato Block II).
Ispezione di `O-C149-on-s0`: 150 chiavi; VAL alle condizioni `0NH3_1.05`, `20NH3_1.1`, `40NH3_0.9` con frame_index 0 (non 1000);
seen a `0NH3_0.85` e `80NH3_1.2` con frame 1600; array `pred_T_K`, `true_T_K`, `kept_sites`, `removed_sites`, `region_mask`.
Mancano, rispetto al contratto dei 30: frame 1000, `error_signed_K` salvato, `original_sites`, il nome `surviving_sites`, il manifest e il controllo di RMSE.

Elenco dei file presenti (formato vecchio) e di ciò che servirebbe al loro posto:

- `O-C149-on-s0`
  - vecchio, se c'è: `/home/bontempi/official_thesis_nonideal_b2_eval/O-C149-on-s0/fields_figure_frames.npz` (presente sul cluster; frame presenti nel file vecchio: [0, 1600]. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30.)
  - richiesto, assente: `/home/bontempi/official_thesis_nonideal_b2_eval/O-C149-on-s0/fields_val_frame1000_best.npz` + `fields_manifest.json`
  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1
- `S7-C149-unet-on-s1`
  - vecchio, se c'è: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-unet-on-s1/fields_figure_frames.npz` (presente sul cluster; frame presenti nel file vecchio: [0, 1600]. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30.)
  - richiesto, assente: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-unet-on-s1/fields_val_frame1000_best.npz` + `fields_manifest.json`
  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1
- `S7-C149-unet-on-s2`
  - vecchio, se c'è: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-unet-on-s2/fields_figure_frames.npz` (presente sul cluster; frame presenti nel file vecchio: [0, 1600]. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30.)
  - richiesto, assente: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-unet-on-s2/fields_val_frame1000_best.npz` + `fields_manifest.json`
  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1
- `S7-C149-glu-on-s0`
  - vecchio, se c'è: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-glu-on-s0/fields_figure_frames.npz` (presente sul cluster; frame presenti nel file vecchio: [0, 1600]. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30.)
  - richiesto, assente: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-glu-on-s0/fields_val_frame1000_best.npz` + `fields_manifest.json`
  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1
- `S7-C149-glu-on-s1`
  - vecchio, se c'è: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-glu-on-s1/fields_figure_frames.npz` (presente sul cluster; frame presenti nel file vecchio: [0, 1600]. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30.)
  - richiesto, assente: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-glu-on-s1/fields_val_frame1000_best.npz` + `fields_manifest.json`
  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1
- `S7-C149-glu-on-s2`
  - vecchio, se c'è: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-glu-on-s2/fields_figure_frames.npz` (presente sul cluster; frame presenti nel file vecchio: [0, 1600]. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30.)
  - richiesto, assente: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-glu-on-s2/fields_val_frame1000_best.npz` + `fields_manifest.json`
  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1
- `O-C149-off-s0`
  - vecchio, se c'è: `/home/bontempi/official_thesis_nonideal_b2_eval/O-C149-off-s0/fields_figure_frames.npz` (presente sul cluster; frame presenti nel file vecchio: [0, 1600]. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30.)
  - richiesto, assente: `/home/bontempi/official_thesis_nonideal_b2_eval/O-C149-off-s0/fields_val_frame1000_best.npz` + `fields_manifest.json`
  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1
- `S7-C149-unet-off-s1`
  - vecchio, se c'è: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-unet-off-s1/fields_figure_frames.npz` (presente sul cluster; frame presenti nel file vecchio: [0, 1600]. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30.)
  - richiesto, assente: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-unet-off-s1/fields_val_frame1000_best.npz` + `fields_manifest.json`
  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1
- `S7-C149-unet-off-s2`
  - vecchio, se c'è: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-unet-off-s2/fields_figure_frames.npz` (presente sul cluster; frame presenti nel file vecchio: [0, 1600]. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30.)
  - richiesto, assente: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-unet-off-s2/fields_val_frame1000_best.npz` + `fields_manifest.json`
  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1
- `S7-C149-glu-off-s0`
  - vecchio, se c'è: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-glu-off-s0/fields_figure_frames.npz` (presente sul cluster; frame presenti nel file vecchio: [0, 1600]. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30.)
  - richiesto, assente: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-glu-off-s0/fields_val_frame1000_best.npz` + `fields_manifest.json`
  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1
- `S7-C149-glu-off-s1`
  - vecchio, se c'è: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-glu-off-s1/fields_figure_frames.npz` (presente sul cluster; frame presenti nel file vecchio: [0, 1600]. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30.)
  - richiesto, assente: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-glu-off-s1/fields_val_frame1000_best.npz` + `fields_manifest.json`
  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1
- `S7-C149-glu-off-s2`
  - vecchio, se c'è: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-glu-off-s2/fields_figure_frames.npz` (presente sul cluster; frame presenti nel file vecchio: [0, 1600]. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30.)
  - richiesto, assente: `/home/bontempi/official_thesis_nonideal_b2_eval/S7-C149-glu-off-s2/fields_val_frame1000_best.npz` + `fields_manifest.json`
  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1

Il recupero di questi campi è un'inferenza in più. Non è stato avviato: va concordato con chi gestisce la valutazione, senza una campagna autonoma.

