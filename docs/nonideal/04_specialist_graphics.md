# Pacchetto grafico dei 30 specialisti

Codice, test e launcher CPU. Nessun training e nessuna inferenza. I numeri scientifici non sono in questo file: li scrive il job, in `NOTE.md`, solo se freeze, campi e pacchetto numerico sono completi e coerenti.

## Contratto di lettura

File: `configs/nonideal/specialist_graphics_contract_v1.json`. È lo stesso formato descritto in `docs/nonideal/03_specialist_handoff.md`.

- Checkpoint `best` di ciascuno dei 30 run `NI-{glu|unet}-on-{N1|N2|G1|G1eq|N1G1}-s{0|1|2}`.
- Campi: `<eval_root>/<run_id>/best/fields_val_frame1000_best.npz` e `fields_manifest.json` (`nonideal_specialist_fields_v1`). Chiavi `<scenario>|<condizione>|<campo>`. Frame 1000, replica 0, ruolo VAL, sei scenari, tre condizioni. Errore firmato = predizione − verità. La temperatura predetta non si ricostruisce dal modulo dell’errore.
- Indice: `<dest>/package/fields_index.json`.
- Metriche: CSV per frame del best e `tables/per_run_scenario.csv` del pacchetto. Il job ricalcola getti, zona calda e picco dai CSV, perché la tabella aggregata non li conserva, e rifiuta se la RMSE non torna entro 1e-4 K.
- Baseline pulite: il CSV per frame del Blocco II, hash `cb96687cf3614c5393bc9658ec3469980313413f495f32930fd2e810ffc070b3`. Gli hash dei siti superstiti devono coincidere con il modello originale della stessa famiglia e dello stesso seed.
- Se manca un run, un gate, una chiave, uno scenario o la compatibilità delle osservazioni, il job scrive `GRAPHICS_BLOCK_REPORT.md` e non pubblica `COMPLETE.json`.

## Cosa produce il job

Sotto `<dest>/graphics`, solo a matrice completa:

- matrici TRAIN × EVAL, U-Net e GLU separate, seen e VAL separate, valori assoluti e differenza rispetto alla baseline nello stesso scenario;
- media, minimo–massimo e i tre seed di training (dispersione fra training, non un intervallo di confidenza); gli scenari sono categorie;
- una PDF per ciascuno dei 30 run best, una pagina per condizione VAL, più i PNG;
- per famiglia e regime di training: media dei tre campi, deviazione standard fra i tre training, errore del campo medio (la sua RMSE non è la media delle RMSE);
- `NOTE.md`, `INDEX.md`, `figure_manifest.json`;
- `BASELINE_FIELDS_NEEDED.md`: i 12 modelli originali hanno `fields_figure_frames.npz` (frame 0 in VAL, frame 1600 in seen, senza il contratto del frame 1000). Le figure pubblicate `fig5_fields_*.png` coprono quattro configurazioni a seed 0. Quei campi non vengono ricalcolati qui.

## Aggancio

Un solo job, `ni_graphics_spec`, sottomesso dallo script di aggregazione già in catena quando trova `<dest>/post_hook.path`. Nessuna seconda valutazione. Lo script è `scripts/engaging/nonideal_graphics.sbatch`. L’output è `<dest>/graphics`. Il job gira sul cluster: non richiede che questo PC resti acceso.

Le figure dei 12 modelli già pubblicate restano in `results/nonideal/block2/package/figures/` e in `results/nonideal/block2/reading/`.

## Esecuzione e correzione (27 novembre 2026)

Il job del post-hook (30685595, checkout `1e498dcc`) è fallito in 2 s con exit 128. Causa: Slurm esegue una copia dello script dalla sua directory di spool, quindi `ROOT` ricavato da `$0` non era il checkout e `git rev-parse HEAD` falliva prima del codice grafico (log: `slurm-30685595.out`, «not a git repository»). Le variabili `NI_*` erano corrette (stesse di `nonideal_aggregate_spec.sbatch`).

Correzione: `NI_GRAPHICS_ROOT` ed `EXPECT_GRAPHICS_SHA` sono obbligatori (nessun default), il launcher [nonideal_graphics_submit.sh](../../scripts/engaging/nonideal_graphics_submit.sh) verifica checkout, SHA e file di ingresso e passa `--chdir`, `--output`, `--error`; il checkout e lo SHA realmente eseguiti compaiono nel log, e lo SHA in `figure_manifest.json`. Test: `tests/test_nonideal_graphics_launcher.py`. Job risottomesso: 30711670, `COMPLETED`, `GRAPHICS_OK` (checkout `c5963373`, `/home/bontempi/projects/official_thesis_spec_graphics_v2`); tracciato in [graphics_hook_v2.json](../../results/nonideal/specialists/graphics_hook_v2.json). Risultati e lettura: [05_specialist_results.md](05_specialist_results.md).
