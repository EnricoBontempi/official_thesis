# Blocco II — runbook (12 modelli congelati)

**Stato al 24 novembre 2026:** codice, configurazione e test CPU pronti e pubblicati. **Smoke GPU e campagna NON ancora eseguiti**: dal ambiente di sviluppo non c'è un socket SSH attivo verso Engaging (richiede password MIT + Duo interattivi) e la sessione non era interattiva.

## Che cosa è pronto
| Cosa | Dove |
|---|---|
| Specifica a 12 modelli (revisione; v1 intatta) | `configs/nonideal/degradation_spec_v1_12models.yaml` |
| Operatori (Blocco I) | `src/combustion_recon/data/nonideal_observations.py` |
| Valutazione, gate, aggregazione | `src/combustion_recon/evaluation/nonideal_eval.py` |
| Task per checkpoint (R0 + gate, poi 5 celle) | `scripts/60_nonideal_eval.py` |
| Verifica / aggregazione + figure + riepilogo | `scripts/61_nonideal_verify.py`, `scripts/62_nonideal_aggregate.py` |
| Catena Slurm | `scripts/engaging/nonideal_{eval,verify,aggregate}.sbatch`, `nonideal_submit.sh` |
| Test | `tests/test_nonideal_eval.py` (+ Blocco I) |

Selftest CPU (pesi casuali, solo frame `seen_train_eval`, nessun VAL): eseguito su un GLU p_off e due U-Net; percorso completo gate → sei celle → readback funzionante. Non è un risultato scientifico.

## Sequenza sul cluster (dopo `ssh engaging` con socket attivo)
1. Checkout congelato (nuovo, mai riusato): `git clone --branch <branch> https://github.com/EnricoBontempi/official_thesis.git ~/projects/official_thesis_nonideal_b2 && cd ~/projects/official_thesis_nonideal_b2 && git checkout <SHA>`.
2. Smoke (4 percorsi famiglia × p, un checkpoint ciascuno, 8 frame per ruolo, tutte le sei celle):
   `scripts/engaging/nonideal_submit.sh --tip $PWD --sha <SHA> --spec configs/nonideal/degradation_spec_v1_12models.yaml --out-root ~/official_thesis_nonideal_smoke --dest ~/official_thesis_nonideal_smoke_pkg --mode smoke --smoke-frames 8 --runs S7-C149-unet-on-s1,S7-C149-unet-off-s1,S7-C149-glu-on-s0,S7-C149-glu-off-s0 --partition mit_normal_gpu --account mit_amf_advanced_gpu --qos mit_amf_advanced_gpu --time 01:00:00 --max-gpus 4`
   Controllare `SMOKE_OK.json`, `setting_costs.csv`, memoria in `provenance.json`; ridimensionare walltime con i tempi misurati.
3. Campagna: stesso comando con `--mode full`, `--out-root` e `--dest` nuovi, senza `--runs`, `--time` dimensionato dallo smoke.
4. Riepilogo automatico: `<dest>/FINAL_SUMMARY.md` oppure `<dest>/BLOCK_REPORT.md`; pacchetto in `<dest>/package/`.

## Regole di arresto
Ogni task esegue prima R0 e il gate (macro entro 0.05 K dal riferimento S7 per ruolo/replica). Se fallisce: `GATE_FAILED.json`, `CAMPAIGN_HALT.json`, `scancel` dell'array, nessuna cella degradata. Nessuna soglia modificata, nessun retraining. La verifica rifiuta matrici incomplete, hash o input diversi fra run e gate mancanti.
