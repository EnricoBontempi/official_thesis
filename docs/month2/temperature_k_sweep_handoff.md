# Sweep K — handoff Cursor

Branch `sweep/c149-temperature-k`, derivato dal contratto congelato `10ff6e5d0226d028f0be96100b35667de9b955fd` (padre S7 `2879c1b`). Il codice assisted di `97b2ad7` non è in questo branch e non è stato modificato. Il manifest e il protocollo di Sonnet non sono stati riscritti.

## Contratto usato

- Manifest `configs/observations/month2_temperature_anchors_nested_v1.json`
  - SHA256 del file `a9c997c96cb44489dd1fc0d4484304b44d8beddbf00b73c25baf4e1922a19e54`
  - `manifest_content_sha256` `83fd3b5f2a07e5febc787c384285e265cb4764884fcbc6fb90d22218932f6ae6` (ricalcolato, coincide)
  - Tre layout, replica 0/1/2, 32 siti unici, annidamento verificato contro `nested_temperature_anchor_permutation`
- Protocollo `configs/experiments/month2_temperature_study_protocol.yaml` (SHA256 del file nel commit del contratto). Il file non è YAML valido: alla riga `metrics` c’è testo dopo una sequenza flow (`] on full_field...`). Non è stato corretto, per non cambiare i byte congelati. Lo sweep legge il manifest JSON, non questo YAML.
- Handoff risorse di Sonnet: al massimo 1 GPU normal Advanced per questo sweep, 0 preemptable. CPU `mit_normal` / `mit_amf_advanced_cpu` fuori dal conteggio GPU.

## Cosa calcola

Dalla stessa predizione base, per K = 0/1/4/9/16/32 e per il layout accoppiato alla replica di velocità: free, AO (tau 1), media senza shrink, mediana ordinaria, oracle di campo (solo diagnostico). K pari: media dei due valori centrali. K = 0: AO, media e mediana sono il campo free. Il ponte T9 storico (siti 32/64/95) è `layout_kind=t9_bridge` e non entra nei tre layout. Il protocollo elenca solo free/mean/median; AO e oracle restano perché sono nel mandato e nello Step 2.7.

Baseline: oggetti C149 già fittati, SHA256 `a18f1bf2f4f5a170a04ebd73fb5d438aaa3c3e390cf44e678377381eecfe3e5a`, senza refit. T-block con la coppia storica del manifest `112af741…`: p_off (λ 0.001, w_T 0.5), p_on (λ 0.01, w_T 0.5). Il w_T = 5 salvato nell’artefatto è il valore del fit free e non è usato come peso del T-block. Nessun campo T-block è stato scelto su VAL in questo sweep.

Ruoli: seen220 e VAL123. Checkpoint neurali: best e last. Nessun best_seen, nessun TEST, nessuna reference760.

## Esecuzione

Primo tentativo `e9f355f`, checkout `/home/bontempi/projects/official_thesis_k_sweep`. Smoke GPU `30032017` e smoke CPU `30032019` sono falliti prima di qualunque `COMPLETE`: la macro del gate rinominava `s7_rmse_K` sopra `full_rmse_K` già presente, e la directory baseline esisteva perché conteneva solo `slurm/`. I dipendenti `30032021`, `30032023`, `30032025` sono stati annullati dallo scheduler, non rilanciati.

Codice del secondo tentativo: `54a3b4d3a177b731ed7eef45030c3a49b0986c16`, checkout detached `/home/bontempi/projects/official_thesis_k_sweep_v2`. I baseline CPU sono partiti quando i quattro slot GPU erano occupati dai job Sonnet `30032333`, `30032338`, `30032339` e `30032341`. La GPU è stata sottomessa dopo, con due slot Sonnet ancora in coda.

| Job | Cosa | Risorse |
| --- | --- | --- |
| 30032411 | smoke CPU baseline, 4 frame, `COMPLETE` exit 0, macro 0 | `mit_normal` / `mit_amf_advanced_cpu`, 8 CPU, 64G |
| 30032412 | baseline CPU completo, `COMPLETE` exit 0, macro 0 su seen220 e VAL123 | stessa CPU, 3 min |
| 30032664 array 0–1 %1 | smoke GPU, U-Net on s1 e GLU on s0, 4 frame | `mit_normal_gpu` / `mit_amf_advanced_gpu`, 1×L40S, 40 min |
| 30032665 array 0–11 %1 | sweep GPU 12 run, `afterok:30032664` | stessa GPU, 3 h per task, una alla volta |
| 30032666 | aggregazione, `afterok:30032665` (il baseline `30032412` è già `COMPLETE`) | CPU, 40 min |
| 30035232 | chiusura, `afterany:30032666`, senza nuovi calcoli | `mit_normal` / `mit_amf_advanced_cpu`, 2 CPU, 8G, 20 min |

Esito riletto da `sacct`, non dalla coda. `30032411`, `30032412`, `30032664` (task 0 e 1) e `30032665` (task 0–11) sono `COMPLETED` exit 0. I dodici `COMPLETE.json` e il baseline `COMPLETE.json` ci sono. Gate: 144 neurali e 24 baseline, tutti `ok`, macro 0 K, scarto per-frame 0 K.

`30032666` è `FAILED` in 13 s: `no per-frame CSV under .../official_thesis_k_sweep_v1`. I CSV stanno in `<run>/per_frame/` (144 file, circa 1.1 GB) e in `baselines_v1/per_frame/` (36 file). `30035232` ha scritto `BLOCCO.md` per l’assenza del pacchetto e non ha rilanciato nulla. Quel file resta come verbale del primo aggregato.

La lettura dei CSV già scritti è il job `30081928`, `COMPLETED` exit 0, codice `30593985337f6367f5e974807cb54865b00d678f`, checkout `/home/bontempi/projects/official_thesis_k_sweep_agg`. Il checkout dei forward è rimasto `54a3b4d`. `30081718` è una lettura precedente, `FAILED` prima di scrivere, perché la chiave univa p_on e p_off. Nessun forward è stato rifatto.

Pacchetto cluster `/home/bontempi/official_thesis_k_sweep_package_v1`: `INDEX.md`, `checks.json`, `by_seed.csv`, `versus_free.csv`, `across_seeds.csv`, `per_condition.csv`, `figures/val_common_rmse_vs_K.png`. La copia in git è `results/month2/temperature_k_sweep/`. La figura in git è ridisegnata dagli stessi `by_seed.csv`: il PNG del job aveva la legenda solo sul pannello GappyPOD. `checks.json` ha differenza 0 a K=0 per AO, media e mediana, e media uguale a mediana a K=1.

Il manifest dei forward, SHA256 `a9c997c96cb44489dd1fc0d4484304b44d8beddbf00b73c25baf4e1922a19e54`, coincide con i file negli checkout assisted `official_thesis_t`, `official_thesis_t_run`, `official_thesis_t_chain` e `official_thesis_t_eval`. Nessun file di Sonnet è stato modificato. Nessuno stimatore è stato scelto.

Alla submission GPU la coda Advanced conteneva due job Sonnet (`30032653`, `30032656`, `t_checks`). Lo sweep ne aggiunge due, di cui solo lo smoke può partire; il full resta `afterok`. Totale in coda: 4.

Smoke GPU `30032664` completato, exit 0, `SMOKE_OK.json` per entrambi i task. U-Net on s1 e GLU on s0, best e last, seen e VAL, repliche 0/1/2: gate true, |macro| sotto 5e-6 K, scarto per-frame massimo sotto 2e-5 K. Il selettore smoke è quello di S7: 4 posizioni più, su VAL, i frame 1000, quindi 7 frame VAL. Lo sweep scientifico non usa quel sottoinsieme. Il task 0 dell’array `30032665` è partito dopo lo smoke.

I forward usano `scripts/38` e `_forward` di `scripts/17`, batch 8, fp32, TF32 spento, una L40S. Il codice GLU non è stato modificato. Il documento dei numeri è `docs/month2/11_temperature_k_sweep.md`.
