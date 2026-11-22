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

Alla submission GPU la coda Advanced conteneva due job Sonnet (`30032653`, `30032656`, `t_checks`). Lo sweep ne aggiunge due, di cui solo lo smoke può partire; il full resta `afterok`. Totale in coda: 4.

Forward neurale, quando partirà: `scripts/38` + `_forward` di `scripts/17`, batch 8, fp32, TF32 spento, una sola L40S, array `%1`. Il codice GLU non è modificato.
