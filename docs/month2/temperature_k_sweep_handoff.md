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

Codice eseguito: `e9f355feb53e7a16e431f355fa540b1e5d6bb5a9`. Checkout detached `/home/bontempi/projects/official_thesis_k_sweep`. Alla submission la coda Advanced conteneva già i job Sonnet `30032014` e `30032016` (`t_checks`, pending). Questo sweep chiede una sola L40S alla volta.

| Job | Cosa | Risorse |
| --- | --- | --- |
| 30032017 array 0–1 %1 | smoke GPU, U-Net on s1 e GLU on s0, 4 frame | `mit_normal_gpu` / `mit_amf_advanced_gpu`, 1×L40S, 40 min |
| 30032019 | smoke CPU baseline, 4 frame | `mit_normal` / `mit_amf_advanced_cpu`, 8 CPU, 64G, 30 min |
| 30032021 array 0–11 %1 | sweep GPU 12 run, `afterok:30032017` | stessa GPU, 3 h per task |
| 30032023 | baseline CPU completo, `afterok:30032019` | stessa CPU, 2 h |
| 30032025 | aggregazione, `afterok:30032021:30032023` | CPU, 40 min |

Se lo smoke fallisce, i job dipendenti vengono annullati (`kill-on-invalid-dep`) e non vengono rilanciati da soli. Forward neurale: `scripts/38` + `_forward` di `scripts/17`, batch 8, fp32, TF32 spento. Il codice GLU non è modificato.
