# S7 — dieci training free C149: consegna operativa (20 novembre 2026)

**Stato:** submission eseguita, training in corso. Nessuna valutazione, nessuna dichiarazione di S7/sparse/Mese 2 conclusi.
**Branch:** `s7/c149-free-campaign`. **SHA eseguito:** `f792b73388d43f008e8979f1aad0d1fa961cb559` (checkout congelato, detached, pulito salvo `.git_commit`).

## Risorse e decisione
Enrico ha chiesto preemptable con account/QoS Advanced. Lo scheduler rifiuta la combinazione: `sbatch --test-only -p mit_preemptable -A mit_amf_advanced_gpu [--qos=mit_amf_advanced_gpu|normal]` → *Invalid account or account/partition combination*. L'associazione `mit_amf_advanced_gpu` esiste solo su `mit_normal_gpu` (QoS `mit_amf_advanced_gpu`, `normal`); su `mit_preemptable` è consentito solo l'account `mit_general` (QoS `normal`). Enrico ha scelto di restare su **`mit_normal_gpu` + `mit_amf_advanced_gpu`, non preemptable**. Nessun cambio silenzioso.
Array **29799680**, `--array=0-9%4`, L40S×1, 8 CPU, 64G, `--time=06:00:00` (MaxTime partizione 6h), cap interno 5.5 h per attempt (`--wall-hours 5.5`), `--requeue`. QoS `mit_amf_advanced_gpu`: max 4 GPU/utente, quindi un solo array con %4; nessun altro job GPU dell'utente al momento della submission (`s7_submit.sh` lo impone).

## Matrice (task → run)
| Task | Run | Famiglia | p | seed | Job ID | Stato osservato 14:18–14:20 EST |
|---|---|---|---|---:|---|---|
| 0 | S7-C149-unet-on-s1 | U-Net | on | 1 | 29799990 | RUNNING, epoca 9, update 855 |
| 1 | S7-C149-unet-off-s1 | U-Net | off | 1 | 29799991 | RUNNING, epoca 9 |
| 2 | S7-C149-unet-on-s2 | U-Net | on | 2 | 29799992 | RUNNING, epoca 9 |
| 3 | S7-C149-unet-off-s2 | U-Net | off | 2 | 29799993 | RUNNING, epoca 9 |
| 4 | S7-C149-glu-on-s0 | GLU | on | 0 | 29799680_4 | PENDING (JobArrayTaskLimit) |
| 5 | S7-C149-glu-off-s0 | GLU | off | 0 | 29799680_5 | PENDING |
| 6 | S7-C149-glu-on-s1 | GLU | on | 1 | 29799680_6 | PENDING |
| 7 | S7-C149-glu-off-s1 | GLU | off | 1 | 29799680_7 | PENDING |
| 8 | S7-C149-glu-on-s2 | GLU | on | 2 | 29799680_8 | PENDING |
| 9 | S7-C149-glu-off-s2 | GLU | off | 2 | 29799680_9 | PENDING |

Verificati in esecuzione: solo task 0–3 (piano `fresh`, hash cache `fa1c9358…`, contratto maschere ok, init U-Net on/off uguale a pari seed, `fit_begin`, epoche reali). I sei GLU **non sono ancora partiti** e non sono dichiarati funzionanti in campagna (lo smoke GLU è PASS).

Riusati, **non rilanciati**: O-C149-on-s0 (training 29645053) e O-C149-off-s0 (29645054), SHA `d6332000…`. File, checkpoint, hash e config verificati in `results/month2/s7_c149_comparison/prep/reused_unet_verification.json` (config SHA identici ai file a `d633200`; early-stop a 660/710 epoche).

## Input e hash
Cache C149: copia in home byte-identica della cache pool (20 file, aggregato `fa1c9358deb3bda0313cd74f9ab6855af87bc969c4069a463073d4c4120f03b4`), `/home/bontempi/official_thesis_data/month2_dense_unet_cache_c149`. Motivo: nel primo smoke l'automount `/orcd/pool/005/bontempi` andava in timeout (`stat` > 120 s) su node3004 e node4102; l'originale in pool è intatto. Stats fit-only 760 `678263fb…`. Maschere: hash file manifest `2e2d8ff6…` e aggregato payload `7be78f35…` verificati separatamente e registrati in `run_identity.json` (`eval_mask_manifest_file_sha256`, `eval_mask_aggregate_payload_sha256`; la chiave storica `mask_payload_sha256` è il hash del file). Nessun fallback di rigenerazione.

## Codice integrato (da d633200 + correzioni)
Dal branch Opus O5 non serviva codice di training: l'evaluator O5 è solo U-Net e resta per la fase di valutazione. Correzioni sul trainer, trovate o motivate dallo smoke: (1) il tracker `best_seen` non sopravviveva al resume; (2) un checkpoint scritto all'early-stop, ripreso, avrebbe continuato ad allenare; (3) resume su GPU: gli stati RNG caricati con `map_location=cuda` andavano riportati su CPU (errore reale dello smoke: `TypeError: RNG state must be a torch.ByteTensor`). Ricetta scientifica invariata (test di confronto con le config Opus C149).

## Smoke (non scientifici), job 29797847 (U-Net) e 29797848 (GLU), L40S, SHA `d1a0ebd`
Artefatti: `results/month2/s7_c149_comparison/prep/smoke/{unet,glu}/`. Entrambi: C149 aperto con id/conteggi 2980/220/123; isolamento (mutare vi/vj nascosti → input, token e predizione bit-identici; mutare gli osservati → cambiano); p_on risponde a p, p_off no; init on/off uguale; evaluator 1/5/20% × repliche 0/1/2 con 164/819/3277 siti; step reale batch 8. Memoria GLU: eval 0.48 GiB al 20% (batch 8), 0.07 GiB batch 1; training batch 8 4.3 GiB. U-Net: train 0.49 GiB.
Resume: kill duro (`os._exit(137)`) a metà epoca 3 → requeue → ripresa da `last.pt` epoca 2: stato, contatori, LR, sampler, stream (epoca 2 servita due volte con identici indici) uguali al continuo; pesi entro rumore GPU (diff max resume 1.5e-5 GLU / 7.2e-5 U-Net; controllo run-to-run senza interruzione 1.8e-5 / 1.0e-5; un campione ciascuno, non bitwise). Continuazione da cap interno di wall-time con **`scontrol requeue` reale** (U-Net: ripartito su altro nodo con `SLURM_RESTART_COUNT=1`, ripreso da `last.pt`); requeue di task di array verificato su un array CPU (29799260).
Limite dichiarato: la preemption vera non è testabile sulla partizione scelta (PreemptMode OFF); il kill duro e il requeue reale coprono lo stesso percorso di ripresa.

## Comportamento a interruzione
`scripts/engaging/s7_run_task.sh` + `src/combustion_recon/training/s7_launch.py`: nuova directory mai sovrascritta; un requeue dello stesso job riprende solo da `checkpoints/last.pt` completo a confine d'epoca (`last.pt` scritto atomico ogni epoca); mai da `partial_interrupt.pt`/`best.pt`; senza checkpoint completo il riavvio è registrato come `fresh_restart_no_checkpoint` (file spostati in `_discarded_attempts`, non cancellati); summary terminale → no-op; `stopped_error` o checkpoint corrotto → rifiuto. Al cap interno 5.5 h il task fa `scontrol requeue` (continuazione registrata, non plateau). `MaxBatchRequeue=5`: oltre 5 requeue il job viene messo in hold. Ledger in `official_thesis_s7_c149_runs/_ledger/<run>.jsonl`. Timeout/SIGTERM sono loggati dal trap.

## Percorsi
- Checkout: `/home/bontempi/projects/official_thesis_s7`
- Output: `/home/bontempi/official_thesis_s7_c149_runs/<run_id>/` (ledger in `_ledger/`)
- Log Slurm: `/home/bontempi/official_thesis_s7_c149_runs/slurm/slurm-29799680_<task>.out|.err`
- Smoke: `/home/bontempi/official_thesis_s7_smoke_runs/`

## Durata stimata (stime, non garanzie)
Misurato U-Net C149 s0: 20.4 s/epoca, 660–710 epoche, 3.7–4.0 h. Misurato GLU S5: 9.4 s/epoca, 890–980 epoche, 2.3–2.6 h (stessa dimensione di epoca). I primi task S7 mostrano ~10 s/epoca all'avvio (da riconfermare). A 1000 epoche un U-Net può superare 5.5 h → una continuazione. Calcolo totale ≈ 4×4 h + 6×2.5 h ≈ 31 GPU-h; con 4 GPU almeno ≈ 8 h di orologio senza coda. Coda: smoke partiti dopo ~40 min (primo giro) e 4 min (secondo), requeue reale ripartito dopo ~4 min; la coda `mit_normal_gpu` aveva ~1200 job pending.

## Comandi manuali
```
ssh engaging
squeue -u bontempi -r -o "%i %j %T %M %r %N"
sacct -j 29799680 -X -o JobID%18,State,ExitCode,Elapsed,NodeList
R=/home/bontempi/official_thesis_s7_c149_runs
~/envs/official_thesis/bin/python ~/projects/official_thesis_s7/scripts/engaging/s7_plan.py last $R/S7-C149-unet-on-s1
tail -f $R/slurm/slurm-29799680_0.out
cat $R/_ledger/S7-C149-unet-on-s1.jsonl
```

## Blocchi residui
- Preemptable+Advanced non consentito (vincolo sopra); scelta di Enrico: normal.
- I sei GLU sono in attesa dei posti (max 4 GPU).
- Nessuna valutazione finale, snapshot, baseline C149 o residuo GappyPOD S6: fuori da questo incarico. L'evaluator O5 (branch `opus/o5-eval-attempt2`) è U-Net-only; per GLU C149 serve estensione in fase di valutazione.
- STOP: nessuna dichiarazione di S7, sparse o Mese 2 conclusi.
