# Sparse reconstruction — S1–S6 technical status

**Status:** **STOP E** — confronto sparse seed 0 consolidato. GappyPOD nested gate 5% **passato** (job **29601906**); neurali best/last 1/5/20% + snapshot-swap completi (best riusati; last+swap jobs **29601907–29601910**). Nessun training S7 / nessuna scelta definitiva di finalisti.  
**Branch:** `mission/dense-unet`.  
**Execution tip SHA (S5/S6/Gappy):** `38ac7fe40178982ac1bd33c5d5602a5e7d928d3b`.  
**Consegna revisori:** `results/month2/s6_sparse_eval/STOP_E_delivery/` (+ Gappy `results/month2/gappypod_nested_1_20/`).  
**Record S4:** [s4_profiling_submission_record.md](s4_profiling_submission_record.md).  
**Record S5:** [s5_neural_freeze_record.md](s5_neural_freeze_record.md).

---

## 1. S1 — Maschere eval persistite

| Voce | Valore |
|---|---|
| Format | `month2_sparse_eval_masks_v1` / algoritmo `nested_subset_extension_v1` |
| Contratto | train_fit 760 + seen_train_eval 220 + val_eval 123; **no TEST**; K=0; 2C; coverages 1/5/20%; replicas 0/1/2 |
| Seed eval | `mask_seed=20261116`, `layout_id=random_sparse_primary` |
| Self-check 5% | **3309/3309** seed regeneration ↔ persisted sites (same `mask_hash`) — **not** by itself full historical baseline equivalence |
| Verified examples @5% | 3/3 match `month2_observations.json` |
| **Baseline join @5%** | **3309/3309** hex `observation_hash` ↔ persisted `hash_5pct` (`scripts/13_join_sparse_masks_vs_baseline.py`); sentinel `dense_full_velocity` rows **6618** counted separately |
| Nesting | 164 ⊂ 819 ⊂ 3277 verified per instance |
| Manifest (Git) | `data/manifests/month2_sparse_eval_masks_manifest.json` |
| Join report (Git) | `data/manifests/month2_sparse_mask_baseline_join_5pct.json` |
| Payload (out of Git) | `data/processed/month2_sparse_eval_masks/` |
| Aggregate SHA256 | `7be78f35ffea2ebc598adce8cec39dc719339aac17ecadde3137f2064891ffeb` |

**Hash semantics:** at K=0, baseline `observation_hash` for sparse methods equals `mask_hash` (SHA256 of uint8 HxW velocity mask). Nested 1%/20% remain **new** artifacts.

### Regeneration / load / verify

```bash
.venv/bin/python scripts/11_materialize_sparse_eval_masks.py --overwrite
.venv/bin/python scripts/13_join_sparse_masks_vs_baseline.py
```

Protocol: `configs/experiments/month2_sparse_protocol.yaml`.

---

## 2. S1 — GLU audit + S3 implementation

Audit: [04_glu_release_adaptation.md](04_glu_release_adaptation.md) (updated for tie-break / fixed domain / non-vacuous T isolation). Upstream tree untouched.  
Wrapper: `src/combustion_recon/models/glu_reconstructor.py` + `configs/models/glu_minimal.yaml`.

---

## 3. S2 — Sparse U-Net (unchanged capacity)

| Voce | Valore |
|---|---|
| Channels | `[vi_obs_norm, vj_obs_norm, Mv, x_scaled, y_scaled, nh3_norm, phi_norm]` |
| Parameters b=32 | **4 903 041** |
| Train masks | `train_mask_seed=20261118` + `set_epoch` |
| Diagnostic trainer | **Incomplete** for science: no warmup/plateau/VAL counters; full scientific resume = **S4 freeze requirement** |

S2 test hardenings (S3 mission): real sample→input isolation path; p_off rebuild from mutated source p; positive controls after learning; tautologies removed; `rtol=0` for absolute equality.

---

## 4. S3 — Adapted deterministic GLU

| Voce | Valore |
|---|---|
| Name | GLU adattato — ablation deterministica (not full paper reproduction) |
| Params | **1 154 578** |
| Encoder | `ThesisDomainAdaptiveEncoder` (upstream structure + additive `W_p` bias-free) |
| Decoder | `ThesisSoftDomainAdaptiveReconstructor` — public release `forward` with **one** selection delta |
| Defaults | d=32, heads=4, layers=4, latents=64, top_k=16, σ=0.05, `per_sensor_sigma=False`, CLS on; encoder `num_freqs=256` (stage-0), decoder `num_freqs=64` (class default) |
| Geometry | Fixed domain from `month2_dense_unet_cache/geometry.npz` (or explicit synth axes); buffers `domain_center` / `domain_L` / extents; `q=2(r-c)/L`, L=max(extent_x,extent_y) |
| Chunk eval | 512 queries; full vs chunk **max_abs=0** (atol declared 1e-5); chunked **backward not claimed** |

### Post-S3 circumscribed corrections (this STOP B delivery)

| # | Problem | Fix |
|---|---|---|
| 1 | Comment claimed `torch.topk` prefers lower indices on ties; N=k=2 test never exercised boundary choice | `topk_distance_then_site_index` (distance↑, site_index↑); N>k expected-index + permutation + unequal-distance tests |
| 2 | `compute_isotropic_geometry` from sensors∪queries (call/batch dependent) | Domain from verified cache axes / explicit geometry; persisted buffers; explicit error if missing |
| 3 | Vacuous T isolation (discarded `t_mut`, same tensors re-inferred) | Mutate source T; rebuild via `build_canonical_observation`; assert target changed, GLU inputs/pred identical |

**Prior diagnostic full-grid inputs:** sensors∪full 16384 queries → center/L **identical** to domain axes (diff 0). Do not declare all prior full/chunk results wrong: chunking already mapped q before chunks. Subset-only call-site maps **can** differ (corner L_diff≈0.060). **No** re-run of diagnostic overfit metrics; manifests/metrics unchanged.

### Geometry invariants (CPU; rtol=0)

Declared: `ATOL_GEOM_PRED=1e-5`, `ATOL_CHUNK=1e-5`, `ATOL_RELOAD=1e-6`.  
Measured max_abs: A subset **0**, B batch **0**, D chunk **0**, E reload **0**. Center/L unchanged across layouts. Env: Python 3.13.13, torch 2.15.0+cu132, CUDA unavailable in check — **no** universal CPU/GPU reproducibility claim.

Domain (cache): center ≈ `(-9.60e-6, 0.111)`, L ≈ `0.09628`.

### Diagnostic overfit (same 8 TRAIN frames as S2, full 16384 queries, CPU) — metrics **not rewritten**

| Run | Epochs | Updates | Loss first→last | Wall |
|---|---:|---:|---|---:|
| Fixed masks | 8 | 64 | **1.262 → 0.367** | ~17 s |
| Variable masks | 5 | 40 | **1.262 → 0.734** | ~19 s |

Artifacts: `results/month2/sparse_s3_glu_diagnostic_overfit/` (`report.json`, maps; `.pt` gitignored with SHA in report). Weights **not** for science.

---

## 5. S4 — Scientific runner + profiling outcomes

| Voce | Stato |
|---|---|
| Runner | `sparse_scientific_loop.py` (warmup 950, plateau VAL 5%×3 repl, early-stop, tranche 1000, extend esplicita) |
| Resume tests | **Verificati in locale** (`tests/test_sparse_scientific_trainer.py`); non riprovati come resume Engaging di un run lungo |
| Profiling attempt-2 | U-Net **29554165** (`node3008` L40S), GLU **29554166** (`node3408` L40S); SHA `740805f`; entrambi **COMPLETED** `0:0` / `completed_diagnostic` |
| Misure chiave | U-Net peak VRAM **0.60 GB**, loop 2ep+VAL **31.6 s**; GLU **4.33 GB**, **40.7 s**; batch **8**, microbatch **null**; query GLU **16384**; CPU/GPU ok (atol/rtol 1e-4); loss/grad smoke finiti |
| Stima tranche 1000 | **Stima** (non misura): U-Net ~2–3 h, GLU ~3–4.5 h incl. eval multi-ruolo ogni 10; incertezza alta (±40%); nessun tetto scientifico in ore |
| Freeze | Candidato operativo: mem 64G, wall scheduler 6:00:00, microbatch null — vedi freeze candidate. **Non** equivale ad autorizzazione S5 |
| S5 | Quattro run **COMPLETED** early-stop su HOME (`~/projects/official_thesis_s5`, SHA `38ac7fe`); artefatti `~/official_thesis_s5_runs/` |
| GappyPOD 1/20% | **Chiuso in S6** via job **29601906** (vedi §7b); λ congelate, nessun retune |

RMSE VAL del profiling sono diagnostici e **non** usati per architettura/LR/variante.

---

## 6. S5 screening outcomes (seed 0) — primary VAL@5%

Esecuzione: jobs **29567988 / 29567990 / 29567992 / 29567994**, tip HOME, cache geometry/file hash allineati al training (`geometry` `97e2e9df…`).  
`eval_history.json` standalone omette l’ultima entry di early-stop; VAL last recuperata da `last.pt` + CSV `evals/epoch_*`.

| Job | Run | Stop | Best VAL RMSE [K] @5% | ep best | Last VAL RMSE [K] @5% | ep last | Train wall |
|---|---|---|---:|---:|---:|---:|---:|
| 29567988 | U-Net p_on | early_stop | **42.964** | 60 | 49.496 | 630 | 52:06 |
| 29567990 | U-Net p_off | early_stop | **43.525** | 140 | 47.672 | 650 | 53:55 |
| 29567992 | GLU p_on | early_stop | **45.801** | 440 | 46.839 | 890 | 2:20:07 |
| 29567994 | GLU p_off | early_stop | **47.535** | 650 | 47.871 | 980 | 2:34:48 |

Confronto primario seed 0 (training progress): **U-Net p_on best**. Gap p_on↔p_off U-Net ~0.56 K; GLU peggiore e più costoso.

Checkpoint SHA256 (Engaging `~/official_thesis_s5_runs/`):

| Run | best.pt | last.pt |
|---|---|---|
| s5_sparse_unet_pon_seed0 | `b03b896e…464b7a` | `fad71e8f…761552` |
| s5_sparse_unet_poff_seed0 | `0d9d93a1…07fb9fc` | `a29eed9d…cd7b7` |
| s5_adapted_glu_pon_seed0 | `bfd126e5…f4639c1` | `a405bffb…0fbb0f4b` |
| s5_adapted_glu_poff_seed0 | `fbae86d5…552d1f4` | `a2d9a08d…b461ba8c` |

Full digests: `results/month2/s6_sparse_eval/STOP_E_delivery/provenance/runs_provenance.json`.

---

## 7. S6 — Eval 1/5/20% + snapshot-swap + Gappy nested (**STOP E**)

### 7a. Job ledger

| Job | Role | Partition | Account / QoS | State | Exit | Elapsed | MaxRSS |
|---:|---|---|---|---|---|---:|---:|
| **29601906** | Gappy nested 1/5/20% | `mit_normal` | `mit_amf_advanced_cpu` | COMPLETED | 0:0 | 00:00:55 | ~4.3 GiB |
| **29601907** | S6 U-Net p_on last+swap | `mit_normal_gpu` | `mit_amf_advanced_gpu` | COMPLETED | 0:0 | 00:26:25 | ~2.3 GiB |
| **29601908** | S6 U-Net p_off last+swap | `mit_normal_gpu` | `mit_amf_advanced_gpu` | COMPLETED | 0:0 | 00:25:57 | ~2.3 GiB |
| **29601909** | S6 GLU p_on last+swap | `mit_normal_gpu` | `mit_amf_advanced_gpu` | COMPLETED | 0:0 | 00:27:38 | ~2.4 GiB |
| **29601910** | S6 GLU p_off last+swap | `mit_normal_gpu` | `mit_amf_advanced_gpu` | COMPLETED | 0:0 | 00:26:57 | ~2.4 GiB |

Best@1/5/20% neurali: prodotti nel primo tentativo S6 (jobs 29584098/100/101/102, falliti allo swap per `KeyError p_conditioning`) e **riusati** dopo verifica 27/27 CSV per run. Last+swap: jobs sopra. Fix swap: `p_conditioning` (+ `baseline_physical`) in `build_sparse_swapped_batch_item`.

OUT Engaging: `~/official_thesis_s6_results/<run_id>/`, Gappy `~/official_thesis_gappypod_nested_1_20_amf_v2/`.  
Git mirror: `results/month2/s6_sparse_eval/STOP_E_delivery/`, `results/month2/gappypod_nested_1_20/`.

### 7b. GappyPOD nested (29601906)

| Voce | Valore |
|---|---|
| Load equivalence (n=8) | **ok** — max_abs vi/vj/T = 0 (`load_equivalence_sample.json`) |
| Fit artifact | `artifacts/gappypod_fitted_pod_p_off_on.npz` SHA256 `a6fbdfd2…0a74ac9`; reload modes ok |
| Peak RSS | ~4.3 GiB (128G requested after prior 64G OOM) |
| Gate 5% | **ok=true**, 6/6 checks (train/seen/val × p_on/p_off), ΔRMSE ~1e-8–1e-9 ≪ atol 0.05 K |
| VAL@5% refit | p_on **70.5985 K**, p_off **94.7987 K** (match storico) |
| Nested 1%/20% | **completed** after gate |

### 7c. Protocol matching

Stessi `mask_root` / `cache_root` / tip SHA `38ac7fe`; coverages 1/5/20%; roles train_fit / seen_train_eval / val_eval; **mask replicas** 0/1/2 (non seed di training); aggregazione mean-condition RMSE.  
**Swap contract** (`scripts/17`): maschera/coordinate/p/target del destinatario fissi; vi/vj del donatore campionati agli stessi siti destinatario; pairing U5 within-condition; riferimento `T_donor` vs `T_dest` salvato.

**Distinzione seed:** tutti i neurali sono **train seed 0**. Le tre “repliche” nei CSV S6 sono **repliche di maschera eval**. Dispersione init/shuffle = S7.

### 7d. Tabella primaria — best VAL @5% (media 3 mask replicas)

Fonte: `STOP_E_delivery/tables/primary_best_val_5pct.csv`.

| Rank | Method | RMSE [K] | \|bias\| [K] | Centered RMSE [K] |
|---:|---|---:|---:|---:|
| 1 | U-Net p_on | **42.964** | 12.884 | 38.365 |
| 2 | U-Net p_off | 43.526 | 17.518 | 38.002 |
| 3 | GLU p_on | 45.801 | 8.774 | 44.609 |
| 4 | GLU p_off | 47.535 | 18.576 | 42.716 |
| — | GappyPOD p_on (29601906) | 70.599 | 15.382 | 68.194 |
| — | GappyPOD p_off (29601906) | 94.799 | 57.579 | 71.692 |

Classifica **deployable @5%**: neurali ≪ GappyPOD matched. Denso resta separato ([03](03_dense_velocity_diagnostic.md)).

### 7e. Last VAL @5% e inversioni

| Rank last | Method | RMSE [K] |
|---:|---|---:|
| 1 | GLU p_on | 46.839 |
| 2 | U-Net p_off | 47.672 |
| 3 | GLU p_off | 47.871 |
| 4 | U-Net p_on | 49.496 |

**Inversione best↔last:** U-Net p_on passa da 1° (best) a 4° (last). Il criterio primario resta **best**.

### 7f. Curve coverage best VAL (media 3 mask replicas)

| Method | 1% | 5% | 20% |
|---|---:|---:|---:|
| U-Net p_on | 75.199 | **42.964** | 44.321 |
| U-Net p_off | 73.696 | 43.526 | 54.076 |
| GLU p_on | 68.085 | 45.801 | 46.844 |
| GLU p_off | 67.203 | 47.535 | 48.516 |
| GappyPOD p_on | 71.373 | 70.599 | 70.457 |
| GappyPOD p_off | 98.436 | 94.799 | 93.808 |

**Inversione @1%:** GLU (soprattutto p_off) batte U-Net; @5%/20% U-Net p_on riprende il primato neurale. U-Net p_off **peggiora** nettamente a 20% (54.1 vs 43.5). GappyPOD varia poco con la copertura.

### 7g. Snapshot swap (best@5%, mask replica 0)

| Method | Role | Aligned | Swapped | Δ (swap−align) | T_donor ref |
|---|---|---:|---:|---:|---:|
| U-Net p_on | VAL | 42.815 | 100.862 | +58.047 | 97.362 |
| U-Net p_off | VAL | 43.625 | 101.427 | +57.802 | 97.362 |
| GLU p_on | VAL | 45.987 | 99.584 | +53.598 | 97.362 |
| GLU p_off | VAL | 48.024 | 103.636 | +55.611 | 97.362 |

Swap ≫ aligned e vicino al riferimento donatore → predizioni **specifiche dell’istantanea**, non solo del layout. Seen mostra Δ ancora maggiori (~86–97 K).

### 7h. Figure / tabelle per revisori

Directory: `results/month2/s6_sparse_eval/STOP_E_delivery/`

| Path | Contenuto |
|---|---|
| `tables/STOP_E_primary.md` | Sintesi primaria |
| `tables/primary_best_val_5pct.csv` | Best VAL@5% |
| `tables/last_val_5pct.csv` | Last VAL@5% |
| `tables/val_coverage_best_last.csv` | Curve 1/5/20% |
| `tables/gappy_val_coverage.csv` | Gappy VAL 1/5/20% |
| `tables/snapshot_swap_summary.csv` | Aligned/swapped/Δ/donor |
| `tables/neural_per_condition_all.csv` | Per-condizione (pesante) |
| `tables/neural_all_settings_rebuilt.csv` | Tutti setting neurali |
| `figures/*__val_frame1000_aligned_vs_swapped.png` | Pannelli VAL frame 1000 |
| `figures/*__val_selection_curve.png` | Curve selezione VAL |
| `figures/*__train_loss_curves.png` | Curve loss (da S5 `curves.csv.gz`) |
| `provenance/stop_e_status.json` | Ledger job + gap + rank notes |
| `provenance/runs_provenance.json` | Checkpoint SHA + swap JSON |
| `../gappypod_nested_1_20/` | report/gate/equiv/summary CSV |

Metriche per-frame includono RMSE/MAE/NRMSE, bias, centered, e regioni supportate (`high_*`, `jet_*`, `peak_*`, `non_high_*`). Tabella regionale aggregata cross-method non pubblicata separatamente (ricavabile dai CSV).

### 7i. Lacune aperte (non bloccanti STOP E)

1. **Montage cross-method a scale comuni** non generato (esistono pannelli per-run).  
2. **Solo train seed 0** — dispersione init/shuffle = S7.  
3. Summary CSV per-job S6 (`summary_all_settings.csv` in ciascun OUT) riflette soprattutto last+swap; la fonte consolidata è `STOP_E_delivery/tables/*`.  
4. Confronto denso resta fuori classifica deployable.

---

## 8. Proposta contrasto S7 (unica, seed 0) — da revisionare

**Candidato:** **U-Net p_on vs U-Net p_off** (stesso `train_mask_seed`, seeds modello/shuffle ≥1).

**Perché:** best VAL@5% U-Net sopra GLU a costo minore; gap p ≈0.56 K architettura-matched; a 20% p_off peggiora; swap conferma sensibilità all’istantanea.  
**Incerto a seed 0:** inversione best↔last su U-Net p_on; ranking @1% favorevole a GLU; nessuna dispersione seed.  
**Non è una decisione finale** — STOP E per revisori.

---

## 9. Tests

Sparse scientific + GLU/sparse/dense epoch: S4. Swap p_conditioning: `tests/test_s6_swap_pconditioning.py` (3 passed) prima del rilancio last+swap.

---

## 10. Limits / stato fasi

- **S4:** profiling + freeze neurale chiusi; Gappy 1/20% chiuso via 29601906 → S4 operativamente completo per sparse.  
- **S5:** screening seed 0 COMPLETED.  
- **S6:** STOP E — evidenze consolidate sopra.  
- **S7:** non iniziato (no training in questa consegna).  
- HONF / pool ORCD non toccati.
