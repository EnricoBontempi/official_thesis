# Sparse reconstruction — S1–S3 technical status

**Status:** Freeze neurale S5 **definitivo** + **submitted** ([s5_neural_freeze_record.md](s5_neural_freeze_record.md)). GappyPOD 1/20% ancora **bloccato** (S4 non tutto chiuso). Run S5 **non** dichiarati completati.  
**Branch:** `mission/dense-unet`.  
**Record submission:** [s4_profiling_submission_record.md](s4_profiling_submission_record.md).

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
| S5 | Quattro config **frozen+authorized**; jobs **submitted** (vedi record); non dichiarati completati |
| GappyPOD 1/20% | **Bloccata**: manca archivio POD fit (`pod_p_off`/`pod_p_on`); λ 5% riusabili; **non** blocca training neurale S5 |

RMSE VAL del profiling sono diagnostici e **non** usati per architettura/LR/variante.

---

## 6. Tests

Sparse scientific + GLU/sparse/dense epoch: eseguiti in consegna S4 (vedi diario).  

---

## 7. Limits / aperti

- Freeze neurale: **definitivo**; S5 **submitted**; esiti a cura di Enrico (nessun polling in consegna).
- Timing TRAIN vs VAL non separati nel report profiling; stime tranche derivate.
- Resume continuo Engaging su payload reale: non rieseguito nel profiling (solo test sintetici).
- GappyPOD nested 1/20%: artifact POD non persistiti da `month2_baselines_v1`.
- Nessuno screening S5, nessun tuning VAL, niente TEST.
- Microbatch/chunked backward: non usati (non necessari ai picchi VRAM misurati).
