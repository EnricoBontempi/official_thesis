# Sparse reconstruction — S1–S3 technical status

**Status:** implementazione/verifiche S1–S3, **nessun risultato scientifico sparse**.  
**Branch:** `mission/dense-unet`.  
**STOP B:** non proseguire a S4 / screening / training scientifici da questo report.

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

Audit: [04_glu_release_adaptation.md](04_glu_release_adaptation.md). Upstream tree untouched.  
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
| Decoder | public `SoftDomainAdaptiveReconstructor.forward` only |
| Defaults | d=32, heads=4, layers=4, latents=64, top_k=16, σ=0.05, `per_sensor_sigma=False`, CLS on; encoder `num_freqs=256` (stage-0), decoder `num_freqs=64` (class default) |
| Geometry | `q=2(r-c)/L`, L=max(extent_x,extent_y); recorded in `geometry_meta` |
| Chunk eval | 512 queries; full vs chunk **max_abs=0** (atol declared 1e-5); chunked **backward not claimed** |

### Diagnostic overfit (same 8 TRAIN frames as S2, full 16384 queries, CPU)

| Run | Epochs | Updates | Loss first→last | Wall |
|---|---:|---:|---|---:|
| Fixed masks | 8 | 64 | **1.262 → 0.367** | ~17 s |
| Variable masks | 5 | 40 | **1.262 → 0.734** | ~19 s |

Artifacts: `results/month2/sparse_s3_glu_diagnostic_overfit/` (`report.json`, maps; `.pt` gitignored with SHA in report). Weights **not** for science.

---

## 5. Tests

`tests/test_sparse_eval_masks.py` + `test_sparse_unet.py` + `test_glu_adapted.py` + dense regression: **41 passed**.

---

## 6. Limits / S4 remaining

- No scientific 760 training, VAL tuning, S4 profiling/freeze, S5 screening, 1/20% baseline campaign, TEST.
- Scientific epoch runner resume (warmup/plateau/VAL counters) **not** verified — required before S4 freeze.
- Isotropic-q vs upstream [-1,1] σ-semantics sensitivity: documented, not profiled as cost study.
- Chunked training backward / grad accumulation: **not supported** until verified.
