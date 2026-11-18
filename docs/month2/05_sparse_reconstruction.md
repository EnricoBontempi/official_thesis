# Sparse reconstruction — S1–S2 implementation & verification

**Status:** implementazione/verifiche S1–S2, **nessun risultato scientifico sparse**.  
**Branch:** `mission/dense-unet`.  
**STOP A:** non proseguire a S3 / profiling / training scientifici da questo report.

---

## 1. S1 — Maschere eval persistite

| Voce | Valore |
|---|---|
| Format | `month2_sparse_eval_masks_v1` / algoritmo `nested_subset_extension_v1` |
| Contratto | train_fit 760 + seen_train_eval 220 + val_eval 123; **no TEST**; K=0; 2C; coverages 1/5/20%; replicas 0/1/2 |
| Seed eval | `mask_seed=20261116`, `layout_id=random_sparse_primary` |
| Join 5% | **3309/3309** matched (seed regeneration ↔ persisted sites → same `mask_hash`) |
| Verified examples @5% | 3/3 match `month2_observations.json` |
| Nesting | 164 ⊂ 819 ⊂ 3277 verified per instance |
| Manifest (Git) | `data/manifests/month2_sparse_eval_masks_manifest.json` |
| Payload (out of Git) | `data/processed/month2_sparse_eval_masks/` |
| Aggregate SHA256 | `7be78f35ffea2ebc598adce8cec39dc719339aac17ecadde3137f2064891ffeb` |

**Hash semantics:** `mask_hash` = SHA256 of uint8 HxW boolean mask bytes (`month2_observations.mask_hash`). Do **not** compare against unrelated `observation_hash` objects. Nested 1%/20% are **new** artifacts; historical independent 1%/20% draws in `verified_examples` are intentionally not overwritten.

### Regeneration / load / verify

```bash
# Materialize (overwrite only when intentional)
.venv/bin/python scripts/11_materialize_sparse_eval_masks.py --overwrite

# Load one packet
.venv/bin/python -c "
from combustion_recon.data.sparse_eval_masks import load_nested_sites, verify_payload_against_manifest
print(verify_payload_against_manifest())
print(load_nested_sites(role='train_fit', sim_id='0NH3_1.h5', frame_index=0, replica_id=0)['hash_5pct'])
"
```

Protocol config: `configs/experiments/month2_sparse_protocol.yaml` (unknown scientific keys refused; `scientific_training_authorized: false`).

---

## 2. S1 — GLU audit (documental only)

See [04_glu_release_adaptation.md](04_glu_release_adaptation.md). Upstream tree `252f0ec5…` at parent `5de0d7be…`; **no** wrapper implementation; `external/` untouched. Approved candidate registered; conflicts R1–R13 and S3/S4 checks listed there.

---

## 3. S2 — Sparse U-Net

| Voce | Valore |
|---|---|
| Config | `configs/models/unet_sparse.yaml` |
| Channels (exact order) | `[vi_obs_norm, vj_obs_norm, Mv, x_scaled, y_scaled, nh3_norm, phi_norm]` |
| Backbone | b=32, GroupNorm/SiLU, 4 down, bilinear decoder, linear T head |
| Parameters (trainable) | **4 903 041** |
| Dense API | preserved (`DenseUNet` still asserts 6 channels) |
| Cache | reused `data/processed/month2_dense_unet_cache` after fingerprint validation |
| Train masks | `train_mask_seed=20261118`; `set_epoch` called every epoch and after resume |
| Eval masks | historical seed + persisted nested payload |

Canonical observation (`training/sparse_view.py`) derives U-Net grid, future GLU tokens, and physical m/s baseline packet from the same sites.

---

## 4. Tests & diagnostic overfit (measured)

**Tests:** `tests/test_sparse_eval_masks.py` + `tests/test_sparse_unet.py` → **14 passed** (~36 s, local CPU `.venv`).

Declared tolerances: bit identity `atol=0`; prediction `atol=1e-6`; resume weight `max_diff < 1e-4`.

| Check | Result |
|---|---|
| Hidden-velocity mutation | input bit-identical; pred unchanged |
| T / forbidden metadata | no forward effect |
| p_off zeros | enforced per batch |
| set_epoch mask variation | epoch 0 ≠ 1 on example; reproducible |
| Resume complete vs continuous | matched within tolerance; partial refused |
| Fixed-mask overfit (8 frames, 40 ep) | loss **1.0015 → 0.1287** (80 updates, ~22 s CPU) |
| Variable-mask brief (15 ep) | loss **1.0015 → 0.3828** (finite; no near-zero claim) |

**Frames (seed=7 subset):**  
`20NH3_1.2` {800,1200}, `20NH3_1.25` {360,920}, `40NH3_0.85` {200,1000}, `80NH3_1.05` {1200,1320}.

Artifacts: `results/month2/sparse_s2_diagnostic_overfit/` (`report.json`, `diag_maps_frame0.npz`; checkpoints local, not for science).

Weights from these runs **must not** be reused in future scientific trainings.

---

## 5. Limits / not done

- No scientific training on 760 frames; no VAL tuning; no GLU wrapper (S3); no S4/S5; no 1/20% baseline re-eval campaign; U4–U7 artifacts untouched.
- Engaging not required for this delivery (CPU sufficient for diagnostic overfit).
- Nested 1%/20% baseline matched evals remain future work (S4/S6).
