# GLU release adaptation — documentary audit (S1)

**Project:** `EnricoBontempi/official_thesis` only.  
**Roadmap:** [roadmapsparse.md](../../roadmapsparse.md) §5 (approved deterministic GLU candidate); S1 deliverable per §8.  
**Scope of this file:** read-only audit of frozen upstream in `external/GLU_Reconstruction/`. **No** edits to `external/`, **no** wrapper implementation (`models/glu_reconstructor.py` is S3).  
**Parent repo (audit time):** branch `mission/dense-unet`, commit `5de0d7bec62a925e63c2a9b86ac27cd12d6c1f50`.

---

## 1. Provenance / upstream identity

| Item | Value | Status |
|---|---|---|
| Separate git submodule | **No** — `external/GLU_Reconstruction/.git` absent | **Verified** (`test -d …/.git`) |
| Parent `HEAD` at audit | `5de0d7bec62a925e63c2a9b86ac27cd12d6c1f50` | **Verified** (`git rev-parse HEAD`) |
| Tree entry at `HEAD` | `external/GLU_Reconstruction` → tree `252f0ec5335dc0798afff57bdb0c69269188f9cb` | **Verified** (`git ls-tree HEAD`) |
| `src/models.py` blob | `f7150976be0c6f7d6eaa0b6d94e8e290414ddd1a` | **Verified** (`git hash-object`) |
| Upstream remote | N/A (vendored tree inside thesis repo) | **Verified** |

**Interpretation:** the “frozen release” is the tree at parent commit `5de0d7be…` / tree `252f0ec5…`, not an independent submodule SHA. Any future pin should record **both** parent commit and tree hash.

**Files read integrally for this audit:** `src/models.py`, `train.py`, `evaluate.py`, `src/dataloading.py`, `README.md`, `configs/collinear_re40_stage0.yaml`, `configs/collinear_re40_stage1.yaml`, `requirements.txt`, `src/__init__.py`.

---

## 2. Module disposition (reuse / modify / exclude)

| Module / path | Role in upstream release | Thesis disposition | Rationale (code-backed) |
|---|---|---|---|
| `DomainAdaptiveEncoder` | Sparse sensor → latent tokens | **Reuse structure, modify I/O** | Core spatial encoder; `forward` ignores `U`; accepts `(B,T,N,2+C)` and optional `original_phi` (`models.py` L319–391). Wrapper must drop time channel, add `p`, enforce site ordering. |
| `build_pos_value_proj`, `build_latent_bank`, `CrossAttention`, `FourierEmbedding` | Encoder building blocks | **Reuse** (via encoder) | Composed inside `DomainAdaptiveEncoder`. |
| `SoftDomainAdaptiveReconstructor` | Latent → field on query coords `Y` | **Reuse structure, modify I/O + fusion** | k-NN + aggregation / optional CLS fusion (`models.py` L1644–1921). Thesis: single channel **T**, uniform `phi`, no `CalRecVar`, inference without targets. |
| `SpatiallyAwareCLSFusion`, `CLSConditionedCoordEncoder` | CLS pathways | **Reuse if `retain_cls=True`** | Used when `retain_cls` (`models.py` L1893–1900). Matches approved candidate. |
| `GLU_Bay_DD` | Full train/infer orchestration | **Exclude from scientific path** | `forward` requires `G_full`, `Y`, `U`, `teacher_force_prob`; wires encoder + temporal + decoder + optional phi MLPs (`models.py` L2110–2279). |
| `TemporalDecoderSoftmax`, `TemporalDecoderHierarchical` | Latent forecasting | **Exclude** | Not needed for single-frame sparse **T** task per roadmap §5. |
| `TemporalDecoderAdapter` | AR rollout wrapper | **Exclude** | |
| `ud_MultiheadSoftmaxAttention`, `DelayEmbedNeuralODE` (stub) | Alternate / legacy paths | **Exclude** | `odeint` stub raises unless `torchdiffeq` installed (`models.py` L15–19). |
| `train.py`, `evaluate.py` | Upstream training/eval | **Exclude** | Full-field `g_f` loss, stage0/1 staging, phi/NLL (`train.py` L368–409). |
| `src/dataloading.py` | HDF5 → `G_down` with **time** in last channel | **Exclude** | `build_cube` appends `t` (`dataloading.py` L407–412). Conflicts with thesis token contract (no **T** in forward features). |
| `configs/*.yaml` | Collinear Re40 demos | **Reference only** | Defaults differ from approved sparse candidate (see §4, §8). |
| `README.md` | Release workflow | **Reference only** | Documents stage-0 uncertainty + adaptive importance — **not** thesis path. |

---

## 3. Tensor shapes and channels (verified from code)

### 3.1 Upstream dataloader → model interface

From `dataloading.py` docstring and `H5SpatialTemporalDatasetConcentrated.__getitem__`:

| Tensor | Shape | Content |
|---|---|---|
| `G_down` | `[B, N_ts, N_xs, 2 + N_c + 1]` | `(x, y, field values…, t)` per sensor |
| `G_down_t` | `[B, N_ts, N_pts, N_c]` | Downsampled field on recon pool (unused in `GLU_Bay_DD.forward` signature) |
| `G_full` | `[B, T_full, N_pts, N_c]` | **Training target** on reconstruction points |
| `Y` (`y` in batch) | `[N_pts, 2]` or batched `[B, N_pts, 2]` | Query coordinates (recon grid subset or full) |
| `U` | `[B, N_para]` | Conditions; **encoder does not use `U` in `forward`** |

Coordinate preprocessing in loader: per-dataset min–max on `xy`, then map to **[-1, 1]** (`dataloading.py` L128–133). **Not** the thesis isotropic `q = 2(r - c)/L` map (roadmap §5.2) — that is a **planned wrapper change**, not present upstream.

### 3.2 `DomainAdaptiveEncoder`

| Arg / return | Shape | Notes |
|---|---|---|
| `coords_tuv` | `(B, T, N_inp, 2+C_sel)` | `xy = [...,0:2]`, values from `[...,2:]` or `channel_to_encode` (`models.py` L329–338) |
| `U` | any | **Unused** in `forward` (`models.py` L320–323) — aligns with roadmap note; still passed by `GLU_Bay_DD` |
| `lat_vec` | `(B, T, S, D)` | `S = N_inp`, or `1+N_inp` if `retain_cls`, or `L+N_inp` if `retain_lat` (`models.py` L371–383) |
| `mask` | `(B, T, S)` | All-ones bool mask (`models.py` L385–386) |
| `coords` | `(B, N_inp, 2)` | From **first time slice only**: `coords_tuv[:, 0, :, :2]` (`models.py` L386) |
| `merged_phi` | `(B, S)` | Ones if `original_phi is None` (`models.py` L387–389) |

Embedding: `tok = pos_tok + val_tok` with `FourierEmbedding(2, num_freqs)` → `pos_linear` → `RMSNorm`, and `val_linear(N_channels, All_dim)` → `RMSNorm` (`models.py` L341–347, `build_pos_value_proj` L229–237). Default `nn.Linear` **includes bias** on `val_linear` / `pos_linear`.

### 3.3 `SoftDomainAdaptiveReconstructor`

| Arg / return | Shape | Notes |
|---|---|---|
| `z` | `[B, T, S, D_raw]` or `[B, T, S+1, D]` with CLS | |
| `Y` | `[B, P, 2]` | Query positions |
| `sensor_coords` | `[B, S, 2]` | Sensor locations (excludes CLS index) |
| `mask` | `[B, T, S]` or `[B, T, S+1]` | |
| `phi_mean` | `[B, S]` or `None` → ones | Detached for distance scaling (`models.py` L1778, L1823–1824) |
| `local_out_mean`, `out_mean` | `[B, T, P, C]` | `C = N_channels`; duplicate refs when `retain_cls=False` (`models.py` L1904–1910) |
| `out_logvar` | `[B, T, P, C]` or `None` | If `CalRecVar` |

Internals (release `forward`, not `_forward`): `torch.cdist(Y, sensor_coords)` → `[B,P,S]`; `top_k` on `d_scaled = d / (phi^gamma)`; weighted aggregation `_topk_aggregate`; then **V-projection hack** from `MultiheadAttention` weights (`models.py` L1876–1887), **not** the full cross-attention path in `_forward` (L2037+). **Verified:** two implementations exist; shipped `forward` uses top-k + V-linear path.

### 3.4 `GLU_Bay_DD.forward` (excluded path, shapes for incompatibility analysis)

| Input | Shape | Used for |
|---|---|---|
| `G_down` | `[B, N_ts, N_xs, F]` | Encoder input (`F = 2+C+1` from loader) |
| `G_full` | `[B, T_full, N_pts, 1]` | Sets `T_full`, MSE target in trainer |
| `Y` | `[N_pts, N_dim]` / `[B, N_pts, 2]` | Decoder queries |
| `teacher_force_prob` | scalar | Temporal training only |

Outputs used in `train.py`: `G_u_mean` vs `g_f`, optional sensor head vs `g_d[...,2:2+C]`, trajectory MSE, NLL, phi ELBO.

### 3.5 Thesis sparse contract (roadmap §3–§5) — target mapping **not yet implemented**

| Quantity | Expected (roadmap) | Upstream analogue |
|---|---|---|
| Sensors / step | `N_s` sites, 2C scalars → **2** value channels in encoder | `N_xs` sensors, `N_c` channels + **time** |
| Queries | `P = 16384` (128×128), single `t` | `P = Num_recon_pts` (often downsampled, e.g. 40% in stage0 YAML) |
| Model output | **T** only (1 channel) | `N_channels` = dataloader `N_c` (configs use `channel: 0` → **1** channel) |
| Batch | Homogeneous `N_s`, **no padding** | Variable `N_xs` via resampling per item; **no padding mask path in encoder** (decoder has `padding_mask` optional) |

---

## 4. Effective defaults: class `__init__` vs `build_model` vs shipped YAML

Legend: **C** = confirmed in `models.py` constructor defaults; **F** = set only in `train.py` / `evaluate.py` `build_model`; **Y0/Y1** = `collinear_re40_stage0.yaml` / `stage1.yaml`.

### 4.1 `DomainAdaptiveEncoder` (`build_model` in `train.py` L154–165)

| Parameter | Class default (`models.py`) | If omitted from YAML | Shipped Y0 | Shipped Y1 | Approved sparse candidate (roadmap §5) |
|---|---|---|---|---|---|
| `All_dim` (`F_dim`) | — | — | 32 | **64** | **32** |
| `num_heads` | — | — | 4 | 4 | **4** |
| `latent_layers` (`num_layers`) | — | — | 4 | 4 | **4** |
| `num_freqs` | **64** (L268) | — | **256** | 128 | **TBD** (roadmap cites 64 latents, not freqs) |
| `latent_tokens` | **8** (L269) | — | 64 | 64 | **64** |
| `pooling` | `"none"` (asserted, **ignored** L270) | `"none"` in Y0/Y1 | `none` | `mean` (unused) | n/a |
| `retain_cls` | **False** | `cfg.get(..., False)` | **true** | **true** | **true** |
| `retain_lat` | **False** | default False | false | false | false (implicit) |
| `channel_to_encode` | `None` (= all) | — | `[0]` | `[0]` | vi/vj (2 ch.) — **wrapper** |

Cross-attn / mixer depth = `latent_layers` for both `cross_attn` and `token_to_latent` (`models.py` L299–311). Dropout **0.0** on cross-attention (L299).

### 4.2 `SoftDomainAdaptiveReconstructor` (`build_model` L183–196)

| Parameter | Class default | Shipped Y0 | Shipped Y1 | Approved candidate |
|---|---|---|---|---|
| `d_model` | — | 32 | 64 | 32 |
| `num_heads` | — | 4 | 4 | 4 |
| `N_channels` | — | 1 (from data) | 1 | **1 (T)** |
| `num_freqs` | **64** (L1653) | (not passed → 64) | idem | likely **64** unless wrapper sets otherwise |
| `overlap_ratio` | **0.02** | (default) | (default) | unused in thesis v0 |
| `importance_scale` | **0.50** | **0.25** | **0.50** | **uniform φ=1** → γ term inactive if φ≡1 |
| `bandwidth_init` | **0.05** | 0.05 | 0.05 | **0.05** (isotropic **q**-space, roadmap) |
| `top_k` | **None** (= all S) | **16** | **16** | **16** |
| `per_sensor_sigma` | **False** (L1661) | **true** | false | **False** |
| `CalRecVar` | **False** | **true** | **true** | **False** |
| `retain_cls` | **False** | true | true | **true** |
| `USE_FINAL_MLP` | **False** | false | false | false (keep release fusion; roadmap “fusione release documentata”) |
| `use_weighted_fusion` | **False** | (default) | (default) | **False** |
| `block_size` | **256** | (default) | (default) | not used in `forward` path |

Fourier `sigma` for positions: **10.0** fixed in reconstructor `__init__` (L1674–1678). Encoder Fourier uses same **10.0** via `build_pos_value_proj` → `FourierEmbedding(..., learnable=True)` (no explicit sigma arg → default 10.0).

### 4.3 Temporal decoders (excluded for thesis v0)

`TemporalDecoderSoftmax` defaults (`models.py` L501–514): `n_layers=4`, `n_heads=4`, `n_window=16`, `dt=0.02` (YAML sets `delta_t: 1.0`), `rope_base=1000`, `checkpoint_every_layer=True`.

`TemporalDecoderHierarchical`: `pooling_kernel=2`, `pooling_layers=2`, `n_window=16`, Euler rollout in `_rollout_loop` (Heun blocks commented L1464–1491).

Factory always builds temporal module even for stage 0; weights frozen in `freeze_for_stage` when `Stage==0` (`train.py` L247–250).

### 4.4 `GLU_Bay_DD` flags

| Flag | Default in class | Y0 | Y1 | Thesis |
|---|---|---|---|---|
| `use_adaptive_selection` | **False** (via `cfg.get`) | **true** | **true** | **False** |
| `Use_imp_in_dyn` | **False** | false | **true** | **False** |
| `CalRecVar` | ctor arg | true | true | **False** |
| `Supervise_Sensors` | `cfg.get` **False** (`train.py` L81) | **true** | false | **False** (semantic mismatch for T head) |
| `N_window` | ctor | 1 | 1 | 1 (single-frame) |

`train.py` `load_cfg` injects many Bayesian/teacher-force defaults (L69–105) even when unused — **verified**; thesis trainer must not inherit these silently.

---

## 5. Target / loss paths vs sparse **T-from-velocity** task

### 5.1 Upstream loss composition (`train.py` `run_epoch` L388–395)

| Term | When active | Target | Incompatibility with thesis |
|---|---|---|---|
| `weighted_channel_mse(out, g_f)` | Always | Full-field `G_full` on recon grid | Thesis: same **idea** (MSE on **T**), but `out` must be 1 ch. normalized **T**; no `G_full` in **inference** signature |
| `sensor_supervision_loss` | `Stage==0` ∧ `Supervise_Sensors` | `g_u_mean_sens` vs `g_d[..., 2:2+C]` | Sensors carry **vi/vj**; head emits **N_channels** from training setup — with `channel:0` upstream supervises **same scalar field at sensors**, not velocity→**T** mapping |
| `trajectory_loss` | `Stage≥1` or `N_window < T` | Latent rollout vs encoded obs | Irrelevant for single-frame deterministic ablation |
| `heteroscedastic_nll` | `CalRecVar` ∧ `Stage==0` | `out_logvar` | **Excluded** by roadmap |
| `compute_phi_loss` (ELBO) | `Use_Adaptive_Selection` ∧ train | Importance vs recon uncertainty | **Excluded** |

### 5.2 Forward-path leakage / conditioning (inference contract)

- `GLU_Bay_DD.forward` **always** takes `G_full`; used to size `T_full` and (in trainer) as regression target — **not** roadmap-compliant inference API.
- `teacher_force_prob` and latent rollout inject temporal machinery unrelated to instant **T**.
- `phi_mlp_*` + `use_adaptive_selection`: builds `original_phi` from sensor coords (`models.py` L2187–2212) — **excluded**; thesis uses φ≡1.
- Encoder **time channel** in `G_down` last column: not stripped in `DomainAdaptiveEncoder` unless wrapper omits it from tensor width (encoder slices values by `val_linear.in_features` / `channel_to_encode` only).

### 5.3 Semantic alignment with Month-2 sparse contract

| Roadmap requirement | Upstream | Gap |
|---|---|---|
| Tokens `[x,y,vi_norm,vj_norm]` no **T** in forward | `(x,y,u,t)` in `G_down` | Strip **t**; 2 value channels; isotropic **q** coords |
| Output **T** only | `N_channels` from dataset channel index | Set `N_channels=1`, supervise normalized **T** from cache |
| `p` additive, no bias: `h_pos + h_val + W_p p` | No **p**; `tok = pos + val` only | **New** `nn.Linear` (bias=False) in wrapper |
| `p` not at decoder | N/A | Enforce in wrapper |
| No target / `G_full` at inference | Required in `forward` | New wrapper `forward` |
| Uniform importance | `phi_mean` default ones OK; adaptive path optional | Disable `use_adaptive_selection` |
| Homogeneous batch, no padding | Decoder supports `padding_mask`; encoder does not | Refuse heterogeneous **N**; do not claim padding support |

---

## 6. Approved candidate registry (roadmap §5 — design contract for S3+)

The following is **approved operationally** in `roadmapsparse.md` §5.1–§5.3; this audit **registers** it against upstream capabilities. Implementation status: **S3 wrapper delivered** (`models/glu_reconstructor.py`, `configs/models/glu_minimal.yaml`); not a full paper reproduction.

| Decision | Approved candidate | Upstream baseline | Action |
|---|---|---|---|
| Encoder / decoder modules | Reuse `DomainAdaptiveEncoder` + `SoftDomainAdaptiveReconstructor` classes | As-is in `models.py` | Thin wrapper in thesis package |
| Wrapper inference signature | **No** `G_full` / targets | `GLU_Bay_DD` requires them | Replace orchestration |
| Conditioning **p** | Additive `W_p p`, **bias=False**; not repeated at decoder | Not present | Wrapper-only |
| Coordinates / distances | Isotropic `q=2(r-c)/L` from **fixed full-domain** axes (cache/`geometry.npz` or explicit synth); buffers persist center/L | Loader [-1,1] anisotropic min–max; `cdist` on `Y` vs `sensor_coords` | Wrapper maps with module buffers; **no** call-site sensors∪queries geometry; error if missing |
| Importance | Uniform (φ=1); no propagator, Beta, NLL, learned imp | Stage0 YAML enables adaptive φ + CalRecVar | Strip |
| `per_sensor_sigma` | **False** | Stage0 YAML **true**; dynamic `nn.Parameter` creation in `forward` if `True` (`models.py` L1780–1784) | Force **False**; S3 torch test on param registration |
| CLS + top-k | `retain_cls=True`, `top_k=16` | Matches Y0/Y1 | Align factory |
| Capacity | d=32, 4 heads, 4 layers, 64 latents | Y0 matches; Y1 uses d=64 | Use **32** for matched screening |
| Value embedding | Keep release `val_linear` + bias + `RMSNorm` initially | `build_pos_value_proj` | Per roadmap §5.2; MLP alt only on evidence |
| Batches | Homogeneous **N**, **no** padding; refuse mixed | Padding hook in decoder only | Explicit errors in wrapper |
| Site order | Sort by site index; **lexicographic** k-NN tie-break `(distance↑, site_index↑)` | `torch.topk` on `d_scaled` — **no** guaranteed lower-index preference on ties | Thesis subclass replaces topk; sorting alone insufficient |
| Temporal stack | None for v0 | Always constructed in `build_model` | Do not instantiate / load |
| Query eval | Chunk decoder (**512** candidate) | Full `P` in one pass | S3 full vs chunk numerical test |
| Training loss | MSE normalized **T** full grid (16384) | MSE on `g_f` | Thesis trainer (S5), not upstream |

---

## 7. Expected costs and verifications (S3 / S4)

### 7.1 Complexity drivers (**verified** structure; magnitudes **need profiling**)

| Kernel | Scaling (symbols) | Code locus | Note |
|---|---|---|---|
| Encoder self/cross on sensors | O(`N_s²`) per layer via token mixing | `token_to_latent` + `TransformerEncoder` on `N_inp` tokens | Roadmap §5.3 warns O(N²) |
| Encoder latents | O(`L·N_s`) with small `L=64` | `cross_attn` latents ← tokens | |
| `cdist` queries × sensors | O(`P·N_s`) | `SoftDomainAdaptiveReconstructor` L1813 | Full grid `P=16384` |
| Top-k + aggregate | O(`P·N_s·log K`) topk; gather O(`P·K`) | L1838–1865 | Top-k does **not** remove dense `cdist` |
| Decoder repeat | Per time `T`; thesis `T=1` | — | |
| Chunking | `P/512` forwards | Not in upstream | Wrapper S3 |

At **5%** coverage (`N_s≈819` per roadmap §3.2), `cdist` and encoder attention likely dominate; **20%** is not guaranteed ~4× encoder cost if `P` fixed at 16384 — **requires S4 profiling**, not guessed here.

### 7.2 S3 verifications (roadmap §8 S3 + §5)

| ID | Verification | Method | Status |
|---|---|---|---|
| S3-V1 | Tensor shapes / finite forward-backward | `tests/test_glu_adapted.py` | **Done** |
| S3-V2 | **T** absent from inference signature | API + **non-vacuous** isolation: mutate source **T**, rebuild via `build_canonical_observation`, assert inputs/pred unchanged and target really changed | **Done** (vacuous discarded-`t_mut` test corrected) |
| S3-V3 | `per_sensor_sigma=False` → no dynamic params; upstream `True` lazy Parameter | Separate torch test (executed) | **Done** (risk confirmed when True; thesis forces False) |
| S3-V4 | Permutation invariance (sensor order) + query equivariance | Numeric tests, atol=1e-5, rtol=0 | **Done** |
| S3-V5 | Tie-break at equal k-NN distances | `topk_distance_then_site_index` (stable double-argsort); **N>k**, surplus equidistant candidates, expected site indices, sensor permutations, unequal-distance control; CPU env recorded; **no** universal CPU/GPU claim | **Done** (corrected: earlier claim that `torch.topk` prefers lower indices was **false**) |
| S3-V6 | Full vs chunked decoder (`chunk=512`) | Declared atol=1e-5; measured **max_abs=0** | **Done** (eval); chunked **backward not claimed** |
| S3-V7 | Collinear (vi,vj) embedding amplitude sensitivity | Torch test (no collapse assumed) | **Done** |
| S3-V8 | Isotropic **q** vs old [-1,1] map sensitivity | Geometry recorded; σ-semantics cost study | **Open → S4** |
| S3-V9 | Parameter identity before/after forward+step | Count/names stable; unused MLP may lack grads | **Done** |
| S3-V10 | Fixed domain geometry invariants A–E | center/L from cache axes as buffers; query-subset / batch / layout / chunk / reload; declared tols; measured max_abs=0 (CPU) | **Done** |

### 7.3 S4 verifications (roadmap §8 S4 + §5.3)

| ID | Verification | Method | Status |
|---|---|---|---|
| S4-V1 | VRAM/RAM at batch 8, full `P=16384`, fp32 | Engaging / local profiler | **Open** |
| S4-V2 | Microbatch / grad accumulation vs single loss | If S3-V6 fails memory | **Open** |
| S4-V3 | GappyPOD 5% + nested 1/20% eval reuse | Artifacts from S1 manifests | **Open** (S1) |
| S4-V4 | Freeze file: factory defaults = `glu_minimal.yaml` + wrapper | Config diff audit | **Open** |
| S4-V5 | Cost report without using VAL perf for model pick | Procedure only | **Open** |
| S4-V6 | GLU blocked → explicit continue/stop U-Net decision | Governance | **Open** |

**Not claimed:** upstream release was executed on thesis combustion data; GPU hours; numerical equivalence between `forward` and `_forward` in reconstructor; paper vs code parity.

---

## 8. Roadmap conflicts and explicit deltas (no silent choices)

| # | Topic | Roadmap / thesis | Upstream release | Resolution |
|---|---|---|---|---|
| R1 | **Coordinate map** | Isotropic `q` from **full-domain** **L, c** (cache axes / explicit) | Per-file min–max → [-1,1] | Wrapper buffers; refuse silent first-batch geometry |
| R2 | **`per_sensor_sigma`** | **False** | Y0 `true`; lazy Parameter in forward | Factory + S3-V3 |
| R3 | **Uncertainty / φ** | Off | Y0/Y1 `Use_Adaptive_Selection`, `CalRecVar` | Do not instantiate phi MLPs |
| R4 | **Time in encoder input** | Forbidden in forward features | Last channel of `G_down` | Wrapper strips **t** |
| R5 | **Channels** | vi/vj in, **T** out | Single `channel:0` scalar field | 2 in / 1 out + norm stats |
| R6 | **`F_dim`** | 32 for screening | Y1 example 64 | Use 32 in `glu_minimal.yaml` |
| R7 | **`num_freqs`** | Roadmap silent | Encoder default 64; Y0 passes 256 only if wired — **Y0 sets 256 in YAML but `build_model` does not pass `num_freqs` to reconstructor** | Reconstructor stays **64** freqs unless ctor updated; encoder gets YAML value — **Verified asymmetry** |
| R8 | **Supervise_Sensors** | Not semantically reusable | Y0 `true` | Disable; different physics |
| R9 | **Padding / mixed N** | Refuse | Decoder `padding_mask` only | No implicit support |
| R10 | **Propagator** | None | Built always in `build_model` | Wrapper omits temporal modules |
| R11 | **Loss on full grid** | 16384 queries TRAIN | `Full_Field_DownS` < 1 subsamples recon points | Thesis uses full grid per roadmap §5.3 |
| R12 | **Reconstructor fusion path** | “Release fusion documented” | `USE_FINAL_MLP=false`; `SpatiallyAwareCLSFusion` when `retain_cls` | Keep; GEGLU optional off |
| R13 | **Dual decoder implementations** | Unclear which is “paper” | `forward` ≠ `_forward` (cross-attn variant) | S3 must **freeze** which path wrapper calls (default: public `forward`) |

---

## 9. Verified facts vs suspected issues

### Verified from static code (this audit)

1. Encoder `forward` does **not** use `U`.
2. Token fusion in encoder is **additive** `pos_tok + val_tok` (no **p** term upstream).
3. `GLU_Bay_DD` ties inference to `G_full` and optional temporal rollout + phi heads.
4. `per_sensor_sigma=True` can register `log_sigma` **during forward** when `S` changes (`models.py` L1780–1784).
5. Shipped stage0 config enables adaptive selection, variance head, sensor supervision, and `per_sensor_sigma: true` — all contrary to approved sparse candidate.
6. `build_model` does **not** pass `num_freqs` into `SoftDomainAdaptiveReconstructor` (reconstructor stays at class default **64**).
7. Upstream HDF5 loader appends **physical time** to every sensor token.

### Suspected / reviewer-flagged — **not** demonstrated defects here

1. **Optimizer identity** when `log_sigma` is created mid-training (roadmap §5.2) — needs **S3-V3**.
2. **Tie-breaking** for equidistant sensors — **addressed in S3-V5** via thesis helper (not `torch.topk`).
3. **Memory** at `P=16384`, `N_s=819` — needs **S4-V1**.
4. Whether public `forward` or `_forward` matches published GLU paper — needs paper diff, not in repo.

### S3 post-delivery corrections (same STOP B, no S4)

| Issue | Prior over-strong claim | Correction |
|---|---|---|
| Tie-break | Comment/tests implied `torch.topk` prefers lower indices | Implemented `topk_distance_then_site_index`; N>k expected-index tests |
| Domain geometry | `compute_isotropic_geometry(sensors∪queries)` at call site | Fixed domain from `geometry.npz` / explicit axes; buffers; removed call-site helper |
| Target isolation | Mutated unused `t_mut` then re-inferred same tensors | Mutate source **T**, rebuild canonical GLU inputs, assert identity |

**Prior full-grid diagnostic limit:** when queries already span full mesh extrema, sensors∪full_queries center/L ≡ domain axes (verified L/c diff=0). Internal chunking that applied the q-map **before** splitting queries remains consistent with that map; do **not** retroactively invalidate prior full/chunk max_abs=0 under full-grid inputs. Subset-only call-site geometry **can** differ (corner example L_diff≈0.060).

---

## 10. Relation to mission phases

| Phase | This document |
|---|---|
| **S1** | Delivers audit portion; manifest/YAML/schema are separate S1 items |
| **S2** | U-Net sparse — out of scope here |
| **S3** | Implement wrapper + §6 registry + §7.2 tests |
| **S4** | Profiling + freeze + GappyPOD matched baselines |

**STOP:** S1 documentary audit only. No scientific training authorized by this file.

---

## 11. Quick reference — file anchors

| Concern | Location |
|---|---|
| Encoder I/O | `external/GLU_Reconstruction/src/models.py` (`DomainAdaptiveEncoder`) |
| Decoder I/O / k-NN | `SoftDomainAdaptiveReconstructor.forward` |
| Factory | `external/GLU_Reconstruction/train.py` `build_model` |
| Losses | `external/GLU_Reconstruction/train.py` `run_epoch` |
| Batch layout | `external/GLU_Reconstruction/src/dataloading.py` |
| Shipped hyperparameters | `configs/collinear_re40_stage0.yaml`, `stage1.yaml` |
| Approved sparse GLU policy | `roadmapsparse.md` §5 |

---

*Audit author: Cursor agent (S1). Parent commit `5de0d7bec62a925e63c2a9b86ac27cd12d6c1f50`; tree `252f0ec5335dc0798afff57bdb0c69269188f9cb`.*