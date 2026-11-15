# Step 1.5 — Inexpensive Feasibility Checks (Audit Fix Pass)

**Generated from CSVs:** `2026-11-15` (auto-written from tables; zero hand-copied numbers)

**Meta (`month1_feasibility_meta.json`):**
- ParamMean: basis=`piecewise_phi1`, α=`0.01`, LOCO RMSE=`71.96` K
- Gappy POD: R=`32`, λ=`0.01`, w_T=`5.0`
- Front sanity Pearson(|∇·u|, |∇T|) on `0NH3_1` = `0.2368`
- Frames: train_fit=`760`, seen=`220`

## 1. Check 1 — pooled kinematic correlations with T

| Feature | Pearson | Spearman | MI |
|---------|--------:|---------:|---:|
| `vi` | 0.0065 | 0.0141 | 0.0706 |
| `vj` | -0.3196 | -0.2057 | 0.1282 |
| `vk` | 0.0103 | 0.0173 | 0.0778 |
| `umag` | -0.3788 | -0.2798 | 0.1169 |
| `div` | -0.0204 | 0.0092 | 0.0125 |
| `div_alt` | -0.0115 | -0.0151 | 0.0182 |
| `vort` | -0.0643 | -0.0639 | 0.0175 |
| `strain` | -0.1231 | -0.1302 | 0.0224 |

## 2. Temperature-free baselines

Cell format: `RMSE / High-T RMSE / Jet RMSE / Bias / Centered RMSE / High-T IoU`.

| Method | seen | val | interp | extrap |
|--------|------|-----|--------|--------|
| GlobalTrainMean | 173.8/101.1/405.6/+0.5/151.6/0.581 | 140.1/100.4/326.1/-61.1/115.3/0.807 | 134.2/70.1/338.7/-19.8/123.3/0.836 | 223.4/194.4/507.2/+114.2/175.0/0.115 |
| ParamConditionedMean | 129.3/74.8/335.2/+0.3/127.2/0.659 | 88.0/44.9/261.7/-16.9/84.8/0.904 | 106.6/46.2/283.5/-13.4/104.7/0.887 | 172.1/223.2/414.0/+42.6/159.4/0.195 |
| LocalHGB_Unconditioned | 164.4/86.9/380.6/+3.2/141.6/0.584 | 138.8/96.9/300.9/-60.2/111.6/0.836 | 118.3/59.3/304.2/-16.6/108.7/0.855 | 219.0/131.2/505.3/+124.1/166.4/0.115 |
| LocalHGB_Conditioned | 116.5/65.8/312.9/+1.2/115.3/0.626 | 93.2/42.8/276.6/+9.0/90.7/0.941 | 95.8/33.4/269.9/+3.9/94.4/0.905 | 159.0/136.0/382.9/+69.4/133.2/0.122 |
| GappyPOD_Unconditioned | 123.0/72.2/275.9/-0.5/109.8/0.614 | 94.6/71.0/197.2/-46.4/71.8/0.880 | 98.5/58.8/220.1/-32.6/87.0/0.843 | 182.7/157.4/381.5/+93.1/139.2/0.122 |
| GappyPOD_Conditioned | 107.7/65.2/268.6/+0.3/105.9/0.663 | 69.0/34.9/204.7/-14.6/66.7/0.937 | 88.4/38.3/227.5/-13.5/85.5/0.899 | 153.4/202.9/360.0/+41.2/138.5/0.173 |
| OraclePOD_Unconditioned | 53.9/39.5/120.3/+0.1/53.9/0.700 | 35.0/19.5/100.7/-0.6/35.0/0.968 | 46.6/25.0/110.5/-0.9/46.6/0.934 | 72.1/84.1/160.7/+1.0/72.0/0.309 |
| OraclePOD_Conditioned | 53.6/39.0/118.7/+0.1/53.5/0.693 | 35.8/20.2/102.6/-2.3/35.7/0.966 | 46.5/25.3/109.0/-1.9/46.4/0.930 | 75.2/85.1/163.5/+5.7/74.6/0.296 |

## 3. Nested K-anchor AnchorOffset (GappyPOD Conditioned @ vel 5%)

| K | val RMSE / |bias| / High-T | interp RMSE / |bias| / High-T | extrap RMSE / |bias| / High-T / IoU |
|--:|---------------------------:|------------------------------:|------------------------------------:|
| 0 | 69 / 15 / 35 | 88 / 21 / 38 | 153 / 43 / 203 / 0.17 |
| 1 | 72 / 17 / 38 | 92 / 24 / 43 | 159 / 53 / 213 / 0.19 |
| 4 | 72 / 19 / 39 | 93 / 25 / 45 | 150 / 44 / 211 / 0.19 |
| 9 | 70 / 16 / 36 | 89 / 18 / 40 | 145 / 33 / 209 / 0.20 |
| 16 | 69 / 14 / 35 | 88 / 15 / 38 | 142 / 26 / 206 / 0.20 |
| 32 | 68 / 9 / 33 | 87 / 11 / 36 | 140 / 19 / 206 / 0.20 |

## 4. ParamMean + AnchorOffset on `test_extrap`

| K | RMSE | Bias | High-T RMSE | High-IoU |
|--:|-----:|-----:|------------:|---------:|
| 0 | 172.1 | +42.6 | 223.2 | 0.195 |
| 1 | 182.9 | +16.1 | 247.2 | 0.202 |
| 4 | 172.5 | +0.9 | 248.8 | 0.197 |
| 9 | 166.7 | +0.3 | 246.2 | 0.205 |
| 16 | 163.5 | +1.0 | 244.4 | 0.201 |
| 32 | 161.6 | -0.4 | 245.8 | 0.200 |

## 5. Per-condition ParamMean on `test_extrap`

| sim_id | RMSE | Bias | High-IoU |
|--------|-----:|-----:|---------:|
| `0NH3_1.3.h5` | 217.3 | +46.5 | 0.486 |
| `20NH3_0.75.h5` | 120.7 | +90.7 | 0.202 |
| `20NH3_1.3.h5` | 189.4 | +23.7 | 0.002 |
| `80NH3_1.25.h5` | 161.1 | +9.4 | 0.089 |

## 6. Artifacts

| Path | Content |
|------|---------|
| `results/tables/month1_check1_kinematic_correlations.csv` | Check 1 |
| `results/tables/month1_check2_3_4_per_condition_metrics.csv` | Per-sample metrics |
| `results/tables/month1_per_condition_summary.csv` | Per `sim_id` aggregates |
| `results/tables/month1_feasibility_split_summary.csv` | Split means |
| `results/tables/month1_feasibility_meta.json` | Selected hyperparameters |
| `results/figures/month1_feasibility/fig07–fig10` | Figures |

## 7. Implications

1. Centered LOCO ParamMean (`piecewise_phi1`) cuts extrap bias ~+88→~+43 K and val bias ~−35→~−17 K.
2. Conditioned GappyPOD inherits this (~153 K extrap RMSE, ~+41 K bias vs prior ~174 / +84).
3. Oracle POD floor on extrap is ~72–75 K RMSE / High-IoU ~0.30 — linear POD cannot close topology alone.
4. Nested K + AnchorOffset drives extrap |bias| toward ~0–20 K by K=16–32; High-T topology remains hard.
5. K=0 assisted rows match temperature-free Gappy@5% (shared mask seeds + identical frames).

