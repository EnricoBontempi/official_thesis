# Month-2 baseline run — automatic report

- experiment_id: `month2_baselines_v1`
- run_id: `consolidation_v1`
- smoke: False
- wall_time_s: 1440.702
- peak_rss_mb: 10357.109
- shard_decodes: 23
- git_head_preflight: `2c800cedd1cccc6935eb64c6af57fbdd6001aee8`

## Selected hyperparameters
```json
{
  "free_p_off_lambda": 0.001,
  "free_p_on_lambda": 0.01,
  "tblock_p_off": {
    "ridge_lambda": 0.001,
    "w_T": 0.5
  },
  "tblock_p_on": {
    "ridge_lambda": 0.01,
    "w_T": 0.5
  },
  "param_mean": {
    "basis": "piecewise_phi1",
    "ridge_alpha": 0.01,
    "loco_rmse_K": 71.96259270478093
  }
}
```

## Search table
```
                       task  p_conditioning  K  ridge_lambda  w_T  R  score_mean_condition_rmse_K  n_val_frames  n_replicas
           temperature_free           False  0        0.0001  NaN 32                   106.190654           123           3
           temperature_free           False  0        0.0010  NaN 32                    94.798655           123           3
           temperature_free           False  0        0.0100  NaN 32                    97.173089           123           3
           temperature_free           False  0        0.1000  NaN 32                   123.226820           123           3
           temperature_free           False  0        1.0000  NaN 32                   136.958806           123           3
           temperature_free            True  0        0.0001  NaN 32                    84.254712           123           3
           temperature_free            True  0        0.0010  NaN 32                    72.506643           123           3
           temperature_free            True  0        0.0100  NaN 32                    70.598528           123           3
           temperature_free            True  0        0.1000  NaN 32                    80.880717           123           3
           temperature_free            True  0        1.0000  NaN 32                    86.881431           123           3
temperature_assisted_Tblock           False  9        0.0010  0.5 32                    70.928506           123           3
temperature_assisted_Tblock           False  9        0.0010  5.0 32                    71.726430           123           3
temperature_assisted_Tblock           False  9        0.0010 50.0 32                    77.531460           123           3
temperature_assisted_Tblock           False  9        0.0100  0.5 32                    75.453483           123           3
temperature_assisted_Tblock           False  9        0.0100  5.0 32                    70.959821           123           3
temperature_assisted_Tblock           False  9        0.0100 50.0 32                    72.673254           123           3
temperature_assisted_Tblock           False  9        0.1000  0.5 32                   109.255190           123           3
temperature_assisted_Tblock           False  9        0.1000  5.0 32                    92.323398           123           3
temperature_assisted_Tblock           False  9        0.1000 50.0 32                    86.691478           123           3
temperature_assisted_Tblock            True  9        0.0010  0.5 32                    69.866421           123           3
temperature_assisted_Tblock            True  9        0.0010  5.0 32                    72.218609           123           3
temperature_assisted_Tblock            True  9        0.0010 50.0 32                    82.183097           123           3
temperature_assisted_Tblock            True  9        0.0100  0.5 32                    67.451238           123           3
temperature_assisted_Tblock            True  9        0.0100  5.0 32                    68.202650           123           3
temperature_assisted_Tblock            True  9        0.0100 50.0 32                    70.842097           123           3
temperature_assisted_Tblock            True  9        0.1000  0.5 32                    78.274077           123           3
temperature_assisted_Tblock            True  9        0.1000  5.0 32                    75.918168           123           3
temperature_assisted_Tblock            True  9        0.1000 50.0 32                    75.968820           123           3
```

## Summary seen_train_eval
```
              method  p_conditioning  K          variant     rmse_K  abs_bias_K  centered_rmse_K
     GappyPOD_Tblock           False  0           Tblock 113.075205   35.473472       103.457899
     GappyPOD_Tblock           False  9           Tblock 101.848368   17.022674        99.654007
     GappyPOD_Tblock           False 32           Tblock  92.535087   13.056458        91.128447
     GappyPOD_Tblock            True  0           Tblock 108.362362   14.947735       106.585260
     GappyPOD_Tblock            True  9           Tblock 103.169793   14.484413       101.455532
     GappyPOD_Tblock            True 32           Tblock  96.896463   13.533153        95.255431
       GappyPOD_free           False  0             free 113.075205   35.473472       103.457899
       GappyPOD_free            True  0             free 108.362362   14.947735       106.585260
    GappyPOD_plus_AO           False  9               AO 107.887590   24.122425       103.457899
    GappyPOD_plus_AO           False 32               AO 104.964520   14.217306       103.457899
    GappyPOD_plus_AO            True  9               AO 111.205277   24.440618       106.585260
    GappyPOD_plus_AO            True 32               AO 108.094428   14.400481       106.585260
     GlobalTrainMean           False  0      zero_sensor 173.684504   71.082906       151.380321
      LocalHGB_dense           False  0 dense_diagnostic 166.685356   76.423329       142.767514
      LocalHGB_dense            True  0 dense_diagnostic 117.008758   12.955970       115.861608
           OraclePOD           False -1    oracle_full_T  53.037219    0.762057        53.028805
           OraclePOD            True -1    oracle_full_T  52.612153    2.573685        52.510039
ParamConditionedMean            True  0      zero_sensor 129.347859   19.079606       127.173042
   ParamMean_plus_AO            True  9               AO 132.884213   30.078738       127.173042
   ParamMean_plus_AO            True 32               AO 128.946661   17.186139       127.173042
```

## Summary val_eval
```
              method  p_conditioning  K          variant     rmse_K  abs_bias_K  centered_rmse_K
     GappyPOD_Tblock           False  0           Tblock  94.798655   57.578823        71.691605
     GappyPOD_Tblock           False  9           Tblock  70.928506   15.044298        68.535153
     GappyPOD_Tblock           False 32           Tblock  63.796644   11.603248        62.213455
     GappyPOD_Tblock            True  0           Tblock  70.598528   15.382384        68.193760
     GappyPOD_Tblock            True  9           Tblock  67.451238   14.972457        65.116916
     GappyPOD_Tblock            True 32           Tblock  63.416589   14.598747        61.119355
       GappyPOD_free           False  0             free  94.798655   57.578823        71.691605
       GappyPOD_free            True  0             free  70.598528   15.382384        68.193760
    GappyPOD_plus_AO           False  9               AO  74.924164   17.213978        71.691605
    GappyPOD_plus_AO           False 32               AO  72.635528    9.339960        71.691605
    GappyPOD_plus_AO            True  9               AO  70.597012   13.689548        68.193760
    GappyPOD_plus_AO            True 32               AO  69.137081    8.930073        68.193760
     GlobalTrainMean           False  0      zero_sensor 139.700336   69.357008       114.836567
      LocalHGB_dense           False  0 dense_diagnostic 136.990950   72.813526       109.663844
      LocalHGB_dense            True  0 dense_diagnostic  92.726538   14.905831        91.224742
           OraclePOD           False -1    oracle_full_T  34.472350    0.567132        34.465240
           OraclePOD            True -1    oracle_full_T  35.258377    1.939905        35.177519
ParamConditionedMean            True  0      zero_sensor  88.008301   18.251545        84.819365
   ParamMean_plus_AO            True  9               AO  87.814235   17.624717        84.819365
   ParamMean_plus_AO            True 32               AO  85.929045   10.915381        84.819365
```

## Invariant checks / costs
```json
{
  "invariant_checks": {
    "n_cases": 4,
    "atol": 1e-08,
    "max_frames": 2,
    "passed": true,
    "records": [
      {
        "sim_id": "0NH3_1.05.h5",
        "frame": 0,
        "p_conditioning": false,
        "hidden_vel_maxabs": 0.0,
        "vk_maxabs": 0.0,
        "free_T_mut_maxabs": 0.0,
        "off_anchor_T_maxabs": 0.0,
        "observed_vel_maxabs": 192.9041419837033,
        "K0_tblock_vs_free": 0.0,
        "AO_centered_rmse_absdiff": 0.0
      },
      {
        "sim_id": "0NH3_1.05.h5",
        "frame": 0,
        "p_conditioning": true,
        "hidden_vel_maxabs": 0.0,
        "vk_maxabs": 0.0,
        "free_T_mut_maxabs": 0.0,
        "off_anchor_T_maxabs": 0.0,
        "observed_vel_maxabs": 62.51869625890686,
        "K0_tblock_vs_free": 0.0,
        "AO_centered_rmse_absdiff": 1.4210854715202004e-14
      },
      {
        "sim_id": "0NH3_1.05.h5",
        "frame": 50,
        "p_conditioning": false,
        "hidden_vel_maxabs": 0.0,
        "vk_maxabs": 0.0,
        "free_T_mut_maxabs": 0.0,
        "off_anchor_T_maxabs": 0.0,
        "observed_vel_maxabs": 138.237522385494,
        "K0_tblock_vs_free": 0.0,
        "AO_centered_rmse_absdiff": 0.0
      },
      {
        "sim_id": "0NH3_1.05.h5",
        "frame": 50,
        "p_conditioning": true,
        "hidden_vel_maxabs": 0.0,
        "vk_maxabs": 0.0,
        "free_T_mut_maxabs": 0.0,
        "off_anchor_T_maxabs": 0.0,
        "observed_vel_maxabs": 53.55275945624362,
        "K0_tblock_vs_free": 0.0,
        "AO_centered_rmse_absdiff": 0.0
      }
    ],
    "summary": {
      "max_hidden_vel": 0.0,
      "max_off_anchor": 0.0,
      "min_observed_response": 53.55275945624362,
      "max_AO_crmse_diff": 1.4210854715202004e-14
    },
    "adapter_vs_historical_maxabs": 0.0,
    "adapter_vs_historical_ok": true
  },
  "fit_times_s": {
    "means": 0.8105595111846924,
    "pod": 5.915160655975342
  },
  "store_stats": {
    "shard_decodes": 23,
    "cache_hits": 1103,
    "cache_misses": 1103,
    "frames_materialized": 1103,
    "time_unvalidated_frames": 1,
    "known_t_mismatch_hits": 49
  },
  "comparison_rule": {
    "metric_atol_K": 1e-06,
    "metric_rtol": 1e-08,
    "ids_must_match_exactly": true,
    "pairing_keys": [
      "method",
      "role",
      "p_conditioning",
      "K",
      "site_fraction",
      "variant",
      "mask_replica_id",
      "sim_id",
      "frame_index",
      "R",
      "ridge_lambda",
      "w_T",
      "observation_hash"
    ]
  }
}
```

Editorial interpretation belongs in docs/month2/mission_baseline_consolidation_report.md.

