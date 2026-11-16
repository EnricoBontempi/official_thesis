# Month 2 — Baseline validation (Step 2.3)

**Experiment:** `month2_baselines_v1`
**Smoke:** False
**Wall time:** 1918.3s; **peak RSS:** 10441 MiB
**Store decodes:** 23

## Preregistered search
```yaml
aggregation_order:
- frame
- condition
- split_VAL
- then_mean_over_mask_replicas
budget_note: "same \u03BB/w_T grids for p_off and p_on; no iterative grid expansion\
  \ after scores"
registered_before_scores: true
selection_metric: mean_over_replicas_of_mean_condition_full_field_RMSE_on_VAL
temperature_assisted_primary:
  K: 9
  includes_historical_pair:
    ridge_lambda: 0.01
    w_T: 5.0
  n_combinations: 9
  ridge_lambda:
  - 0.001
  - 0.01
  - 0.1
  w_T:
  - 0.5
  - 5.0
  - 50.0
temperature_free:
  K: 0
  extends_beyond_historical_upper_edge_0.01: true
  includes_historical_lambda: 0.01
  n_combinations: 5
  ridge_lambda:
  - 0.0001
  - 0.001
  - 0.01
  - 0.1
  - 1.0
  w_T_fixed_unused: 5.0
```

### Selected hyperparameters
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

### Search table (head)
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

## Summary — seen_train_eval
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

## Summary — val_eval
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

## VAL per-condition (selected sparse methods)

```
              method       sim_id  p_conditioning  K       variant  mask_replica_id     rmse_K  abs_bias_K  centered_rmse_K
     GappyPOD_Tblock 0NH3_1.05.h5           False  0        Tblock                0 124.962991   81.741403        91.732744
     GappyPOD_Tblock 0NH3_1.05.h5           False  0        Tblock                1 124.389030   81.653895        91.177014
     GappyPOD_Tblock 0NH3_1.05.h5           False  0        Tblock                2 124.459647   81.546306        91.717883
     GappyPOD_Tblock 0NH3_1.05.h5            True  0        Tblock                0  93.902311   19.983158        90.649749
     GappyPOD_Tblock 0NH3_1.05.h5            True  0        Tblock                1  94.363699   19.770520        91.189436
     GappyPOD_Tblock 0NH3_1.05.h5            True  0        Tblock                2  94.132062   19.791637        91.001102
     GappyPOD_Tblock 20NH3_1.1.h5           False  0        Tblock                0  98.977716   72.042965        66.647163
     GappyPOD_Tblock 20NH3_1.1.h5           False  0        Tblock                1  98.118158   69.986685        67.221936
     GappyPOD_Tblock 20NH3_1.1.h5           False  0        Tblock                2  97.932932   70.009301        66.873245
     GappyPOD_Tblock 20NH3_1.1.h5            True  0        Tblock                0  69.726098    9.370923        68.957008
     GappyPOD_Tblock 20NH3_1.1.h5            True  0        Tblock                1  69.448998    9.204235        68.705928
     GappyPOD_Tblock 20NH3_1.1.h5            True  0        Tblock                2  69.624463    9.344143        68.849698
     GappyPOD_Tblock 40NH3_0.9.h5           False  0        Tblock                0  61.483419   21.286584        56.541450
     GappyPOD_Tblock 40NH3_0.9.h5           False  0        Tblock                1  60.532511   18.625197        56.113164
     GappyPOD_Tblock 40NH3_0.9.h5           False  0        Tblock                2  62.331488   21.317067        57.199848
     GappyPOD_Tblock 40NH3_0.9.h5            True  0        Tblock                0  47.986645   16.882740        44.772687
     GappyPOD_Tblock 40NH3_0.9.h5            True  0        Tblock                1  48.074575   17.130601        44.759241
     GappyPOD_Tblock 40NH3_0.9.h5            True  0        Tblock                2  48.127900   16.963504        44.858994
     GappyPOD_Tblock 0NH3_1.05.h5           False  9        Tblock                0  93.038341   23.511669        89.238956
     GappyPOD_Tblock 0NH3_1.05.h5           False  9        Tblock                1  90.396044   17.451950        87.749193
     GappyPOD_Tblock 0NH3_1.05.h5           False  9        Tblock                2  93.231017   24.054140        88.841940
     GappyPOD_Tblock 0NH3_1.05.h5            True  9        Tblock                0  90.204132   19.648461        87.094227
     GappyPOD_Tblock 0NH3_1.05.h5            True  9        Tblock                1  91.363464   19.031278        88.382232
     GappyPOD_Tblock 0NH3_1.05.h5            True  9        Tblock                2  90.370574   19.340417        87.448377
     GappyPOD_Tblock 20NH3_1.1.h5           False  9        Tblock                0  66.544931   13.172471        64.506544
     GappyPOD_Tblock 20NH3_1.1.h5           False  9        Tblock                1  67.122865   14.099281        64.743312
     GappyPOD_Tblock 20NH3_1.1.h5           False  9        Tblock                2  67.339320   16.476314        64.410453
     GappyPOD_Tblock 20NH3_1.1.h5            True  9        Tblock                0  65.743948    9.389426        64.882865
     GappyPOD_Tblock 20NH3_1.1.h5            True  9        Tblock                1  66.487506    9.191927        65.582170
     GappyPOD_Tblock 20NH3_1.1.h5            True  9        Tblock                2  64.504299    9.928208        63.426913
     GappyPOD_Tblock 40NH3_0.9.h5           False  9        Tblock                0  51.815138    7.426153        50.996190
     GappyPOD_Tblock 40NH3_0.9.h5           False  9        Tblock                1  54.534947   10.382951        53.104764
     GappyPOD_Tblock 40NH3_0.9.h5           False  9        Tblock                2  54.333950    8.823754        53.225021
     GappyPOD_Tblock 40NH3_0.9.h5            True  9        Tblock                0  45.807624   15.834534        42.850765
     GappyPOD_Tblock 40NH3_0.9.h5            True  9        Tblock                1  46.360058   16.110994        43.317696
     GappyPOD_Tblock 40NH3_0.9.h5            True  9        Tblock                2  46.219534   16.276871        43.066994
     GappyPOD_Tblock 0NH3_1.05.h5           False 32        Tblock                0  83.008145   16.470001        80.857685
     GappyPOD_Tblock 0NH3_1.05.h5           False 32        Tblock                1  79.256112   14.541677        77.287268
     GappyPOD_Tblock 0NH3_1.05.h5           False 32        Tblock                2  83.239418   16.970055        80.888499
     GappyPOD_Tblock 0NH3_1.05.h5            True 32        Tblock                0  85.224780   18.444861        82.316798
     GappyPOD_Tblock 0NH3_1.05.h5            True 32        Tblock                1  83.931878   18.413429        81.089584
     GappyPOD_Tblock 0NH3_1.05.h5            True 32        Tblock                2  85.598617   18.856598        82.675426
     GappyPOD_Tblock 20NH3_1.1.h5           False 32        Tblock                0  61.823512   12.767561        59.644388
     GappyPOD_Tblock 20NH3_1.1.h5           False 32        Tblock                1  61.536615   12.651505        59.650135
     GappyPOD_Tblock 20NH3_1.1.h5           False 32        Tblock                2  60.423557   11.949376        58.666915
     GappyPOD_Tblock 20NH3_1.1.h5            True 32        Tblock                0  61.188695    9.122741        60.298120
     GappyPOD_Tblock 20NH3_1.1.h5            True 32        Tblock                1  62.234815    9.253570        61.359536
     GappyPOD_Tblock 20NH3_1.1.h5            True 32        Tblock                2  60.189795    9.952865        59.150196
     GappyPOD_Tblock 40NH3_0.9.h5           False 32        Tblock                0  46.547202    5.603311        46.042950
     GappyPOD_Tblock 40NH3_0.9.h5           False 32        Tblock                1  49.172719    7.637066        48.275033
     GappyPOD_Tblock 40NH3_0.9.h5           False 32        Tblock                2  49.162515    5.838684        48.608219
     GappyPOD_Tblock 40NH3_0.9.h5            True 32        Tblock                0  43.785816   15.881413        40.680041
     GappyPOD_Tblock 40NH3_0.9.h5            True 32        Tblock                1  44.344952   15.741931        41.303506
     GappyPOD_Tblock 40NH3_0.9.h5            True 32        Tblock                2  44.249954   15.721316        41.200985
       GappyPOD_free 0NH3_1.05.h5           False  0          free                0 124.962991   81.741403        91.732744
       GappyPOD_free 0NH3_1.05.h5           False  0          free                1 124.389030   81.653895        91.177014
       GappyPOD_free 0NH3_1.05.h5           False  0          free                2 124.459647   81.546306        91.717883
       GappyPOD_free 0NH3_1.05.h5            True  0          free                0  93.902311   19.983158        90.649749
       GappyPOD_free 0NH3_1.05.h5            True  0          free                1  94.363699   19.770520        91.189436
       GappyPOD_free 0NH3_1.05.h5            True  0          free                2  94.132062   19.791637        91.001102
       GappyPOD_free 20NH3_1.1.h5           False  0          free                0  98.977716   72.042965        66.647163
       GappyPOD_free 20NH3_1.1.h5           False  0          free                1  98.118158   69.986685        67.221936
       GappyPOD_free 20NH3_1.1.h5           False  0          free                2  97.932932   70.009301        66.873245
       GappyPOD_free 20NH3_1.1.h5            True  0          free                0  69.726098    9.370923        68.957008
       GappyPOD_free 20NH3_1.1.h5            True  0          free                1  69.448998    9.204235        68.705928
       GappyPOD_free 20NH3_1.1.h5            True  0          free                2  69.624463    9.344143        68.849698
       GappyPOD_free 40NH3_0.9.h5           False  0          free                0  61.483419   21.286584        56.541450
       GappyPOD_free 40NH3_0.9.h5           False  0          free                1  60.532511   18.625197        56.113164
       GappyPOD_free 40NH3_0.9.h5           False  0          free                2  62.331488   21.317067        57.199848
       GappyPOD_free 40NH3_0.9.h5            True  0          free                0  47.986645   16.882740        44.772687
       GappyPOD_free 40NH3_0.9.h5            True  0          free                1  48.074575   17.130601        44.759241
       GappyPOD_free 40NH3_0.9.h5            True  0          free                2  48.127900   16.963504        44.858994
    GappyPOD_plus_AO 0NH3_1.05.h5           False  9            AO                0  95.268507   21.350820        91.732744
    GappyPOD_plus_AO 0NH3_1.05.h5           False  9            AO                1  94.552174   19.894532        91.177014
    GappyPOD_plus_AO 0NH3_1.05.h5           False  9            AO                2  97.080024   24.060380        91.717883
    GappyPOD_plus_AO 0NH3_1.05.h5            True  9            AO                0  94.218377   18.133985        90.649749
    GappyPOD_plus_AO 0NH3_1.05.h5            True  9            AO                1  93.934246   17.363020        91.189436
    GappyPOD_plus_AO 0NH3_1.05.h5            True  9            AO                2  95.196963   19.759540        91.001102
    GappyPOD_plus_AO 20NH3_1.1.h5           False  9            AO                0  69.963399   15.862826        66.647163
    GappyPOD_plus_AO 20NH3_1.1.h5           False  9            AO                1  69.657326   14.439656        67.221936
    GappyPOD_plus_AO 20NH3_1.1.h5           False  9            AO                2  70.270137   18.111472        66.873245
    GappyPOD_plus_AO 20NH3_1.1.h5            True  9            AO                0  71.313881   14.113280        68.957008
    GappyPOD_plus_AO 20NH3_1.1.h5            True  9            AO                1  70.847549   13.318633        68.705928
    GappyPOD_plus_AO 20NH3_1.1.h5            True  9            AO                2  71.965366   16.495068        68.849698
    GappyPOD_plus_AO 40NH3_0.9.h5           False  9            AO                0  58.758942   12.335408        56.541450
    GappyPOD_plus_AO 40NH3_0.9.h5           False  9            AO                1  58.392519   12.939030        56.113164
    GappyPOD_plus_AO 40NH3_0.9.h5           False  9            AO                2  60.374448   15.931680        57.199848
    GappyPOD_plus_AO 40NH3_0.9.h5            True  9            AO                0  45.802423    7.144852        44.772687
    GappyPOD_plus_AO 40NH3_0.9.h5            True  9            AO                1  46.129151    8.951587        44.759241
    GappyPOD_plus_AO 40NH3_0.9.h5            True  9            AO                2  45.965150    7.925966        44.858994
    GappyPOD_plus_AO 0NH3_1.05.h5           False 32            AO                0  92.673703   11.294595        91.732744
    GappyPOD_plus_AO 0NH3_1.05.h5           False 32            AO                1  92.188634   10.206552        91.177014
    GappyPOD_plus_AO 0NH3_1.05.h5           False 32            AO                2  92.991408   13.102021        91.717883
    GappyPOD_plus_AO 0NH3_1.05.h5            True 32            AO                0  91.781286   12.101667        90.649749
    GappyPOD_plus_AO 0NH3_1.05.h5            True 32            AO                1  92.253138   10.374640        91.189436
    GappyPOD_plus_AO 0NH3_1.05.h5            True 32            AO                2  92.186849   12.419624        91.001102
    GappyPOD_plus_AO 20NH3_1.1.h5           False 32            AO                0  67.661184    9.283319        66.647163
    GappyPOD_plus_AO 20NH3_1.1.h5           False 32            AO                1  68.201682    9.053466        67.221936
    GappyPOD_plus_AO 20NH3_1.1.h5           False 32            AO                2  67.741484    8.205812        66.873245
    GappyPOD_plus_AO 20NH3_1.1.h5            True 32            AO                0  70.194985   10.649623        68.957008
    GappyPOD_plus_AO 20NH3_1.1.h5            True 32            AO                1  69.589386    8.362459        68.705928
    GappyPOD_plus_AO 20NH3_1.1.h5            True 32            AO                2  70.054409    9.917275        68.849698
    GappyPOD_plus_AO 40NH3_0.9.h5           False 32            AO                0  57.270149    6.959458        56.541450
    GappyPOD_plus_AO 40NH3_0.9.h5           False 32            AO                1  57.014275    8.322226        56.113164
    GappyPOD_plus_AO 40NH3_0.9.h5           False 32            AO                2  57.977228    7.632194        57.199848
    GappyPOD_plus_AO 40NH3_0.9.h5            True 32            AO                0  45.462856    5.779318        44.772687
    GappyPOD_plus_AO 40NH3_0.9.h5            True 32            AO                1  45.346251    5.591747        44.759241
    GappyPOD_plus_AO 40NH3_0.9.h5            True 32            AO                2  45.364568    5.174308        44.858994
     GlobalTrainMean 0NH3_1.05.h5           False  0   zero_sensor                0 214.503958  142.749245       155.071260
     GlobalTrainMean 0NH3_1.05.h5           False  0   zero_sensor                1 214.503958  142.749245       155.071260
     GlobalTrainMean 0NH3_1.05.h5           False  0   zero_sensor                2 214.503958  142.749245       155.071260
     GlobalTrainMean 20NH3_1.1.h5           False  0   zero_sensor                0 106.647226   53.339141        92.241423
     GlobalTrainMean 20NH3_1.1.h5           False  0   zero_sensor                1 106.647226   53.339141        92.241423
     GlobalTrainMean 20NH3_1.1.h5           False  0   zero_sensor                2 106.647226   53.339141        92.241423
     GlobalTrainMean 40NH3_0.9.h5           False  0   zero_sensor                0  97.949823   11.982637        97.197018
     GlobalTrainMean 40NH3_0.9.h5           False  0   zero_sensor                1  97.949823   11.982637        97.197018
     GlobalTrainMean 40NH3_0.9.h5           False  0   zero_sensor                2  97.949823   11.982637        97.197018
           OraclePOD 0NH3_1.05.h5           False -1 oracle_full_T                0  44.977308    0.737511        44.967905
           OraclePOD 0NH3_1.05.h5           False -1 oracle_full_T                1  44.977308    0.737511        44.967905
           OraclePOD 0NH3_1.05.h5           False -1 oracle_full_T                2  44.977308    0.737511        44.967905
           OraclePOD 0NH3_1.05.h5            True -1 oracle_full_T                0  45.674929    2.582402        45.562364
           OraclePOD 0NH3_1.05.h5            True -1 oracle_full_T                1  45.674929    2.582402        45.562364
           OraclePOD 0NH3_1.05.h5            True -1 oracle_full_T                2  45.674929    2.582402        45.562364
           OraclePOD 20NH3_1.1.h5           False -1 oracle_full_T                0  32.373234    0.334909        32.370475
           OraclePOD 20NH3_1.1.h5           False -1 oracle_full_T                1  32.373234    0.334909        32.370475
           OraclePOD 20NH3_1.1.h5           False -1 oracle_full_T                2  32.373234    0.334909        32.370475
           OraclePOD 20NH3_1.1.h5            True -1 oracle_full_T                0  33.916460    1.467706        33.862418
           OraclePOD 20NH3_1.1.h5            True -1 oracle_full_T                1  33.916460    1.467706        33.862418
           OraclePOD 20NH3_1.1.h5            True -1 oracle_full_T                2  33.916460    1.467706        33.862418
           OraclePOD 40NH3_0.9.h5           False -1 oracle_full_T                0  26.066509    0.628976        26.057342
           OraclePOD 40NH3_0.9.h5           False -1 oracle_full_T                1  26.066509    0.628976        26.057342
           OraclePOD 40NH3_0.9.h5           False -1 oracle_full_T                2  26.066509    0.628976        26.057342
           OraclePOD 40NH3_0.9.h5            True -1 oracle_full_T                0  26.183743    1.769607        26.107775
           OraclePOD 40NH3_0.9.h5            True -1 oracle_full_T                1  26.183743    1.769607        26.107775
           OraclePOD 40NH3_0.9.h5            True -1 oracle_full_T                2  26.183743    1.769607        26.107775
ParamConditionedMean 0NH3_1.05.h5            True  0   zero_sensor                0 132.846251   28.358165       127.629807
ParamConditionedMean 0NH3_1.05.h5            True  0   zero_sensor                1 132.846251   28.358165       127.629807
ParamConditionedMean 0NH3_1.05.h5            True  0   zero_sensor                2 132.846251   28.358165       127.629807
ParamConditionedMean 20NH3_1.1.h5            True  0   zero_sensor                0  81.407575    7.030853        81.052473
ParamConditionedMean 20NH3_1.1.h5            True  0   zero_sensor                1  81.407575    7.030853        81.052473
ParamConditionedMean 20NH3_1.1.h5            True  0   zero_sensor                2  81.407575    7.030853        81.052473
ParamConditionedMean 40NH3_0.9.h5            True  0   zero_sensor                0  49.771076   19.365616        45.775815
ParamConditionedMean 40NH3_0.9.h5            True  0   zero_sensor                1  49.771076   19.365616        45.775815
ParamConditionedMean 40NH3_0.9.h5            True  0   zero_sensor                2  49.771076   19.365616        45.775815
```

## Scales / isolation / costs
```json
{
  "pod_vs_normalizer_scales": {
    "pod_scale_T": 331.73782983147095,
    "pod_scale_vi": 5.673608242065474,
    "pod_scale_vj": 11.909608084386177,
    "month2_sigma_T": 331.7378431423074,
    "note": "POD uses its own fit-only train_fit std for state scaling; Month-2 normalizer is separate and used for NRMSE / neural path."
  },
  "invariant_checks": {
    "same_velocity_mask_K0_K9": true,
    "K0_tblock_matches_free_maxabs": 0.0,
    "AO_centered_rmse_free": 80.54356426622638,
    "AO_centered_rmse_ao": 80.54356426622638,
    "AO_centered_rmse_absdiff": 0.0,
    "hidden_velocity_mutation_maxabs": 0.0,
    "observed_only_containers": true,
    "off_anchor_T_not_in_model_inputs": true,
    "vk_placeholder_zero": true,
    "same_velocity_mask_ok": true,
    "adapter_vs_historical_maxabs": 0.0,
    "adapter_vs_historical_ok": true
  },
  "fit_times_s": {
    "means": 1.7011091709136963,
    "pod": 9.340945959091187
  },
  "hgb_included": true
}
```

## Compact comparator table (VAL / seen, mean condition RMSE)

Units K. Site fraction 5%, 2C, R=32. Replicas 0/1/2 averaged after per-config aggregation.

| Role | Method | p | K | variant | RMSE | \|bias\| | Centered-RMSE |
|---|---|---:|---:|---|---:|---:|---:|
| VAL | GlobalTrainMean | off | 0 | zero | 139.7 | 69.4 | 114.8 |
| VAL | ParamConditionedMean | on | 0 | zero | 88.0 | 18.3 | 84.8 |
| VAL | GappyPOD_free | off | 0 | free | 94.8 | 57.6 | 71.7 |
| VAL | GappyPOD_free | on | 0 | free | 70.6 | 15.4 | 68.2 |
| VAL | GappyPOD_plus_AO | off | 9 | AO | 74.9 | 17.2 | 71.7 |
| VAL | GappyPOD_plus_AO | on | 9 | AO | 70.6 | 13.7 | 68.2 |
| VAL | GappyPOD_plus_AO | off | 32 | AO | 72.6 | 9.3 | 71.7 |
| VAL | GappyPOD_plus_AO | on | 32 | AO | 69.1 | 8.9 | 68.2 |
| VAL | GappyPOD_Tblock | off | 9 | Tblock | 70.9 | 15.0 | 68.5 |
| VAL | GappyPOD_Tblock | on | 9 | Tblock | 67.5 | 15.0 | 65.1 |
| VAL | GappyPOD_Tblock | off | 32 | Tblock | 63.8 | 11.6 | 62.2 |
| VAL | GappyPOD_Tblock | on | 32 | Tblock | 63.4 | 14.6 | 61.1 |
| VAL | OraclePOD (full T) | off | — | oracle | 34.5 | 0.57 | 34.5 |
| VAL | OraclePOD (full T) | on | — | oracle | 35.3 | 1.94 | 35.2 |
| seen | GappyPOD_free | off | 0 | free | 113.1 | 35.5 | 103.5 |
| seen | GappyPOD_free | on | 0 | free | 108.4 | 14.9 | 106.6 |
| seen | GappyPOD_Tblock | on | 9 | Tblock | 103.2 | 14.5 | 101.5 |
| seen | OraclePOD (full T) | off | — | oracle | 53.0 | 0.76 | 53.0 |

Dense diagnostic (not sparse-comparable): LocalHGB_dense VAL p_off RMSE 137.0 / p_on 92.7.

## Free vs AO vs T-block vs Oracle

- **Free** = velocity-only MAP (K=0). Selected λ: p_off `0.001`, p_on `0.01`.
- **AO** = free prediction + uniform anchor offset (τ=1); Centered-RMSE identical to free (verified).
- **T-block** = joint MAP with T anchors; selected at K=9 then evaluated at K=32 without retuning. Selected: p_off `(λ=0.001, w_T=0.5)`, p_on `(λ=0.01, w_T=0.5)`.
- **OraclePOD** = least-squares projection of true T onto the TRAIN-fit T POD basis (uses full T; not deployable). VAL oracle ~34–35 K vs best sparse T-block ~63–67 K → estimation gap remains large relative to representation error.

No monotonic improvement with K is assumed; AO RMSE can drop mainly via bias while Centered-RMSE stays flat.

## Costs and local defect

- Wall ~1918 s (~32 min); peak RSS ~10.4 GiB; shard decodes = 23 (20 TRAIN + 3 VAL), one decode per condition for fit+seen.
- POD fit ~9.3 s; means+LOCO ~1.7 s; HGB dense ~1.6 s.
- Matrix sizes: state dim ≈ 3×128×128 (vi,vj,T) for 2C; R=32; 760 fit frames.
- **Local data defect:** shard `40NH3_1.2.h5` has `shape_t=2001` numerical frames but `t_shape=2000`. `ShardStore` pads the missing last timestamp by linear extrapolation (`t_coord_pads=1`). Fields at index 2000 are real; only the time coordinate was short.

## Questions for independent review (before Step 2.4)

1. Is the estimation gap (Oracle ~35 K vs sparse ~65–95 K on VAL) large enough to prioritize neural capacity, or should R=64 / richer priors be tried first on the same protocol?
2. Assisted T-block prefers `w_T=0.5` (lower edge of the preregistered grid) for both p modes — discuss whether to extend below 0.5 or treat as evidence that light T weighting is enough at K=9.
3. LocalHGB dense with p still trails ParamMean / GappyPOD_free p_on on VAL RMSE — keep as dense diagnostic only, or drop from neural comparisons?
4. Confirm that AO and T-block remain separately labeled in any future neural assisted protocols.
5. Engaging/GPU still unverified; Step 2.4 should not assume cluster readiness.

## Status: results written for independent review before Step 2.4

No U-Net / GLU / Step 2.4 started.

