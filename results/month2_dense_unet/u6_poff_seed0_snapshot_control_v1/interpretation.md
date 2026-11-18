# U6 interpretation — p_off ablation + snapshot control (editorial)

**Ambito:** un seed accoppiato (0) confronta U4 p_on e U6 p_off; sul best p_off si ripete lo scambio istantanee con i pairing U5. Non è valutazione finale indipendente.  
**Numeri:** verificati dagli artefatti di training/eval (vedi anche `u7_closure_val_level_structure.csv`).  
**Revisioni Opus (riferite da Enrico):** U6 valido / utilizzabile con riserve; non ratifica supervisori.

## Quanto cambia togliere p? (best vs best)

Su VAL macro RMSE, best p_off **39.67 K** vs best p_on **26.93 K** (Δ **+12.7 K**).

**Correzione di ranking:** 39.67 K è **inferiore** (migliore) a GappyPOD denso p_on **70.46 K**, e a ParamMean 88.01 / GappyPOD p_off 93.66 / GlobalMean 139.70.  
Quindi la U-Net p_off **supera in accuratezza tutti i riferimenti baseline densi riportati**, pur restando **peggiore** della U-Net p_on.

Per condizione (aligned VAL RMSE K, best):

| Condizione | p_on | p_off | Δ |
|---|---:|---:|---:|
| 0NH3_1.05 | 32.01 | 45.68 | +13.7 |
| 20NH3_1.1 | 27.61 | 35.65 | +8.0 |
| 40NH3_0.9 | 21.17 | 37.68 | +16.5 |

La macro nasconde che `0NH3_1.05` e `40NH3_0.9` pagano il costo maggiore senza p.

## Livello vs struttura — distinguere best e last

VAL macro (media sulle 3 condizioni; **non** ricomporre RMSE da |bias| e C-RMSE mediati):

| | RMSE | \|bias\| | Centered-RMSE |
|---|---:|---:|---:|
| p_on best @160 | 26.93 | 7.09 | 25.15 |
| p_off best @30 | 39.67 | 15.98 | 34.57 |
| p_on last @730 | 28.06 | 8.91 | 24.44 |
| p_off last @600 | 44.10 | 30.45 | 28.94 |

- **Best vs best:** p_off peggiora **livello e struttura**.
- **Durante il training p_off:** dal best al last la struttura **migliora** (C-RMSE 34.57→28.94) mentre il livello sulle condizioni VAL **peggiora** (|bias| 15.98→30.45) e il RMSE sale.
- **Last vs last:** il divario p_on/p_off è soprattutto di **livello**; la struttura è più vicina.
- Il best precoce @30 è **coerente** con la selezione su RMSE (primo minimo assoluto): non prova da solo un bug o un budget insufficiente.
- **Last** resta diagnostico aggiuntivo; **non** sostituisce retroattivamente il checkpoint selezionato.
- Non usare best−last come stima della variabilità fra seed.

## Dipendenza dall’istantanea in p_off

Swap VAL: 39.67 → **102.41** (Δ **+62.7**); seen +102.  
Come in U5, `0NH3_1.05` domina; le altre condizioni ~+16–19 K, soprattutto Centered-RMSE.  
L’allineamento velocità–T istantaneo resta importante anche senza canali p — per questo seed.

Confronto controllo: p_on U5 Δ VAL ≈+69.6; p_off Δ ≈+62.7 (ordine simile). Ciò **non** implica che p sia irrilevante: il livello aligned peggiora comunque senza p.

## Tracciabilità (limiti documentati, non patch retroattive ai manifest)

- Provenance U6 `clean: false` per **untracked** outputs; `dirty_entries` tracciati vuoto.
- Righe swapped U6 senza `src_frame_index`: pairing ricostruibile da `swap_pairs_reused_from_u5.json`. Raccomandazione futura: annotare il donatore sulle metriche swapped.
- U5 rifittò baseline a λ/iperparametri U3 (scostamento); Opus ≈U3 per-frame. U6 riusò U3 senza refit.

## Cosa non concludere

- Peggioramento p_off ≠ insufficienza fisica della sola velocità.
- p_off meglio delle baseline dense ≠ inutilità universale dei parametri, né garanzia che p_on resti migliore con pochi sensori.
- Nessuna causalità, ceiling informativo, robustezza multi-seed o successo sparse.
- VAL usata per selezione/scheduler/arresto in U4 e U6.
