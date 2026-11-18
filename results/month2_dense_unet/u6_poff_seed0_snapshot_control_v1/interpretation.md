# U6 interpretation — p_off ablation + snapshot control (editorial)

**Ambito:** un seed accoppiato (0) confronta U4 p_on e U6 p_off; sul best p_off si ripete lo scambio istantanee con i pairing U5. Non è valutazione finale indipendente.

## Quanto cambia togliere p?

Su VAL macro RMSE, best p_off **39.67 K** vs best p_on **26.93 K** (Δ **+12.7 K**).  
p_off resta sotto GappyPOD denso p_off (93.66 K) e GlobalMean (139.70 K), ma **sopra** GappyPOD p_on (70.46 K) e nettamente sopra U-Net p_on.  
Last diagnostico: p_off 44.10 @600 vs p_on 28.06 @730 — non usare best−last (~4 K qui, ~1 K in U4) come stima di variabilità fra seed.

Per condizione (aligned VAL RMSE K):

| Condizione | p_on | p_off | Δ |
|---|---:|---:|---:|
| 0NH3_1.05 | 32.01 | 45.68 | +13.7 |
| 20NH3_1.1 | 27.61 | 35.65 | +8.0 |
| 40NH3_0.9 | 21.17 | 37.68 | +16.5 |

## Livello vs struttura

Su p_off aligned, `|bias|` è rilevante (≈8–20 K per condizione) e Centered-RMSE resta alto (≈30–39 K): peggiorano **entrambi**, non solo un offset globale.  
Rispetto a p_on, il gap non è spiegabile come puro shift di livello.

## Dipendenza dall’istantanea in p_off

Swap VAL: 39.67 → **102.41** (Δ **+62.7**); seen +102.  
Come in U5, `0NH3_1.05` domina (Δ +153.8); le altre condizioni +16–19 K, soprattutto Centered-RMSE.  
L’allineamento velocità–T istantaneo resta importante anche senza canali p espliciti — per questo seed.

Confronto controllo: p_on U5 Δ VAL ≈+69.6; p_off Δ ≈+62.7. Ordine di grandezza simile; non implica che p sia irrilevante (il livello aligned peggiora comunque senza p).

## Cosa non concludere

- Peggioramento p_off ≠ insufficienza fisica della sola velocità.
- p_off ancora utile vs GlobalMean/GappyPOD p_off ≠ inutilità universale dei parametri.
- Nessuna causalità, ceiling informativo, robustezza multi-seed o successo sparse.
- Un seed accoppiato controlla il confronto, non misura la dispersione fra inizializzazioni.
- VAL usata per selezione/scheduler/arresto in U4 e U6.

## Anomalie / limiti operativi

- Best p_off molto precoce (@30) rispetto a p_on (@160); la curva non migliora il minimo assoluto dopo.
- Primo tentativo eval `29479222` fallito per path symlink; riprovato con `815bb7a` / job `29479532` (aligned già OK al primo tentativo).
- Baseline matched riusate da U3 (no refit); ParamMean non entra nel forward p_off.
- Altri seed U6 non eseguiti (rinviati).
