# Lettura dei 30 specialisti

Numeri letti dai CSV per frame e dai campi congelati di questo job, dopo il controllo di completezza.
Sono stress test sintetici. VAL123 è già servita alla selezione dei modelli originali. seen220 non è un dominio nuovo.
Tre seed di training; le repliche di maschera sono mediate prima e non sono training indipendenti.
Nessun test di significatività. Il minimo–massimo fra seed non è un intervallo di confidenza.

Guadagno = RMSE della baseline pulita − RMSE dello specialista, stesso scenario. Positivo: lo specialista ha errore più basso.
Il costo sul pulito è la differenza su R0. Il recupero nello scenario di training è il guadagno sulla diagonale.
Il trasferimento è il guadagno sugli altri scenari. Una diagonale favorevole non dice nulla, da sola, sugli altri scenari.

## Guadagno nello scenario di training (media sui seed, minimo e massimo fra seed), checkpoint best

### seen220

| famiglia | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|
| U-Net | -1.83 [-7.07, +2.40] | -1.44 [-5.51, +2.69] | +0.73 [-2.94, +7.43] | +0.32 [-3.67, +5.24] | +3.00 [+0.65, +7.02] |
| GLU | +0.35 [-4.12, +4.83] | -2.03 [-6.91, +1.06] | -1.58 [-10.91, +7.74] | +0.22 [-1.40, +2.30] | -5.41 [-21.60, +5.34] |

### VAL123

| famiglia | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|
| U-Net | -0.85 [-1.02, -0.70] | -2.65 [-3.84, -1.80] | -1.88 [-4.77, +0.48] | -0.42 [-1.80, +2.06] | +0.29 [-1.19, +3.24] |
| GLU | -1.26 [-3.85, +1.52] | +0.10 [-2.67, +2.67] | -2.18 [-7.01, +2.63] | -1.90 [-4.23, -0.12] | -2.81 [-9.53, +1.92] |

Il bias è il livello (errore medio con segno). La centered-RMSE è la struttura dopo aver tolto quel livello.
Sulle medie di frame e condizioni, bias e centered-RMSE non si sommano alla RMSE.
La deviazione standard fra seed, nelle schede di variabilità, è dispersione fra training, non incertezza fisica.
La RMSE del campo medio non è la media delle RMSE dei singoli modelli: i due numeri sono scritti separati su quelle pagine.

