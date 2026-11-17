# U5 — interpretazione (separata dall’auto-report)

**Ambito:** controllo di scambio istantanee sul best U4 (epoca 160), p_on seed 0.  
**Non affermato:** causalità fisica, information ceiling, robustezza multi-seed, successo sparse, superiorità universale sul lineare.

## Cosa mostra lo scambio

Su **VAL**, lo scambio vi/vj entro condizione peggiora chiaramente la macro RMSE: **26.93 → 96.55 K** (Δ **+69.6 K**). Su **seen** il peggioramento è ancora più grande: **37.68 → 150.30 K** (Δ **+112.6 K**).

Questo è **evidenza empirica** che l’allineamento fra velocità e temperatura della stessa istantanea conta per questo modello/dati/seed: senza di esso le prestazioni collassano verso (o oltre) i riferimenti lineari matched.

Non prova che la rete “usi solo” l’allineamento, né che l’informazione fisica sia assente quando lo scambio cambia poco in qualche sottoinsieme.

## Livello vs struttura (VAL)

| Condizione | Δ RMSE | Δ Centered-RMSE | Δ \|bias\| | Lettura prudente |
|---|---:|---:|---:|---|
| 0NH3_1.05 | +164.2 | +161.7 | +26.7 | Peggiora soprattutto la **struttura**; anche il livello medio si sposta |
| 20NH3_1.1 | +21.8 | +22.0 | ~0 | Quasi solo **struttura** |
| 40NH3_0.9 | +22.8 | +25.3 | ~0 | Quasi solo **struttura** |

La media VAL nasconde l’anomalia di **0NH3_1.05**, dove lo scambio è molto più dannoso (anche regione jet: Δ jet RMSE ≈ +493 K). Le altre due condizioni peggiorano in modo più moderato ma ancora netto.

Regioni: il fronte/getto (T<1800) peggiora più del bulk caldo (T≥2100) in tutte e tre le VAL; i conteggi `high_n_voxels` restano invariati (soglia sul true).

## Confronto matched (VAL macro RMSE, allineato)

| Metodo | RMSE_K |
|---|---:|
| DenseUNet aligned (best@160) | **26.93** |
| DenseUNet swapped | 96.55 |
| GappyPOD denso p_on (λ=0.01 U3) | 70.46 |
| ParamMean | 88.01 |
| GlobalMean | 139.70 |

Con velocità allineata, la U-Net batte i comparatori densamente matched su questo seed. Con velocità scambiata, **non** resta superiore a GappyPOD/ParamMean: il vantaggio aligned non si trasferisce automaticamente a uno scenario in cui l’istantanea è disallineata.

I metodi ParamMean/GlobalMean non usano la velocità istantanea: non sono soggetti a questo controllo (restano reference di livello/forma media).

## Limiti obbligatori

1. **VAL ha guidato selezione, ReduceLROnPlateau e early stop U4** → non è una valutazione finale indipendente.  
2. **Un solo seed** → nessuna robustezza.  
3. Controllo densamente osservato → **nessuna** inferenza su osservazioni sparse.  
4. Peggioramento allo scambio ≠ dimostrazione causale del meccanismo fisico.  
5. Shard temporalmente anomalo: pairing per **indici**, senza Δt fisici certificati.  
6. Hardware U5: L40S `node3102` (U4 training era `node3008`); replay aligned entro atol 0.05 K (di fatto Δ=0).

## Prossimo passo (fuori incarico)

STOP C: discussione con Enrico/ChatGPT prima di p_off, altri seed o cambi modello.
