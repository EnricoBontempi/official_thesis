# 11 — Sweep delle correzioni a K temperature (C149)

**Stato:** codice e controlli locali pronti; i numeri di questo documento sono scritti solo dopo i gate sullo sweep completo. Nessuna selezione di stimatore.

## 1. Contratto e perimetro

Manifest congelato `configs/observations/month2_temperature_anchors_nested_v1.json`, SHA256 del file `a9c997c96cb44489dd1fc0d4484304b44d8beddbf00b73c25baf4e1922a19e54`, contenuto `83fd3b5f2a07e5febc787c384285e265cb4764884fcbc6fb90d22218932f6ae6`. Tre permutazioni annidate 1 ⊂ 4 ⊂ 9 ⊂ 16 ⊂ 32, una per replica di maschera di velocità, siti fissi su tutti i frame. La griglia storica T9 (righe e colonne 32, 64, 95) è un ponte a parte, `layout_kind=t9_bridge`, e non è uno dei tre layout.

Velocità al 5% persistita, seen220 e VAL123, repliche 0/1/2, checkpoint best e last dei dodici run S7. Nessun best_seen, nessun TEST, nessun training. Il protocollo YAML comune non è parsabile (testo dopo la sequenza `metrics`); i siti usati sono quelli del JSON, non una griglia alternativa.

## 2. Correzioni

Su una sola predizione: free; AO_K con tau = 1, `sum(residui)/(K+1)`; media dei residui senza shrink; mediana ordinaria (K pari: media dei due valori centrali); oracle, media del residuo sull’intero campo, solo diagnostico. A K = 9 le tre correzioni con ancore passano dalle funzioni già testate di median9. A K = 0, AO, media e mediana restituiscono il campo free; l’oracle no. Nessun clipping.

Il T-block di GappyPOD non è una traslazione. Usa gli oggetti C149 già fittati (artefatto `a18f1bf2…`) e la coppia storica p_off (0.001, 0.5), p_on (0.01, 0.5). Il w_T = 5 dell’artefatto non entra nel T-block.

## 3. Metriche e gate

Campo completo, dominio comune (griglia meno i 32 siti, uguale a ogni K), off-anchor(K), errore sulle ancore. Regioni con conteggio: un insieme vuoto resta NaN e viene dichiarato; un RMSE globale non finito ferma il run. Il gate free confronta `s7_rmse_K` con il CSV S7 della stessa chiave (run, checkpoint o metodo, ruolo, replica, sim_id, frame), file e hash espliciti, tolleranza macro 0.05 K. Gli scarti per-frame sono scritti anche quando il macro passa. `COMPLETE.json` solo dopo gate e rilettura del CSV.

## 4. Risultati

Non ancora: lo sweep scientifico non è stato letto. Le tabelle e le figure stanno in `results/month2/temperature_k_sweep/` quando il pacchetto è stato riletto. Fino a quel momento questo paragrafo non riporta un valore di RMSE.
