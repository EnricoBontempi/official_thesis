# Astra median9 — handoff tecnico della second pass

Branch `astra/median9-second-pass`, worktree `/home/enric/projects/official_thesis_astra_median9`.
Codice del calcolo: `8621c2006d5a72e347caed45d9a90ab7a2057b66`.
Base: report revisionato `82bff35726fb70af9987081802ece3ef718e42bd`. Il pacchetto tecnico `86a925af61accdef0b235ac084827cf639c1040c` ne è il nonno (padre di `54d8df9`). Nessun merge con `mission/dense-unet`. Checkout frozen, worktree di Sonnet e training Opus `29671719`/`29671721` non toccati.

Second pass esplorativa. Non conferma indipendente, non sostituisce A0–A3, non è A4.

## Contratto

`delta_median9` è la mediana ordinaria dei nove residui `T_obs − T_pred`, il quinto valore ordinato. `T_median9 = T_pred + delta_median9`. Nessun tau, shrink, clipping o tuning.

`delta_mean9 = sum(residui) / 9`, senza shrink. AO9 resta `sum(residui) / (9 + 1) = 0.9 * delta_mean9`. La differenza median9−AO9 mescola il cambio di stimatore e la rimozione dello shrink. mean9−AO9 isola lo shrink; median9−mean9 isola lo stimatore. T-only costante non riceve questa ricetta.

## Cosa c’è sugli artefatti

Lettura del 2026-11-19 su Engaging `~/official_thesis_astra_results/level_structure_v1/`, senza forward.

| Artefatto | Contenuto | Basta per la mediana sull’insieme? |
| --- | --- | --- |
| `per_frame_{seen,val}.csv.gz` neurali e baseline | metriche, `delta_ao9_K`, `anchor_mean_resid_K` | No. Manca il vettore dei nove residui. |
| `val_frame1000_rep0_fields.npz` dei quattro neurali | pred e true 128×128, tre condizioni VAL, frame 1000, replica 0, float32, circa 294 KB | Solo quei 12 campi. |
| Baseline | nessun npz | No. |
| Figure PNG | mappe già pubblicate | No. |

`inspect-csv` sul VAL U-Net p_on (`92fb62042bec684fc5ce550ea2a4cbdd5ee44eaac7dfe3cf6ab928c5abdfa19e`) esce 2: la media non determina la mediana. Righe neurali: 1107 VAL e 1980 seen, cioè 123 e 220 frame per tre repliche e tre trattamenti primari.

## Controlli

`tests/test_astra_level.py`: 19 passed, incluso offset uniforme di 40 K (median9 e mean9 = 40 K, AO9 = 36 K), permutazione, uno e due residui estremi, mutazione di T fuori dalle ancore, centered-RMSE invariata dopo la traslazione, oracle come minimo RMSE fra traslazioni sul campo completo. Le righe free/oracle/AO9 di `treat_frame` restano quelle primarie.

Sui 12 campi esemplificativi, free/oracle/AO9 rigenerati contro il CSV primario: scarto massimo 4.1e-5 K (`delta_applied` di AO9, U-Net p_on, `20NH3_1.1`, frame 1000). Tolleranza dichiarata 0.05 K. I riferimenti non sono stati aggiornati. Codice già committato prima di questa lettura.

## Cosa è stato calcolato

Solo i 12 campi esemplificativi. Tabelle in `results/month2/astra_median9_second_pass/example_frame1000_rep0/`: metriche free, oracle, AO9, mean9, median9; nove residui e contributo di ciascuna ancora ad AO9 (`residuo/10`), con indice, riga e colonna; differenze appaiate, inclusi i segni negativi. Nessuno stimatore è stato scelto.

Esempio già citato dal report, un solo frame: U-Net p_off, `40NH3_0.9`, frame 1000, replica 0. Oracle 15.85 K, AO9 55.22 K, mean9 61.36 K, median9 22.24 K. Nello stesso pacchetto ci sono frame in cui median9 allontana la RMSE da AO9, per esempio U-Net p_on su `0NH3_1.05` (median9 43.47 K, AO9 42.55 K). Questo non è l’andamento su seen o su VAL.

I due siti `(32, 95)` e `(95, 95)` sono identificabili nella tabella dei residui. Il loro ruolo sul complesso dei frame non è misurato: i residui completi non esistono.

## Inferenza ancora necessaria

Per seen220, VAL123, repliche 0/1/2, quattro best S5 e le baseline Gappy/ParamMean/GlobalMean serve un nuovo forward, oppure salvare i nove residui durante quel forward. Non è stato sottomesso.

Misura già disponibile, stessi percorsi di A2: i quattro job neurali 29612403–29612406 hanno impiegato 1 min 58 s–2 min 08 s su L40S (4 CPU, 32G, `mit_normal_gpu` / `mit_amf_advanced_gpu`). Il job baseline 29612407 ha impiegato 22 s su CPU. Una ripetizione che aggiunge mean9, median9 e i nove residui ha lo stesso costo di ordine, con uscita nuova `~/official_thesis_astra_results/median9_second_pass/`, senza sovrascrivere `level_structure_v1`. Prima di accettare median9, free/oracle/AO9 vanno confrontati con i CSV primari entro 0.05 K; se falliscono, i riferimenti non si aggiornano. T-only resta etichettato a parte.

A4, sui checkpoint Opus, non è eseguita. Le funzioni non dipendono dal checkpoint, ma il tempo non è stato misurato su quei run e non viene promesso. Opus e la scelta S7 restano aperti.
