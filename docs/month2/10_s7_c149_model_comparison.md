# S7 — Confronto C149 U-Net / GLU adattato, temperature-free (21 novembre 2026)

**Stato:** valutazioni, controlli e residui S7 eseguiti e verificati nel perimetro temperature-free. Le metriche sono della **valutazione finale S7** (non del loop). Nessuna regola di selezione preregistrata è stata applicata; la lettura finale è una decisione di sviluppo post-risultati (§9). Pacchetto e indice: `results/month2/s7_c149_comparison/package/INDEX.md`.

## 1. Matrice e provenienza
12 run C149 (2980 frame, 149 per condizione TRAIN), velocità TRAIN 5%, K=0 senza T nel forward: U-Net masked e GLU adattato deterministico × p_on/p_off × seed 0/1/2 (seed modello = seed shuffle). Dieci nuovi training (SHA `f792b73`, array 29799680; glu-on-s1 esteso da 1000 a 1080 epoche con continuazione autorizzata, SHA `501d41a`, job 29966737, tranche 1 preservata con manifest) e due U-Net seed 0 riusati (O-C149-on/off-s0, SHA `d633200`, file e hash verificati). Tutti chiusi per early-stop con la stessa regola; best primario = primo minimo assoluto VAL 5% del loop. Matrice di valutazione congelata `configs/experiments/month2_s7_c149_eval_matrix.json` (SHA256 `a76bac9b…`); valutazione `acc8e2b` (job 29984795/29984812), pacchetto `aadf05c` (job 29988745). Gate loop-parity 5% (macro 0.05 K) 18/18 per run, scarto massimo 0 K; 357372 righe, 972 CSV, `mask_hash` per riga = payload verificato.

## 2. Risultati per famiglia e p (VAL123, 5%, best; media dei 3 seed, [min–max])
| | RMSE | centered | |bias| medio per frame | bias firmato |
|---|---:|---:|---:|---:|
| U-Net p_on | 39.7 [38.2–42.2] | 34.4 | 14.0 | −7.8 |
| U-Net p_off | 40.6 [39.8–41.6] | 35.1 | 16.4 | −8.3 |
| GLU p_on | 42.7 [41.5–43.5] | 41.0 | 9.6 | +1.8 |
| GLU p_off | 44.5 [43.8–45.2] | 40.9 | 15.3 | +0.1 |

I range dei seed si sovrappongono (U-Net p_on seed 1: 42.2; GLU p_on seed 1: 41.5). Risultati individuali: `package/tables/global_by_setting.csv`, `headline_best_5pct_by_run.csv`. **Livello vs struttura:** la U-Net ha centered RMSE più bassa (circa 6 K su VAL) ma bias firmato negativo (≈ −8 K); il GLU ha bias firmato vicino a zero e |bias| minore con p_on. **seen220** (5%, best): U-Net 42.3 (p_on) / 44.8 (p_off); GLU 50.1 / 53.3. **reference760** (parte del pool di fit): U-Net 28 K, GLU 43 K. **best vs last:** su VAL last − best è positivo per tutti (U-Net +1.0/+2.4, GLU +1.5/+0.5 K) mentre sul TRAIN la U-Net scende a ~14 K (last): la VAL peggiora mentre il fit migliora. VAL123 è stata usata per scegliere il best: non è conferma indipendente.

## 3. Coperture 1/5/20% (best, VAL, media seed)
1%: U-Net 68.3 (p_on) / 80.0 (p_off); GLU 63.6 / 70.6. 20%: U-Net 37.7 / 48.6; GLU 42.4 / 43.4. La U-Net p_off peggiora da 5% a 20% (+8 K), con bias firmato +7.5 K; il GLU resta quasi piatto. 1% e 20% generalizzano la densità di TRAIN del 5% (nessun TRAIN multidensità): non sono una curva ottima del numero di sensori. Tabelle: `family_p_seed_dispersion.csv`, `coverage_degradation.csv`.

## 4. Snapshot-swap neurale (best, 5%, replica 0)
Stessi siti del destinatario, velocità del donatore a quei siti, coordinate/p/target del destinatario; pairing S6 ricostruito e confrontato con quello persistito; tag di famiglia/p/seed/epoca/hash dalla matrice; aligned riusati dalla valutazione verificata (stessi frame, stesso `mask_hash`) e rigiocati (scarto per-frame 0.0). Riferimento RMSE(T_donatore, T_destinatario): seen 153.9 K, VAL 97.4 K. Swapped, media seed: seen 148–150 K (tutti i modelli), VAL: U-Net 100.0 (p_on) / 101.4 (p_off), GLU 96.7 / 99.4; aligned VAL 39–45 K. In ogni condizione e per ogni run lo swap peggiora l'RMSE di almeno 3 K (seen) / 5.8 K (VAL); fino a 20% dei frame di una condizione VAL ha swapped < aligned solo per scostamenti per-frame, mai in media. Le predizioni dipendono dall'istantanea osservata, non solo dal layout; lo swap misura sensibilità all'istantanea, non causalità fisica. Tabelle in `residuals/tables/swap_neural_*.csv`.

## 5. Swap GappyPOD residuo S6 (contesto A38 congelato)
Oggetti A38 storici (artefatto SHA `a6fbdfd2…`, non rifittati), stesso pairing e osservazioni; aligned rigiocati = righe S6 (scarto 0.0). VAL: p_on 70.5 → 98.4 K, p_off 95.1 → 129.0 K (donatore 97.4); seen: p_on 108.4 → 130.8, p_off 113.2 → 153.6. Contesto separato dal fit C149.

## 6. Baseline C149 a pari dati
GlobalTrainMean, ParamConditionedMean, GappyPOD R=32 p_on/off fittati sull'intero pool C149 (2980 frame, 20×149), iperparametri **congelati** (λ p_off 1e-3, p_on 1e-2; ParamMean piecewise_phi1, α 0.01; w_T 5, seed 0; normalizzatore storico 760 usato solo per denormalizzare le velocità), nessun tuning su VAL, nessun refit A38. Controllo: lo stesso codice sul sottoinsieme A38 della stessa cache riproduce i 12 valori storici (scarto max 2.5e-8 K). VAL 5% C149: GlobalMean 139.6, ParamMean 87.7, GappyPOD p_off 93.8, p_on 70.6 K (A38 storico: 139.7, 88.0, 94.8, 70.6). L'effetto del pool sui baseline è piccolo (≤ 2.1 K, VAL ≤ 1.0 K). I neurali (VAL 5% ≈ 40–45 K) sono ben sotto tutti i baseline; GappyPOD p_on varia poco con la copertura. Tabelle `residuals/tables/baselines_*.csv`.

## 7. Diagnostico batch del GLU (separato dai risultati)
Frame TRAIN/VAL dello smoke, best GLU p_on e p_off e controllo U-Net (non modifica pesi né CSV). **Ripetizione identica:** differenza 0. **B=8 fisso, cambiano solo gli altri 7 esempi** (reali, riordinati, estremi con velocità ×10 e p +3): differenza 0.000 K per GLU e U-Net → nessuna dipendenza funzionale dagli altri esempi. **Forma del batch:** B=2,3,6 ≤ 2.0e-3 K; **B=1** differisce fino a 114 K su singoli pixel (media per frame 0.02–0.05 K, RMSE di frame ≤ 0.03 K) con cambi dell'insieme top-k su 2623 query per frame di prova; `cdist` a distanze dirette cambia i top-k su ~5000 query e l'RMSE di frame fino a 0.08 K. Le query con distanza k-esima/(k+1)-esima a meno di 1e-6 sono ~1067 per frame: **pareggi di distanza sulla griglia regolare** risolti diversamente dall'arrotondamento. Conclusione: sensibilità numerica dei pareggi top-k alla forma del calcolo, non errore funzionale; U-Net: ≤ 6e-4 K. Impatto: la valutazione usa batch 8 come il loop (parity bit-identico); scegliere un'altra forma del batch può spostare un frame fino a ~0.03–0.08 K. Gli scarti dello smoke (≤ 0.04 K) sono coerenti. Tabelle `batch_diagnostic_*.csv`.

## 8. Figure, costi, risultati sfavorevoli
Figure: `package/figures/*` (curve per copertura, punti 5%, differenze appaiate, VAL per condizione, centered vs bias con etichetta corretta «mean over frames of per-frame |bias|»), `fields/fields_val_frame1000_{best,last,best_seen_diagnostic}.png` (tre VAL, 12 run, scale comuni dichiarate: T [208, 2652] K, errore ±300 K clippato per la visualizzazione, massimo reale 720 K; campi coerenti con i CSV entro 2.1e-5 K), `residuals/figures/*`. Costi (sacct): training compute U-Net 3.8–4.2 h, GLU 5.5–7.8 h (due attempt per cap 5.5 h; l'attesa GLU include il limite %4 dell'array); valutazione 53–82 min per run, uguale per le due famiglie. Sfavorevoli: la U-Net p_off peggiora a 20% e a 1%; la U-Net ha bias VAL fortemente negativo; il GLU ha centered più alta e RMSE seen/reference760 peggiori; tutti i modelli peggiorano molto sotto swap; VAL non indipendente; glu-on-s1 ha richiesto estensione oltre la tranche.

## 9. Letture e decisione (attribuite)
- **Revisori Opus e Neon** (analisi di tabelle e figure): concordano che la U-Net sia complessivamente preferibile nella ricetta temperature-free studiata (accuratezza/struttura migliori e costo minore); il GLU conserva vantaggi nel livello e in alcuni casi fuori densità.
- **Decisione del coordinatore** (sviluppo post-risultati, **non** regola preregistrata né superiorità generale dell'architettura): U-Net p_on è il candidato operativo free; p_off resta nel confronto. Lo sweep K successivo includerà entrambe le famiglie e p_on/off; la matrice dei training con T sarà decisa dopo lo sweep, senza escludere definitivamente il GLU.
- Non sono stati applicati i criteri restrittivi proposti nelle review (range disgiunti, vittoria su ogni metrica, costo mai superiore): non approvati.

## 10. Limiti
Seed 0/1/2 solo per C149; VAL usata nella selezione; tre repliche di maschera non sono tre training; il seed nominale non rende uguali i pesi iniziali fra architetture; own_pool_train non calcolato (opzionale); baseline A38 restano riferimento storico.
