# S7 — piano delle valutazioni finali C149 (21 novembre 2026)

Metriche del loop ≠ metriche della valutazione finale. Nessun vincitore, nessuna scelta di seed. Nessuno sweep K, nessun training con T, nessun TEST.

## Controlli conclusivi eseguiti prima della submission
- **glu-on-s1 esteso (job 29966737, SHA 501d41a):** chiuso per early-stop a epoca 1080 (15/15 eval senza miglioramento relativo 0.001), exit 0, 36 min. Storia 10→1080 contigua; le prime 100 eval sono identiche alla tranche 1 preservata (`tranche1_epoch1000/`, 12 file, SHA256 del manifest riverificati). `best.pt` identico alla tranche 1 (epoca 630, 41.501 K): nessuna riselezione. `last.pt` = epoca 1080; `best_seen_diagnostic` è passato da 960 a 1070 (tracker diagnostico predefinito, non selezione). Ledger: `extension_start`, `attempt_start action=extend`, `attempt_end rc=0`.
- **Parity al 5% degli undici run finiti (array 29968643):** 11/11 PARITY_OK, 18/18 confronti per run, scarto macro massimo **0.0** (bit-identico); scarto per-frame massimo **0.0** su train_fit 760, seen 220, VAL 123×3 maschere, tutti i checkpoint, entrambe le famiglie. Il gate scientifico resta macro RMSE 0.05 K.
- **Anomalia dello smoke (non sistematica):** nello smoke a sottoinsieme di frame il GLU VAL differiva dal loop fino a 0.025–0.040 K per frame. Confrontando smoke e parity completo sugli stessi frame: GLU VAL fino a 2.5e-2 K (media 2e-3), GLU train/seen ≤ 1e-5 K, U-Net 0 / ≤ 9e-6 K. Interpretazione supportata dai dati: il GLU su VAL è sensibile alla composizione dei batch (sottoinsieme = batch diversi); con gli stessi batch del loop il risultato è identico. Nessuna soglia nuova: è una nota di riproducibilità, non un criterio.

## Lanciato ora
Valutazione completa dei 12 run con `scripts/38_s7_eval.py --mode full` e matrice `configs/experiments/month2_s7_c149_eval_matrix.json`:
reference760 / seen220 / VAL123; coperture 1/5/20%; repliche 0/1/2; checkpoint best, last, best_seen_diagnostic distinti, con forward condiviso solo a pesi identici (glu-off-s0 e glu-on-s2: last = best_seen_diagnostic, SHA pesi uguali). 81 impostazioni per run (54 per i due con forward condiviso), readback di ogni CSV, gate 5% vs loop per run/checkpoint/ruolo/replica; `COMPLETE.json` solo se readback e gate passano, altrimenti `GATE_FAILED.json` con diagnostica. Output nuovi in `/home/bontempi/official_thesis_s7_c149_eval/<run_id>/`, mai sovrascrivibili (rifiuto se esiste). Nessun riuso di output O5: i due U-Net riusati sono rivalutati con lo stesso evaluator (checkpoint verificati per file/pesi SHA).

## Resta da eseguire (NON lanciato)
1. **Snapshot-swap** best@5% replica 0 su seen/VAL, stesso pairing/osservazioni/riferimento donatore, per i 12 run: funzioni S6 già family-aware (`evaluate_swap`, `build_within_condition_swap_pairs`); manca un wrapper S7 guidato dalla matrice con readback e provenance.
2. **Residuo GappyPOD p_on/off dello swap S6** (CPU, `mit_normal` + `mit_amf_advanced_cpu`; script `15_gappypod_nested_coverage_eval.py`, `submit_gappy_s6_repair.sh`): da verificare/completare, etichettando pool e oggetti fit.
3. **Baseline C149** con pool e ricetta di fit versionati (nessun refit A38).
4. **Aggregazione e pacchetto:** tabelle macro/per-condizione, differenze appaiate famiglia/p, seed, regionali con conteggi, costi, figure su tutte le VAL; specifica `configs/experiments/month2_s7_c149_protocol.yaml` con la regola di scelta del seguito fissata PRIMA di leggere gli esiti; report `docs/month2/10_s7_c149_model_comparison.md`.

## Risorse e stime (misurate)
- `mit_normal_gpu`, account/QoS `mit_amf_advanced_gpu`, 1×L40S, 8 CPU, 64G per task; massimo 4 GPU complessive.
- Costo forward misurato (parity, 11 run): U-Net e GLU costano uguale in valutazione, ~540–570 s per i 15 forward del parity, dominati da train_fit (~160 s per checkpoint, 760 frame, lato CPU). Estrapolazione a 81 impostazioni: ~84 min di forward + ~2 min di avvio per run (≈56–67 min per i due con forward condiviso). Walltime richiesto 3 h per task (margine ~1.8×). Memoria picco GPU misurata: U-Net 0.17 GB, GLU 0.51 GB (smoke).
- 12 run su 4 GPU: ~3 ondate, ≈ 4–5 h di orologio dopo la coda.
