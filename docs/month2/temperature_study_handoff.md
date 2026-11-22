# Studio con temperatura (Step 2.7) — contratto comune e coordinamento Sonnet/Cursor

**Pubblicato il 21 novembre 2026, PRIMA della preparazione lunga.** Branch `temperature/assisted-c149` (derivato da `s7/c149-free-campaign` @ `2879c1b5`). Il contratto è congelato: Cursor e il percorso assisted usano gli stessi file, byte per byte.

## File del contratto
- `configs/experiments/month2_temperature_study_protocol.yaml` — protocollo comune.
- `configs/observations/month2_temperature_anchors_nested_v1.json` — manifest dei tre layout T (indici, coordinate, algoritmo, hash); generato da `scripts/50_t_write_anchor_manifest.py` con geometria C149 (sha nel manifest). Ordine/siti non sono mai stati scelti con T, errori o risultati.

## Cosa fa Cursor (solo questo)
Sweep delle correzioni (free / media_K / mediana_K) sui 12 checkpoint free S7 congelati e sui baseline C149, K={0,1,4,9,16,32}, stessi layout e dominio comune del manifest, griglia T9 storica solo come ponte separato. Non modifica checkpoint o risultati free, non tocca il manifest, non aspetta i training assisted.

## Cosa fa Sonnet
Implementazione T-input (U-Net e GLU), test, smoke, profiling, checkpoint/resume e 12 training assisted (famiglia × p × seed 0/1/2), un modello per (famiglia, p, seed) a K variabile, valutato su tutto lo sweep.

## Piano di assegnazione GPU (totale massimo 4 normal + 4 preemptable, inclusi smoke, sweep e altri job dell'utente)
| Chi | normal Advanced (L40S) | preemptable | Note |
|---|---:|---:|---|
| Cursor (inferenza sui checkpoint free, se serve GPU) | ≤ 1 | 0 | i job CPU (`mit_normal`, `mit_amf_advanced_cpu`) non contano |
| Sonnet (smoke, profiling, training assisted) | ≤ 3 | ≤ 4 | array separati non devono superare insieme i limiti |
Regola: ogni run assisted ha **una sola submission attiva** (nessun duplicato normal/preemptable); l'assegnazione run→pool sarà scritta qui al momento della submission. Preemptable: `mit_preemptable` con account `mit_general` (QoS normal; `mit_amf_advanced_gpu` su preemptable è rifiutato, non ritentato); solo con checkpoint/resume verificati. Prima di sottomettere, controllare `squeue -u bontempi` per i GPU già in uso.

## Stato
Contratto pubblicato. Preparazione, test, smoke e training assisted in corso (aggiornamenti in questo file).
