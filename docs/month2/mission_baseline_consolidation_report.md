# Mission report — Baseline consolidation (pre-neural)

**Stato finale:** **COMPLETATA** (revisione Opus accettabile dopo correzione metadati temporali)  
**Branch:** `mission/baseline-consolidation`  
**Commit codice eseguito (freeze M4):** `2c800cedd1cccc6935eb64c6af57fbdd6001aee8`  
**Consegna risultati replica:** `129f2982100437654da9dadf59588f2d86c59a35`  
**Base scientifica storica:** `822ed52c07eded37bba4f79af0f2177da4aa36eb`  
**Run replica:** `results/baseline_consolidation/consolidation_v1/` (non ricalcolata)  
**Nessun lavoro neurale avviato.**

## Checklist M0–M7

| Step | Esito |
|---|---|
| M0 ambiente/branch | OK — `main` ff a `f34ce8c`, branch dedicato, raw locali 30 Arrow, ~25 GiB RAM / 819 GiB liberi |
| M1 timestamp | OK — diagnosi raw + policy senza pad generico |
| M2 isolation | OK — battery su frame VAL reali; pytest |
| M3 contratto/output | OK — YAML enforced; output sotto `results/baseline_consolidation/` |
| M4 freeze | OK — commit `2c800ce` prima del full run; preflight + tolleranze fissate |
| M5 replica | OK — full CPU PASS |
| M6 confronto | OK — `identical_or_within_tolerance` (max abs metric = 0) |
| M7 consegna | questo report; PR da aprire; **STOP** |

## Correzioni (confermate)

1. **C3 / timeline:** SHA/size dello shard `data-00022-of-00030.arrow` coincidono con l’inventario remoto. `t` (2000) span 1.5→2.0 (audit); gap irregolare **interno** (diff×2 a i=1098); frame **2000 ≡ 1999** bitwise. Il pad generico “manca l’ultimo t” è **rifiutato**. Policy: eccezione identity-bound, `time_validated=False`, `t=NaN`, niente coordinate inventate. Dettaglio: `docs/month2/mission_baseline_timestamp_audit.md`.
2. **C2 / isolation:** tautologie sostituite da mutazioni sul frame **prima** del sample; assert su predizioni; controlli positivi/negativi; AO CRMSE; K=0 stesso modello T-block.
3. **C4 / aggregazione:** fingerprint esteso (`variant`, `R`, `λ`, `w_T`, `AO_tau`, …); NaN-safe; fail-fast (niente `except` che salta gruppi).
4. **C1 / YAML+manifest:** runner consuma il contratto o rifiuta; preflight con HEAD, status, hash sorgenti/config/stats, rule di confronto **prima** dei fit.
5. **C5 / output:** nuovi artefatti separati; per-frame gzip; report automatico distinto dall’interpretazione; storico `results/tables/` intatto.

## Diagnosi dati / limiti temporali

- Single-frame per indice resta valido per il confronto baseline.
- Uso temporale multi-frame / forecasting su `40NH3_1.2` **non autorizzato**.
- Stats fit-only **non** rigenerate.

### Correzione metadati post-replica (dopo revisione Opus su `129f298`)

Problema residuo: `time_validated=True` sugli indici `0…1999` perché esisteva `t[i]`, nonostante gap interno e packing ambiguo. **Corretto senza rerun.**

- Su **tutto** lo shard anomalo: `field_time_association_verified=False` e `time_validated=False`.
- Distinzione esplicita: presenza coordinata originale vs regolarità timeline vs associazione verificata vs `duplicate_of=1999` (solo i=2000).
- Flag propagati in `Month2ReconDataset.metadata` (non in `model_inputs`); disponibili per futuri CSV per-frame; artefatti `consolidation_v1` / storici **non** riscritti.
- Guardia `require_verified_field_time_association` rifiuta l’uso temporale non verificato.
- Il nuovo hash del lettore (`shard_access.py`) deriva da questa modifica di metadata; il freeze eseguito della replica resta `2c800ce`.

## Ambiente e comandi

```text
python: 3.13.13 (venv)
numpy / sklearn / pyarrow / torch: presenti
freeze HEAD: 2c800cedd1cccc6935eb64c6af57fbdd6001aee8
pytest: tests/test_month2_baselines.py (+ guards/pipeline/baselines/isolation) → 43 passed
smoke: scripts/06_run_month2_baselines.py --smoke --run-id smoke_m3
full:  scripts/06_run_month2_baselines.py --full --run-id consolidation_v1
compare: scripts/compare_month2_baseline_runs.py --new-dir results/baseline_consolidation/consolidation_v1
```

## Replica vs storico `822ed52`

- Conteggio per-frame: 66180 = 66180; common keys 66180.
- Metriche comuni: **max abs diff = 0** (atol 1e-6 K, rtol 1e-8).
- Search scores: max abs ~1e-14.
- Iperparametri selezionati: **identici** (free λ 0.001/0.01; T-block (0.001,0.5)/(0.01,0.5); ParamMean piecewise_phi1 α=0.01).
- Unica differenza di tracciabilità spiegata: storico LocalHGB senza tag `R`; replica marca `R=0` (nessun effetto numerico).
- Classificazione: `identical_or_within_tolerance`.

## Costi misurati (replica)

| Voce | Valore |
|---|---|
| Wall | ~1441 s (~24 min) |
| Peak RSS | ~10357 MiB |
| Shard decode | 23 |
| time_unvalidated_frames | 1 |
| known_t_mismatch_hits | 49 |

## Test

- Sintetici: timeline reject, adapter isolation, AO CRMSE.
- Reali: policy `40NH3_1.2` (skip se raw assenti).
- Runner isolation battery sulla replica: **passed**.
- Non svolti: cluster/GPU, TEST payloads, LFS pointer GitHub (usato SHA inventario locale verificato sul file).

## Questioni ancora aperte (non espansa qui)

Come da revisioni: gap Oracle↔stima; `w_T=0.5` al bordo; AO non monotono su seen; VAL non tipico high-NH3; p_off/on non è un ablazione causale di un solo modulo. **R=64 / nuove griglie / neurali non avviati.**

## File persistenti

- `results/baseline_consolidation/consolidation_v1/` (tabelle, manifest, comparison, auto_report)
- `results/baseline_consolidation/timestamp_diagnosis_40NH3_1.2.json`
- Storico `results/tables/month2_baseline_*.csv` **non modificato**
