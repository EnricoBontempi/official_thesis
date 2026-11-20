# Ledger Opus O0–O6

Stato tecnico per la revisione Neon. Fonti: `roadmapopus.md`, [opus_overnight_handoff.md](opus_overnight_handoff.md), `docs/RESEARCH_LOG.md`, test e artefatti nominati. «Verificato» significa che esiste un test o un artefatto citato; «aperto» significa che non l'ho trovato o non l'ho verificato in questo mandato.

| Passo | Voce | Stato | Evidenza o nota |
| --- | --- | --- | --- |
| O0 | YAML del protocollo sparse di nuovo parsabile (`authorized_s6_offline`) | corretto | RESEARCH_LOG 2026-11-19 wave-1 |
| O0 | Payload/hash delle maschere realmente consumati, niente rigenerazione silenziosa | corretto, verificato | `require_eval_mask_payload`; `tests/test_sparse_eval_masks.py`, `tests/test_opus_o5_mask_contract.py` (hash del file manifest separato dall'aggregato) |
| O0 | Gate Gappy rifiuta riferimento vuoto e richiede le sei chiavi | corretto | RESEARCH_LOG wave-1; verifica dei test non rieseguita in questo mandato |
| O0 | `eval_history.json` scritto anche su early-stop | corretto per i run futuri | handoff «Common corrections»; lo storico S5 non è stato alterato, R0 ha unito l'ultima eval dal CSV |
| O0 | Precisione esplicita (fp32, AMP, TF32, compile spenti) | verificato | `progress.json` S5 con `allow_tf32=false`; bridge LEDGER |
| O0 | Separare SHA di training, evaluator/launcher, checkpoint e CSV riusati | fatto per O5 | matrice con SHA di training; due SHA di launcher dichiarati; nessuno SHA storico inventato |
| O0 | Riconciliare ultima eval fra checkpoint, CSV e JSON | parziale | R0 PROVENANCE; non rifatto per ogni run S5 |
| O0 | Controllo figure loss/colonne, regioni e casi VAL di S6 | verificato, con limiti delle figure | `scripts/31_o0_s6_delivery_check.py`, `results/month2/opus_data_study/o0_checks/S6_DELIVERY_CHECK.md`: 20/20/3 condizioni, repliche 0/1/2, coperture 1/5/20%, 369 frame VAL per setting, colonne regionali complete e senza mancanti. Figure di loss su scala lineare e figure swap con barre colore per pannello: osservate, non rigenerate |
| O0 | Aggiornare i riepiloghi obsoleti in `roadmapsparse.md` e report | fatto per `roadmapsparse.md` e `roadmapopus.md` | stato corrente aggiunto, stato del 18 novembre conservato come storico; `roadmapastra.md` e i report 05 non toccati (Astra fuori mandato) |
| O1 | R0 sulle quattro history S5 | fatto | `results/month2/opus_data_study/r0_selection/`; non è una validazione indipendente; nessun peso intermedio |
| O2 | Manifest e cache C149; nesting; 760 frame condivisi bit-identici | fatto | handoff «C149 data path»; verify 29613544; `tests/test_opus_c149_cache_contract.py` |
| O2 | Unità di 95 update, stream, resume | fatto | `tests/test_scientific_epoch_sampler.py`, `tests/test_sparse_scientific_trainer.py` |
| O2 | Ponte A38 seed 0 contro S5 | fatto, con limite | `results/month2/opus_data_study/bridge_s5/LEDGER.md`: ordine dei frame diverso; replay di inferenza passato, equivalenza del training non testata; due riferimenti A38 seed 0 nel runner nuovo |
| O3 | Freeze, config, `best_seen_diagnostic` salvato | fatto | matrice `configs/experiments/opus_o5_matrix.json`, checkpoint verificati per hash |
| O3 | Launcher robusto, preflight leggero, directory non sovrascrivibili | fatto | `scripts/engaging/opus_o5_eval.sbatch`; preflight per run da `o5_preflight_paths.py` |
| O4 | Otto training conclusi (early-stop), nessun partial | fatto | handoff «outcome» e «verified outcomes»; `summary.completed` resta false (early-stop) |
| O5 | Valutazione finale degli otto run | fatto (otto run validi) | array 29728465 e sostitutivo 29732630_1; due SHA di launcher; `COMPLETE.json` degli otto run e controlli incrociati superati dal job 29771069 |
| O5 | Aggregazione, tabelle, confronto col loop, figure | fatto | `scripts/29_opus_o5_package.py`, `scripts/28_opus_o5_figures.py`; confronto col loop entro 0.05 K su 72 su 72 righe (scarto massimo 0.0011 K); figure principali controllate visivamente (elenco nell’indice) |
| O5 | Report `docs/month2/06_opus_data_and_training_variability.md` | **non fatto** | spetta alla revisione Neon |
| O6 | Decisione e passaggio ad Astra | **non fatto** | spetta al coordinatore; nessuna scelta S7 |

Obblighi tecnici ancora aperti: nessun obbligo bloccante. Restano dichiarati: le figure S6 non sono state rigenerate, i riepiloghi dei report e di `roadmapastra.md` non sono stati rivisti, e la riconciliazione completa dell'ultima eval dei quattro S5 non è stata rifatta oltre R0.

## Stato O5 verificato dal job finale

- Otto run con `COMPLETE.json` e controlli di completezza/provenienza superati (job finale, SHA `258106bc6e0e6dc7a721be2645b5ba66472acc8b`).
- SHA evaluator/launcher dichiarati: 918286809d3cbb33d155b70a6e03a9b898bcab9e, 29b94345f0cac3f725662ad687a15ff841c7c730.
- Confronto con il loop al 5% entro 0.05 K su tutte le righe: **True**.
- Figure dei campi VAL frame 1000: best: ok; best_seen_diagnostic: ok; last: ok.
- Report scientifico e decisione O6: **non fatti**, spettano a Neon e al coordinatore.
