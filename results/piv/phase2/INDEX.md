# Missione 02 — Fase 2 (F2.0–F2.6): indice degli artefatti

Report: [../../../mese3/02_virtual_piv/fase2_report_R2.md](../../../mese3/02_virtual_piv/fase2_report_R2.md) · Specifica congelata: [../../../configs/observations/piv_observation_spec_v1.yaml](../../../configs/observations/piv_observation_spec_v1.yaml) (+ `.sha256`) · Manifest dei download: [../qualification/download_manifest.tsv](../qualification/download_manifest.tsv)
Riproduzione: `bash scripts/piv_phase2/run_phase2.sh` · Checksum: [SHA256SUMS](SHA256SUMS) · I `*provenance.json` registrano SHA del codice, stato "dirty", versioni e file letti con sha256.

| Cartella | Contenuto | Script |
|---|---|---|
| `f20/` | F2.0: v_ref sull'intero record (`vref_full_record.csv`), registro dei kernel (`kernel_registry_v2.json/csv`, monotonicità), contenuto delle finestre B (`b_windows_content.json`), indice swapcells (`swapcells_index.csv`) | `f20_preliminaries.py` |
| `f26/` | F2.6: RMS per configurazione/condizione/regione, autocorrelazione dell'errore, distribuzione, distanza dal bordo, morfologia dei rigetti, conteggi; `figures/` (Figg. A–I), `tables/` | `f26_stats.py`, `f26_figures.py` |
| `f24/` | F2.4 Neal: statistiche dei residui, analisi di correlazione, `figures/` (N1, N2). I residui completi stanno fuori dalla repository | `f24_neal.py`, `f24_analyze.py` |
| `f25/` | F2.5 Cardiff: validazione del decoder e della maschera, conteggi per frame, transizioni dei codici, confronto delle regole di sostituzione, `figures/` (N3) | `f25_fetch.py`, `f25_analyze.py` |
| `tables/` | Tabelle Markdown incluse nel report | `f2_tables.py` |
| `../qualification/` | Manifest di provenienza dei download (Neal e Cardiff) | `f2_manifest.py` |

`f26/z_samples_PIV-A_B-in.npz` (campioni dell'errore standardizzato, 29 MB) non è versionato: si rigenera con `f26_stats.py`.
Dati scaricati e ambiente OpenPIV: fuori dalla repository, in `~/piv_qualification_data/`.
