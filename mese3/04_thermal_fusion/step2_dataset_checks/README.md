# Step 2 — Reproducible Thermal Dataset Inspection Scripts

Questa cartella contiene gli script riproducibili, il manifest di provenienza dei campioni, il riepilogo JSON delle verifiche e le 4 figure diagnostiche prodotte nello **Step 2 (Agente B — Dataset sperimentali di temperatura)** della Missione 04.

**Nessun dataset pesante e nessun PDF integrale è committato in questa cartella.** I campioni temporanei (82 file, 49,6 MB totali su 16 dataset) possono essere riscaricati integralmente usando URL e hash SHA-256 registrati in [`samples_manifest.json`](samples_manifest.json).

## Contenuto

- [`samples_manifest.json`](samples_manifest.json): elenco puntuale di tutti gli 82 file ispezionati (ID dataset, nome file, URL sorgente, data UTC di download, stato HTTP, `Content-Type`, `ETag`/`Last-Modified`, dimensione in byte, `SHA-256`).
- [`inspect_all_samples.py`](inspect_all_samples.py): script di ispezione in sola lettura su tutti i 16 dataset (timestamp, lacune, duplicati, valori non finiti, codici sentinella `999.0 -> 0.0`, sensori bloccati, differenze grezzo/corretto, sovrapposizione termocoppia/Raman, replicati, discrepanze dichiarato vs effettivo).
- [`inspection_summary.json`](inspection_summary.json): output strutturato prodotto da `inspect_all_samples.py` (tutti i numeri citati in `../step2_dataset_termici.md` e `../step2_dataset_fonti_e_verifiche.md` derivano direttamente da questo file).
- [`generate_verification_figures.py`](generate_verification_figures.py): script che rigenera le 4 figure diagnostiche in `figures/`:
  - `figures/fig1_nist_rse2007_radiation_bias.png` — Bias radiativo di segno opposto fra termocoppia a giunzione nuda e aspirata co-localizzata (`NIST_RSE_2007` Test 01 e Test 12).
  - `figures/fig2_nist_nrc_and_katurji_faults.png` — Anomalie reali di misura: canale bloccato a 3320 °C e codice sentinella `999.0 -> 0.0` in `NIST/NRC Test 01`; mappa dei canali morti e dei dropout a 50 Hz in `Katurji EE1`.
  - `figures/fig3_nist_pool_inertia_and_rad.png` — Differenza media `TG - TC` e fattore di attenuazione inerziale `TG_RMS / TC_RMS` in funzione del diametro della giunzione (`39–199 µm`) in `NIST_Pool_Fires`.
  - `figures/fig4_hylon2_tc_vs_raman_and_walls.png` — Confronto spaziale diretto fra termocoppia a doppia giunzione e termometria Raman 1D (inclusa la regione esclusa nel nucleo di `Flame A.1` e la sovrastima in CRZ), più profili di temperatura di parete LIP (quarzo vs metallo) in `HYLON Phase 2` (TNF 2026-01).

## Riproduzione

1. Scaricare i file elencati in `samples_manifest.json` in una cartella locale `SAMPLES_DIR` mantenendo la struttura `<relative_path>` e verificando gli hash `sha256` (esempio rapido in Python):

```bash
python3 -c '
import hashlib, json, urllib.request
from pathlib import Path

mf = json.loads(Path("samples_manifest.json").read_text())
root = Path("/tmp/step2_thermal_samples")
for e in mf["files"]:
    dst = root / e["relative_path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(e["url"], headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=45).read()
    assert hashlib.sha256(data).hexdigest() == e["sha256"], f"SHA mismatch: {e[\"relative_path\"]}"
    dst.write_bytes(data)
print("Verified", len(mf["files"]), "files")
'
```

*(Nota: i 3 file PDF di verifica in `docs_verification/` servono solo per la verifica documentale e non sono letti da `inspect_all_samples.py`.)*

2. Rieseguire l'ispezione e la rigenerazione delle figure (richiede Python ≥ 3.10, `numpy`, `matplotlib`, e `scipy` per i file `.mat` di `CAM.94523` e `Katurji`):

```bash
python3 inspect_all_samples.py --samples-dir /tmp/step2_thermal_samples --out-json inspection_summary.json
python3 generate_verification_figures.py --samples-dir /tmp/step2_thermal_samples --out-dir ./figures
```
