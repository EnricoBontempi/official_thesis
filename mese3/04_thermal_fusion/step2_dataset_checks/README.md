# Step 2 — Reproducible Thermal Dataset Inspection Scripts

Questa cartella contiene gli script riproducibili, il manifest di provenienza dei campioni, il riepilogo JSON delle verifiche e le 4 figure diagnostiche prodotte nello **Step 2 (Agente B — Dataset sperimentali di temperatura)** della Missione 04 (revisione v2).

**Nessun dataset pesante e nessun PDF integrale è committato in questa cartella.** I campioni temporanei (**85 file, 49,6 MB totali su 16 dataset**) possono essere riscaricati integralmente usando URL e hash SHA-256 registrati in [`samples_manifest.json`](samples_manifest.json).

## Contenuto

- [`samples_manifest.json`](samples_manifest.json): elenco puntuale di tutti gli 85 file ispezionati (ID dataset, nome file, URL sorgente, data UTC di download, stato HTTP, `Content-Type`, `ETag`/`Last-Modified`, dimensione in byte, `SHA-256`).
- [`inspect_all_samples.py`](inspect_all_samples.py): script di ispezione in sola lettura su tutti i 16 dataset (timestamp, lacune, valori non finiti, pattern `999.0 -> 0.0`, anomalie di canale, confronto grezzo/corretto da modello, sovrapposizione TC–Raman con scarti di posizione/gradienti/finestra `x ± err_x`, confronto PIV `2S2Q`–`4Q` su 5 quote reagente+freddo, audit delle chiavi duplicate ULB, inventario completo `.mat` e verifica esplicita `shape/dtype/value` su `CAM.94523`).
- [`inspection_summary.json`](inspection_summary.json): output strutturato prodotto da `inspect_all_samples.py` (tutti i numeri calcolati citati in `../step2_dataset_termici.md` e `../step2_dataset_fonti_e_verifiche.md` derivano direttamente da questo file).
- [`generate_verification_figures.py`](generate_verification_figures.py): script che rigenera le 4 figure diagnostiche in `figures/`:
  - `figures/fig1_nist_rse2007_radiation_bias.png` — Differenza fra termocoppia a giunzione nuda e aspirata a identiche coordinate nominali `(X,Y,Z)` (`NIST_RSE_2007` Test 01 e Test 12).
  - `figures/fig2_nist_nrc_and_katurji_faults.png` — Anomalia non segnalata (`TC South U-3-1` a `3320–3330 °C`) e guasto documentato con pattern `999.0 -> 0.0` (`Tree 4-9`) in `NIST/NRC Test 01`; mappa di disponibilità (`finite` vs `NaN`) a 50 Hz in `Katurji EE1`.
  - `figures/fig3_nist_pool_inertia_and_rad.png` — Correzione da modello (`TG - TC`) e rapporto da modello (`TG_RMS / TC_RMS`) fra casi diversi (combustibili, diametri vasca e quote differenti), con evidenza dei 3 punti sovrapposti nella stessa fiamma `Methanol 30 cm` (`r=0, z=41–61 cm`, `52 µm` vs `150 µm`).
  - `figures/fig4_hylon2_tc_vs_raman_and_walls.png` — Confronto spaziale diretto fra doppia TC (`±10% T`, `±4 mm x`) e Raman 1D (`Tpg ± Tpg_rms`, banda di fluttuazione + precisione single-shot), più profili di temperatura di parete LIP (quarzo vs metallo) in `HYLON Phase 2` (TNF 2026-01).

## Dipendenze e riproduzione

- **Dipendenze Python:** Python `>= 3.10`, `numpy` (richiesto da `inspect_all_samples.py`), `scipy` (richiesto per i file `.mat` di `CAM.94523` e `Katurji 50 Hz`: se assente, lo script segnala `"error": "scipy not available"` nelle sole sezioni `.mat`), `matplotlib` (richiesto solo da `generate_verification_figures.py`). Nessuna dipendenza esterna è necessaria per leggere CSV, TXT, XLSX, ZIP o DOC.

1. Scaricare i file elencati in `samples_manifest.json` in una cartella locale `SAMPLES_DIR` mantenendo la struttura `<relative_path>` e verificando gli hash `sha256` (compatibile con Python 3.10+ e senza problemi di escaping bash):

```bash
python3 - <<'PYEOF'
import hashlib, json, urllib.request
from pathlib import Path

mf = json.loads(Path("samples_manifest.json").read_text(encoding="utf-8"))
root = Path("/tmp/step2_thermal_samples")
for entry in mf["files"]:
    rel = entry["relative_path"]
    dst = root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(entry["url"], headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=60).read()
    got = hashlib.sha256(data).hexdigest()
    assert got == entry["sha256"], f"SHA mismatch on {rel}: expected {entry['sha256']}, got {got}"
    dst.write_bytes(data)
print("Verified", len(mf["files"]), "files (", mf["total_bytes"], "bytes )")
PYEOF
```

*(Nota: i 3 file di verifica in `docs_verification/` servono solo per la verifica documentale e non sono letti da `inspect_all_samples.py`.)*

2. Rieseguire l'ispezione e la rigenerazione delle figure:

```bash
python3 inspect_all_samples.py --samples-dir /tmp/step2_thermal_samples --out-json inspection_summary.json
python3 generate_verification_figures.py --samples-dir /tmp/step2_thermal_samples --out-dir ./figures
```

## Separazione tra verifiche automatiche e verifiche documentali

| Categoria | Che cosa è verificato **automaticamente** da `inspect_all_samples.py` | Che cosa deriva da **verifica documentale / ispezione umana** |
|---|---|---|
| **HYLON Phase 2** | Matching TC–Raman (`dx`, interpolazione lineare, `dT/dx`, finestra `x ± err_x`, overlap delle barre), replicati TC, confronto PIV `2S2Q`–`4Q` su 5 quote (`Reactive` e `Cold`, incluso `z_col` e `n`), LIP pareti, estrazione intestazioni commento (incluso typo `m_a/m_h` nei CSV `L1` TC). | Geometria iniettore, equazione (4) doppia giunzione, procedura calibrazione Raman/LIP, avvertimento CRZ §5.7, restrizione d'uso NO-LIF (da `HYLON_TNF_2026_01_Phase2-1_PDF.pdf`). |
| **NIST RSE 2007 & NIST/NRC** | Regolarità $\Delta t$, conteggio `NaN`, differenze `BB - ASP` pre-ignizione e attive, traiettoria anomala `TD70LBB` Test 12, conteggio colonne `999.0 -> 0.0` e valori `>1500 °C` (`TC South U-3-1`). | Coordinate nominali `(X,Y,Z)` identiche delle 5 coppie porta e sfalsamento di `VD*` (da `RSE_01.fds`, `RSE_12.fds`); conferma documentale del guasto `Tree 4-9` e assenza di conferma per `TD70LBB` / `TC South U-3-1` (da FDS Validation Guide). |
| **NIST Pool Fires** | Schemi CSV (1 vs 2 intestazioni), `TG - TC`, `TG_RMS / TC_RMS`, confronto diretto sui 3 punti sovrapposti `Methanol 30 cm` (`52` vs `150 µm`), serie liquida Acetone, profilo $U(z)$ Metano. | Definizione di `TG`/`TG_RMS` come uscite del modello radiativo+inerziale NIST, emissività Pt vergine e incertezza `5,2%` (da `NIST_Pool_Fires/README.md` §4.1). |
| **Cambridge `CAM.94523`** | Inventario completo `(nome, shape, dtype)` di tutte le variabili nei 7 `.mat`; match parole chiave TC (`0` hit); verifica esplicita `shape=(1,1), dtype=uint8, value=0` vs tracce `float64` su tutti i `4 × 1000` colpi. | Inferenza finale che nessuna delle 61 variabili di `FigureData.mat` corrisponde a dati TC (ispezione di nomi/forme/range rispetto all'abstract DataCite); interpretazione dei `uint8(0)` come colpi rigettati (non essendoci uno script README nel deposito). |
| **ULB 20 kW & Wang MWIR** | Conteggio colonne, righe mancanti in `A`, molteplicità chiavi duplicate (`2211` coppie), bias sistematici `2nd - 1st`, span/semispan (`ULB`); differenze `MWIR - TC` e parsing `analysis_error_budget.csv` (`Wang`). | Contraddizione `°C` vs `K` e `T_std` con `README_V4.txt` e significato delle colonne omesse nel foglio `units` (`ULB`); definizione del modello di emissività e dell'incertezza `U = 4,1%` da `README.txt` (`Wang`). |
| **Altri dataset** | Byte LF/CR in `Waterloo`; celle vuote e integrali PDF in `FM Burner`; `NaN` vs `0.0` in `Brzosko`; `NaN` per canale e passi `|\Delta T|` in `Katurji`; `dt = -10 s` e intestazioni `SBG` in `Pelzmann`; match testuali `Mikron M90-D` / `no swirl` in `Surface_Temp_BB.doc`; outlier `<250 K` in `SM1.zip`; simmetrizzazione in `Smyth`. | Assenza di file digitali su `VT`, `PoliMi Solero 2017`, `TU Delft Sampat/van den Berg`; verifica `HTTP 403` (`Deep Blue`, `TUdatalib`) e TOC remoti (`NETL ESSB`, `4TU Link 2025`, `4TU Neeser`). |
