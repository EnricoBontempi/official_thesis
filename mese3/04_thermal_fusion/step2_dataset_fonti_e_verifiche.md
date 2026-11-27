# Missione 04 — Step 2: fonti, registro delle ricerche e verifiche dei dataset termici

**Documento collegato a:** [step2_dataset_termici.md](step2_dataset_termici.md) (v2).  
**Artefatti riproducibili:** [step2_dataset_checks/README.md](step2_dataset_checks/README.md), [samples_manifest.json](step2_dataset_checks/samples_manifest.json), [inspect_all_samples.py](step2_dataset_checks/inspect_all_samples.py), [inspection_summary.json](step2_dataset_checks/inspection_summary.json), [generate_verification_figures.py](step2_dataset_checks/generate_verification_figures.py).  
**Versione:** v2 (revisione mirata a partire dal commit `6ae6795`, 27 novembre 2026) — **Branch:** `mission04/step2-thermal-datasets`.  
**Nota di integrità:** questo registro è integralmente autosufficiente (non dipende da appunti privati di subagenti). Nessun dataset pesante e nessun PDF integrale è stato committato nella repository; tutti gli **85 file ispezionati** sono riproducibili tramite URL e hash SHA-256 registrati qui e in [samples_manifest.json](step2_dataset_checks/samples_manifest.json) (verificati `85/85` su disco).

---

## 1. Metodologia di ricerca e canali interrogati

La ricerca è stata condotta il 26–27 novembre 2026 in due fasi integrate, seguite da una revisione mirata v2 sulle attribuzioni causali e sui controlli di confrontabilità (senza riaprire lo scouting né ampliare l'inventario dei 40 dataset):
1. **Esplorazione parallela ad ampio spettro** (archivi TNF/DLR/Sydney/Adelaide, archivi NIST/MaCFP/`firemodels`, repository universitari EU/UK/US, e misure dinamiche/calibrazione/deriva di termocoppie).
2. **Verifica diretta autonoma tramite API REST e probe HTTP** (12 round di interrogazione live + controllo intestazioni `HEAD`/`Range` + download selettivo di 85 file rappresentativi, `49,63 MB` totali in locale temporaneo, e analisi in sola lettura via Python). Nella revisione v2 sono stati inoltre consultati direttamente i file FDS primari `RSE_01.fds`, `RSE_12.fds` e i sorgenti LaTeX della FDS Validation Guide (`HGL_Chapter.tex`, `Surface_Temperature_Chapter.tex`, `Experiment_Chapter.tex`) per verificare le coordinate nominali delle coppie porta in `NIST_RSE_2007` e distinguere guasti documentati da anomalie osservate, e sono stati scaricati i 3 file radiali `Methanol_30_cm_TC_z={41,51,61}_cm_Sung_2024.csv` (`990 B` totali, stesso dataset `NIST_Pool_Fires`) per verificare l'unico punto di sovrapposizione fra due diametri di giunzione nella stessa fiamma.

### 1.1 Canali e API interrogati

| Canale / Repository | Metodo di interrogazione | Ambito / query principali | Esito operativo |
|---|---|---|---|
| **TNF Workshop** (`tnfworkshop.org`) | Scraping pagine `/data-archives/*`, `/pilotedjet/`, `/swirlflames/`, `/hydrogen-flames/`, `HEAD` sugli ZIP | HYLON Phase 1/2/3, Sandia C–F, KAUST Cracked NH$_3$, Delft III, DLR A/B, AHEAD, NTNU | **HTTP 200** sugli ZIP diretti; ispezionati `HYLON_TNF_2026_01_Phase2-1.zip` e `pmCDEF.zip` |
| **Sydney Thermofluids** (`web.aeromech.usyd.edu.au/thermofluids/`) | Verifica pagina `/database.php`, `/bluff_files/`, `/swirl_files/` | Bluff body `HM1–3` (`Surface_Temp_BB.doc`), swirl `SM1.zip` | **HTTP 200** live; scaricati e ispezionati `Surface_Temp_BB.doc` e `SM1.zip` |
| **MaCFP Database** (`github.com/MaCFP/macfp-db`) | GitHub API + `raw.githubusercontent.com` | `NIST_Pool_Fires`, `Waterloo_Methanol`, `FM_Burner` (ott. 2025), `UMD_Line_Burner` | **HTTP 200**; scaricati e ispezionati **31** file + README |
| **FDS Experimental Archive & Validation Guide** (`github.com/firemodels/exp` & `firemodels/fds`) | GitHub API + `raw.githubusercontent.com` (`RSE_01.fds`, `RSE_12.fds`, `HGL_Chapter.tex`, `Surface_Temperature_Chapter.tex`, `Experiment_Chapter.tex`) | `NIST_RSE_2007`, `NIST_NRC`, `Smyth_Slot_Burner`, coordinate nominali e documentazione guasti | **HTTP 200**; scaricati e ispezionati 10 file dati + verificati input FDS e capitoli Validation Guide |
| **Zenodo REST API** (`zenodo.org/api/records`) | Query mirate (`swirl thermocouple`, `combustor temperature`, `thermocouple drift`, `pyrometer`, `MWIR thermocouple`, `suction pyrometer`, `ULB furnace`) | `7624702`, `7624630`, `23054326`, `8189490`, `8186928`, `4972091`, `21371063`, `23062725`, `23037088`, `19004211`, `3693670`, `20446928` | **HTTP 200**; scaricati e ispezionati 5 record completi/campioni |
| **DataCite REST API** (`api.datacite.org/dois`) | Interrogazione su 9 prefissi istituzionali (`10.4121` 4TU: 194 hit; `10.48328` TUdatalib: 34 hit; `10.7302` Deep Blue: 12 hit; `10.17035` Cardiff: 77 hit; `10.17863` Cambridge: 29 hit; `10.5445` KIT: 25 hit; `10.7294` VT: 3 hit; `10.25909` Adelaide: 1 hit; `10.3929` ETH: 0 hit; `10.14469` Imperial: 1 hit) | Verifica sistematica dell'esistenza di DOI dati sui rig dello Step 1 | **HTTP 200** sui metadati DataCite; confermati sia i record esistenti bloccati (`403`) sia le assenze nei repository interrogati |
| **Cardiff Research Data / Figshare API** (`api.figshare.com/v2`) | Query articoli & file su `10.17035/*` e `10.6084/m9.figshare.16615174` | `cardiff.31141438`, `cardiff.30234904`, `d.2022.0217184646`, `d.2022.0214909050`, `Katurji 16615174.v2` | **HTTP 200**; scaricati e ispezionati 5 file (1 positivo Cardiff, 2 controlli negativi Cardiff, 2 `.mat` Katurji) |
| **4TU.ResearchData API v2** (`data.4tu.nl/v2/articles`) | Metadati + download README + HTTP `Range` TOC su ZIP | `Link 2025` (`7e680c48...`, `133 MB`), `Neeser 2025` (`dbaa8f0f...`, `2,05 GB`), `Navarro` (`19355384`) | **HTTP 200**; `REAME.pdf` + TOC `Link 2025` (negativo T) e `README.md` `Neeser 2025` (TC parete 100 Hz) |
| **Cambridge Apollo** (`repository.cam.ac.uk`) | Handle pages + bitstream UUID direct download | `CAM.94523` (LIGS swirl), `1810/226470` (Stratified Slot), `CAM.96660`, `CAM.90803`, `CAM.106960` | **HTTP 200** su bitstream diretti (`API REST /rest` 404); scaricati tutti i 7 `.mat` di `CAM.94523` |
| **OSTI.gov API + NETL EDX CKAN API** (`edx.netl.doe.gov/api/3/action/package_show`) | Verifica `10.2172/1463254` (VT), `10.18141/2478453` (NETL ESSB), `2552943` (NETL NH$_3$) + HTTP `Range` ZIP TOC | `NETL_ESSB_Data_Release_1.0.zip` (`96,9 MB`, 115 file), `2025_FlameSpeed_FTIRSpecies.zip` (`19,0 MB`) | **HTTP 200**; verificato TOC completo `NETL ESSB`; VT confermato solo report PDF |
| **Mendeley Data API** (`data.mendeley.com/api/datasets`) | Verifica `jvvvgpdy48.6` (Pelzmann) e `jtmk3pg36k.1` (Fischer FWI) | File CSV/TXT McKenna e BC parete FWI | **HTTP 200**; scaricati e ispezionati 3 file di `jvvvgpdy48.6` |
| **PoliMi IRIS** (`re.public.polimi.it`) | Ricerca autore (`Solero`, `Olivani`) + verifica handle `11311/1152656` e `11311/552372` | Verifica presenza dataset digitali e PDF completi | **HTTP 200** su `IJETR2102.pdf` (scaricato temporaneamente e letto integralmente); `etfsPAPER2007.pdf` bloccato da login; **zero dataset digitali** |
| **Dryad API** (`datadryad.org/api/v2`) & **NIST PDR** (`data.nist.gov`) | Verifica `10.5061/dryad.4cj4k`, `10.18434/mds2-2314`, `M32135`, `T4/1502568`, `mds2-2812` | Metadati e disponibilità file | **HTTP 200** metadati; Dryad `302 MB` non scaricato; NIST FCD `mds2-2314` richiede portale interattivo |
| **ISF Workshop** (`isfworkshop.org` & `adelaide.edu.au/cet/isfworkshop`) | Probe HTTP diretto delle pagine dataset | ISF-3 DLR pressurizzato e archivi soot Adelaide | **Link live rotti** (redirect generico Adelaide University post-fusione; `isfworkshop.org` punta a snapshot Wayback Machine del 2 giugno 2025) |

---

## 2. Esito puntuale delle piste iniziali (Step 1 §14.2) e delle piste aggiuntive

| Pista iniziale (Step 1 §14.2 / incarico) | Esito della verifica | Classificazione accesso | Note documentate |
|---|---|---|---|
| **Virginia Tech Optical Swirl Combustor** (`OSTI 1463254`, DOE-VT-001176-2) | Verificati API OSTI e prefisso DataCite `10.7294` (`data.lib.vt.edu`: solo 3 record combustion, nessuno del rig DE-FE0011762) | **`non individuato` / `su richiesta`** | Esiste solo il rapporto finale PDF (264 pp.); i 320 nodi TC tipo B e le mappe IR non risultano archiviati pubblicamente nei repository interrogati. |
| **TU Delft (Sampat *Fuel* 2025/GT2022; van den Berg *AECS* 2024; X. Huang 2018; DJHC)** | Interrogati 4TU.ResearchData (`10.4121`, 194 record combustion) e TU Delft Repository | **`non individuato` / `su richiesta`** | Nessuno di questi rig ha un dataset pubblico su 4TU. Individuati su 4TU: `Link 2025` (swirl CH$_4$/H$_2$, **zero dati T** verificato via TOC+README) e `Neeser 2025` (accenditore GOx/CH$_4$, **TC parete 100 Hz** + `sensors.py`, `2,05 GB`). |
| **Archivi TNF: HYLON, Sandia D–F, Delft III, DLR A–B, KAUST NH$_3$** | Verificati link diretti su `tnfworkshop.org` | **`accessibile`** (`HYLON Phase 2` e `Sandia pmCDEF` **ispezionati**; altri HTTP 200) | **`HYLON Phase 2` (gennaio 2026)** è il ritrovamento principale per combustori swirl confinati: doppia TC + Raman 1D + LIP pareti + PIV (`4Q`/`2S2Q`) + CAD. |
| **TNF/Darmstadt TSF & SWQ, CSSB scalari/fosfori, TECFLAM, Lund LSB** | Verificate pagine TNF, TUdatalib e Cambridge Apollo | **`su richiesta` / `non individuato`** | Su Cambridge Apollo esiste `1810/243259` (CSSB **sola LDA**, senza T) e `1810/226470` (Stratified Slot Raman/Rayleigh, *All Rights Reserved*). TSF/SWQ e fosfori bluff body non scaricabili apertamente. |
| **DLR GTMC / PRECCINSTA / SFB 606 / HBK / FRS** | Verificati elib DLR, TNF e DataCite | **`su richiesta`** | I PDF descrittivi sono aperti su `elib.dlr.de`, ma i file numerici (Raman, CARS 5 kHz, fosfori di parete) sono forniti tramite richiesta/accordo con DLR. |
| **ISF-3 (DLR pressurizzato) & Adelaide JHC** | Probe HTTP su `isfworkshop.org`, `adelaide.edu.au/cet/isfworkshop/` e prefisso DataCite `10.25909` (1 record) | **`su richiesta` / link rotti** | Le pagine CET di Adelaide reindirizzano alla home generica post-fusione; `isfworkshop.org` rinvia a un backup Wayback Machine del 2025. Nessun dataset numerico ISF-3 o Adelaide JHC scaricabile direttamente. |
| **Cardiff e altri rig ad ammoniaca** | Interrogati prefisso `10.17035` (77 hit), Figshare API, Zenodo, NETL EDX, TNF KAUST | **`accessibile`** (misto positivo / negativo) | **Positivo ispezionato:** `cardiff.31141438` (Sato 2026: TC scarico + parete int/est su 9 $\Phi$). **Negativi verificati aprendo i file:** `cardiff.30234904` e `d.2022.0217184646` (zero T). Altri NH$_3$ aperti: `TNF KAUST Cracked NH3`, `ULB 7624630`, `NETL 2552943`, `Zenodo 19004211`. |
| **PoliMi (Solero / Olivani / Coghe)** | Interrogata `re.public.polimi.it` (IRIS), scaricato `IJETR2102.pdf` (`11311/1152656`), verificato `11311/552372` | **`non individuato` (solo PDF / login)** | **Verificato su testo integrale (`Solero 2017`):** TC tipo R `0,3 mm` (500 Hz, 5000 campioni), corr. radiativa Morgan (1991) con velocità PIV/LDA — **sole figure in PDF, zero dataset digitale**. `Olivani 2007` PDF protetto da login. |
| **UMich Deep Blue TCC-III** | Interrogata DataCite (`10.7302`, 12 record) + probe HTTP su `deepblue.lib.umich.edu` | **`bloccato dall'ambiente` (`HTTP 403`)** | Confermati via DataCite `z2hq3x4s` (Near-Wall Flow: PIV + TC coassiale rapida di flusso termico), `z2ms3qp3`, `z2h41pcd`, `z2cc0xnp`. Download bloccato da HTTP 403 (mai aggirato). |
| **TUdatalib Darmstadt** | Interrogata DataCite (`10.48328`, 34 record) + probe HTTP su `tudatalib.ulb.tu-darmstadt.de` | **`bloccato dall'ambiente` (`HTTP 403` Anubis)** | Confermati via DataCite `tudatalib-2291` (pistone fosfori H$_2$, 2026, CC BY 4.0), `1819` (Raman NH$_3$/H$_2$/N$_2$), `2135` (Raman H$_2$), `2194` (TDLAS), `1535/1532` (motori ottici), `1113/1114` (flow bench). Bloccati da Anubis HTTP 403 (mai aggirata). |
| **Sydney Thermofluids (Bluff Body & Swirl)** | Probe HTTP diretto su `web.aeromech.usyd.edu.au/thermofluids/` | **`accessibile (ispezionato)`** | `SM1.zip` (`133.389` colpi Raman/Rayleigh/LIF, `9` colpi `<250 K`) e `Surface_Temp_BB.doc` (**pirometro IR Mikron M90-D**, non TC, su hardware scambiato). |
| **EngineBench (`arXiv:2406.03325`) & OSU McManus (FRS+PIV)** | Verifica metadati/paper e repository pubblici | **`controllo negativo`** (EngineBench) / **`non individuato`** (McManus) | `EngineBench` contiene solo PIV motored a freddo (nessuna combustione né T). I dati McManus & Sutton non risultano depositati negli archivi aperti interrogati. |
| **Calibrazioni sperimentali e misure dinamiche TC (pista ampia)** | Ricerca ampia su Zenodo, Figshare, MaCFP, `firemodels/exp`, Mendeley, 4TU | **Misto (`accessibile` per bias/dropout/deriva; `non individuato` per step-response kHz)** | **Trovati e ispezionati:** `NIST_RSE_2007`, `NIST/NRC`, `NIST_Pool_Fires`, `Katurji 50 Hz`, `MetForTC` (`8189490`), `Pearce` (`4972091`), `Waterloo`, `ULB`, `Wang MWIR`, `Brzosko`. **Non individuati nei canali interrogati:** serie raw kHz a doppia TC in fiamma e prove raw di risposta a gradino/impulso laser di TC. |

---

## 3. Registro delle esclusioni, controlli negativi, accessi bloccati e separazione automatico vs documentale

### 3.1 Controlli negativi verificati aprendo i file o i TOC
- **`TU Delft Swirl Burner` (`10.4121/7e680c48-7408-4e35-8880-10220814afd5.v1`, Link et al. 2025):** scaricato `REAME.pdf` (5 pp.) e letto l'indice ZIP remoto (`95` file, `133 MB`) — contiene esclusivamente PIV, SPOD, OH-PLIF ed emissioni NO/CO$_2$; **zero dati termici**.
- **`Cardiff COG/NH3` (`10.17035/cardiff.30234904`) & `Mashruk 2022` (`10.17035/d.2022.0217184646`):** aperti e parsati via `inspect_all_samples.py`; **zero colonne o fogli di temperatura**.
- **`Pelzmann & Robert McKenna` (`10.17632/jvvvgpdy48.6`):** controllo automatico delle intestazioni conferma `SBG` = sensore di flusso termico Schmidt-Boelter (`U [mV], Q [W], Q [kW]`), non TC gas.
- **`Cambridge CAM.94523` (`10.17863/CAM.94523`) — assenza parziale:** valido per LIGS ottica (`4 × 1000` colpi verificati), ma `0` match automatici di parole chiave TC su tutte le variabili dei 7 `.mat` e nessuna serie TC individuata all'esame dell'inventario completo.
- **`EngineBench`, `Cambridge CCB` (`CAM.96660`), `LEAFINNOX` (`CAM.90803`), `Yoko` (`CAM.106960`), `CSSB LDA` (`1810/243259`), `TU Berlin AHEAD`, `NTNU Bluff-Body H2`:** esclusi per assenza di misure di temperatura nei metadati/file depositati.

### 3.2 Accessi bloccati dall'ambiente, protetti da login o link rotti (mai aggirati)
- **`UMich Deep Blue Data`:** `HTTP 403 Forbidden` sui download diretti per `10.7302/z2hq3x4s`, `z2ms3qp3`, `z2h41pcd`, `z2cc0xnp`. Metadati verificati solo via DataCite.
- **`TU Darmstadt TUdatalib`:** `HTTP 403 Forbidden` (sfida anti-bot Anubis) per `tudatalib-2291`, `1535`, `1532`, `1113`, `1114`, `1819`, `2135`, `2194`. Metadati verificati solo via DataCite.
- **`PoliMi IRIS` — `Olivani et al. 2007` (`11311/552372`):** bitstream PDF protetto da login istituzionale (`Solero 2017` `11311/1152656` invece pubblico `HTTP 200`, letto integralmente).
- **`ISF Workshop` / `Adelaide CET`:** link live storici rotti post-fusione universitaria; `isfworkshop.org` rinvia a snapshot Wayback Machine (`20250602034109`).
- **`Cambridge Apollo REST v6` (`/rest/items/...`):** `HTTP 404`; accesso regolare effettuato tramite i collegamenti pubblici `/bitstreams/<uuid>/download` (`HTTP 200`).
- **`NIST Fire Calorimetry Database` (`10.18434/mds2-2314`):** API PDR riporta `0` file diretti (portale web dinamico).

### 3.3 Esclusioni per dimensione / formato / priorità (archivi aperti non scaricati integralmente)
- **`NETL ESSB Rel. 1.0` (`96,9 MB`)** e **`4TU Neeser GOx/CH4` (`2,05 GB`):** verificati via indice ZIP remoto (`HTTP Range`) e `README.md` scaricato.
- **`Dryad ETH Bonciolini` (`302 MB`)**, **`Zenodo 20446928` (`668 MB`)**, **`TNF HYLON Phase 1` (`35,2 MB`)**, **`TNF KAUST Cracked NH3` (`24,9 MB`)**, **`TU Berlin AHEAD` (`3,47 GB`)**: verificati via API/`HEAD`, non scaricati per dimensione/priorità.
- **`Pearce Drift Survey` (`10.5281/zenodo.4972091`):** scaricati tutti i 9 `.opju` (`3,1 MB`) + mappa figure, ma il formato binario proprietario OriginPro `.opju` non è leggibile nativamente in Python.

### 3.4 Separazione esplicita fra verifiche automatiche e verifiche documentali
L'esatta attribuzione di ogni risultato a controllo automatico (`inspect_all_samples.py`) vs verifica documentale/ispettiva è dettagliata nella tabella dedicata in [step2_dataset_checks/README.md](step2_dataset_checks/README.md):
- **Automatiche (script):** regolarità temporale e salti (`dt`), conteggi `NaN`/celle vuote/sentinelle `999.0 -> 0.0` e `(1,1) uint8 == 0`, inventario `(nome, shape, dtype)` dei `.mat` e match keyword, differenze `BB - ASP`, traiettoria `TD70LBB` Test 12, confronto TC–Raman (`dx`, interpolazione, `dT/dx`, finestra `x ± err_x`, overlap barre), confronto PIV `2S2Q–4Q` su 5 quote (`Reactive`+`Cold`), confronti `NIST_Pool_Fires` inclusi i 3 punti sovrapposti `Methanol 30 cm`, audit chiavi duplicate `ULB`, parsing `analysis_error_budget.csv` (`Wang`), byte LF/CR (`Waterloo`), simmetria (`Smyth`), match testuali in `Surface_Temp_BB.doc` e `Pelzmann`.
- **Documentali / ispettive:** geometrie e procedure nei PDF (`HYLON`, `Solero 2017`, `Link 2025`, `Neeser 2025`), coordinate nominali `(X,Y,Z)` da `RSE_01.fds`/`RSE_12.fds`, distinzione fra guasto confermato (`Tree 4-9` in `HGL_Chapter.tex`) e anomalie non documentate (`TD70LBB`, `TC South U-3-1`), natura modellistica di `TG`/`TG_RMS` (`NIST_Pool_Fires/README.md`), inferenza finale di assenza TC in `CAM.94523`, interpretazione del budget di errore in `Wang` e delle colonne omesse in `ULB`.

---

## 4. Provenienza, integrità e riproduzione dei campioni ispezionati

Tutti gli **85 file** scaricati temporaneamente in locale (totale `49.627.572` byte = `49,63 MB`, tutti `HTTP 200`, **85/85 verificati per dimensione e hash SHA-256** su disco prima della rigenerazione v2) sono inventariati integralmente in [samples_manifest.json](step2_dataset_checks/samples_manifest.json).

### 4.1 Riepilogo per dataset e metodo di ispezione

| `dataset_id` (manifest) | File | Byte totali | Sorgente / versione | Metodo di ispezione ([inspect_all_samples.py](step2_dataset_checks/inspect_all_samples.py)) |
|---|---:|---:|---|---|
| `tnf_hylon_phase2` | 1 ZIP (`73` file reali) | `9.273.876` | `tnfworkshop.org` (`2026-01`, Last-Mod `2026-01-24`) | Lettura in memoria da ZIP; check intestazioni commento (typo `L1` TC); overlap TC–Raman su tutti i punti in `[rx_min, rx_max]` (`dx`, `T_lin`, `dT/dx`, `x±err_x`); PIV `2S2Q` vs `4Q` su 5 quote `Reactive`+`Cold`; LIP pareti; lettura PDF 24 pp. |
| `cam_94523_ligs_tc` | 7 `.mat` | `11.475.530` | `repository.cam.ac.uk` (`10.17863/CAM.94523`, `2023-03-03`) | `scipy.io.loadmat`; inventario completo `(nome, shape, dtype)` su tutti i 7 `.mat`; match keyword TC; verifica esplicita `shape=(1,1), dtype=uint8, value=0` su tutti i `4 × 1000` colpi |
| `tnf_sandia_def` | 1 ZIP (`144` file) | `8.053.410` | `tnfworkshop.org` (Rel. 2.0, Last-Mod `2019-02-03`) | Lettura in memoria di `D15.Yall` (`9.339` colpi) e `D15.Yave` |
| `sydney_thermofluids` | 2 (`SM1.zip`, `.doc`) | `4.302.915` | `web.aeromech.usyd.edu.au` (Last-Mod `2013-10-01`) | Parsing 9 file `.YAL` (`133.389` colpi, check `<250 K` e `TDNR`); estrazione testo e match automatici da `Surface_Temp_BB.doc` |
| `brzosko_tc_grid` | 2 (`.npz`, `.py`) | `4.031.563` | `zenodo.org/records/21371063` (2026-07-15) | `numpy.load`; verifica shape `(211,532,12)`, `mask`, `NaN` vs `0.0` |
| `wang_mwir_tc` | 1 ZIP | `3.735.615` | `zenodo.org/records/23054326` (v. R1-02) | Lettura in memoria `campaign2_station_data.csv`, `analysis_error_budget.csv`, `analysis_tc_radiation_error.csv` e `README.txt` |
| `pearce_drift_survey` | 10 (`9 .opju`, `1 .txt`) | `3.089.342` | `zenodo.org/records/4972091` (2021-06-17) | Verifica integrità archivio e mappa figure; formato proprietario `.opju` |
| `katurji_50hz_tc` | 2 `.mat` | `1.298.824` | `ndownloader.figshare.com` (`16615174.v2`) | `scipy.io.loadmat`; mappa `NaN` per canale, run contigui, distribuzione passi `|\Delta T|` a `20 ms` |
| `nist_nrc` | 3 `.csv` | `1.028.143` | `firemodels/exp` (`master`) | Parser CSV 1-intestazione; check $\Delta t=10\text{ s}$, pattern `999.0 -> 0.0`, soglia `>1500 °C` (`TC South U-3-1`), `NaN` |
| `metfortc_drift` | 1 `.xlsx` | `907.052` | `zenodo.org/records/8189490` (2023-07-27) | Parser XML sui 4 fogli raw Keithley (`2021 K`, `2022 K`, `2022 BRML S`, `2023 S`) |
| `docs_verification` | 3 (`2 PDF`, `1 MD`) | `883.876` | `data.4tu.nl` (`Link`, `Neeser`), `re.public.polimi.it` (`Solero 2017`) | Verifica documentale testuale; non letti dallo script numerico |
| `cardiff_nh3_swirl` | 3 (`2 .xlsx`, `1 .zip`) | `627.262` | `ndownloader.figshare.com` (`31141438`, `30234904`, `27277428`) | Parser XML su fogli Excel; check automatico intestazioni T e RMS/incertezza |
| `nist_rse_2007` | 3 `.csv` | `276.747` | `firemodels/exp` (`master`) | Parser CSV 2-intestazioni; check $\Delta t=30\text{ s}$, `NaN`, 5 coppie `*BB` vs `*A`, traiettoria anomala `TD70LBB` Test 12, 9 `VD*` |
| `ulb_furnace` | 3 (`1 .xlsx`, `2 .txt`) | `185.712` | `zenodo.org/records/7624702` & `7624630` | Parser XML su `data` e `units`; audit `545` righe senza `A` e `2.211` coppie duplicate (shift sistematici, span vs semispan) |
| `fm_burner` | 9 (`8 .csv`, `1 .py`) | `149.946` | `MaCFP/macfp-db` (update ott. 2025) | Celle vuote (`9/693`), integrazione numerica PDF 200 bin, velocità 2C |
| `pelzmann_mckenna` | 3 (`2 .csv`, `1 .txt`) | `73.157` | `data.mendeley.com` (`jvvvgpdy48.6`) | Check automatico intestazioni `SBG` e monotonia timestamp `TPH_log.txt` (`dt = -10 s`) |
| `smyth_slot_burner` | 4 `.txt` | `42.326` | `firemodels/exp` (`master`) | Asimmetria destra-sinistra e confronto diretto file simmetrizzato vs originale; check LDV |
| `nist_pool_fires` | **13 `.csv`** *(+3 in v2)* | `23.258` | `MaCFP/macfp-db` (`Sung_2021` & `Sung_2024`) | Parser dual-schema; comparazione fra casi + confronto sui 3 punti sovrapposti `Methanol 30 cm` (`52` vs `150 µm`); serie liquida `Acetone`; velocità `Methane` |
| `waterloo_methanol` | 10 `.csv` | `23.153` | `MaCFP/macfp-db` (`master`) | Verifica byte LF/CR (`0 LF/0 CR` sui compositi); parsing slice `z=10 cm` |
| `umd_line_burner` | 4 (`3 .csv`, `1 .md`) | `7.658` | `MaCFP/macfp-db` (`master`) | Parsing `TCz`, `TCx`, `TC_Data` e confronto unità con `README.md` |

### 4.2 File aggiunti in v2 (e selezione dei file primari; elenco integrale di tutti gli 85 file in [samples_manifest.json](step2_dataset_checks/samples_manifest.json))

| Dataset | File | Byte | SHA-256 (primi 16 caratteri) | URL sorgente |
|---|---|---:|---|---|
| `nist_pool_fires` *(v2)* | `Methanol_30_cm_TC_z=41_cm_Sung_2024.csv` | 331 | `b8f5f46e18c02400` | `https://raw.githubusercontent.com/MaCFP/macfp-db/master/Liquid_Pool_Fires/NIST_Pool_Fires/Experimental_Data/Methanol_30_cm_TC_z=41_cm_Sung_2024.csv` |
| `nist_pool_fires` *(v2)* | `Methanol_30_cm_TC_z=51_cm_Sung_2024.csv` | 331 | `90c7eb7c466c5cfe` | `https://raw.githubusercontent.com/MaCFP/macfp-db/master/Liquid_Pool_Fires/NIST_Pool_Fires/Experimental_Data/Methanol_30_cm_TC_z=51_cm_Sung_2024.csv` |
| `nist_pool_fires` *(v2)* | `Methanol_30_cm_TC_z=61_cm_Sung_2024.csv` | 328 | `dfc238c0041ca7dc` | `https://raw.githubusercontent.com/MaCFP/macfp-db/master/Liquid_Pool_Fires/NIST_Pool_Fires/Experimental_Data/Methanol_30_cm_TC_z=61_cm_Sung_2024.csv` |
| `tnf_hylon_phase2` | `HYLON_TNF_2026_01_Phase2-1.zip` | 9.273.876 | `4dc63f18651f6145` | `https://tnfworkshop.org/wp-content/uploads/2026/01/HYLON_TNF_2026_01_Phase2-1.zip` |
| `nist_rse_2007` | `NIST_RSE_2007_{01,02,12}.csv` | 276.747 | `52347762` / `0a69af10` / `152f8028` | `https://raw.githubusercontent.com/firemodels/exp/master/NIST_RSE_2007/...` |
| `nist_nrc` | `NIST_NRC_Test_01_file_{1,2}.csv`, `Test_04_file_1.csv` | 1.028.143 | `d97758dc` / `eaf78bd8` / `96ccad1b` | `https://raw.githubusercontent.com/firemodels/exp/master/NIST_NRC/...` |
| `cam_94523_ligs_tc` | `FigureData.mat` + `Signalss_*.mat` (7 file) | 11.475.530 | `c060cff1` + 6 hash nel manifest | `https://www.repository.cam.ac.uk/bitstreams/<uuid>/download` *(UUID integrali nel manifest)* |
| `katurji_50hz_tc` | `TC_Array_50Hz_left_front_EE{1,3}.mat` | 1.298.824 | `2009e315` / `bb1a4709` | `https://ndownloader.figshare.com/files/{30748849,30748852}` |
| `ulb_furnace` | `Flame_Temperatures_data_V4.xlsx` + `README_V4.txt` | 182.390 | `a1d4fca8` / `e1f0300a` | `https://zenodo.org/api/records/7624702/files/.../content` |
| `cardiff_nh3_swirl` | `Radiation_Supplementary.xlsx` + 2 controlli negativi | 627.262 | `36f17f51` / `8a3b50e9` / `bc8fbe0c` | `https://ndownloader.figshare.com/files/{61298647,58362748,49921563}` |
| `wang_mwir_tc` | `data_IJTS_R1.zip` | 3.735.615 | `fa752904a9098c9f` | `https://zenodo.org/api/records/23054326/files/data_IJTS_R1.zip/content` |
| `sydney_thermofluids` & `tnf_sandia_def` | `Surface_Temp_BB.doc`, `SM1.zip`, `pmCDEF.zip` | 12.356.325 | `a85e1073` / `e23c6570` / `83089b2f` | `web.aeromech.usyd.edu.au` & `tnfworkshop.org` *(URL integrali nel manifest)* |

**Procedura di recupero e riproduzione:** seguire [step2_dataset_checks/README.md](step2_dataset_checks/README.md) (script heredoc Python 3.10+ con verifica SHA-256 automatica su `samples_manifest.json`, seguito da `inspect_all_samples.py` e `generate_verification_figures.py`).

---

## 5. Riferimenti bibliografici e identificatori permanenti

- **HYLON Phase 2 (TNF):** M. Vilespy et al., *Combust. Flame* 283 (2026) 114536, DOI `10.1016/j.combustflame.2025.114536`; database `HYLON_TNF_2026_01_Phase2-1.zip` su `https://tnfworkshop.org/hydrogen-flames/`.
- **NIST RSE 2007:** M. Bundy et al., NIST TN 1483 (2007); dati su `https://github.com/firemodels/exp/tree/master/NIST_RSE_2007`; coordinate verificate su `firemodels/fds` (`Validation/NIST_RSE_2007/FDS_Input_Files/RSE_01.fds`, `RSE_12.fds`).
- **NIST/NRC:** A. Hamins et al., NIST SP 1013-1 (2006); dati su `https://github.com/firemodels/exp/tree/master/NIST_NRC`; documentazione guasti verificata su `firemodels/fds` (`Manuals/FDS_Validation_Guide/HGL_Chapter.tex`, `Surface_Temperature_Chapter.tex`).
- **NIST Pool Fires (MaCFP):** K. Sung, A. Hamins et al., NIST TN 2083r1 / TN 2162r1; dati e README §4.1 su `https://github.com/MaCFP/macfp-db/tree/master/Liquid_Pool_Fires/NIST_Pool_Fires`.
- **Cambridge Pressurized Swirl LIGS:** L. Weller et al. (2023), Apollo Univ. Cambridge, DOI `10.17863/CAM.94523`.
- **Katurji 50 Hz In-Fire TC Array:** M. Katurji et al., Figshare, DOI `10.6084/m9.figshare.16615174.v2`.
- **ULB 20 kW Furnace:** M. Cafiero et al. (2022), Zenodo, DOI `10.5281/zenodo.7624702`; A. Parente et al. (2023), DOI `10.5281/zenodo.7624630`.
- **NIST Smyth Slot Burner:** T. S. Norton, K. C. Smyth et al., *Combust. Sci. Technol.* 90 (1993) 1–34; `https://github.com/firemodels/exp/tree/master/Smyth_Slot_Burner`.
- **Waterloo Methanol Pool Fire:** E. J. Weckman & A. B. Strong, *Combust. Flame* 105 (1996) 245–266; `https://github.com/MaCFP/macfp-db/tree/master/Liquid_Pool_Fires/Waterloo_Methanol`.
- **Cardiff NH$_3$ Swirl:** Sato et al. (2026), DOI `10.17035/cardiff.31141438`; negativi: `10.17035/cardiff.30234904`, `10.17035/d.2022.0217184646`, `10.17035/d.2022.0214909050`.
- **FM Burner & UMD Line Burner (MaCFP):** `https://github.com/MaCFP/macfp-db/tree/master/Extinction/{FM_Burner,UMD_Line_Burner}`.
- **Wang MWIR vs TC:** Zenodo, DOI `10.5281/zenodo.23054326` (v. R1-02, 2026-09-30).
- **MetForTC & Pearce Drift:** E. Turzo-Andras (2023), `10.5281/zenodo.8189490`; D. Sestan et al. (2023), `10.5281/zenodo.8186928`; J. Pearce (2021), `10.5281/zenodo.4972091`.
- **Brzosko 12-TC Grid:** M. Brzosko et al. (2026), TU Berlin, DOI `10.5281/zenodo.21371063`.
- **Pelzmann & Robert McKenna:** `10.17632/jvvvgpdy48.6`; **Fischer FWI:** `10.17632/jtmk3pg36k.1`.
- **Sydney Thermofluids & Sandia D–F:** `https://web.aeromech.usyd.edu.au/thermofluids/database.php`; `https://tnfworkshop.org/data-archives/pilotedjet/ch4-air/`.
- **NETL ESSB Rel. 1.0:** C. Bedick, P. Strakey, N. Weiland (2024), DOI `10.18141/2478453`.
- **4TU TU Delft:** Neeser & De Domenico (2025), DOI `10.4121/dbaa8f0f-59d2-4d58-8cc9-494af1b966f1.v1`; S. Link et al. (2025, negativo T), DOI `10.4121/7e680c48-7408-4e35-8880-10220814afd5.v1`.
- **PoliMi Swirl:** G. Solero, *IJETR* 7(2) (2017) 13–17, handle `11311/1152656`; A. Olivani et al., *ETFS* 31 (2007) 427–436, handle `11311/552372`.
- **Virginia Tech:** DOE-VT-001176-2, DOI `10.2172/1463254`.
- **Archivi bloccati `HTTP 403` (verificati via DataCite):** UMich Deep Blue TCC-III (`10.7302/z2hq3x4s`, `z2ms3qp3`, `z2h41pcd`, `z2cc0xnp`); TU Darmstadt TUdatalib (`10.48328/tudatalib-2291`, `1535`, `1532`, `1113`, `1114`, `1819`, `2135`, `2194`).
