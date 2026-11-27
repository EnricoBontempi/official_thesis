# Missione 04 — Step 2: fonti, registro delle ricerche e verifiche dei dataset termici

**Documento collegato a:** [step2_dataset_termici.md](step2_dataset_termici.md).  
**Artefatti riproducibili:** [step2_dataset_checks/README.md](step2_dataset_checks/README.md), [samples_manifest.json](step2_dataset_checks/samples_manifest.json), [inspect_all_samples.py](step2_dataset_checks/inspect_all_samples.py), [inspection_summary.json](step2_dataset_checks/inspection_summary.json), [generate_verification_figures.py](step2_dataset_checks/generate_verification_figures.py).  
**Data:** 27 novembre 2026 — **Branch:** `mission04/step2-thermal-datasets`.  
**Nota di integrità:** questo registro è integralmente autosufficiente (non dipende da appunti privati di subagenti). Nessun dataset pesante e nessun PDF integrale è stato committato nella repository; tutti gli 82 file ispezionati sono riproducibili tramite URL e hash SHA-256 registrati qui e in [samples_manifest.json](step2_dataset_checks/samples_manifest.json).

---

## 1. Metodologia di ricerca e canali interrogati

La ricerca è stata condotta il 26–27 novembre 2026 in due fasi integrate:
1. **Esplorazione parallela ad ampio spettro** (quattro filoni tematici: archivi TNF/DLR/Sydney/Adelaide, archivi NIST/MaCFP/`firemodels`, repository universitari EU/UK/US, e misure dinamiche/calibrazione/deriva di termocoppie).
2. **Verifica diretta autonoma tramite API REST e probe HTTP** (12 round di interrogazione live + controllo intestazioni `HEAD`/`Range` + download selettivo di 82 file rappresentativi, `49,62 MB` totali in locale temporaneo, e analisi in sola lettura via Python). Nessun dataset è stato dichiarato "accessibile" sulla base della sola citazione in un articolo o negli appunti di un filone: ogni accesso riportato nel rapporto è stato verificato direttamente via HTTP/API.

### 1.1 Canali e API interrogati

| Canale / Repository | Metodo di interrogazione | Ambito / query principali | Esito operativo |
|---|---|---|---|
| **TNF Workshop** (`tnfworkshop.org`) | Scraping pagine `/data-archives/*`, `/pilotedjet/`, `/swirlflames/`, `/hydrogen-flames/`, `HEAD` sugli ZIP | HYLON Phase 1/2/3, Sandia C–F, KAUST Cracked NH$_3$, Delft III, DLR A/B, AHEAD, NTNU | **HTTP 200** sugli ZIP diretti; ispezionati `HYLON_TNF_2026_01_Phase2-1.zip` e `pmCDEF.zip` |
| **Sydney Thermofluids** (`web.aeromech.usyd.edu.au/thermofluids/`) | Verifica pagina `/database.php`, `/bluff_files/`, `/swirl_files/` | Bluff body `HM1–3` (`Surface_Temp_BB.doc`), swirl `SM1.zip` | **HTTP 200** live; scaricati e ispezionati `Surface_Temp_BB.doc` e `SM1.zip` |
| **MaCFP Database** (`github.com/MaCFP/macfp-db`) | GitHub API + `raw.githubusercontent.com` | `NIST_Pool_Fires`, `Waterloo_Methanol`, `FM_Burner` (ott. 2025), `UMD_Line_Burner` | **HTTP 200**; scaricati e ispezionati 28 file + README |
| **FDS Experimental Archive** (`github.com/firemodels/exp`) | GitHub API + `raw.githubusercontent.com` + FDS Validation Guide TeX | `NIST_RSE_2007`, `NIST_NRC`, `Smyth_Slot_Burner`, `NIST_FSE_2008`, `NIST_RSE_1994` | **HTTP 200**; scaricati e ispezionati 10 file |
| **Zenodo REST API** (`zenodo.org/api/records`) | Query mirate (`swirl thermocouple`, `combustor temperature`, `thermocouple drift`, `pyrometer`, `MWIR thermocouple`, `suction pyrometer`, `ULB furnace`) | `7624702`, `7624630`, `23054326`, `8189490`, `8186928`, `4972091`, `21371063`, `23062725`, `23037088`, `19004211`, `3693670`, `20446928` | **HTTP 200**; scaricati e ispezionati 5 record completi/campioni |
| **DataCite REST API** (`api.datacite.org/dois`) | Interrogazione per 9 prefissi istituzionali (`10.4121` 4TU: 194 hit; `10.48328` TUdatalib: 34 hit; `10.7302` Deep Blue: 12 hit; `10.17035` Cardiff: 77 hit; `10.17863` Cambridge: 29 hit; `10.5445` KIT: 25 hit; `10.7294` VT: 3 hit; `10.25909` Adelaide: 1 hit; `10.3929` ETH: 0 hit; `10.14469` Imperial: 1 hit) | Verifica sistematica dell'esistenza di DOI dati sui rig dello Step 1 | **HTTP 200** sui metadati DataCite; ha permesso di confermare sia i record esistenti bloccati (`403`) sia le assenze (`VT`, `TU Delft Sampat/van den Berg`) |
| **Cardiff Research Data / Figshare API** (`api.figshare.com/v2`) | Query articoli & file su `10.17035/*` e `10.6084/m9.figshare.16615174` | `cardiff.31141438`, `cardiff.30234904`, `d.2022.0217184646`, `d.2022.0214909050`, `Katurji 16615174.v2` | **HTTP 200**; scaricati e ispezionati 5 file (2 positivi TC, 2 controlli negativi Cardiff, 2 `.mat` Katurji) |
| **4TU.ResearchData API v2** (`data.4tu.nl/v2/articles`) | Metadati + download README + HTTP `Range` TOC su ZIP | `Link 2025` (`7e680c48...`, `133 MB`), `Neeser 2025` (`dbaa8f0f...`, `2,05 GB`), `Navarro` (`19355384`) | **HTTP 200**; `REAME.pdf` + TOC `Link 2025` (negativo T) e `README.md` `Neeser 2025` (TC parete 100 Hz) |
| **Cambridge Apollo** (`repository.cam.ac.uk`) | Handle pages + bitstream UUID direct download | `CAM.94523` (LIGS swirl), `1810/226470` (Stratified Slot), `CAM.96660`, `CAM.90803`, `CAM.106960` | **HTTP 200** su bitstream diretti (`API REST /rest` 404); scaricati tutti i 7 `.mat` di `CAM.94523` |
| **OSTI.gov API + NETL EDX CKAN API** (`edx.netl.doe.gov/api/3/action/package_show`) | Verifica `10.2172/1463254` (VT), `10.18141/2478453` (NETL ESSB), `2552943` (NETL NH$_3$) + HTTP `Range` ZIP TOC | `NETL_ESSB_Data_Release_1.0.zip` (`96,9 MB`, 115 file), `2025_FlameSpeed_FTIRSpecies.zip` (`19,0 MB`) | **HTTP 200**; verificato TOC completo `NETL ESSB` (`Thermal Imaging/` + LDA + OH-PLIF); VT confermato solo report PDF |
| **Mendeley Data API** (`data.mendeley.com/api/datasets`) | Verifica `jvvvgpdy48.6` (Pelzmann) e `jtmk3pg36k.1` (Fischer FWI) | File CSV/TXT McKenna e BC parete FWI | **HTTP 200**; scaricati e ispezionati 3 file di `jvvvgpdy48.6` |
| **PoliMi IRIS** (`re.public.polimi.it`) | Ricerca autore (`Solero`, `Olivani`) + verifica handle `11311/1152656` e `11311/552372` | Verifica presenza dataset digitali e PDF completi | **HTTP 200** su `IJETR2102.pdf` (scaricato temporaneamente e letto integralmente); `etfsPAPER2007.pdf` bloccato da login; **zero dataset digitali** |
| **Dryad API** (`datadryad.org/api/v2`) & **NIST PDR** (`data.nist.gov`) | Verifica `10.5061/dryad.4cj4k`, `10.18434/mds2-2314`, `M32135`, `T4/1502568`, `mds2-2812` | Metadati e disponibilità file | **HTTP 200** metadati; Dryad `302 MB` non scaricato; NIST FCD `mds2-2314` richiede portale interattivo |
| **ISF Workshop** (`isfworkshop.org` & `adelaide.edu.au/cet/isfworkshop`) | Probe HTTP diretto delle pagine dataset | ISF-3 DLR pressurizzato e archivi soot Adelaide | **Link live rotti** (redirect generico Adelaide University post-fusione; `isfworkshop.org` punta a snapshot Wayback Machine del 2 giugno 2025) |

---

## 2. Esito puntuale delle piste iniziali (Step 1 §14.2) e delle piste aggiuntive

| Pista iniziale (Step 1 §14.2 / incarico) | Esito della verifica | Classificazione accesso | Note documentate |
|---|---|---|---|
| **Virginia Tech Optical Swirl Combustor** (`OSTI 1463254`, DOE-VT-001176-2) | Verificati API OSTI e prefisso DataCite `10.7294` (`data.lib.vt.edu`: solo 3 record combustion, nessuno del rig DE-FE0011762) | **`non individuato` / `su richiesta`** | Esiste solo il rapporto finale PDF (264 pp.); i 320 nodi TC tipo B e le mappe IR non risultano archiviati pubblicamente. |
| **TU Delft (Sampat *Fuel* 2025/GT2022; van den Berg *AECS* 2024; X. Huang 2018; DJHC)** | Interrogati 4TU.ResearchData (`10.4121`, 194 record combustion) e TU Delft Repository | **`non individuato` / `su richiesta`** | Nessuno di questi rig ha un dataset pubblico su 4TU. Individuati invece su 4TU: `Link 2025` (swirl CH$_4$/H$_2$, **zero dati T** verificato via TOC+README) e `Neeser 2025` (accenditore GOx/CH$_4$, **TC parete 100 Hz** + `sensors.py`, `2,05 GB`). |
| **Archivi TNF: HYLON, Sandia D–F, Delft III, DLR A–B, KAUST NH$_3$** | Verificati link diretti su `tnfworkshop.org` | **`accessibile`** (`HYLON Phase 2` e `Sandia pmCDEF` **ispezionati**; altri HTTP 200) | **`HYLON Phase 2` (gennaio 2026)** è il ritrovamento principale per combustori swirl confinati: doppia TC + Raman 1D + LIP pareti + PIV (`4Q`/`2S2Q`) + CAD. |
| **TNF/Darmstadt TSF & SWQ, CSSB scalari/fosfori, TECFLAM, Lund LSB** | Verificate pagine TNF, TUdatalib e Cambridge Apollo | **`su richiesta` / `non individuato`** | Su Cambridge Apollo esiste `1810/243259` (CSSB **sola LDA**, senza T) e `1810/226470` (Stratified Slot Raman/Rayleigh, *All Rights Reserved*). TSF/SWQ e fosfori bluff body non scaricabili apertamente. |
| **DLR GTMC / PRECCINSTA / SFB 606 / HBK / FRS** | Verificati elib DLR, TNF e DataCite | **`su richiesta`** | I PDF descrittivi sono aperti su `elib.dlr.de`, ma i file numerici (Raman, CARS 5 kHz, fosfori di parete) sono forniti solo tramite richiesta/accordo con DLR. |
| **ISF-3 (DLR pressurizzato) & Adelaide JHC** | Probe HTTP su `isfworkshop.org`, `adelaide.edu.au/cet/isfworkshop/` e prefisso DataCite `10.25909` (solo 1 record) | **`su richiesta` / link rotti** | Le pagine CET di Adelaide reindirizzano alla home generica post-fusione universitaria; `isfworkshop.org` rinvia a un backup Wayback Machine del 2025. Nessun dataset numerico ISF-3 o Adelaide JHC scaricabile direttamente. |
| **Cardiff e altri rig ad ammoniaca** | Interrogati prefisso `10.17035` (77 hit), Figshare API, Zenodo, NETL EDX, TNF KAUST | **`accessibile`** (misto positivo / negativo) | **Positivo ispezionato:** `cardiff.31141438` (Sato 2026: TC scarico + parete int/est su 9 $\Phi$). **Negativi verificati aprendo i file:** `cardiff.30234904` e `d.2022.0217184646` (zero T). Altri NH$_3$ aperti: `TNF KAUST Cracked NH3` (HTTP 200), `ULB 7624630` (pirometro a suzione NH$_3$/H$_2$), `NETL 2552943` (McKenna flame speed/FTIR), `Zenodo 19004211`. |
| **PoliMi (Solero / Olivani / Coghe)** | Interrogata `re.public.polimi.it` (IRIS), scaricato `IJETR2102.pdf` (`11311/1152656`), verificato `11311/552372` | **`non individuato` (solo PDF / login)** | **Verificato su testo integrale (`Solero 2017`):** TC tipo R `0,3 mm` (500 Hz, 5000 campioni), corr. radiativa Morgan (1991) con velocità PIV/LDA — **sole figure in PDF, zero dataset digitale**. `Olivani 2007` PDF protetto da login. |
| **UMich Deep Blue TCC-III** | Interrogata DataCite (`10.7302`, 12 record) + probe HTTP su `deepblue.lib.umich.edu` | **`bloccato dall'ambiente` (`HTTP 403`)** | Confermati via DataCite `z2hq3x4s` (Near-Wall Flow: PIV + TC coassiale rapida di flusso termico), `z2ms3qp3`, `z2h41pcd`, `z2cc0xnp`. Download bloccato da HTTP 403 (non aggirato). |
| **TUdatalib Darmstadt** | Interrogata DataCite (`10.48328`, 34 record) + probe HTTP su `tudatalib.ulb.tu-darmstadt.de` | **`bloccato dall'ambiente` (`HTTP 403` Anubis)** | Confermati via DataCite `tudatalib-2291` (pistone fosfori H$_2$, 2026, CC BY 4.0), `1819` (Raman NH$_3$/H$_2$/N$_2$), `2135` (Raman H$_2$), `2194` (TDLAS), `1535/1532` (motori ottici), `1113/1114` (flow bench). Landing/download bloccati da protezione Anubis HTTP 403 (non aggirata). |
| **Sydney Thermofluids (Bluff Body & Swirl)** | Probe HTTP diretto su `web.aeromech.usyd.edu.au/thermofluids/` | **`accessibile (ispezionato)`** | `SM1.zip` (133.389 colpi Raman/Rayleigh/LIF, 9 outlier `<250 K`) e `Surface_Temp_BB.doc` (**pirometro IR Mikron M90-D**, non TC, su hardware scambiato). |
| **EngineBench (`arXiv:2406.03325`) & OSU McManus (FRS+PIV)** | Verifica metadati/paper e repository pubblici | **`controllo negativo`** (EngineBench) / **`non individuato`** (McManus) | `EngineBench` contiene solo PIV motored a freddo (nessuna combustione né T). I dati della tesi/paper McManus & Sutton (FRS+PIV) non risultano depositati in archivi aperti. |
| **Calibrazioni sperimentali e misure dinamiche TC (pista ampia)** | Ricerca ampia su Zenodo, Figshare, MaCFP, `firemodels/exp`, Mendeley, 4TU | **Misto (`accessibile` per bias/dropout/deriva; `non individuato` per step-response kHz)** | **Trovati e ispezionati:** `NIST_RSE_2007`, `NIST/NRC`, `NIST_Pool_Fires`, `Katurji 50 Hz`, `MetForTC` (`8189490`), `Pearce` (`4972091`), `Waterloo`, `ULB`, `Wang MWIR`, `Brzosko`. **Non individuati:** serie raw kHz a doppia TC in fiamma e dataset raw di risposta a gradino/impulso laser di TC. |

---

## 3. Registro delle esclusioni, controlli negativi e accessi falliti / bloccati

### 3.1 Controlli negativi verificati aprendo i file o i TOC
- **`TU Delft Swirl Burner` (`10.4121/7e680c48-7408-4e35-8880-10220814afd5.v1`, Link et al. 2025):** scaricato `REAME.pdf` (5 pp.) e letto l'indice ZIP remoto (`95` file, `133 MB`) — contiene esclusivamente PIV reagente/non-reagente, modi SPOD, OH-PLIF qualitativo ed emissioni NO/CO$_2$; **zero dati termici**.
- **`Cardiff COG/NH3` (`10.17035/cardiff.30234904`, `COGNH3_Supplementary.xlsx`) & `Mashruk 2022` (`10.17035/d.2022.0217184646`, `220819_underpinning data_Energy.xlsx`):** aperti e parsati integralmente via `inspect_all_samples.py`; entrambi contengono **zero colonne o fogli di temperatura** (solo concentrazioni specie e chemiluminescenza/OES).
- **`Pelzmann & Robert McKenna` (`10.17632/jvvvgpdy48.6`):** aperti `20200511_SBG_FFC_condition01/03.csv` e `20201104_TPH_log.txt`; confermato che `SBG` è uno **Schmidt-Boelter heat flux gauge** (`U [mV], Q [W], Q [kW]`), non una termocoppia nel gas (le temperature gas del dataset provengono da DLR CARS/Cantera).
- **`Cambridge CAM.94523` (`10.17863/CAM.94523`) — assenza parziale:** valido per LIGS ottica (inclusi 4000 colpi raw con dropout), ma **negativo per i dati TC** citati nell'abstract (`0` variabili TC su `61` in `FigureData.mat`).
- **`EngineBench`, `Cambridge CCB` (`CAM.96660`), `LEAFINNOX` (`CAM.90803`), `Yoko` (`CAM.106960`), `CSSB LDA` (`1810/243259`), `TU Berlin AHEAD`, `NTNU Bluff-Body H2`:** esclusi per assenza di misure di temperatura nei metadati/file depositati.

### 3.2 Accessi bloccati dall'ambiente, protetti da login o link rotti (mai aggirati)
- **`UMich Deep Blue Data` (`deepblue.lib.umich.edu/data/concern/generic_works/*`):** `HTTP 403 Forbidden` sui download diretti per `10.7302/z2hq3x4s`, `z2ms3qp3`, `z2h41pcd`, `z2cc0xnp`. Metadati verificati solo via API DataCite.
- **`TU Darmstadt TUdatalib` (`tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/*`):** `HTTP 403 Forbidden` (sfida anti-bot Anubis) per `tudatalib-2291`, `1535`, `1532`, `1113`, `1114`, `1819`, `2135`, `2194`. Metadati verificati solo via API DataCite.
- **`PoliMi IRIS` — `Olivani et al. 2007` (`re.public.polimi.it/retrieve/.../etfsPAPER2007.pdf`):** restituisce pagina HTML di login istituzionale (accesso riservato). A differenza di questo, `Solero 2017` (`IJETR2102.pdf`) era pubblico (`HTTP 200`) ed è stato letto integralmente.
- **`ISF Workshop` / `Adelaide CET` (`adelaide.edu.au/cet/isfworkshop/data-sets`):** tutti i sottopercorsi storici reindirizzano alla pagina generica della nuova Adelaide University (`189 kB` HTML senza file dati); `isfworkshop.org` rinvia esplicitamente a uno snapshot Wayback Machine (`20250602034109`).
- **`Cambridge Apollo REST v6` (`/rest/items/...`):** `HTTP 404` (endpoint deprecato in DSpace 7+); risolto in modo regolare usando i collegamenti pubblici `/bitstreams/<uuid>/download` presenti nelle pagine handle (`HTTP 200`).
- **`NIST Fire Calorimetry Database` (`10.18434/mds2-2314`):** l'API JSON del record PDR riporta `0` file diretti perché i dati sono serviti tramite web-app interattiva separata; non prioritario rispetto ai CSV completi già ispezionati su `firemodels/exp` e `MaCFP`.

### 3.3 Esclusioni per dimensione / formato / priorità (archivi aperti ma non scaricati)
- **`NETL ESSB Release 1.0` (`96,9 MB`)** e **`4TU Neeser GOx/CH4` (`2,05 GB`):** verificati rispettivamente tramite indice ZIP remoto (`HTTP Range`, 115 file) e `README.md` scaricato, non scaricati integralmente per rispettare il vincolo "senza download massivi".
- **`Dryad ETH Bonciolini` (`302 MB` `.tdms`)**, **`Zenodo 20446928` (`668 MB`)**, **`TNF HYLON Phase 1` (`35,2 MB`)**, **`TNF KAUST Cracked NH3` (`24,9 MB`)**, **`TU Berlin AHEAD` (`3,47 GB`)**: verificati via API/`HEAD` ma non scaricati per priorità o dimensione.
- **`Pearce Drift Survey` (`10.5281/zenodo.4972091`):** scaricati tutti i 9 `.opju` (`3,1 MB`) + `FILES_CORRESPONDING_TO_FIGURES.txt`, ma il formato binario proprietario OriginPro `.opju` non è leggibile direttamente con librerie Python standard senza Origin; documentato il limite di formato.

---

## 4. Provenienza, integrità e riproduzione dei campioni ispezionati

Tutti gli **82 file** scaricati temporaneamente in locale (totale `49.626.582` byte = `47,33 MiB` / `49,63 MB`, scaricati il `2026-11-27T02:49–02:52 UTC`, tutti `HTTP 200`) sono inventariati con URL completo, `ETag`/`Last-Modified`, `Content-Type`, dimensione esatta e hash **SHA-256** integrale in [samples_manifest.json](step2_dataset_checks/samples_manifest.json).

### 4.1 Riepilogo per dataset e metodo di ispezione

| `dataset_id` (manifest) | File | Byte totali | Sorgente / versione | Metodo di ispezione ([inspect_all_samples.py](step2_dataset_checks/inspect_all_samples.py)) |
|---|---:|---:|---|---|
| `tnf_hylon_phase2` | 1 ZIP (`73` file reali) | `9.273.876` | `tnfworkshop.org` (`2026-01`, Last-Mod `2026-01-24`) | Estrazione in memoria da ZIP; parsing profili TC, Raman 1D, LIP pareti, PIV `4Q`/`2S2Q`; interpolazione overlap TC–Raman e `2S2Q`–`4Q`; lettura PDF 24 pp. (PyMuPDF) |
| `cam_94523_ligs_tc` | 7 `.mat` | `11.475.530` | `repository.cam.ac.uk` (`10.17863/CAM.94523`, `2023-03-03`) | `scipy.io.loadmat`; inventario di tutte le 61 variabili di `FigureData.mat`; conteggio colpi rigettati (`uint8([0])`) su `4 × 1000` colpi raw |
| `tnf_sandia_def` | 1 ZIP (`144` file) | `8.053.410` | `tnfworkshop.org` (Rel. 2.0, Last-Mod `2019-02-03`) | Lettura in memoria di `pmD.scat/D15.Yall` (`9.339` colpi) e `D15.Yave` |
| `sydney_thermofluids` | 2 (`SM1.zip`, `.doc`) | `4.302.915` | `web.aeromech.usyd.edu.au` (Last-Mod `2013-10-01`) | Parsing 9 file `.YAL` (`133.389` colpi, check `<250 K` e `TDNR`); estrazione testo binario da `Surface_Temp_BB.doc` |
| `brzosko_tc_grid` | 2 (`.npz`, `.py`) | `4.031.563` | `zenodo.org/records/21371063` (2026-07-15) | `numpy.load`; verifica shape `(211,532,12)`, `mask`, conteggio `NaN` vs `0.0` |
| `wang_mwir_tc` | 1 ZIP | `3.735.615` | `zenodo.org/records/23054326` | Lettura in memoria CSV campagna 2 (`0°`) e `analysis_tc_radiation_error.csv` |
| `pearce_drift_survey` | 10 (`9 .opju`, `1 .txt`) | `3.089.342` | `zenodo.org/records/4972091` (2021-06-17) | Verifica integrità archivio e mappa figure; rilevato formato proprietario `.opju` |
| `katurji_50hz_tc` | 2 `.mat` | `1.298.824` | `ndownloader.figshare.com` (`16615174.v2`) | `scipy.io.loadmat`; mappa `NaN` per canale, run contigui di dropout, distribuzione passi `|\Delta T|` a `20 ms` |
| `nist_nrc` | 3 `.csv` | `1.028.143` | `firemodels/exp` (`master`) | Parser CSV 1-intestazione; check regolarità $\Delta t=10\text{ s}$, rilevamento schema `999.0 -> 0.0`, soglia `>1500 °C` (`TC South U-3-1`) e `NaN` |
| `metfortc_drift` | 1 `.xlsx` | `907.052` | `zenodo.org/records/8189490` (2023-07-27) | Parser XML `zipfile`+`ElementTree` sui 4 fogli raw Keithley (`2021 K`, `2022 K`, `2022 BRML S`, `2023 S`) |
| `docs_verification` | 3 (`2 PDF`, `1 MD`) | `883.876` | `data.4tu.nl` (`Link`, `Neeser`), `re.public.polimi.it` (`Solero 2017`) | Verifica documentale testuale (PyMuPDF/UTF-8); non letti dallo script numerico |
| `cardiff_nh3_swirl` | 3 (`2 .xlsx`, `1 .zip`) | `627.262` | `ndownloader.figshare.com` (`31141438`, `30234904`, `27277428`) | Parser XML su fogli Excel; verifica valori T (`31141438`) e conferma negativa assenza T (`30234904`, `27277428`) |
| `nist_rse_2007` | 3 `.csv` | `276.747` | `firemodels/exp` (`master`) | Parser CSV 2-intestazioni; check $\Delta t=30\text{ s}$, `NaN`, 5 coppie co-localizzate `*BB` vs `*A` (pre-ignizione vs combustione attiva), 9 colonne velocità `VD*` |
| `ulb_furnace` | 3 (`1 .xlsx`, `2 .txt`) | `185.712` | `zenodo.org/records/7624702` & `7624630` | Parser XML su `data` e `units`; confronto con `README_V4.txt`; conteggio righe incomplete (`545/5060`) e dispersione sui `2.211` replicati |
| `fm_burner` | 9 (`8 .csv`, `1 .py`) | `149.946` | `MaCFP/macfp-db` (update ott. 2025) | Conteggio celle vuote (`9/693`), integrazione numerica PDF a 200 bin, verifica velocità 2C |
| `pelzmann_mckenna` | 3 (`2 .csv`, `1 .txt`) | `73.157` | `data.mendeley.com` (`jvvvgpdy48.6`) | Verifica intestazioni `SBG` (flusso termico, non TC) e controllo monotonia timestamp `TPH_log.txt` (`dt = -10 s`) |
| `smyth_slot_burner` | 4 `.txt` | `42.326` | `firemodels/exp` (`master`) | Confronto asimmetria destra-sinistra e interpolazione diretta file simmetrizzato vs originale; check file LDV |
| `nist_pool_fires` | 10 `.csv` | `22.268` | `MaCFP/macfp-db` (`Sung_2021` & `Sung_2024`) | Parser dual-schema (1 vs 2 intestazioni); calcolo `TG - TC` e `TG_RMS / TC_RMS` stratificato per `D_bead`; check serie liquida `Acetone` e velocità `Methane` |
| `waterloo_methanol` | 10 `.csv` | `23.153` | `MaCFP/macfp-db` (`master`) | Verifica byte LF/CR (bug upstream `0 LF/0 CR` sui compositi); parsing slice `z=10 cm` (`T-bar`, `T-rms`, `<u'T'>`, `<v'T'>`) |
| `umd_line_burner` | 4 (`3 .csv`, `1 .md`) | `7.658` | `MaCFP/macfp-db` (`master`) | Parsing `TCz`, `TCx`, `TC_Data` e confronto unità con `README.md` |

### 4.2 Tabella puntuale dei file chiave ispezionati (selezione dei file primari; elenco completo di tutti gli 82 file in [samples_manifest.json](step2_dataset_checks/samples_manifest.json))

| Dataset | File | Byte | SHA-256 (primi 16 caratteri) | URL sorgente |
|---|---|---:|---|---|
| `tnf_hylon_phase2` | `HYLON_TNF_2026_01_Phase2-1.zip` | 9.273.876 | `4dc63f18651f6145` | `https://tnfworkshop.org/wp-content/uploads/2026/01/HYLON_TNF_2026_01_Phase2-1.zip` |
| `nist_rse_2007` | `NIST_RSE_2007_01.csv` | 74.849 | `52347762e40e57eb` | `https://raw.githubusercontent.com/firemodels/exp/master/NIST_RSE_2007/NIST_RSE_2007_01.csv` |
| `nist_rse_2007` | `NIST_RSE_2007_02.csv` | 112.433 | `0a69af1091d3f558` | `https://raw.githubusercontent.com/firemodels/exp/master/NIST_RSE_2007/NIST_RSE_2007_02.csv` |
| `nist_rse_2007` | `NIST_RSE_2007_12.csv` | 89.465 | `152f80285beb3dc0` | `https://raw.githubusercontent.com/firemodels/exp/master/NIST_RSE_2007/NIST_RSE_2007_12.csv` |
| `nist_nrc` | `NIST_NRC_Test_01_file_1.csv` | 437.995 | `d97758dcc5380d1d` | `https://raw.githubusercontent.com/firemodels/exp/master/NIST_NRC/NIST_NRC_Test_01_file_1.csv` |
| `nist_nrc` | `NIST_NRC_Test_01_file_2.csv` | 367.075 | `eaf78bd8dc855249` | `https://raw.githubusercontent.com/firemodels/exp/master/NIST_NRC/NIST_NRC_Test_01_file_2.csv` |
| `nist_nrc` | `NIST_NRC_Test_04_file_1.csv` | 223.073 | `96ccad1b950baac8` | `https://raw.githubusercontent.com/firemodels/exp/master/NIST_NRC/NIST_NRC_Test_04_file_1.csv` |
| `nist_pool_fires` | `Methanol_100_cm_TC_r=0_cm_Sung_2021.csv` | 486 | `7fd7c7ca106b7f4e` | `https://raw.githubusercontent.com/MaCFP/macfp-db/master/Liquid_Pool_Fires/NIST_Pool_Fires/Experimental_Data/Methanol_100_cm_TC_r=0_cm_Sung_2021.csv` |
| `nist_pool_fires` | `Methanol_30_cm_TC_r=0_cm_Sung_2024.csv` | 545 | `fa4ce9e59d42ac0f` | `https://raw.githubusercontent.com/MaCFP/macfp-db/master/Liquid_Pool_Fires/NIST_Pool_Fires/Experimental_Data/Methanol_30_cm_TC_r=0_cm_Sung_2024.csv` |
| `nist_pool_fires` | `Ethanol_30_cm_TC_r=0_cm_Sung_2024.csv` | 425 | `1967df48e2742f2b` | `https://raw.githubusercontent.com/MaCFP/macfp-db/master/Liquid_Pool_Fires/NIST_Pool_Fires/Experimental_Data/Ethanol_30_cm_TC_r=0_cm_Sung_2024.csv` |
| `nist_pool_fires` | `Propane_37_cm_50_kW_TC_r=0_cm_Sung_2024.csv` | 529 | `e7b7c9ce8edee290` | `https://raw.githubusercontent.com/MaCFP/macfp-db/master/Liquid_Pool_Fires/NIST_Pool_Fires/Experimental_Data/Propane_37_cm_50_kW_TC_r=0_cm_Sung_2024.csv` |
| `nist_pool_fires` | `Acetone_30_cm_TC_Sung_2024.csv` | 18.981 | `41ec5eefc3107496` | `https://raw.githubusercontent.com/MaCFP/macfp-db/master/Liquid_Pool_Fires/NIST_Pool_Fires/Experimental_Data/Acetone_30_cm_TC_Sung_2024.csv` |
| `cam_94523_ligs_tc` | `FigureData.mat` | 7.695.591 | `c060cff1289f1d3a` | `https://www.repository.cam.ac.uk/bitstreams/b264bf63-442c-4624-b736-b7369f1b7d02/download` |
| `cam_94523_ligs_tc` | `Signalss_thermal.mat..4.mat` (4 file) | 3.761.229 | `dd103fbe` / `c06b631b` / `bbb65515` / `eab960d4` | `https://www.repository.cam.ac.uk/bitstreams/{f6e5a9cd,89d38b66,4a910ba1,b5132b18}-.../download` *(UUID integrali nel manifest)* |
| `katurji_50hz_tc` | `TC_Array_50Hz_left_front_EE1.mat` | 963.963 | `2009e315c621b984` | `https://ndownloader.figshare.com/files/30748849` |
| `katurji_50hz_tc` | `TC_Array_50Hz_left_front_EE3.mat` | 334.861 | `bb1a4709fa541a4b` | `https://ndownloader.figshare.com/files/30748852` |
| `ulb_furnace` | `Flame_Temperatures_data_V4.xlsx` | 175.002 | `a1d4fca8f50d3186` | `https://zenodo.org/api/records/7624702/files/Flame_Temperatures_data_V4.xlsx/content` |
| `ulb_furnace` | `README_V4.txt` | 7.388 | `e1f0300ae8f18e48` | `https://zenodo.org/api/records/7624702/files/README_V4.txt/content` |
| `cardiff_nh3_swirl` | `Radiation_Supplementary.xlsx` | 13.775 | `36f17f513de00253` | `https://ndownloader.figshare.com/files/61298647` |
| `cardiff_nh3_swirl` | `COGNH3_Supplementary.xlsx` (neg.) | 16.679 | `8a3b50e9b12fbc6f` | `https://ndownloader.figshare.com/files/58362748` |
| `cardiff_nh3_swirl` | `d.2022.0217184646.zip` (neg.) | 596.808 | `bc8fbe0c7a9d3b04` | `https://ndownloader.figshare.com/files/49921563` |
| `smyth_slot_burner` | `Temperature-thermocouple.txt` | 10.866 | `c409f3101c56c79c` | `https://raw.githubusercontent.com/firemodels/exp/master/Smyth_Slot_Burner/Temperature-thermocouple.txt` |
| `smyth_slot_burner` | `Temperature-thermocouple-radiation-corrected-original.txt` | 12.220 | `943da3bf6ccbbbea` | `https://raw.githubusercontent.com/firemodels/exp/master/Smyth_Slot_Burner/Temperature-thermocouple-radiation-corrected-original.txt` |
| `waterloo_methanol` | `Mean_and_rms_values_z10cm.csv` & `Estimated_correlations_z10cm.csv` | 971 | `6e90f4d5` / `dcb57ff3` | `https://raw.githubusercontent.com/MaCFP/macfp-db/master/Liquid_Pool_Fires/Waterloo_Methanol/Experimental_Data/...` |
| `fm_burner` | `temperature_mean.csv` & `temperature_PDF_OC20.9_1.5D.csv` | 22.355 | `cab15521` / `73620e21` | `https://raw.githubusercontent.com/MaCFP/macfp-db/master/Extinction/FM_Burner/Experimental_Data/FM_Burner_dataset_update_Oct2025/...` |
| `umd_line_burner` | `TCz_Data.csv`, `TCx_Data.csv`, `TC_Data.csv` | 2.924 | `2f2e1998` / `881eb905` / `a9afd9ee` | `https://raw.githubusercontent.com/MaCFP/macfp-db/master/Extinction/UMD_Line_Burner/Experimental_Data/...` |
| `wang_mwir_tc` | `data_IJTS_R1.zip` | 3.735.615 | `fa752904a9098c9f` | `https://zenodo.org/api/records/23054326/files/data_IJTS_R1.zip/content` |
| `metfortc_drift` | `MetForTC_Drift_TDMF_2023.xlsx` | 907.052 | `92552b0ce670878f` | `https://zenodo.org/api/records/8189490/files/MetForTC%20-%20Drift%20tests%20of%20TCs%20with%20TDMF%20-%202023.07.27.xlsx/content` |
| `brzosko_tc_grid` | `experimental_pin_temperature_dataset.npz` | 4.029.073 | `60947ebb0d5e8688` | `https://zenodo.org/api/records/21371063/files/experimental_pin_temperature_dataset.npz/content` |
| `pelzmann_mckenna` | `20200511_SBG_FFC_condition01.csv` & `20201104_TPH_log.txt` | 30.278 | `82805939` / `08cad7da` | `https://data.mendeley.com/public-files/datasets/jvvvgpdy48/files/{e765d7c5,faf84754}-.../file_downloaded` |
| `sydney_thermofluids` | `Surface_Temp_BB.doc` & `SM1.zip` | 4.302.915 | `a85e1073` / `e23c6570` | `https://web.aeromech.usyd.edu.au/thermofluids/{bluff_files/data/Surface_Temp_BB.doc,swirl_files/data/SM1.zip}` |
| `tnf_sandia_def` | `pmCDEF.zip` | 8.053.410 | `83089b2f0bb59101` | `https://tnfworkshop.org/wp-content/uploads/2019/02/pmCDEF.zip` |
| `docs_verification` | `PoliMi_Solero_2017_IJETR2102.pdf`, `TUD_swirlburner_REAME.pdf`, `TUD_neeser_README.md` | 883.876 | `3eb49be8` / `8ebd30dc` / `c5f29172` | `re.public.polimi.it` & `data.4tu.nl` (URL completi nel manifest; nessun PDF committato in git) |

**Procedura di recupero e riproduzione:** seguire le istruzioni in [step2_dataset_checks/README.md](step2_dataset_checks/README.md) (download verificato per SHA-256 da `samples_manifest.json` in una cartella temporanea, seguito da `inspect_all_samples.py` e `generate_verification_figures.py`).

---

## 5. Riferimenti bibliografici e identificatori permanenti

- **HYLON Phase 2 (TNF):** M. Vilespy et al., *Combust. Flame* 283 (2026) 114536, DOI `10.1016/j.combustflame.2025.114536`; database `HYLON_TNF_2026_01_Phase2-1.zip` su `https://tnfworkshop.org/hydrogen-flames/`.
- **NIST RSE 2007:** M. Bundy, A. Hamins, E. L. Johnsson, S. C. Kim, G. H. Ko, D. B. Lenhert, *Measurements of Heat and Combustion Products in Reduced-Scale Ventilation-Limited Compartment Fires*, NIST TN 1483 (2007); dati su `https://github.com/firemodels/exp/tree/master/NIST_RSE_2007`.
- **NIST/NRC:** A. Hamins et al., *Report of Experimental Results for the International Fire Model Benchmarking and Validation Exercise #3*, NIST SP 1013-1 (2006); dati su `https://github.com/firemodels/exp/tree/master/NIST_NRC`.
- **NIST Pool Fires (MaCFP):** K. Sung, A. Hamins et al., NIST TN 2083r1 / TN 2162r1; dati su `https://github.com/MaCFP/macfp-db/tree/master/Liquid_Pool_Fires/NIST_Pool_Fires`.
- **Cambridge Pressurized Swirl LIGS:** L. Weller, P. Shah, A. Giles, F. De Domenico, S. Morris, B. A. O. Williams, S. Hochgreb (2023), Apollo Univ. Cambridge, DOI `10.17863/CAM.94523`.
- **Katurji 50 Hz In-Fire TC Array:** M. Katurji et al., *Flame thermocouple data*, Figshare, DOI `10.6084/m9.figshare.16615174.v2`.
- **ULB 20 kW Furnace:** M. Cafiero, S. Sharma, M. M. Kamal, A. Coussement, A. Parente (2022), Zenodo, DOI `10.5281/zenodo.7624702`; A. Parente et al. (2023), MILD non-conventional fuels, DOI `10.5281/zenodo.7624630`.
- **NIST Smyth Slot Burner:** T. S. Norton, K. C. Smyth, J. H. Miller, M. D. Smooke, *Combust. Sci. Technol.* 90 (1993) 1–34; dati su `https://github.com/firemodels/exp/tree/master/Smyth_Slot_Burner`.
- **Waterloo Methanol Pool Fire:** E. J. Weckman & A. B. Strong, *Combust. Flame* 105 (1996) 245–266; dati su `https://github.com/MaCFP/macfp-db/tree/master/Liquid_Pool_Fires/Waterloo_Methanol`.
- **Cardiff NH$_3$ Swirl:** Sato et al. (2026), DOI `10.17035/cardiff.31141438`; controlli negativi: `10.17035/cardiff.30234904`, `10.17035/d.2022.0217184646` (Mashruk *Energy* 2022), `10.17035/d.2022.0214909050`.
- **FM Burner & UMD Line Burner (MaCFP):** `https://github.com/MaCFP/macfp-db/tree/master/Extinction/FM_Burner` (update ott. 2025); J. P. White et al., *Fire Safety J.* 76 (2015) / PhD UMD (2016), `https://github.com/MaCFP/macfp-db/tree/master/Extinction/UMD_Line_Burner`.
- **Wang MWIR vs TC:** *Aero-Engine Exhaust Plume Radiance-Temperature Dataset from Single-Band MWIR Thermometry*, Zenodo, DOI `10.5281/zenodo.23054326`.
- **MetForTC & Pearce Drift:** E. Turzo-Andras (2023), DOI `10.5281/zenodo.8189490`; D. Sestan et al. (2023), DOI `10.5281/zenodo.8186928`; J. Pearce (2021), DOI `10.5281/zenodo.4972091`.
- **Brzosko 12-TC Grid:** M. Brzosko et al. (2026), TU Berlin, Zenodo, DOI `10.5281/zenodo.21371063`.
- **Pelzmann & Robert McKenna:** Mendeley Data v6, DOI `10.17632/jvvvgpdy48.6`; **Fischer FWI:** Mendeley Data v1, DOI `10.17632/jtmk3pg36k.1`.
- **Sydney Thermofluids & Sandia D–F:** `https://web.aeromech.usyd.edu.au/thermofluids/database.php` (`SM1.zip`, `Surface_Temp_BB.doc`); `https://tnfworkshop.org/data-archives/pilotedjet/ch4-air/` (`pmCDEF.zip`).
- **NETL ESSB Rel. 1.0:** C. Bedick, P. Strakey, N. Weiland (2024), OSTI/EDX, DOI `10.18141/2478453`.
- **4TU TU Delft:** Neeser & De Domenico (2025, GOx/CH$_4$ igniter), DOI `10.4121/dbaa8f0f-59d2-4d58-8cc9-494af1b966f1.v1`; S. Link, F. De Domenico, A. Gangoli Rao, G. Eitelberg (2025, swirl CH$_4$/H$_2$, negativo T), DOI `10.4121/7e680c48-7408-4e35-8880-10220814afd5.v1`.
- **PoliMi Swirl:** G. Solero, *Experimental Analysis of a Swirl Burner for Propulsion Applications: Influence of Thermal and Fluid Dynamic Field on Pollutant Emissions*, *IJETR* 7(2) (2017) 13–17, handle `11311/1152656`; A. Olivani et al., *ETFS* 31 (2007) 427–436, handle `11311/552372`.
- **Virginia Tech:** *Heat Transfer Methodologies in a Reacting Gas Turbine Combustor*, DOE-VT-001176-2, DOI `10.2172/1463254` (solo rapporto PDF).
- **Archivi bloccati `HTTP 403` (verificati via DataCite):** UMich Deep Blue TCC-III (`10.7302/z2hq3x4s`, `z2ms3qp3`, `z2h41pcd`, `z2cc0xnp`); TU Darmstadt TUdatalib (`10.48328/tudatalib-2291`, `1535`, `1532`, `1113`, `1114`, `1819`, `2135`, `2194`).
