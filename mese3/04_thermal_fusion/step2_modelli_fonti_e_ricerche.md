# Missione 04 — Step 2 (agente A): fonti, ricerche e registro d'accesso

**Documento collegato a:** [step2_modelli_di_misura.md](step2_modelli_di_misura.md).
**Data:** 27 novembre 2026.
**Stato:** consegna da revisionare, non approvata.
**Nessun PDF in repository.** Le fonti sono indicate con DOI o URL.

---

## 0. Livelli di lettura e metodo

**Livelli di lettura:** come nello step 1.

| Sigla | Significato |
|---|---|
| **TI** | documento intero |
| **TS** | testo integrale disponibile; lette le sezioni pertinenti |
| **PZ** | lettura parziale o per parole chiave |
| **AB** | solo abstract o metadati |
| **SEC** | fonte secondaria o estratto del motore di ricerca |

**Chi ha letto:**

- **C**: il coordinatore;
- **S1**: subagente su dinamica e compensazione;
- **S2**: subagente su configurazioni con termocoppia e velocimetria;
- **S3**: subagente su sensore virtuale e alternative.

Più letture della stessa fonte non sono conferme indipendenti.

**Organizzazione:**

- Contesto letto in repository:
  - `mese3/README.md`;
  - roadmap e README della missione 04;
  - step 1 (rapporto e registro);
  - `data_inventory.md` della missione 01;
  - scouting delle misure della missione 01;
  - roadmap 02.
- **Non esistono file `AGENTS.md`** in repository.
- Il coordinatore ha letto direttamente le configurazioni decisive: VT, Delft JSC, Delft JHC, EM2C. Ha lanciato tre subagenti con un brief comune, salvato in [step2_handoff/brief_subagenti.md](step2_handoff/brief_subagenti.md).
- **Incidente:** i tre subagenti sono stati interrotti da un limite d'uso dell'API. Il coordinatore ha salvato un passaggio di consegne parziale (commit `2f8758b`, cartella [step2_handoff/](step2_handoff/)). Dopo il ripristino della quota i subagenti sono stati ripresi dal punto d'arresto e hanno consegnato.
- **Report dei subagenti:**
  - S3 non ha potuto scrivere il proprio REPORT.md (la scrittura di file è vietata ai subagenti) e ha consegnato il report come testo;
  - S1 e S2 hanno scritto REPORT.md nella scratchpad, che è effimera.

  Tutti i riferimenti usati nel rapporto sono riportati qui sotto, con DOI o URL. Nessun riferimento dipende dalle note dei subagenti.

---

## 1. Verifiche sui testi da parte del coordinatore

| Affermazione | Fonte | Esito |
|---|---|---|
| Sonda tipo B traversata; liner e transition piece scollegati; 2 Hz con std < 0,5 %; isteresi; errore transitorio entro 10 %; giunto di riferimento −30 K / −19,2 K; Tab. 6-1 (formule e 11,1 K a 1600 K); refuso sull'area della mappa | VT, OSTI 1463254, Cap. 6, pp. 188–199 | **Confermato** |
| TC tipo T da 40 AWG, τ ~0,05 s a 20 m/s (costruttore), sonde spostate dal flusso | VT, Cap. 4, pp. 81–82 | **Confermato** |
| Bead K da 0,762 mm, ~1,5 s (costruttore); guaine ceramiche; rotture; HTC falsato | VT, Cap. 4, pp. 134–138 | **Confermato** |
| PIV reagente con TiO₂ 1–2 µm, 7,4 Hz; caso R1 = φ 0,65, pilota 6 %, Re 50k | VT, Cap. 5, pp. 165–167 | **Confermato** |
| Camera in quarzo o d'acciaio; tipo S con guaina di Pt; boccola; 200 campioni su 2d; Eq. (1); T_wall alla parete; PIV solo a H₂ = 0 % | Sampat 2025, Fuel, §2.1–2.2, §3.2 | **Confermato** |
| Nu di Hilpert; ε = 0,175 da Bradley & Entwistle; WSGG; limiti di intrusività; modulo NI-9213 | Tesi Sampat 2025, §3.2.5, §6.2.2, App. C.2.1, §7.3 | **Confermato.** Diametro della sonda assente |
| Piastra con porte al posto della finestra; tipo S da 3 mm; 0,62 mm/s; 10 Hz; 71 punti; Eq. 1–3; SPIV non reagente in altre condizioni | van den Berg 2024, §2, §2.2, conclusioni | **Confermato.** Definizione di T_w assente |
| K da 1,5 mm nella ORZ attraverso una barra; < 1200 K; ≤ 5 %; non corrette; confronto con RANS | Mercier 2016, HAL, §2.5, §3.3, Fig. 6 | **Confermato** |
| ERAU: bead tipo B con la sezione di prova rimossa, misura all'ingresso; correzione radiativa | OSTI 1846098, §2.4–2.5 | **Confermato** (segnalato da S2) |
| Collis & Williams per Re < 49; bead-only insensibile fino a ~250 µm | Shaddix 2017, OSTI 1456684 | **Confermato** (segnalato da S1) |
| Lunghezza fredda 0,94 mm; Ranz–Marshall; std 237 / 385 K | NIST TN 2083r1 | **Confermato** (segnalato da S1; già nella missione 01) |
| Equazione FDS con U/4; default 1 mm, ε 0,85, tipo K; Nu = 2 + 0,6 Re^½ Pr^⅓ | FDS User Guide, righe 10516–10519; `func.f90` riga 3358 | **Confermato** (segnalato da S3) |
| Bead da 90 µm; "lags behind" | Hubbard 2022, OSTI 1887423, §3.3 | **Confermato.** La frase sui ~100 K di T_rms non è stata ricercata dal coordinatore: livello del subagente |

**Non riverificati dal coordinatore** (livello del subagente):

- Kim 2024;
- Smith 1978;
- Pousse 2009;
- Yilmaz 2008;
- Fuego;
- Nau 2018, Arndt 2021, Koren 2018;
- Dennis 2016;
- Kuang 2025;
- Labahn 2019;
- TU Berlin;
- Deng 2020;
- Zhang 2025.

---

## 2. Bibliografia ragionata

### 2.1 Configurazioni decisive

| Rif. | Fonte (DOI / URL) | Lettura | Sezioni, pagine e figure | Chi |
|---|---|---|---|---|
| VT 2018 | Ekkad et al., rapporto finale DOE/NETL DE-FE0011762, Virginia Tech, doi 10.2172/1463254, [OSTI 1463254](https://www.osti.gov/servlets/purl/1463254) | TS | Cap. 3 (p. 46), Cap. 4 (pp. 40, 79–82, 134–138), Cap. 5 (pp. 165–167, 183), Cap. 6 (pp. 188–199, Fig. 6-1–6-17, Tab. 6-1) | C, S1 |
| Sampat 2025 | Sampat, Goselink, Dave, Schrijer, Gangoli Rao, Fuel 395 (2025) 135059, [10.1016/j.fuel.2025.135059](https://doi.org/10.1016/j.fuel.2025.135059); [PDF TU Delft](https://repository.tudelft.nl/file/File_1f7eab10-17ca-49e4-8a3d-10a7b58f6c14) | TS | §2.1, §2.2 (Eq. 1), §3.2, Fig. 1, Fig. 6 | C |
| Tesi Sampat | R. P. Sampat, *Phenomenology of combustion in a jet-stabilized combustor*, tesi di dottorato TU Delft (2025), [PDF](https://repository.tudelft.nl/file/File_bd189a65-5831-4433-8cbf-4f73c2857607); record uuid:132eb207-492d-435f-9292-3bdae208b4de | TS | §3.1 (pp. 34–35), §3.2.5, §6.2.2 (Eq. 6.1), App. C.2.1 (Eq. C.1–C.5), Fig. C.4, §7.3 | C |
| van den Berg 2024 | van den Berg, Sampat, Gangoli Rao, AECS 20 (2024) 100298, [10.1016/j.jaecs.2024.100298](https://doi.org/10.1016/j.jaecs.2024.100298); [PDF](https://repository.tudelft.nl/file/File_3e5ff27b-6bbc-43d9-8e1e-0c40a3c3a845) | TS | §2, §2.2, Eq. 1–3, Tab. 1–2, conclusioni | C |
| Mercier 2016 | Mercier et al., Combust. Flame 171 (2016) 42–58, [10.1016/j.combustflame.2016.05.006](https://doi.org/10.1016/j.combustflame.2016.05.006); [HAL hal-01405801](https://hal.science/hal-01405801/document) | TS | §2.3–2.5, §3.2–3.3, Fig. 2–6 | C |
| ERAU 2022 | Martin et al., *Improving NOx Entitlement with Axial Staging*, rapporto DOE DE-FE0031227, [OSTI 1846098](https://www.osti.gov/servlets/purl/1846098) | TS | §2.4–2.5 (pp. 16–18), p. 82 circa (K esposta), C-type | S2, C |

### 2.2 Modelli della sonda e dinamica

| Rif. | Fonte | Lettura | Uso | Chi |
|---|---|---|---|---|
| NIST TN 2083r1 | Sung, Chen, Bundy, Fernandez, Hamins, *The Thermal Character of a 1 m Methanol Pool Fire*, [10.6028/NIST.TN.2083r1](https://doi.org/10.6028/NIST.TN.2083r1) | TS | §2.1–2.1.1, §3.7, Tab. 5, App. A, D, G | S1 (missione 01) |
| Shaddix 2017 | *A new method to compute the radiant correction of bare-wire thermocouples*, SAND2017-4404C, [OSTI 1456684](https://www.osti.gov/servlets/purl/1456684) (anche [OSTI 1456686](https://www.osti.gov/servlets/purl/1456686)) | TI | Eq. 1–14, Tab. 1, Fig. 3–6 | S1, S3, C |
| Kim 2024 | Kim et al., Combust. Flame (2024) 113628, [10.1016/j.combustflame.2024.113628](https://doi.org/10.1016/j.combustflame.2024.113628), OSTI 2574405 | TS | §2, Tab. 1 | S1 |
| Yilmaz 2008 | Sensors 8 (2008) 7882, [10.3390/s8127882](https://doi.org/10.3390/s8127882), OSTI 1628567 | TI | §2: τ e funzione di trasferimento | S1 |
| Smith 1978 | SAND78-8726, [OSTI 6417340](https://www.osti.gov/servlets/purl/6417340) | PZ | p. 7: silice, conduzione | S1 |
| Pousse 2009 | [arXiv 0903.4429](https://arxiv.org/abs/0903.4429) | PZ | Rivestimento BeO-Y₂O₃ su tipo B | S1 |
| Bradley & Entwistle 1961 | Br. J. Appl. Phys. 12 (1961) 708 | Citato da Sampat, non letto | ε di Pt-10Rh | — |
| Jiménez 2022 | Combust. Flame 240, [10.1016/j.combustflame.2022.112022](https://doi.org/10.1016/j.combustflame.2022.112022) | AB | Conduzione (riferimento di van den Berg) | S1 |

Letti solo come abstract o metadati (S1, via OpenAlex/OSTI):

- Miles & Gouldin, [10.1080/00102209308924108](https://doi.org/10.1080/00102209308924108);
- Tagawa et al. 1998, [10.1063/1.1149103](https://doi.org/10.1063/1.1149103);
- Tagawa & Ohta 1997, [10.1016/S0010-2180(97)00044-8](https://doi.org/10.1016/S0010-2180(97)00044-8) (solo titolo);
- Hung 2005, [10.1063/1.1847412](https://doi.org/10.1063/1.1847412);
- McLoone 2008, [10.1177/0142331208092148](https://doi.org/10.1177/0142331208092148);
- O'Reilly 2001, [10.1063/1.1384428](https://doi.org/10.1063/1.1384428);
- Kar 2004, [10.4271/2004-01-1418](https://doi.org/10.4271/2004-01-1418);
- Chandran 1985, [10.1080/00102208508960292](https://doi.org/10.1080/00102208508960292);
- Ballantyne & Moss, [10.1080/00102209708946813](https://doi.org/10.1080/00102209708946813);
- Katsuki 1987, [10.1016/0010-2180(87)90011-3](https://doi.org/10.1016/0010-2180(87)90011-3);
- Yule 1978, [10.2514/3.62378](https://doi.org/10.2514/3.62378);
- Petit 1982, [10.1088/0022-3735/15/7/017](https://doi.org/10.1088/0022-3735/15/7/017);
- Kato 2007, [10.1088/0957-0233/18/3/030](https://doi.org/10.1088/0957-0233/18/3/030);
- Collis & Williams 1959, [10.1017/S0022112059000696](https://doi.org/10.1017/S0022112059000696);
- Bradbury & Castro 1972, [10.1017/S0022112072002319](https://doi.org/10.1017/S0022112072002319);
- Bradley & Matthews 1968, [10.1243/jmes_jour_1968_010_048_02](https://doi.org/10.1243/jmes_jour_1968_010_048_02);
- Hindasageri 2013, [10.1063/1.4790471](https://doi.org/10.1063/1.4790471);
- Bahlawane 2007, [10.1063/1.2426876](https://doi.org/10.1063/1.2426876);
- Okumura 2024, [10.1016/j.fuel.2024.133331](https://doi.org/10.1016/j.fuel.2024.133331).

Solo dati bibliografici, nessun abstract:

- Lenz & Günther 1980, [10.1016/0010-2180(80)90071-1](https://doi.org/10.1016/0010-2180(80)90071-1);
- Heitor & Moreira 1993, [10.1016/0360-1285(93)90017-9](https://doi.org/10.1016/0360-1285(93)90017-9);
- Yanagi & Mimura 1981, [10.1016/S0082-0784(81)80107-5](https://doi.org/10.1016/S0082-0784(81)80107-5);
- Kent 1970, [10.1016/S0010-2180(70)80040-2](https://doi.org/10.1016/S0010-2180(70)80040-2).

### 2.3 Velocimetria con termocoppie: altri casi

| Rif. | Fonte | Lettura | Chi |
|---|---|---|---|
| Olivani 2007 | Exp. Therm. Fluid Sci. 31 (2007) 427, [10.1016/j.expthermflusci.2006.05.003](https://doi.org/10.1016/j.expthermflusci.2006.05.003); [OSTI 20864947](https://www.osti.gov/biblio/20864947) | AB | S2 |
| Coghe 2004 | ETFS 28 (2004), [10.1016/j.expthermflusci.2003.12.007](https://doi.org/10.1016/j.expthermflusci.2003.12.007) | SEC | S2 |
| Solero 2017 | IJETR 7(2) | SEC (estratto del motore di ricerca: bead Pt/Pt-13Rh da 0,3 mm; non verificato) | S2 |
| Bouziane 2009 | CFM 2009, [HAL hal-03378716](https://hal.science/hal-03378716v1/file/bitstream_66018.pdf) | TS (pp. 2–4) | S2 |
| Archer 2005 | Tesi UMD, hdl 1903/2949; OSTI ETDE 20013560 | AB | S2 |
| Heitor 1985 | Exp. Fluids 3 (1985) 323, [10.1007/BF01830191](https://doi.org/10.1007/BF01830191); OSTI ETDE 6435338 | PZ (abstract, simboli, riferimenti) | S2 |
| Heitor 1987 | JFM | AB | S2 |
| Ferrão & Heitor 1998 | Exp. Fluids 24, [10.1007/s003480050189](https://doi.org/10.1007/s003480050189) | AB. La corrispondenza DOI–record ETDE era già in dubbio nello step 1 | S2 |
| TU Berlin | [arXiv 1605.03110](https://arxiv.org/abs/1605.03110) | TS | S2 |
| Deng 2020 | ACS Omega, PMC7178801 (testo via NCBI BioC API) | TS | S2 |
| Zhang 2025 | PMC12591111 | TS | S2 |
| XJTU | PCI 39 (2023), [10.1016/j.proci.2022.08.066](https://doi.org/10.1016/j.proci.2022.08.066) | SEC | S2 |
| OSTI 775039 | (S2, uso marginale) | PZ | S2 |

### 2.4 Sensore virtuale, alternative, assimilazione

| Rif. | Fonte | Lettura | Chi |
|---|---|---|---|
| FDS | [github.com/firemodels/fds](https://github.com/firemodels/fds), tag FDS-6.11.2 e master 46536679: User Guide (sezione Thermocouples), Validation Guide (Waterloo), `dump.f90`, `func.f90`, `read.f90`, `cons.f90` | TS | S3, C |
| Fuego | SAND2022-12437, [OSTI 1888153](https://www.osti.gov/servlets/purl/1888153), §2.15 | TS | S3 |
| Hubbard 2022 | ASME J. Heat Mass Transfer, [10.1115/1.4054204](https://doi.org/10.1115/1.4054204); [OSTI 1887423](https://www.osti.gov/servlets/purl/1887423) | TS (§3.3) | S3, C |
| Brundage et al. | J. Fire Sci., [10.1177/0734904110386188](https://doi.org/10.1177/0734904110386188) | AB | S3 |
| Braun, Lu, Paniagua | GT2017-64669, [10.1115/GT2017-64669](https://doi.org/10.1115/GT2017-64669) | AB | S3 |
| Blevins & Pitts | NISTIR 6310 | PZ | S3 |
| Nau 2018 | JEGTP 141, [10.1115/1.4040716](https://doi.org/10.1115/1.4040716); [elib 123633](https://elib.dlr.de/123633/1/2018_Nau_GTP_author_final.pdf) | TS | S3 |
| Arndt 2021 | PCI 38, [10.1016/j.proci.2020.06.088](https://doi.org/10.1016/j.proci.2020.06.088); [elib 141835](https://elib.dlr.de/141835/1/2021_Arndt_PCI_author_final.pdf) | PZ | S3 |
| Koren 2018 | FTaC 101, [10.1007/s10494-018-9895-5](https://doi.org/10.1007/s10494-018-9895-5); [hal-01744260](https://hal.science/hal-01744260/document) | PZ | S3 |
| Dennis 2016 | [10.1016/j.combustflame.2016.02.033](https://doi.org/10.1016/j.combustflame.2016.02.033); [OSTI 1533658](https://www.osti.gov/servlets/purl/1533658) | PZ/TS | S3 |
| Giezendanner-Thoben 2005 | Appl. Opt., [10.1364/AO.44.006565](https://doi.org/10.1364/AO.44.006565) | AB | S3 |
| Zhou 2007 (TDLAS) | AIAA J, [10.2514/1.26624](https://doi.org/10.2514/1.26624) | AB | S3 |
| Kuang 2025 (TDLAS) | Sensors, [10.3390/s25072256](https://doi.org/10.3390/s25072256); PMC11991275 | PZ | S3 |
| Liu 2017 | MST, [10.1088/1361-6501/aa5aee](https://doi.org/10.1088/1361-6501/aa5aee) | AB | S3 |
| Powers 2022 (FRS) | preprint, [10.21203/rs.3.rs-2370966/v1](https://doi.org/10.21203/rs.3.rs-2370966/v1) | AB | S3 |
| PIV termografica | Fond 2012 (AB); Abram 2013, [10.1007/s00340-013-5411-8](https://doi.org/10.1007/s00340-013-5411-8) (PZ); Fan 2017, [10.1007/s00348-017-2313-2](https://doi.org/10.1007/s00348-017-2313-2) (PZ); Yin 2017, [elib 113064](https://elib.dlr.de/113064/1/2017_Yin_CnFComm_author_final.pdf) (TS) | vedi | S3 |
| Lin & Wang | Fire Technol., [10.1007/s10694-016-0619-x](https://doi.org/10.1007/s10694-016-0619-x) | AB | S3 |
| Jahn, Rein, Torero 2011 | IAFSS 10 | AB | S3 |
| Koo et al. 2010 | Fire Saf. J., [10.1016/j.firesaf.2010.02.003](https://doi.org/10.1016/j.firesaf.2010.02.003) | solo titolo | S3 |
| Labahn 2019 | FTaC, [10.1007/s10494-019-00093-1](https://doi.org/10.1007/s10494-019-00093-1); [OSTI 1772036](https://www.osti.gov/pages/servlets/purl/1772036) | PZ | S3 |
| Welsh & Rubini 1997 | IAFSS 5, doi 10.3801/IAFSS.FSS.5-1009 | SEC | S3 |
| Brübach 2013; Abram 2018 | PECS | non accessibili | — |

### 2.5 Altri file scaricati, non usati

- Tesi Link (TU Delft, swirl CH₄/H₂ parzialmente premiscelato con PIV): [PDF](https://repository.tudelft.nl/file/File_b0ecc49d-bba9-478f-9050-0c421c9539f8). Ricerca per parole chiave: solo analizzatore allo scarico, nessuna termocoppia in camera emersa.
- Paper Link: File_53187369-4937-4622-b2f2-6e6ff9d776bf. Non letto.

---

## 3. Query principali

**Coordinatore:**

- OpenAlex per DOI (VT, Sampat, van den Berg, Mercier);
- WebSearch:
  - "repository.tudelft.nl Sampat … jet stabilized combustor";
  - "van den Berg enclosed jet-in-hot-coflow … pdf";
- ricerche per parole chiave nei PDF (`thermocouple`, `emissivity`, `Nusselt`, `seed`, `Hz`).

**S1:**

- Shaddix critical review; Hung/McLoone a due termocoppie;
- catalisi del Pt in NH₃ e rivestimenti (4 varianti);
- Tagawa Ohta su J-STAGE; dual thermocouple in fiamme turbolente;
- circa 40 titoli su OpenAlex;
- OSTI: "thermocouple flame", "compensated thermocouple", "thermocouple time constant", "thermocouple catalytic", autore Gouldin;
- arXiv.

**S2:**

- Olivani, Solero, Coghe thermocouple; titolo IJETR; Pt/Pt-13 %Rh 0,3 mm;
- Archer micro-thermocouple; filo di Heitor 1985;
- SJTU Liu; Cardiff R-type; XJTU PCI;
- OpenAlex per autore (Solero, Cozzi, Archer);
- 6 query OSTI; 5 ricerche PMC.

**S3:**

- virtual thermocouple LES; thermocouple model MILD furnace;
- two-thermocouple DNS; EnKF LES flame thermocouple;
- Jahn, Rein, Torero; TDLAS in un combustore swirl;
- circa 15 ricerche OpenAlex; 4 query OSTI; clone del repository FDS.

---

## 4. Accessi falliti

| Host | Errore | Che cosa non è stato letto |
|---|---|---|
| re.public.polimi.it, politesi | CONNECT 403 / 000 | PoliMi: Solero 2017, tesi |
| drum.lib.umd.edu, api.drum | 403 | Archer 2005 |
| sciencedirect.com | 403 | Olivani 2007, Abram 2018, Brübach 2013 |
| link.springer.com (PDF) | Challenge JS | Heitor 1985 (testo), Ferrão 1998 |
| pubs.aip.org | 000 / 403 | Tagawa 1998, Hung 2005, Hindasageri 2013, Liu 2024 PoF |
| mural.maynoothuniversity.ie | 403 / ENOTFOUND | Hung, McLoone (Kee) |
| jstage.jst.go.jp, eprints.lib.hokudai.ac.jp | 403 | Tagawa |
| escholarship.org, ntrs.nasa.gov, spiral.imperial.ac.uk, cambridge.org | 403 / 000 | Lisbona/Imperial, NASA |
| iopscience (PDF) | Paywall | Petit 1982, Kato 2007 |
| core.ac.uk, PMC (pagine), mdpi.com, zenodo.org, orca.cardiff.ac.uk | Cloudflare / captcha / 403 | Copie alternative; Cardiff |
| publications.iafss.org, nist.gov, tsapps.nist.gov | 403 | Welsh & Rubini; documenti NIST |
| api.archives-ouvertes.fr, theses.hal.science, ifp.hal.science | 403 | Ricerche su HAL (i PDF diretti funzionavano) |
| export.arxiv.org | DNS | API arXiv (le pagine dirette funzionavano) |
| api.crossref.org | 429 | Metadati (sostituiti da OpenAlex) |
| api.github.com | 403 | Sostituito da git clone |
| repository.tudelft.nl, File_072bee9b… (GT2022-81808) e File_12cf7b04… | *«campus only property … under embargo or closed»* | Sampat GT2022; supplementi |
| Appendice supplementare B.1 di Fuel 2025 | Non reperita | Sostituita dalla tesi, App. C.2.1 |

**Controllo dei permessi della sessione.** Un subagente (S2) ha chiesto di inviare uno User-Agent da browser per superare un 403. La richiesta è stata negata e non è stata aggirata. S3 non ha potuto scrivere un file di report: non ha aggirato il divieto e ha consegnato il report come testo.

---

## 5. Piste di dataset incontrate (per l'agente dataset; non ispezionate)

- Mappe TC e PIV di Virginia Tech (OSTI 1463254): disponibilità [M].
- TU Delft: dati di Fuel 2025 e AECS 2024; possibile 4TU.ResearchData [M].
- Waterloo, pool fire di metanolo (Weckman & Strong 1996): termocoppia + LDA. Usato come caso di validazione FDS: i dati sono nel repository `fds-repo`/`exp` [I, da verificare].
- Nessun dataset trovato con serie di termocoppie + velocità in combustori a gas (S2).

---

## 6. File temporanei

I file seguenti sono nella scratchpad della sessione, effimera, e si perdono a fine sessione. Si possono riscaricare dagli URL dei §2.

- `me/vt/vt.pdf|txt`;
- `me/delft/*.pdf|txt`;
- `me/em2c/m.pdf`;
- `me/erau/e.pdf`;
- `sa1/dl_*`, `sa2/d_*`, `sa3/dl_*`;
- `sa3/fds/repo`.

**Calcolo illustrativo di τ:** copiato in repository ([step2_handoff/stima_tau_correlazioni.py](step2_handoff/stima_tau_correlazioni.py)). Requisiti: `pip install cantera numpy`, quindi `python -I stima_tau_correlazioni.py`.
