# Missione 04 — Step 1: fonti, ricerche e registro di accesso

**Documento collegato a:** [step1_pratica_sperimentale_temperatura.md](step1_pratica_sperimentale_temperatura.md).  
**Data:** 26 novembre 2026.  
**Stato:** consegna da revisionare; non approvata.  
**Nota:** nessun PDF o copia integrale è stato aggiunto alla repository. Le fonti sono collegate tramite DOI o URL.

**Livello di lettura:**

- **TI:** testo integrale, oppure sezioni pertinenti lette per intero;
- **PZ:** lettura parziale o ricerca per parole chiave nel testo;
- **AB:** solo abstract o metadati;
- **SEC:** fonte secondaria, oppure estratto prodotto dal motore di ricerca, da verificare.

---

## 1. Metodo

### 1.1 Organizzazione

Contesto letto in repository:

- `mese3/README.md`;
- README e roadmap della missione 04;
- roadmap 02;
- matrice, inventario e scouting della missione 01;
- voce del diario del 25 novembre con le indicazioni di Linzheng e Jason.

Non risultano file `AGENTS.md` nella repository.

La ricerca è stata divisa in **sette filoni paralleli** (subagenti). Tutti seguivano lo stesso brief:

- etichette D/I/M;
- posizione precisa nella fonte;
- "non riportato" distinto da "non misurato";
- simultaneità da dimostrare, non da presumere.

| Filone | Ambito | Esito |
|---|---|---|
| A | Combustori modello swirl per turbine a gas: DLR, EM2C, KIT, ETH, GE/Toronto, Cambridge, MIT | Completo per DLR ed EM2C; parziale per gli altri; MIT non verificato |
| B | Rig swirl ad ammoniaca e origine sperimentale di RealPDEBench | Quasi solo abstract, per blocchi d'accesso |
| C | Fiamme da laboratorio aperte: PIV + termometria del gas | Completo per i casi accessibili da OSTI, DLR, Lund e TU Delft |
| D | Motori ottici: TCC-III, Darmstadt, Sandia, IFPEN | Completo per TCC-III e Darmstadt; abstract per LIF+PIV e PIV termografica |
| E | Strumenti: termocoppie, fosfori, IR, finestre, termometria ottica nei combustori | Completo; review principali non accessibili |
| F | Scopi dichiarati e assenze documentate | Completo; review di Hochgreb e Barlow non accessibili |
| G | RealPDEBench dalle fonti originali | Completo per paper, card e codice; rig d'origine solo abstract |

**Incidenti di percorso:**

- Tutti i filoni sono stati interrotti da un limite d'uso dell'API e poi ripresi dal punto in cui si erano fermati.
- Durante il lavoro è stata approvata un'estensione della lista di host consentiti dell'ambiente: ORCA, HAL, KAUST, Lund LUP, MIT DSpace, AIP, Europe PMC, Semantic Scholar. HAL è diventato leggibile. Molti altri host hanno continuato a rispondere con pagine anti-bot, 429 o errori DNS (§5).
- Il controllo dei permessi della sessione ha negato alcune richieste: un follow-up del filone B e un download diretto da Lund LUP. Non sono state aggirate.

### 1.2 Integrazione del coordinatore

Ho letto tutti gli appunti dei filoni. Le cose che contano:

- I casi sono stati unificati sotto identificativi comuni (GT, NH, LF, EN, RP).
- Le schede in conflitto sono state confrontate: quarzo PRECCINSTA da 1,5 o 1,7 mm; Nau JEGTP datato 2018 o 2019. Non toccano le conclusioni.
- Ho verificato a campione sui testi originali le affermazioni decisive (§2).

---

## 2. Verifiche puntuali sui testi originali (coordinatore)

Ho scaricato i PDF in una cartella temporanea (nessun file in repository) e cercato le frasi.

| Affermazione | Fonte verificata | Esito |
|---|---|---|
| *«it is not possible to perform Raman and PIV measurements simultaneously»*; zone escluse con h < 8 mm e r > 27 mm; T dalla legge dei gas ideali; GT spesso senza condizioni al contorno | Arndt et al. 2015, manoscritto [elib 96436](https://elib.dlr.de/96436/1/Arndt2015ExpFluids_manuscript.pdf) | **Confermato** testualmente |
| *«Consideration of heat loss to surfaces have proven critical in improving large eddy simulations»*; calibrazione con termocoppia in forno | Yin, Nau, Meier 2017, [elib 112323](https://elib.dlr.de/112323/1/2017_Yin_ETFS_author_final.pdf) | **Confermato** |
| Termocoppie *«intrusive … only point measurements»*; *«PIV measurements have not been performed so far for y < 0 mm»* | Nau et al. 2019, [elib 123633](https://elib.dlr.de/123633/1/2018_Nau_GTP_author_final.pdf) | **Confermato** |
| *«If available, measured temperatures can be used as boundary conditions … just not available»*; preriscaldamento *«not precisely measured»*; Yin misurato *«in operating conditions different»* | Agostinelli et al. 2021, preprint [Zenodo 6450237](https://zenodo.org/records/6450237) | **Confermato** |
| FRS e PIV *«frame straddle»*; segnale FRS *«directly proportional to temperature»* | McManus & Sutton 2022, [OSTI 2420963](https://www.osti.gov/servlets/purl/2420963) | **Confermato** |
| *«~6 minutes»* per l'equilibrio; *«flame temperature or the burned gas was not measured»* | Wang et al., [arXiv 2210.04181](https://arxiv.org/abs/2210.04181) | **Confermato** |
| *«This additional measurement was not performed»* riferito alla temperatura del pistone | Henrion et al. 2022, [OSTI 1976995](https://www.osti.gov/servlets/purl/1976995) | **Corretto.** La frase riguarda la trasmissione della finestra a caldo. Per il pistone il testo dice *«challenges left unaddressed during this study»*. Rapporto §8.1 e §12 aggiornati |
| Associazioni DOI–titolo–autori di 10 lavori chiave | OpenAlex API | **Confermate** |

Non ho riverificato personalmente HAL (Mercier, Kraus, Koren, Schiffmann) né le pagine Springer (Gövert, Kumar, Welch, Chu). Per questi la lettura integrale è dei filoni.

- **Mercier 2016:** citato in modo coerente da tre filoni indipendenti (A, E, F).

---

## 3. Bibliografia ragionata

### 3.1 Combustori confinati swirl e affini

| Rif. | Fonte | Lettura | Sezioni o pagine usate | Per che cosa |
|---|---|---|---|---|
| Arndt 2015 | C. M. Arndt et al., *Experimental analysis of thermo-acoustic instabilities in a generic gas turbine combustor by phase-correlated PIV, chemiluminescence, and laser Raman scattering measurements*, Exp. Fluids 56 (2015) 69. [10.1007/s00348-015-1929-3](https://doi.org/10.1007/s00348-015-1929-3); [elib 96436](https://elib.dlr.de/96436/) | TI (manoscritto) | §1, §2.1–2.6, Results | Raman ↔ PIV non simultanei; zone escluse; uso acustico di T |
| Arndt 2021 | C. M. Arndt, P. Nau, W. Meier, *Characterization of wall temperature distributions in a gas turbine model combustor measured by 2D phosphor thermometry*, PCI 38 (2021) 1867–1875. [10.1016/j.proci.2020.06.088](https://doi.org/10.1016/j.proci.2020.06.088); elib 141835 | TI | Intro, metodo, Tab. 2, Fig. 1/6 | Pareti come condizioni al contorno; strisce sulle finestre; flusso termico ≈10 % |
| Meier 2010 | W. Meier, I. Boxx, M. Stöhr, C. D. Carter, Exp. Fluids 49 (2010) 865; elib 65267 | TI | §2.1, §2.5, §3.1 | Raman nel GTMC e in PRECCINSTA |
| Sadanandan 2009 | Manoscritto, elib 60286 | PZ | §2 | Ventilatori per il quarzo |
| Dennis 2016 | C. N. Dennis et al., fs-CARS a 5 kHz nel GTMC, Combust. Flame (2016). [10.1016/j.combustflame.2016.02.033](https://doi.org/10.1016/j.combustflame.2016.02.033); [OSTI 1533658](https://www.osti.gov/servlets/purl/1533658) | PZ | §2.3, §3.1–3.2 | CARS a 5 kHz; finestre ~1000 K; PIV da campagne precedenti |
| Yin 2017 | Z. Yin, P. Nau, W. Meier, Exp. Therm. Fluid Sci. 82 (2017) 50–57. [10.1016/j.expthermflusci.2016.11.004](https://doi.org/10.1016/j.expthermflusci.2016.11.004); [elib 112323](https://elib.dlr.de/112323/) | TI | §1–4 | Bluff body a 1 kHz; PVC; simultaneità con OH-PLIF |
| Nau 2019 | P. Nau et al., J. Eng. Gas Turbines Power 141 (2019) 041021. [10.1115/1.4040716](https://doi.org/10.1115/1.4040716); [elib 123633](https://elib.dlr.de/123633/) | TI | pp. 2, 4–10 | Pareti a 8 bar; eccitazione dal retro; pro e contro delle termocoppie |
| Nau 2017 | P. Nau, Z. Yin, K. P. Geigle, W. Meier, Appl. Phys. B 123 (2017) 279. [10.1007/s00340-017-6856-y](https://doi.org/10.1007/s00340-017-6856-y); [elib 115444](https://elib.dlr.de/115444/) | TI (§1–3) | §1–3.3 | Finestre interne ed esterne a 3 bar; *«had to rely on estimated wall temperatures»* |
| Nau 2024 | P. Nau et al., PCI 40 (2024) 105357. [10.1016/j.proci.2024.105357](https://doi.org/10.1016/j.proci.2024.105357); [elib 209840](https://elib.dlr.de/209840/) | TI | §2.1–3.2 | Fosfori con fibra; T_ex riportata senza metodo |
| Stöhr 2018 | M. Stöhr et al., J. Eng. Gas Turbines Power (2018). [10.1115/1.4037724](https://doi.org/10.1115/1.4037724); elib 116931 | PZ | §2.3, §5.1 | PIV e OH-PLIF a 10 kHz; densità dalla luce Mie (QLS) |
| Agostinelli 2021 | P. W. Agostinelli et al., Combust. Flame 234 (2021) 111728. [10.1016/j.combustflame.2021.111728](https://doi.org/10.1016/j.combustflame.2021.111728); [Zenodo 6450237](https://zenodo.org/records/6450237) | TI (Intro, §2–3.1, App. A) | Intro, §2, §3.1 | Motivi delle assenze; condizioni al contorno di PRECCINSTA |
| Gövert 2018 | S. Gövert et al., Flow Turb. Combust. 100 (2018) 503–534. [10.1007/s10494-017-9848-4](https://doi.org/10.1007/s10494-017-9848-4) | TI | §3.1–3.2, Tab. 1 | *«wall temperature was not measured»*; LDV e Raman a φ diversi |
| Kumar 2023 | A. Kumar et al., Flow Turb. Combust. 111 (2023) 995–1028. [10.1007/s10494-023-00452-z](https://doi.org/10.1007/s10494-023-00452-z) | TI | §2, p. 1000, §4.1 | *«not available for the case considered»*; PIV solo a freddo |
| Pries 2026 | Pries & Fiolitakis, AECS 27 (2026) 100524; elib 225168 | PZ | §3.6 | Pareti misurate riusate fuori dalla loro condizione |
| Karpowski 2022 | GT2022-83810; [arXiv 2308.05431](https://arxiv.org/abs/2308.05431) | PZ | §2–4 | LES con pareti misurate e CHT |
| Mercier 2016 | R. Mercier et al., Combust. Flame 171 (2016) 42–58. [10.1016/j.combustflame.2016.05.006](https://doi.org/10.1016/j.combustflame.2016.05.006); [HAL hal-01405801](https://hal.science/hal-01405801) | TI | §1, §2.1–2.5, Fig. 2–6, §3.2 | Caso EM2C completo: barre, fosfori, ORZ, motivazioni |
| Koren 2018 | C. Koren, R. Vicquelin, B. Gicquel, Flow Turb. Combust. 101 (2018) 77. [10.1007/s10494-018-9895-5](https://doi.org/10.1007/s10494-018-9895-5); hal-01744260 | PZ | §3, conclusioni | Richiesta di misure esterne e allo scarico |
| Guiberti 2015 | T. F. Guiberti et al., PCI 35 (2015) 1385. [10.1016/j.proci.2014.06.016](https://doi.org/10.1016/j.proci.2014.06.016) | SEC | — | Originale EM2C (HAL anti-bot) |
| Kraus 2017 | C. Kraus et al., J. Eng. Gas Turbines Power 139 (2017) 051503. [10.1115/1.4035143](https://doi.org/10.1115/1.4035143); hal-01688648 | TI | §1, §2, §4, §5.1 | Plenum 450/350 K; colori di ricottura; frequenze |
| Kraus 2018 | C. Kraus, L. Selle, T. Poinsot, Combust. Flame 191 (2018) 239–251. [10.1016/j.combustflame.2018.01.007](https://doi.org/10.1016/j.combustflame.2018.01.007); hal-01730351 | PZ (§1, §4) | §1, Tab. 1 | Pratiche sulle condizioni al contorno termiche; motivi delle assenze |
| Poinsot 2017 | T. Poinsot, PCI 36 (2017) 1–28. [10.1016/j.proci.2016.05.007](https://doi.org/10.1016/j.proci.2016.05.007); hal-01502419 | PZ (§4.1, §5) | §4.1, Fig. 10 | Pareti e termoacustica |
| Gicquel 2012 | L. Gicquel, G. Staffelbach, T. Poinsot, PECS 38 (2012) 782–817. [10.1016/j.pecs.2012.04.004](https://doi.org/10.1016/j.pecs.2012.04.004) | PZ (parole chiave) | §4.1 | Contesto |
| Papafilippou 2024 | Flow Turb. Combust. 113 (2024) 1189–1214. [10.1007/s10494-024-00558-y](https://doi.org/10.1007/s10494-024-00558-y) | TI | §2, §4 | Pareti in quarzo a temperatura ambiente nella LES |
| Bonciolini 2019 | G. Bonciolini et al., PCI 37 (2019). [arXiv 1807.02428](https://arxiv.org/abs/1807.02428) | TI | Fig. 1 | Termocoppia di parete come variabile di stato |
| Passarelli | GE/Toronto, 1 MPa. [arXiv 2006.09948](https://arxiv.org/abs/2006.09948) | PZ | §2 | Preriscaldamento *«approximately 500 K»*, nessuno strumento |
| Chen (annulare Cambridge) | [arXiv 2111.00731](https://arxiv.org/abs/2111.00731) | PZ | §2.3 | Pareti a 1000 K da LES precedenti |
| Tay-Wo-Chong & Polifke 2013; Han/Morgans 2017 | J. Eng. Gas Turbines Power [10.1115/1.4007734](https://doi.org/10.1115/1.4007734); Combust. Flame [10.1016/j.combustflame.2017.06.018](https://doi.org/10.1016/j.combustflame.2017.06.018) | AB | — | Effetto delle pareti su FTF e ciclo limite |
| Stella 2001; Vanselow 2019 | Exp. Fluids [10.1007/s003480000151](https://doi.org/10.1007/s003480000151); [10.1007/s00348-019-2795-1](https://doi.org/10.1007/s00348-019-2795-1) | AB | — | Termoforesi; beam steering nella PIV |

### 3.2 Ammoniaca e RealPDEBench

| Rif. | Fonte | Lettura | Uso |
|---|---|---|---|
| Hu 2026 (RealPDEBench) | Hu et al., ICLR 2026. [arXiv 2601.01829](https://arxiv.org/abs/2601.01829) v2; card HF `AI4Science-WestlakeU/RealPDEBench`; [sito](https://realpdebench.github.io); codice GitHub (clone superficiale, HEAD 469217e) | TI | §3.2 p. 5; App. B.2 pp. 26–27; B.3 pp. 27–30; B.4 pp. 30–31; D.2 p. 34; Tab. 7; Fig. 2d, 12 |
| Liu 2024 PoF | C. Liu, H. Yang, C. Ruan, L. Yu, X. Lü, Phys. Fluids 36 (2024) 034123. [10.1063/5.0196764](https://doi.org/10.1063/5.0196764) | AB | Candidato rig d'origine; OH* + PIV simultanee |
| Liu 2024 Fuel | Fuel 357 (2024) 129796. [10.1016/j.fuel.2023.129796](https://doi.org/10.1016/j.fuel.2023.129796) (OA, ma 403 da ScienceDirect; Lund LUP non raggiungibile) | AB | Rig SJTU |
| Viswamithra 2022 | [arXiv 2206.12015](https://arxiv.org/abs/2206.12015) | TI | NH-LSU: §2.2, §2.3, §4.1–4.2 |
| Cardiff HPOC (99 % NH₃ crackizzata) | Energy & Fuels, PMC12926937 | TI | Preriscaldamento; temperatura di zona di fiamma come condizione incerta |
| Mashruk 2023 (PROCI 39); Davies 2025 (C&F 274, tesi ORCA 183189); Mashruk 2022 (IJHE) | Cardiff | AB/SEC | Termocoppie K/R su faccia del bruciatore e RZ [SEC] |
| Hayakawa 2017; Okafor 2021 (Tohoku); Zhang 2021/2023 (XJTU); Khateeb 2020/2021, Wang 2023 (KAUST); Ma 2026 (BIT) | Vedi appunti del filone B | AB | Pareti isolate o raffreddate; OH/NO-PLIF; T non determinabile |

### 3.3 Fiamme da laboratorio

| Rif. | Fonte | Lettura | Uso |
|---|---|---|---|
| McManus & Sutton 2022 | PCI 39 (2022). [10.1016/j.proci.2022.07.052](https://doi.org/10.1016/j.proci.2022.07.052); [OSTI 2420963](https://www.osti.gov/servlets/purl/2420963) | TI | FRS + PIV simultanee (§2, p. 3) |
| Sutton, rapporto DOE 2022 | [OSTI 1887222](https://www.osti.gov/servlets/purl/1887222) | PZ | FRS + PIV + LII; soot che impedisce l'FRS |
| McManus & Sutton 2020 | Exp. Fluids. [10.1007/s00348-020-02973-z](https://doi.org/10.1007/s00348-020-02973-z) | AB | Sfide della combinazione FRS + PIV |
| Most, Dinkelacker, Leipertz 2002 | PCI 29 (2002) 2669–2677 | AB/SEC | Precedente FRS + PIV |
| Petersson 2007; tesi Petersson 2014 | Appl. Opt. 46 (2007) 3928, [10.1364/AO.46.003928](https://doi.org/10.1364/AO.46.003928); [tesi Lund](https://lup.lub.lu.se/search/files/3031390/4255515.pdf) | TI (articolo nella tesi) | FRS + PIV fallita; polvere e breakdown |
| Omrane 2008 | Appl. Phys. B 92 (2008) 99 (nella tesi Petersson) | TI | PIV con fosfori, non reagente |
| Yin et al. 2017 (TPIV) | Combust. Flame 184 (2017) 249–251; [elib 113064](https://elib.dlr.de/113064/1/2017_Yin_CnFComm_author_final.pdf) | TI | Limite di 900–1000 K |
| Frank 2021 | J. Chem. Phys. 154 (2021); [OSTI 1772034](https://www.osti.gov/servlets/purl/1772034) | PZ | Mie contro Rayleigh; CARS con particelle (§3, §4.1) |
| Bohlin 2017 | PCI 36 (2017); [OSTI 1251595](https://www.osti.gov/servlets/purl/1251595) | TI | CARS vicino a parete; pareti non misurate |
| Schneider 2019 (TSF-H₂) | PCI 37; OSTI 1497651 | PZ (§2) | Raman/Rayleigh a Sandia |
| Wabel 2019 | AIAA; OSTI 1602154 | PZ | Campagne separate |
| van den Berg 2024 | AECS 20 (2024) 100298. [10.1016/j.jaecs.2024.100298](https://doi.org/10.1016/j.jaecs.2024.100298) | TI (§2) | Termocoppia tipo S traversata; PIV a freddo |
| Mazellier et al. | [arXiv 1004.0074](https://arxiv.org/abs/1004.0074) | PZ (§3) | Isoterma di 500 K dall'olio d'oliva |
| Rivera Lizarralde et al. | [arXiv 2510.25109](https://arxiv.org/abs/2510.25109) | PZ (§2.3) | Isoterma DEHS |
| Wang et al. (SJTU) | [arXiv 2210.04181](https://arxiv.org/abs/2210.04181) | TI | Assenza esplicita; T da modello |
| De, Zhu, Acharya | [arXiv 2103.02792](https://arxiv.org/abs/2103.02792) | TI | T non riportata |
| TNF12, TNF13 | [OSTI 1242775](https://www.osti.gov/servlets/purl/1242775); OSTI 1505369 | TI (sintesi) / PZ (atti) | Condizioni al contorno termiche dei bruciatori TNF; richiesta di condizioni coincidenti |
| Kildare 2024 | PCI 40; OSTI 2577073 | PZ | Precedente di T ricostruita con ML da PLIF (nessuna PIV) |
| Euler 2014; Jainski 2017/2018; Kosaka 2018; Pfadler 2007/2008; Fond 2012; Abram 2013/2018 | Vedi appunti dei filoni C ed E | AB/SEC | Fosfori sul bluff body, SWQ, PIV condizionata, PIV termografica |

### 3.4 Motori ottici

| Rif. | Fonte | Lettura | Uso |
|---|---|---|---|
| Schiffmann 2016 | OGST 71 (2016). [10.2516/ogst/2015028](https://doi.org/10.2516/ogst/2015028); HAL hal-01707465 | TI | §1.1 p. 4; §1.2; §2.1 p. 8; Tab. A1–A2 |
| Henrion 2022 | Combust. Flame (2022) 112083. [10.1016/j.combustflame.2022.112083](https://doi.org/10.1016/j.combustflame.2022.112083); [OSTI 1976995](https://www.osti.gov/servlets/purl/1976995) | TI | §2.1–2.2 |
| Wu, Patel, Ameen 2022 | Flow Turb. Combust.; OSTI 2427277 | PZ | Pareti a 400 K e port a 350 K imposti |
| Welch 2023 | Flow Turb. Combust. 110 (2023) 185–208. [10.1007/s10494-022-00347-5](https://doi.org/10.1007/s10494-022-00347-5) | TI | §2.1, Tab. 1, Fig. 1a, §3.1 |
| Chu 2023 | Flow Turb. Combust. 110 (2023) 3–29. [10.1007/s10494-022-00353-7](https://doi.org/10.1007/s10494-022-00353-7) | TI | Tab. 1, pp. 7–8 |
| Schmidt 2021 | Flow Turb. Combust. 106 (2021) 597–611. [10.1007/s10494-020-00147-9](https://doi.org/10.1007/s10494-020-00147-9) | TI | §2.1–2.2, Tab. 1 |
| Ding 2019 | PCI 37; [arXiv 2009.01739](https://arxiv.org/abs/2009.01739) | TI | pp. 5, 7, 15 |
| Dec & Hwang 2009 | SAE 2009-01-0650. [10.4271/2009-01-0650](https://doi.org/10.4271/2009-01-0650); OSTI 1142409 | TI | pp. 1, 5, 7–9 |
| Sandia SAND2021-13491C | OSTI 1894434 | TI | p. 9: T bulk dalla legge dei gas, parete a 90 °C assunta |
| Reggeti 2022 | Meas. Sci. Technol. 33 125201; OSTI 2339639 | PZ (p. 3) | Pirometria del soot |
| Baum 2014; Peterson 2013/2014/2015; Ding 2017; Kopf 2023; Wu/Alzuabi/Sick 2020; Ye 2026 | Vedi appunti del filone D | AB | Database Darmstadt; LIF di toluene + PIV; fosfori sul pistone; PIV termografica |

### 3.5 Strumenti e review

| Rif. | Fonte | Lettura | Uso |
|---|---|---|---|
| Shaddix 2017 | SAND2017-4404C, [OSTI 1456684](https://www.osti.gov/servlets/purl/1456684) | TI | Correzione radiativa e conduttiva; sensibilità ai gradienti; uso delle termocoppie come riferimento |
| Shaddix 1999 | HTD99-282, OSTI 20002672 | AB | — |
| Heitor & Moreira 1993 | PECS 19 (1993) 259–278. [10.1016/0360-1285(93)90017-9](https://doi.org/10.1016/0360-1285(93)90017-9) | SEC | Sonde in fiamma (da recuperare) |
| Ballantyne & Moss 1977; Chandran 1985; Miles & Gouldin 1993; Heitor, Taylor, Whitelaw 1985; Tagawa & Ohta 1997 | — | AB/SEC | Risposta dinamica; compensazione; LDA + termocoppia |
| Yilmaz 2008 | Sensors 8 (2008) 7882 (PMC3790995) | PZ | τ di termocoppie inguainate (contesto incendi) |
| Rodrigues 2018 | J. Eng. Gas Turbines Power; elib 123635 | PZ | Quarzo opaco all'IR; raffreddamento delle finestre |
| Dues 2018 | Simposio di Lisbona; elib 121147 | TI | FRS: stesso accesso ottico della PIV; limiti del Rayleigh nei flussi interni |
| Giezendanner-Thoben 2005 | Appl. Opt. 44 (2005) 6565 | AB | Two-line OH-PLIF |
| Review non accessibili | Hochgreb PCI 37 (2019); Barlow PCI 31 (2007); Dreizler & Böhm PCI 35 (2015); Masri PCI 38 (2021); Driscoll PECS (2020); Aldén PCI 33 (2011); Brübach PECS 39 (2013); Abram PECS 64 (2018); Eckbreth (libro); Kohse-Höinghaus & Jeffries (libro) | — | Da recuperare |

### 3.6 Fonti interne riusate

- [Scouting delle misure, missione 01](../01_observation_state_matrix/scouting/measurement_scouting.md): A-measurements-scout-001, termocoppia NIST, τ ≈ 57 ms.
- [Matrice delle osservazioni](../01_observation_state_matrix/observation_state_matrix.md).
- [Inventario dei dati](../01_observation_state_matrix/data_inventory.md): B-data-audit-001/002/003, contraddizione C2 sulla frequenza.

---

## 4. Registro delle ricerche (canali e query)

**Canali:**

- WebSearch, standard ed esteso;
- API OpenAlex: ricerche per titolo e autore, stato open access, abstract;
- API OSTI;
- arXiv;
- DLR elib (link diretti ai PDF);
- HAL, dopo l'estensione degli host;
- Zenodo API;
- Springer open access;
- TU Delft repository;
- Lund LUP (tesi Petersson, raggiunta dai filoni prima del blocco);
- Hugging Face API e card;
- sito realpdebench.github.io;
- clone superficiale del codice RealPDEBench (solo codice).

**Query rappresentative per filone** (l'elenco completo è negli appunti dei filoni):

- **A:**
  - Meier GTMC Raman thermocouple;
  - PRECCINSTA wall temperature phosphor LES;
  - EM2C Guiberti Mercier wall temperature;
  - Taamallah / Hong flameholder heat conductivity;
  - Worth Dawson annular thermocouple;
  - Cambridge confined bluff body thermocouple;
  - phosphor wall temperature gas turbine model combustor.
- **B:**
  - RealPDEBench combustion experiment;
  - SJTU swirl intensity ammonia methane;
  - Okafor wall heat loss;
  - Khateeb KAUST swirl ammonia;
  - Cardiff thermocouple burner face R-type;
  - Mashruk chemiluminescent footprint;
  - XJTU co-firing PIV OH-PLIF;
  - NTNU NH₃/H₂ swirl;
  - phosphor wall temperature ammonia.
- **C:**
  - filtered Rayleigh PIV simultaneous;
  - Kaiser Frank Rayleigh PIV jet flame;
  - thermographic PIV flame;
  - side-wall quenching CARS PIV phosphor;
  - oil droplet evaporation isotherm flame front;
  - thermocouple PIV open flame;
  - jet-in-hot-coflow thermocouple;
  - CSSB bluff body temperature.
- **D:**
  - TCC-III benchmark thermocouples heated;
  - Baum 2014 database;
  - Peterson toluene LIF PIV;
  - Kopf thermographic PIV engine;
  - Ding phosphor piston;
  - Darmstadt engine coolant 60 °C;
  - two-colour soot pyrometry.
- **E:**
  - Shaddix radiation correction;
  - Heitor Moreira probes;
  - Bradley Matthews silica coating;
  - Tagawa Ohta two-thermocouple;
  - Heitor Taylor Whitelaw LDA thermocouple;
  - fused quartz limits;
  - IR thermography combustor liner;
  - TDLAS swirl combustor;
  - forum / ResearchGate thermocouple flame.
- **F:**
  - wall temperature not measured LES swirl;
  - thermal boundary conditions LES PRECCINSTA;
  - Hochgreb mind the gap;
  - TNF workshop thermal boundary conditions;
  - PIV combustion thermophoresis refractive index.

---

## 5. Accessi falliti e blocchi

| Host o risorsa | Errore | Che cosa non è stato letto |
|---|---|---|
| www.sciencedirect.com | HTTP 403 (anche per articoli open access) | Liu 2024 Fuel; Abram 2018; Ye 2026; vari paper Elsevier |
| pubs.aip.org | Connessione rifiutata, poi blocco | Liu 2024 PoF (candidato rig RealPDEBench) |
| lup.lub.lu.se | DNS; poi richiesta negata dal controllo dei permessi | Record di Liu 2024 Fuel (la tesi Petersson era stata letta prima) |
| orca.cardiff.ac.uk, jae.cardiffuniversitypress.org | DNS; poi pagina anti-bot | Tesi Davies 2025; paper Mashruk; J. Ammonia Energy 38 (rig SJTU annulare) |
| repository.kaust.edu.sa | 403 | Khateeb e Wang (KAUST) |
| hal.science (parte), hal-centralesupelec | Pagina JS anti-bot / 403 | Guiberti 2015; Degenève 2019 (Mercier e Koren invece letti) |
| dspace.mit.edu | "Web Firewall 429" | Taamallah 2015; Hong 2012 (MIT) |
| deepblue.lib.umich.edu | 403 | Readme e dati TCC-III; tesi Schiffmann e Greene |
| tuprints / tudatalib (TU Darmstadt) | Pagina anti-bot Anubis | Tesi Kosaka, Johe, Schmidt, Lee; database Baum |
| www.repository.cam.ac.uk | 429 / 403 | Hochgreb 2019; tesi Cambridge |
| link.springer.com (alcuni PDF) | Challenge JS o redirect a idp | Exp. Fluids e Appl. Phys. B non open access |
| opg.optica.org | Challenge JS | Fond 2012 (Opt. Express, open access) |
| core.ac.uk, mdpi.com, researchgate, reddit, tandfonline, asme, wiley, vtechworks, drum.umd, re.public.polimi.it, oatao, ifp.hal.science, duepublico2, eprints.whiterose, tnfworkshop.org | 403, DNS o anti-bot | PoliMi IJETR 2017; tesi VT, UMD e Duisburg; forum |
| Strumento di lettura web (WebFetch) | DNS per host non consentiti nella propria lista | Copie alternative degli stessi testi |

---

## 6. Esclusioni motivate

- Termometria industriale generica e incendi: usati solo come fonte di strumento (Yilmaz 2008), non come casi.
- Dataset: nessuna ricerca sistematica, nessuno scaricamento (vincolo dello step 1). Solo elenco di piste (§7).
- Detonazioni e fireball (LAS) e pirometria del soot: citati solo per completezza; non pertinenti a un combustore NH₃/CH₄ premiscelato.
- Fiamme laminari a bassa pressione con sonde MBMS: già nella missione 01, non ripresi.
- Lavori senza PIV inclusi **solo** quando chiariscono una pratica termica decisiva: ETH (termocoppia di parete come stato), Sandia HCCI (LIF calibrato sulla pressione), NH-LSU (termocoppie d'ingresso e scarico), Cardiff HPOC.

---

## 7. Piste incontrate (non indagate: da valutare nello step 2)

- **Archivi TNF:**
  - Sandia flames D–F;
  - Cambridge/Sandia stratified swirl (CSSB), con temperature del bluff body di Euler 2014;
  - Darmstadt TSF e SWQ;
  - Lund/Darmstadt LSB;
  - dati di contorno DLR GTMC/PRECCINSTA.
- **ISF-3** (fiamma fuligginosa DLR a 3 bar): pareti, CARS, PIV.
- **Deep Blue TCC-III:**
  - "Near-Wall Flow", doi 10.7302/Z2HQ3X4S (flusso termico in testa + PIV);
  - "CFD Input Dataset", hdl 2027.42/108382;
  - serie accese.
- **TUdatalib Darmstadt:**
  - condizioni al contorno del motore;
  - temperatura del pistone con H₂, doi 10.48328/tudatalib-2291.
- **Cardiff research data:** dataset 30234904; Pugh 2018, doi 10.17035/d.2018.0049851796.
- **Adelaide JHC:** Kildare et al. 2024, T da Rayleigh con PLIF.
- **Tesi:**
  - McManus (Ohio State 2019), FRS + PIV;
  - Kopf (Duisburg), PIV termografica in motore;
  - Davies (Cardiff 2025), termocoppie nei rig NH₃.
- **Sydney:** file "Surface_Temp_BB" (superficie del bluff body).
- **Bruciatore swirl PoliMi** (IJETR 2017): PIV + LDA + termocoppia con correzione basata sulla velocità. **Pertinente al contesto PoliMi.**
- **EngineBench:** già nella missione 01 (solo velocità).

---

## 8. Lacune residue

1. Il rig sperimentale da cui deriva RealPDEBench non è confermato. Strumentazione termica del rig SJTU: [M].
2. Originali DLR del 2006–2007 (Weigand, Meier) ed EM2C (Guiberti 2015): [M].
3. MIT (laboratorio Ghoniem) e PoliMi: [M] o [SEC].
4. Rig NH₃ di Cardiff, KAUST, Tohoku e XJTU: solo abstract o [SEC].
5. Review metodologiche (Heitor & Moreira, Brübach, Abram, Hochgreb): non lette.
6. Pratica informale (forum): non raccolta.
7. Gruppi cinesi, giapponesi e industriali: sottorappresentati. La selezione è sbilanciata verso archivi aperti (DLR elib, HAL, OSTI, arXiv) e gruppi con LES.
