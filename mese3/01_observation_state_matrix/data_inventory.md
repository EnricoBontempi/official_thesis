# Inventario dei dati — Observation–State Matrix (incarico `data-audit`)

**Data:** 25 novembre 2026 · **Base:** `plan/month3-missions` @ `30d11e0` · **Branch:** `matrix/f1-data-audit`
**Roadmap:** [roadmap01.md](roadmap01.md) §4 · **Template:** [templates_abc.md](templates_abc.md) (scheda B e regole comuni)
**Natura:** ricognizione documentale con controlli locali leggeri. Nessun training, inferenza, download, apertura di TEST; nessun dato, raw o codice modificato.

Legenda delle evidenze: **D** documentato (fonte indicata, con tipo di lettura) · **I** inferito (premesse esplicite) · **M** mancante/non verificato (domanda concreta).
Tipo di fonte per ogni affermazione: **[L]** lettura/controllo diretto eseguito oggi in questa ricognizione · **[P]** documentazione primaria letta · **[A]** audit o rapporto preesistente della repository, non rieseguito · **[S]** riportato da sintesi, non letto verbatim.

---

## 1. Sintesi

1. **Nel perimetro consultato, l'unico dataset di campi effettivamente presente è la parte numerica (LES) di RealPDEBench/combustion.** Le 30 traiettorie sono presenti e leggibili sul computer locale (30 shard Arrow, 2,098 GB ciascuno; le dimensioni coincidono oggi con l'inventario remoto). Ogni traiettoria è una condizione operativa NH₃/CH₄ × φ: 2001 frame, griglia 128×128, 15 canali, T in K e tre componenti di velocità. [L][A]
2. **Il contenuto sperimentale (OH\*, 8 shard, ≈3,7 GB) non risulta sul computer.** È dichiarato sull'hub; nei percorsi controllati non c'è alcun file `real` e nessun documento letto ne riporta un'ispezione. Di esso conosciamo soltanto metadati e schema. Non esiste alcuna PIV e nessuna temperatura misurata per la combustione. [L][P]
3. **I campi "veri" sono simulazioni LES, non misure.** Il paper dichiara una LES 3D implicita instazionaria; il piano di sezione, l'interpolazione a 128×128 e i filtri non sono dichiarati nelle fonti lette. I valori istantanei sono stati LES, non stati sperimentali. [P][A]
4. **Simulazione e esperimento condividono la condizione operativa, non lo stato istantaneo.** Non c'è registrazione campo per campo; il paper (sintesi) confronta segnali OH\* integrati o in frequenza. Nessun riferimento sperimentale simultaneo e co-localizzato per T o velocità. [S][A]
5. **Tempo.** Quattro livelli, da non confondere (dettaglio in B-data-audit-001 e 005):
   - *regolarità delle coordinate* — controllata oggi su `t` di 23 shard non-TEST (20 train + 3 val): 22 regolari (Δt=0,25 ms, nessun salto), 1 con lacuna interna (`40NH3_1.2`, passo 1098→1099, `t` lungo 2000) [L];
   - *associazione campo–tempo accettata dal contratto/loader esistente* — `t[i]` è assunto come tempo del frame `i` quando `t` è finito, non decrescente e di lunghezza `shape_t` [A];
   - *verifica indipendente dell'esportazione* (secondo riferimento temporale, confronto con il raw HDF5 o con gli autori) — **non disponibile**: manca per tutte le traiettorie, ed è una lacuna di verifica, non un errore dimostrato [M];
   - *anomalia dimostrata* — solo `40NH3_1.2`: la lacuna in `t` è riscontrata oggi [L]; il frame 2000 identico al 1999 proviene dall'audit del 16 novembre e **non è stato rieseguito** [A].
   I 7 shard TEST non sono stati aperti: per loro restano solo i metadati dei manifest.
6. **Geometria: gli assi dell'array sono verificati, l'orientamento fisico no.** Colonne = x, righe = y, celle anisotrope (Δx≈0,581 mm, Δy≈0,758 mm). Il bruciatore "a destra" o "sotto" è una lettura storica incoerente fra audit; il significato fisico di i/j/k e il segno non sono dichiarati dalla fonte. [L][A]
7. **Dati motore PoliMi, RANS/URANS, casi laminari: nessun accesso individuato.** Nella repository e nelle directory consultate (§6) non ho trovato dati motore né laminari. Le sole tracce sono l'indicazione riferita da Enrico (colloquio D'Errico) e il messaggio di Linzheng. Questo non esclude che esistano o siano stati inviati su canali non consultati (posta, cluster, altri computer). [L][A]
8. **Cosa è supportato oggi:** ricostruzione/diagnostica su campi LES numerici (velocità planari → T) con osservazioni degradate sintetiche, valutate sulle sole condizioni TRAIN/VAL. **Cosa richiede verifiche o ipotesi dichiarate:** usi multi-frame (forecasting, sensori lenti: verifiche proporzionate in §9), orientamento fisico, ogni affermazione su OH\*, su PIV reale, su validazione sperimentale, su dati motore (§9).

---

## 2. Disponibilità effettiva dei dati

| Livello | Risorsa | Stato | Evidenza |
|---|---|---|---|
| **Presente nella repository (Git)** | Manifest e metadati piccoli: `data/manifests/` (inventario remoto, riassunto audit 30 condizioni, `operating_conditions.yaml`, statistiche, hash delle cache, manifest maschere); `data/manifests/hf_small_metadata/` (README hub, `channels.json`, `dataset_info`, `state`, indici JSON, parametri); `configs/splits/*.yaml`; codice dei loader | Presente | [L] i file sono nel worktree |
| **Presente sul computer, fuori da Git** | 30 shard Arrow numerici `data/raw/combustion/hf_dataset/numerical/` (≈62,9 GB; 2 098 221 016–120 B ciascuno), `dataset_info.json`, `state.json`, `mean_std.pt` | Presente, leggibile | [L] dimensioni di tutti i 30 confrontate oggi con l'inventario remoto: nessuna differenza; SHA256 **non** ricalcolati oggi |
| | Cache derivate in `data/processed/`: `month2_dense_unet_cache` (760 frame train_fit), `month2_dense_unet_cache_c149` (2980 train_fit = 20×149), 220 `seen_train_eval`, 123 `val_eval`; maschere sparse `month2_sparse_eval_masks` | Presente | [L] forme `.npy` lette dall'intestazione; contengono solo condizioni train e val |
| **Dichiarato sul computer ma assente** | Shard sperimentali OH\* `combustion/hf_dataset/real/` (8 shard) | **Assente**: nessun file locale | [L] controllo di esistenza di tutti i percorsi dell'inventario remoto |
| **Dichiarato su cluster/altro sistema** | Checkpoint, risultati ed esperimenti dell'ambiente di calcolo Engaging, riferiti nei documenti Mese 2 e `docs/nonideal/` | Non verificato: nessun accesso eseguito oggi | [A] |
| **Solo citato** | Raw HDF5 sul sito RealPDEBench; EngineBench (Oxford, Kaggle); archivi TNF/Sandia (Sydney swirl, TECFLAM); dati motore RANS/URANS PoliMi | Non individuati nel perimetro consultato (nessun file locale) | [A] |
| **Percorsi consultati senza esito** | `~/projects/` vicini (`master-thesis`, `thesis-research`, `ModularDT`): solo elenco di primo livello e ricerca di termini; nessun dato combustione, motore o laminare. `external/GLU_Reconstruction/Dataset/` contiene solo `.gitkeep` | Nessun dato pertinente | [L] |

---

## 3. Righe sintetiche B

| ID | Dataset/configurazione | Contesto e provenienza | Variabili | Spazio e tempo | Struttura dei file | Accesso, riferimento e lacune |
|---|---|---|---|---|---|---|
| B-data-audit-001 | RealPDEBench combustion, **numerico LES**, 30 traiettorie (raw Arrow v2.0.1) | Simulazione LES 3D, fiamme swirl NH₃/CH₄/aria, STAR-CCM+ (EDC); stati istantanei; premiscelazione: vedi scheda | 15 canali: P, HRR, 7 frazioni molari, T, 3 velocità, ch14 | 128×128, x∈[−36,9; +36,9] mm, y∈[62,9; 159,1] mm; 2001 frame, Δt=0,25 ms (0,5 s); `t` regolare nei 22 shard non-TEST controllati | 1 traiettoria/riga/shard, `observed (T,H,W)` + `numerical (T,H,W,15)` float32 + `x,y,t` float64 1D | Locale, CC BY-NC 4.0; orientamento fisico e piano non dichiarati; associazione campo–tempo accettata dal loader, non verificata in modo indipendente |
| B-data-audit-002 | RealPDEBench combustion, **sperimentale OH\***, 30 traiettorie in 8 shard | Esperimento: chemiluminescenza OH\* ad alta velocità, stesse condizioni operative dichiarate | 1 canale, intensità OH\* | 128×128 dichiarata; 2001 frame | `observed (T,H,W)` + `x,y,t` | **Non locale**; nessun file letto; geometria e ottica non verificate |
| B-data-audit-003 | Canale `observed` dei file **numerici** (intensità "surrogate-mapped") | Segnale proxy/sintetico derivato dalla simulazione (mappatura non descritta nelle fonti lette) | 1 canale | come 001 | 1° array della riga | Locale; contenuto mai caratterizzato nelle audit (solo scansione NaN/Inf) |
| B-data-audit-004 | Cache di progetto derivate da 001 (A38, C149, maschere sparse) | Preprocessing locale, solo TRAIN/VAL | vi, vj normalizzate, T in K, p=(NH₃, φ) normalizzati | 128×128; frame ogni 40 (A38) o 10 (C149) indici | `.npy` float32 `(N,128,128)`, `geometry.npz`, `frames.json`, manifest | Locale, payload gitignored; vk assente |
| B-data-audit-005 | Caso anomalo `40NH3_1.2` (shard 22) | Stesso dataset, packaging temporale anomalo | come 001 | `t` lungo 2000; lacuna interna [L]; frame 2000 = frame 1999 [A] | come 001 | Policy attuale del codice: singolo frame per indice consentito, multi-frame rifiutato su questo shard |
| B-data-audit-006 | EngineBench (Oxford TPSRG, PIV in motore ottico TCC-III, motorizzato) | Esperimento PIV, nessuna combustione | u,v su 50×49 vettori (padding a 128×128) | piano di swirl inferiore, 1300 rpm; fasi in CAD | Kaggle/GitHub | **Non scaricato**; solo loader letto |
| B-data-audit-007 | Archivi TNF/Sandia swirl (Sydney swirl, TECFLAM) | Esperimenti non premiscelati | LDV puntuale; Raman/Rayleigh/LIF | profili/punti | file separati | **Non scaricati**; nessuna PIV, nessun accoppiamento V+T verificato |
| B-data-audit-008 | Dati motore PoliMi (RANS/URANS) | Nessuna evidenza di contenuto | — | — | — | **Nessun accesso o dato individuato nel perimetro consultato** |
| B-data-audit-009 | Casi laminari e campi medi | Nessuna fonte individuata | — | — | — | **Nessun dato individuato nel perimetro consultato** |

---

## 4. Schede B

### B-data-audit-001 — RealPDEBench combustion: campi numerici LES (30 condizioni)

**Sintesi.** È l'unico insieme di campi completo che abbiamo in mano. Contiene stati istantanei LES su una sezione 128×128 di una fiamma swirl, con T e velocità a 3 componenti come riferimento numerico. Verificati direttamente oggi: presenza e dimensione dei 30 file, struttura di schema, coordinata `t` (non i campi) di 23 shard non-TEST, griglie. Il resto proviene da audit preesistenti.

| Campo | Contenuto | Evidenza |
|---|---|---|
| Identità e accesso | `AI4Science-WestlakeU/RealPDEBench`, hub versione dati 2.0.1 (formato `lazy_slicing_v2`, rilasciata 2026-09-18 per la card); paper arXiv:2601.01829 (ICLR 2026). Licenza CC BY-NC 4.0 (card HF e codice). Locale in `data/raw/…/numerical/` | D [P][A]; presenza D [L] |
| Contesto fisico e provenienza | LES 3D implicita, instazionaria; STAR-CCM+ 2022.1, solver segregato, EDC, meccanismo ridotto 38 specie/184 reazioni, CVODE; due inlet a portata massica, outlet in pressione, pareti no-slip, 1 bar. Dati "cross-sectional". **Non dichiarati nelle fonti lette:** mesh, modello SGS, geometria di swirler/combustore, piano della sezione, interpolazione alla griglia 128×128, filtri/crop. Stati istantanei (non medi). Regime: swirl turbolento. **Regime di miscelazione (premiscelato o no): non determinato.** B.2 descrive due inlet a portata massica, aria e miscela combustibile: ciò non stabilisce se la miscela sia premiscelata a monte del swirler né l'uniformità delle frazioni all'ingresso. Una verifica del 24 novembre su 15 canali citata in una nota di sessione non è rintracciabile (nessuno script né output nella repository, nessuna fonte che io abbia potuto aprire): **non è usata come evidenza** | D [P][A] (solver, EDC, inlet, B.2 verbatim); M (mesh, SGS, piano, ricampionamento, regime di miscelazione; verifica in C9) |
| Condizioni operative | NH₃ ∈ {0,20,40,60,80}% (CH₄ 100…20%) × φ; 30 file, copertura **non cartesiana**: 0%: 9 (φ 0,85–1,30); 20%: 9 (0,75–1,30); 40%: 6; 60%: 1 (φ=0,95); 80%: 5. La documentazione elenca φ senza 0,95 e 1,15, che i file contengono | D [A][L: 30 shard e `remain_params` coerenti nel manifest] |
| Variabili | Indici `numerical[...,c]`: 0 Absolute_Pressure, 1 Chemistry_Heat_Release_Rate, 2 CH₄, 3 CO, 4 CO₂, 5 H₂O, 6 NH₂, 7 NH₃, 8 OH (frazioni molari), 9 Pressure, **10 Temperature**, **11/12/13 Velocity[i/j/k]**, 14 Velocity_Magnitude. Unità: non dichiarate nelle fonti; T in K e velocità in m/s **inferiti** dagli intervalli fisici (T medio ≈1860–2190 K per condizione, velocità O(1–40)). Tutti i campi sono simulati; nessuna misura. Il canale 14 non è il modulo euclideo (r≈0,997, negativo fino a ≈−8 m/s): ricalcolare da 11–13 | D [A] per canali e ordine; I per unità |
| Qualità valori | Nessun NaN/Inf (30/30, audit Mese 1). T<0 in 15/30 condizioni e T>2600 K in 24/30, frazione ≪0,01–0,03 %: artefatti LES. Frazioni molari frequentemente negative; HRR negativo. Frame consecutivi identici: controllati nell'audit del Mese 1 solo sui 3 shard pilota; sul `40NH3_1.2` il duplicato 1999/2000 è stato trovato dall'audit del 16 novembre (non rieseguito oggi) | D [A]; duplicati sulle altre 26 traiettorie (30 − 3 pilota − `40NH3_1.2`): **M** |
| Spazio | Array: asse 0 = y (righe), asse 1 = x (colonne); x ∈ [−0,03691; 0,03689] m, y ∈ [0,06285; 0,15913] m, uniformi e **identiche per le 23 shard non-TEST controllate**; Δx≈0,581 mm, Δy≈0,758 mm (celle anisotrope). Le coordinate sono 1D float64 (128). **Orientamento fisico:** non dichiarato (vedi §7 C1). Nessuna maschera di dominio nelle cache (`mask_all_true`) | Griglie e ordine assi: D [L][A]; orientamento: M |
| Tempo | 2001 frame, Δt=0,25 ms per tutte le 30 (manifest); finestre assolute diverse: 20 su [1,5; 2,0] s, 8 su [3,5; 4,0], 1 su [3,8; 4,3] (`0NH3_0.9`), 1 su [1,7; 2,2] (`20NH3_0.85`). Controllo [L] su `t` di 23 shard non-TEST: finito, crescente, passo mediano 2,5×10⁻⁴ s, nessuna lacuna in 22; lacuna interna in 1 (B-005). Livelli di evidenza e conteggi: tabella sotto. Le finestre assolute non sono allineate fra condizioni: il tempo assoluto non è una variabile di confronto | D [L] (regolarità di `t`, 23 shard) / D [A] (manifest, 30); 7 shard TEST: M |
| Relazione fra modalità | Tutti i canali sono la stessa simulazione: simultanei e co-localizzati per costruzione, ma generati dalla stessa LES. T, velocità e specie *non* sono modalità indipendenti | D [A] |
| Struttura dei file | Arrow IPC stream, **1 riga = 1 traiettoria = 1 shard**; colonne `sim_id`, `observed` (bytes → float32 `(2001,128,128)`), `numerical` (bytes → float32 `(2001,128,128,15)`), `numerical_channels`, `shape_t/h/w`, `x,y,t` (+ `*_shape`). Mappa shard→`sim_id`: ordine lessicografico, verificata sui 30 (Mese 1). Leggibili con `pa.ipc.open_stream` su memory map; decodifica di un frame = decodifica dell'intera riga (~2 GB) | D [L][A] |
| Riferimenti utilizzabili | T (K) come riferimento numerico di ricostruzione da velocità; per i 30 casi come verità **simulata**. Non fornisce riferimento sperimentale | D |
| Verifiche e lacune | **Tracciabilità SHA256:** vedi C4 (lacuna di tracciabilità, non contraddizione); oggi solo le dimensioni sono riconfrontate. Orientamento, piano, ricampionamento, SGS, mesh, unità, regime di miscelazione: aperti. Aperti anche i campi 3D: la sezione non è un volume | M |
| Collegamenti | Audit: `docs/month1/02_data_audit_report.md`, `docs/month1/02_pilot_data_audit_report.md`, `docs/month2/mission_baseline_timestamp_audit.md`, `docs/nonideal/00_ricognizione_osservazioni.md`; codice: `src/combustion_recon/data/shard_access.py`, `temporal_quality.py` | — |

**Split locali (non ufficiali).** Gli indici ufficiali non sottraggono intere condizioni: numerico `val`/`test` vuoti, `in/out_dist` vuoti, 30 condizioni in `remain`. Il progetto ha congelato 20 train / 3 val / 3 test_interp / 4 test_extrap per condizione (`configs/splits/splits_interp_extrap.yaml`). [L][A]

**Tempo: livelli di evidenza e conteggi.** Sono quattro affermazioni diverse; ciascuna ha il proprio stato.

| Livello | Che cosa afferma | Stato | Evidenza |
|---|---|---|---|
| 1. Regolarità delle coordinate | `t` finito, non decrescente, passo mediano 2,5×10⁻⁴ s, nessun passo anomalo | **Controllata** su 23 shard non-TEST (20 train + 3 val): 22 regolari, 1 con un passo doppio (`40NH3_1.2`). Non controllata sui 7 TEST | D [L] |
| 2. Associazione accettata dal contratto/loader | Il loader assume `t[i]` come tempo del frame `i` se `t` è finito, non decrescente e lungo `shape_t`; la accetta (`field_time_association_verified=True`) e la rifiuta per lo shard anomalo | È una **regola di accettazione**, non una prova | D [A] (`shard_access.py`, `temporal_quality.py`) |
| 3. Verifica indipendente dell'esportazione | Confronto con una fonte esterna (raw HDF5, convertitore, autori) che confermi la corrispondenza indice↔tempo | **Non effettuata** per alcuna traiettoria; mancano un secondo riferimento temporale e un log del convertitore. È una lacuna di verifica, **non un errore dimostrato** nelle traiettorie regolari | M |
| 4. Anomalia dimostrata | Solo `40NH3_1.2`: `t` ha 2000 campioni con un passo doppio a 1098→1099 (riscontrato oggi); frame 2000 bit-a-bit identico al 1999 (audit del 16 novembre, **non rieseguito oggi**) | Dimostrata per questo shard; la spiegazione "campione interno mancante" è un'interpretazione dell'audit | D [L] (`t`) / D [A] (duplicato) / I (spiegazione) |

Conteggi: 30 condizioni in totale = 20 train + 3 val + 7 TEST (3 interp + 4 extrap). Controllate oggi: 23 (train + val, incluso `40NH3_1.2`) → 22 regolari + 1 anomala. I "29 casi non anomali" sono 30 − 1: i 22 regolari controllati più i 7 TEST non controllati. Le cache contengono solo le 23 condizioni train/val: 22 con flag di associazione `True` e 1 (`40NH3_1.2`) con `False`.

---

### B-data-audit-002 — RealPDEBench combustion: OH\* sperimentale (assente in locale)

**Sintesi.** Esiste sull'hub, ma non l'abbiamo. Tutto ciò che sappiamo viene da card, schema e paper. Non è un campo di temperatura né di velocità e non è la frazione molare di OH.

| Campo | Contenuto | Evidenza |
|---|---|---|
| Identità e accesso | Stessa release di 001, cartella `hf_dataset/real/`: 8 shard (6×≈500,3 MB, 2×≈375,2 MB, totale ≈3,67 GB), 30 traiettorie impacchettate 2–4 per shard; CC BY-NC 4.0 | D [A] (inventario remoto); **assenza locale D [L]** |
| Contesto | Chemiluminescenza OH\* ad alta velocità; §3.2 del paper: solo OH\* (1 canale). Nessuna PIV né temperatura misurata per la combustione; la PIV è dichiarata solo per i sistemi fluidi | D [P] |
| Variabili | Schema (`dataset_info`): `observed (T,H,W)`, `x`, `y`, `t`. Unità e scala dell'intensità: non dichiarate nelle fonti lette. Proprietà ottiche (esposizione, filtro, camera, lunghezza di visione) non lette | D schema [L sui metadati]; M resto |
| Spazio | 128×128 dichiarata; campo di vista, proiezione (OH\* è integrato lungo la linea di vista), orientamento e rapporto con la sezione LES **non verificati** | M |
| Tempo | 2001 frame dichiarati; card: "4000 Hz per 1 s" non compatibile con 2001×0,25 ms = 0,5 s; camera "2000 fps" da B.3 [S]; coordinate `t` reali non lette | contraddizione D, M per `t` |
| Relazione fra modalità | Stesse condizioni operative nominali delle simulazioni. Nessun contenuto letto indica registrazione, sincronizzazione o co-localizzazione campo-per-campo; il confronto riportato è su segnali integrati/spettrali [S] | D [S]/M |
| Indici | `train_index_real` 46 223, `val_index_real` 5 718, `test_index_real` 5 719 finestre su 30 condizioni (separazione per tronco temporale, non per condizione) | D [L sui JSON di Git] |
| Verifiche | Scaricare un solo shard (≈0,4–0,5 GB) per ispezionare scala, `t`, FOV e verifica visiva sarebbe un passo separato da autorizzare | M |

---

### B-data-audit-003 — `observed` nei file numerici: intensità "surrogate-mapped"

**Sintesi.** Ogni shard numerico contiene anche un array `observed (2001,128,128)` che la card definisce "surrogate-mapped intensity": un segnale sintetico/proxy, non una misura OH\* e non il canale 8 (frazione molare di OH). Nessuna audit ne ha caratterizzato il contenuto.

| Campo | Contenuto | Evidenza |
|---|---|---|
| Provenienza | Card e `channels.json`: modalità "misurabile" riempita con intensità mappata dalla simulazione; il metodo (proporzionale a OH? a HRR? integrale di linea?) non è descritto nelle fonti lette. Il loader legacy cita `surrogate/{sim_id}` | D (esistenza) / M (metodo) |
| Natura | Proxy sintetico. Può essere campo planare, non proiezione integrata, e non include risposta ottica: **non assumere equivalenza con OH\* sperimentale** né con OH numerico | I |
| Verifiche | Scansione NaN/Inf: nessuno (pilota). Statistiche, correlazione con canali 8/1/10, unità: mai calcolate | M |
| Uso potenziale | Eventuale input ottico della missione `05_optical_diagnostics` solo dopo la sua caratterizzazione | I |

---

### B-data-audit-004 — Cache di progetto

**Sintesi.** Le cache sono derivate di B-001 e contengono *meno* informazione del raw. Sono tutte TRAIN/VAL: nessuna condizione TEST.

| Campo | Contenuto | Evidenza |
|---|---|---|
| Identità | `data/processed/month2_dense_unet_cache` (A38: 760 train_fit, 220 seen_train_eval, 123 val_eval), `…_c149` (2980 train_fit = 20 cond × 149, stesse 220 e 123), `month2_sparse_eval_masks` (1103 frame, 3309 istanze, seed 20261116, siti 164/819/3277 per 1/5/20%). Payload gitignored; manifest in Git | D [L] (forme lette dagli header `.npy`); hash: manifest [A] |
| Cosa conservano | `vi_norm`, `vj_norm` (z-score globale fit-only), `target_T_K` (K), `target_T_norm`, `p_norm = (NH₃, φ)` normalizzati, `x/y` (float64) e `x_scaled/y_scaled` (128×128 float32) | D [L] |
| Cosa **non** conservano | `vk` (componente fuori piano; presente solo nel raw), le 12 altre variabili fisiche, `observed`, i timestamp (salvati in `frames.json` come coordinate archivistiche, non come variabile del modello). Input solo planari `planar_2C` | D [L][A] |
| Sottocampionamento | A38: indici 0,40,…,1480 (train_fit) e 1600,…,2000 (seen_train_eval); C149: indice ogni 10 (2,5 ms) da 0 a 1480. Frame di valutazione VAL: 41 per condizione (123 su 3) | D [L][A] |
| Statistiche | μ,σ vi=−0,060, 5,674; vj=4,272, 11,910 m/s (760 frame); T medio 2039,5 K, σ 331,7 K | D [A] |
| Differenze raw/cache | Orientamento e coordinate invariati; nessuna maschera di dominio; nessuna correzione di T<0 o T>2600 K (T in K non ritagliata). Il flag `field_time_association_verified=True` in `frames.json` per le 22 condizioni regolari presenti in cache (su 23) registra l'accettazione da parte del loader, in base a forma e monotonia di `t`: non è una verifica indipendente dell'esportazione | D [L][A]; interpretazione I |

---

### B-data-audit-005 — `40NH3_1.2`: anomalia temporale documentata

**Sintesi.** Caso noto, non riparato. La lacuna in `t` è stata riscontrata di nuovo oggi; il frame duplicato proviene dall'audit del 16 novembre e non è stato rieseguito.

| Campo | Contenuto | Evidenza |
|---|---|---|
| Identità | `40NH3_1.2.h5`, `data-00022-of-00030.arrow`, 2 098 221 056 B, SHA256 `3d908c0d…29e4dbdb` coincidente con l'inventario (audit del 16 novembre) | D [A]; dimensione D [L] |
| Struttura | `shape_t=2001`, `numerical` rimodellabile a (2001,128,128,15); ma `t` ha 2000 elementi (`t_shape=2000`), finito, crescente, da 1,5 a 2,0 | D [L] riscontrato oggi su `t` |
| Anomalia | **Dimostrata:** un solo passo irregolare, `diff[1098]≈5×10⁻⁴ s` (doppio del mediano) [L]; frame 1999 e 2000 **identici bit a bit** [A, non rieseguito]. **Interpretazione dell'audit [I]:** la lacuna è compatibile con un campione interno mancante, indice uniforme 1099 (t=1,77475 s). L'ipotesi precedente "manca l'ultimo timestamp, aggiungere t[2000]=2,00025" è stata respinta dall'audit sulla base di queste evidenze | D [L] lacuna in `t`; D [A] duplicato e rifiuto; I spiegazione |
| Associazione campo–tempo | Non verificata in modo indipendente per alcun indice di questo shard, e marcata `False` dal codice. Non è noto se i campi dopo l'indice 1098 siano traslati di un campione rispetto a `t`, né quale dei due frame duplicati corrisponda al campione mancante. Manca una fonte indipendente (secondo timestamp, raw HDF5) | M |
| Stato nel codice | Policy attuale (non modificata qui): `has_original_time_coordinate=True` per i<2000; `timeline_regular=False`; `field_time_association_verified=False` per tutti i 2001 indici; `duplicate_of=1999` per 2000; multi-frame/forecasting rifiutati su questo shard; singolo frame per indice consentito | D [A] `temporal_quality.py`, `shard_access.py` |
| Presenza nelle cache | In C149: 149 frame train_fit e 11 seen_train_eval (incluso indice 2000, duplicato di 1999) hanno associazione non verificata | D [L] |
| Limiti | Nessuna riparazione o riscrittura del raw. Le altre 22 condizioni controllate hanno coordinate regolari; per esse l'associazione è accettata dal loader ma non verificata in modo indipendente (nessuna anomalia dimostrata). Per le 7 TEST non ho controllato `t` | — |

---

### B-data-audit-006 — EngineBench (PIV in motore ottico)

**Sintesi.** Precedente metodologico per osservazioni di velocità realistiche. Non scaricato. Non contiene temperatura né combustione.

| Campo | Contenuto | Evidenza |
|---|---|---|
| Identità e accesso | Oxford TPSRG; arXiv:2406.03325; `github.com/sambkr/EngineBench`. Pagina Oxford non raggiungibile il 24 novembre (HTTP 500); pagine Kaggle non leggibili. Licenza: testo CC BY 4.0, appendici [S] CC BY-NC-SA 4.0: **da chiarire** | D [A] (letto via API GitHub); M licenza |
| Contesto | PIV in motore ottico TCC-III motorizzato (nessuna combustione), piano di swirl inferiore; subset `LSP_small` 5 205 snapshot, 1300 rpm, 40 kPa; fase in gradi di manovella | D [P][A] |
| Variabili/spazio | u,v su 50×49 vettori con padding a 128×128; spaziatura ≈1,25–1,4 mm [S]; finestra, sovrapposizione, validazione: non verificate | D/[S]/M |
| Vettori mancanti | Nel rilascio sono **zeri esatti**; maschera `u==0 | v==0`: non distingue mancante da zero vero. Le lacune del benchmark sono **artificiali** (poligoni di bordo fino al 25%, maschere fisse di test) | D [A] (codice letto) |
| Incertezza | "average velocity error 1,5–8%" (da Schiffmann et al. 2016); nessun indicatore per vettore | D [P] |
| Riferimento | Nessun riferimento termico o scalare | D |
| Uso | Ordini di grandezza e logica delle lacune; **non** maschera realistica del nostro combustore | I |

---

### B-data-audit-007 — Archivi TNF/Sandia (swirl)

**Sintesi.** Scouting mirato del 24 novembre, non riesaminato oggi. Nessun dataset letto offre velocità e temperatura simultanee e co-localizzate a singolo colpo.

| Campo | Contenuto | Evidenza |
|---|---|---|
| Sydney swirl | Non premiscelato; LDV puntuale 2–3 componenti; composizione Raman/Rayleigh/LIF istantanei misurati a Sandia; campagne e file separati ("may not overlap"); nessuna PIV | D [P][A] |
| TECFLAM | Dati su richiesta al gruppo di Darmstadt; diagnostiche non dichiarate nella pagina | D [P][A] |
| Altri | Fiamme pilotate e bluff-body: non premiscelate e senza swirl | D [A] |
| Combustore modello DLR (PIV/OH-PLIF/Raman) | Non nell'archivio TNF letto; non verificato altrove | M |
| Accesso | Nessun dato scaricato | — |

---

### B-data-audit-008 — Dati motore PoliMi (RANS/URANS)

**Sintesi.** Nei percorsi e sistemi consultati (§6) non ho individuato né dati né accessi. Le sole tracce sono una richiesta di indagine e una descrizione di terza mano; questo non esclude materiale presente su canali non consultati.

| Campo | Contenuto | Evidenza |
|---|---|---|
| Fonti | Messaggio di Linzheng: "explore what engine data are available from PoliMi, including RANS/URANS". Jason: chiede "how much of [RANS data] is available, what does it contain"; il suo lavoro sul ammonia DT è stato fatto con RANS/URANS. Colloquio D'Errico (24 novembre, riferito da Enrico, **non verificato**): dispone soprattutto di RANS motoristiche; disponibilità di LES e condivisibilità da verificare; non si occupa di PIV in combustori ottici | D [A] (diario 25 nov., `roadmaptests.md` §2): messaggi riportati |
| Contenuto | Geometria, variabili, risoluzione, regime di carico, tempo/crank angle, mesh, formato, licenza: **nessuna informazione** | M |
| Natura attesa | RANS/URANS descrivono trattamento numerico; un target RANS è un campo medio/mediato d'insieme, non uno stato istantaneo | I |
| Cosa non si può dire | Nessuna affermazione di disponibilità, accesso, co-localizzazione con misure | — |
| Verifiche proposte | Domande minime da inviare tramite Enrico (non effettuate): geometria/cilindrata, RANS vs URANS vs LES, quantità esportate, risoluzione spaziale/temporale (crank angle), regime, esistenza di misure sperimentali associate, condivisibilità | M |

---

### B-data-audit-009 — Casi laminari e campi medi

**Sintesi.** Nessuna fonte è emersa dal perimetro consultato. Non è una prova di assenza di fonti esterne.

| Campo | Contenuto | Evidenza |
|---|---|---|
| Repository | Le occorrenze di "laminar" nei documenti riguardano citazioni bibliografiche (RF-PINNs; Mese 1) e le definizioni delle roadmap; nessun dato laminare | D [L] |
| `external/GLU_Reconstruction/` | `Dataset/` contiene soltanto `.gitkeep`; configurazioni per `collinear_re40` (flusso non reattivo di fluidodinamica) non accompagnate dai dati | D [L] |
| Altri percorsi | Nessuno pertinente | D [L] (ricerca limitata) |
| Resta aperto | Scouting di dataset laminari o a campi medi appartiene alla fase F2 | M |

---

## 5. Relazione fra modalità e validazione

| Domanda | Risposta | Evidenza |
|---|---|---|
| Quali modalità sono simultanee e co-localizzate? | Le 15 grandezze numeriche e il proxy `observed` (B-003) della stessa traiettoria: stessa simulazione, stessa griglia, stesso indice di frame. L'OH\* sperimentale (B-002) **non** è registrato fra il frame i della LES e quello sperimentale | D [A]/[S]; M (sincronizzazione) |
| "Stessa condizione operativa" = stesso istante? | **No.** Il paper dichiara il confronto soltanto a livello di condizione operativa, con segnali OH\* integrati o in frequenza; le finestre temporali assolute delle simulazioni non sono allineate fra condizioni; per l'esperimento le `t` non sono state lette | [S]; D [L] per le finestre numeriche |
| Cosa può fare da riferimento? | (a) T e velocità LES come verità **numerica** per ricostruzione da osservazioni sintetiche; (b) campi medi o statistici ottenuti dalle traiettorie; (c) per OH\*: eventuale confronto spettrale/statistico con l'esperimento, non punto a punto | D/I |
| Cosa non può validare? | Accuratezza **sperimentale** di T o velocità (nessuna misura); la stima del rumore PIV reale; fluttuazioni istantanee con un riferimento medio; un volume 3D con una sola sezione; il paradigma PIV (la slice LES non è una PIV: nessuna media di finestra, lacune, errore correlato); trasferimento OH\*→T senza dati simultanei | D/I |
| Validazione disponibile oggi | Numerica su simulazione, per condizioni escluse dal training (val: 3; test: 7 mai toccati da questa ricognizione) | D [A] |

Le grandezze `T`, velocità e `observed` vanno tenute distinte come tre ruoli: **campo numerico** (LES), **segnale proxy sintetico** (`observed` numerico), **immagine sperimentale** (OH\* reale). Il canale 8 (frazione molare OH) è una quantità di stato, non un segnale ottico.

---

## 6. Controlli locali eseguiti oggi e perimetro

| # | Controllo | Cosa copre | Cosa non copre |
|---|---|---|---|
| L1 | Stato Git, creazione del worktree `matrix/f1-data-audit` da `origin/plan/month3-missions` | Isolamento | — |
| L2 | Elenco `data/raw/`, `data/processed/`, `data/manifests/` e presenza dei percorsi dell'inventario remoto | 30 shard numerici presenti; 0 shard `real` presenti | Contenuto |
| L3 | Confronto di `os.path.getsize` dei 30 shard numerici con `size_bytes` dell'inventario | Dimensioni identiche per 30/30 | SHA256 non ricalcolati; nessuna lettura di byte di payload |
| L4 | Lettura (memory map, Arrow IPC) dei soli campi `shape_t`, `t_shape`, `numerical_channels`, `t`, `x`, `y` dei **23 shard non-TEST**; analisi di `diff(t)` | Esistenza/regolarità di `t`, coordinate identiche (primi/ultimi valori), 2001 vs 2000 campioni di `t` | **7 shard TEST non aperti** (`0NH3_1.3`, `20NH3_0.75`, `20NH3_1`, `20NH3_1.3`, `40NH3_1.15`, `60NH3_0.95`, `80NH3_1.25`); campi `numerical` e `observed` non decodificati, quindi il duplicato 1999/2000 **non è stato rieseguito** (dato [A]); duplicati di campo non cercati |
| L5 | Intestazioni `.npy`/`.npz` di 2 cache e `frames.json` (flag temporali, `mask_all_true`) | Forme, dtype, flag, conteggi | Valori dei campi |
| L6 | Lettura di metadati JSON in Git (schema, indici, parametri, README dell'hub) | Struttura dichiarata | Nulla sul remoto vivo: i metadati sono copie del 13 novembre |
| L7 | Elenco di primo livello e ricerca di termini in `master-thesis`, `thesis-research`, `ModularDT` e `external/GLU_Reconstruction/Dataset` | Nessun dato motore/laminare | Non è una scansione del computer |

**Limiti.** Un controllo su campione non dimostra proprietà su tutti i file. L4 è lettura diretta attuale di `t` su 23 shard non-TEST (22 regolari + `40NH3_1.2`); per gli altri 7 vale solo il manifest preesistente, e le affermazioni sul duplicato di campo valgono come audit preesistente. Non ho consultato rete, cluster né fonti primarie ex novo: le affermazioni su paper e codice esterno sono quelle degli audit del 24 novembre (testo integrale per §3.2, Appendice B.1, B.2 e Tabella 7; sintesi [S] per B.3, parte combustione di B.4 e D.2).

---

## 7. Contraddizioni, lacune di tracciabilità e domande aperte

| ID | Contraddizione o lacuna | Fonti | Conseguenza pratica |
|---|---|---|---|
| C1 | **Orientamento del combustore.** Pilota (13 nov.): bruciatore *sotto* la finestra, y assiale, `Velocity[j]` assiale con +j≈+y. Audit completo (14 nov.), `operating_conditions.yaml`, `splits_interp_extrap.yaml`, `06_supervisor_review_brief.md`: getti freddi *a destra* (col≈127), V verso sinistra, `Velocity[j]` assiale. Ricognizione del 24 nov.: definizione di i/j/k, segno e piano **non dichiarati**; `vj` lungo le colonne è inferenza dalle statistiche TRAIN. Inoltre i getti "freddi" sono individuati tramite T, che la ricognizione del 24 nov. dichiara di non usare come tracciante | `02_pilot_data_audit_report.md`, `02_data_audit_report.md`, `00_ricognizione_osservazioni.md` §1.2, §9.2 | Non usare "bruciatore a destra/sotto" come fatto. Il segno di vj rispetto a +x resta aperto. Gli operatori isotropi non ne dipendono |
| C2 | **Durata e frequenza.** "4000 Hz per 1 s" (docs/card) vs 2001 frame × 0,25 ms = 0,5 s; camera OH\* 2000 fps [S] | `01_remote_reconnaissance.md`, `00_ricognizione…` §1.1, §9.1 | Usare Δt=0,25 ms e 0,5 s misurati. Per l'esperimento la frequenza non è verificata |
| C3 | **Tipo di `x,y,t`.** Card: float32 `(H,W)` e float32 `(T,)`; shard locali: `x`,`y` 1D float64 (128), `t` float64 | `hf_small_metadata/README.md`; L4 | Il loader del progetto legge i tipi effettivi; non assumere la card |
| C4 | **Tracciabilità della verifica SHA256 (lacuna, non contraddizione).** Le dichiarazioni hanno oggetto e data diversi. (a) Pilota 13 nov. e audit completo 14 nov.: "All sizes/SHA256 match" / "Verified" per gli **shard raw scaricati** rispetto all'inventario LFS. (b) Audit timestamp 16 nov.: dimensione e SHA256 locali di **un** shard (`data-00022`) uguali all'inventario. (c) Manifest della cache, 17 nov.: la **costruzione della cache** usa gli hash dell'inventario più la dimensione locale e "non ricalcola" gli SHA: una dichiarazione sul build, non sullo stato dei raw. Una verifica al download può coesistere con il mancato ricalcolo durante la cache. Non dimostro alcun conflitto; manca però, nei file che ho letto, un registro per-shard degli hash locali ricalcolati e del metodo (gli script `02_audit_*` non contengono calcolo di SHA) | `02_data_audit_report.md` §1, `02_pilot_data_audit_report.md` §1, `mission_baseline_timestamp_audit.md`, `month2_dense_unet_cache_manifest.json` | Citare la verifica dei raw come "dichiarata dall'audit del Mese 1", senza attribuirle un registro. Oggi ho riconfrontato solo le dimensioni |
| C5 | **Controllo dei duplicati (copertura parziale).** Pilota: nessun frame consecutivo identico su 3 shard; audit completo "focused on finite-value scans"; il duplicato di `40NH3_1.2` è emerso nell'audit del 16 novembre. Non è una contraddizione: i controlli avevano coperture diverse | `02_data_audit_report.md` §5, `mission_baseline_timestamp_audit.md` | Duplicati sulle altre 26 traiettorie (non pilota, non `40NH3_1.2`): non verificati, né provati assenti |
| C6 | **Associazione accettata dal loader ≠ verifica indipendente (lacuna).** Le cache segnano `field_time_association_verified=True` per le 22 condizioni regolari presenti (su 23) in base alla regola di accettazione del loader (forma, finitezza, monotonia di `t`). La regolarità è stata confermata oggi per 22 shard non-TEST; l'indipendenza della verifica non è dimostrata, ma non c'è neanche evidenza di errore nelle regolari | `shard_access.py` (policy temporale), `frames.json`, tabella "Tempo" in B-001 | Etichettare l'associazione come "accettata, non verificata indipendentemente", senza trattarla né come errore né come prova. Un secondo riferimento temporale non è disponibile |
| C7 | **Componenti di velocità.** Mese 1: 3 componenti (i, j, k) nei test di fattibilità; Mese 2: solo 2C planari in cache | `month2_dense_unet_cache_manifest.json`; `00_context_and_decisions.md` §3 | Non confrontare risultati 3C storici con 2C |
| C8 | **φ documentate vs presenti.** Docs: {0,75; 0,85; 0,9; 1,0; 1,05; 1,1; 1,2; 1,25; 1,3}; file: anche 0,95 e 1,15 | `01_remote_reconnaissance.md` §3.1–3.2 | Usare l'insieme dei file |
| C9 | **Regime di miscelazione non determinato.** Il documento non può affermare che il campo sia premiscelato. Una verifica citata il 24 nov. non è rintracciabile e non è usata. B.2 elenca due inlet (aria e miscela combustibile) ma non basta a decidere. **Verifica proposta e suoi limiti:** (i) fonte primaria: geometria di iniezione/swirler e condizioni d'ingresso dagli autori o dal codice di conversione, che è la via decisiva; (ii) sui dati, un controllo sulle zone fredde dei reagenti (X_CH₄+X_NH₃ e rapporto X_NH₃/X_CH₄ in accordo con φ e miscela dichiarati e spazialmente uniformi) è solo una condizione **necessaria**. Non può stabilire la premiscelazione perché i canali esportati non includono O₂, N₂, H₂ né specie C intermedie oltre CH₄/CO/CO₂, quindi non si può calcolare né una frazione di miscela (es. Bilger) né un rapporto elementare locale; la frazione molare di carbonio non è conservata nella reazione (cambia il numero totale di moli); un plateau di T vicino a T_ad(φ) coerente col φ globale non distingue premiscelato e non premiscelato | `02_data_audit_report.md`, `00_ricognizione_osservazioni.md` §9.1 (B.2) | Non attribuire alla LES un regime di miscelazione; non usare il punto come premessa di T–φ locale |
| Q1 | Piano di sezione, ricampionamento, SGS, mesh | paper B.2/§3.2: non dichiarati | Contatto con gli autori o lettura del codice HDF5 |
| Q2 | Mappatura `observed` numerico: formula e unità | — | Lettura del convertitore dell'hub o confronto con canali 1/8 su TRAIN |
| Q3 | Allineamento fra OH\* sperimentale e simulato; `t` reali | — | Scaricare un singolo shard reale (da autorizzare) |
| Q4 | Verifica indipendente dell'esportazione indice↔tempo e duplicati sulle altre traiettorie | — | Proporzionata all'uso (§9): confronto con raw HDF5/autori; controllo di continuità sui soli frame effettivamente usati |
| Q5 | Controllo di `t` sui 7 shard TEST | — | Rinviato: i `t_start/t_end/dt` del manifest sono preesistenti |
| Q6 | Che cosa contengono, quanti e in che formato i dati motore PoliMi | — | Domande a PoliMi/D'Errico tramite Enrico |

---

## 8. Fonti e stato della lettura

| ID | Fonte | Parti lette | Localizzatore |
|---|---|---|---|
| S01 | Hu et al., RealPDEBench, arXiv:2601.01829 (ICLR 2026) | **Non riletto oggi.** Citato da `docs/nonideal/00_ricognizione_osservazioni.md` §1, §9.1 (HTML v2: §3.2, App. B.1, B.2, Tab. 7 verbatim; B.3, parte combustione B.4, D.2 solo sintesi) | [A] |
| S02 | Card, `channels.json`, `dataset_info`, `state`, indici, parametri (copia del 13 nov. 2026) — hub `AI4Science-WestlakeU/RealPDEBench` v2.0.1 | Letti oggi | `data/manifests/hf_small_metadata/` |
| S03 | Inventario remoto 2026-11-13 | Letto oggi (campi `entries`, `summary`) | `data/manifests/remote_file_inventory.json` |
| S04 | `docs/month1/01_remote_reconnaissance.md` | Letto per intero | — |
| S05 | `docs/month1/02_pilot_data_audit_report.md` | Letto per intero | — |
| S06 | `docs/month1/02_data_audit_report.md` | Letto per intero | — |
| S07 | `docs/month2/mission_baseline_timestamp_audit.md` | Letto per intero; solo `t` ricontrollato oggi (L4), duplicato di campo non rieseguito | + `results/baseline_consolidation/timestamp_diagnosis_40NH3_1.2.json` non letto |
| S08 | `docs/nonideal/00_ricognizione_osservazioni.md` | Letto per intero | — |
| S09 | `docs/RESEARCH_LOG.md` (voce 2026-11-25), `docs/proposal.md` (§1–5), `mese3/*` | Letti | — |
| S10 | `roadmaptests.md` (§2), `docs/month2/00_context_and_decisions.md` (§§1–5) | Letti nelle parti citate | — |
| S11 | Codice: `src/combustion_recon/data/shard_access.py`, `temporal_quality.py`, `t_anchors.py` (intestazioni) | Letto nelle parti citate | — |
| S12 | Manifest: `operating_conditions.yaml`, `full30_audit_summary.json`, `month2_dense_unet_cache_manifest.json`, `month2_sparse_eval_masks_manifest.json`; cache locali C149/A38 | Letti/ispezionati (L5) | `data/manifests/`, `data/processed/` |
| S13 | `configs/splits/splits_interp_extrap.yaml`, `month2_frame_protocol.yaml` | Letti | — |
| S14 | EngineBench: arXiv:2406.03325; `github.com/sambkr/EngineBench` | Non riletti oggi; da S08/S10 | [A] |
| S15 | TNF/Sandia, Sciacchitano et al. | Non riletti oggi; da S08 | [A] |
| S16 | Nie et al. (2024); Liu et al. (2025); Sciacchitano (2019); DMF-Gen | **Non letti** (fuori perimetro: altre missioni) | — |

**Registro delle ricerche.** Ricerca di termini `laminar|RANS|URANS|engine|PoliMi|Politecnico` in documenti, configurazioni e codice della repository (esclusi `external`, `overleaf`): occorrenze solo in proposta, roadmap, diario, `roadmaptests.md` e `docs/month1`; nessun dato. Stesso tipo di ricerca in `master-thesis`, `thesis-research` (solo "engine" come sottostringa di "Engaging"). Non è stata eseguita alcuna ricerca web, di rete o su cluster.

---

## 9. Implicazioni pratiche

**Supportati dall'evidenza attuale**
- Ricostruzione/studio di T o di altre variabili numeriche da osservazioni sintetiche di velocità planare su stati LES istantanei, con split per condizione operativa (train 20, val 3) e cache già disponibili. [D]
- Stress test con degradazioni sintetiche dichiarate come tali (rumore, lacune, medie di finestra); non sono misure PIV qualificate. [D][A]
- Uso a singolo frame, per indice, delle condizioni TRAIN/VAL, compreso `40NH3_1.2` (consentito dalla policy attuale). [D][A]
- Uso di T, velocità e specie LES come quantità numeriche distinte da OH\*. [D]

**Richiedono ulteriori verifiche**
- **Usi multi-frame, forecasting, sensori lenti o ritardi:** nelle 22 condizioni regolari controllate non c'è evidenza di anomalia temporale; resta non verificata in modo indipendente l'ipotesi "`t[i]` è il tempo del frame `i`". Non è un divieto: verifiche proporzionate e ipotesi da dichiarare, in funzione dell'uso:
  1. dichiarare l'ipotesi indice↔tempo nel protocollo dell'esperimento;
  2. per le traiettorie usate, confermare la regolarità di `t` sulla finestra impiegata (già fatto per le 22 non-TEST; da fare per eventuali altre);
  3. per finestre di forecasting, un controllo di continuità sui soli frame realmente usati (frame ripetuti, salti improvvisi), a costo contenuto e senza scansione completa;
  4. per sensori lenti, dichiarare la scala temporale: se è ≫ Δt=0,25 ms, un errore di pochi campioni ha effetto trascurabile [I, da verificare caso per caso];
  5. se l'esito dipendesse in modo decisivo dal tempo assoluto, confrontare con il raw HDF5 o chiedere agli autori;
  6. `40NH3_1.2`: la policy attuale rifiuta il multi-frame (non modificata qui); per usi temporali la scelta è escluderla o dichiarare esplicitamente la lacuna 1098→1099.
- **Fiducia nell'orientamento** (bruciatore, segno di vj, vk come swirl): non dedurre fisica da C1.
- **Regime di miscelazione e legame T–φ locale:** non determinati; da stabilire con fonte primaria prima di usarli (C9).
- **OH\* e ricostruzione ottica:** scaricare e ispezionare almeno uno shard reale; caratterizzare `observed` (Q2, Q3). Senza dati simultanei di T non c'è validazione sperimentale della ricostruzione.
- **Uso dei 7 TEST:** invariato; non toccare. La regolarità di `t` in quegli shard non è stata controllata.
- **Dati motore RANS/URANS:** nessun accesso individuato nel perimetro consultato; ogni affermazione resta M finché PoliMi non descrive i dati. Un target RANS medio non validerebbe stati istantanei.
- **Validazione sperimentale:** oggi nessuna per T e velocità; il confronto sperimentale possibile è solo statistico/integrato con OH\*.
- **Equivalenza dei dataset:** non confrontare numeri fra A38 e C149, né 3C storici con 2C.

**Prossime verifiche (da autorizzare separatamente)**
1. Colmare la lacuna di tracciabilità SHA256 (C4): registrare gli hash locali per-shard; il ricalcolo completo comporta la lettura di ≈63 GB, quindi da autorizzare.
2. Un solo shard reale OH\* per scala, `t`, FOV e confronto qualitativo con `observed` numerico.
3. Confronto di frame adiacenti su TRAIN/VAL per i duplicati, proporzionato all'uso (C5).
4. Domande agli autori su piano della sezione, ricampionamento e SGS.
5. Domande minime a PoliMi sul contenuto dei dati motore.
