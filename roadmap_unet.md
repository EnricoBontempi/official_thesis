# Roadmap U-Net — Diagnostico da velocità densa a temperatura istantanea

**Progetto:** esclusivamente `EnricoBontempi/official_thesis`, MIT/Deng Lab.  
**Preparata:** 2026-11-17, America/New_York.  
**Stato:** su `mission/dense-unet`, **U0–U6 (p_off seed 0 + snapshot control) eseguiti**; altri seed U6 **rinviati**; U7 / chiusura missione da discussione. Ratifica supervisori **non** documentata/non inventata. Revisione Opus esterna su U4/U5: «utilizzabile con riserve» (non ratifica supervisori).  
**Piano principale:** [roadmap_month2.md](roadmap_month2.md), dettaglio dello **Step 2.4**, non roadmap concorrente del Mese 2.  
**Missione precedente:** [roadmap_missione.md](roadmap_missione.md); consolidamento completato sul branch dedicato, integrazione in main ancora pendente alla ricognizione.  
**Esecutore:** Cursor IDE, un incarico delimitato alla volta, con restituzione dei risultati a Enrico/ChatGPT.

## 1. Domanda, ambito e conclusioni ammissibili

**Domanda:** con tutti i valori istantanei di vi/vj sulla griglia, una U-Net ricostruisce T meglio della media parametrica e di GappyPOD denso? Quanto del risultato dipende dall'allineamento fra velocità e temperatura della stessa istantanea?

Il problema finale resta **poca velocità → tutta T**, su condizioni operative escluse dall'addestramento. Questa missione rimuove temporaneamente la sparsità per studiare il primo modello non lineare. Predice valori numerici di temperatura, non soltanto colori né campi medi. T è disponibile per supervisione, ma assente dagli input.

Un risultato positivo è evidenza empirica sul modello/dati/condizioni valutati; non è un information ceiling, una dimostrazione causale o garanzia di successo sparse. Un risultato negativo non dimostra che l'informazione fisica sia assente. Un vantaggio sui seen ma non su VAL suggerisce di indagare la generalizzazione, senza diagnosticarne automaticamente l'unica causa.

**Fuori ambito:** masked U-Net, implementazione/adattamento GLU, sensori T, AO/teste residue, FiLM, prior ParamMean nel forward, 3C, finestre temporali, forecasting, loss PDE, incertezza, R=64, nuovi split/fold e campagne TEST. Sono attività successive da decidere usando i risultati.

## 2. Provenienza e autorizzazioni

### 2.1 Stato remoto controllato alla preparazione

| Riferimento | SHA | Significato |
|---|---|---|
| main | `f34ce8c60d0a381ad830f960b8f1c0b4cf44dadb` | Base documentale prima di questa roadmap |
| mission/baseline-consolidation | `3cf7bf03f041369a976ee8c9994ded21a8870daf` | Dipendenza tecnica aggiornata, non ancora integrata |
| Freeze replica | `2c800cedd1cccc6935eb64c6af57fbdd6001aee8` | Codice effettivamente eseguito nella replica |
| Risultati replica | `129f2982100437654da9dadf59588f2d86c59a35` | Tabelle/report della replica |

[Report della dipendenza al commit verificato](https://github.com/EnricoBontempi/official_thesis/blob/3cf7bf03f041369a976ee8c9994ded21a8870daf/docs/month2/mission_baseline_consolidation_report.md). La replica riporta 66.180 righe per-frame e differenza massima delle metriche pari a zero rispetto allo storico; non è stata rieseguita durante la preparazione di questo piano.

Su tutto `40NH3_1.2.h5` l'associazione campo-tempo è non verificata; i=2000 duplica i=1999, non selezionato nel seen corrente. Conservare campi, indici e flag; niente dedupe, esclusioni o tempi inventati. L'esperimento è single-frame per indice.

### 2.2 Mandato e fonti

Enrico ha scelto di procedere con il diagnostico U-Net e richiesto questa roadmap e il push. La proposta dettagliata Opus è stata fornita in conversazione il 17 novembre; questa roadmap ne registra le scelte operative e le precisazioni discusse con ChatGPT. Non attribuire ai supervisori né ad Astra una revisione di questo piano U-Net.

La pubblicazione del piano non significa implementazione eseguita o autorizzazione a percorrere tutte le fasi automaticamente. L'incarico successivo a Cursor specificherà la fase attiva. Gli stop sono punti di esame degli output nel workflow concordato, non nuove approvazioni generiche per azioni di routine.

### 2.3 Gerarchia documentale

Split, frame, stats e isolamento restano quelli del protocollo comune. Per il dettaglio del solo Step 2.4, questo documento sostituisce il precedente suggerimento FiLM con concatenazione e rinvia ogni opzione temporale. In caso di conflitti scientifici ulteriori, registrare il conflitto e fermare il run interessato, senza modificare silenziosamente il contratto storico.

## 3. Contratto dell'esperimento

### 3.1 Dati e significato degli input

- Griglia 128×128. **vi/vj = canali 11/12**, senza assumere automaticamente la loro equivalenza agli assi fisici x/y. T target = canale 10, grezza, senza clipping.
- `temperature_free`, K=0, `site_fraction=1.0`: tutta la velocità planare osservata, nessun valore T in input.
- Sei canali ordinati: `[vi_norm, vj_norm, x_scaled, y_scaled, nh3_norm, phi_norm]`.
- x/y da coordinate ammesse nel sample, trasformazione geometrica fissa in [-1,1]. Verificare forma, orientazione e identità delle griglie fra TRAIN/VAL; non basta avere gli stessi estremi. Non leggere payload TEST per questa verifica.
- p_on principale: frazione NH3 nella miscela NH3/CH4 e rapporto di equivalenza phi, normalizzati con la funzione esistente e ripetuti sulla griglia. Non sono campi locali di specie.
- p_off: stessa architettura con i due canali p a zero, nessun p letto da metadata o prior. Restano le stesse stats globali fit-only, non stats per condizione.
- Ingressi costruiti da `model_inputs` con allowlist esplicita; T solo nel percorso target/loss/evaluator. Esclusi vk, velocità 3D derivata, specie, heat release, sim_id, indice frame, tempo e split.
- Maschera densa tutta vera verificata, non aggiunta come canale costante. Nessuna feature ricavata dalla T target.

### 3.2 Split e preprocessing

| Ruolo | Campioni | Uso |
|---|---:|---|
| train_fit | 20×38 = 760; indici 0…1480 passo 40 | Gradienti e fitting |
| seen_train_eval | 20×11 = 220; indici 1600…2000 passo 40 | Diagnostica, mai selezione |
| val_eval | 3×41 = 123; indici 0…2000 passo 50 | Selezione dichiarata |
| TEST | Nessun payload | Fuori missione |

Usare `configs/splits/month2_frame_protocol.yaml` e `data/manifests/train_norm_stats_month2_fit.json`, verificandone fingerprint e provenienza. Niente refit con seen/VAL. La nuova config densa deve referenziare il protocollo senza modificare il contratto storico sparse al 5%.

### 3.3 Architettura iniziale concreta

| Parte | Specifica di partenza |
|---|---|
| Encoder / risoluzioni | 128²: b; 64²: 2b; 32²: 4b; 16²: 8b |
| Bottleneck | 8²: 8b, dopo il quarto max-pool ×2 |
| Blocco | Due Conv 3×3, padding zero 1, ciascuna seguita da GroupNorm(8 gruppi) e SiLU |
| Decoder | Quattro upsampling bilineari ×2, align_corners=False, concatenazione skip e doppio blocco conv; uscite 8b, 4b, 2b, b |
| Testa | Conv 1×1 lineare, 1 canale T normalizzata |
| Condizionamento | Solo concatenazione dei canali p all'ingresso |
| Larghezza candidata | b=32; b=16 alternativa di costo decisa dal profiling, prima di valutare prestazioni scientifiche |

Misurare parametri/MAC se disponibili; le stime Opus (~4,9M per b=32) non sono numeri certificati. Registrare architettura esatta, bias delle conv e convenzione di conteggio. Niente dropout o augmentation iniziali.

**Inizializzazione:** conv interne Kaiming compatibile con la convenzione implementata; testa con pesi piccoli ma non tutti nulli (std=1e-3, bias=0), seed registrato. Output iniziale vicino alla media fit, non necessariamente identico. Questo evita l'incompatibilità tra testa tutta zero e richiesta di gradienti non nulli nei blocchi precedenti al primo backward.

### 3.4 Training: default e ciò che va fissato dopo profiling

- MSE su T normalizzata, uniforme su tutti i pixel e batch. Valutazione in K con inversione delle stats fit-only.
- AdamW, lr=1e-3, weight_decay=1e-4; **warmup lineare 950 aggiornamenti**, poi **ReduceLROnPlateau** su RMSE macro-condizione VAL (factor 0.5, patience 5 eval, threshold rel 0.001, min_lr 1e-5). **Niente cosine** legato al massimo di epoche. Gradient clipping norma 1.
- fp32 senza TF32 per il primo confronto; batch 8; shuffle riproducibile; num_workers=0 sulla cache compatta.
- **Budget iniziale autorizzato Enrico:** massimo **1000** epoche su Engaging (1 GPU non preemptable AMF se accessibile). Continuazione oltre 1000 solo con decisione successiva. Nessun fallback automatico lr=3e-4.
- Valutazioni VAL/seen/**TRAIN** ogni 10 epoche e all'ultima; selezione solo VAL (primo minimo RMSE macro-condizione); conservare anche last. Seen/TRAIN non selezionano.
- Early stop operativo: non prima di 200 epoche; 15 eval VAL consecutive senza miglioramento relativo 0,1%; reset pazienza se il LR scende; best assoluto conservato anche sotto soglia 0,1%. Convenzione operativa, non certificazione fisica.
- Autorizzato soltanto **p_on seed 0**. p_off / altri seed fuori da questo incarico.

## 4. Step numerati e punti di arresto

Tutti gli step sono inizialmente **[ ] non eseguiti**. Codice presente non equivale a esperimento completato.

### U0 — Allineare checkout e dipendenze (preparazione)

**Problema / motivo:** main non contiene ancora le correzioni da cui dipende il nuovo lavoro.  
**Dipendenze:** lettura istruzioni, diario, questa roadmap, protocollo e report della missione precedente.

**Attività**
1. Git status/fetch, verifica dei due branch e preservazione di modifiche concorrenti. Niente force/reset hard/clean distruttivo.
2. Creare un branch dedicato, proposta `mission/dense-unet`, contenente sia questa roadmap sia la dipendenza `3cf7bf0` o suoi discendenti verificati. Integrare le storie sul branch di lavoro risolvendo README/diario/roadmap preservando entrambe; non serve alterare main in questo incarico.
3. Verificare Python, torch, raw locali, RAM/disco, CPU/GPU realmente disponibili. Non scaricare raw né presumere Engaging.
4. Registrare commit di partenza e dipendenze; ricercare prima eventuali implementazioni concorrenti.

**Output:** preflight e stato Git nel futuro report `docs/month2/03_dense_velocity_diagnostic.md`.  
**Verifica / completamento:** dipendenze presenti, nessuna modifica altrui perduta, raw accessibili.  
**Passaggio:** U1; in mancanza raw distinguere test sintetici da controlli reali non eseguiti.

### U1 — Configurazione e cache densa verificabile (implementazione)

**Problema / motivo:** leggere gli shard Arrow a ogni epoca sprecherebbe tempo e RAM; la cache non deve introdurre leakage.  
**Dipendenze:** U0 e pipeline Month-2.

**Attività**
1. Nuova config `configs/experiments/month2_dense_unet.yaml` con experiment_id distinto; config modello `configs/models/unet_small.yaml`. Consumare valori o rifiutarli esplicitamente.
2. Cache costruita via dataset esistente: ruoli separati, velocità e target separati, metadata separati; cache locale ignorata da Git. Conservare il minimo necessario, non sei mappe duplicate per ogni frame.
3. Manifest: ID/ruoli/indici, ordine canali, geometria, dtype, hash split/protocollo/stats/config, commit, provenance shard e flag temporali. Verificare fingerprint al riuso.
4. Prefetch per shard, evitando cache thrashing e parallelismo Arrow aggressivo. Misurare byte reali della cache e decode, senza promettere dimensioni teoriche.
5. Verifica indipendente di alcuni valori contro sample del dataset. Momenti normalizzati TRAIN compatibili con la definizione delle stats, tolleranza numerica e convenzione di std; non pretendere uguaglianza esatta in float32.

**Output:** `src/combustion_recon/training/dense_cache.py`, config e piccolo cache manifest versionato; payload in data/processed ignorato o percorso configurabile.  
**Verifica / completamento:** conteggi 760/220/123, nessun TEST, fingerprint validi, coordinate e maschera densa controllate, nessuna stats rifittata.  
**Passaggio:** U2; cambiare campioni o normalizzatore richiede una decisione distinta.

### U2 — Rete, trainer e verifiche piccole (implementazione)

**Problema / motivo:** separare bug e mancato apprendimento tecnico dai risultati scientifici.  
**Dipendenze:** U1.

**Attività**
1. Implementare `models/unet.py`, `training/trainer.py`, `scripts/07_train_reconstructor.py`, riusando evaluator e normalizzatore.
2. Forward (B,6,128,128)→(B,1,128,128); loss/gradienti finiti e flusso del gradiente ai blocchi. Modalità train/eval corrette.
3. Mutation tests: alterare target T, vk e metadata senza cambiare input ammessi non cambia il forward in eval. Controlli positivi su vi e p dopo piccoli aggiornamenti; p_off realmente indipendente da p. Non richiedere che ogni singolo peso abbia gradiente non nullo.
4. Overfit su 8 frame train_fit di 2–4 condizioni, seed e ID registrati: fino a 500 aggiornamenti, salvo tetto di wall time di 30 minuti. Documentare loss iniziale/finale, mappe e confronto ParamMean sugli stessi sample. Se non c'è apprendimento netto o il tetto scade, diagnosi e STOP; non iniziare tuning automatico.
5. Checkpoint completo (modello, optimizer, scheduler, RNG, step/config/stats hash); reload con predizioni coerenti sullo stesso ambiente. Se supportato resume, provarne la continuità.
6. Testare evaluator in K e aggregazione fail-fast: ruoli separati e identità completa incluse variante, seed e velocità corretta/scambiata.
7. Preparare il controllo U5 di scambio istantanee senza analizzarne ancora risultati VAL.

**Output:** test `tests/test_unet_dense.py` ed eventuali test cache/trainer, log smoke/overfit e report aggiornato; checkpoint fuori Git.  
**Verifica / completamento:** test appropriati passati, isolamento reale, apprendimento dimostrato sul piccolo subset, nessun risultato VAL usato per modificare il modello.  
**STOP A:** consegnare commit, test, overfit, parametri effettivi, limiti. Il primo incarico Cursor termina qui, salvo limite più stretto nel prompt.

### U3 — Profiling e riferimento lineare denso (esperimenti preparatori)

**Problema / motivo:** fissare costo fattibile e un confronto con le stesse osservazioni prima del training scientifico.  
**Dipendenze:** esame di U2 e incarico successivo esplicito.

**Attività**
1. Profilare 1–2 epoche train_fit e una valutazione, misurando tempi, RAM/VRAM, cache e thread. Le metriche VAL del profiling non sono usate per scegliere architettura o lr; niente mappe/prestazioni per guidare il pilot.
2. Scegliere b=32 o 16, batch, dispositivo, E_max e limite wall in base a costo/stabilità TRAIN. Stimare costo totale inclusi eval, p_off e repliche possibili. Salvare la decisione prima del run.
3. CPU ammessa se fattibile. GPU locale/Engaging solo dopo verifica effettiva; nessun accesso cluster o costo esterno implicito. Le 24 GPU-ore sono ceiling proposto, non allocazione.
4. Eseguire GappyPOD free denso, K=0, R=32, p_on/off sugli stessi frame/2C/stats. Nuovo experiment_id/config e runner separato o estensione che non alteri il contratto sparse storico.
5. Griglia lambda=[1e-4,1e-3,1e-2,1e-1,1], selezione VAL esistente e tutti i trial registrati. La riselezione è un controllo matched; 1/Mu non rende automaticamente invalido il lambda sparse.
6. A copertura 100% la maschera è unica: nessuna triplicazione artificiale di risultati identici per replica maschera.
7. Riutilizzare ParamMean/GlobalMean consolidati soltanto dopo verifica di corrispondenza frame/fit/stats; LocalHGB denso resta diagnostico secondario.

**Output:** `scripts/08_dense_linear_reference.py` (o equivalente motivato), `configs/experiments/month2_dense_linear.yaml`, tabelle/search/manifest sotto `results/month2_dense_linear/<run_id>/`, profiling e decisione budget nel report.  
**Verifica / completamento:** comparatore matched riproducibile, nessun TEST o artefatto storico sovrascritto; risorse misurate e budget definitivo registrato.  
**STOP B:** esaminare costi e protocollo congelato prima di U4. Non promettere minuti di CPU/GPU senza misure.

### U4 — Primo training p_on, seed 0 (esperimento scientifico)

**Problema / motivo:** valutare il primo ricostruttore non lineare denso nel contratto fissato.  
**Dipendenze:** U3 concluso, incarico di run e budget registrato.

**Attività**
1. Committare codice/config prima del run, checkout pulito; manifest con HEAD e hash sorgenti/config/stats/cache, versioni e dispositivo.
2. Nuova inizializzazione seed 0: niente pesi o optimizer riusati da overfit/profiling.
3. Addestrare soltanto train_fit, con selezione VAL dichiarata. Registrare tentativi e interruzioni; nessun tuning occulto o sostituzione dei run deboli.
4. Salvare curve, migliore e ultimo checkpoint e output per-frame/condizione/split in K. Valutare train_fit come diagnostico separato; seen non seleziona nulla.
5. Misurare wall time, aggiornamenti, picco RAM/VRAM e costo di selezione.

**Output:** `results/month2_dense_unet/<run_id>/` con preflight, manifest, curves, selection, per_frame.csv.gz, per_condition, summary e auto_report; checkpoint fuori Git con percorso persistente/hash.  
**Verifica / completamento:** run completato o interrotto esplicitamente, valutazione riproducibile, risultati completi e nessun cambio di config durante il run.  
**Passaggio:** U5 sul checkpoint selezionato, senza ulteriore ottimizzazione.

### U5 — Controllo delle istantanee e lettura dei risultati (esperimento/decisione)

**Problema / motivo:** distinguere una buona mappa tipica dall'uso della velocità specifica dell'istante.  
**Dipendenze:** U4.

**Attività**
1. Per ogni condizione e ruolo di eval separatamente, ordinare i frame selezionati e sostituire vi/vj del frame j con quelli di (j+floor(n/2)) mod n. Coordinate, p e target restano del destinatario.
2. Nessun punto fisso, nessuno scambio tra condizioni/ruoli; pairing deterministico salvato con indici sorgente/destinatario. Nessun re-training o selezione sul controllo.
3. Non convertire scarti di indice in tempi fisici per lo shard non verificato. Il wrapping produce separazioni diverse; non descrivere tutti i pairing come un intervallo unico.
4. Confrontare corretto/scambiato per frame e condizione: RMSE, bias, |bias|, Centered-RMSE; regionali getto T<1800 K e hot T>=2100 K, coverage/count e casi vuoti.
5. Tabelle matched con ParamMean/GlobalMean e GappyPOD denso. Sparse 5% eventualmente solo in tabella distinta, dichiarando più informazione nel caso denso.
6. Mostrare predizione/verità/errore con scale coerenti su frame fissati prima dei risultati (es. indice 1000 per ciascuna VAL), ed errori medi; non scegliere solo immagini riuscite.
7. Riportare media macro per condizione, tutte le condizioni VAL, seen e peggioramenti. Nessuna regola “vince su tutte e tre o non esiste miglioramento”. Un seed non prova robustezza; VAL è stata usata per selezione.
8. Scambio che peggiora: evidenza di uso dell'allineamento istantaneo. Scambio che cambia poco: nessuna evidenza forte con questo controllo, non prova che la rete ignori la velocità.

**Output:** pairing, tabelle, figure e interpretazione editoriale separata dall'auto_report nel report 03.  
**Verifica / completamento:** confronto paired completo, nessuna contaminazione o inferenza di information ceiling.  
**STOP C:** discutere risultati con Enrico/ChatGPT prima di p_off, altri seed o cambi modello.

### U6 — Ablation p_off e ripetizioni delimitate (esperimenti successivi)

**Problema / motivo:** misurare il contributo dei parametri noti e qualificare la variabilità del training.  
**Dipendenze:** U5 esaminato e scelta registrata; nessun obbligo di eseguire tutto automaticamente.

**Attività**
1. p_off seed 0: stessa rete (canali p zero), dati, normalizzazione globale, budget e selezione, nuova inizializzazione. Confronto matched con GlobalMean e GappyPOD denso p_off.
2. Ripetere U5 per p_off, con gli stessi pairing. Non usare ParamMean(p) nel suo forward.
3. Per consolidare un confronto p_on/p_off, preferire seed accoppiati 0/1/2 per entrambi se fattibile. Se si ripete soltanto p_on, dichiarare che la variabilità di p_off rimane non misurata.
4. Riportare risultati individuali e dispersione fra seed senza trasformare tre seed o tre condizioni in una certificazione statistica.
5. Esito debole: prima diagnosi di implementazione/convergenza/generalizzazione, poi al massimo un esperimento correttivo motivato e deciso separatamente. FiLM, residuo ParamMean, più frame o capacità diversa non scattano automaticamente da una soglia di bias.

**Output:** run card e tabelle aggiuntive; decisione esplicita su ablation/seed eseguiti o rinviati.  
**Verifica / completamento:** parità del confronto documentata, variabilità qualificata e nessuna estensione nascosta.  
**Passaggio:** U7 anche con optional rinviati, con limiti dichiarati.

### U7 — Chiusura della missione e prossimo incarico (documentazione/decisione)

**Problema / motivo:** chiudere una domanda senza confondere software pronto, risultati scientifici e fase Mese 2 completa.  
**Dipendenze:** U5 almeno, stato U6 registrato.

**Attività**
1. Report 03 con domanda, config, risultati, controlli, costi e limiti; collegare commit eseguiti, run ID, checkpoint e comandi riproducibili.
2. Aggiornare questa checklist, roadmap Mese 2, registro D4/D8, README e diario. Ogni stato ha evidenza e data.
3. Consegnare branch/commit e PR se disponibile; integrare solo nell'ambito dell'incarico ricevuto. Se il client gh non funziona, riportare compare link senza dichiarare PR aperta.
4. Proporre il prossimo incarico: U-Net sparse (2.6) e audit GLU (2.5), entrambi verso T e con stessi input nel futuro confronto. Nessuna architettura ibrida implicita.

**Completamento minimo:** U0–U5 verificati, report e consegna; p_off/seed o correttivi rinviati esplicitamente se non fatti. Per dichiarare “confronto p_on/off completato” serve U6 pertinente; per dichiarare robustezza servono repliche appropriate.  
**Esiti ammessi:** conclusa tecnicamente con risultato positivo/misto/negativo; preliminare per variabilità non misurata; parziale/bloccata con ragione. Nessuna vittoria obbligatoria.  
**Limite finale:** chiudere questa missione non chiude il Mese 2 e non autorizza training GLU/sparse/assisted.

## 5. Regole di tracciabilità e limiti

- Versionare config, test, manifest, tabelle compatte e report; non raw/cache/checkpoint o per-frame enormi non compressi. Non sovrascrivere `consolidation_v1`, `results/tables/`, proposal, documenti inviati o `external/GLU_Reconstruction/`.
- Fingerprint scientifico di aggregazione distingue ruolo, metodo/variante, p, K, copertura, seed, architettura/config, checkpoint e controllo aligned/swapped. Parametri POD non applicabili esplicitamente null/N/A secondo schema, senza falsi valori fisici.
- Normalizzazione, fitting e loss soltanto train_fit; VAL può valutare e selezionare secondo piano ma non fare gradienti o fit di preprocessing.
- TEST già esaminati storicamente: limite esplicito, non un test mai visto. In questa missione nessun payload TEST né freeze per abilitarli.
- 760 frame correlati, 20 condizioni TRAIN, una traiettoria per condizione; seen non è replica indipendente. VAL ha solo tre condizioni e non copre regimi ricchi/high-NH3. Non generalizzare i risultati a tutto l'inviluppo operativo.
- Due componenti di velocità su una slice 2D non chiudono la fisica 3D reattiva; nessuna loss PDE introdotta per deduzione.
- Prima di commit/push rileggere remoto, preservare concorrenti; push fast-forward con controllo dello SHA remoto, mai force. Non committare modifiche altrui per ottenere stato pulito.
- Nessuna risorsa cluster, download pesante o training è stato avviato per scrivere questo piano.

## 6. Checklist iniziale

- [x] U0 — Checkout con roadmap e consolidamento aggiornato. (branch `mission/dense-unet`; evidenza in `docs/month2/03_dense_velocity_diagnostic.md`)
- [x] U1 — Contratto denso e cache verificati. (760/220/123; payload gitignored; stub `data/manifests/month2_dense_unet_cache_manifest.json`; validazione rafforzata post-STOP A)
- [x] U2 — Implementazione/test/overfit consegnati. **STOP A.** + correzioni isolamento/grad/YAML; Step 2.4 **non** dichiarato completo
- [x] U3 — Profiling e GappyPOD denso; budget proposto. **STOP B.** (U4 non avviato)
- [x] U4 — Primo training p_on seed 0 eseguito su Engaging (job `29465586`, L40S, early stop ep. 730, best VAL 26.93 K @160).
- [x] U5 — Controllo istantanee (job `29471596`): aligned replay OK; VAL 26.93→96.55 K allo scambio. **STOP C.**
- [x] U6 — p_off seed 0 + snapshot control eseguiti (best VAL 39.67 K @30; swap VAL +62.7). Altri seed **rinviati**.
- [ ] U7 — Report finale e scelta del prossimo incarico.
