# Roadmap sparse — U-Net e GLU adattato per temperatura istantanea

**Progetto:** esclusivamente `EnricoBontempi/official_thesis`, MIT / Deng Lab.  
**Data:** 2026-11-18, America/New_York.  
**Base verificata:** `87d1e3f40826a9343c3c2ce19d140d481b911619`, chiusura U7, branch `mission/dense-unet`.  
**Stato:** S0–S3 chiusi; S4 profiling COMPLETED; freeze neurale S5 **definitivo** e **submitted**; GappyPOD 1/20% ancora bloccato (S4 non tutto chiuso).  
**Piano principale:** [roadmap_month2.md](roadmap_month2.md). Questo documento dettaglia **2.5–2.6 e la sola conferma sparse pertinente di 2.8**; non chiude il Mese 2 e non autorizza assisted o TEST.  
**Esecutore:** Cursor, per incarichi delimitati successivi. Pubblicare il piano non autorizza a percorrerlo automaticamente.

## 1. Domande e punto di partenza

**Obiettivo:** ricostruire tutto il campo istantaneo T da poche misure co-locate di vi/vj, su condizioni operative escluse dal training. Separare prestazioni dei metodi, contributo dei parametri noti e sensibilità a copertura/layout.

Il diagnostico [U-Net denso](roadmap_unet.md) è chiuso, con report [03](docs/month2/03_dense_velocity_diagnostic.md). VAL best: U-Net p_on 26.93 K, p_off 39.67 K; GappyPOD denso p_on 70.46 K, p_off 93.66 K; ParamMean 88.01 K. Fra i last il gap neurale riguarda soprattutto il livello (p_on/p_off RMSE 28.06/44.10; |bias| 8.91/30.45; Centered-RMSE 24.44/28.94 K). I controlli swap indicano predizioni specifiche dell'istantanea.

Sono risultati a **una coppia di seed**, selezionati su VAL; repliche dense rinviate. Il vantaggio cambia insieme rappresentazione, stimatore, capacità e apprendimento: non dimostra necessità universale della non linearità. L'oracle POD riguarda il sottospazio scelto, non tutti i ricostruttori. Il divario dense→sparse sarà un costo empirico della ricetta, non un information ceiling né una decomposizione pura della perdita informativa.

Domande della fase:
1. Una U-Net allenata allo sparse supera le baseline sullo stesso pacchetto di osservazioni?
2. Quale vantaggio/costo ha GLU adattato rispetto alla U-Net?
3. Quanto aiutano NH3 e phi con poche velocità?
4. Che cosa cambia a 1/5/20%, e quanto dipende da layout e inizializzazione?
5. Le predizioni restano specifiche dell'istantanea?

**Fuori ambito:** ancore T/AO neurali (2.7), TEST/pilot (2.8 separato), forecasting, finestre temporali, 3C, PDE loss, FiLM, prior ParamMean nel forward, incertezza/Beta, sweep architetturali, nuovi split, training misto di densità o interpolazioni iniziali. Non modificare HONF o `external/`.

## 2. Provenienza, decisioni e gerarchia

Enrico ha discusso due proposte indipendenti, Opus e Astra, basate su `87d1e3f`, e approvato la sintesi in conversazione il 18 novembre. Le proposte sono consulenze, non ratifiche dei supervisori. I revisori non avevano raw/cache/checkpoint; Opus riferisce riaggregazioni per-frame e controlli NumPy, Astra controlli sintetici e macro per condizione. Non equivalgono a re-inferenza indipendente. I rilievi GLU da lettura richiedono test nel nostro ambiente.

**Scelte operative approvate:** una sola roadmap; U-Net sette canali da zero; maschere persistite condivise e annidate; quattro screening p_on/off per entrambe le famiglie se tecnicamente fattibili; GLU deterministico con condizionamento additivo; batch GLU omogenei senza padding iniziale; seed distinti; audit e profiling prima del freeze scientifico; repliche mirate.

**Scelta risorse esplicita di Enrico:** nessun tetto arbitrario di ore per run o campagna; non applicare il precedente ceiling proposto di 24 GPU-ore a questa missione. Misurare e rendere visibile il costo, arrestare su criteri operativi di plateau e discutere continuazioni sulla base delle curve. Questo non cambia retroattivamente i limiti dei vecchi incarichi U4/U6.

Gerarchia:
- Month-2 resta fonte di split/frame/stats/metriche e piano generale.
- Questa roadmap è fonte delle decisioni operative sparse e del superamento del ceiling orario proposto.
- Futuro `configs/experiments/month2_sparse_protocol.yaml`: unica specifica eseguibile del contratto; deve corrispondere al testo, senza default scientifici ignorati.
- Futuri report `04_glu_release_adaptation.md` e `05_sparse_reconstruction.md`: evidenze tecniche e risultati, non roadmap alternative.
- Conflitti non risolti: registrarli prima del run interessato. Nessuna riscrittura di proposal o risultati storici.

## 3. Contratto comune e osservazioni

### 3.1 Dati e isolamento

Conservare split e `configs/splits/month2_frame_protocol.yaml`: train_fit 760 (20×38, indici 0…1480 passo 40), seen_train_eval 220 (20×11, 1600…2000 passo 40), val_eval 123 (3×41, 0…2000 passo 50). Fit, normalizzazione, POD e prior solo train_fit. Riusare `train_norm_stats_month2_fit.json` e cache compatta previa verifica dei fingerprint e hash payload.

Griglia 128×128; target T grezza canale 10, nessun clipping; velocità 11/12, senza identificazione automatica con assi fisici. Coordinate ammesse, p = frazione NH3 nella miscela e phi, non campi locali di specie. K=0, 2C, single-frame. Nessun sim_id, indice, tempo, T o vk nel forward. Metadata ammessi solo a registrazione, grouping e costruzione indipendente da target delle maschere.

La cache può contenere campi completi per target e sorgente osservazioni; **estrarre i valori osservati prima di costruire feature**. Niente statistiche per-frame, norme o interpolazioni che leggano velocità nascoste. Costruire griglia e token dalla stessa osservazione canonica, conservando valori fisici e normalizzati distinti. Gli adapter GappyPOD non devono ricevere velocità normalizzate come se fossero m/s.

Su `40NH3_1.2` mantenere flag e indici; niente tempi fisici dedotti, dedupe o esclusioni nuove. Una traiettoria per condizione, frame correlati e solo tre VAL: nessuna falsa indipendenza dei pixel/frame.

### 3.2 Maschere di valutazione persistite e annidate

| Copertura nominale | Siti | Scalari vi/vj | Token sensore GLU |
|---|---:|---:|---:|
| 1% | 164 | 328 | 164 |
| 5% | 819 | 1638 | 819 |
| 20% | 3277 | 6554 | 3277 |

CLS e latenti non sono misure e si contano separatamente.

- Al 5% preservare seed storico `20261116`, layout `random_sparse_primary`, repliche 0/1/2 e identità delle maschere baseline. Rigenerare una volta, confrontare gli hash e **persistire indici ordinati**; non affidarsi in futuro al solo comportamento di NumPy.
- Opus riferisce 3309/3309 corrispondenze (1103 frame nei tre ruoli ×3 repliche). È una verifica da ripetere/documentare in S1, non già eseguita da questa roadmap. Escludere qualsiasi payload TEST.
- Per ogni frame/replica, costruire 1% come sottoinsieme del 5% storico, 20% come sua estensione dal complemento. Usare RNG separati con namespace per copertura; salvare algoritmo/versioni/seed/hash.
- Il 5% resta storico; 1/20% annidati sono nuovi artefatti: nuove valutazioni matched baseline, mai sovrascrivere lo storico.
- Manifest versionato con identità ruolo/condizione/frame/copertura/replica, geometria, conteggi, payload SHA256 e formato. Payload fuori Git, persistente e trasferibile; verificare hash al consumo.
- Distinguere hash della sola maschera da observation hash esistente: verificare la semantica, non confrontare digest di oggetti differenti.
- Tutte le famiglie ricevono stessi siti e valori; join completo per chiave e hash. Manifest di osservazioni dipendente solo da geometria/identità/seed, non da T o modello.

### 3.3 Maschere TRAIN e seed

Primo training soltanto 5%, maschera ricampionata per frame ed epoca con `training_mask_seed`; il runner deve chiamare `set_epoch` anche dopo resume. Dataset order e worker non cambiano il layout assegnato al frame.

Separare `model_seed`, `shuffle_seed`, `train_mask_seed` e `eval_mask_seed`. Per screening: model_seed=0, shuffle_seed=0, train_mask_seed=20261118; eval storico come sopra. Condividere sequenza di maschere e shuffle fra p_on/off e famiglie, indipendentemente da microbatch. Per repliche inizializzazione 1/2, shuffle_seed=1/2; mantenere train_mask_seed fisso per isolare variabilità init/shuffle condizionata alla stessa sequenza di maschere. Questa scelta non stima tutta la variabilità possibile delle maschere TRAIN; una sua replica indipendente richiederebbe un esperimento distinto.

TRAIN ed eval usano namespace distinti, senza riusare intenzionalmente layout eval. Verificare riproducibilità e variazione fra epoche su esempi, non imporre l'impossibilità matematica di una coincidenza casuale.

## 4. U-Net sparse

Input ordinati: `[vi_obs_norm, vj_obs_norm, Mv, x_scaled, y_scaled, nh3_norm, phi_norm]`. Zero normalizzato nei siti nascosti; Mv=1 se vi/vj entrambe osservate. p_off: ultimi due canali zero in tutti i batch; niente prior o normalizzatori per condizione.

Stesso backbone b=32, GroupNorm/SiLU e decoder bilineare del denso, prima conv a sette canali e testa T lineare. Nuova inizializzazione seed registrato; nessun warm start denso. Riusare moduli senza allentare silenziosamente i vecchi assert: config/classe/percorso sparse espliciti, vecchi checkpoint e contratti densi validi. Contare parametri effettivi.

Test prima del run:
- mutazioni velocità nascoste → input bit-identico e predizione invariata in eval;
- mutare T/vk/metadata vietati, e p sorgente in p_off → nessun effetto;
- controllo positivo su valori osservati e su p_on dopo piccolo apprendimento;
- p_on/off stessa inizializzazione nella famiglia e stesso shuffle/layout; p_off zero per ogni batch;
- forward/backward, loss/gradienti finiti, clipping controllato;
- overfit su 8 frame TRAIN con layout fissati come sanity; breve prova separata di apprendimento con layout variabili, senza pretendere perdita nulla per input diversi/incompleti;
- checkpoint/reload e resume continuo vs ripreso a confine d'epoca, inclusi RNG, epoca maschere, scheduler, contatori e history.

Mascherare soltanto il modello denso in inferenza non sostituisce il training sparse e resta fuori dal minimo.

## 5. GLU adattato — ablation deterministica

### 5.1 Confini dell'adattamento

Audit su release congelata in `external/GLU_Reconstruction/`: registrare commit/tree/hash, classi, forme, default effettivi e differenze paper/codice. Conservare upstream intatto; wrapper/estensioni nel pacchetto di tesi.

| Parte | Scelta iniziale |
|---|---|
| Encoder | Riusare struttura DomainAdaptiveEncoder; token vettoriali [x,y,vi_norm,vj_norm], senza T |
| p | Proiezione additiva senza bias: h_i = h_pos + h_val + W_p p; nessun secondo ingresso p al decoder |
| Decoder | SoftDomainAdaptiveReconstructor, output solo T, fusione della release documentata |
| Globale/locale | CLS attivo, top_k=16, importanza uniforme=1 |
| Temporale/incertezza | Nessun propagatore, varianza, Beta, NLL/ELBO o importance appresa |
| Pipeline | Dataset/trainer/evaluator della tesi; escludere loader/trainer/wrapper upstream dal percorso scientifico |
| Bandwidth | per_sensor_sigma=False; kernel condiviso fisso iniziale sigma=0.05 nella scala isotropa definita sotto |
| Capacità candidata | dimensione 32, 4 teste, 4 layer, 64 latenti; controllare factory e default prima del freeze |

Il wrapper non deve accettare target/G_full; T solo nella loss. La supervisione upstream ai sensori non è semanticamente riutilizzabile quando i sensori misurano vi/vj e la testa emette T. L'uso di T TRAIN per loss non è di per sé leakage: il vincolo riguarda informazione inferenziale e split.

p_off usa stesso modulo W_p con p zero; parametri non letti da vie indirette. Encoder upstream ignora U secondo i revisori: verificare, non affidarsi al suo passaggio per condizionare.

### 5.2 Geometria, embedding, padding

Per GLU preservare rapporti fisici: q=2*(r-centro)/L, L=max(estensione_x,estensione_y) della geometria, stesso fattore per entrambi gli assi. Usare coerentemente q per sensori, query, embedding geometrico e distanze locali; registrare L/centro/unità. Nessun fit sui target. U-Net conserva coordinate del diagnostico. È la stessa informazione geometrica in rappresentazioni diverse, da dichiarare. Sigma=0.05 è una scelta convenzionale in queste coordinate, non una lunghezza fisica universale né equivalenza garantita al paper; eventuale cambio prima del freeze motivato, nessun tuning occulto su VAL.

Mantenere inizialmente embedding valori Linear con bias + RMSNorm. Testare vettori collineari a moduli diversi nel range TRAIN, distinguendo embedding e output dopo piccolo overfit. Non presumere collasso esatto. MLP alternativa solo su evidenza e nuova decisione.

Batch iniziali a copertura omogenea: N costante, **nessun padding**. Rifiutare esplicitamente input padded o N misti; non accettare zeri padded come sensori. Non dichiarare supporto padding verificato. Se in futuro serve batch misto, aggiungere validità end-to-end (due cross-attention, mixer, CLS e decoder) e test single/padded e mutazioni padding prima dell'uso. Questo dettaglio delimita il requisito generale 2.5 senza introdurre falsa sicurezza.

Ordinare i token per indice di sito nel wrapper, dopo qualsiasi permutazione in ingresso, e definire tie-break deterministico dei vicini equidistanti. Testare invarianza del risultato a permutazioni sensori e equivarianza rispetto all'ordine query, comprese parità k-NN. Non accettare deviazioni non spiegate come tolleranza generica.

Il parametro per-sensore creato nel forward è segnalato dai revisori come rischio per optimizer e identità dei sensori: confermarlo con piccolo test torch, non presentarlo già verificato eseguendo.

### 5.3 Query e memoria

Encoder una volta per campo; decoder in chunk in eval, candidato 512 query. Test full/chunk su stessi pesi/input con tolleranza dichiarata prima e differenze massime registrate.

Training iniziale MSE su tutte le 16384 query. Se memoria insufficiente: prima microbatch/accumulo a batch effettivo 8, poi chunking con backward correttamente pesato e verifica dei gradienti rispetto alla loss full-field (attenzione al grafo condiviso encoder). Concatenare tutti i grafi non garantisce risparmio. Niente sottocampionamento automatico di query solo per GLU: sarebbe una nuova scelta di supervisione da discutere.

Profilare self-attention sensori O(N²), cdist e aggregazione query×sensori: top-k non elimina automaticamente le matrici dense. 20% può essere più costoso del 5%; non togliere sensori a GLU per farlo entrare. Report esplicito di eventuali limiti, non una falsa prestazione matched.

## 6. Matrice minima e selezione

| Blocco | Modello / p | Nuovi training | Regola |
|---|---|---:|---|
| Medie | GlobalMean off; ParamMean on | 0 | Riuso oggetti/config/fit consolidati |
| Lineare sparse | GappyPOD R=32 on/off | 0 | Riuso 5% dopo join; nuove eval 1/20% annidate |
| Screening | U-Net on, U-Net off, GLU-det on, GLU-det off | 4, seed 0 | Tutti allenati 5%, dopo test/profiling/freeze |
| Snapshot | Quattro best di screening | 0 | 5%, replica 0, seen e VAL; stesso controllo anche per GappyPOD |
| Conferma | Due configurazioni del contrasto prioritario | 4, seed 1/2 | Decidere dopo screening; riportare tutti i seed |
| Denso | U4/U6 | 0 | Tabella separata: più informazione, seed singolo |

Quattro screening sono la destinazione pianificata se GLU tecnicamente fattibile; non cancellare GLU off solo perché on perde. Se fattibilità non risolta, registrare GLU incompleto/rinviato e discutere, senza bloccare indefinitamente il risultato U-Net.

GappyPOD: lambda scelti al 5% congelati nella curva 1/5/20%, nessun nuovo tuning iniziale. Verificare fit/statistiche/p/frame/valori fisici prima del riuso; se mancano oggetti salvati, segnalare e concordare ricostruzione a config fissata, senza rifit silenziosi. Dense e sparse non nella stessa classifica.

Repliche di maschera eval 0/1/2 distinte da seed training. Confermare U-Net vs GLU p_on non rende automaticamente robusti gli effetti p_on/off: per quello servono repliche delle relative coppie. Nessuna replica densa necessaria per avviare questo percorso.

### 6.1 Training e plateau

| Voce | Default operativo da implementare e congelare prima dei run |
|---|---|
| Loss | MSE T normalizzata uniforme sull'intero campo, stats fit-only |
| Optimizer | AdamW lr iniziale 1e-3, wd 1e-4; clipping norma 1 |
| Precisione/batch | fp32 senza AMP/TF32/compile; batch effettivo 8 |
| Warmup | 950 aggiornamenti optimizer; contare aggiornamenti, non microbatch |
| Scheduler | ReduceLROnPlateau su VAL 5%: min, factor 0.5, patience 5 eval, threshold rel 0.001, min_lr 1e-5 |
| VAL | Ogni 10 epoche e a fine tranche/arresto; tutte e tre le repliche fissate |
| Best | Primo minimo assoluto del punteggio VAL, separato dalla soglia early stop |
| Arresto | Non prima di 200 epoche; 15 eval senza miglioramento relativo 0.1%; reset pazienza al calo LR |
| Tranche iniziale | 1000 epoche (95000 update con batch effettivo 8); checkpoint e discussione se ancora in progresso |

La tranche 1000 **non è un tetto scientifico definitivo**: se esaurita mentre migliora, decidere continuazione esplicita dallo stato completo, senza azzerare LR/optimizer/pazienza e senza cambiare criterio di selezione. Nessun ciclo infinito automatico. Se raggiunge il plateau definito, fermare e analizzare; queste sono convenzioni operative, non prove matematiche di convergenza.

GLU conserva candidato lr=1e-3: smoke/overfit TRAIN verificano sicurezza, non fanno una gara di iperparametri. Se fallisce, diagnosi e proposta di singola modifica prima del freeze; non scegliere lr leggendo VAL. Config scientifiche effettive, differenze e tentativi falliti registrati.

Selezione S = media sulle repliche della media sulle condizioni della media frame di RMSE in K. Confrontare solo gruppi completi; mai accorpare impropriamente pixel, ruoli, checkpoint o repliche. Se stesso numero frame/condizione e stesse repliche, ordini di media equivalenti; conservarne comunque una definizione univoca.

TRAIN/seen diagnostici ogni 10 epoche al 5% replica 0; best/last finali: tre ruoli × tre coperture × tre repliche. La misura TRAIN usa maschere eval fisse, distinta dalla loss TRAIN con maschere ricampionate. Non selezionare su seen, 1/20%, snapshot o TEST. 1/20% sono fuori densità di training, checkpoint scelto solo al 5%.

### 6.2 Hardware, durata e concorrenza

**Nessun limite prefissato di ore per training o campagna.** Il precedente ceiling proposto 24 GPU-h e le riserve 2 h/run dei revisori non si applicano qui (decisione Enrico 18 novembre). Non troncare un run per rispettarli, non ridurre un budget scientifico per riempire uno slot.

Profilare tempi/epoca, valutazioni, checkpoint, memoria e latenza campo completo batch 1. Pubblicare stima e costo reale; il tempo U-Net denso non predice GLU. Se costo inatteso, comunicarlo e decidere come proseguire, senza inventare un tetto retroattivo.

Distinguere **walltime scheduler** da budget scientifico: rispettare MaxTime/QoS realmente disponibili, richiedere durata appropriata e salvare checkpoint completi a confine d'epoca prima del limite. Se serve dividere un run fra job, usare continuazione programmata e registrata dallo stesso stato, testata prima; un timeout non vale plateau né inferiorità scientifica. Nessun riavvio cieco dopo errori.

Enrico riferisce quattro slot normal utilizzabili in parallelo. Verificare uso reale, associazioni, GPU e code al momento; non sono quattro GPU fisiche garantite. Preferenza una GPU normale non preemptable per run, L40S se disponibile per confrontare costi. Non modificare job/env HONF, non calcolare sui login node. Risorse diverse ammesse solo registrandole e verificando coerenza numerica.

Dopo validazione comune e incarico di screening, fino a quattro run indipendenti possono essere concorrenti se gli slot effettivi lo consentono: directory/log/config/checkpoint/RNG separati, nessuna scrittura condivisa. Parallelismo non autorizza seed/modelli extra e non sostituisce gli STOP.

## 7. Valutazione, figure e controllo istantanee

Per frame/condizione/ruolo/copertura/replica: RMSE, MAE e NRMSE fit-only, bias firmato, |bias|, Centered-RMSE; regioni termiche esistenti (T<1800, T>=2100 e T>=2200 ove previste), coverage/conteggi, picchi e metriche già supportate. Regioni vuote: non definite con conteggio zero, mai errore 0 inventato. Identità RMSE²=bias²+C-RMSE² solo prima di mediare.

Figure: frame indice 1000 per ciascuna VAL, stessi pannelli e scale T per confronti; errore divergente comune e viste strette dichiarando saturazione; mappe errori medi, curve loss TRAIN e metriche eval, LR, best e stop, coperture e costi. Riportare tutte le condizioni e peggioramenti, non solo esempi favorevoli.

Best è il confronto primario selezionato. Last e seen servono a comprendere stabilità e meccanismo, non sono ulteriori condizioni obbligatorie per decretare un vincitore. Differenze di segno, condizioni discordi e variabilità vanno descritte; nessuna soglia fisica arbitraria in K, nessun “vince su ogni condizione oppure indistinguibile”.

### Snapshot sparse

Riutilizzare pairing U5 entro ruolo/condizione, senza punti fissi. Fissi: siti/maschera del destinatario, coordinate, p, target, copertura e replica. Cambiano insieme solo vi/vj, presi **dal donatore agli stessi siti del destinatario**; tutti gli altri siti restano nascosti.

La predizione swapped riguarda il donatore osservato col layout del destinatario, non coincide necessariamente con la predizione aligned del donatore già salvata. Salvare indici destinatario/donatore e observation hash del layout, checkpoint e variante, con test della trasformazione effettiva.

Sul best di ciascuno screening, 5%, replica 0, seen/VAL; stesso controllo per GappyPOD matched. Aggiungere riferimento full-field e regionale RMSE(T_donatore,T_destinatario), calcolato dai target solo nel percorso diagnostico. Non dividere automaticamente per valori piccoli/nulli; eventuali rapporti vanno definiti con casi limite.

Peggioramento sostiene specificità dell'istante, non causalità; assenza di peggioramento non prova che la velocità sia ignorata o che il modello produca soltanto una media. Leggere variabilità intrinseca, mappe, condizioni e pairing.

## 8. Fasi per Cursor: output, verifiche e STOP

Tutti i percorsi nuovi sotto sono **output futuri**, non implementazioni già presenti. Per ogni incarico leggere istruzioni, remoto e diario, preservare concorrenti, committare prima di eseguire, consegnare risultati e aggiornare stato. Uno STOP chiude l'incarico: il successivo viene delimitato con Enrico, non serve chiedere permesso per ogni operazione ordinaria già inclusa.

### S0 — Piano e decisioni (documentale)

**Output:** questa roadmap; riferimenti Month-2/README, registro e diario aggiornati.  
**Completamento:** sintesi approvata da Enrico e push, nessun codice o run. **STOP 0.**

### S1 — Osservazioni persistite e audit GLU

**Attività:** materializzare manifest/payload eval, annidamento e join storico; definire schema macchina `configs/experiments/month2_sparse_protocol.yaml`; audit release e forme/default in `docs/month2/04_glu_release_adaptation.md`. Confermare numeri/tie-break/condizionamento/coordinate; nessun training scientifico.  
**Verifiche:** 3309 hash 5% o spiegazione delle differenze; 1⊂5⊂20 per frame/replica; ruoli completi/no TEST; upstream invariato; contratto input/loss univoco.  
**Output:** generatori/test, config protocollo, manifest con hash persistenti, report audit. S1 e S2 possono far parte del primo incarico delimitato; niente GLU trainer upstream.

### S2 — U-Net sparse e percorso trainer

**Attività:** vista cache a sette canali, maschere per epoca e resume, evaluator con identità complete e unità; test §4 e piccolo overfit TRAIN. Riutilizzare moduli preservando il denso.  
**Output:** moduli sparse, `configs/models/unet_sparse.yaml`, config esperimento, CLI, test/report.  
**Verifiche:** isolamento fino al forward, osservazioni identiche ai baseline, apprendimento piccolo, resume e set_epoch effettivi. **STOP A: consegna S1–S2 prima del training scientifico.**

### S3 — GLU adattato e prove sintetiche/piccole

**Attività:** wrapper `models/glu_reconstructor.py` o equivalente, config `glu_minimal.yaml`, p additivo, distanze isotrope, nessun padding implicito; test §5 e piccolo overfit TRAIN.  
**Verifiche:** forme/gradienti, T assente dalla firma inferenziale, permutazioni/tie, chunked/full e gradienti se applicabile, parametri tutti presenti prima dell'optimizer, sensibilità scala documentata.  
**Output:** codice/config/test e audit aggiornato. **STOP B.**

### S4 — Baseline matched, profiling e freeze

**Attività:** riuso 5% e valutazioni GappyPOD 1/20% con lambda congelati; verifica medie consolidate. Profiling U-Net e GLU su risorse effettive, inclusi eval 1/5/20%; pesi diagnostici non riusati. Non usare prestazioni VAL del profiling per scegliere modello.  
**Verifiche:** memoria, loss/gradienti finiti, costi misurati, scheduler/checkpoint/resume, ambiente e hash; batch effettivo e query full-field. Se GLU bloccato, report e decisione esplicita sul proseguimento U-Net, nessun rinvio indefinito automatico.  
**Output:** configurazioni effettive congelate, stime senza tetti orari, comandi esatti e piano scheduler. **STOP C prima dei run scientifici.**

### S5 — Quattro screening seed 0

**Attività:** un run per U-Net on/off e GLU on/off da inizializzazione fresca; possibile concorrenza fino agli slot normal effettivi secondo incarico. Stesse regole congelate, nessun tuning intermedio guidato dall'altro run.  
**Output:** curve, best/last, per-frame/per-condizione, costi, manifest/checkpoint hash; stato reale e tentativi falliti. Se fine tranche in miglioramento: consegna e decisione continuazione, non fallimento né restart da zero.  
**Verifiche:** no leakage, campioni/layout, stato scheduler/stop, provenienza e replay best. **STOP D.**

### S6 — Confronto, coperture e snapshot

**Attività:** valutazione finale best/last completa §6; controlli §7, riferimento donatore; tabelle matched, mappe e limiti. Separare denso/oracle/swap dalla classifica deployable al 5%.  
**Output:** `docs/month2/05_sparse_reconstruction.md` e directory risultati dedicate, senza sovrascrivere storico.  
**Decisione:** quale contrasto confermare? Se differenze piccole o discordi, non proclamare vincitore a seed 0; se GLU costa troppo documentare il limite. **STOP E per discutere finalisti e repliche.**

### S7 — Repliche mirate e chiusura sparse

**Attività:** seed modello/shuffle 1/2 delle due configurazioni scelte, stesso contratto e train_mask_seed; nessuna replica di ogni iperparametro. Se occorre confermare anche altri contrasti, scelta esplicita aggiuntiva.  
**Output:** risultati individuali seed 0/1/2, dispersione training separata da repliche layout; report e stato 2.5–2.6 aggiornati, rimandi a 2.8 senza dichiararlo tutto completato.  
**Completamento:** ogni fase eseguita o rinvio motivato; dati/codice/config/output ricostruibili, confronto GLU presente oppure apertamente incompleto. **STOP F.** Assisted 2.7 e TEST restano incarichi separati.

## 9. Decisioni successive e limiti

- U-Net competitiva e meno costosa: mantenerla; non serve che GLU vinca.
- GLU migliore con evidenza ripetuta: investire nel suo adattamento documentato.
- Entrambi apprendono male anche piccolo TRAIN: diagnosi tecnica prima di aumentare complessità.
- Buon TRAIN e cattiva VAL: analizzare trasferimento, livello/struttura, condizioni e copertura; non impossibilità fisica.
- Difficoltà solo a densità mai viste: possibile singolo training multi-densità comune, nuova decisione/config, non scelta automatica.
- Nearest-observed con distanza sensore, embedding MLP GLU, comparatore lineare supervisionato: fuori dal minimo, al massimo un correttivo motivato alla volta.
- Nessun obbligo di nuove repliche dense; se si vuole sostenere necessità della non linearità o robustezza p densa, serviranno comparatori/repliche pertinenti.
- Ratifica supervisori distinta dalle decisioni operative Enrico. Non spacciare pubblicazione di questo piano per approvazione scientifica del laboratorio.

## 10. Tracciabilità e checklist

Per ogni run: SHA e hash sorgenti/config, working tree con tracked/untracked distinti, dati/stats/maschere e valori osservati, canali/unità/coordinate, seed separati, p/K/copertura/replica/layout, architettura/default reali, optimizer/scheduler/RNG/epoca/update, best e criterio significativo distinti, hash checkpoint, ambiente/GPU/QoS/job, queue/setup/compute e RAM/VRAM/latency. Payload pesanti fuori Git, persistenti con hash; comandi esatti, niente soli `...` nel report di consegna. Nessun overwrite di raw, storico, upstream o manifest passati.

Estensioni di tranche esplicite, checkpoint completi distinti da partial; no reset stato e no continuazione da epoca incompleta. Pubblicare tabelle/figure/script e risultati negativi. Prima di push rileggere remoto; fast-forward senza force; nessuna PR/merge implicita nell'incarico.

- [x] S0 — Piano documentale approvato operativamente da Enrico; pubblicazione, nessun esperimento.
- [x] S1 — Osservazioni persistite + join baseline storico 3309/3309 @5% (hex) + audit GLU documentale.
- [x] S2 — U-Net sparse/test/overfit/resume diagnostico. STOP A.
- [x] S3 — GLU adattato deterministico, test e piccolo overfit TRAIN. **STOP B.**
- [ ] S4 — Baseline/profiling/freeze. Profiling **COMPLETED**; freeze neurale fatto; GappyPOD 1/20% **aperto**. STOP C parziale.
- [ ] S5 — Screening seed 0 (**submitted**; non dichiarato completato). STOP D.
- [ ] S6 — Confronti/coperture/snapshot e scelta finalisti. STOP E.
- [ ] S7 — Repliche/chiusura con limiti o rinvii dichiarati. STOP F.

**Prossimo incarico previsto:** Enrico controlla esiti S5; eventuale resume/tranche; GappyPOD 1/20% separato.
