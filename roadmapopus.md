# ROADMAP OPUS — Più istantanee aiutano davvero la ricostruzione sparse?

**PROBLEMA CENTRALE: separare l'effetto della quantità di istantanee supervisionate dalla variabilità del training e dalla selezione del checkpoint.**

**DOMANDA: con le stesse poche velocità osservate, aumentare i frame TRAIN migliora la temperatura su istantanee mai viste e su condizioni operative escluse?**

**UTILITÀ PER LA TESI: decidere se raccogliere/sfruttare più esempi di combustione prima di cambiare architettura o sistema di misura.**

- Repository: `EnricoBontempi/official_thesis`; branch `mission/dense-unet`.
- Decisione: Enrico, 18 novembre 2026, America/New_York.
- Stato iniziale: **PIANIFICATA, nessun passo eseguito da questo documento**.
- Analisi di riferimento: [analisi tripla revisore s7.md](analisi%20tripla%20revisore%20s7.md), commit `6e2d176fa74d097aa8128c472103cf76c5a8a7a6`.
- Evidenze di partenza: STOP E `a6b2bad7445170e6d821a14de9f7af9a1905ab2c`; diario `925f556823214e37f3abf9d73b41d6e5ed815e58`.
- Piani collegati: [roadmapsparse.md](roadmapsparse.md), [roadmap_month2.md](roadmap_month2.md), [roadmapastra.md](roadmapastra.md).

## 1. Contesto e motivazione ingegneristica

La tesi studia la ricostruzione del campo istantaneo di temperatura nella combustione da osservazioni limitate. Il contesto applicativo riguarda accesso ottico/PIV, campionamento incompleto delle velocità e disponibilità di poche misure termiche nelle camere di combustione. Le misure sparse attuali sono campioni ideali di simulazione: non riproducono ancora integralmente rumore, geometria, risposta dinamica o intrusività della strumentazione reale.

La U-Net densa U0–U7 è conclusa nel perimetro dichiarato. Lo screening sparse S5 e la valutazione S6 sono consegnati: al 5% i neurali superano GappyPOD, ma non c'è un vincitore robusto fra U-Net e GLU né un effetto p dimostrato su più training. Tutte le reti sparse sono state addestrate con 38 frame per ciascuna delle 20 condizioni TRAIN, soltanto 760 istantanee.

Opus evidenzia un forte divario fra frame già usati nel fit e frame nuovi delle stesse condizioni. Per U-Net last riferisce circa 7–8 K su train_fit contro 56–62 K su seen. Questo motiva l'ipotesi di insufficiente varietà delle istantanee; non dimostra che aggiungere frame risolverà il problema. Frame ravvicinati sono correlati e potrebbero aggiungere poca informazione.

La campagna mantiene **U-Net sparse, due velocità, K=0, copertura TRAIN 5%**. Varia la quantità di esempi e replica inizializzazione/shuffle. Non è una gara di nuove architetture e non sostituisce la futura indagine di Neon sulla densità delle osservazioni.

## 2. Domande, misure ed esiti richiesti

| Domanda | Confronto | Misura che deve comparire nel report |
| --- | --- | --- |
| Più frame aiutano a ricostruire nuove istantanee? | 19/38/149 frame per condizione, seed 0, stesso p | RMSE seen e VAL, bias e centered; andamento rispetto agli aggiornamenti |
| Il ranking attuale dipende dal training? | 38 frame, seed 0 contro 1, on/off | Delta appaiati per condizione e dispersione descrittiva fra i due run |
| Quanto conta il minimo scelto su VAL? | Best, last e rianalisi R0 | Epoca/update selezionato, prestazione e sensibilità del ranking |
| p diventa più utile con più dati? | On/off in ciascun regime | Delta appaiato on−off a seed e frame fissati |
| Il miglioramento è utile per la combustione? | Stessi casi e soglie in tutti i bracci | Errori nelle regioni calde, area e IoU oltre alla RMSE globale |

La risposta finale deve distinguere nuove istantanee delle condizioni TRAIN (`seen_train_eval`) da trasferimento alle tre condizioni VAL. Un miglioramento su seen senza miglioramento su VAL sostiene una migliore generalizzazione temporale nel campione, non un trasferimento operativo risolto.

## 3. Perimetro, gerarchia e stato dei piani

1. Questa è una sotto-missione della roadmap sparse, prima di Astra e prima di scegliere la prosecuzione S7.
2. S4–S6 rimangono consegnati con i rilievi aperti delle revisioni; non vengono riscritti come esperimenti mai avvenuti.
3. S7 resta sospeso nella decisione scientifica; `roadmapsparse.md` e Mese 2 restano aperti.
4. La preparazione comune di O0 serve anche ad Astra: non duplicarla in un secondo framework.
5. Pubblicare questa roadmap autorizza la pianificazione. Cursor riceverà incarichi per singoli passi; nessuna esecuzione autonoma dell'intera campagna o submission implicita.

Esclusi: GLU training, multidensità, T in ingresso, sweep di iperparametri, nuovi split di condizioni, TEST, nuovi fit POD/prior e modifiche HONF/ModularDT o upstream `external/`. Le idee Opus su ridge supervisionata e CV per condizione restano documentate nell'analisi tripla per una decisione successiva.

## 4. Contratto scientifico da congelare in O3

### 4.1 Dati e confini

| Pool | Indici per condizione | Condizioni | Frame complessivi | Funzione |
| --- | --- | ---: | ---: | --- |
| B19 | 0…1440, passo 80 | 20 TRAIN | 380 | Riduzione dati |
| REF/A38 | 0…1480, passo 40 | 20 TRAIN | 760 | Riferimento S5 e replica |
| C149 | 0…1480, passo 10 | 20 TRAIN | 2980 | Aumento dati |
| seen_train_eval | 1600…2000, passo 40 | Stesse 20 TRAIN | 220 | Nuove istantanee, diagnostico |
| val_eval | 0…2000, passo 50 | Tre VAL storiche | 123 | Selezione e confronto storico |

B19 ⊂ A38 ⊂ C149, verificato sugli identificativi reali. Nessun frame della finestra seen entra in TRAIN. Nessun TEST caricato. Prima di usare C149 verificare che ogni indice corrisponda al frame previsto, incluse anomalie note di `40NH3_1.2`; niente deduplicazione o rimozione silenziosa.

Normalizzatore: mantenere quello storico fit-only sui 760 frame per isolare l'effetto del pool di ottimizzazione. Per B19 questo usa statistiche dei 760, inclusi frame non ottimizzati da B19: dichiararlo come **esperimento sul numero di esempi supervisionati a preprocessing congelato**, non come pipeline integralmente fittata su soli 380 frame. Nessun uso di seen/VAL per statistiche. Non introdurre un secondo normalizzatore senza un contrasto separato.

Cache C149 separata e versionata. Confrontare i 760 frame condivisi con la cache storica e controllare canali, unità, orientamento, coordinate e denormalizzazione sui frame nuovi. Accesso raw/pool è un prerequisito da verificare, non una disponibilità presunta. Se mancano i raw, A/B possono procedere su incarico, ma la domanda sull'aumento dei dati resta aperta.

### 4.2 Unità di ottimizzazione e campionamento

Un'**epoca scientifica** contiene sempre 760 presentazioni, batch 8, **95 optimizer update**, anche per B19 e C149. Non significa un passaggio completo nel pool. La tranche iniziale di 1000 epoche equivale a 95 000 update e 760 000 presentazioni, salvo early stop.

Usare uno stream deterministico di permutazioni del pool senza rimpiazzo: consumare 760 elementi per epoca scientifica e conservare gli elementi non consumati per quella successiva; a esaurimento generare la permutazione seguente. Per B19 si attraversano due permutazioni per epoca; per C149 una permutazione attraversa più epoche. È consentita una ripetizione attraverso il confine fra due permutazioni, non un riequilibrio in base a T o loss. Persistire posizione, generatore e contatore di ciclo. Stampare conteggi per condizione e copertura cumulativa del pool: l'equilibrio è garantito su passaggi completi, non su ciascuna epoca C149.

Per il pool 38 verificare che lo shuffle resti equivalente al percorso S5: una permutazione nuova di 760 a ogni epoca. Usare un generatore del sampler separato da inizializzazione e maschere. Se il runner storico consuma gli RNG in modo incompatibile, dichiarare il cambiamento e attivare il controllo ponte O2: non attribuire ai dati una differenza introdotta dal runner.

La maschera TRAIN dipende da identità frame, epoca scientifica e seed, come S5. Le due occorrenze B19 dello stesso frame nella stessa epoca condividono la maschera: documentare che più presentazioni non creano una nuova istantanea o un nuovo layout. Nessun cambio implicito verso seed per occorrenza. Il confronto misura l'utilità dei pool con questa ricetta, a budget di aggiornamenti controllato; non stima un effetto puro della sola indipendenza statistica dei frame.

### 4.3 Modello, training, selezione

| Voce | Scelta |
| --- | --- |
| Modello | SparseUNet b=32, 7 canali storici; nuova inizializzazione in ciascun run |
| p_on/off | NH3/phi disponibili oppure due canali zero; tutto il resto uguale |
| Informazione | Single-frame, vi/vj al 5%, coordinate; K=0; nessun T nel forward |
| Precisione | fp32, AMP/TF32/compile disabilitati esplicitamente e registrati |
| Loss/optimizer | MSE T normalizzata full-field; AdamW lr 1e-3, wd 1e-4, clipping norma 1 |
| Batch | Effettivo 8; nessuna modifica di batch implicita |
| Warmup | 950 optimizer update |
| Scheduler | VAL 5% tre repliche; ReduceLROnPlateau min, factor .5, patience 5 eval, threshold rel .001, min_lr 1e-5 |
| Eval | Ogni 10 epoche scientifiche, cioè 950 update, e all'arresto |
| Best primario | Primo minimo assoluto VAL mean-condition RMSE K, media delle tre maschere |
| Early stop | Minimo 200 epoche scientifiche; 15 eval senza miglioramento relativo .001; reset pazienza al calo LR |
| Orizzonte iniziale | 1000 epoche scientifiche; estensione solo esplicita con resume completo |
| Seed | modello/shuffle 0 o 1 come matrice; train_mask_seed 20261118 invariato; eval_mask_seed 20261116 |

Si conserva la selezione VAL di S5 per un confronto interpretabile con lo storico. R0 non la sostituisce retroattivamente. VAL influenza checkpoint, scheduler e arresto: i risultati sono evidenze di sviluppo, non una prova su holdout indipendente.

Valutazione TRAIN comune: usare i **760 frame storici** e chiamarla `train_reference760`, anche per B19/C149. Per B19 distinguere i 380 ottimizzati dagli altri 380; per C149 i 760 sono un sottoinsieme del pool. Non confrontare come se fossero la stessa metrica medie calcolate su 380, 760 e 2980 frame. L'eventuale `own_pool_train` va pubblicata separatamente.

Maschere dei frame storici: payload persistiti condivisi, nessun fallback. Nuovi frame C149 richiedono maschere TRAIN deterministiche; eventuali eval own_pool richiedono un manifest separato, senza alterare quello storico.

## 5. Matrice di training

| ID | Pool | p | Model/shuffle seed | Nuovo training |
| --- | --- | --- | ---: | --- |
| REF38-on/off | 38 | on/off | 0 | No: S5 originale, salvo ponte motivato |
| O-A38-on-s1 | 38 | on | 1 | Sì |
| O-A38-off-s1 | 38 | off | 1 | Sì |
| O-B19-on-s0 | 19 | on | 0 | Sì |
| O-B19-off-s0 | 19 | off | 0 | Sì |
| O-C149-on-s0 | 149 | on | 0 | Sì |
| O-C149-off-s0 | 149 | off | 0 | Sì |

**Matrice principale: sei nuovi training**, secondo la proposta Opus; non viene ridotta al solo p_on. Nella stessa famiglia e seed confrontare gli hash dell'inizializzazione prima dell'apprendimento. Il seed TRAIN-mask resta fisso: il contrasto seed 0/1 misura inizializzazione/shuffle condizionati a quel processo di mascheramento, non tutta la variabilità possibile.

La matrice è incompleta rispetto a una replica di ogni regime: B19 e C149 hanno inizialmente un seed. Non promettere significatività dall'unica coppia 38. Se l'effetto dati è ambiguo rispetto alla variabilità osservata, proporre come passo O6 separato la replica seed 1 di B19 e/o C149 on/off. Non avviarla automaticamente.

## 6. Passi operativi e STOP

### O0 — Preparazione comune e verifica dei rilievi

- [ ] Leggere diario, analisi tripla, roadmap mese 2/sparse, report 03/04/05 e istruzioni locali; verificare branch/remoto e preservare modifiche concorrenti.
- [ ] Verificare sintassi e schema di `month2_sparse_protocol.yaml`, incluso S6; correggere solo difetti riprodotti, senza cambiare di nascosto protocollo.
- [ ] Rendere obbligatori payload/hash realmente consumati; niente rigenerazione silenziosa dei layout mancanti. Gate Gappy deve rifiutare riferimenti vuoti e richiedere le sei chiavi previste.
- [ ] Separare SHA training, script evaluator, patch e checkpoint/CSV riusati. Se il sorgente storico non è ricostruibile, dichiarare il limite anziché inventarne lo SHA.
- [ ] Uniformare precisione esplicita; riconciliare ultima eval fra checkpoint, CSV e JSON; correggere il salvataggio futuro senza alterare lo storico.
- [ ] Controllare figure loss/colonne, regioni e casi VAL. Aggiornare i riepiloghi di stato obsoleti in `roadmapsparse.md`/report con link alle sotto-missioni, preservando decisioni e risultati storici.

**Consegna:** ledger sintetico verificato/corretto/aperto, test pertinenti, SHA. **STOP O0**, niente training. Queste verifiche valgono anche per Astra.

### O1 — R0: rianalisi della selezione sui quattro S5

- [ ] Recuperare una volta le metriche per epoca di tutti i quattro S5; usare history completa nel checkpoint e CSV quando il JSON omette la finale.
- [ ] Calcolare: selezione minima seen, leave-one-VAL-condition-out (selezione sulle altre due, misura sulla terza), media mobile causale delle ultime cinque eval e scelta del primo minimo della curva smussata. Prime quattro eval escluse da quest'ultima regola.
- [ ] Conservare best VAL e last originali; riportare per ogni regola epoca/update, score usato e VAL per condizione disponibile a quell'epoca.
- [ ] Nessuna ricostruzione fittizia di pesi mancanti: le tabelle per epoca possono supportare la rianalisi RMSE senza un checkpoint intermedio. Mappe/inferenza di un checkpoint non salvato restano non disponibili.
- [ ] Dichiarare la contaminazione residua da scheduler/early stop VAL e l'uso esplorativo di seen per questa selezione. Nessuna di queste varianti diventa un risultato TEST o una CV indipendente.

**Consegna:** `results/month2/opus_data_study/r0_selection/`, tabella e lettura sintetica. **STOP O1**. Nessun training e nessuna scelta nascosta della regola più favorevole.

### O2 — Dataset, sampler e compatibilità del runner

- [ ] Manifest dei tre pool con conteggi/nesting/disgiunzione; verificare disponibilità raw e cache C149 senza download massivi impliciti.
- [ ] Costruire C149 in streaming/mmap/chunk su compute node, conservando il preprocessing storico. Verificare payload condivisi e campioni nuovi; niente load pesante su WSL o login.
- [ ] Implementare unità 95 update riusando il trainer scientifico; testare conteggi, stream continuo, maschere/epoca e assenza di leakage.
- [ ] Test continuo contro resume: sampler anche a ciclo non allineato, RNG, scheduler, contatori, history, update e best/last; rifiutare partial e resume incompatibili.
- [ ] Controllo ponte 38 seed 0: confrontare ordine/mask/init/LR con S5 su percorso ridotto. Replay del best storico @5% con tolleranza macro RMSE **0.05 K** dichiarata prima.
- [ ] Se emerge una differenza scientifica del runner non eliminabile, documentarla. La matrice dovrà aggiungere due riferimenti 38 seed 0 nel nuovo runner, con incarico esplicito prima dei confronti; niente attribuzione indebita ai dati.

**Consegna:** manifest, controlli memoria, test e valutazione della confrontabilità. **STOP O2**. Se C149 non disponibile, presentare il blocco e i passi eseguibili separatamente.

### O3 — Freeze e preparazione dei sei run

- [ ] Config separate con pool, sampler, selezione, precisione, seed e output realmente consumati; mantenere autorizzazioni nuove disabilitate fino all'incarico di lancio.
- [ ] Salvare anche `best_seen_diagnostic` in futuro, senza cambiare scheduler/arresto/primario, così la sensibilità seen sarà valutabile senza riselezione irreversibile.
- [ ] Definire set di curve ed eval finali: best e last su reference760/seen/VAL, 1/5/20%, tre maschere; own_pool solo separato. Preferire completamento finale offline per evitare costi inutili nel loop.
- [ ] Verificare percorsi HOME stabili e dipendenze; nessun ritorno implicito al pool non accessibile. Hash sorgenti/config/dati prima dell'esecuzione.
- [ ] Lanciatore robusto alla copia Slurm in spool: niente include relativo a `$0`; preflight leggero e nuove directory non sovrascrivibili.
- [ ] CPU `mit_normal` + account/QoS `mit_amf_advanced_cpu`; GPU `mit_normal_gpu` + account/QoS `mit_amf_advanced_gpu`, verificati al lancio. Preferire L40S normal; nessun fallback general/preemptable.

**Consegna:** freeze leggibile, sei config e piano onde. **STOP O3**, ancora nessuna submission.

### O4 — Submission per onde, su incarico

Prima onda: A38 on/off seed 1 e B19 on/off seed 0 (quattro GPU normal se disponibili). Seconda: C149 on/off seed 0 quando i dati sono verificati e gli slot disponibili. Nessuna dipendenza automatica o attesa attiva per riempire gli slot: Enrico riporta lo stato e autorizza il passo successivo.

- [ ] Committare/pushare codice prima dei run; registrare SHA reale, config/hash e ID.
- [ ] Nuova inizializzazione, nessun warm start da S5/profiling o fra pool.
- [ ] Checkpoint completi a confine d'epoca, salvataggio atomico; partial separato.
- [ ] Nessun tetto scientifico arbitrario in ore. Walltime Slurm e margine di checkpoint sono vincoli operativi; fine walltime non equivale a plateau. Estensione/resume solo da stato completo su incarico.
- [ ] Dopo submission consegnare ID, risorse, log e comandi manuali, quindi **STOP senza polling, attese o rilanci automatici**. HONF intatto.

### O5 — Recupero, eval finale e confronto

Solo dopo che Enrico segnala la fine: leggere stato e output una volta, verificare arresto, NaN/OOM, epoca/update e checkpoint. Eventuali riparazioni precedono il confronto scientifico.

- [ ] Eval finale nei setting congelati, manifest completo e nessun riuso basato sulla sola esistenza dei CSV.
- [ ] Tabelle separate per best, last e best_seen diagnostico; mostrare tutte le condizioni, seed e repliche, non soltanto macro favorevoli.
- [ ] Curve rispetto a update, presentazioni e passaggi equivalenti del pool. Riportare numero di frame distinti effettivamente visitati.
- [ ] Confronti 19/38/149 allo stesso p/seed e confronto seed 0/1 a 38. A parità di step usare curve condivise fino al minimo orizzonte completato; stop diversi sono anche effetti della ricetta. Un singolo checkpoint comune non salvato non può essere inventato.
- [ ] RMSE, MAE, bias firmato/assoluto, centered, metriche regionali con definizioni storiche, area/IoU calda, tempi e memoria. Pannelli sulle tre VAL, scale comuni e errore firmato.
- [ ] Coperture 1/20% etichettate come generalizzazione dalla densità TRAIN 5%, non curva ottimale numero sensori.

**Consegna:** report `docs/month2/06_opus_data_and_training_variability.md`, `results/month2/opus_data_study/`, ledger e diario aggiornati. **STOP O5**, nessuna scelta S7 automatica.

### O6 — Decisione e passaggio ad Astra

| Esito | Conclusione consentita | Azione da discutere |
| --- | --- | --- |
| Più frame migliorano seen e VAL in entrambi i p | I frame aggiunti sono utili con questa ricetta | Regime ricco candidato; conservarne i checkpoint per estensione Astra |
| Migliora solo seen | Generalizzazione alle nuove istantanee migliora, trasferimento fra condizioni ancora limitato | Astra/Neon e analisi per condizione restano centrali |
| Effetto dati piccolo o non monotono | Questa aggiunta e questa ottimizzazione non risolvono il problema | Valutare correlazione, esposizione e poi osservazioni/architettura |
| Differenze simili a quelle seed 0/1 | Evidenza insufficiente per attribuire piccoli gap al pool | Proporre replica mirata dei pool ambigui, senza lancio automatico |
| C149 non disponibile | La metà crescente della curva non è stata misurata | Registrare limite e decidere recupero dati; A/B non sostituiscono C |

Niente test di significatività costruito dai pixel o dalle tre maschere. Due seed sono un metro descrittivo del rumore, non una distribuzione stimata robustamente. Nessuna soglia automatica «due volte il gap seed».

## 7. Chiusura verificabile

- [ ] Sei training principali consegnati, oppure righe esplicitamente bloccate con motivazione: in tal caso campagna parziale, non domanda risolta.
- [ ] Dati, durata, selection e runner confrontabili o differenze chiaramente dichiarate.
- [ ] Risposta esplicita alle cinque domande della sezione 2, con limiti della correlazione e dei seed.
- [ ] Opzioni di proseguimento documentate senza scegliere un vincitore universale.
- [ ] Passaggio a [roadmapastra.md](roadmapastra.md), mantenendo S5 come riferimento per la diagnosi originale e identificando separatamente eventuali checkpoint Opus da aggiungere.
- [ ] Diario/report/stato dei piani aggiornati; sparse e Mese 2 restano aperti finché i rispettivi passi non sono svolti.

La chiusura di Opus risponde a una decisione sul regime di dati; non certifica PIV reale, successo assisted, necessità della non linearità o generalizzazione fuori dal dominio studiato.
