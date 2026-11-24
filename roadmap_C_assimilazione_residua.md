> **Nota del 24 novembre 2026.** **SOSPESA — NON ESEGUIRE.** La priorità corrente è la definizione del problema applicativo e lo studio di robustezza a osservazioni non ideali descritto in [roadmaptests.md](roadmaptests.md). Eventuale riattivazione soltanto dopo riesame e decisione esplicita di Enrico.
>
> Il piano storico sotto questa nota è conservato integralmente. Secondo quanto riferito da Enrico, non è stato eseguito.

# Roadmap C — Assimilazione spaziale dei residui di temperatura

**Studio preparatorio successivo al Mese 2 · 22 novembre 2026 · America/New_York**  
**Responsabile scientifico:** Enrico Bontempi  
**Esecutore previsto:** Sonnet  
**Coordinamento:** nuova chat amministratrice delle attività A/B/C.

## Stato e mandato

- [x] Piano C preparato e pubblicato su richiesta di Enrico.
- [ ] C0 — Contratto e riferimenti congelati.
- [ ] C1 — Baseline RBF, modulo residuo, test e smoke.
- [ ] C2 — Tre correttori sul riferimento storico, sviluppo.
- [ ] C3 — Valutazione e decisione sul trasferimento ai fold B.
- [ ] C4 — Eventuale conferma fold-specifica, sei correttori.
- [ ] C5 — Consegna e sintesi.

**Questa è una roadmap preparatoria, non la roadmap complessiva del Mese 3 e non una submission.** La nuova chat assegnerà esplicitamente implementazione ed esecuzione, precisando se autorizza soltanto C0/C1 oppure anche i tre training C2. C4 è un'estensione condizionata e richiede una decisione distinta; non parte da sola.

Mese 2 resta concluso nel perimetro sperimentale e il report è stato inviato ai supervisori. C non riapre il Mese 2 e non richiede nuovi dati o termocoppie reali. Il confronto Polimi e il riscontro MIT possono ancora cambiare la direzione applicativa.

## 1. Domanda scientifica

**Fissata una buona ricostruzione dalla velocità, un piccolo modulo che vede gli errori misurati nei siti T può correggere anche la struttura, oltre al livello, meglio di un offset mediano e di un interpolatore spaziale semplice?**

Per ogni istantanea:
1. la U-Net free congelata produce T0(x);
2. nei soli K siti osservati si calcola r_k = T_obs(x_k) − T0(x_k);
3. un correttore trasforma residui, posizioni e contesto ammesso in una mappa ΔT(x);
4. la predizione finale è T0(x) + ΔT(x).

Con K=0 si restituisce **esattamente T0**, senza passare attraverso il modulo. Il modello non deve degradare il backbone quando non arrivano misure.

Il risultato desiderato non è una curva RMSE più bella sulle VAL note: è capire quando le misure contengono un segnale spaziale utilizzabile, quando basta correggere il livello e dove il metodo fallisce. Un risultato negativo chiude la prova della ricetta; non dimostra impossibilità fisica o assenza d'informazione nelle T.

## 2. Motivazione e fonti congelate

Il Mese 2 ha mostrato:
- U-Net free competitiva nella struttura al 5% di velocità;
- mediana efficace sul livello di alcune condizioni VAL ma non sulla centered-RMSE;
- guadagno piccolo nella maggior parte delle ricette end-to-end con T in input;
- beneficio misurabile nel GLU p_off assisted, senza superare complessivamente U-Net free + mediana;
- fallimenti di correzione globale con una singola ancora sfavorevole;
- differenze già presenti a K=0 fra training free e assisted, che confondono il confronto diretto.

C elimina quest'ultimo problema: ogni confronto usa **lo stesso backbone, lo stesso checkpoint e la stessa predizione base**.

| Fonte | Riferimento |
|---|---|
| Contratto dei fold e valutazione comune | [roadmap_B_generalizzazione.md](roadmap_B_generalizzazione.md), prima pubblicazione 81f59438204cd5b0927ef4b3988b9a32026627f5 |
| Sintesi delle tre revisioni | [reviewmese2.md](reviewmese2.md), origine 9a6277f271652a2b6416cad14a6609ddb2c1a021 |
| S7 free finale | 2879c1b5270be30fb10375f0b300a50cd37d5413 |
| Sweep media/mediana | be18ad007e66fbf2565ad3d5486c6513fc5c3e4c |
| Assisted v2 e pacchetto congiunto | d20232ef04873e94708995d119bedc2f6eca2f50; results/month2/joint_step23/package/INDEX.md |
| Manifest geometrico T | configs/observations/month2_temperature_anchors_nested_v1.json |
| Domanda generale della tesi | docs/proposal.md |

Questi riferimenti sono fonti, non una richiesta di rifare le missioni precedenti. Non modificare codice upstream GLU, checkpoint free, tabelle storiche o report consegnato.

Il protocollo YAML storico delle T non è parsabile: conservarlo e creare un contratto C valido distinto; riusare il manifest JSON dei siti, non la sintassi errata del protocollo.

## 3. Disegno e dipendenze: sviluppo prima, trasferimento separato

### 3.1 Blocco iniziale C2: tre training

| Elemento | Specifica |
|---|---|
| Backbone | U-Net masked b32 free S7 p_on, checkpoint best |
| Backbone seed | 0, 1, 2, tutti e tre; nessuna scelta del seed migliore |
| Correttore | Una sola architettura descritta al §6, seed rispettivamente 0, 1, 2 |
| Pool | C149 storico: 20 condizioni, 2980 frame, normalizzatore storico del backbone |
| Osservazioni | vi/vj 2C al 5%; T ideali con K variabile |
| Nuovi training | **3 correttori**, non tre nuove U-Net |
| Selezione | VAL123 originale, criterio unico ai cinque K non nulli |
| Ruoli finali | reference760, seen220, VAL123 |

p_on significa che phi e percentuale NH3 sono input noti del backbone. Il correttore non riceve parametri operativi aggiuntivi assenti dal backbone. Nessun p_off, GLU o training per ogni K nel blocco minimo.

La dispersione dei tre sistemi comprende variazione del backbone e del correttore; non isola la sola inizializzazione del correttore. Si accoppiano base e trattamento entro ciascun sistema, senza prodotto cartesiano 3×3.

### 3.2 Cosa può procedere mentre gira B

C0/C1 e C2 sul protocollo storico possono essere preparati ed eseguiti in parallelo a B, nei limiti assegnati. Sono sviluppo sulle VAL già note, non nuova conferma indipendente.

Il contratto B0 è necessario per progettare correttamente il trasferimento; non occorre attendere i risultati outer di B per scrivere il codice C. La specifica C e le eventuali regole di attivazione dei fold vanno congelate prima di consultarne i risultati outer, se si vuole limitare ulteriore adattamento a quei risultati.

### 3.3 Conferma C4, solo dopo decisione

Se C mostra una ragione per proseguire, la chat amministratrice può assegnare:
- due fold B-lean/B-rich × tre backbone seed = **6 ulteriori correttori**;
- ciascuno appoggiato al best U-Net del proprio fold, già prodotto da B;
- fit del correttore e calibrazione RBF sulle sole 19 condizioni ammesse;
- normalizzatore e maschere specifici di quel fold;
- outer escluso anche dalla selezione del correttore.

**Massimo descritto dalla roadmap: 3 correttori iniziali + 6 condizionati; zero nuovi backbone in C.** C4 non è implicito nell'autorizzazione di C2. Se mancano i checkpoint B, si può attendere la dipendenza senza duplicarne i training.

Usare una U-Net S7 originale per valutare come held-out una condizione B sarebbe contaminato: l'ha già vista in training. Il relativo risultato può essere soltanto storico/diagnostico, non il confronto C4.

## 4. Dati, siti e domini

### Blocco storico

Stesse condizioni, frame e normalizzatore del riferimento S7:
- fit C149: 2980 frame 0..1480 passo 10;
- normalizzazione e fit reference: 760 frame storici 0..1480 passo 40;
- seen220: 1600..2000 passo 40;
- VAL123: 0..2000 passo 50 su tre condizioni;
- nessun TEST.

### Fold

Consumare senza reinterpretazioni il contratto pubblicato da Hydrogen:
- fit 2831 frame di 19 condizioni; norm/train_reference 722;
- seen_retained209 e selection_val123;
- outer_fold41, consultato soltanto dopo freeze;
- nessun fit o calibratore basato sulla condizione esclusa.

### Osservazioni

Velocità TRAIN 5%, stesse maschere del backbone. Usare backbone.eval(), parametri senza gradiente e comportamento numerico identico alla valutazione del riferimento.

K eval = **0, 1, 4, 9, 16, 32**; tre layout T storici annidati, ciascuno accoppiato alla replica di velocità. Manifest atteso:
a9c997c96cb44489dd1fc0d4484304b44d8beddbf00b73c25baf4e1922a19e54.

Siti TRAIN nuovi, casuali, deterministici da seed/epoca/frame, indipendenti dai tre layout eval. Riutilizzare il generatore esistente con namespace C distinto e pubblicato. Nessuna scelta usando target, gradienti, errori VAL o convenienza dei risultati.

Domini:
- common: griglia meno tutti i 32 siti del layout, **16352 pixel**, uguale per ogni K;
- full: tutti i pixel;
- off_anchor(K): esclusi solo i K siti;
- anchors(K): soli siti osservati, indefinito a K=0.

**Primario cross-K: common.** Riportare anche full, off_anchor e anchors. Per stimare l'effetto delle misure agli stessi siti, calcolare esplicitamente l'errore del backbone K=0 nei siti S_K, anche se K=0 non li osserva: è una metrica diagnostica etichettata, non un input.

## 5. Baseline spaziale unica: RBF regolarizzata sui residui

Prima del modulo appreso implementare un interpolatore gaussiano, **senza training neurale**:
- coordinate nella stessa unit square per tutte le configurazioni;
- k(x,a) = exp(−||x−a||²/(2 ell²));
- ΔT_RBF(x) = k(x,S) [k(S,S) + alpha I]^{-1} r;
- K=0 restituisce esattamente T0;
- risoluzione numericamente stabile via solve/Cholesky, non inversa esplicita.

Usare i residui grezzi: non sommare automaticamente la mediana. Così a K=1 la baseline può dare una correzione localizzata, distinguibile da uno shift globale. L'assenza di un termine costante è una scelta della baseline, da dichiarare quando un bias globale non viene recuperato.

**Calibrazione limitata e solo TRAIN:** ell ∈ {0.05, 0.15, 0.30}, alpha ∈ {0.001, 0.1}, kernel con diagonale unitaria. Sei combinazioni, non una ricerca aperta. Per ogni backbone scegliere una coppia sulla media uniforme della RMSE common sui cinque K>0, usando i frame train_reference ammessi e tre layout di calibrazione nuovi, deterministici e separati da quelli eval. Congelare siti di calibrazione, ordine dei candidati e tie-break primo minimo prima di eseguire. Non usare VAL/seen/outer per selezionare ell o alpha.

Il backbone ha già allenato quei frame: la calibrazione vede residui in-sample. Dichiararlo e misurare lo scarto fra distribuzioni dei residui TRAIN/VAL/seen; non descriverla come una validazione indipendente.

La calibrazione non allena un modello a vedere target fuori dai suoi dati ammessi: i residui completi dei soli frame TRAIN sono leciti per calibrare il metodo. In inferenza RBF vede soltanto r nei siti osservati.

Salvare risultati di tutti i sei candidati di calibrazione e quello scelto. Non cambiare griglia o aggiungere un secondo metodo se i risultati sono sfavorevoli. Baseline POD/GappyPOD storiche restano contesto, non ulteriori fit richiesti da C.

La valutazione RBF su VAL con manifest fisso è sviluppo; quella outer segue le stesse protezioni dei correttori. Non anticipare le metriche outer “perché è soltanto CPU”.

## 6. Un solo modulo residuo leggero

### Specifica da implementare e congelare in C0

Per evitare una nuova ricerca architetturale, il default è un **correttore a singola cross-attention**, un head, dimensione latente 32, senza rete che ricostruisca da sola un campo indipendente dalle misure.

Feature ammesse:
- query per pixel: coordinate, T0 normalizzata e sue due differenze spaziali calcolate con convenzione geometrica dichiarata;
- token per ancora: coordinate, T0 nel sito e residuo r_k normalizzato con la scala T del backbone;
- embedding query/key con MLP a due strati, larghezza 32, attivazione SiLU;
- valori della somma di attenzione: i residui scalari r_k stessi;
- score: prodotto query/key scalato più un termine di distanza geometrica con ell=0.15 fisso;
- gate per pixel g = tanh(MLP a due strati di larghezza 32), che riceve le feature query e il contesto di attenzione; ultimo strato inizializzato a zero;
- correzione: ΔT(x) = g(x) × somma_k attention(x,k) r_k, poi ritorno in kelvin.

Il contesto per il gate è la combinazione pesata degli embedding delle ancore. Nessun indice d'ordine delle ancore o embedding del nome della condizione; masking corretto e normalizzazione softmax sui soli token validi.

Con K=0 bypass esatto. Con tutti i residui nulli, ΔT=0 esattamente: il modulo non deve migliorare il campo grazie a un ramo che ignora le osservazioni. All'inizializzazione la correzione è zero per qualunque K; il gate può imparare ad attivarla. Questo non impone consistenza esatta ai siti osservati.

La forma scelta limita l'ampiezza della correzione al massimo residuo osservato e può essere troppo restrittiva. È parte dell'ipotesi della ricetta, non una proprietà fisica. Se emerge un problema numerico/di espressività già su casi sintetici, segnalarlo prima dei run scientifici e proporre una sola revisione del contratto; niente varianti parallele o tuning occulto.

Non leggere T vera fuori dalle ancore per costruire query, chiavi, gradienti o gating. Le differenze spaziali provengono da T0, non dal target. Non imporre media nulla alla correzione: serve correggere sia livello sia struttura, che saranno misurati separatamente.

### Loss e K TRAIN

- K uniforme su **{1,4,9,16,32}** per optimizer batch, deterministico dal numero di update; stesso K su microbatch.
- K=0 non richiede training: il bypass rende gradiente e correzione nulli. Si valuta comunque a ogni checkpoint.
- MSE normalizzata sui pixel **fuori dalle K ancore**, media per frame/batch. Target pieno lecito soltanto nella loss e nelle metriche.
- Nessuna loss regionale, penalità alle ancore o termine extra: appartengono ad altre ablation.
- Nessun oversampling dei casi con residuo elevato.

Questa ricetta differisce dal precedente assisted full-field/K incluso zero: dichiararlo. Il confronto principale è però fra correttori sulla stessa base, non una pretesa ablation isolata del vecchio assisted.

### Ottimizzazione, stop e checkpoint

Un unico default: AdamW lr 1e-3, weight decay 1e-4, clip 1, batch effettivo 8, 760 presentazioni/95 update per epoca scientifica, warmup 950 update, fp32 senza AMP/TF32/compile. Backbone sempre congelato; verificare anche buffer e modalità eval.

Eval ogni 10 epoche. Un solo criterio per scheduler, best e plateau: **macro RMSE common media uniforme dei cinque K>0** su selection_val e tre repliche. I valori K=0 sono riportati, ma non diluiscono il criterio relativo dello scheduler.

ReduceLROnPlateau come B (fattore .5, patience 5 eval, rel .001, minimo 1e-5); early-stop minimo 200 epoche, patience 15 eval, rel .001 e reset alla riduzione LR. Tranche iniziale 1000; eventuale estensione unica fino a 1200 solo se inclusa nel mandato e con stato invariato, come B.

Salvare e valutare **anche lo stato iniziale a correzione zero**; includerlo nella scelta del primo minimo assoluto. Se nessuna epoca lo migliora, il best resta il bypass: è un risultato negativo valido. Last rimane distinto. Nessun checkpoint per K o best_outer.

Il budget uguale in update è una convenzione, non costo uguale: i forward del backbone e dell'attenzione vanno misurati. Non promettere tempi dalle vecchie ricette.

## 7. Previsioni del backbone e rischio dei residui in-sample

Il backbone resta immutabile, ma la sua predizione dipende dalla maschera di velocità. Un residuo memorizzato per una maschera non può essere usato per un'altra.

Due percorsi:
1. default: forward congelato a ogni batch con la maschera TRAIN prevista, senza gradienti sul backbone;
2. cache soltanto se include esattamente checkpoint, normalizzatore, frame, maschera/epoca, precisione e contesto necessari e ne viene verificata la parità.

Non fissare poche maschere TRAIN solo per velocizzare il correttore. Riutilizzare il forward entro la stessa istantanea per K e metodi; valutare a chunk senza materializzare tutte le mappe di una campagna in GPU.

I residui su frame ottimizzati dal backbone possono avere ampiezza e struttura diverse dai residui di nuove condizioni. Non nascondere questo scarto con riscalamenti stimati su VAL. Non fare automaticamente cross-fitting con decine di backbone: richiederebbe una nuova decisione se necessario. In C4 la condizione outer è esclusa da entrambi gli stadi, ma le condizioni di fit del correttore restano in-sample rispetto al proprio backbone.

## 8. Verifiche prima dei training scientifici

### Contratto e isolamento
- YAML/JSON parsabili, count e membership reali, hash degli input consumati.
- Mutare T fuori dai siti osservati non cambia il forward.
- Mutare velocità nascoste non cambia il forward.
- Permutare ancore insieme a coordinate/residui non cambia la predizione entro la tolleranza numerica pubblicata.
- K=0 bit-identico al backbone; tutti i residui zero restituiscono backbone.
- Backbone pesi/buffer e hash invariati dopo aggiornamenti del correttore.
- Minimo aggiornamento di training effettivo: gate inizialmente zero ma gradienti finiti; nessun ramo accidentalmente morto.
- Un errore costante sintetico, una perturbazione localizzata e un caso con un'ancora estrema servono a verificare segni, formule e limiti; non si impongono prestazioni scientifiche da questi test.
- Regione vuota e insieme ancore vuoto gestiti esplicitamente.

### Parità e smoke reale
1. Sul backbone storico col normalizzatore storico, riprodurre il free S7 al 5%, riferimenti non vuoti e macro entro 0.05 K, scarti per-frame dichiarati.
2. Nei fold, riprodurre il riferimento B con i suoi input e statistiche; nessun confronto numerico diretto fra normalizzatori diversi.
3. Inizializzazione ΔT=0: tutti i K coincidono col backbone sul medesimo dominio.
4. Smoke GPU su un sistema storico, con training breve, eval sui sei K, output distinti e readback.
5. Kill/resume reale: ripresa dall'ultimo checkpoint completo, stessi stream, optimizer/scheduler/RNG e contatori. L'eventuale rumore numerico va confrontato con la ripetibilità del percorso non interrotto; non inventare identità bitwise del training.
6. Profilare RAM/VRAM/tempo per epoca, eval e costruzione residui; poi congelare risorse.

Zero beneficio nel breve smoke non è un bug da “correggere” cambiando architettura. L'overfit su pochi esempi può essere un controllo positivo, non una selezione basata su VAL. Nessun forward pesante sul login.

## 9. Valutazione scientifica e diagnostici di uso delle T

Per best e last, stessa base e stesso pacchetto di osservazioni:

| Metodo | Scopo |
|---|---|
| Free T0 | Riferimento K=0, riportato a ogni K sugli stessi pixel |
| Free + mean_K | Continuità col Mese 2 e controllo non robusto |
| Free + median_K | Comparatore di livello |
| Free + RBF_K | Comparatore spaziale economico |
| Free + modulo C_K | Metodo appreso |
| Oracle-offset | Solo limite della traslazione uniforme |

AO tau=1 può essere citato dallo storico, ma non richiede nuova campagna nel blocco minimo.

Riportare RMSE/MAE, bias firmato e |bias| per frame, centered-RMSE, regioni e IoU secondo B, tutti i K e domini, per seed/replica/condizione. Includere frame count, regioni vuote e p95. Nessun NaN globale tollerato; NaN legittimi alle ancore K=0 o regioni vuote espliciti.

Calcolare differenze appaiate entro lo stesso backbone, frame, replica e dominio. Per confronti di struttura, non basta battere free nella RMSE: confrontare centered e getti anche con mediana e RBF. Nessuna vittoria definita soltanto sulle ancore osservate.

### Uso delle T, da misurare direttamente
Su VAL e seen del blocco storico, dopo il freeze:
- confrontare errore agli stessi siti con K=0 e K=32;
- a K=32 mantenere siti/velocità fissi e aggiungere +100 K alle T osservate;
- a K=1 ripetere la perturbazione nei tre siti dei layout fissati;
- misurare variazione della predizione vicino/lontano dai siti e sul livello medio.

Sono stress test sintetici, non rumore realistico o misure corrette. Non usarli per tuning. Il cambiamento delle predizioni prova sensibilità ai valori, non accuratezza o correttezza fisica. Non richiedere una risposta di +100 K in tutto il dominio. Nei fold evitare ulteriori analisi outer non previste dal contratto congelato.

### Artefatti e gate

Gate tra loop ed evaluator del correttore: macro common ai cinque K>0 per replica/ruolo effettivamente registrati, tolleranza 0.05 K e scarti per-frame. Riferimenti mancanti bloccano la validazione. K=0 deve inoltre passare il confronto esatto col backbone a input e batch identici.

Salvare per-frame metriche, residui alle ancore, delta, chiavi e provenienza; statistiche della correzione, norme/gradienti del gate e confronto del backbone prima/dopo. Per le figure conservare frame1000 rep0 per le tre VAL, con scale comuni e tutti i metodi; non occorre salvare ogni campo.

Per outer non esiste una storia di loop: verificare cardinalità, maschere, domini e ricalcolo delle metriche, non inventare un replay storico. COMPLETE soltanto dopo verifiche e readback; output diagnostici conservati in caso di fallimento.

## 10. Decisione fra sviluppo C2 e fold C4

Dopo C2/C3, consegnare una tabella che mostri:
- differenze dal free, dalla mediana e dalla RBF su VAL e seen;
- centered, getti, livello, casi peggiori e tutti i tre seed;
- guadagni o peggioramenti a K=1 e ai K maggiori;
- se best è rimasto lo stato iniziale;
- sensibilità ai valori e costo aggiunto.

La chat amministratrice ed Enrico decidono se:
1. fermare C con un risultato negativo o con una baseline semplice già sufficiente;
2. trasferire **la ricetta congelata** ai due fold B, se il segnale strutturale è promettente o la domanda resta sostanzialmente aperta;
3. rinviare in attesa del contributo applicativo.

Nessuna soglia in kelvin inventata dopo i risultati, nessuna esclusione del seed sfavorevole e nessun avvio automatico di C4. Una sola deviazione favorevole di un seed non basta a dichiarare progresso strutturale generale.

La sola assenza di vantaggio della RBF non proibisce il modulo neurale C2: la RBF è un comparatore, non un oracle dell'informazione. Allo stesso modo un fallimento del modulo non dimostra che un'altra assimilazione non possa funzionare.

## 11. Organizzazione, risorse e catena sul cluster

Percorsi nuovi proposti:
- roadmap_C_assimilazione_residua.md (questa roadmap);
- configs/experiments/residual_c_protocol_v1.yaml;
- configs/experiments/residual_c_matrix_v1.json;
- docs/preparatory/c_residual_handoff.md;
- docs/preparatory/c_residual_report.md;
- results/preparatory/c_residual_assimilation/INDEX.md, con blocchi development e folds distinti.

Sonnet lavora in un worktree dedicato, indicativamente preparatory/c-residual-assimilation. Non modifica il codice B o A in parallelo. Consuma il contratto B con versione/hash, espone una API del correttore e un evaluator senza riscrivere i risultati free.

Hydrogen produce i sei backbone B e il contratto fold-specifico. C non li duplica, non ne cambia normalizzazione o selezione e non usa i checkpoint della variante A come base senza nuova ablation decisa.

Stime: profiling richiesto; modulo piccolo non significa campagna istantanea. Sei K in eval e forward del backbone hanno costo reale. Non ripetere automaticamente tutta l'eval S7 a 1/5/20%: qui solo velocità 5%.

Risorse possibili, nei limiti attuali da verificare e condividere con A/B:
- normal Advanced: mit_normal_gpu, account/QoS mit_amf_advanced_gpu, L40S;
- preemptable: mit_preemptable, mit_general, QoS ammesso, L40S, dopo resume verificato;
- CPU: mit_normal con account/QoS consentiti per baseline e aggregazione.

I quattro slot normal e quattro preemptable non sono una quota per agente. La chat amministratrice assegna slot e priorità; nessun job altrui va modificato.

Dopo autorizzazione di esecuzione, preparare training → verifica/freeze → eval → aggregazione con dipendenze Slurm, output nuovi e checkout congelati, senza dipendenza dal PC. Una catena non deve consumare output parziali o dichiarare successo se manca un gate. Se il mandato prevede stop dopo submission, consegnare job ID/percorsi/risorse e fermarsi senza polling locale. Nessun recupero infinito, duplicazione o nuova famiglia per aggirare un errore.

## 12. Limiti, stop e criterio di consegna

Questo studio resta numerico: ancore ideali senza rumore, tre layout accoppiati a tre maschere velocità, tre sistemi seed, frame correlati e VAL già usata nello sviluppo. K=4 non è un numero universale di termocoppie.

Non imporre hard consistency ai sensori reali futuri: rumore, ritardo, intrusività e posizioni accessibili richiedono un modello dell'osservazione e il contributo del domain expert. Nuovi layout/rumore/dataset non sono autorizzati da questa roadmap.

**STOP C sviluppo:** baseline e tre correttori validamente valutati, risultati negativi inclusi, best/last e identità K=0 verificati, costi e limiti pubblicati. C4 può restare non attivato con motivazione esplicita.

**STOP C conferma, se attivata:** sei correttori fold-specifici completati nel budget concordato, outer valutato dopo freeze, nessun uso della condizione esclusa nel fit di entrambi gli stadi, confronto con le baseline sullo stesso backbone e contratto B.

Nessuna delle due chiusure sceglie da sola il Mese 3, garantisce pubblicabilità o autorizza TEST. L'esito serve a Enrico per decidere, con il contributo dei supervisori, se perseguire l'assimilazione spaziale o mantenere la soluzione più semplice.
