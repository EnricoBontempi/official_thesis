# Roadmap missione — Consolidamento delle baseline prima dei neurali

**Progetto:** esclusivamente `EnricoBontempi/official_thesis`, tesi MIT/Deng Lab.  
**Base scientifica revisionata:** `822ed52c07eded37bba4f79af0f2177da4aa36eb`.  
**Data di preparazione:** 2026-11-16, America/New_York.  
**Stato:** missione autorizzata da Enrico, **non ancora eseguita**.  
**Esecutore previsto:** Cursor IDE nel PC/WSL locale con gli shard esistenti.  
**Piano principale:** [roadmap_month2.md](roadmap_month2.md), consolidamento Step 2.2–2.3. Questo documento ne dettaglia soltanto la chiusura tecnica; non autorizza Step 2.4–2.9.

## Obiettivo e limite finale

Rendere verificabile il riferimento numerico delle baseline: correggere prove deboli e gestione temporale, identificare esattamente codice/dati/ambiente, ripetere **una volta** l'esperimento scientificamente invariato e confrontarlo con i risultati storici.

Procedere autonomamente fra i passi sotto quando i controlli passano. **Fermarsi prima di qualsiasi implementazione, preparazione di training o training neurale.** La consegna sarà letta da Enrico e ChatGPT prima del prossimo incarico. Non chiedere conferme generiche mentre Enrico è assente; in caso di decisione sostanziale salvare il lavoro e consegnare un blocco motivato.

### Esiti ammessi

- **COMPLETATA, IN ATTESA DI REVISIONE:** correzioni verificate, diagnosi dati sufficiente, rerun riproducibile e confronto superato.
- **PARZIALE, BLOCCATA DA DATI/AMBIENTE:** codice/test eseguiti, ma raw o dipendenze reali mancanti, risorse insufficienti o run interrotto.
- **BLOCCATA DA DECISIONE SCIENTIFICA:** serve cambiare frame, fit set, split, campi, obiettivo o impostazioni dell'esperimento; oppure l'anomalia dei raw ne mette in dubbio l'identità/validità.

Un report con esito parziale è una consegna valida; non sostituire controlli reali mancanti con PASS fittizi.

## Decisione concordata dopo Opus e Astra

Entrambi hanno revisionato indipendentemente `822ed52` e concluso **«pronti dopo correzioni circoscritte»**. Hanno verificato matematica/coerenza degli artefatti con modalità differenti, ma **non avevano i raw e non hanno rieseguito il run completo**. La concordanza non certifica da sola la catena raw→CSV.

Rilievi da risolvere qui:

| ID | Evidenza/questione | Azione della missione |
|---|---|---|
| C1 | Manifest senza SHA eseguito, stato dirty, hash sorgenti e ambiente; YAML/runner/risultati nel medesimo commit | Identità dei sorgenti prima del nuovo run e manifest completo. Vecchia preregistrazione cronologica non attestabile. |
| C2 | Nel runner, `hidden_velocity_mutation_maxabs` confronta gli stessi input; due flag sono `True` letterali; copia `mi_poison` inutilizzata | Mutazioni effettive prima di costruire il sample, predizioni reali e assert. |
| C3 | `t_shape=2000` vs `shape_t=2001` per `40NH3_1.2.h5`; pad generico presume che manchi la coda | Diagnosi raw, SHA e confronto con audit; eliminare riparazioni generiche non giustificate. |
| C4 | Fingerprint di aggregazione incompleto e `except Exception` che può saltare gruppi; alcune scelte sono hardcoded | Fail-fast, identità completa delle configurazioni e contratto YAML realmente rispettato o rifiuto esplicito. |
| C5 | Report generato mescolato a commento editoriale; file per-frame ~65 MB | Nuovi output separati, report automatico distinto dall'interpretazione; nessuna riscrittura della storia Git. |

Precisazioni scientifiche da **registrare**, non trasformare in nuovi esperimenti:
- AO τ=1 a K=9 peggiora seen p_on in 18/20 condizioni; media RMSE ~108.36→111.21 K. Non garantisce miglioramento delle ancore.
- T-block migliora anche Centered-RMSE: non sostenere che tutte le correzioni lineari cambino soltanto il livello.
- VAL contiene tre condizioni moderate e non rappresenta tutta la difficoltà dei regimi ricchi/high-NH3. Non cambiare split né usare seen come nuova selezione.
- p_off/on confronta pipeline selezionate, con prior/base/λ differenti, non il solo effetto causale di un modulo.
- Oracle proietta T vera sul blocco T della **base congiunta** TRAIN-fit; non è una predizione da sensori, una POD T-only ottimale o un limite fisico universale.
- Normalizzazione 1/Mu e w_T/K definisce un particolare stimatore regolarizzato, non un modello di rumore già calibrato.
- Non usare insuccessi di HGB o di una futura rete per dichiarare non identificabilità fisica.

**Rinviati:** U-Net, GLU, controllo di disallineamento temporale della velocità, R=64, nuovi λ/w_T, τ calibrato, altri K/coperture, fold di pseudo-extrap, incertezza/assimilazione appresa. Rimangono domande per la prossima decisione.

## Invarianti scientifici del rerun

Usare il contratto `configs/experiments/month2_baselines.yaml` della base revisionata, senza ottimizzarlo di nuovo a posteriori:

- 20 TRAIN-fit × 38 frame = **760**; seen = **220**; VAL = **123**; nessun payload TEST.
- 2C ch.11/12, single-frame, sparse 5%, T ch.10 grezza in K.
- seed maschera **20261116**, repliche **0/1/2**, stesso layout/ricette/maschere.
- POD **R=32**; p_off/on come prima; GlobalMean/ParamMean, Gappy free/AO/T-block, Oracle e HGB dense diagnostico come nel run originale.
- ParamMean LOCO invariato; HGB stessi pixel, seed e iperparametri.
- Free λ = **[1e-4,1e-3,1e-2,1e-1,1]**.
- Assisted K=9: λ = **[1e-3,1e-2,1e-1]**, w_T = **[0.5,5,50]**.
- Valutazione K = **0/9/32**; AO τ = **1**; selezione assisted a K=9, senza rituning a K=32.
- Stesso ordine delle medie: frame→condizione→VAL→repliche. TRAIN-fit e seen restano diagnostici distinti.
- Stats numeriche fit-only esistenti invariate, salvo difetto dimostrato: in tal caso **STOP**, non rigenerare silenziosamente.
- Non pretendere gli stessi iperparametri vincitori con un assert imposto: confrontare l'intera ricerca e indagare qualsiasi differenza.

Sono autorizzati cambi di interfaccia, controlli, metadati, destinazioni output e rappresentazione CSV, purché non cambino l'esperimento. Le nuove esecuzioni sono repliche di verifica; non rendono retroattivamente preregistrato il run storico.

## M0 — Sincronizzazione sicura e inventario

**Attività**
1. Leggere istruzioni correnti, diario, questo documento, report Step 2.2/2.3, YAML e codice reale.
2. Eseguire `git status`, fetch e controllo divergenze. Non perdere modifiche locali; niente reset hard, clean distruttivo, stash indiscriminato o force.
3. Con working tree pulito aggiornare `main` con fast-forward e creare un branch dedicato, proposta `mission/baseline-consolidation` (nome diverso se già usato). Per modifiche locali sovrapposte o divergenze non risolvibili con certezza fermarsi e documentare; non committare lavoro altrui per ottenere artificialmente uno stato pulito.
4. Verificare raw locali, Python/venv, RAM/disco e inventario. Usare percorsi configurabili; non presupporre che il path WSL sia identico su altri host.
5. Registrare base iniziale ed esistenza del CSV storico prima di qualsiasi esecuzione.

**Output:** preflight nel futuro report di missione; nessuna modifica dei raw.  
**Passaggio:** ambiente sufficiente. Dati assenti: si possono completare correzioni e test sintetici, ma non M1 reale/M5 completo. Niente download, trasferimenti o cluster.

## M1 — Diagnosi raw del timestamp, senza inventare una timeline

**Problema:** il lettore aggiunge la coda a qualunque t corto, anche vuoto/non monotono. L'audit storico riporta già t_start=1.5 e t_end=2.0; la diagnosi «manca l'ultimo» non è dimostrata.

**Attività**
1. Sullo shard originale di `40NH3_1.2.h5` misurare byte size/SHA256 e confrontarli con la voce esatta dell'inventario remoto (LFS SHA256 se presente). Controllare sim_id, shape campi, t_shape, lunghezza byte effettiva e dtype, senza passare dal pad.
2. Salvare estremi, finitezza, monotonia, distribuzione di diff(t), posizioni di duplicati/salti e confronto con l'audit. Non inferire il punto mancante dalla sola mediana.
3. Esaminare differenze fra campi numerici consecutivi per T/vi/vj, con calcolo a blocchi e senza grandi copie; cercare duplicati esatti, salti anomali e verificare validità/continuità dell'indice 2000. Differenze piccole/grandi sono indizi, non prova definitiva della sincronizzazione.
4. Correggere il comportamento generale: timeline valide accettate; dati temporali non finiti, decrescenti o mismatch inattesi rifiutati. Nessuna estrapolazione generica, replica di un timestamp unico o NaN silenzioso.
5. Se il caso noto è spiegato con evidenza sufficiente, gestirlo con eccezione strettamente identificata (sim_id e identità raw), warning e provenienza per condizione. Qualsiasi coordinata ricostruita va distinta dall'originale.
6. Se i campi sono verificati validi ma l'associazione temporale rimane ambigua, è consentito mantenere il solo percorso **single-frame per indice** con flag esplicito di tempo non validato e senza coordinate inventate. Se l'interfaccia non lo consente con una piccola modifica sicura, fermarsi dopo la diagnosi invece di forzare il pad.

**Arresto scientifico:** SHA non coincidente non spiegato, frame invalidi, necessità di escludere/spostare frame, rifittare statistiche o cambiare sincronizzazione. Non scambiare «nessun duplicato trovato» per dimostrazione di campionamento uniforme.

**Output:** `docs/month2/mission_baseline_timestamp_audit.md` e piccolo JSON di diagnosi/provenienza. Raw intatti; niente uso temporale autorizzato.  
**Passaggio:** indici/campi del confronto single-frame confermati e anomalia circoscritta; oppure STOP motivato.

## M2 — Controlli di isolamento realmente eseguiti

**Attività**
- Sostituire tautologie, flag costanti e copie inutilizzate nel runner con prove che falliscono se l'invariante è violato.
- Su alcuni frame reali TRAIN/VAL (nessun TEST), per entrambe le modalità p e le varianti pertinenti:
  1. copiare il frame sorgente;
  2. alterare velocità non osservate, T fuori ancore e vk separatamente;
  3. ricostruire sample/maschere;
  4. confrontare input ammessi e **predizioni effettive**.
- Free: mutare tutta T target non deve cambiare la predizione; assisted: T alle ancore resta fissata nel test negativo.
- p_off: mutare metadata p non deve influire sul forward; non mutare sim_id/seed e poi attribuire differenze alle sole p.
- Controllo positivo: una perturbazione osservata deve poter cambiare l'output del metodo che la usa. GlobalMean deve invece restare invariato ai sensori; ParamMean risponde a p, non alla velocità.
- K=0: usare il modello T-block effettivo con gli stessi iperparametri a flag free/assisted; non confrontare oggetti diversamente fittati per imporre uguaglianza.
- AO: invarianza di Centered-RMSE sul singolo frame, senza supporre che RMSE o bias assoluto migliorino.
- Registrare valore misurato, tolleranza, numero di casi e assert; non un semplice PASS scritto a mano.
- Test raw mancanti marcati `pytest.skip` con ragione, non `return`. Aggiungere test sintetici della gestione temporale anche senza raw.

**Output:** test e riepilogo invarianti effettivi.  
**Passaggio:** test negativi/positivi appropriati passano. Una fuga reale richiede correggere il metodo e documentare l'effetto prima del rerun; se cambia la scienza, STOP.

## M3 — Runner vincolato al contratto e output separati

**Attività**
1. Consumare i parametri del YAML oppure rifiutare valori incompatibili. R/K/griglie/seeds/scaling non devono sembrare configurabili quando il runner li ignora.
2. Completare identità dei gruppi: ruolo, metodo, variante, p, K, copertura/layout, replica, seed, R, λ, w_T e τ pertinenti. Non mediare iperparametri differenti; nulla di questo richiede riscrivere il comportamento storico del Mese 1.
3. Sostituire eccezioni generiche che saltano gruppi con fallimento esplicito. Verificare completezza/identità delle configurazioni e righe rispetto al piano, con diagnostici/oracle distinti.
4. Selezione sempre su VAL; esportare ricerca e vincitori. Non introdurre nuovi trial.
5. Aggiungere destinazione output/run_id al runner. **Non sovrascrivere** tabelle/meta/report del run `822ed52` o artefatti del Mese 1.
6. Separare report automatico da interpretazione manuale. Salvare i risultati nuovi sotto una directory run dedicata, per esempio `results/baseline_consolidation/<run_id>/`; i path definitivi devono risultare nel manifest.
7. Per il nuovo per-frame grande usare CSV gzip con verifica di lettura, o altra rappresentazione compatta già supportata; summary e metadati restano facilmente leggibili. Non cancellare né riscrivere la storia del CSV storico da ~65 MB. Checkpoint/POD/cache pesanti restano fuori Git, con percorso e hash se salvati.
8. Test suite nell'ambiente reale con torch/pyarrow; niente moduli segnaposto usati per dichiarare esecuzione production. Smoke sintetico o reale compatto prima di congelare il codice.

**Output:** codice/config di replica/test pronti; test, comandi e limiti registrati.  
**Passaggio:** stessi dati e algoritmo previsti, test appropriati passati; nessun trial scientifico aggiunto.

## M4 — Congelamento prima del rerun

Questo passaggio è **sequenziale e obbligatorio**.

1. Committare su branch dedicato codice corretto, configurazione finale e strumenti diagnostici. Non includere file estranei. Il commit precede i nuovi risultati.
2. Verificare checkout pulito prima del run (output futuri in posizione ignorata/esterna finché in esecuzione). Non far riscrivere al runner config o report tracciati durante il preflight.
3. Registrare prima del fit: run_id, timestamp con fuso, HEAD, stato Git, hash sorgenti eseguiti e configurazione, Python/dependencies/BLAS e thread/seed rilevanti. Non esportare variabili ambiente complete, token o credenziali.
4. Registrare fit set e frame, normalizzatore/prior/POD e hash/provenienza disponibili; per gli oggetti ancora da fittare registrarli dopo il fit. Identificare gli shard tramite inventario e checksum **effettivamente verificati**, distinguendoli da hash solo attesi; almeno il caso anomalo richiede verifica raw.
5. Prima di leggere i risultati nuovi fissare la regola di confronto: colonne comuni, pairing tramite identità, comportamento NaN e tolleranze. Default per metriche in K: atol **1e-6 K**, rtol **1e-8**. Hash/ID/conteggi/griglie devono coincidere esattamente; anomalie non numeriche non sono tolleranze floating-point.
6. Il commit finale che aggiunge risultati non coincide col commit eseguito: conservarli entrambi in modo esplicito.

**Passaggio:** sorgenti e protocollo identificabili; il vecchio run resta storico con provenienza incompleta, senza attestazioni retroattive inventate.

## M5 — Una replica completa CPU invariata

- Autorizzata **una** nuova esecuzione completa dopo smoke/test/congelamento, non una ricerca aperta.
- Stessa matrice e griglie del contratto storico. R=32, K=0/9/32, τ=1 e nessuna estensione di λ/w_T.
- Fare prefetch per traiettoria e riusare frame/fit; contare decode, tempi e picco memoria. Il vecchio run riporta 23 decode, ~32 min e ~10.4 GiB: riferimenti da verificare, non promesse o assert di performance.
- Monitorare RAM/spazio e avanzamento. Non modificare swap, servizi, configurazione globale del PC o ambiente condiviso per far entrare il run. Partire senza parallelismo aggressivo. Se risorse insufficienti, checkpoint/log e STOP.
- Limite operativo: **2 ore per il singolo full run**, salvo completamento imminente dimostrabile; nessun ciclo automatico di rerun finché i numeri coincidono. Documentare un'interruzione; non ridurre la matrice in silenzio.
- Nessun TEST/freeze, GPU, Engaging, training neurale o accesso esterno a persone.
- Gli output riportano tutti gli invarianti e l'esito è fallimento se uno obbligatorio fallisce.

**Output:** replica completa, manifest, ricerca, tabelle, report automatico e log separati.  
**Passaggio:** output completi, oppure consegna parziale esplicita.

## M6 — Confronto con il run storico

1. Confrontare nuovi CSV (anche compressi) con quelli conservati di `822ed52`: pairing per tutte le identità pertinenti, conteggi, repliche, frame, observation hash, griglie e selezioni.
2. Verificare summary dal per-frame e dal per-condition su entrambe le esecuzioni. Non confrontare solo le celle migliori.
3. Applicare le tolleranze fissate in M4 a metriche comuni e gestione dei mancanti. Colonne di tracciabilità nuove sono ammesse; campi scientifici non vanno esclusi per ottenere PASS.
4. Produrre `comparison.json` e tabella con max differenze, righe mancanti/aggiunte, selezioni e classificazione:
   - identico/entro tolleranza;
   - differenza spiegata e localizzata;
   - differenza non spiegata.
5. Se oltre tolleranza, diagnosticare con piccoli controlli (dtype, RNG, ordine frame, solver, versioni). Non allargare tolleranze dopo i risultati e non rilanciare l'intera matrice ripetutamente.
6. Se un bug scientificamente rilevante è dimostrato, conservare entrambi i risultati e dichiarare quelli precedenti superati; STOP per revisione prima di scegliere una nuova baseline. Il rerun invariato non obbliga a riprodurre un errore per essere «corretto».

**Passaggio:** confronto motivato; differenze non risolte impediscono lo stato COMPLETATA, non la consegna del lavoro.

## M7 — Consegna e arresto

Creare `docs/month2/mission_baseline_consolidation_report.md` con:
- stato finale e checklist M0–M7;
- problemi riprodotti e correzioni, separati dai rilievi non confermati;
- diagnosi timestamp, validità single-frame e limitazioni temporali;
- ambiente, commit eseguito/finale e comandi riproducibili;
- test reali, sintetici, skip e controlli non svolti;
- risultati della replica e differenze rispetto allo storico;
- costi/memoria/decode misurati;
- file/output persistenti e hash;
- questioni ancora aperte prima dei neurali.

Aggiornare diario in forma breve, link README e nota della roadmap Mese 2. Correggere tramite voce datata le vecchie affermazioni troppo forti («preregistrato», gap necessariamente non lineare, sensori che garantiscono miglioramento); non cancellare la storia e non modificare il proposal o il report LaTeX inviato.

Committare risultati compatti/report sul branch della missione, push senza force e aprire PR verso main se lo strumento è disponibile. **Non fare merge e non integrare il codice della missione direttamente in main.** Se apertura PR non disponibile, consegnare branch e link compare senza bloccare il push. Preservare modifiche concorrenti e verificare SHA remoto.

Messaggio finale dell'agent:
1. stato finale;
2. correzioni e risultati principali;
3. timestamp e dati;
4. test e confronto numerico;
5. risorse;
6. blocchi o decisioni;
7. branch, commit del codice eseguito, commit finale e PR;
8. conferma esplicita: **nessun lavoro neurale avviato**.

**STOP finale:** anche se tutto passa, attendere Enrico/ChatGPT. Non preparare U-Net, GLU, nuovi esperimenti o nuovi prompt per continuare automaticamente.

## Checklist della missione (compilazione dell'esecutore)

- [ ] M0 — ambiente e branch sicuri.
- [ ] M1 — diagnosi raw e politica temporale documentata.
- [ ] M2 — invarianti effettivi e test.
- [ ] M3 — contratto/aggregazione/output corretti.
- [ ] M4 — codice/config committati prima del run, provenienza e confronto fissati.
- [ ] M5 — replica invariata completa oppure blocco esplicito.
- [ ] M6 — confronto completo e differenze spiegate.
- [ ] M7 — report, diario, branch/PR; arresto prima dei neurali.
