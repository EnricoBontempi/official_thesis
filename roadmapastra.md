# ROADMAP ASTRA — Livello termico, struttura e nove temperature osservate

**PROBLEMA CENTRALE: distinguere una mappa spaziale sbagliata da una mappa traslata al livello termico sbagliato, e misurare quanto correggano poche T.**

**DOMANDA: quanto recuperiamo con l'offset perfetto del campo e quanto di quel recupero otteniamo davvero con nove misure termiche prefissate?**

**UTILITÀ PER LA TESI: valutare se poche ancore T possono completare la ricostruzione da PIV/velocità sparse nelle camere di combustione.**

- Repository: `EnricoBontempi/official_thesis`; branch `mission/dense-unet`.
- Decisione: Enrico, 18 novembre 2026, America/New_York.
- Stato iniziale: **PIANIFICATA, esecuzione dopo la sotto-missione Opus**.
- Analisi sorgente: [analisi tripla revisore s7.md](analisi%20tripla%20revisore%20s7.md), `6e2d176fa74d097aa8128c472103cf76c5a8a7a6`.
- Evidenze storiche: STOP E `a6b2bad7445170e6d821a14de9f7af9a1905ab2c`; diario `925f556823214e37f3abf9d73b41d6e5ed815e58`.
- Piani collegati: [roadmapopus.md](roadmapopus.md), [roadmapsparse.md](roadmapsparse.md), [roadmap_month2.md](roadmap_month2.md).

## 1. Contesto e significato ingegneristico

Lo scopo della tesi è ricostruire la temperatura istantanea della combustione da misure limitate, con motivazione PIV/accesso ottico e poche temperature disponibili. Il task free corrente usa due componenti di velocità su pochi siti, coordinate ed eventualmente NH3/rapporto di equivalenza phi. Nessuna temperatura è fornita al modello (`K=0`).

Nei risultati sparse alcune mappe hanno una struttura relativamente buona ma un livello termico spostato. La soglia fisica di una regione calda può amplificare molto questo errore: abbassare uniformemente il campo può far scomparire una grande porzione della regione oltre soglia anche senza distruggere la forma generale.

Astra cita `40NH3_0.9`: U-Net p_on best ha RMSE circa 39.92 K, bias −29.35 K e centered-RMSE 26.80 K; GLU p_on ha RMSE 29.24 K, bias −3.25 K e centered 28.55 K. L'IoU calda è circa .310 contro .925. Questi valori motivano una prova: **non bastano per concludere che correggere il bias renda equivalenti i campi**. Bisogna correggerli e misurare nuovamente regioni, errori e casi peggiori.

La campagna confronta tre livelli di informazione: free, offset oracolo, offset stimato con nove T. L'ultimo è un problema assisted, diverso dal free. L'oracolo usa tutta la T vera ed è soltanto un riferimento diagnostico.

Le nove temperature sono osservazioni ideali di simulazione, senza rumore né ritardo termico. Non equivalgono ancora a nove termocoppie reali. Il loro costo informativo va dichiarato; l'esperimento prepara una decisione ingegneristica, non certifica già uno strumento sperimentale.

## 2. Domande precise e uscite attese

| Domanda | Esperimento | Risposta richiesta |
| --- | --- | --- |
| Quanto errore dipende da una traslazione uniforme del livello? | Free contro oracle-offset | Recupero RMSE e regioni, residuo centered e casi in cui la forma resta errata |
| Nove T stimano abbastanza bene quella traslazione? | AO9 contro oracle e free | Errore di stima dell'offset e recupero sul campo, soprattutto fuori dalle ancore |
| Il vantaggio GLU nelle regioni calde sopravvive alla correzione? | Quattro neurali, stessi siti e misure | Ranking per condizione e metriche spaziali, senza sola macro |
| p serve ancora con poche T? | On/off nella stessa famiglia, free e AO9 | Variazione del gap p a informazione termica identica |
| La rete resta utile quando aggiungiamo T? | Neurale+AO9 contro baseline+AO9 e riferimento T-only | Quanto aggiungono velocità e ricostruzione neurale rispetto al solo livello termico |

Non è una nuova campagna di training di default: **zero training neurali necessari nel blocco principale**. Si applica una correzione scalare a predizioni congelate, con valutazioni offline. Se in futuro servirà apprendere dalle T o cambiare architettura, sarà una decisione successiva.

## 3. Ordine, fonti e rapporti fra roadmap

1. Prima completare o discutere la chiusura della campagna Opus e leggere il suo report.
2. La preparazione tecnica comune O0 di Opus si riusa; verificare che sia stata consegnata, non implementare una seconda pipeline parallela.
3. Il riferimento primario Astra sono i **quattro best S5 originali**, scelti su VAL 5%, seed 0. Questo preserva la domanda che ha generato la revisione.
4. I checkpoint Opus possono costituire un blocco secondario separato, congelato in A0 dopo averne visto la consegna. Nessuna sostituzione silenziosa del riferimento storico con il nuovo run migliore.
5. L'ingresso delle T estende esplicitamente il perimetro K=0 della roadmap sparse per questa sotto-missione e si collega allo Step 2.7 Mese 2. Non autorizza in generale tutti gli assisted o uno sweep K.
6. S7 e il completamento sparse/Mese 2 rimangono aperti. Le proposte Neon verranno riesaminate dopo entrambe le sotto-missioni.

Cursor riceverà incarichi per singolo passo. Nessuna submission o esecuzione automatica dell'intera roadmap deriva dalla pubblicazione di questo documento.

## 4. Disegno sperimentale principale

### 4.1 Matrice congelata

| Fattore | Scelta principale |
| --- | --- |
| Modelli neurali | U-Net p_on, U-Net p_off, GLU adattato p_on, GLU adattato p_off |
| Pesi | Quattro `best.pt` S5 seed 0, hash dal ledger STOP E |
| Trattamenti | Free K=0; oracle-offset; assisted AO9 tau=1 |
| Velocità | 5%, 819 siti, payload storico persistito |
| Repliche velocità | 0/1/2, stesse fra tutti i metodi |
| Ruoli | seen_train_eval 220 frame, val_eval 123 frame |
| Parametri operativi | Come il checkpoint, on oppure off; nessun'altra via indiretta |
| Layout T | Nove siti fissi, uguali fra metodi, frame e repliche velocità |
| Rumore T | Zero nel blocco principale, dichiarato |
| Selezione | Nessuna nuova selezione: checkpoint, siti, tau e metriche fissati prima dell'eval |

Le tre repliche velocità non diventano tre repliche termiche o tre seed di training. Per un dato frame le nove T restano le stesse. Eventuale sovrapposizione di siti T e velocità è consentita e va contata: sono comunque due grandezze osservate diverse.

### 4.2 Correzioni e unità

Siano `T_hat` la predizione free in kelvin e `T` il target, su N=16384 siti. Tutte le correzioni si applicano **dopo la denormalizzazione**, in K. Definire il bias firmato come media di `T_hat−T`; l'offset correttivo ha segno opposto.

**Oracle-offset**:

`delta_oracle = mean_grid(T − T_hat)`

`T_oracle = T_hat + delta_oracle`

Con pesi spaziali uniformi, sullo stesso intero campo il nuovo RMSE coincide col centered-RMSE originale. Questa identità vale per frame: la media dei quadrati e il quadrato della media di RMSE aggregate non sono intercambiabili. Non è un limite inferiore di tutti i ricostruttori, ma il minimo RMSE ottenibile traslando uniformemente quella predizione.

**AO9**, K=9 e tau=1 congelato:

`delta_AO9 = sum_anchor(T_obs − T_hat_at_anchor) / (9 + 1)`

`T_AO9 = T_hat + delta_AO9`

È la soluzione di una regressione dell'offset con penalità `tau*delta²`, usando somma dei nove residui quadratici. Il denominatore è K+tau: non confondere con una loss media o interpretare tau come varianza fisica. Per K=0 l'offset deve essere esattamente zero. Nessun clipping dei campi corretti, soglie variate, tau scelto su VAL o prior termico aggiuntivo nel forward.

Riusare l'implementazione AO consolidata della tesi se conforme; controllare segno, unità e convenzione della regolarizzazione, anziché creare una variante omonima incompatibile. Se il codice storico differisce dalla formula, fermare il freeze interessato e documentare la differenza prima di confrontare.

### 4.3 Siti termici e informazione consentita

Riutilizzare il layout fisso K=9 della tesi se esiste ed è verificabile, con gli stessi indici per tutte le famiglie e baseline. In assenza di un layout storico K=9 univoco, adottare una griglia 3×3: righe e colonne `[32,64,95]` zero-based nella griglia 128×128. Salvare i nove indici, le coordinate fisiche e il disegno prima della prima valutazione. Verificare che tutti i siti appartengano al dominio valido; se il dominio è mascherato, risolvere la geometria in A0 prima di eseguire, senza usare T per scegliere punti.

Nessuna selezione in base a hot spot, target del frame, errore del modello, gradiente T o risultato VAL. Il risultato riguarda questo layout, non dimostra robustezza al posizionamento. Variazioni di K/layout/rumore appartengono al riesame Neon o a un'estensione successiva.

Il modello free riceve soltanto l'osservazione canonica vi/vj, coordinate ed eventuale p. AO riceve separatamente i nove scalari T e le predizioni negli stessi siti. Soltanto il percorso oracle e le metriche leggono il campo target completo. Il renderer non deve diventare una via nascosta di correzione scientifica.

### 4.4 Baseline matched

| Riferimento | Informazione ammessa | Trattamenti |
| --- | --- | --- |
| GappyPOD p_on/off | Stesse velocità 5%, eventuale p; fit e lambda storici | Free e AO9; oracle come diagnostico separato |
| ParamMean | Solo p, prior fit-only congelato | Free e AO9; oracle separato |
| GlobalMean | Prior fit-only, nessuna velocità o p | Free e AO9; oracle separato |
| T-only costante | Soltanto le nove T | Campo costante uguale alla loro media; nessun uso di vi/vj o target fuori ancore |

Il T-only costante è un controllo di livello a basso costo, non un'interpolazione spaziale ottimizzata. Se si aggiunge un interpolatore spaziale T-only, congelarne prima formula e parametri e pubblicarlo come riferimento ulteriore. Non inventare un comparatore dopo aver visto quale batte la rete.

Riusare gli oggetti fittati e verificati di GappyPOD/medie e il gate storico. Se non disponibili, segnalarlo in A0: nessun nuovo fit o tuning silenzioso. ParamMean usa p ed è quindi una baseline p_on; non descriverla come matched al divieto p di un neurale p_off. GlobalMean+AO9 e T-only restano confronti senza p. Il confronto fra ricette con diversa informazione va etichettato.

## 5. Metriche e lettura del risultato

Per ciascun frame/replica/trattamento salvare: RMSE, MAE, bias firmato, |bias|, centered-RMSE; delta applicato, delta_oracle ed errore `delta_AO9−delta_oracle`. Pubblicare medie per condizione e poi macro uniforme fra condizioni e repliche. Riportare separatamente seen e VAL.

Campi e regioni:

- Intero campo e **off-anchor**, cioè gli altri 16375 siti, identici fra metodi. Le nove T non devono migliorare la valutazione soltanto nei punti usati dalla correzione.
- Soglie e definizioni `high_/jet_/peak_` storiche, identificate con la loro fonte. Per la regione calda motivante usare la soglia storica T≥2100 K.
- Area calda vera e predetta, IoU, RMSE regionale e conteggi validi. Per regioni vuote usare una convenzione esplicita e conteggi: non trasformare NaN in zero errore o uno di IoU senza dichiararlo.
- Metriche regionali off-anchor con gli stessi siti esclusi; le maschere di riferimento possono dipendere da T vera per la sola valutazione e devono restare uguali fra trattamenti.
- Pannelli sulle **tre** condizioni VAL, incluso `40NH3_0.9`, a frame 1000 ove disponibile; più frame possono essere mostrati se scelti con regola preventiva. Nessuna scelta solo dei casi riusciti.

La traslazione uniforme lascia invariata la centered-RMSE della stessa predizione sullo stesso insieme di pixel: verificarlo. L'IoU può invece migliorare o peggiorare. Non promettere miglioramento monotono di tutte le metriche, neppure per l'oracle, che ottimizza soltanto RMSE globale.

Per descrivere quanta parte del margine oracle viene recuperata, si può riportare per frame il recupero in MSE `[(MSE_free−MSE_AO9)/(MSE_free−MSE_oracle)]`, con denominatore >1e-6 K²; altrimenti non definito. Non troncare valori negativi né aggregare soltanto frame favorevoli. È una misura descrittiva secondaria, non la nuova metrica di selezione.

## 6. Passi operativi e STOP

### A0 — Consegna Opus, preparazione comune e freeze Astra

- [ ] Leggere report Opus, diario, analisi tripla e report 05; verificare quali rilievi O0 sono stati davvero risolti.
- [ ] Identificare quattro best S5 e hash, manifest dei dati e maschere, baseline fittate e layout T9.
- [ ] Confermare matrice primaria S5. Elencare separatamente eventuali run Opus candidati al blocco secondario; non attendersi che Opus abbia risolto l'errore di livello.
- [ ] Congelare siti, tau, metriche, soglie, campi invalidi, convenzioni IoU e tolleranze prima di lanciare le valutazioni.
- [ ] Preparare config e manifest dedicati `configs/experiments/month2_astra_level_structure.yaml`, senza cambiare la selezione o i pesi storici.
- [ ] Rendere esplicita l'eccezione assisted A0–A4 nel contratto: non rimuovere genericamente tutti i divieti K>0 né autorizzare training assisted.

**Consegna:** freeze e inventario, prerequisiti mancanti distinti da decisioni. **STOP A0**, nessun training né inferenza massiva.

### A1 — Implementazione AO e test dell'informazione

- [ ] Riusare il percorso di inferenza S6 corretto, con adattatore post-processing comune a neurali e baseline.
- [ ] Test sintetico con offset noto: formula e segno, shrink tau, K=0 identità, oracle RMSE=centered e centered invariata dopo traslazione.
- [ ] Test AO permutando l'ordine delle nove ancore: risultato invariato entro tolleranza dichiarata.
- [ ] Mutare T vera fuori dalle nove ancore: predizione free e AO9 invariate; oracle e metriche possono cambiare. Mutare una T osservata: offset cambia secondo la formula, senza modifica dei pesi.
- [ ] Verificare p_off da mutazione dei p sorgente e isolamento delle velocità nascoste nel percorso reale; riusare i contratti già testati.
- [ ] Test off-anchor: esclusi esattamente nove siti distinti; regioni vuote e conteggi gestiti correttamente.
- [ ] Policy fp32 senza AMP/TF32/compile esplicita. Replay del K=0 sui checkpoint S5: macro RMSE entro **0.05 K** dal consolidato a ruoli/repliche corrispondenti; investigare scarti, non aggiornare automaticamente il riferimento.

**Consegna:** codice/test e replay piccolo, SHA prima dell'esecuzione scientifica. **STOP A1**. Il test oracle legge T per definizione; questo non autorizza il suo uso in un metodo deployable.

### A2 — Valutazione primaria offline, nessun training

- [ ] Inferenza congelata dei quattro best su seen e VAL al 5%, repliche 0/1/2. Calcolare free/oracle/AO9 dalla stessa predizione base; nessun nuovo forward necessario per ogni offset.
- [ ] Calcolare i riferimenti matched, riusando predizioni solo se corredate da hash e identità completa, altrimenti inferenza nuova dagli oggetti congelati.
- [ ] Salvare per-frame e per-condizione, offset, conteggi, manifest e tempi; output nuovi in `results/month2/astra_level_structure/` e directory remote dedicate.
- [ ] Non materializzare simultaneamente tutti i campi di tutti i metodi se non necessario: elaborazione a batch/chunk e salvataggio compatto.
- [ ] Se si usa Engaging: CPU con `mit_amf_advanced_cpu`/QoS omonima su `mit_normal`; GPU con `mit_amf_advanced_gpu`/QoS omonima su `mit_normal_gpu`. Preferire normal L40S per inferenza neurale; evitare GPU solo per sottrarre offset se le predizioni sono già disponibili.
- [ ] Committare sorgenti/config prima del job e registrare SHA/hash; niente calcolo pesante su login/WSL, niente dipendenze implicite dal pool guasto.
- [ ] Dopo ogni submission: ID, log, output e comandi manuali, quindi **STOP senza polling, attese o rilanci automatici**. HONF intatto.

La matrice primaria contiene quattro modelli × due ruoli × tre repliche velocità × tre trattamenti; questi sono setting di valutazione, **non 72 training**. Le baseline si aggiungono con la loro corretta informazione. La durata sarà misurata, senza tetti scientifici arbitrari in ore.

### A3 — Recupero e report decisionale

Quando Enrico segnala fine job, una lettura di accounting/log e controllo completezza per chiavi. Risolvere errori di esecuzione prima di proclamare una classifica.

- [ ] Tabelle free/oracle/AO9 all-field e off-anchor, per ciascuna condizione e macro; nessuna sostituzione dei risultati STOP E.
- [ ] Confronto dei delta on/off e U-Net/GLU dopo ciascun trattamento; baseline T-only/medie/Gappy nello stesso quadro informativo.
- [ ] Diagrammi/pannelli a scale comuni di temperatura ed errore firmato, siti T visibili e soglia calda dichiarata.
- [ ] Risposta esplicita: quanto errore è livello, quanto è forma, quanto recuperano nove T, e dove restano fallimenti rilevanti per la combustione.
- [ ] Distinguere miglioramento dello score VAL da validazione indipendente: checkpoint scelti su VAL e analisi motivata da casi VAL già osservati.
- [ ] Due seed non esistono nel blocco S5 primario; tre maschere non consentono inferenza sulla dispersione fra training.

**Consegna:** `docs/month2/07_astra_level_structure_and_anchors.md`, risultati con provenienza, figure e diario. **STOP A3**, nessuna scelta S7 o training nuovo automatica.

### A4 — Estensione con checkpoint Opus, se utile e concordata

Questo blocco collega le due domande senza confonderle. Dopo O6/A3, proporre la matrice prima di eseguirla.

Se Opus ha consegnato C149 on/off, il confronto naturale è S5-38 seed 0 contro C149 seed 0, stessi trattamenti e osservazioni; se necessario aggiungere A38 seed 1 per una sensibilità descrittiva. **Nessun nuovo training Astra** è richiesto per questo blocco. Non scegliere, fra molti checkpoint Opus, solo quello che massimizza il guadagno AO9.

Domanda aggiuntiva: più istantanee riducono il bias che AO9 deve correggere, migliorano la struttura residua, oppure le due azioni si completano? Tenere tabelle S5 e Opus distinte e pubblicare il confronto appaiato. Se C149 manca, l'estensione ricca resta non svolta; non rietichettare B19 come aumento dati.

**STOP A4** dopo la consegna autorizzata; nessuno sweep K/layout, rumore o multidensità implicito.

## 7. Scenari e decisioni successive

| Esito misurato | Interpretazione consentita | Direzione da discutere |
| --- | --- | --- |
| Oracle migliora molto, AO9 recupera una parte consistente anche off-anchor | Il livello è un limite importante e nove T aiutano con questo layout ideale | Dare centralità a free/assisted e studiare fattibilità delle misure termiche |
| Oracle migliora, AO9 aiuta poco o peggiora | La media globale è correggibile, ma nove residui locali non la stimano bene in questa ricetta | Analizzare distribuzione dei residui, siti e shrink; eventuale sweep K dopo decisione |
| Oracle migliora poco e restano errori spaziali forti | L'offset globale non basta | Priorità a ricostruzione della struttura, regime dati/densità o rappresentazione |
| GLU conserva un vantaggio regionale dopo oracle/AO9 | Il vantaggio non si esaurisce nel solo livello | Considerare replica matched U-Net p_on contro GLU p_on |
| U-Net+AO9 raggiunge un profilo utile a costo minore | Ricetta assisted candidata, nel perimetro misurato | Considerare conferma on/off o architetturale senza dichiarare vittoria universale |
| T-only o media+AO9 sono vicini alle reti | Una parte importante della metrica è già spiegata dalle poche T | Misurare esplicitamente il contributo incrementale delle velocità e delle regioni dinamiche |
| Correzione aiuta macro ma danneggia casi caldi critici | La media nasconde fallimenti operativi | Non scegliere il modello sulla sola RMSE macro |

Non stabilire a posteriori una soglia di «recupero sufficiente» solo perché un risultato la supera. La decisione finale deve riportare guadagni in kelvin, regioni, casi peggiori e costo osservativo; eventuali tolleranze applicative richiedono motivazione e confronto con i supervisori.

## 8. Chiusura e ritorno al percorso della tesi

- [ ] Matrice S5 primaria completa oppure lacune esplicite; nessun metodo escluso perché sfavorevole.
- [ ] Oracle chiaramente separato da metodi che usano informazioni disponibili.
- [ ] K=0 replay verificato; AO9 usa soltanto i nove valori dichiarati e mantiene il medesimo campo base.
- [ ] Baseline matched, metriche off-anchor, regioni e tutte le condizioni documentate.
- [ ] Risposta alle cinque domande della sezione 2 con limiti di seed, layout, rumore ideale e selezione.
- [ ] Estensione Opus consegnata oppure rinviata esplicitamente; nessun nuovo training necessario attribuito per errore ad Astra.
- [ ] Proposte Neon riesaminate: multidensità, replica delle famiglie, K/layout e pseudo-estrapolazione, scegliendo cosa risponde ai limiti rimasti.
- [ ] Aggiornare decisione S7, roadmap sparse, Step 2.7 e impegni 2.8–2.9 Mese 2 secondo risultati reali. Un primo AO9 non chiude automaticamente tutto lo Step 2.7 o il Mese 2.

Questa missione deve produrre una decisione su quali osservazioni servono, non soltanto una nuova classifica di algoritmi. La direzione finale rimane la ricostruzione sparse utile alla combustione; il rigore serve a distinguere il beneficio reale da quello ottenuto cambiando informazioni o criteri a confronto.
