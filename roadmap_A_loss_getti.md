> **Nota del 24 novembre 2026.** **SOSPESA — NON ESEGUIRE.** La priorità corrente è la definizione del problema applicativo e lo studio di robustezza a osservazioni non ideali descritto in [roadmaptests.md](roadmaptests.md). Eventuale riattivazione soltanto dopo riesame e decisione esplicita di Enrico.
>
> Il piano storico sotto questa nota è conservato integralmente. Secondo quanto riferito da Enrico, non è stato eseguito.

# Roadmap A — Loss regionale per migliorare i getti freddi

**Studio preparatorio successivo al Mese 2 · 22 novembre 2026 · America/New_York**  
**Responsabile scientifico:** Enrico Bontempi  
**Esecutore previsto:** Cursor  
**Coordinamento:** nuova chat amministratrice delle attività A/B/C.

## Stato e mandato

- [x] Roadmap preparata e pubblicata su richiesta di Enrico.
- [ ] A0 — Contratto B acquisito e variante congelata.
- [ ] A1 — Modifica minima, test, replay e smoke.
- [ ] A2 — Sei training della variante, sui due fold B.
- [ ] A3 — Valutazione matched con i sei riferimenti B.
- [ ] A4 — Report e consegna.

**Questa pubblicazione è documentale: non lancia job e non costituisce la roadmap generale del Mese 3.** L'incarico della nuova chat deve autorizzare esplicitamente implementazione e matrice prima della submission. Non sono autorizzati altri pesi, soglie, modelli o dati per ottenere un risultato favorevole.

Il Mese 2 e il report già inviato restano conclusi nel loro perimetro. A è una prova preparatoria della proposta di loss regionale, non un requisito retroattivo della chiusura.

## 1. Domanda scientifica

**Aumentare moderatamente il peso degli errori nei getti freddi durante il training migliora la loro ricostruzione su condizioni escluse, senza un costo eccessivo sul resto del campo?**

A cambia **soltanto la loss** della U-Net masked free p_on. Tutto ciò che riguarda dati, osservazioni, inizializzazione, stream, budget, selezione e valutazione proviene da B.

Il problema nasce dagli errori regionali elevati del Mese 2, che una traslazione uniforme non corregge. Non assumere che un miglioramento regionale sia già un miglioramento generale: la variante può spostare l'errore verso la zona calda, alterare il livello o specializzarsi alla soglia scelta.

La T vera serve per la supervisione e per definire la regione **solo nella loss**. In inferenza non è disponibile e non viene usata come input, maschera o criterio di selezione dei pixel da correggere. K=0 per tutto A.

## 2. Fonti e dipendenze

| Fonte | Riferimento |
|---|---|
| Contratto scientifico di confronto | [roadmap_B_generalizzazione.md](roadmap_B_generalizzazione.md), prima pubblicazione 81f59438204cd5b0927ef4b3988b9a32026627f5 |
| Sintesi delle proposte | [reviewmese2.md](reviewmese2.md), origine 9a6277f271652a2b6416cad14a6609ddb2c1a021 |
| Studio C, distinto da A | [roadmap_C_assimilazione_residua.md](roadmap_C_assimilazione_residua.md), prima pubblicazione f9d8486e19f8e51aa0f45b45cd9de4e569776d60 |
| Riferimento free storico | S7, 2879c1b5270be30fb10375f0b300a50cd37d5413 |
| Config modello/ricetta | configs/experiments/s7/S7-C149-unet-on-s1.yaml a d20232ef04873e94708995d119bedc2f6eca2f50 |
| Metriche e split | docs/proposal.md; configs/splits/splits_interp_extrap.yaml; report 10 del Mese 2 |

Hydrogen possiede implementazione e contratto B. Cursor deve acquisire il suo contratto B0, con membership, normalizzatori, maschere, API dei loader e recipe manifest; non crearne una copia divergente.

A può:
- preparare subito codice e test sintetici;
- allenare la variante non appena B0/B1 hanno qualificato input e percorso fold-specifici;
- procedere in parallelo ai training B, senza aspettarne le metriche outer.

A **non** deve:
- duplicare i sei training baseline di B;
- modificare normalizzazione, split o loader comuni senza accordo con Hydrogen;
- usare un backbone S7 come warm start;
- cambiare la specifica dopo aver visto quale condizione outer è difficile.

Congelare la loss e le regole di confronto prima di leggere i risultati outer di B. Se vengono consultati prima di una modifica successiva, quella modifica sarà dichiarata esplorativa. Non serve trattenere artificiosamente la consegna B in attesa di A: serve rispettare il freeze della specifica.

## 3. Una sola loss, nessuna ricerca di pesi

Sia e_i = T_pred_norm,i − T_true_norm,i e J_i = 1[T_true_K,i < 1800 K].

Per ciascun frame definire:
- w_i = 1 + lambda J_i;
- L_frame = somma_i(w_i e_i²) / somma_i(w_i);
- L_batch = media dei L_frame.

**Variante unica: lambda=1, quindi peso 2 nei getti e 1 altrove.** Soglia 1800 K, già usata nelle metriche del Mese 2. La normalizzazione per somma dei pesi impedisce che la sola crescita della regione faccia crescere banalmente la scala della loss. Non rende la variante equivalente a un semplice cambio di learning rate: cambia il peso relativo degli errori.

**Controllo B: lambda=0**, cioè la loss MSE full-field originale. Quando lambda=0, il codice deve richiamare direttamente la loss storica per preservare il percorso di riferimento.

La scelta del fattore 2 è una convenzione moderata fissata prima dei risultati, non un optimum fisico. Non provare 3, 5 o 10 se la variante non migliora. Non sostituire la maschera con T predetta, gradiente, distanza dal bordo o percentile di ogni frame. Nessuna loss su derivate, fronti o ancore, nessun secondo termine aggiunto.

Costruire J dalla T vera in kelvin, usando gli array fisici o l'inversa esatta del normalizzatore del fold. Una soglia in unità normalizzate, se usata, deve essere derivata dalle statistiche corrette e dimostrata equivalente. Non interpretare 1800 come un valore normalizzato e non riusare una soglia convertita con statistiche di un altro fold.

Calcolare la loss per frame, non normalizzare tutti i pixel del batch insieme: altrimenti cambierebbe il peso relativo delle condizioni in funzione della loro area fredda.

Conservare anche la MSE ordinaria come diagnostico TRAIN, oltre alla loss pesata, alla frazione di pixel jet e ai contributi pesati delle due regioni. Non confrontare numericamente le due loss come se fossero lo stesso obiettivo.

## 4. Matrice minima e confronto appaiato

| Blocco | Condizione esclusa | Modello | Seed | Nuovi training A |
|---|---|---|---|---:|
| A-lean | 40NH3_0.85.h5 | U-Net masked b32 p_on, lambda=1 | 0, 1, 2 | 3 |
| A-rich | 80NH3_1.2.h5 | Stessa variante | 0, 1, 2 | 3 |

**Totale A: sei training nuovi.** I sei riferimenti non pesati sono già la matrice di B; non risottometterli.

p_on significa phi e percentuale NH3 forniti come input noti, oltre a velocità 2C sparse al 5% e coordinate. Non si aggiungono T osservate all'input. Nessun p_off, GLU o variante sul pool storico a 20 condizioni richiesta da questa roadmap.

Per ogni coppia A/B usare:
- stesso insieme di 19 condizioni e frame C149;
- stesso normalizzatore fold-specifico;
- stesso hash dei pesi iniziali a seed fissato;
- identici ordine di presentazione, maschere TRAIN e batch, fino al rispettivo punto di arresto;
- stessa architettura, optimizer e regole di selezione.

La traiettoria successiva di pesi e LR può cambiare proprio a causa della loss; non forzarla a seguire il baseline. Stesso seed da solo non dimostra un confronto appaiato: verificare init e stream reali.

## 5. Dati, selezione e budget

Applicare il contratto B senza estenderlo:
- fit: 19 ×149 = 2831 frame;
- norm_fit e train_reference: 19 ×38 = 722 frame;
- seen_retained: 19 ×11 = 209 frame;
- selection_val: VAL123 originale;
- outer_fold: 41 frame della condizione esclusa, tutte le sue traiettorie escluse da fit e normalizzazione;
- TEST originali chiusi.

La U-Net di A parte da zero. Il normalizzatore storico su 760 frame non è valido per i fold: contiene la condizione esclusa. A usa gli artefatti B, non li rifitta autonomamente.

Ricetta uguale a B: 760 presentazioni/95 update per epoca scientifica, batch 8, AdamW lr1e-3, wd1e-4, clip1, warmup950; stesso plateau scheduler, precisione fp32 e maschere. Nessuna modifica AMP/TF32/compile per accelerare soltanto uno dei due metodi.

**Best primario:** primo minimo assoluto della macro RMSE full-field ordinaria su selection_val, al 5%, media delle tre repliche. Non selezionare sul jet RMSE o sulla loss regionale. Così A risponde a “serve dare più peso ai getti mantenendo la selezione globale?”, senza cambiare contemporaneamente il criterio di checkpoint.

**Secondario:** last. Se una singola metrica ai getti migliora soltanto in un checkpoint selezionato ex post, è un'osservazione esplorativa e non sostituisce il primario. Nessun best_outer o best_jet scientifico.

Budget e stop come B:
- eval ogni 10 epoche;
- plateau minimo 200 epoche, 15 eval senza miglioramento relativo .001, reset al calo LR;
- tranche1000;
- una sola estensione fino a1200, soltanto se inclusa nel mandato di esecuzione e con stato invariato;
- mancato plateau a1200 dichiarato come budget esaurito, non occultato.

Riportare update, presentazioni e stop. Stesse regole e budget massimo non significano necessariamente stesso tempo o stesso checkpoint: affiancare best/last e curve ai punti di update comuni già disponibili, senza richiedere nuovi checkpoint intermedi o training.

## 6. Implementazione circoscritta per Cursor

Output nuovi proposti:
- configs/experiments/jet_loss_a_protocol_v1.yaml;
- configs/experiments/jet_loss_a_matrix_v1.json;
- una funzione di loss testata sotto src/combustion_recon/;
- docs/preparatory/a_jet_loss_handoff.md;
- docs/preparatory/a_jet_loss_report.md;
- results/preparatory/a_jet_loss/INDEX.md.

Lavorare in worktree dedicato, indicativamente preparatory/a-jet-loss. Introdurre un flag esplicito per la loss, con default storico invariato. Non copiare l'intero trainer per cambiare poche righe. Se Hydrogen deve aggiungere supporto fold nello stesso file, concordare prima una base/API comune e integrare i commit necessari, senza sovrascriversi o fare merge indiscriminati dei branch.

Il checkpoint deve registrare identificativo e parametri della loss. Il resume deve rifiutare un tentativo di passare da lambda0 a lambda1 o da un fold all'altro sotto lo stesso run ID.

Nessuna modifica a modelli, dati storici, report supervisors, codice C o roadmap comuni durante l'esecuzione.

## 7. Test e smoke prima della matrice

### Test della loss

Su tensori piccoli con risultato calcolabile:
1. lambda0 richiama la MSE originale e ne preserva valore e gradiente.
2. Peso2 dentro J, peso1 fuori, normalizzazione per frame corretta.
3. Nessun pixel jet e tutti i pixel jet restituiscono la MSE del frame.
4. La soglia è stretta: T=1800 K non appartiene a J.
5. Rimescolare i frame non cambia la media della loss; duplicare un frame pesa come previsto.
6. Nessun NaN con J vuoto e gradienti finiti.
7. Ricostruire J con ciascun normalizzatore B dà la stessa maschera di T fisica.
8. Il forward del modello non riceve J o target; cambiare target a input fisso modifica la loss, non la predizione.

### Verifiche del percorso reale

- Acquisire manifest e conteggi B; nessuna condizione esclusa in statistiche o stream.
- Verificare init e primi batch A/B a seed fissato.
- Controllare che la loss non consumi RNG e non alteri il sampler.
- Uno smoke reale GPU per fold, separato dai risultati scientifici: training breve, eval standard, checkpoint/resume e readback.
- Verificare il default lambda0 contro il trainer baseline a pari input. Usare i riferimenti tecnici B senza ripetere tutta la sua campagna di smoke.
- Provare isolamento da velocità nascoste e assenza di target nell'input.
- Verificare che scheduler, stop e best usino la macro VAL ordinaria e non la nuova loss.
- Profilare overhead di loss, RAM/VRAM e tempi; correggere solo errori tecnici, non cercare un peso “che funzioni” nello smoke.
- Checkpoint completo, optimizer/scheduler/RNG/storia e sampler conservati al resume.

**Uscita A1:** codice/config testati e pushati, input B verificati, matrice sei run pronta, checkout remoto congelato. Se l'incarico autorizza già la matrice, procedere dopo il gate tecnico senza un'altra richiesta puramente rituale.

## 8. Training e valutazione

### A2 — Training

Solo i sei run autorizzati, output nuovi non sovrascrivibili, senza metriche outer nel loop. Una sola submission attiva per run. Preemption e continuazioni non sono nuovi seed. Nessun rilancio da best o warm start silenzioso.

### A3 — Freeze ed eval

Congelare i sei best/last A e acquisire quelli B verificati. I risultati B possono essere disponibili prima; non usarli per cambiare lambda, soglia, modello o selezione.

Valutare tutte le coppie sulla stessa membership e sui medesimi frame/maschere di B:
- outer_fold primario per la domanda di generalizzazione;
- seen_retained e selection_val separati;
- train_reference soltanto diagnostico;
- velocità5%, K0, repliche0/1/2.

Replay su selection_val contro il loop, riferimenti non vuoti e macro entro0.05 K; riportare scarti per-frame. Per outer nessun replay storico inventato: verificare chiavi, cardinalità, maschere e ricalcolo diretto delle metriche. COMPLETE soltanto dopo controlli e readback, altrimenti mantenere la diagnostica del fallimento.

Non rieseguire i forward baseline B se i CSV e i campi necessari sono disponibili e compatibili: riusarli dopo verifica. Se manca una metrica, aggiungerla in modo concordato al percorso comune, documentando la nuova inferenza eventualmente necessaria; non alterare numeri già pubblicati.

## 9. Come misurare il beneficio e il compromesso

Usare le metriche B e pubblicare per ciascun seed, replica, condizione e checkpoint:
- RMSE globale, MAE, bias firmato e |bias|;
- centered-RMSE;
- RMSE jet, hot e peak, con conteggi e regioni vuote;
- hot IoU ed errore di area;
- p95 della RMSE sui frame, calcolato dai dati per-frame;
- medie e distribuzione della T vera e predetta;
- tempo GPU e overhead della variante.

Aggiungere **RMSE nei getti dopo sottrazione del bias globale del frame**, sullo stesso insieme J definito dal target. Non sottrarre il bias calcolato solo dentro il getto: nasconderebbe un errore termico regionale diverso. Chiamare la metrica in modo non ambiguo, per esempio jet_rmse_after_global_debias_K.

Calcolare A−B appaiato a seed/frame/maschera:
- segno negativo sugli errori = miglioramento;
- segno negativo su IoU = peggioramento, quindi non riusare etichette generiche di “gain” per tutte le metriche.

Mostrare singoli seed, media/min/max e deviazione standard descrittiva, senza test di significatività basati su pixel/frame correlati. Le tre repliche delle maschere non sono tre training.

Una tabella deve accostare, per ogni fold:
1. Δjet RMSE;
2. Δjet RMSE dopo debias globale;
3. Δglobal e Δcentered;
4. Δhot RMSE e Δhot IoU;
5. dispersione fra seed e risultato peggiore.

Non dichiarare “senza degradare il resto” perché la media globale cambia poco: riportare direttamente grandezze e condizioni del compromesso. Nessuna soglia in kelvin da scegliere dopo aver visto i risultati. Con due fold mirati si descrive evidenza limitata, non una garanzia di generalità.

### Figure

- curve TRAIN MSE ordinaria e loss pesata, con VAL ordinaria, assi e unità distinti;
- differenze A−B per seed e fold, globali e regionali;
- confronto bias/centered per distinguere livello da struttura;
- frame1000 rep0 di ciascun outer, riferimento vero, B, A ed errori con scale comuni;
- peggior frame dopo eval, dichiarato come selezionato per diagnosi e non per tuning.

B può fornire i campi del suo frame1000: non rigenerarli solo per motivi di percorso o nome. Salvare quelli A e le metriche per-frame; non è necessario salvare tutte le mappe.

## 10. Lettura finale e stop

La relazione deve rispondere:
- la loss migliora davvero i getti sulle due condizioni escluse o soltanto nel fit?
- il miglioramento rimane dopo aver rimosso il bias globale?
- migliora tutti i seed o è dominato da uno solo?
- quanto costa su regione calda, livello e campo complessivo?
- best e last raccontano la stessa storia?
- il risultato giustifica un seguito applicativo o conviene mantenere la loss ordinaria?

**STOP A:** una sola variante e i sei confronti matched conclusi nel budget, oppure blocco tecnico esplicito; report e pacchetto pubblicati, risultati negativi inclusi. Non si cerca un secondo peso o una nuova regione per recuperare una vittoria.

Se il beneficio resta limitato a un fold o viene pagato con errori caldi importanti, chiudere documentando il compromesso. Se i getti migliorano coerentemente, proporre una verifica ulteriore di trasferimento, senza avviarla. A non dimostra automaticamente utilità su motori, sensori reali o altro dataset.

## 11. Risorse e lavoro autonomo sul cluster

Il costo atteso per run è dell'ordine della U-Net free B/S7 (storicamente 3.7–4.2 h); la loss dovrebbe aggiungere poco calcolo, ma il nuovo percorso va profilato. Non usare le stime della campagna assisted e non promettere tempi di coda.

Limiti e quote sono condivisi con B e C, assegnati dalla chat amministratrice:
- normal: mit_normal_gpu, account/QoS mit_amf_advanced_gpu, L40S;
- preemptable: mit_preemptable, mit_general e QoS ammesso, L40S, dopo resume verificato;
- CPU per preparazione/aggregazione con account e QoS ammessi.

Verificare i limiti correnti: quattro normal e quattro preemptable menzionati da Enrico non significano otto slot per ogni agente. Nessuna modifica o cancellazione dei job di Hydrogen/Sonnet.

Quando l'incarico autorizza l'esecuzione, predisporre dipendenze cluster training → verifica/freeze → eval → aggregazione, output non sovrascrivibili e controllo degli artefatti. Le dipendenze Slurm da sole non validano il risultato. Se il mandato richiede stop dopo submission, riportare ID, risorse e percorsi e fermarsi senza polling locale.

Un errore reale produce un report di blocco e checkpoint conservati. Nessun recupero indefinito, duplicazione di run o modifica scientifica mascherata da fix operativo.

## 12. Confini

Fuori da A: T in input o correzioni mediane/RBF, GLU, p_off, multidensità, nuove condizioni/dataset, TEST, tuning di soglia/peso, integrazione generale dei branch e modifiche al report inviato.

Il contratto comune è di B; il correttore è di C; la loss è di A. La chat amministratrice integra i risultati e aggiorna diario/roadmap comuni dopo le consegne. Nessuno dei tre agenti può dichiarare da solo approvata la direzione definitiva del Mese 3.
