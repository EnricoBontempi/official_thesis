# Specialisti — criteri di smoke non superati (GLU) e politica `budget_exhausted`

**Data:** 24 novembre 2026 · **Contesto:** training specialistici già sottomessi (codice `b6fc9fe`, invariato). Questo documento non modifica soglie né training.

## 1. I due criteri GLU non superati dallo script di smoke
Lo script di smoke (`scripts/65_nonideal_smoke_checks.py`) segnala `all_pass: false` per il GLU a causa di due criteri distinti. Si riportano separatamente evidenze e limiti. (Un terzo segnale, l'identità della U-Net, era falso per una soglia fissa di 10⁻⁶ scritta prima di misurare; contro il rumore GPU misurato la U-Net passa: sotto.)

### 1A. Identità: specialista senza degradazione contro il training GLU originale
- **Cosa confronta.** Stesso seed (0), stessa configurazione, 4 epoche piccole: percorso originale (batch 8, un forward) contro percorso specialistico con regime identità (microbatch 1, accumulo a batch effettivo 8). L'unica differenza di ricetta voluta è il microbatch.
- **Evidenza (GPU, smoke).**

| confronto | pesi (max abs) | loss per update (max abs) | VAL macro, 6 frame (K) |
|---|---|---|---|
| originale contro identità specialista | 7.0·10⁻⁵ | 2.5·10⁻⁴ | 0.077 |
| originale (replica) contro identità | 7.3·10⁻⁵ | 2.6·10⁻⁴ | 0.081 |
| rumore: due run originali identici | 1.1·10⁻⁵ | 1.0·10⁻⁵ | 0.0035 |
| rumore: due run specialisti N1G1 identici | 1.0·10⁻⁵ | 9.1·10⁻⁵ | 0.0049 |

  La differenza di identità supera il rumore fra run identici di circa 7× sui pesi, 25× sulla loss e 20× sulla VAL: **non è spiegata dal solo rumore GPU**. Per la U-Net identità e rumore sono dello stesso ordine (pesi 5.9·10⁻⁵ contro 5.6·10⁻⁵; loss 3·10⁻⁶ contro 3·10⁻⁶): passa, giudicata contro il rumore misurato.
- **Evidenze a favore dell'interpretazione "numerica, non di ricetta".**
  1. Test CPU (modello piccolo, batch omogeneo): loss, norma del gradiente, gradienti per parametro e aggiornamento coincidono fra microbatch 1 e batch 8 (`tests/test_nonideal_specialist.py`); la loss pesa ogni campione 1/8 anche con numero di token diverso.
  2. Elementi del modello che dipendano dal batch: nessuno (normalizzazioni per token; diagnostico S7 senza dipendenza dagli altri esempi).
  3. Sensibilità nota alla forma del batch: in inferenza, batch 1 contro batch 8 differisce fino a 0.094 K per frame sui modelli allenati (macro ≤ 0.003 K).
- **Limiti (non superati da queste evidenze).**
  - **I controlli CPU non provano l'equivalenza della traiettoria GPU.** Provano che la matematica del gradiente accumulato è corretta. Non provano che, su GPU, kernel con B=1 e B=8, ordini di somma e amplificazione lungo centinaia di epoche conducano a pesi e metriche statisticamente indistinguibili.
  - Lo smoke è di 24 update su un modello quasi non allenato: la grandezza della deriva su traiettorie di 300–1000 epoche **non è nota**.
  - Non esiste, e non è autorizzato, un controllo "GLU pulito con microbatch 1" da confrontare con il GLU originale. Quindi, per i GLU, il confronto specialista contro originale **mescola** (i) l'effetto del regime, (ii) la differenza numerica di batching e (iii) la variabilità fra traiettorie. Non si possono separare con i dati disponibili.
  - Come unità di giudizio ragionevole, ma non come soglia: la dispersione fra seed dei modelli originali (Blocco II) è di circa 1–4 K sull'RMSE macro; guadagni o costi specialistici del GLU molto più piccoli di quella dispersione non vanno interpretati. Per la U-Net il confronto è più pulito.
- **Se servisse chiuderlo:** proposta (non lanciata, richiede autorizzazione) di 3 training GLU puliti con microbatch 1 (un seed ciascuno) per misurare la differenza di sola ricetta numerica.

### 1B. Resume: ripresa reale contro run continuo
- **Cosa fallisce.** `resume_ok` è falso per il GLU perché `scheduler_state_equal` è falso: il campo float `best` dello scheduler segue la metrica VAL, che porta rumore GPU. I contatori interi dello scheduler coincidono.
- **Evidenza.** Stato (epoca, update, campioni, sampler, learning rate, contatori di early stop), stream di presentazioni e ledger (fresh poi resume) identici al run continuo. Pesi: 1·10⁻⁵–2·10⁻⁵. Metrica VAL a 6 frame dopo 4 epoche: ripreso contro continuo 1.1·10⁻²–3.3·10⁻² K; due riprese indipendenti fra loro 2.2·10⁻² K; due run continui identici 4.9·10⁻³ K. Lo stesso `scheduler_state_equal: false` compare nello smoke S7 storico (tutte e tre le confronti, U-Net e GLU), accettato allora.
- **Limiti.**
  - La ripresa non è bitwise: dopo una preemption, la traiettoria di un run devia di quantità dell'ordine del rumore GPU. Su un modello quasi non allenato questo valeva 0.01–0.03 K sulla metrica VAL.
  - Early stop e scheduler usano un miglioramento relativo di 10⁻³ su un valore vicino a ~40 K (cioè ~0.04 K): il rumore dopo un resume **potrebbe** spostare di un'epoca una decisione di plateau; con modelli allenati l'ampiezza del rumore non è stata misurata.
  - Le riprese reali sono tracciate nel ledger e nel `run_identity`; il freeze registra tentativi, estensioni e segnali per ogni run.

## 2. Politica `budget_exhausted` (a 1200 epoche)
- **Definizione.** Un run che raggiunge 1000 epoche senza il plateau dell'early stop riceve l'unica continuazione fino a 1200 (stesso stato e regole). Se a 1200 il plateau non c'è: `budget_exhausted`, nessuna ulteriore estensione automatica.
- **Come entra nel pacchetto.** Se gli artefatti sono validi, **è incluso** e **non è dichiarato convergente**: etichetta esplicita `budget_exhausted_not_declared_converged` in `freeze.json`, `tables/training_diagnostics.csv`, `tables/per_run_scenario.csv`, `package/fields_index.json`, `COMPLETE.json`, `INDEX.md`, `FINAL_SUMMARY.md` e nelle figure (marcatori vuoti; conteggi nei titoli delle matrici). Anche il "plateau" è una convenzione operativa, non una certificazione fisica di convergenza: l'etichetta `plateau_by_operational_early_stop_rule` lo dice.
- **Validità richiesta (tutte):** stato terminale dichiarato; per `budget_exhausted` anche continuazione registrata nel ledger (`extension_start`) e ultima epoca 1200; identità (run, famiglia, p_on, regime, SHA di training, hash della specifica); checkpoint best/last completi, finiti, riletti; storia VAL pulita e metrica specialistica presenti e mai usate per decidere; ultima epoca di valutazione; init uguale fra i cinque regimi; gate R0 passato per best e last (macro entro 0.05 K dalla metrica del loop); conteggi, chiavi uniche e hash di input coerenti.
- **Cosa blocca (nessun pacchetto completo, dati conservati, `BLOCK_REPORT`):** errore tecnico (`stopped_error`, valori non finiti), stop per walltime non concluso, 1000 epoche senza continuazione eseguita, 1200 epoche senza continuazione registrata, checkpoint non valido, gate fallito, matrice incompleta, chiavi duplicate, input non confrontabili. Le soglie (plateau, 0.05 K) non sono cambiate.

## 3. Versione tracciata di verifica e aggregazione
Il checkout dei training (`b6fc9fe`, `/home/bontempi/projects/official_thesis_nonideal_spec_run`) resta intatto e continua a eseguire training ed estensione. Verifica, freeze, valutazione e aggregazione ora girano da un checkout separato e tracciato (SHA nel report di consegna); lo stage B è stato sostituito (stessa dipendenza sui due array di training) per distinguere `NI_TRAIN_TIP`/`NI_TRAIN_SHA` (training, estensione) da `NI_TIP`/`EXPECT_SHA` (post-processing). `freeze.json` registra entrambi gli SHA.
