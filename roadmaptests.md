# Roadmap — Robustezza a osservazioni di velocità non ideali

**Progetto:** *Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion* — Enrico Bontempi, MIT Deng Lab / Politecnico di Milano
**Preparata:** 24 novembre 2026, America/New_York
**Branch:** `plan/nonideal-observations` (da `plan/abc-preparatory-studies` @ `9d2c3af`); aggiornato dal Blocco I su `nonideal/block1-degradation-spec` (da `8d72a81`)
**Nota di ricognizione:** [docs/nonideal/00_ricognizione_osservazioni.md](docs/nonideal/00_ricognizione_osservazioni.md)

---

## 0. Riepilogo operativo (prevale sulle indicazioni operative precedenti)

1. **Priorità corrente:** definire il problema applicativo e misurare quanto i modelli già allenati reggono osservazioni di velocità meno idealizzate, motivate da misure reali. Poi, solo se giustificato, riaddestrare un insieme circoscritto di modelli.
2. **Roadmap A/B/C sospese** (loss nei getti, fold di generalizzazione, assimilazione residua): non eseguire. Restano nel repository con il piano storico integrale e una nota di sospensione datata 24 novembre 2026. Riattivazione solo dopo riesame e decisione esplicita di Enrico. Il vecchio incarico amministrativo che ne prevedeva l'esecuzione è superato.
3. **Questa roadmap non autorizza esecuzioni.** Ogni blocco parte con un incarico separato e delimitato. Nessun job è stato lanciato preparandola.
4. **Non cambia retroattivamente** proposal, report del Mese 2 (`documents_thesis` @ `a1dfb39`, inviato il 22 novembre) né i risultati congelati.
5. **Scelta motori/combustore non presa.** Questa roadmap lavora sul dataset attuale e non presenta risolto il problema della validazione sperimentale.

6. **Aggiornamento del Blocco I (24 novembre 2026, dopo il confronto con Linzheng).** Specifica scritta in [configs/nonideal/degradation_spec_v1.yaml](configs/nonideal/degradation_spec_v1.yaml), operatori in `src/combustion_recon/data/nonideal_observations.py`, test CPU, qualificazione di G1 su geometria e maschere TRAIN, handoff in [docs/nonideal/01_block1_handoff.md](docs/nonideal/01_block1_handoff.md). Correzioni concordate che **prevalgono** sul testo sottostante:
   - i valori σ = 0.5/1.5 m/s e la geometria di G1 sono **default di scenario**; la letteratura sostiene il fenomeno e la conversione, non questi valori per il nostro combustore;
   - **nessun cancello automatico di retraining** (cadono «2 K / 5 %, 3/3 seed, conferma su seen» di §8 e D6): dopo il Blocco II si riportano grandezze, dispersione, regioni e singole condizioni e Enrico decide esplicitamente; il Blocco III è previsto ma non autorizzato;
   - **0.04 K (GLU) era una differenza osservata**, non un limite: il replay ideale si fa con lo stesso batching della cella e ogni discrepanza si diagnostica;
   - il troncamento al minimo numero di siti del batch (§10, III.0) **non è una scelta definitiva**: si decide nel Blocco III;
   - l'asse S (media a finestra), il campionamento regolare, le lacune dipendenti da T, gli spuri come errori di valore e le temperature osservate **restano fuori** dal primo blocco e non si aggiungono automaticamente.

7. **Ampliamento del Blocco II a 12 modelli (autorizzato da Enrico il 24 novembre 2026).** U-Net/GLU × p_on/p_off × seed 0/1/2, tutti `best` della matrice S7 finale (nessun `last`/`best_seen`, nessuna scelta del seed). Specifica originale preservata; revisione distinta [configs/nonideal/degradation_spec_v1_12models.yaml](configs/nonideal/degradation_spec_v1_12models.yaml) con operatori, livelli, geometrie e generatori invariati (test). Valutazioni nominali: 12 × 6 × 3 × 343 = 74 088 (replay tecnici aggiuntivi contati a parte). Domanda aggiuntiva: conoscere phi e NH3 (p_on) attenua il peggioramento? p non viene mai degradato; nessuna T in input. Batching uniforme per famiglia (U-Net 8, GLU 1 anche per R0); gate R0 entro 0.05 K dal riferimento S7 per ruolo/replica; se fallisce la campagna si ferma e si diagnostica. Runbook: [docs/nonideal/02_block2_runbook.md](docs/nonideal/02_block2_runbook.md).

8. **Blocco III autorizzato da Enrico il 24 novembre 2026 con matrice esplicita di 30 training specialistici.** 2 famiglie (U-Net masked, GLU adattato) × solo p_on × seed 0/1/2 × 5 regimi (N1, N2, G1, G1-eq, N1G1), da zero, un solo regime per modello, nessun campione pulito, nessun training misto. **La precedente limitazione a «al massimo 6 training» (§10, III.0) è superata esplicitamente da questa decisione, non reinterpretata.** Selezione invariata (primo minimo assoluto della macro RMSE VAL pulita al 5 %); la metrica VAL dello scenario specialistico è solo registrata. GLU: microbatch 1 con accumulo a batch effettivo 8 (nessun troncamento al minimo del batch). Valutazione di best e last su tutti i sei scenari contro le baseline pulite. Contratto: [configs/nonideal/specialist_protocol_v1.yaml](configs/nonideal/specialist_protocol_v1.yaml); matrice: `configs/experiments/nonideal_specialist_matrix.json`; baseline: `configs/nonideal/specialist_baselines_v1.json`; handoff: [docs/nonideal/03_specialist_handoff.md](docs/nonideal/03_specialist_handoff.md).

## 1. Obiettivo e legame con la tesi

Domanda della tesi: quanto si può ricostruire la temperatura della singola istantanea da osservazioni sparse di velocità, e come cambia il risultato quando osservazioni e condizioni diventano meno favorevoli.

Finora le osservazioni erano ideali: valori esatti della slice LES, in siti casuali al 5% sull'intero dominio. Questa roadmap rende esplicito il **modello dell'osservazione** e ne misura le conseguenze in tre blocchi:

- **I. Audit e modello dell'osservazione.** Che cosa contiene il dato e quali degradazioni sono motivate da misure, con quale grado di supporto.
- **II. Robustezza dei modelli congelati.** Stessi checkpoint, stessi frame, input degradati, T numerica originale come target.
- **III. Retraining con osservazioni degradate.** Fase condizionata: previsto ma non autorizzato: dopo il Blocco II Enrico decide esplicitamente se procedere (nessuna soglia automatica, §8).

Il Blocco II **misura** il problema; non modifica i modelli per farli funzionare meglio.

## 2. Stato e limiti dell'autorizzazione

| Attività | Stato |
|---|---|
| Ricognizione, roadmap, nota di ricognizione, note di sospensione A/B/C | Eseguite con l'incarico del 24 novembre 2026 |
| Blocco I (step I.1–I.5) | Eseguito il 24 novembre 2026 con incarico separato: ricognizione documentale completata nei limiti registrati, specifica scritta, operatori e test CPU, qualificazione G1 su TRAIN. Nessun calcolo GPU, nessun download EngineBench. STOP I: revisione di Enrico |
| Blocco II | Eseguito il 24 novembre 2026 su 12 modelli (gate R0 12/12, 74 088 valutazioni): [docs/nonideal/02_block2_robustness.md](docs/nonideal/02_block2_robustness.md). STOP II: decisione di Enrico sul Blocco III |
| Blocco III | Autorizzato il 24 novembre 2026 nella forma dei 30 training specialistici (voce 8 del riepilogo); nessun'altra configurazione; **eseguito**: 30/30 training validi (plateau), 60/60 valutazioni e gate, aggregazione e grafici completati il 27 novembre 2026. Sintesi verificata: [docs/nonideal/05_specialist_results.md](docs/nonideal/05_specialist_results.md); risultati: [results/nonideal/specialists/](results/nonideal/specialists/). Nessun recupero consistente su VAL, costo sul pulito positivo; pannelli spaziali appaiati con le baseline non disponibili (frame diversi, inferenza non autorizzata) |
| TEST | Chiuso. La storia dei TEST esaminati nel Mese 1 va preservata e non va estesa |
| Termocoppie all'uscita | Tema separato, fuori da questa roadmap |
| `documents_thesis`, HONF, presentazioni, contatti con persone (incl. Franzelli) | Fuori perimetro |

Contesto riferito da Enrico (non verificato qui): colloquio del 24 novembre con il Prof. D'Errico (ICE, PoliMi). Dispone soprattutto di RANS motoristiche; disponibilità di LES e condivisibilità dei dati da verificare; non si occupa direttamente di PIV in combustori ottici; ha suggerito dati pubblici (EngineBench, archivi Sandia) come riferimento. Linzheng e Jason sono stati aggiornati; risposta attesa.

## 3. Fonti

### 3.1 Repository e versioni congelate

| Fonte | Riferimento |
|---|---|
| S7 free (confronto U-Net/GLU) | `s7/c149-free-campaign` @ `2879c1b5`; `docs/month2/10_s7_c149_model_comparison.md`; `results/month2/s7_c149_comparison/package/INDEX.md` |
| Sweep K | `sweep/c149-temperature-k` @ `be18ad00`; `docs/month2/11_temperature_k_sweep.md` (non presente su questo branch; leggerlo dal commit) |
| Assisted e pacchetto congiunto | `temperature/assisted-c149` @ `d20232ef`; `results/month2/joint_step23/package/INDEX.md`; `docs/month2/12_temperature_input_study.md` |
| Sintesi delle revisioni | `reviewmese2.md` @ `9a6277f2` |
| Roadmap sospese | `plan/abc-preparatory-studies` @ `9d2c3af4`: `roadmap_A_loss_getti.md`, `roadmap_B_generalizzazione.md`, `roadmap_C_assimilazione_residua.md` |
| Report supervisori | `documents_thesis` main @ `a1dfb391` (nessun commit successivo al 24 novembre); Mese 1 in `archive/` |

**Erratum noto:** `configs/experiments/month2_temperature_study_protocol.yaml` non è parsabile (riga 72, documentato in `reviewmese2.md` §10). Non va corretto silenziosamente né riusato come configurazione eseguibile. Le configurazioni di questa roadmap sono nuove, con test di parsing.

### 3.2 Fonti esterne (dettaglio e grado di verifica nella nota di ricognizione)

- RealPDEBench: arXiv:2601.01829; https://realpdebench.github.io/datasets/combustion/; dataset card HF.
- EngineBench: arXiv:2406.03325; https://github.com/sambkr/EngineBench (loader e generatori di lacune letti); pagina Oxford non raggiungibile il 24 novembre.
- TNF: https://tnfworkshop.org/data-archives/ (pagine swirl, piloted, bluff-body; Sydney swirl).
- Incertezza PIV: Sciacchitano et al. 2015 (MST 26 074004); Wieneke 2015 (MST 26 074002); Sciacchitano 2019 (MST 30 092001).

### 3.3 Che cosa risulta oggi (sintesi della ricognizione)

- La velocità in input è la **componente planare di una slice LES interpolata a 128×128**, senza maschera di dominio; procedura di slicing e interpolazione non dichiarata nelle fonti lette. Non è una PIV.
- Il dataset non contiene PIV né temperatura misurata per la combustione: l'unica modalità reale è OH*.
- EngineBench: vettori mancanti codificati come zero, maschera derivata da `u==0|v==0`; **le lacune del benchmark sono artificiali** (poligoni di bordo ≤25%, maschere fisse di test). Errore medio dichiarato 1.5–8% (da Schiffmann et al.).
- TNF: nessun dataset letto con velocità e temperatura simultanee e co-localizzate a singolo colpo; nessuna PIV nei dataset swirl dell'archivio.
- Letteratura PIV: errore casuale tipico 0.03–0.1 px, fino a qualche decimo in condizioni reali; il moto fuori piano e il seeding disomogeneo sono fonti principali.
- Una media a finestra di 3–7 celle cambia la velocità di 0.6–1.6 m/s RMS sui frame TRAIN: la risoluzione PIV è un asse distinto e non trascurabile.

## 4. Ruoli di dati e checkpoint

| Oggetto | Ruolo in questa roadmap |
|---|---|
| C149 TRAIN (20 condizioni × 149 = 2980 frame) | Solo training del Blocco III; fonte delle statistiche di scala (Blocco I) |
| reference760 / normalizzatore fit-only | Congelato (`train_norm_stats_month2_fit.json`, `678263fb…`). Mai rifittato, né su VAL né su realizzazioni degradate |
| seen220 (11 frame successivi per condizione TRAIN) | Valutazione, riportata **separata** da VAL |
| VAL123 (3 condizioni × 41) | Valutazione descrittiva. Già usata ripetutamente per selezione e sviluppo: **non è conferma indipendente** |
| TEST | Chiuso |
| Maschere eval 5% storiche (seed 20261116, repliche 0/1/2) | Base dei siti per tutto il Blocco II. Repliche di maschera ≠ seed di training |
| Sei checkpoint `best` free p_on | `O-C149-on-s0`, `S7-C149-unet-on-s1`, `S7-C149-unet-on-s2`, `S7-C149-glu-on-s0`, `S7-C149-glu-on-s1`, `S7-C149-glu-on-s2` (hash e epoche nella nota di ricognizione §7) |
| Esclusi per default | p_off, assisted, last, best_seen. Nessuna scelta del seed migliore |

## 5. Ordine delle operazioni dell'osservazione

```
campo fisico denso (m/s, 128×128, dalla cache con statistiche congelate o dal grezzo)
  → [S] operatore spaziale opzionale (media a finestra sul campo denso)
  → campionamento ai siti (maschere 5% storiche)
  → [N] rumore sui valori, in m/s, per vettore
  → [G] invalidazione (validità = 0; nessun riempimento)
  → normalizzazione con le statistiche congelate
  → input del modello (U-Net: valore 0 e Mv=0 ai siti invalidati; GLU: token rimosso)
```

Motivazioni:
- **S prima del campionamento:** una PIV restituisce lo spostamento medio in una finestra; l'operatore agisce quindi sul campo fisico, non sui valori già campionati. Usa il campo vero attorno al sito solo come modello fisico della misura, mai per riempire siti mancanti.
- **N in m/s, prima della normalizzazione:** l'incertezza è una proprietà della misura; la stessa σ fisica corrisponde a 0.088 (vi) e 0.042 (vj) normalizzati per σ=0.5 m/s. Mai mescolare le due unità.
- **N generato per tutti i siti prima di G:** con generatori indipendenti, i siti che sopravvivono hanno lo stesso rumore in N e in N+G. Questo rende appaiati i confronti tra celle.
- **G dopo N:** un vettore invalidato non è visto, quindi l'ordine non cambia i valori; conta solo per l'accoppiamento delle realizzazioni.
- **Spuri:** dopo la validazione PIV i vettori spuri diventano in gran parte mancanti; vanno modellati come invalidazione, non come grandi errori di valore, salvo decisione contraria (D3).

Generatori: seed di degradazione in un namespace nuovo, distinto dai seed di training e dai seed delle maschere storiche, derivato da `(degradation_seed, operator_id, livello, sim_id, frame_index, replica_id)`. Stesse realizzazioni per tutti i sei modelli.

## 6. Blocco I — Audit e modello dell'osservazione

### I.1 Audit RealPDEBench del campo di velocità

- **Domanda:** che cosa rappresenta esattamente la velocità in input, e che cosa manca per confrontarla con una misura?
- **Dipendenze:** nessuna.
- **Attività:** leggere verbatim l'Appendice B (PDF arXiv:2601.01829) per piano, interpolazione, mesh e SGS; verificare segno e asse delle componenti (vj risulta allineata alle colonne; segno rispetto a +x da chiarire, per esempio con un controllo di trasporto del campo T su frame TRAIN consecutivi); leggere vk dal dato grezzo per pochi frame TRAIN e mappare |vk|/|u_planare|; annotare l'incoerenza della nota pilot del Mese 1 senza riscriverla.
- **Output:** sezione aggiornata della nota di ricognizione; tabella "verificato / mancante / richiede domain expert".
- **Verifiche:** ogni affermazione con fonte o script; nessun frame VAL/TEST letto.
- **Completamento:** tutte le voci della tabella 1.1 della nota hanno stato [P], [V] o "non dichiarato dalla fonte".
- **Stop/decisione:** nessuno; alimenta I.3.

### I.2 Modello fisico dell'osservazione PIV in questo combustore

- **Domanda:** quali fenomeni separano la slice numerica da una PIV planare nello stesso piano, e quali sono trasferibili come operatori?
- **Dipendenze:** I.1.
- **Attività:** per ogni fenomeno (media a finestra e spessore del foglio, campionamento regolare in un campo di vista, errore casuale e sua correlazione, vettori rigettati, seeding ridotto nel gas combusto, perdita di coppie per moto fuori piano, riflessi a pareti/finestre) compilare la scheda di §6.6.
- **Output:** schede compilate nella nota; elenco delle domande per un esperto di diagnostica.
- **Verifiche:** nessun valore numerico senza fonte o senza etichetta "ipotetico".
- **Completamento:** ogni fenomeno ha classe A/B/C e uno stato "incluso nel core / asse separato / escluso con motivazione".
- **Stop/decisione:** nessuno.

### I.3 Quantificazione dei parametri e scelta dei default

- **Domanda:** quali valori numerici usare, e su quale base?
- **Dipendenze:** I.1, I.2; I.4 se approvato.
- **Attività:** convertire l'incertezza PIV da px a m/s con formula dichiarata (σ_u ≈ σ_px · U_fs / d_max, U_fs da TRAIN); tabulare l'effetto della media a finestra in m/s (fatto in forma preliminare: 0.63/1.12/1.56 m/s RMS per finestre 3/5/7 celle); definire il generatore di lacune contigue indipendente da T e calcolarne, sui siti 5% storici, la frazione effettiva di siti rimossi e la distribuzione del numero di token per frame; verificare che il numero minimo di siti restanti sia ≥ 16 (top-k GLU) per ogni frame/replica.
- **Output:** tabella parametri con unità, fonte, classe, conversione.
- **Verifiche:** calcoli su TRAIN e su maschere (che non contengono T); nessuna metrica di modello.
- **Completamento:** i default di §9 sono confermati o sostituiti con motivazione scritta.
- **Stop/decisione:** nessuno.

### I.4 Campione EngineBench (opzionale)

- **Domanda:** nella PIV reale rilasciata, quante lacune *interne* al campo di vista esistono, e con che geometria?
- **Dipendenze:** decisione D2 (download e licenza).
- **Attività:** scaricare solo il subset LSP_small (o un singolo file) se accessibile e la licenza lo consente; contare vettori a zero interni alla porzione osservata, dimensioni dei cluster, variazione tra snapshot e fasi; verificare il codice dei loader anche per i file di test.
- **Output:** statistiche classe A *per EngineBench*; scheda di trasferimento classe B al combustore.
- **Verifiche:** separare i bordi/padding dalle lacune interne; non usare la varianza dei campi come stima del rumore.
- **Completamento:** statistiche riportate o rinuncia motivata.
- **Stop/decisione:** se non autorizzato o inaccessibile, G resta classe C.

### I.5 Congelamento della specifica di degradazione

- **Domanda:** quale matrice esatta valutare nel Blocco II?
- **Dipendenze:** I.1–I.3 (I.4 se eseguito).
- **Attività:** scrivere `configs/nonideal/degradation_spec_v1.yaml` (nuovo, parsabile) con operatori, livelli, unità, classi, namespace dei seed, ordine delle operazioni, ruoli, repliche, metriche, criteri di reporting del Blocco II (§8: nessuna soglia automatica). Test di schema e di parsing. Registrare lo SHA256.
- **Output:** specifica + hash + voce nel diario.
- **Verifiche:** test che rifiutano livelli in unità normalizzate, ruoli TEST, seed coincidenti con quelli storici.
- **Completamento:** approvazione esplicita di Enrico.
- **STOP I:** nessun passo del Blocco II prima dell'approvazione. Dopo l'approvazione, livelli e regole non cambiano in base agli esiti.

**Stato del Blocco I (24 novembre 2026).** I.1: Appendice B letta (B.1/B.2 verbatim; B.3/B.4 in sintesi) e orientamento riclassificato in nota di ricognizione §1.2/§8. I.3: conversioni e G1 definiti e qualificati su TRAIN. I.4: non eseguito (nessun download). I.5: specifica `degradation_spec_v1.yaml` scritta e testata; **approvazione di Enrico ancora da registrare**.

### I.6 Scheda per ogni degradazione

Ogni operatore della specifica riporta: fenomeno rappresentato; fonte; parametro e unità; informazione effettivamente disponibile; conversione al nostro dominio; ipotesi; **che cosa non permette di affermare**; classe:

- **A** supportata quantitativamente da misure pertinenti;
- **B** ispirata a misure, con trasferimento da giustificare;
- **C** stress test sintetico con intensità ipotetica.

Stato proposto oggi (da confermare in I.3/I.5):

| Operatore | Fenomeno | Fonte | Parametro | Classe | Non permette di affermare |
|---|---|---|---|---|---|
| N — rumore i.i.d. per componente | Errore casuale PIV | Letteratura metrologica 0.03–0.1 px tipico, decimi di px nel peggiore (supporta fenomeno e ordine di grandezza in px) | σ in m/s; σ_u = σ_px·U_fs/d_max | **B** (σ_px misurato in altri esperimenti; d_max, U_fs e quindi σ = 0.5/1.5 m/s sono ipotesi di scenario, non verificate sul nostro combustore) | Che sia l'errore di una PIV in questo bruciatore; correlazioni spaziali, bias e peak-locking non modellati |
| G — lacune contigue indipendenti da T | Regioni senza vettori (riflessi, ombre, seeding locale, campo di vista) | Fenomeni elencati da EngineBench e dalla letteratura; dimensioni e frazione non misurate per il combustore | unione di dischi: area 25 %, diametro 3–10 mm (famiglia scelta per semplicità, non per la fisica) | **C** (ordine di grandezza della frazione mutuato dal ≤25% di EngineBench, che è a sua volta artificiale) | Che la geometria corrisponda a un esperimento; nulla sulle lacune correlate alla fiamma |
| S — media a finestra | Risoluzione spaziale PIV / finestra di interrogazione | Spaziatura 1.25–1.4 mm in EngineBench [S]; finestra in fiamma non verificata | lato in mm, convertito in celle anisotrope | **B/C** | Che il campo LES sottostante abbia la stessa scala di filtro della misura; scala del filtro LES ignota |
| D — lacune correlate a T (seeding nel gas combusto, perdita di coppie fuori piano) | Fenomeni fisici reali | Ragionamento fisico (densità ∝ 1/T; vk forte) | probabilità di invalidazione in funzione di campi veri | **C**, e **rischio di leakage** (la maschera porta informazione su T) | Qualsiasi miglioramento attribuibile alla velocità; va confrontato con G a pari numero di siti |
| Campionamento regolare a passo PIV | Griglia densa in un campo di vista | Pratica PIV | passo in mm, campo di vista | **B** per il passo, **C** per il campo di vista | Separabile dal cambiamento di densità solo con controlli; fuori distribuzione per modelli allenati al 5% casuale |

## 7. Blocco II — Robustezza dei modelli congelati

### II.0 Precondizioni e identità

- **Domanda:** gli oggetti congelati sono disponibili e identici a quelli valutati in S7?
- **Dipendenze:** STOP I approvato.
- **Attività:** sul cluster, verificare file e `file_sha256` dei sei checkpoint, hash della cache C149, delle statistiche, del manifest delle maschere; recuperare dal pacchetto S7 (`2879c1b5`) i valori per-frame/per-condizione di riferimento (best, 5%, seen220, VAL123, repliche 0/1/2).
- **Output:** `results/nonideal/II0_identity.json`.
- **Verifiche:** ogni hash coincide con la matrice S7.
- **Completamento:** 6/6 checkpoint e 4/4 input verificati.
- **Stop:** una discrepanza ferma il blocco e si riporta a Enrico; nessun checkpoint sostitutivo.

### II.1 Implementazione degli operatori e test di isolamento

- **Domanda:** il percorso degradato è corretto e, con operatori identità, riproduce esattamente S7?
- **Dipendenze:** II.0.
- **Attività:** modulo nuovo (es. `src/combustion_recon/data/nonideal_observations.py`) che costruisce, dalla specifica congelata, pacchetti `CanonicalSparseObservation` degradati; nessuna modifica a evaluator, modelli o maschere storiche oltre a un punto di ingresso per il pacchetto di osservazione.
- **Test (CPU, obbligatori prima dello smoke):**
  1. identità: N=0, G vuoto, S assente ⇒ input bit-identici a quelli S7 per U-Net e GLU;
  2. unità: σ fornita in m/s, normalizzazione solo dopo; test che fallisce con σ normalizzata;
  3. determinismo e accoppiamento: stessa realizzazione per modelli diversi; rumore uguale sui siti superstiti tra N e N+G;
  4. separazione dei namespace: seed di degradazione ≠ seed di training e di maschera;
  5. validità: U-Net con Mv=0 e valore 0 ai siti invalidati; GLU senza token corrispondenti; nessun sito invalidato riempito;
  6. nessun uso di T nella generazione di N, G, S (test che l'output non cambia permutando T);
  7. casi limite: N_siti < 16 e N_siti = 0 producono errore esplicito (o politica dichiarata in specifica), mai un input silenziosamente vuoto;
  8. GLU con N diverso tra frame: valutazione a batch omogeneo (in pratica batch 1 per le celle con lacune); registrare la composizione dei batch (`glu_batching_policy`).
- **Output:** codice, test, commit.
- **Completamento:** tutti i test passano in CI locale.
- **Stop:** nessuno.

### II.2 Replay ideale e smoke reale

- **Domanda:** sul cluster, il percorso nuovo riproduce i riferimenti congelati e quanto costa?
- **Dipendenze:** II.1.
- **Attività:** un job GPU breve (partizione normal) su tutti i sei modelli: replay ideale completo di VAL123 e seen220 replica 0, più un sottoinsieme di frame per ogni cella della matrice; misurare tempo per frame e memoria.
- **Verifiche:** replay ideale entro 0.05 K macro rispetto a S7 per U-Net (gate S7 esistente; ogni scarto maggiore si diagnostica). Per GLU **non si fissa una soglia**: 0.04 K era una differenza osservata nel diagnostico S7, non un limite garantito. Il replay ideale usa il batching reale della cella (R0 a batch 8 per R0/N1/N2; R0 a batch 1 per G1/N1G1/G1-eq) ed è il riferimento appaiato delle differenze; una discrepanza di replay si diagnostica (forma del batch, pareggi top-k), senza modificare modello né tie-break dei vicini.
- **Output:** report di smoke con profiling e stima del costo dell'array.
- **Completamento:** replay riportato e discrepanze diagnosticate.
- **Stop:** discrepanza non spiegata ⇒ diagnosi, nessun array.

### II.3 Campagna di valutazione

- **Dipendenze:** II.2.
- **Attività:** array Slurm (un task per modello, o per modello × cella se il profiling lo suggerisce), catena con aggregazione dipendente; output per-frame su storage del cluster; resume a livello di task.
- **Verifiche:** completezza (righe attese = modelli × celle × repliche × frame), unicità delle chiavi, hash di specifica e checkpoint in ogni riga.
- **Output:** CSV per-frame, provenienza, log `sacct`.
- **Completamento:** 100% delle righe attese.

### II.4 Analisi

- **Attività:** per ogni modello, cella, ruolo, condizione e replica:
  - RMSE, MAE, bias, |bias|, centered RMSE in kelvin; getti (T<1800 K), zona calda (T≥2100 K), IoU zona calda;
  - per G: RMSE **dentro e fuori** dalla regione invalidata (definita dalla geometria della lacuna, non da T);
  - differenze **appaiate** rispetto al replay ideale (stesso modello, frame, replica); media, min, max sui seed; conteggio seed/condizioni che peggiorano;
  - G contro G-eq (pari numero di siti) per separare geometria e quantità di informazione;
  - peggiori frame e condizioni; casi di fallimento;
  - 3–4 figure di campi leggibili (stessi frame per tutti i modelli, scelti con regola fissata in specifica, non dopo aver visto gli errori).
- **Output:** `results/nonideal/block2/package/` con `INDEX.md`, tabelle, figure, manifest e hash.
- **Verifiche:** ricalcolo indipendente di almeno quattro macro dai CSV grezzi.
- **Completamento:** pacchetto validato.

### II.5 Report e decisione

- **Output:** `docs/nonideal/02_block2_robustness.md`, narrativo, senza hash nel testo (rimandati al pacchetto); risultati positivi e negativi; limiti di §16.
- **STOP II:** report di grandezze, dispersione, regioni e singole condizioni (§8); decisione esplicita di Enrico sul Blocco III.

## 8. Regole di decisione (riformulate)

**Non esiste alcuna soglia automatica** per attivare il Blocco III. Le soglie «2 K / 5 %», «3/3 seed» e «conferma su seen» della versione precedente sono rimosse e non vengono sostituite da altre.

Dopo il Blocco II si riportano, senza giudizio automatico: variazioni appaiate (K e %) rispetto al replay ideale con lo stesso batching, per modello, seed, cella, ruolo (seen220 e VAL123 separati), condizione e replica; dispersione fra seed e repliche; RMSE dentro e fuori dalla regione G1 e nelle regioni di getti e zona calda; peggiori frame e condizioni. Enrico decide esplicitamente se, per quali famiglie e con quali celle procedere al Blocco III, tenendo presente la classe delle degradazioni (B o C) e che VAL123 non è una conferma indipendente. Il Blocco III resta previsto ma **non autorizzato**.

## 9. Matrice del Blocco II (congelata nella specifica v1)

Base: dodici checkpoint `best` free (U-Net/GLU × p_on/p_off × seed 0/1/2; ampliamento approvato, originariamente sei p_on); siti 5% storici, repliche 0/1/2; ruoli seen220 e VAL123 (separati); T numerica originale come target. Specifica: [configs/nonideal/degradation_spec_v1.yaml](configs/nonideal/degradation_spec_v1.yaml). Tutti i valori sono **default di scenario**, non misure del nostro combustore.

| ID | Contenuto | Parametri | Classe | Ruolo |
|---|---|---|---|---|
| R0 | Input ideale | — | — | Riferimento appaiato (replay) |
| N1 | Rumore gaussiano i.i.d. per componente, in m/s | σ = 0.5 m/s (0.0881 vi, 0.0420 vj normalizzati) | B (fenomeno e conversione supportati; valore ipotetico) | Scenario core |
| G1 | Unione di dischi, indipendente da T | area 25 % del dominio, diametro U[3,10] mm, bordi ritagliati | C (stress test sintetico) | Core |
| N1G1 | N1 + G1 con le stesse realizzazioni | rumore uguale sui superstiti, stessa regione | B+C | Core |
| G1-eq | Rimozione casuale uniforme dello stesso numero di siti di G1 | per frame e replica | controllo | Contiguità vs informazione |
| N2 | Rumore severo | σ = 1.5 m/s (stessa realizzazione standardizzata di N1, ×3) | C | Stress |

Definizione completa di G1 (costruzione, area richiesta/effettiva, bordi, posizioni, seed, rifiuti) e qualificazione su TRAIN: spec e [handoff](docs/nonideal/01_block1_handoff.md). Dimensione: 6 celle × 12 modelli × 3 repliche × 343 frame = 74 088 valutazioni nominali (più 6 174 righe di replay tecnico GLU a batching S7). Nessun prodotto cartesiano.

Assi **separati, non nel core**: S (media a finestra), campionamento regolare a passo PIV, lacune dipendenti da T, spuri come errori di valore, temperature osservate. Non si aggiungono automaticamente; ciascuno richiede decisione e un proprio controllo.

## 10. Blocco III — Retraining con osservazioni degradate (condizionato)

### III.0 Contratto (prima di qualunque training)

Da congelare e approvare:
- **Famiglie e seed:** solo le famiglie che Enrico decide esplicitamente di attivare (§8); seed 0/1/2 come gli originali. ~~Al massimo 2 famiglie × 3 seed = 6 training.~~ **Superato dalla decisione esplicita di Enrico del 24 novembre 2026: 30 training specialistici (2 famiglie × p_on × 3 seed × 5 regimi).** Nessun training per altre intensità o geometrie.
- **Da zero, non fine-tuning (default):** stessa architettura, budget, cadenza di valutazione, early stopping (nessun miglioramento relativo 0.001 per 15 valutazioni) e selezione degli originali. Il fine-tuning è un'alternativa che richiede un controllo dedicato: stesso numero di epoche aggiuntive su input ideali.
- **Distribuzione TRAIN delle degradazioni (proposta, non autorizzata):** per campione, probabilità 0.25 ideale; altrimenti σ ~ U[0, 1.5] m/s e frazione di area invalidata ~ U[0, 0.35] con il generatore di lacune in un namespace TRAIN distinto da quello di valutazione; siti di base come nel training originale (5% ricampionati per epoca). Asse S incluso solo se attivato nel Blocco II.
- **Normalizzazione:** statistiche storiche congelate.
- **Osservazioni consentite:** velocità degradata, coordinate, p. Nessuna T in input.
- **GLU e N variabile:** il wrapper vieta batch eterogenei. **Procedura NON decisa.** La proposta precedente (troncare i siti superstiti al minimo del batch, anche per la U-Net) può alterare l'informazione disponibile e non è più un default: si risolve nel Blocco III, valutando alternative (batch omogenei, bucketing per N, controllo a pari informazione). Il supporto a padding sarebbe una modifica architetturale e richiede controllo separato. Per l'inferenza del Blocco II vale invece il batching di §7 / specifica (`inference_batching`).
- **Checkpoint:** criterio primario identico agli originali (VAL ideale 5%), per confrontabilità e per non aumentare l'uso di VAL; VAL degradata registrata solo come diagnostico, con realizzazioni di un namespace "selezione" diverso da quello di valutazione.
- **Budget e resume:** riferimento S7 ~4 h per U-Net, 5.5–7.8 h per GLU su L40S; costo dell'augmentation da misurare nello smoke. Checkpoint completi per epoca e resume testato.
- **Nessuna modifica** ad architettura o loss insieme all'augmentation. A/B/C restano sospese.

### III.1 Implementazione e controllo di equivalenza

- **Attività:** degradazione nel data loader TRAIN con la stessa libreria del Blocco II.
- **Verifica chiave:** con probabilità di degradazione 0, il nuovo percorso riproduce le prime E epoche di un training originale (loss e metrica di loop) entro tolleranza dichiarata.
- **Stop/decisione:** se l'equivalenza non regge, "stessa ricetta" non basta: servono training di controllo su input ideali con il nuovo codice (fino a 3 per famiglia), da approvare per budget.

### III.2 Smoke e profiling

Un training breve per famiglia sulla partizione normal; verifica di resume dopo interruzione simulata; tempo per epoca; stima della campagna.

### III.3 Training

Array Slurm con catena indipendente dal PC; preemptable solo con resume verificato e account/QoS ammessi.

### III.4 Valutazione 2×2

Stessa specifica, stesse realizzazioni del Blocco II:

| | Input ideali | Input degradati |
|---|---|---|
| Originale | R0 (Blocco II) | celle Blocco II |
| Riaddestrato | R0 sul nuovo modello | stesse celle sul nuovo modello |

Più una cella di **trasferimento** a una geometria di lacuna non usata in TRAIN (es. poligono di bordo alla EngineBench, congelato in I.5), valutata su originali e riaddestrati.

Si riportano separatamente: recupero di robustezza (degradato_riaddestrato − degradato_originale) e costo sul caso ideale (ideale_riaddestrato − ideale_originale). Robustezza nella distribuzione TRAIN e trasferimento a degradazioni non viste sono conclusioni distinte.

### III.5 Report

`docs/nonideal/03_block3_retraining.md` e pacchetto con la stessa struttura del Blocco II. **STOP III.**

## 11. Dipendenze e parallelismo

```
I.1 → I.2 → I.3 → I.5 (STOP I)
        I.4 (opz., D2) ┘
I.5 → II.0 → II.1 → II.2 → II.3 → II.4 → II.5 (STOP II)
II.5 → [decisione esplicita di Enrico] → III.0 → III.1 → III.2 → III.3 → III.4 → III.5 (STOP III)
```

Parallelizzabili: I.1 e la lettura per I.2; I.4 con I.1–I.3; i test CPU di II.1 possono essere scritti durante I.5 ma non eseguiti sul cluster prima dell'approvazione; nel Blocco III, i training delle due famiglie.

## 12. Regole contro leakage e tuning sugli esiti

- Livelli, geometrie, regole di decisione, frame delle figure fissati in I.5, prima di qualsiasi inferenza degradata.
- Nessun livello scelto o modificato guardando VAL o seen.
- Nessuna informazione su T nella generazione di N, G, S. Le lacune correlate a T sono un asse dichiarato con controllo a pari numero di siti.
- Nessun sito invalidato riempito con velocità vere; nessuna interpolazione nascosta.
- Normalizzatore, checkpoint e maschere storiche immutati.
- Seed di degradazione separati da seed di training e di maschera; realizzazioni TRAIN, selezione e valutazione in namespace distinti.
- TEST chiuso; la storia Mese 1 resta registrata.
- seen220 e VAL123 sempre riportati separatamente.

## 13. Risorse, profiling e conservazione degli artefatti

- **Engaging:** fino a 4 GPU normal (`mit_amf_advanced_gpu`, partizione `mit_normal_gpu`); fino a 4 preemptable (`mit_general` con QoS ammesso, partizione `mit_preemptable`). Advanced non è ammesso sulla preemptable. L40S finora. Limiti e occupazione da verificare prima di ogni submission.
- Normal preferita per job lunghi o critici (secondo Enrico leggermente più veloce: osservazione, non garanzia).
- Nessun calcolo pesante sul nodo di login. Smoke prima di ogni array. Catene Slurm con dipendenze, indipendenti dal PC. Resume per task.
- Stima indicativa Blocco II: ~37 000 forward, ordine di poche ore GPU totali; da misurare in II.2. GLU a batch omogeneo può essere più lento.
- Artefatti: per-frame e campi sul cluster; nel repository solo pacchetti aggregati, specifiche, hash, manifest e `INDEX.md`. Nessuna modifica di risultati storici; nuovi percorsi sotto `results/nonideal/`.
- Ogni riga di risultato porta: hash della specifica, del checkpoint, della maschera, SHA del codice.

## 14. Decisioni sostanziali ancora da prendere

| ID | Decisione | Quando | Default proposto |
|---|---|---|---|
| D1 | Livelli e geometrie del core (σ di N1/N2, frazione e scala di G1) | STOP I | §9 (specifica v1, default di scenario) |
| D2 | Scaricare un campione EngineBench (licenza da chiarire) | Prima di I.4 | Sì, solo LSP_small, se la licenza lo consente |
| D3 | Spuri come invalidazione (default) o come outlier di valore | STOP I | Invalidazione |
| D4 | Attivare assi S, campionamento regolare, lacune correlate a T | Dopo il Blocco II | Fuori dal primo blocco; nessuna aggiunta automatica |
| D5 | Trattamento delle lacune correlate a T (leakage) | Se D4 lo attiva | Solo con controllo G-eq e conclusioni separate |
| D6 | ~~Soglie della regola §8.1~~ | Rimossa | Nessuna soglia automatica; decisione esplicita di Enrico dopo il Blocco II |
| D7 | Famiglie del Blocco III e da zero vs fine-tuning | STOP II | Famiglie attivate; da zero |
| D8 | Criterio di checkpoint del Blocco III | III.0 | Identico agli originali |
| D9 | Coinvolgimento di un esperto di diagnostica (contatti solo tramite il gruppo MIT) | Indipendente | Domande pronte da I.2 |

Le scelte implementative ordinarie (nomi dei file, struttura dei job, formati) sono lasciate agli esecutori, nel rispetto di questa roadmap.

## 15. Criteri di consegna

- **Blocco I:** nota di ricognizione completa, specifica congelata e testata, approvazione registrata nel diario.
- **Blocco II:** identità 6/6, test II.1 verdi, replay riportato con discrepanze diagnosticate, completezza 100%, pacchetto validato, report secondo §8 (nessuna soglia automatica).
- **Blocco III (se attivato):** contratto approvato prima dei training, equivalenza III.1 documentata, training conclusi con regola di arresto dichiarata, valutazione 2×2 + trasferimento, report.
- In tutti i casi: commit e push su branch dedicato, verifica del contenuto remoto, voce nel diario; nessun merge indiscriminato su `main`.

## 16. Limiti delle conclusioni

- La T di riferimento è un campo LES interpolato, non una misura; la velocità "vera" è una slice LES a scala di filtro non dichiarata.
- Le degradazioni sono di classe B o C: misurano la **sensibilità** dei modelli a perturbazioni plausibili, non le prestazioni su una PIV reale di questo bruciatore.
- Una maschera motoristica o un rumore preso da un altro esperimento non rendono "realistico" l'input del combustore.
- VAL123 è già stata usata per lo sviluppo: i risultati sono descrittivi, non una conferma indipendente; tre condizioni, tre seed, frame correlati.
- I modelli originali sono stati allenati al 5% casuale: una geometria di campionamento molto diversa misura anche il fuori-distribuzione, non solo la qualità della misura.
- Un miglioramento dopo retraining nella distribuzione TRAIN non implica robustezza a degradazioni non viste.
- Nulla di questo studio valida la ricostruzione su dati sperimentali né decide fra motori e combustore.
