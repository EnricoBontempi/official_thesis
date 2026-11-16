# Diario della ricerca

Questo file è il diario del progetto di tesi in `official_thesis` (ricostruzione di campi di temperatura da osservazioni sparse in combustione turbolenta). Il precedente progetto esplorativo HONF resta separato in `thesis-research`.

---

## 2026-11-13 — Invio del proposal e avvio del Mese 1

- **Contesto:** Inviata la proposta di tesi (`docs/proposal.md`) alla Prof.ssa Sili Deng e creata la repository `official_thesis` con la roadmap del Mese 1 (`roadmap_month1.md`, commit `75c5db2`, `7dff7d6`). La prof ha risposto che il piano va bene, dicendosi curiosa di vedere i primi aggiornamenti perché un po' scettica sulla possibilità di ricostruire la temperatura *istantanea* dalla sola velocità.
- **Primi controlli sui dati (`Step 1.1`–`1.2` e pilota `Step 1.3`):**
  - Fatta la ricognizione del dataset `RealPDEBench` (sezione combustione) su Hugging Face (`docs/month1/01_remote_reconnaissance.md`).
  - Scoperta importante: negli split ufficiali di RealPDEBench i set `val` e `test` numerici sono vuoti (`[]`), perché loro usano tutte le 30 simulazioni LES (sottocampionate a $64 \times 64$) per fare pre-training di forecasting temporale prima del fine-tuning sui dati sperimentali $\text{OH}^*$. Per il nostro problema dobbiamo quindi definire noi uno split pulito per condizione operativa a piena risoluzione $128 \times 128$.
  - Scaricati 3 shard pilota (`0NH3_1`, `40NH3_1`, `80NH3_1.05`) per controllare formato e canali (`02_pilot_data_audit_report.md`).

---

## 2026-11-14 — Download completo delle 30 condizioni, split e prime baseline

- **Audit completo delle 30 simulazioni LES (`Step 1.3`, `docs/month1/02_data_audit_report.md`):**
  - Scaricati tutti i 30 shard numerici (~58.6 GB, 2001 frame a $4000\text{ Hz}$ per condizione, miscele $\text{NH}_3 \in \{0, 20, 40, 60, 80\}\%$ e $\phi \in [0.75, 1.30]$).
  - **Chiarita la geometria fisica della fetta 2D ($128 \times 128$):** rispetto all'ipotesi iniziale del pilota, le mappe mostrano chiaramente una sezione diametrale simmetrica rispetto alla riga centrale (`row 64`). Due getti freddi di miscela fresca ($T \sim 300\text{ K}$) entrano dal **bordo destro (`col 127`)** e si aprono a V verso sinistra, racchiudendo la zona centrale calda di ricircolo (IRZ, $T \sim 1900\text{--}2280\text{ K}$). I canali di velocità sono `Ch 11` ($v_i$, radiale/trasversale), `Ch 12` ($v_j$, assiale) e `Ch 13` ($v_k$, swirl fuori dal piano). Il modulo $|\mathbf{u}|$ viene sempre ricalcolato da `Ch 11–13` perché `Ch 14` ha piccoli artefatti negativi di interpolazione.
- **Definizione degli split e del protocollo (`Step 1.4`, `docs/month1/03_validation_and_splits_protocol.md`):**
  - Poiché l'autocorrelazione sul fronte di fiamma dura 50–100 frame (`fig04`), abbiamo diviso le 30 condizioni tenendo fuori intere traiettorie (`configs/splits/splits_interp_extrap.yaml`, `fig06`): **20 `train`**, **3 `val`** (`0NH3_1.05`, `20NH3_1.1`, `40NH3_0.9`), **3 `test_interp`** (`20NH3_1.0`, `40NH3_1.15`, `60NH3_0.95`) e **4 `test_extrap`** ai bordi (`20NH3_0.75`, `0NH3_1.3`, `20NH3_1.3`, `80NH3_1.25`).
  - Statistiche di normalizzazione calcolate solo sulle 20 condizioni di `train`.
- **Prima esecuzione delle baseline (`Step 1.5`, commit `567b76f`) e successiva revisione (`analisi_agents.md`, commit `659815a`):**
  - Lanciate le prime baseline non neurali (`GlobalTrainMean`, `ParamConditionedMean`, `LocalHGB`, `GappyPOD`).
  - Facendo revisionare il codice e il PDF di GLU a Fable (Claude) e Sol (ChatGPT), sono emersi due problemi di regolarizzazione nel codice di `baselines.py` (un `ridge = 1e-2` non scalato che schiacciava sia il contributo delle $K$ termocoppie in `GappyPOD` sia la curvatura quadratica in $\phi$ di `ParamMean`) e sono stati chiariti i dettagli esatti del paper GLU (`arXiv:2603.26023`).

---

## 2026-11-15 — Fix delle baseline, risultati definitivi del Mese 1 e report LaTeX per i supervisori

- **Fix delle baseline e risultati verificati (`commit 4fdbcb1`, `docs/month1/04_feasibility_check_report.md`):**
  - Corretto `ParamConditionedMean` (feature centrate attorno a $\phi=1.0$ scelte in `LOCO-CV` sulle 20 condizioni di train) e `GappyPOD` ($R=32$ modi con regolarizzazione scalata sulle varianze modali, più variante con offset sulle ancore `+AO` e proiezione oracolo `OraclePOD`).
  - **Cosa ci dicono davvero i dati del Mese 1:**
    1. **Livello termico medio vs chimica (`fig07`):** a parità di velocità del fluido, la temperatura cambia di $200\text{--}360\text{ K}$ tra $0\%$ e $80\%\text{ NH}_3$. Però basta conoscere i parametri $(\text{NH}_3\%, \phi)$ perché già un modello semplice (`ParamMean`, 0 sensori) stimi il livello medio entro $\sim 18\text{--}19\text{ K}$ (`|Bias|`) in distribuzione e $\sim 43\text{ K}$ in estrapolazione.
    2. **Il vero collo di bottiglia è stimare la forma istantanea del fronte di fiamma:** proiettando la $T$ vera sui 32 modi POD (`OraclePOD`), l'errore di forma (`Centered-RMSE`) scende a $36\text{--}75\text{ K}$ (la base lineare cattura $\sim 80\%$ dell'energia delle fluttuazioni). Ma quando stimiamo quei 32 modi linearmente dal $5\%$ di velocità (`GappyPOD Cond`), il `Centered-RMSE` resta a $67\text{--}139\text{ K}$ (e $205\text{--}360\text{ K}$ lungo i getti e il fronte di fiamma). Il problema non è la base POD, ma il legame fortemente non lineare tra velocità e temperatura nei vortici.
    3. **Fiamme ricche ($\phi \ge 1.20$) e complementarità globale/locale:** l'errore sulla forma del fronte cresce con $\phi$ ($r = 0.73$) sia in training sia in estrapolazione. Sulle fiamme stabili (`0NH3_1.0`) va meglio `GappyPOD` (globale), mentre su quelle ricche e corrugate (`20NH3_1.3`, `80NH3_1.25`) va meglio `LocalHGB` (locale non lineare), motivando l'uso di un'architettura globale-locale come GLU.
    4. **Termocoppie sparse ($K \in \{1\dots 32\}$):** con $K \le 9$ termocoppie i metodi lineari (`+AO` e `+T-block`) non migliorano la forma del fronte; servono $K = 16\text{--}32$ sensori per ridurre il `Centered-RMSE` da $139\text{ K}$ a $119\text{ K}$.
- **Report LaTeX per i supervisori (`overleaf_month1_report/main.tex` e repo `documents_thesis`, commit `3f7c0e2`):**
  - Scritto e compilato il report completo in LaTeX con tutte le figure (`fig01`–`fig10`), le tabelle verificate sui CSV e la proposta in 4 passi per il Mese 2–3:
    - *Step 0:* U-Net da velocità densa $(v_i, v_j) \rightarrow T$ (1 frame vs finestra di 3 frame) per misurare empiricamente quanta informazione sul fronte di fiamma c'è nella velocità prima di introdurre la sparsità;
    - *Step 1:* Adattamento di GLU alla velocità sparsa ($1\%, 5\%, 20\%$) condizionato a $(\text{NH}_3\%, \phi)$;
    - *Step 2:* Modulo residuo a due stadi per assimilare poche termocoppie ($K > 0$) senza farle schiacciare dai token di velocità;
    - *Step 3 (Mese 3):* Base di correzione a basso rango dipendente dallo snapshot (inclusi modi di spostamento del fronte) per rendere utili anche $K = 4\text{--}9$ termocoppie.

---

## 2026-11-16 — Ricezione del codice di GLU da Jason Chen

- **Aggiornamento:** In attesa del riscontro della Prof.ssa Deng e di Linzheng sul report, Jason Chen ci ha girato il repository con il codice ufficiale di GLU (`https://github.com/cosmos2w/GLU_Reconstruction`).
- **Organizzazione (`commit ebaae9d`):** Clonato il codice originale intatto dentro `external/GLU_Reconstruction/` e documentato nel `README.md` come baseline di riferimento del lab (i nostri adattamenti andranno in `src/combustion_recon/`, e i training verranno lanciati su Engaging).
- **Cosa abbiamo visto ispezionando `external/GLU_Reconstruction/`:**
  - Lo Stage-0 (ricostruzione spaziale) usa solo `DomainAdaptiveEncoder`, `phi_mlp_1` e `SoftDomainAdaptiveReconstructor`, saltando il modulo di forecasting temporale.
  - Nel loro `DomainAdaptiveEncoder` c'è già sia il filtro `channel_to_encode` (pensato proprio per osservare solo un sottoinsieme di canali, come la velocità) sia l'argomento `U` per passare i parametri operativi della condizione (`# unused here` nei loro test a singola condizione).
  - La fusione globale-locale nel decoder usa un gate spaziale adattivo (`SpatiallyAwareCLSFusion`) e la testa di varianza è addestrata con loss Gaussian NLL (`training_procedure_pseudocode.tex`).
