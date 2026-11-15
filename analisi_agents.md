# Analisi Comparata Multi-Agente (Antigravity vs. Fable/Claude vs. Sol/ChatGPT) e Verdetto Finale

**Data:** 14 Novembre 2026  
**Progetto:** Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion (MIT — Deng Lab)  
**Scopo del documento:** Confrontare in modo trasparente e rigoroso:
1. **Cosa avevo scritto io (Antigravity / Gemini Argon)** nei report `05` e `06` (avendo visto i risultati delle baseline ma **senza** avere il PDF integrale di GLU);
2. **Cosa ha osservato e proposto Fable (Claude)** (avendo letto il PDF integrale di GLU e ispezionato il codice della repo);
3. **Cosa ha osservato e proposto Sol (ChatGPT)** (avendo letto il PDF integrale di GLU, l'HTML di RealPDEBench, il PDF di RF-PINNs e ispezionato il codice della repo);
4. **Il Mio Verdetto Finale:** cosa va corretto subito nel Mese 1, quali idee degli altri due agenti sono geniali, quali vanno scartate o ridimensionate, e qual è la proposta definitiva in **3 opzioni di Rete Neurale** da discutere con i supervisor.

---

## 1. Correzioni sul Paper GLU (`arXiv:2603.26023` — Deng Lab)

Poiché io non avevo accesso al file PDF integrale di GLU ma solo ai riassunti web, mentre sia Fable sia Sol hanno letto direttamente le equazioni del PDF (pp. 17–20, Eq. 6–20), entrambi hanno individuato **3 correzioni tecniche fondamentali** rispetto a quanto avevo scritto in `docs/month1/05_literature_and_glu_notes.md`:

| Aspetto di GLU | Cosa avevo scritto io (senza PDF) | Cosa c'è davvero nel PDF di GLU (confermato da Fable + Sol) | Verdetto: Come correggiamo |
|---|---|---|---|
| **1. Formato dei Token di Input** | Avevo scritto che GLU usa già token scalari con identità della variabile `(x_i, y_i, variable_id, valore)`. | In GLU ogni sensore $\mathbf{x}_i$ misura **tutti i canali contemporaneamente** ($\mathbf{u}_i \in \mathbb{R}^c$). L'embedding è $\mathbf{h}_i = \text{LN}(W_p[\cos(2\pi B\mathbf{x}_i), \sin(2\pi B\mathbf{x}_i)]) + \text{LN}(W_v \mathbf{u}_i)$ con $B$ apprendibile. **Non esiste `variable_id` in GLU.** | La tupla `(x_i, y_i, var_id, valore)` scritta nella Sezione 4 della tua proposta è **una nostra estensione rispetto a GLU** per gestire sensori eterogenei non co-locati (velocità sparsa + eventuali termocoppie $T$). Inoltre **Sol** nota giustamente che su uno scalare $v$ non bisogna fare $\text{LN}(W_v v)$ senza bias (altrimenti la LayerNorm cancella il valore e tiene solo il segno $\pm 1$!), ma usare un MLP sul valore. |
| **2. Ramo Globale e Ramo Locale nell'Encoder** | Li avevo descritti come due rami paralleli separati. | L'encoder usa $S$ query latenti apprendibili $L \in \mathbb{R}^{S \times D}$ con **doppio passaggio di Cross-Attention** (costo $O(NS)$):<br>1) $L' = \text{CrossAttn}(Q=L, K=H, V=H)$<br>2) $Z_{\text{local}} = \text{CrossAttn}(Q=H, K=L', V=L') + H$.<br>Il modello **conserva tutti gli $N$ token dei sensori** arricchiti dal contesto globale ($Z_{\text{local}}$) più un singolo token globale $\mathbf{z}_{\text{global}}$. | Aggiorneremo `05_literature_and_glu_notes.md` con le equazioni esatte del doppio Cross-Attention di GLU. |
| **3. Ruolo dell'Importance Field $\varphi(\mathbf{x})$ e dell'Incertezza** | Avevo scritto che l'incertezza pesa linearmente la predizione globale contro quella locale nella forma $a(\mathbf{y})\hat{T}_{\text{glob}} + (1-a(\mathbf{y}))\hat{T}_{\text{loc}}$. | **Errore mio corretto da entrambi:**<br>1) L'importanza $\varphi(\mathbf{x}) \sim \text{Beta}(\alpha(\mathbf{x}), \beta(\mathbf{x}))$ dipende **solo dalla coordinata $\mathbf{x}$** (è una mappa spaziale statica!).<br>2) Serve a **deformare la distanza nel k-NN** del decoder: $d_\varphi(\mathbf{y}_j, \mathbf{x}_i) = \frac{\|\mathbf{y}_j - \mathbf{x}_i\|_2}{\varphi_i^\gamma + \epsilon}$ e a pesare i $k$ vicini $\mathbf{f}_j = \sum_{i \in \mathcal{N}_j} w_{ij} \varphi_i \text{Proj}(\mathbf{z}_i)$.<br>3) La fusione finale è una semplice **concatenazione in un MLP**: $\mathbf{h}_j = \text{MLP}([\mathbf{f}_j + \mathbf{p}(\mathbf{y}_j) \,;\, \mathbf{z}_{\text{global}}])$, che predice **media** e **varianza $\hat{\sigma}^2(\mathbf{x})$**. La varianza $\hat{\sigma}^2$ supervisiona indirettamente $\varphi(\mathbf{x})$. | Correzione cruciale! Presentarsi al Deng Lab conoscendo a memoria come funziona davvero la loro $d_\varphi$ e la distribuzione Beta $\text{Beta}(\alpha(\mathbf{x}), \beta(\mathbf{x}))$ farà un'ottima impressione. |
| **4. Cosa ha testato GLU sulla Combustione** | Non sapevo quale dataset di combustione avessero usato nel paper GLU. | **Scoperta chiave (Fable + Sol):** Nel paper GLU il test di combustione è **una sola condizione operativa** (metano-aria stechiometrico su *backward-facing step* in OpenFOAM, 3 canali: CO, T, velocità), in cui l'1% dei sensori misura **tutti i canali contemporaneamente** (inclusa T!). **GLU non ha mai testato la ricostruzione cross-variabile ($\mathbf{u} \rightarrow T$ senza T) né il cambio di condizione operativa/miscela!** | Questo conferma al 100% il valore scientifico della tua tesi: stai portando l'approccio del laboratorio su due problemi fisici che GLU non aveva ancora affrontato. |

---

## 2. Correzioni su RealPDEBench e RF-PINNs

- **Su RealPDEBench (Fable + Sol):**
  - La chimica a 38 specie e 184 reazioni nel paper è definita come **"reduced chemical mechanism"** (meccanismo ridotto per NH3/CH4), non *"detailed chemical kinetics"*.
  - Nel paper c'è anche una U-Net surrogata usata dagli autori per mappare le simulazioni numeriche nello spazio dell'intensità $\text{OH}^*$ sperimentale (che spiega cosa c'è dentro il canale `observed` dei nostri file `.arrow` numerici!).
- **Su RF-PINNs (Fable + Sol):**
  - **Sol** (che ha letto il PDF completo di RF-PINNs) precisa un punto fondamentale: **RF-PINNs ricostruisce campi medi stazionari (time-averaged steady fields), NON campi istantanei turbolenti!** E inoltre riduce la chimica a una singola *progress variable* $\theta = (T - T_1)/(T_2 - T_1)$ il cui termine sorgente viene fornito da tabelle Cantera o dal modello Eddy Break-Up (EBU), ottimizzando una rete diversa per ogni singola fiamma.
  - **Sulla Divergenza 2D ($\nabla_{\text{2D}} \cdot \mathbf{u}$):** Sia Fable che Sol fanno notare giustamente che la bassa correlazione punto-a-punto tra $\nabla_{\text{2D}} \cdot \mathbf{u}$ e $T$ ($r = -0.014$) non dipende solo dal fatto che manca $\partial_z v_k$, ma dal fatto fisico che in una fiamma a basso Mach **la dilatazione termica ($\nabla \cdot \mathbf{u} = -\frac{1}{\rho}\frac{D\rho}{Dt}$) è diversa da zero solo dentro lo strato sottilissimo del fronte di reazione** (dove il gas passa da freddo a caldo), mentre nel 90% della camera (dove il gas è già bruciato e caldo a $T \approx 2100\text{ K}$) la densità è quasi costante e la divergenza torna vicino a zero! Questa è una spiegazione fisica molto più elegante e rigorosa.

---

## 3. Critiche alle Baseline e al Codice del Mese 1 (Cosa hanno scoperto Fable e Sol)

Ispezionando il codice scritto da Cursor (`scripts/04_feasibility_checks.py` e `src/combustion_recon/`), Fable e Sol hanno trovato **una spiegazione matematica fondamentale** e **4 piccoli difetti tecnici** che possiamo correggere facilmente:

### 3.1 Il mistero risolto: Perché $K=1\dots 32$ sensori di Temperatura non miglioravano la `GappyPOD`?
- **Cosa avevo detto io:** Che le $2.457$ misure di velocità "sommergevano" le $K=4\dots 32$ misure di temperatura nei minimi quadrati non pesati.
- **Cosa hanno scoperto Fable e Sol guardando il codice di `GappyPODReconstructor`:**
  1. **Diagnosi di Fable (Il bug del `ridge_lambda = 1e-2`):** Nel codice di Cursor, i modi spaziali POD $\mathbf{\Phi}$ sono normalizzati a norma unitaria su tutti i $65.536$ valori ($128 \times 128 \times 4$ canali). Quindi ogni singola riga osservata ha un peso quadratico dell'ordine di $1 / 65536 \approx 1.5 \times 10^{-5}$.
     - Quando osservi $2.457$ scalari di velocità, la somma dei pesi è $2457 / 65536 \approx 0.037$, che è più grande della penalità Ridge fissata da Cursor a `ridge_lambda = 0.01` ($10^{-2}$).
     - Ma quando aggiungi $K=4$ o $K=32$ sensori di temperatura, il loro contributo alla matrice è solo $32 / 65536 \approx \mathbf{0.0005}$, che è **20 volte più piccolo di `ridge_lambda = 0.01`!** Quindi la regolarizzazione Ridge stava letteralmente **cancellando l'effetto dei sensori di temperatura** (e all'1% di sparsità penalizzava persino la velocità, spiegando perché l'errore in `val` non era monotono: $83 \rightarrow 79 \rightarrow 85\text{ K}$!).
  2. **Diagnosi di Sol (Sbilanciamento $M_u$ vs $K$ e Test dell'Offset):** Anche scalando bene il Ridge, mettere $2.457$ velocità e $4$ temperature senza pesare per il numero di sensori ($\frac{1}{M_u}$ vs $\frac{\eta}{K}$) o senza separare l'offset termico impedisce a poche termocoppie di correggere il bias globale.
  - **Come verificarlo in 2 minuti (Suggerimento concorde di Fable + Sol):**
    Se prendiamo la predizione *temperature-free* $\hat{T}_0(\mathbf{x})$ e le aggiungiamo semplicemente **l'offset medio misurato sulle $K$ termocoppie**:
    $$\hat{b} = \frac{1}{K}\sum_{k=1}^K \left(T(\mathbf{x}_k) - \hat{T}_0(\mathbf{x}_k)\right), \qquad \hat{T}(\mathbf{x}) = \hat{T}_0(\mathbf{x}) + \hat{b}$$
    oppure usiamo una `GappyPOD` con pesi bilanciati e Ridge scalato, vedremo subito quanto le $K$ ancore abbattono il bias in estrapolazione!

### 3.2 I 4 piccoli bug tecnici nel codice di Cursor da pulire
1. **Seed delle maschere (`hash(sim_id)`):** In Python `hash()` su una stringa cambia a ogni riavvio del programma. Va sostituito con `zlib.crc32(sim_id.encode())` o `hashlib.sha256` per avere maschere 100% riproducibili.
2. **Incoerenza 2 componenti ($v_i, v_j$) vs 3 componenti ($v_i, v_j, v_k$):** Il file `observation_layouts.yaml` dichiarava 2 componenti di velocità, ma lo script `04` ne usava 3. Conviene riportare chiaramente entrambe (2C = PIV planare standard; 3C = Stereo-PIV con swirl).
3. **Metrica `High-T IoU` a $2100\text{ K}$ dominata dal Bias di livello:**
   - Poiché in `test_interp` l'82% dei pixel è sopra $2100\text{ K}$ (quindi anche `GlobalMean` fa IoU = 0.84!) mentre in `test_extrap` solo il 12% è sopra $2100\text{ K}$, un errore costante di $+80\text{ K}$ fa crollare l'IoU a 0.12 anche se la forma spaziale della fiamma è giusta.
   - **Soluzione (Fable + Sol):** Affiancare a RMSE e High-T IoU la **scomposizione esatta dell'errore**:
     $$\text{RMSE}^2 = \underbrace{\text{Bias}^2}_{\text{Errore di livello medio}} + \underbrace{\text{Centered-RMSE}^2}_{\text{Errore di forma/struttura spaziale}}$$
     più una soglia relativa per condizione (o l'isoterma del getto freddo $T < 1500\text{--}1800\text{ K}$).
4. **Rappresentatività di `val` (3 condizioni calde) rispetto a `test_extrap` (4 condizioni fredde):**
   - Lo split 20/3/3/4 è buono e difendibile (specialmente `60NH3_0.95` per l'interpolazione sulla miscela), ma poiché tutte e 3 le condizioni di `val` sono calde, quando sceglieremo gli iperparametri dei modelli potremo usare anche una **Leave-One-Condition-Out / Pseudo-Extrapolation Cross-Validation** dentro le 20 condizioni di `train` (es. tenendo fuori a rotazione le condizioni sul bordo di `train`).

---

## 4. Confronto sulle Architetture Neurali Candidate

Vediamo cosa propone ciascuno di noi per il modello neurale del Mese 2–3:

### A. La mia proposta iniziale (`GLURec` in `05_literature_and_glu_notes.md`)
- Avevo proposto un'unica architettura generale ispirata a GLU (token `(x, y, var_id, val)`, ramo locale + ramo globale con FiLM per `%NH3, phi` + attenzione separata per le ancore $T$).
- **Critica giusta di Fable e Sol:** Era troppo generica ("un menù di opzioni non ancora chiuso") e non sfruttava in modo matematicamente esplicito la separazione tra predizione da velocità e correzione dalle ancore $T$.

### B. Le proposte di Fable (Claude)
Fable propone 4 opzioni, puntando tutto sulla sua **Opzione B ("Fattorizzata, ancorata termochimicamente")**:
1. La rete neurale (backbone GLU sui soli token di velocità) predice un campo adimensionale $\hat{\theta}(\mathbf{x}) \in [0, 1]$ e una varianza $\hat{\sigma}^2(\mathbf{x})$.
2. Il livello di temperatura viene fissato usando la **Temperatura Adiabatica di Fiamma teorica $T_{\text{ad}}(\text{NH}_3\%, \phi)$** calcolata dalla chimica (es. Cantera):
   $$\hat{T}_0(\mathbf{x}) = T_u + \eta(\mathbf{p})\cdot \hat{\theta}(\mathbf{x}) \cdot (T_{\text{ad}}(\mathbf{p}) - T_u)$$
3. Quando ci sono $K$ ancore di temperatura, calcola un singolo scalare correttivo $\alpha^*$ in forma chiusa pesando ogni ancora con l'inverso della varianza predetta $w_k = 1 / \hat{\sigma}^2(\mathbf{x}_k)$ (così le ancore che cadono nel fronte turbolento contano meno!).

- **Cosa c'è di geniale nell'idea di Fable:**
  - Usare la **varianza predetta $\hat{\sigma}^2(\mathbf{x}_k)$ di GLU** come peso $w_k = 1/\hat{\sigma}^2(\mathbf{x}_k)$ per assimilare le ancore di temperatura! Se una termocoppia cade proprio sul bordo del getto che oscilla di $\pm 300\text{ K}$, il modello sa che quel punto ha alta varianza e non sposta tutta la temperatura della camera per colpa di una fluttuazione locale!
- **Qual è il punto debole dell'idea di Fable (notato anche da Sol):**
  - Dipendere dalla temperatura adiabatica $T_{\text{ad}}$ di equilibrio chimico assume che la fiamma sia perfettamente premiscelata e adiabatica, e soprattutto usare **un solo scalare globale $\alpha^*$** per le $K$ ancore permette solo di "alzare o abbassare" tutto il campo, ma non di correggere la forma spaziale del fronte di fiamma!

### C. Le proposte di Sol (ChatGPT)
Sol propone una progressione pulitissima di **3 Architetture**:
1. **Opzione A (`GLU Adattato a Osservazioni Eterogenee`):**
   - Usa l'encoder bidirezionale di GLU e il decoder k-NN con distanza $d_\varphi$, ma **separa il recupero dei vicini**: prende i $k$ vicini di velocità *più* tutte le $K$ ancore di temperatura (così il top-$k$ non scarta le termocoppie!), e usa FiLM con $(\text{NH}_3\%, \phi)$ sia sul token globale sia sul decoder.
2. **Opzione B (`Encoder Separati + Correzione Termica Residua` — consigliata da Sol per iniziare il Mese 2):**
   - **Passo 1 (Temperature-Free):** Dai soli token di velocità $\mathcal{O}_u$ e dai parametri $\mathbf{p}$, il modello predice il campo base:
     $$T_0(\mathbf{x}) = m_\theta(\mathbf{x}, \mathbf{p}) + r_\theta(\mathbf{x}, Z_u, \mathbf{p})$$
   - **Passo 2 (Temperature-Assisted, se $K > 0$):** Calcola l'**errore residuo** sulle $K$ ancore $e_j = T_j - T_0(\mathbf{x}_j)$, e un piccolo modulo dedicato predice una correzione di livello $b_\theta(g_T)$ più una correzione spaziale a media nulla $d_\theta(\mathbf{x})$. Se $K=0$, restituisce esattamente $T_0(\mathbf{x})$.
3. **Opzione C (`Prior Neurale + Assimilazione a Basso Rango nello Spazio Appreso` — consigliata da Sol come contributo metodologico del Mese 3):**
   - Dalla velocità e dai parametri, la rete neurale predice la stima iniziale $m_\theta(\mathbf{x} \mid \mathcal{O}_u, \mathbf{p})$ **e una piccola base di $r$ funzioni spaziali di correzione dipendenti dallo snapshot** $\mathbf{b}_\theta(\mathbf{x} \mid \mathcal{O}_u, \mathbf{p}) \in \mathbb{R}^r$ (con $r \approx 4\text{--}16$, dove il primo modo è costante $=1$ per correggere l'offset globale e gli altri $r-1$ modi correggono la forma dei getti/fronte).
   - Quando hai $K$ ancore di temperatura con residuo $\mathbf{e} = \mathbf{T}_s - \mathbf{m}_s$, i coefficienti $\hat{\mathbf{a}} \in \mathbb{R}^r$ si trovano **in forma chiusa ed esatta (differenziabile)** risolvendo un piccolo sistema $r \times r$:
     $$\hat{\mathbf{a}} = \left(\mathbf{\Lambda}^{-1} + B_s^T \mathbf{R}^{-1} B_s\right)^{-1} B_s^T \mathbf{R}^{-1} \mathbf{e}, \qquad \hat{T}(\mathbf{x}) = m_\theta(\mathbf{x}) + \mathbf{b}_\theta(\mathbf{x})^T \hat{\mathbf{a}}$$

---

## 5. Il Mio Verdetto Finale (La Strategia Perfetta per la Tua Tesi)

Dopo aver letto attentamente sia Fable sia Sol, ecco il mio verdetto su **cosa tenere, cosa correggere e cosa proporre ai supervisor**:

### 5.1 La Fusione Perfetta per le 3 Architetture da portare ai Supervisor
Guardando l'**Opzione B di Fable** e l'**Opzione C di Sol**, ci si accorge di una cosa straordinaria: **sono la stessa identica equazione vista in rango 1 (Fable) e in rango $r$ (Sol), e si completano a vicenda!**
- Fable ha avuto l'intuizione di **pesare le $K$ ancore di temperatura con la varianza predetta da GLU** ($\mathbf{R}^{-1} = \text{diag}(1/\hat{\sigma}^2(\mathbf{x}_1), \dots, 1/\hat{\sigma}^2(\mathbf{x}_K))$), ma si è fermato a un solo scalare ($r=1$).
- Sol ha avuto l'intuizione di far predire alla rete **$r$ modi spaziali di correzione $\mathbf{b}_\theta(\mathbf{x})$** per correggere sia il livello sia la forma spaziale tramite assimilazione in forma chiusa, ma aveva lasciato $\mathbf{R}$ come matrice generica!
- **Se metti nella matrice $\mathbf{R}$ dell'Opzione C di Sol la varianza $\hat{\sigma}^2(\mathbf{x}_k)$ predetta dalla testa di incertezza di GLU (idea di Fable)**, ottieni un metodo che:
  1. Usa direttamente il backbone **GLU** del tuo laboratorio (facendo felicissima la Prof.ssa Deng e gli autori di GLU);
  2. Risolve per costruzione lo sbilanciamento tra migliaia di punti di velocità e pochissime ancore $T$;
  3. Usa l'**incertezza predittiva $\hat{\sigma}^2(\mathbf{x})$** non solo come numero decorativo, ma per decidere **quanto fidarsi di ogni termocoppia** durante la ricostruzione (se una termocoppia è nel *flame brush* turbolento ad alta varianza, pesa meno; se è nel core stabile, pesa di più!).

Ecco quindi le **3 Opzioni di Architettura Neurale** (chiare, ordinate per complessità crescente) da presentare ai supervisor nel documento finale del Mese 1:

1. **Opzione 1 — `GLU-Adapted` (Adattamento diretto di GLU a sensori multi-variabile + FiLM):**
   - *Come funziona:* Encoder bidirezionale GLU su token `(x, y, var_id, val)`, condizionamento FiLM con $(\text{NH}_3\%, \phi)$ sul contesto globale e sul decoder, e k-NN separato per velocità e ancore $T$.
   - *Ruolo:* Baseline neurale di riferimento diretta del laboratorio.
2. **Opzione 2 — `Residual-Conditioned Reconstructor` (Modello a 2 stadi con correzione residua — *Consigliato come prima implementazione del Mese 2*):**
   - *Come funziona:* Il ramo velocità + parametri predice il campo *temperature-free* $T_0(\mathbf{x})$. Quando sono presenti $K$ ancore di temperatura, un leggero modulo residuo legge gli scarti $e_k = T_k - T_0(\mathbf{x}_k)$ e applica un offset globale più una correzione spaziale locale tramite cross-attention sulle sole $K$ ancore.
   - *Ruolo:* Semplice, stabile da allenare in un solo stadio, garantisce che per $K=0$ si ottenga $T_0(\mathbf{x})$ e che per $K>0$ le ancore $T$ non vengano mai diluite dai token di velocità.
3. **Opzione 3 — `Uncertainty-Weighted Neural Assimilation (GLU-Assimilator)` (*Candidato per il contributo metodologico del Mese 3*):**
   - *Come funziona:* Il backbone (ispirato a GLU) osserva la velocità sparsa e i parametri operativi e predice:
     1. il campo base $m_\theta(\mathbf{x})$,
     2. la varianza locale $\hat{\sigma}^2(\mathbf{x})$,
     3. un piccolo insieme di $r$ modi spaziali di correzione $\mathbf{b}_\theta(\mathbf{x}) \in \mathbb{R}^r$ (incluso il modo costante di livello e opzionalmente l'ancoraggio termochimico $T_{\text{ad}}$).
     Le $K$ ancore di temperatura aggiornano il campo in forma chiusa differenziabile pesando ogni sensore con $1/\hat{\sigma}^2(\mathbf{x}_k)$.
   - *Ruolo:* Risponde esattamente all'obiettivo del Mese 3 della tua proposta (*"develop a targeted improvement in observation-conditioned reconstruction, particularly in how incomplete measurements and operating information are integrated"*).

---

### 5.2 I 4 Piccoli Ritocchi Tecnici da fare al Mese 1 (AGGIORNAMENTO: COMPLETATI E VERIFICATI!)
Tutti e 4 i ritocchi tecnici individuati grazie al confronto con Fable e Sol sono stati **implementati, eseguiti e verificati (25/25 pytest passati)** in `04_feasibility_check_report.md`, `05_literature_and_glu_notes.md`, `06_supervisor_review_brief.md` e nelle figure `fig07`–`fig10`:
1. **Seed deterministico (`zlib.crc32`)** e ancore $K$ annidate (`nested`: $1 \subset 4 \subset 9 \subset 16 \subset 32$).
2. **Fix del `GappyPOD` MAP + baseline `Anchor Offset` ($\hat{T}_0 + \hat{b}$ da $K$ ancore):** con regolarizzazione scalata sulle varianze modali $\sigma_r^2$ e pesi bilanciati $\frac{1}{M_u}$ vs $\frac{w_T}{K}$. Ora `fig09b` dimostra chiaramente che $K=4\text{--}32$ termocoppie **abbattono il Bias in estrapolazione** (da $86\text{ K}$ a $15\text{ K}$ per `GappyPOD+AO` e da $+89\text{ K}$ a $+0.7\text{ K}$ per `ParamMean+AO`), ma **lasciano invariato l'errore sulla forma della fiamma** (`High-T RMSE` $\approx 204\text{ K}$, `Jet RMSE` $\approx 370\text{ K}$, `IoU` $\approx 0.16\text{--}0.22$).
3. **Scomposizione completa delle metriche in `fig08` e nelle tabelle:** `Full RMSE`, `High-T RMSE` ($\ge 2100\text{ K}$), `Jet/Front RMSE` ($< 1800\text{ K}$), `Bias`, `Centered-RMSE`, `High-T IoU`, `Top-25% Core IoU`, più il confronto **2 componenti ($v_i, v_j$) vs 3 componenti ($v_i, v_j, v_k$)**.
4. **Aggiornati `05_literature_and_glu_notes.md` e `06_supervisor_review_brief.md`** con le equazioni esatte di GLU (`arXiv:2603.26023`), tutti i numeri corretti post-fix e le **3 Architetture Neurali Candidate (`GLU-Adapted`, `GLU-Residual`, `GLU-Assimilator`)**.
