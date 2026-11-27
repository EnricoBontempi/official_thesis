# Missione 04 — Step 2 (agente A): modelli di misura per osservazioni termiche realistiche

**Data:** 27 novembre 2026. **Versione:** v2, revisione mirata della v1 (commit `445544f`) dopo il confronto con la consegna dataset (`mission04/step2-thermal-datasets`, commit `cf318f6`). Nota di revisione al §R.
**Stato:** **proposta da discutere nella chat di coordinamento della missione 04. Non approvata.**
**Branch:** `mission04/step2-measurement-models`, base `a19ccc4c9c5a9416eefe40e2679a6fa08366e524`.
**Documento collegato:** [step2_modelli_fonti_e_ricerche.md](step2_modelli_fonti_e_ricerche.md). Contiene: registro delle fonti con livello di lettura, verifiche sui testi, query, accessi falliti e piste di dataset incontrate.
**Materiale di passaggio:** [step2_handoff/](step2_handoff/). Contiene gli appunti di lavoro del coordinatore e il calcolo illustrativo di τ. È stato prodotto durante un'interruzione per limite d'uso e conservato per tracciabilità.
**Perimetro:** solo letteratura e modelli. Esclusi:

- training, inferenza e implementazione del generatore;
- scelta di sensori, target, dataset o architettura;
- contatti con gli autori;
- PDF in repository.

---

## R. Nota di revisione (v1 → v2)

**Correzioni di coerenza e di formulazione.**

1. **Simultaneità.** In v1 la sintesi diceva che tutte e quattro le configurazioni erano in campagne separate. Per EM2C la scheda indicava invece simultaneità non riportata. Ora distinguo tre casi (§1, §3, §11):
   - separazione **documentata**: VT, Delft JSC, Delft JHC;
   - simultaneità **non riportata** e separazione solo **inferita**: EM2C;
   - "non trovato" non significa impossibile.
2. **Irraggiamento.**
   - Il segno dipende dallo scambio radiativo netto. La lettura è sotto T_g solo se l'ambiente "visto" dalla giunzione è radiativamente più freddo.
   - Ho tolto l'affermazione che l'irraggiamento verso pareti fredde sia il termine dominante nei combustori. Nei casi letti non lo è in modo dimostrato: in VT sono maggiori l'errore del giunto di riferimento e l'isteresi (§2.2, §4.1, §10.2).
3. **Modello dinamico.** Distinguo ora (§2.3):
   - tempo convettivo ρcV/(hA);
   - bilancio completo non lineare;
   - tempo effettivo della linearizzazione;
   - primo ordine lineare a coefficienti costanti.

   Il decadimento exp(−t/τ) vale solo per il modello lineare ed è dichiarato come approssimazione.
4. **Durata e inizializzazione.** Ho tolto "0,5 s bastano" e "non rappresentabile / dominata dall'inizializzazione". Al §6.1 elenco da quali ipotesi dipende ogni conclusione:
   - τ stimato;
   - stato iniziale ignoto;
   - transitorio scartato;
   - obiettivo dinamico o statistico;
   - convergenza della media.

   Ho distinto l'intervallo di salvataggio dei campi RealPDEBench (0,25 ms) dal passo del solutore LES, che non è noto.
5. **Geometria.** "La LES non ha pareti" è sostituito da "pareti, stato termico e geometria non sono identificati né disponibili nella slice e nella documentazione usate". "Non supportato dai dati verificati" non significa impossibile. Il tipo di sonda non si sceglie sulla temperatura media del dominio (§6.3).

**Nuove evidenze integrate dalla consegna dataset `cf318f6`.**

- **HYLON Phase 2** (§3.6bis, §11):
  - verificato da me il paper Vilespy et al. 2026, copia HAL;
  - non raggiungibili da questa sessione il PDF TNF e i profili, quindi per montaggio in camera, metodo di estrapolazione e posizioni dipendo dalla verifica dell'agente dataset;
  - il paper usa la sonda a doppia giunzione **all'uscita** del convergente e ricava la T della CRZ dal Raman.
- **PoliMi, Solero 2017:**
  - l'host `re.public.polimi.it` resta bloccato in questa sessione (CONNECT 403);
  - i dettagli (tipo R da 0,3 mm, 500 Hz, 5000 campioni, correzione con velocità PIV/LDA) **dipendono dalla lettura dell'agente dataset**; li riporto come tali.
- **Waterloo e NIST** (MaCFP; README verificati da me):
  - distinguo dati misurati, stime da modello e statistiche compensate;
  - la deviazione standard del NIST compensata (385 K) è un'uscita del modello stesso, non una conferma indipendente;
  - per Waterloo velocità e temperatura sono acquisite **simultaneamente** (125 Hz, 40 000 campioni), ma sono rilasciate solo statistiche.
- **Hubbard 2022:** la frase sui ~100 K è verificata e circoscritta: T_rms sulla linea centrale del plume, sopra 6 cm, LES contro i dati di Weckman (§10.1).

**Aggiunta:** tabella di confronto configurazione → modello → parametri → dataset → lacune → fattibilità (§11).

---

## 0. Come leggere

**Etichette** (come nello step 1):

- **[D]** documentato nella fonte, con posizione;
- **[I]** mia inferenza;
- **[M]** mancante: non riportato nelle fonti lette, oppure fonte non accessibile.

**Livello di lettura delle fonti:** TI, TS, PZ, AB, SEC, come nello step 1 ([registro §0](step2_modelli_fonti_e_ricerche.md)).

**Provenienza dei parametri** (requisito centrale dell'incarico):

| Sigla | Significato |
|---|---|
| **MIS** | misurato o dichiarato nella configurazione stessa |
| **COR** | stimato dagli autori, o stimabile da noi, con una correlazione validata. Si indica quale correlazione e se l'hanno usata gli autori |
| **TRA** | trasferito da un'altra fonte o configurazione, con la motivazione |
| **IGN** | sconosciuto |
| **SENS** | valore proposto solo per un'analisi di sensibilità; **non** è un parametro qualificato |

**Regole che applico:**

- Un'incertezza dichiarata non è rumore gaussiano indipendente.
- La variabilità turbolenta del segnale non è rumore di misura.
- Una traversata in più punti non è una rete di sensori simultanei.
- "Stessa condizione nominale" non significa "stessa realizzazione".

---

## 1. Sintesi

**1. La lettura di una termocoppia nel gas non è "T + rumore".** Nei lavori letti la catena è sempre questa:

1. temperatura del gas attorno alla sonda, già perturbata dalla sonda stessa;
2. temperatura della giunzione (o della guaina), fissata da un bilancio di energia: convezione, irraggiamento verso pareti e gas, conduzione lungo fili e supporti, inerzia termica, eventuale calore catalitico;
3. forza elettromotrice, riferita a un giunto freddo con la sua compensazione;
4. campionamento, media e criteri d'acquisizione: equilibrio, media mobile, numero di campioni;
5. **solo poi**, e non sempre, una stima della temperatura del gas. Si ottiene invertendo il bilancio con parametri in parte misurati e in parte presi da correlazioni (§4).

**2. Configurazioni documentate in modo coerente: quattro**, più HYLON Phase 2 (§3.6bis) emerso dal confronto con l'agente dataset. In tutte la termocoppia è nel gas, dentro o accanto alla regione studiata con PIV. Il rapporto temporale con la PIV non è lo stesso nei quattro casi (schede §3):

- **separazione documentata:** VT, Delft JSC, Delft JHC. Hardware o condizione sono diversi per dichiarazione degli autori;
- **simultaneità non riportata, separazione solo inferita:** EM2C.

Le quattro configurazioni:

- Virginia Tech (VT): sonda tipo B traversata nella zona primaria, **con liner e transition piece scollegati**;
- TU Delft, combustore a getti (JSC): tipo S inguainata, inserita in una **camera d'acciaio gemella** di quella in quarzo;
- TU Delft, jet-in-hot-coflow chiuso (JHC): tipo S da 3 mm in una **piastra che sostituisce una finestra**. La correzione usa h ricavato da una PIV **non reagente**;
- EM2C: tipo K da 1,5 mm nella **zona di ricircolo esterna fredda**, inserita attraverso una barra del telaio delle finestre; nessuna correzione. Il rapporto temporale con la PIV non è riportato.

In tutte e quattro la grandezza pubblicata è un **valore di equilibrio o una media**, non una serie temporale. Lo stesso vale per HYLON (solo medie della temperatura estrapolata, secondo l'agente dataset). Le serie temporali di sonde nel gas che ho trovato hanno uno di questi limiti:

- sono di termocoppie di servizio lente (VT, bead da 0,762 mm, ~1,5 s dichiarati dal costruttore);
- vengono da contesti diversi (pool fire NIST);
- sono termocoppie compensate in sincronia con l'**LDV**, note solo da abstract.

**3. Nessuna termocoppia nel gas caldo acquisita in sincronia con la PIV** nei casi esaminati: separazione documentata o simultaneità non riportata. In un caso (Waterloo, pool fire) termocoppia e **LDA** sono acquisite simultaneamente; sono rilasciate solo statistiche. La ricerca è stata ampliata in questo step:

- ERAU/UCF;
- TU Berlin (non reagente);
- Deng 2020;
- rig cinesi;
- PoliMi;
- UMD.

L'esito non cambia (§3.6). È ancora "non trovato", non "inesistente".

**4. Il modello fisico della lettura è ben documentato in forma e meno nei parametri.** La forma è un bilancio di energia sul bead o sulla guaina. I casi in cui ho trovato **tutti** i parametri della correzione nella stessa configurazione sono:

- Delft JHC, quasi completo; manca la temperatura di parete;
- Delft JSC, tramite la tesi; mancano il diametro della sonda e la lunghezza di percorso radiativa;
- NIST, per il filo fine, ma in un pool fire.

**La risposta dinamica delle sonde da combustore** (bead grandi, guaine) non è mai misurata in situ nei casi letti. Al più è un valore del costruttore, oppure è dedotta dall'isteresi delle traversate (VT).

**5. Per i dati LES** (RealPDEBench: sezione 128×128, campi salvati ogni 0,25 ms, 0,5 s per condizione; passo del solutore non noto) il generatore richiede:

- la **storia temporale** locale di T, della velocità a tre componenti e della composizione;
- l'**ambiente radiativo**: pareti, loro posizione e loro temperatura. Nella slice e nella documentazione usate non sono identificate né disponibili;
- una scelta esplicita del supporto spaziale.

Due cose non si ottengono con il solo post-processing della slice:

- la perturbazione del flusso dovuta alla sonda;
- l'hardware modificato: liner rimosso, camera d'acciaio, finestra sostituita.

**Utilità della finestra di 0,5 s.** Dipende da τ (stimato, non misurato per le sonde da combustore), dallo stato iniziale ignoto, dal transitorio che si accetta di scartare e dall'obiettivo: seguire la dinamica oppure stimare una statistica.

- Con τ stimato dell'ordine di 0,5–1,5 s, tipico di bead grandi o sonde inguainate, la finestra è paragonabile a τ. La lettura simulata dipenderebbe in modo rilevante da un'inizializzazione non nota.
- Con fili fini (τ stimato di decine di ms) il peso dell'inizializzazione si riduce dopo un transitorio da scartare (§6.1).

**Pronto per la discussione:**

- schede VT, Delft JSC, Delft JHC, EM2C;
- forme del modello e loro dominio;
- tabella delle provenienze;
- requisiti per la LES.

**Richiede ancora evidenze:**

- parametri dinamici in situ per sonde da combustore;
- testi integrali di PoliMi, UMD e Lisbona/Imperial (LDV + termocoppia);
- effetto catalitico del Pt in fiamme NH₃/CH₄;
- deposito della semina sulla sonda;
- metodi a due termocoppie.

---

## 2. La catena di misura: generatore e stima

### 2.1 Due cose diverse

- **Generatore della lettura grezza** (modello diretto). Data la storia del gas vicino alla sonda, produce la temperatura della giunzione T_j(t), quindi la tensione e la lettura acquisita. È ciò che servirebbe per creare osservazioni sintetiche dalla LES.
- **Procedura di stima** (modello inverso). Data la lettura, gli autori stimano T_g:
  - correzione radiativa ed eventualmente inerziale;
  - medie e criteri d'equilibrio.

  È ciò che gli autori pubblicano come "temperatura del gas". Contiene ipotesi (h, ε, T_w) che **non** coincidono con la fisica del generatore.

[I] Un'osservazione realistica può essere di due tipi:

- **lettura grezza** (T_j, lenta e con errore di irraggiamento);
- **stima pubblicata** (T_g corretta: errore sistematico ridotto, incertezza diversa, spesso già mediata).

La scelta non è ancora presa. Cambia sia che cosa entra nella rete sia quale incertezza va dichiarata.

### 2.2 Il bilancio di energia (forma generale nelle fonti)

Per un bead sferico di diametro d, uniforme (Biot piccolo), la forma più completa letta è quella del NIST [D, NIST TN 2083r1, §2.1.1, TS]:

```
T_g = T_b + τ dT_b/dt + (ε σ / h) (T_b⁴ − T_surr⁴)        τ = ρ_b c_b d_b² / (6 Nu λ_g)
```

con Nu = 2 + 0,6 Re^½ Pr^⅓ (Ranz–Marshall, 0 < Re < 200).

| Variabile | Significato | Unità |
|---|---|---|
| T_g | temperatura del gas | K |
| T_b | temperatura del bead | K |
| T_surr | temperatura radiativa dell'ambiente | K |
| ε | emissività del bead | — |
| σ | costante di Stefan–Boltzmann | W m⁻² K⁻⁴ |
| h | coefficiente di scambio convettivo | W m⁻² K⁻¹ |
| ρ_b, c_b | densità e calore specifico del bead | kg m⁻³; J kg⁻¹ K⁻¹ |
| λ_g | conducibilità del gas | W m⁻¹ K⁻¹ |
| Re, Pr | numeri di Reynolds (sul diametro del bead) e di Prandtl | — |

Ipotesi NIST:

- conduzione nei fili trascurata, se la lunghezza del filo supera 10 lunghezze fredde l_c = (d_w/2)·√(λ_w/(λ_g·Nu)): l_c = 0,94 mm per un filo da 50 µm a 2 m/s (App. G, Tab. G1);
- dT_b/dt stimata con un fit di 2° ordine su 3 punti (App. D).

**Conduzione nei fili** [D, Shaddix 2017, OSTI 1456684, Eq. 4–11, TI]. Il bilancio del bead include il flusso nei due fili:

```
h_sph (SA_sph − 2A_cyl)(T_g − T_TC) = ε σ SA_sph (T_TC⁴ − T_w⁴) + 2 A_cyl γ dT/dx
```

Profilo esponenziale lungo il filo: T(x) = T_wire + (T_bead − T_wire)·exp(−x√(4H/γd)).

Per i fili Shaddix raccomanda correlazioni valide per Re < 49 e indica Collis & Williams: Nu = (0,24 + 0,56 Re^0,45)(T_m/T_∞)^0,17, per 0,02 < Re < 44. Secondo lo stesso lavoro:

- la correzione con il solo bead e Nu = 2 **sottostima** la correzione vera;
- la lettura è sensibile a variazioni di T_g su una scala *«much larger than the thermocouple bead»* (abstract, Fig. 6).

**Il bilancio stazionario usato dai combustori letti** (forma inversa, senza termine inerziale):

**Segno della correzione radiativa.** Nella forma (ε σ / h)(T_b⁴ − T_surr⁴) la lettura è sotto T_g solo se T_b > T_surr, cioè se l'ambiente radiativo "visto" dalla giunzione è più freddo. Con un ambiente più caldo della giunzione, per esempio gas fresco che vede fiamma o pareti calde, la lettura è sopra T_g [I: conseguenza diretta della forma dell'equazione].

- **Delft JHC** [D, van den Berg 2024, Eq. 1–3, TS]:
  - (Nu·k_g/d)(T_g − T_th) = ε σ (T_th⁴ − T_w⁴);
  - Nu da Churchill–Bernstein per cilindri;
  - ε_S(T) = −0,1 + 3,24·10⁻⁴T − 1,25·10⁻⁷T² + 2,18·10⁻¹¹T³, attribuita a Shaddix 1999. È lo stesso polinomio usato dal NIST per il Pt.
- **Delft JSC** [D, Sampat 2025, Eq. 1; tesi Sampat, App. C.2.1, TS]:
  - ε_tc(σ ε_g T_g⁴ + σ ε_wall T_wall⁴) + h(T_g − T_tc) − σ ε_tc T_tc⁴ = 0;
  - h = Nu·k/D, con Nu da Hilpert (cinque intervalli di Re, tramite Çengel & Ghajar);
  - ε_tc = 0,175 fissa (Bradley & Entwistle 1961: fili Pt-10Rh, 0,1287–0,2243 fra 600 e 1450 °C);
  - ε_g dal modello WSGG (4 gas grigi, HITEMP2010);
  - T_wall = lettura della termocoppia a r = 0, cioè alla parete.
  - [I] Nella forma scritta i pesi di emissione del gas e della parete non sono normalizzati con fattori di vista. Lo riporto come scritto.
- **VT, tabella degli errori** [D, OSTI 1463254, Tab. 6-1, TS]:
  - E_C = (T_r − T_M)/cosh[L(4h_c/(d k_s))^½];
  - E_R = K_R σ ε A_R (T_J⁴ − T_W⁴)/(h_c A_c);
  - E_tr = τ dT_J/dt = (ρ c d/(4h_c))·√6·(STD).
  - **Valori di h_c, ε, L e d non riportati [M].**

### 2.3 Quando basta il primo ordine e quando no

**Quattro oggetti diversi.** Tengo distinti:

1. **Tempo inerziale convettivo:** τ_c = ρcV/(hA). Per la sfera vale ρcd/(6h), per il cilindro ρcd/(4h). È il τ del NIST, di FDS e della tabella VT, e quello della mia stima (§4.3). Non contiene l'irraggiamento.
2. **Bilancio completo non lineare:**

   ```
   ρcV dT_j/dt = hA(T_g − T_j) − εσA(T_j⁴ − T_surr⁴) [− conduzione + catalisi]
   ```

   h, ε e T_surr variano nel tempo e con lo stato locale. La risposta non è un filtro lineare.
3. **Tempo effettivo della linearizzazione:** linearizzando il termine radiativo attorno a T_j,

   ```
   τ_eff = ρcV / [A(h + 4εσT_j³)]
   ```

   È minore di τ_c, perché l'irraggiamento aggiunge una "conduttanza" [I: derivazione mia, non presa da una fonte].
4. **Primo ordine lineare a coefficienti costanti:** dT_j/dt = (T_g − T_j)/τ, con τ fisso. Solo in questo caso valgono la funzione di trasferimento 1/√(1+ω²τ²) (Yilmaz 2008) e il decadimento exp(−t/τ) dell'effetto della condizione iniziale.

**Approssimazione dichiarata.** Quando in questo rapporto uso exp(−t/τ) per ragionare sull'inizializzazione (§6.1), applico il modello 4 come **approssimazione** del 2. Nel modello non lineare la memoria della condizione iniziale decade con un ritmo locale (1/τ_eff(t)), non con una costante unica.

**Il primo ordine con τ costante** è giustificato, nelle fonti, quando valgono insieme:

1. bead o filo nudo con conduzione interna rapida: Biot piccolo [D, NIST; Yilmaz 2008, TI];
2. fili lunghi rispetto a l_c, quindi conduzione verso i supporti trascurabile [D, NIST App. G; Shaddix 2017];
3. h approssimativamente costante nel tempo.

La terza condizione è la più debole in fiamma:

- h dipende da velocità, temperatura e composizione locali;
- [D, AB] Chandran 1985: *«there is no unique time constant in a turbulent flame with widely fluctuating temperature»*;
- [I] il calcolo con correlazioni (§4.3) dà una viscosità cinematica che passa da 1,5·10⁻⁵ m²/s (300 K) a 3,6·10⁻⁴ m²/s (1900 K). Fra gas fresco e combusto τ cambia quindi di un fattore apprezzabile.

**Un modello coerente con le fonti è il bilancio non lineare (oggetto 2)**, con h(t) = f(u(t), T(t), composizione) e il termine radiativo. Un "primo ordine con τ(t)" ne è solo la versione linearizzata localmente (oggetto 3).

**Serve altro quando:**

- **la sonda è inguainata o ceramica** (Delft: guaina di Pt; VT: canna ceramica e tubi ceramici sui fili). Più strati e più capacità termiche: il modello concentrato con il diametro esterno non è validato nelle fonti lette.
  - [D] VT, Cap. 4: le guaine ceramiche *«could further lead to flow stagnation at the bead and hence higher response times»*.
  - [D] Yilmaz 2008: guaina K da 1,575 mm con τ ≈ 5 s in gas di fiamma (valore citato).
- **la conduzione verso i supporti** aggiunge una costante lenta: Petit 1982 (AB) e il metodo dei due tempi caratteristici. Equazioni non lette [M].
- **ci sono forti gradienti lungo i fili.**
  - [D, Smith 1978, SAND78-8726, PZ] Trascurando la conduzione, in un gradiente ripido la termocoppia legge più del gas.
  - [D, Kim et al. 2024, CNF, TS] Fiamma laminare CH₄, tipo R nudo da 125 µm:
    - letture grezze 1640–1431 K contro 1712–1733 K da fluorescenza X;
    - la correzione è dominata dalla conduzione;
    - piccole variazioni dei gradienti assunti portano l'accordo entro 20–50 K.
- **sopra la banda utile** la compensazione amplifica il rumore.
  - [D, AB] Kar 2004: banda limitata per l'amplificazione del rumore.
  - [D, AB] Tagawa 1998: rumore e risoluzione bassa sottostimano τ.
- **in regime transitorio dell'apparato** (riscaldamento delle pareti) la lettura insegue anche la deriva termica: VT, Cap. 4, HTC falsato nelle transizioni di fiamma.

### 2.4 Effetto sulle statistiche turbolente (non è rumore)

- [D, NIST TN 2083r1, Tab. 5, TS] In un punto del pool fire:
  - deviazione standard 237 K misurata, 385 K dopo la compensazione inerziale (rapporto ≈ 0,62);
  - correzione inerziale della media 0,2 K, correzione radiativa 22 K.

  La risposta lenta **attenua le fluttuazioni** e sposta poco la media.

  **Limite epistemico:** i 385 K e la T_g media (TG, TG_RMS nel database MaCFP) sono **stime prodotte dal modello radiativo e inerziale NIST**. Il README MaCFP lo dice: *«The mean gas temperatures (TG) were estimated considering radiative loss and thermal inertia associated with the thermocouples»*; incertezza 5,2 %. Non sono misure indipendenti, quindi:
  - il rapporto 237/385 quantifica l'ampiezza della correzione **secondo quel modello**;
  - non può validare lo stesso modello né la sua funzione di risposta.

  L'agente dataset arriva alla stessa conclusione sui file MaCFP (D04).
- [D, AB] Ferrão & Heitor 1998: la termocoppia *«attenuates the measured velocity-temperature correlations»* rispetto al Rayleigh.
- [D, AB] Chandran 1985: rms robusto, media potenzialmente distorta. Nel loro confronto con il Rayleigh la compensazione a τ unico *«behaved admirably»*.
- [I] La non linearità dell'irraggiamento (T⁴) introduce un **bias della media** che dipende dalla PDF di T, non solo dalla media.
- **Conclusione per il generatore:** varianza ridotta e correlazioni attenuate sono effetti della **fisica della sonda** e vanno generati dal filtro, non aggiunti come rumore.

### 2.5 Catena elettrica e acquisizione

- **Giunto di riferimento** [D, VT Cap. 6, TS]: il riferimento è alla connessione con l'estensione in rame, raffreddato ad aria.
  - Testo: fino a 200 °C → circa −3 %, cioè circa −30 K a 1000 K.
  - Tabella 6-1: massimo 430 K → 1,2 %, cioè −19,2 K a 1600 K; *«Voltage error: 0.1 mV / 8.2 mV»*.
  - Le due stime sono incoerenti nel rapporto. È un errore **sistematico con segno noto**, non rumore.
- **Delft JSC** [D, tesi §3.2.5]: modulo NI-9213 con compensazione interna del giunto freddo. Errore residuo [M].
- **Acquisizione VT** [D]: NI TC01; tre campioni a 2 Hz per punto; salvataggio quando la deviazione standard scende sotto la soglia (tipicamente 0,5 %).
- **Acquisizione Delft JHC** [D]: 10 Hz; media mobile su un diametro di sonda (71 campioni).
- **Acquisizione Delft JSC** [D]: 200 campioni lungo due diametri di sonda, poi media mobile su un diametro. La frequenza di campionamento [M].
- **Disturbo di rete:** visibile a 60 Hz in Yilmaz 2008 [D, TI]. Nei combustori letti non è caratterizzato [M].

### 2.6 Intrusività, catalisi, semina, rotture

- **Perturbazione del flusso**, dichiarata:
  - Delft JHC: *«As the insertion of a probe perturbs the flow, the measured values were averaged over a characteristic length»*;
  - tesi Sampat §7.3: *«probe measurements are intrusive … may affect the flow field»*; la distribuzione di temperatura sulla guaina cambia con l'inserimento;
  - VT Cap. 6: la misura intrusiva non permette di ripetere le prove con geometrie diverse a valle.
- **Spostamento della sonda dovuto al flusso** [D, VT Cap. 4, p. 82]: *«the combustor flow was sufficient to shift the thermocouples out of position»* (tipo T da 40 AWG, flusso non reagente).
- **Rotture** [D, VT]: *«Thinner thermocouples could not be used during the flame experiments due to breakage»*.
- **Catalisi sul Pt.**
  - Pratica documentata: rivestimenti
    - silice in fiamma H₂ (Smith 1978, PZ);
    - BeO-Y₂O₃ su tipo B (Pousse 2009, arXiv, PZ);
    - Y₂O₃-BeO in fiamma NH₃/H₂/N₂ (Okumura 2024, solo estratto);
    - allumina CVD (Bahlawane 2007, AB: elimina la catalisi ma aumenta la perdita radiativa).
  - [D] Kim 2024: nessun effetto catalitico visibile per il tipo R nudo in CH₄.
  - **Entità dell'errore catalitico in fiamme NH₃/CH₄: [M].**
  - [I] Il Pt/Pt-Rh è il catalizzatore industriale dell'ossidazione di NH₃, quindi il rischio è plausibile per RealPDEBench. Non è quantificato.
- **Semina su sonde.**
  - [D, AB] Heitor 1985: l'esposizione alle particelle LDA causa *«large differences»* nella PDF della variabile di progresso.
  - [D, AB] Yanagi & Mimura 1981: deposito compensato con un τ maggiorato.
  - [D, AB] Ferrão & Heitor 1998: campionamento fino a 2000 Hz *«without particle accretion»*.
  - Dettagli quantitativi [M].
- **Fouling delle finestre con la semina** [D, VT Cap. 3]. Riguarda la PIV, non la termocoppia; è un vincolo pratico in più per la simultaneità.

---

## 3. Schede delle configurazioni coerenti

Per ogni scheda distinguo il **montaggio documentato [D]** dalla **ricostruzione [I]**.

### 3.1 Scheda A — Virginia Tech: sonda tipo B traversata nella zona primaria (GT-VT)

Fonte: Ekkad et al., rapporto finale DOE/NETL DE-FE0011762 (2018), [OSTI 1463254](https://www.osti.gov/biblio/1463254). Lettura **TS**: Cap. 4 pp. 81–82 e 134–138; Cap. 5 pp. 165–167; Cap. 6 pp. 188–199. Verificato dal coordinatore sul PDF.

| Voce | Contenuto |
|---|---|
| **Apparato e condizioni** | [D] Combustore a lattina ottico, liner in quarzo (fused silica, spessore 4 mm, lunghezza 216 mm compreso il pezzo metallico; diametro interno non leggibile nel PDF [M]); ugello swirl industriale premiscelato (SoLoNOx); CH₄; 1 atm [I: nessuna pressurizzazione dichiarata per queste prove]. Scansioni TC: Re = 50 000 (sul diametro dell'ugello D_N), φ = 0,65, pilota 6–7 %. Raggio interno del liner = 1,45 D_N |
| **Installazione documentata** | [D] Fili tipo B (PtRh/Pt) in una canna ceramica di ~45 cm, montata su un tubo d'acciaio di 1 m e su un traslatore 2D, inserita **dall'apertura di scarico**. Connettore in un box isolante (piastrelle ceramiche, pasta di zirconia) raffreddato ad aria. **«the liner and transition piece were disconnected during thermocouple scanning in the primary zone»**. Diametro del filo e forma del giunto [M] |
| **Posizioni** | [D] Uscita: 5 stazioni A–E (x/D_c = 1,64–3,34), profili radiali fino a r/D_T = 0,45. Zona primaria: linee assiali a r/D_N = 0; 0,71; vicino alla parete. Mappa 2D di 320 nodi (16×20) in 0 < r/D_N < 1,36 [D]. L'intervallo assiale stampato, *«0.14 < x/DN < 0.29»*, è incoerente con le linee mostrate fino a x/D_N ≈ 2 [I: probabile refuso] |
| **Velocimetria** | [D] PIV planare reagente con liner (Cap. 5): Nd:YAG 140 mJ, camera 2048², TiO₂ da 1–2 µm iniettato nella settling chamber, 7,4 Hz. Il caso R1 (φ = 0,65, pilota 6 %, Re 50 000) **coincide con la condizione nominale** delle scansioni. **Campagne separate**; il confronto TC–PIV è indicato come sviluppo futuro |
| **Grandezza e segnale** | [D] Temperatura della giunzione all'equilibrio, **non corretta**. Tre campioni a 2 Hz per punto; salvataggio con deviazione standard < 0,5 %. Profili e mappe di valori di equilibrio |
| **Errori documentati** | [D] Isteresi secondo la direzione di scansione: *«thermal equilibrium of the probe tip could not perfectly be accomplished»*; *«Transient error is within 10%»*. Giunto di riferimento −30 K a 1000 K (testo) oppure −19,2 K a 1600 K (tabella). Irraggiamento 11,1 K, conduzione 0,15 K, transitorio 0,0016 K a 1600 K (Tab. 6-1). [I] Il transitorio di tabella e l'isteresi misurata non sono coerenti: la tabella non descrive il non-equilibrio osservato. Zona di ricircolo d'angolo misurata più fredda della CFD (1500–1700 K), attribuito dagli autori alle condizioni al contorno ideali della CFD |
| **Modello degli autori** | Formule d'errore della Tab. 6-1 (§2.2), senza correzione applicata ai dati |
| **Simultaneità** | No. Hardware diverso (liner e transition piece assenti durante la scansione) |
| **Altri sensori della stessa campagna** | [D] Cap. 4: TC K con bead da 0,762 mm nel ricircolo e all'uscita, registrate **nel tempo** durante prove transitorie con liner (Re ~12 000–24 000), *«approximate time response of 1.5 seconds»* secondo il costruttore; guaine ceramiche sui fili. In prove non reagenti: tipo T da 40 AWG (0,076 mm), τ ≈ 0,05 s in aria a 20 m/s (costruttore); spostate dal flusso |
| **Per la LES** | Confinamento diverso: quello misurato è senza liner, quello della LES con camera. [I] Le mappe VT non sono un riferimento per uno stato confinato. Utili come **ordine di grandezza** dei gradienti e come esempio di protocollo (equilibrio, isteresi) |

### 3.2 Scheda B — TU Delft: combustore a getti, tipo S inguainata in camera gemella (GT-DelftJSC)

Fonti:

- Sampat, Goselink, Dave, Schrijer, Gangoli Rao, *Fuel* 395 (2025) 135059, [10.1016/j.fuel.2025.135059](https://doi.org/10.1016/j.fuel.2025.135059), CC BY, [copia nel repository TU Delft](https://repository.tudelft.nl/file/File_1f7eab10-17ca-49e4-8a3d-10a7b58f6c14): §2.1–2.2, §3.2 — **TS**;
- R. Sampat, tesi di dottorato TU Delft (2025), [repository](https://repository.tudelft.nl/file/File_bd189a65-5831-4433-8cbf-4f73c2857607): §3.1, §3.2.5, §6.2.2, App. C.2.1, §7.3 — **TS**.

| Voce | Contenuto |
|---|---|
| **Apparato e condizioni** | [D] 12 ugelli da 6,67 mm. Camera cilindrica: in quarzo (diametro interno 208 mm, spessore 3 mm) **oppure** in acciaio 304 delle stesse dimensioni con 6 porte P1–P6. Aria a 350 °C, velocità bulk agli ugelli ≈ 100 m/s, 30–80 kW, φ 0,3–1,0, H₂ 0–100 %, 1 atm |
| **Installazione documentata** | [D] Tipo S con guaina di Pt-10 % (tesi: *«to endure higher temperatures in an oxidative environment»*). Inserimento radiale dalle porte con boccola (foro 0,5 mm più largo della sonda). La camera in sovrapressione esclude la diluizione con aria esterna. Traslatore Zaber. **Camera d'acciaio** *«used during intrusive measurements»*. Diametro della sonda [M] |
| **Velocimetria** | [D] PIV planare reagente **solo** in camera di quarzo, H₂ = 0 %, 60 kW, φ = 0,6 e 0,8; TiO₂ da 0,5 µm tramite un solo ugello; 15 Hz, ~2000 immagini; finestra finale 1,22 mm, passo 0,31 mm |
| **Grandezza e segnale** | [D] Traversata a velocità costante: 200 campioni ogni due diametri di sonda, media mobile su un diametro. Pubblicata la T del gas **corretta** (Fig. 6); dati grezzi in Fig. C.4 della tesi |
| **Modello degli autori** | Bilancio stazionario con irraggiamento di gas e parete più convezione (§2.2). h da Hilpert con D = diametro della sonda e k dei prodotti. *«corrected for radiation and convection based on the velocity field»*. ε_tc = 0,175 (TRA, da fili Pt-Rh). ε_g WSGG (COR); lunghezza di percorso [M]. T_wall = lettura alla parete |
| **Errori dichiarati** | [D] Intrusività; distribuzione di T sulla guaina variabile con la posizione radiale; proposta di verifica con CARS. Incertezza numerica della T corretta [M]; τ [M] |
| **Simultaneità** | No: camera diversa, campagna diversa. [I] La velocità per h viene dalla PIV a H₂ = 0 %; come sia stata usata per gli altri casi non è chiarito [M] |
| **Per la LES** | [I] Fra i casi letti è il più completo per ricostruire l'**inverso** degli autori. Il diametro sonda e la lunghezza di percorso mancano. Geometria e regime (jet-stabilized, MILD/flameless) diversi da uno swirl NH₃/CH₄ |

### 3.3 Scheda C — TU Delft: jet-in-hot-coflow chiuso (LF-DelftJHC)

Fonte: van den Berg, Sampat, Gangoli Rao, *AECS* 20 (2024) 100298, [10.1016/j.jaecs.2024.100298](https://doi.org/10.1016/j.jaecs.2024.100298), CC BY, [repository](https://repository.tudelft.nl/file/File_3e5ff27b-6bbc-43d9-8e1e-0c40a3c3a845): §2, §2.2, Eq. 1–3, Tab. 1–2, conclusioni — **TS**.

| Voce | Contenuto |
|---|---|
| **Apparato e condizioni** | [D] Camera ottagonale (diametro interno 120 mm) con 8 finestre di quarzo da 40×300 mm. Coflow di riferimento a 1000 °C (bulk 1,32 m/s, 11 kW); getto premiscelato a 300 °C, 97–119 m/s; 1 atm; regime flameless |
| **Installazione documentata** | [D] *«One of the quartz windows is replaced by a stainless steel plate with equally spaced ports»*: 10 porte a passo 30 mm. Tipo S, guaina Pt-10 %, **diametro esterno 3 mm**. Traversata radiale alla porta 1 (Y/D = 19,5), dal centro alla parete, a 0,62 mm/s; profili lungo l'asse alle porte 1, 3, 5, 7, 10 |
| **Velocimetria** | [D] Stereo-PIV **solo non reagente** (DEHS ~1 µm nel coflow; 800 immagini; campo di vista circa 58–92 mm dal getto), a condizioni *«different from those of the reacting condition measurements»* |
| **Grandezza e segnale** | [D] 10 Hz; media su un diametro di sonda (71 punti); intervallo di confidenza al 95 % dalla media mobile. T del gas corretta |
| **Modello degli autori** | (Nu·k_g/d)(T_g − T_th) = εσ(T_th⁴ − T_w⁴); Churchill–Bernstein; ε(T) polinomiale. *«The non-reacting flow field measurements were used to correct the thermocouple measurements»*. Definizione di T_w [M]. Riferimento metodologico: Jiménez 2022 (conduzione), letto solo AB |
| **Simultaneità** | No. Livello L4 dello step 1: condizioni diverse per PIV e termocoppia |
| **Per la LES** | [I] Mostra che h dipende da una velocità **non della stessa realizzazione né della stessa condizione**: è una fonte di errore sistematico dell'inverso, da dichiarare se lo si imita |

### 3.4 Scheda D — EM2C: tipo K nella zona di ricircolo esterna fredda (GT-EM2C)

Fonte: Mercier et al., *Combust. Flame* 171 (2016) 42–58, [HAL hal-01405801](https://hal.science/hal-01405801): §2.3–2.5, §3.2–3.3, Fig. 3–6 — **TS**. Il lavoro originale delle termocoppie nella ORZ (Guiberti 2015) non è stato letto [M].

| Voce | Contenuto |
|---|---|
| **Apparato e condizioni** | [D] Swirl premiscelato CH₄/H₂ (X_H₂ = 0,60 nel combustibile), φ = 0,7, 4 kW, reagenti a 293 K, camera quadrata con finestre in quarzo e 4 barre d'acciaio verticali; T_ad = 1881 K |
| **Installazione documentata** | [D] TC tipo K da 1,5 mm *«through one of the four vertical steel bar holding the quartz windows»*. Posizioni in Fig. 6 (punti nella ORZ fino a circa 45 mm dalla barra, letti dal grafico [I]) |
| **Velocimetria** | [D] PIV a 10 Hz reagente (ZrO₂) e non reagente (olio). Simultaneità con le termocoppie [M]; [I] probabilmente separata |
| **Grandezza e segnale** | [D] Temperatura media del gas dopo l'equilibrio termico dell'apparato (~45 min, monitorato da una TC in una barra). **Non corretta**: *«measurement errors related to radiance … do not exceed 5%»* per T < 1200 K |
| **Uso** | [D] Confronto con RANS che impone le pareti misurate (Fig. 6, *«good agreement»*): validazione delle condizioni termiche al contorno |
| **Per la LES** | [I] Tipo B/D dello step 1: un valore medio in una regione relativamente fredda vicina alle pareti. Rappresentabile solo se il dominio numerico contiene la ORZ e le pareti |

### 3.5 Scheda E (incompleta) — termocoppia a filo fine compensata, simultanea con LDV

Fonti:

- Heitor, Taylor, Whitelaw, *Exp. Fluids* 3 (1985) 323, [10.1007/BF01830191](https://doi.org/10.1007/BF01830191): pagina Springer con abstract, lista dei simboli e bibliografia — **PZ**;
- JFM 1987 — **AB**;
- Ferrão & Heitor, *Exp. Fluids* 24 (1998) — **AB**.

**Documentato [D]:**

- fiamma premiscelata stabilizzata da un disco di 0,056 m all'uscita di un tubo da 0,080 m; φ = 0,79; Re 46 800; 9 m/s (JFM 1987, AB);
- compensazione digitale con τ funzione di velocità e temperatura;
- la lista dei simboli definisce Δx/Δr come distanza fra il bead e il volume LDV;
- effetto della semina sulla PDF;
- fino a 2000 Hz senza accrezione (1998).

**Mancante [M]:**

- diametro e materiale del filo;
- valori di Δx/Δr;
- metodo di stima di τ;
- rumore dopo la compensazione.

**Stato:** non è una configurazione qualificata, solo una pista. La forma del modello è quella del §2.2 con τ(u, T). I parametri andrebbero trasferiti (TRA) dal NIST (filo da 50 µm, bead da 153 µm) con la motivazione dichiarata.

### 3.6 Casi annotati, non promossi a scheda

| Caso | Che cosa c'è | Perché non è una scheda | Lettura |
|---|---|---|---|
| PoliMi (Olivani 2007; Coghe 2004; Solero 2017) | Swirl non premiscelato a gas naturale confinato. PIV + LDV + *«thermocouples for the temperature distribution»*. **Secondo l'agente dataset**, che ha letto per intero `IJETR2102.pdf`: TC nuda tipo R (Pt/Pt-13 %Rh), bead da 0,3 mm, 500 Hz, 5000 campioni (10 s); correzione radiativa (Morgan 1991) con h dalla velocità PIV/LDA; campagne separate; solo figure, nessun dato digitale | **Non verificato da me**: `re.public.polimi.it` dà CONNECT 403 in questa sessione. Diametro e tipo sono **dipendenti dalla verifica dell'altro agente**. Parametri della correzione (ε, T_w, correlazione di Nu) [M] in quanto riportato | Per me AB/SEC; agente dataset TI. Bouziane 2009 (HAL, TS) descrive lo stesso rig con termocoppia solo allo scarico |
| Waterloo, pool fire di metanolo (Weckman & Strong 1996; MaCFP) | **Velocità (LDA a 2 componenti) e temperatura acquisite simultaneamente**: *«Four channels of velocity and temperature data were collected simultaneously … 40,000 … at a fixed sampling rate of 125 Hz»*. Filo Pt-Pt10Rh da 50 µm, bead 75–100 µm. Compensazione inerziale digitale con *«several different techniques»*, metodo scelto per coerenza. Nessuna correzione radiativa o catalitica (*«estimated to be less than 5%»*) | Incendio, non combustore. Rilasciate **solo statistiche**: medie, rms, correlazioni ⟨u′T′⟩, scale. Non c'è la serie temporale né il metodo di compensazione adottato | TS: README MaCFP verificati da me |
| UMD, Archer 2005 | Swirl a doppio anello, propano, 33 kW. PIV + *«computer compensated micro-thermocouple»* (medie, rms, scale temporali) | Tesi su DRUM non raggiungibile | AB |
| ERAU/UCF, [OSTI 1846098](https://www.osti.gov/servlets/purl/1846098) | Tipo B a bead esposto, fili saldati, Swagelok piegato a 90°; correzione radiativa (Lemaire & Menanteau 2017) | Misura la **condizione d'ingresso** del crossflow (1620 K) con la sezione di prova **rimossa**; non è nella fiamma della PIV | TS (§2.4–2.5, verificato dal coordinatore) |
| TU Berlin, getto swirl riscaldato ([arXiv 1605.03110](https://arxiv.org/abs/1605.03110)) | K da 0,25 mm, filtro con −3 dB a 421 Hz; incertezza ≤ 1,38 % di T_max (446 °C) | Non reagente | TS (subagente) |
| Deng et al., ACS Omega 2020 (PMC7178801) | Micro-termocoppia S, media di 5000 campioni, collocata nella posizione del vortice individuata con la PIV | Separata dalla PIV; diametro e correzione [M] | TS (subagente) |
| Zhang et al. 2025 (PMC12591111) | 30 tipo B all'uscita di un combustore a kerosene, risposta 2 s; PIV solo a freddo | Uscita, non nella regione PIV | TS (subagente) |
| NIST pool fire (TN 2083r1) | Modello completo con τ, irraggiamento e statistiche. TC e TC_RMS sono **misurati** (giunzione); TG e TG_RMS sono **stime del modello** (README MaCFP) | Non è un combustore. La velocità per h viene da FDS, non da una misura. TG non può validare il modello che lo produce | TS |
| Kim 2024 (CNF) | Tipo R nudo da 125 µm contro fluorescenza X, fiamma laminare | Laminare: utile per l'errore di conduzione, non per lo scenario | TS (subagente) |

### 3.6bis Scheda F (parziale) — HYLON Phase 2: combustore swirl confinato H₂/aria

**Fonti e livelli di verifica:**

- Vilespy et al., *Combust. Flame* 283 (2026) 114536, [10.1016/j.combustflame.2025.114536](https://doi.org/10.1016/j.combustflame.2025.114536), [HAL hal-05347258](https://hal.science/hal-05347258v1/document): §1, §2 (setup, PIV, Raman, temperature), §3.3, Tab. 2 — **TS, verificato da me**.
- Database TNF `HYLON_TNF_2026_01_Phase2-1.zip` (PDF 24 pp. e profili): **non raggiungibile da questa sessione** (tnfworkshop.org, CONNECT 403). Per tutto ciò che viene da lì **dipendo dalla verifica dell'agente dataset** (`cf318f6`, §3.1 e §4.4), marcato **[DS]**.

| Voce | Contenuto |
|---|---|
| **Apparato e condizioni** | [D, paper] Iniettore coassiale a doppio swirl (H₂ centrale, aria anulare). Camera quadrata 78 × 145 mm: **due pareti in quarzo e due in acciaio inox**, queste ultime forate con 8 fori ciascuna per PIV e Raman. Convergente d'uscita (rapporto 0,51). 1 atm; Φ_g = 0,42; fiamma A (2,5 kW, ancorata), fiamma L (lifted). LES AVBP con scambio termico coniugato con le pareti |
| **Termocoppia: che cosa dice il paper** | [D] *«Gas temperature measurements at the combustion chamber outlet were conducted using a double-bead thermocouple»*: due bead di diametro diverso. Poiché il flusso radiativo è proporzionale all'area, il contributo radiativo si elimina. Metodo *«Reduced Radiation Error»* (Brohez, Delvosalle, Marlair 2004; rif. 46: Lemaire & Menanteau 2017). In Tab. 2 la T d'uscita sperimentale vale 945 K (A) e 1110 K (L). La T della CRZ (1050 K e 1300 K) è **dedotta dal Raman**, non dalla termocoppia. Materiale, diametri, campionamento: [M] nel paper |
| **Termocoppia in camera [DS]** | Secondo l'agente dataset il PDF TNF (§5.7) descrive una sonda a doppia giunzione traversata in camera **attraverso una piastra forata** (fori da 1 cm sigillati con lana di vetro). Sei quote z = 2,7–62,7 mm; supporto dichiarato ±4 mm; estrapolazione lineare a superficie nulla T_fluid = T_s + S_s(T_s − T_l)/(S_s − S_l) (Eq. 4, campo omogeneo nel volume della sonda); barre ±10 %. Pubblicata solo la media estrapolata (né T_s/T_l grezzi né rms). Gli autori avvertono che la sonda *«induces local perturbations of the flow and the flame, which may affect the measurements in the CRZ»*. Lega, diametri e rivestimento [M] |
| **Velocimetria** | [D, paper] PIV planare nel piano y = 0: TiO₂ con D₃₂ = 200 nm, 10 Hz. Sessioni da 30 s che iniziano 30 s dopo l'accensione (circa 200 immagini); finestre pulite fra le sessioni per lo sporcamento; ≥ 1000 campi. Prove con il foglio laser attraverso le finestre oppure attraverso i fori delle pareti metalliche: effetto dei fori *«negligible … sufficiently far from the holes»* (App. C, non letta). [DS] Configurazioni `4Quartz` e `2Steel2Quartz` |
| **Relazione temporale fra modalità** | Il paper (novelty statement) parla di *«multiple and simultaneous diagnostics»*. Il §2 però descrive procedure distinte: sessioni PIV dedicate; Raman a KAUST con il proprio sistema laser; termocoppia all'uscita. [DS] Il PDF TNF dichiara campagne separate. [I] La simultaneità istantanea TC–PIV **non** è documentata nel paper; la separazione dipende dalla lettura del PDF TNF da parte dell'agente dataset |
| **Grandezza pubblicata** | T del gas **già corretta** (estrapolata o RRE), solo media. Raman con media e rms d'insieme (precisione a singolo colpo da 4,6 K in aria a 77 K a stechiometria; accuratezza entro 60 K: verificato nel paper) |
| **Confronto TC–Raman [DS]** | 16 punti sovrapposti: 13 compatibili entro ±10 % e ±4 mm; 3 punti nella CRZ a basso gradiente con scarti di +181, +318 e +513 K. Questo scarto **non è una misura esclusiva dell'intrusività**. Concorrono: campagne e laboratori diversi (IMFT e KAUST), accessi diversi (piastra forata contro finestre forate), ipotesi di campo omogeneo dell'Eq. 4, possibile catalisi su H₂ (lega ignota), definizioni diverse di media (estrapolazione di due letture medie contro media d'insieme del Raman) |
| **Pertinenza** | [I] Fra le configurazioni viste è la più vicina per tipo d'apparato: swirl confinato, 1 atm, pareti in parte metalliche, CHT. Diverso combustibile (H₂ e non NH₃/CH₄). Unica con un riferimento ottico (Raman) alla stessa condizione nominale. Per la termocoppia in camera è però **meno documentata** di Delft per il modello: nessuna lettura grezza, parametri della sonda [M] |

---

## 4. Modelli ed errori con le evidenze

### 4.1 Tabella dei meccanismi

| Meccanismo | Effetto sulla lettura | Dipende da | Evidenza (caso) | Generabile dalla LES? |
|---|---|---|---|---|
| Inerzia termica | Ritardo, attenuazione dell'ampiezza: 1/√(1+ω²τ²) (Yilmaz 2008), varianza ridotta | d, ρc, h(u, T, Y) | NIST (std 237 → 385 K); VT (isteresi; τ ~1,5 s del costruttore) | Sì, se c'è una storia temporale locale abbastanza lunga |
| Irraggiamento | Scostamento da T_g di **segno dato dallo scambio netto**: sotto T_g se l'ambiente visto è più freddo, sopra se più caldo; non lineare in T⁴ | ε(T), h, T_surr/T_w e fattori di vista, ε_g | VT 11,1 K a 1600 K (stima); EM2C ≤ 5 % sotto 1200 K; NIST 22 K (modello); Waterloo < 5 % (stima) | Solo con ipotesi su pareti ed emissività, non identificate nei dati verificati |
| Conduzione nei fili e supporti | Bias di segno dipendente dal gradiente; scala spaziale > bead | geometria dei fili, gradiente lungo il filo | Shaddix 2017; Kim 2024; Smith 1978; VT 0,15 K | Solo con orientamento e 3D; una sezione 2D non basta |
| Catalisi superficiale | Lettura sopra T_g | materiale, rivestimento, radicali (H, NH_x) | Solo pratica dei rivestimenti; entità [M] | Non qualificabile |
| Perturbazione della sonda | Flusso e fiamma modificati | dimensione della sonda, guaina, supporto | Delft (dichiarata); VT (geometria non ripetibile) | **No** con il post-processing |
| Hardware modificato | Stato termico e confinamento diversi | liner rimosso, camera d'acciaio, finestra sostituita | VT; Delft JSC; Delft JHC; ERAU | **No** |
| Giunto di riferimento e catena | Bias con segno | T del connettore, compensazione | VT −19/−30 K | Sì, come bias parametrico (SENS) |
| Media spaziale / traversata | Profilo lisciato sul diametro della sonda; isteresi | velocità di traversata, τ | Delft (71 campioni su 3 mm); VT | Sì, in forma di operatore sul campo medio |
| Spostamento / rotture | Posizione incerta; dati mancanti | rigidezza, T | VT (spostamento in non reagente; rotture) | Solo come incertezza di posizione o lacune (SENS) |
| Semina | Alterazione di d ed ε; PDF distorte | carico di particelle | Heitor 1985 (AB); Yanagi & Mimura (AB) | Non qualificabile |
| Rumore elettrico | Componente a 60 Hz; quantizzazione | cablaggio | Yilmaz 2008 | SENS; nei combustori [M] |

### 4.2 Dove l'incertezza dichiarata non è rumore

- VT, "transient error within 10 %": è un'**isteresi** dipendente dalla direzione di scansione, cioè errore correlato e con segno.
- Delft: intervallo di confidenza al 95 % della **media mobile**. Rappresenta la variabilità statistica del segnale fisico più il rumore, non la sola incertezza strumentale.
- PIV di Delft JSC: l'incertezza della media (4 % σ_u) è statistica di campionamento.

### 4.3 Ordine di grandezza di τ e dell'errore radiativo (stima illustrativa, non parametro)

Il calcolo è in [step2_handoff/stima_tau_correlazioni.py](step2_handoff/stima_tau_correlazioni.py), output in [stima_tau_output.txt](step2_handoff/stima_tau_output.txt).

**Ipotesi:**

- correlazioni: Churchill–Bernstein (fili), Ranz–Marshall (bead);
- proprietà da Cantera/GRI 3.0 per i prodotti di CH₄/aria in equilibrio;
- ρc assunti: 3,0–4,2 MJ m⁻³ K⁻¹;
- ε = 0,2 e T_w = 600 K assunti.

Etichetta: **COR + SENS**.

| Sensore | u = 2–30 m/s, T_g = 1000–1900 K | Termine di confronto |
|---|---|---|
| Filo da 50–76 µm | τ ≈ 10–80 ms | NIST: 57 ± 3 ms (filo da 50 µm con bead da 153 µm, circa 1300 K) |
| Bead da 0,3 mm | τ ≈ 70–190 ms; ΔT_rad ≈ 30–120 K a 1500–1900 K | — |
| Bead da 0,762 mm | τ ≈ 0,5–1,4 s | VT: ~1,5 s dichiarato dal costruttore |

[I] I valori stimati sono coerenti in ordine di grandezza con i due valori documentati. Non sostituiscono una misura in situ. Non valgono per sonde inguainate (Delft): manca anche il diametro.

---

## 5. Tabella dei parametri e della loro provenienza

| Parametro | VT (A) | Delft JSC (B) | Delft JHC (C) | EM2C (D) | Filo fine (E) |
|---|---|---|---|---|---|
| Tipo e materiale | B: MIS | S, guaina Pt-10 %: MIS | S, guaina Pt-10 %: MIS | K: MIS | IGN (AB) |
| Diametro sonda / filo / bead | IGN | IGN | 3 mm: MIS | 1,5 mm: MIS | TRA dal NIST (50 µm filo, 153 µm bead) |
| Posizioni | MIS (320 nodi + profili) | MIS (porte P2, P3, P5, P6; radiali) | MIS (porta 1 radiale; asse) | MIS (Fig. 6, solo grafica) | IGN |
| Campionamento | 2 Hz, 3 campioni: MIS | 200 campioni / 2d: MIS; f IGN | 10 Hz: MIS | IGN | fino a 2000 Hz: MIS (AB) |
| Criterio di media o equilibrio | std < 0,5 %: MIS | media mobile su d: MIS | media su d (71 punti): MIS | medie dopo equilibrio dell'apparato: MIS | — |
| τ | IGN (sonda B); 1,5 s per il bead K da 0,762 mm: TRA (costruttore) | IGN | IGN | IGN | COR (τ(u, T) dagli autori, AB) |
| ε sonda | IGN | 0,175: TRA (Bradley & Entwistle, fili) | ε(T): TRA (Shaddix 1999) | non usata | — |
| h / Nu | IGN | Hilpert: COR; velocità da PIV reagente a H₂ = 0 %: MIS | Churchill–Bernstein: COR; velocità da PIV non reagente: MIS, ma altra condizione | — | COR |
| T parete / ambiente radiativo | IGN | lettura alla parete: MIS | IGN | pareti misurate (fosfori, TC): MIS | IGN |
| ε_g | — | WSGG: COR; lunghezza di percorso IGN | non usata | — | — |
| Errore radiativo | 11,1 K a 1600 K: COR (formula, parametri IGN) | incluso nella correzione | incluso | ≤ 5 %: COR | — |
| Errore del giunto di riferimento | −19 / −30 K: MIS (stima) | IGN (NI-9213) | IGN | IGN | IGN |
| Isteresi / transitorio | ≤ 10 %: MIS | IGN | IGN | IGN | — |
| Simultaneità con la velocità | No (hardware diverso) | No (camera diversa) | No (condizione diversa) | IGN | Sì, con LDV |
| Catalisi | IGN | IGN | IGN | IGN | IGN |

---

## 6. Requisiti per i dati numerici e limiti di trasferibilità

### 6.1 Che cosa serve dalla LES per generare la lettura

1. **Storia temporale continua** nella posizione del sensore, con una condizione iniziale per T_j.

   **Che cosa è noto della risoluzione temporale.** RealPDEBench salva i campi ogni 0,25 ms: 2001 frame, 0,5 s per condizione (inventario della missione 01, B-data-audit-001).
   - È l'**intervallo di salvataggio**. Il passo interno del solutore LES non è dichiarato nelle fonti lette [M], e l'intervallo di salvataggio non lo dimostra.
   - [I] Per integrare un'equazione con τ ≥ 10 ms l'intervallo di 0,25 ms è fine. Il contenuto temporale del segnale sotto 0,25 ms (filtrato o sottocampionato dal salvataggio) però non è noto.

   **Da che cosa dipendono le conclusioni sulla finestra di 0,5 s.** Le conclusioni seguenti valgono **nell'approssimazione lineare** del §2.3, punto 4.

   | Ipotesi | Effetto sulle conclusioni |
   |---|---|
   | **(a) τ stimato** con correlazioni (§4.3), non misurato in situ | Se τ reale è diverso, cambiano tutte le soglie |
   | **(b) Stato iniziale e storia precedente** non disponibili: T_j(0) va posto per ipotesi (per esempio uguale a T_g(0) o a una media) | **Una storia artificiale non è un dato fisico verificato** e non va presentata come tale |
   | **(c) Transitorio scartato:** si accettano solo i campioni dopo un tempo t₀ | Nel lineare il peso residuo dell'inizializzazione è circa exp(−t₀/τ); per esempio ~5 % dopo 3τ, ~1 % dopo 5τ. Nel non lineare decade con 1/τ_eff(t), da verificare numericamente |
   | **(d) Obiettivo dinamico** (seguire la realizzazione filtrata) oppure **statistico** (media, varianza della lettura) | Nel secondo caso conta anche la convergenza della statistica su ciò che resta dopo t₀. La finestra contiene circa 70–90 periodi della frequenza osservata nell'OH* del benchmark (~140–175 Hz; step 1, §11.2, lettura grafica). Eventuali scale più lente non sono note, e il filtro della sonda riduce ulteriormente il contenuto temporale utile |

   Ne segue [I], **condizionato ad (a)–(d)**:

   - **fili fini** (τ stimato 10–80 ms): dopo t₀ di alcuni τ resta la maggior parte della finestra;
   - **bead da 0,3 mm** (τ stimato 0,07–0,19 s): t₀ = 3–5τ consuma da circa metà a tutta la finestra;
   - **bead da 0,76 mm o sonde inguainate** (τ stimato 0,5–1,5 s, o ignoto): la finestra è paragonabile a τ. La lettura dipende in modo rilevante dall'ipotesi (b). Senza un'informazione sulla storia precedente la lettura simulata non è qualificabile come osservazione realistica. Non è "impossibile": è non supportata dai dati attualmente verificati.

   Lo shard `40NH3_1.2` ha un'associazione campo–tempo non verificata ed è da escludere per usi multi-frame (missione 01).
2. **Velocità locale a tre componenti** per Re. La componente fuori piano c'è nei dati grezzi, non nelle cache (missione 01, B-data-audit-004).
3. **Composizione e proprietà del gas** (k, μ, Pr). I canali LES contengono 7 frazioni molari: CH₄, CO, CO₂, H₂O, NH₂, NH₃, OH. [I] N₂, O₂ e H₂ vanno inferiti o assunti; le frazioni negative note (missione 01) vanno trattate.
4. **Ambiente radiativo**: temperature ed emissività delle pareti, fattori di vista, lunghezza di percorso per ε_g. Pareti, stato termico e geometria **non sono identificati né disponibili** nella slice e nella documentazione usate: condizioni termiche non dichiarate (step 1, §11), orientamento e posizione della slice non dichiarati (missione 01). Non è escluso che la simulazione d'origine li contenga. [I] Con i dati attualmente verificati l'errore radiativo è un parametro di sensibilità, e anche il suo **segno** dipende dall'ipotesi sull'ambiente (§2.2).
5. **Gradienti lungo i fili e orientamento della sonda** per la conduzione: richiedono campo 3D o ipotesi. [I] Una sezione 2D non basta. Per fili lunghi e sottili si può dichiarare la conduzione trascurabile (criterio NIST di 10 l_c), da verificare caso per caso.
6. **Supporto spaziale.** Celle 0,58 × 0,76 mm (missione 01). [I]
   - Bead fini: puntuali rispetto alla griglia.
   - Sonde da 1,5–3 mm: 2–5 celle.
   - Delft media su un diametro di sonda durante la traversata.
7. **Posizione accessibile.** [I] Una sonda reale entra da una parete, da una porta, da una barra o dallo scarico. Nella sezione LES non ci sono pareti identificate né un orientamento fisico dichiarato. Con i dati verificati la plausibilità dell'accesso non è verificabile.

### 6.2 Che cosa il post-processing della slice non può riprodurre

- La perturbazione del flusso e della fiamma dovuta alla sonda e al supporto, dichiarata da Delft e VT.
- Le modifiche dell'apparato:
  - liner rimosso (VT);
  - camera d'acciaio con pareti e stato termico diversi (Delft JSC);
  - finestra sostituita da piastra (Delft JHC);
  - sezione di prova rimossa (ERAU).

  Lo stato misurato in quelle prove non coincide con lo stato dell'apparato ottico confinato. Rappresentarlo richiederebbe una simulazione dell'hardware modificato, non un post-processing.
- La catalisi superficiale in NH₃/CH₄: non quantificata.
- La deposizione di semina in una prova simultanea con la PIV: mai documentata con la PIV.

### 6.3 Compatibilità con lo scenario RealPDEBench (inferenze)

- **In comune con il caso:** combustore confinato con accesso ottico, 1 bar, velocità O(1–40) m/s (missione 01).
- **Scelta del tipo di sonda.** Non va fatta sulla temperatura media del dominio (~1860–2190 K per condizione, missione 01). Dipende da:
  - le **condizioni locali previste** nella posizione scelta (gas fresco, ricircolo, zona di reazione), compresi picchi ed escursioni;
  - l'**intervallo operativo** del tipo di termocoppia e della guaina;
  - la **compatibilità** con l'ambiente: ossidante, ricco di H e di radicali azotati (catalisi; §2.6);
  - la sopravvivenza meccanica (rotture VT).

  Nei casi letti i tipi K sono usati in zone sotto ~1200 K (EM2C) e i tipi S/B nel gas più caldo (VT, Delft). È una constatazione sui casi, non una regola di scelta.
- **Diverso:**
  - combustibile NH₃/CH₄ (catalisi, radicali azotati: [M]);
  - geometria swirl ignota;
  - nessuna temperatura sperimentale e nessuna PIV in combustione nel benchmark.
- [I] Lo scenario plausibile non è "il laboratorio RealPDEBench". È una **sonda documentata in un'altra configurazione**, trasferita dichiarandone i limiti: velocità, temperatura, diametro, accesso.

---

## 7. Domande per l'agente dataset

1. Esistono **serie temporali grezze** di termocoppie nel gas (non solo medie o profili) in combustori o fiamme con velocità misurata? Con quale frequenza e durata, e con T_j o T_g corretta?
2. Il rapporto VT (OSTI 1463254) o il repository di Virginia Tech rilasciano le mappe di temperatura (320 nodi) e i campi PIV R1? Con le letture grezze?
3. TU Delft: i dati del combustore a getti (Fig. 6, Fig. C.4 grezza) e del JHC sono depositati (4TU.ResearchData)? Contengono diametro della sonda, h e T_w usati?
4. Esistono dataset con **termocoppia + LDV simultanee** (Lisbona/Imperial, Sydney, Darmstadt) oppure termocoppia + Rayleigh/Raman nello stesso punto, utili a qualificare la funzione di trasferimento?
5. Archivi TNF (Sydney bluff body, Darmstadt): oltre alle temperature di superficie, ci sono termocoppie nel gas con metadati di montaggio?
6. Per i dataset trovati: i dati sono già corretti? Ci sono dati mancanti o rigetti dichiarati? Sono documentati la temperatura del giunto freddo e la catena di acquisizione?
7. Ci sono dati di fiamme **NH₃ o NH₃/CH₄** con termocoppie Pt nel gas, rivestite o no, che permettano di valutare l'effetto catalitico?
8. Il rig SJTU (Liu 2024), probabile origine di RealPDEBench, ha dati termici pubblici? I subagenti non hanno trovato termocoppie nella camera.

---

## 8. Alternative per lo step 3 (proposte, non approvate)

| Opzione | Che cosa rappresenta | Generatore | Pro | Contro e limiti |
|---|---|---|---|---|
| **S1. Sonda puntuale fissa con primo ordine e τ(t)** | Lettura grezza lenta di un filo o bead Pt-Rh in una posizione accessibile | Bilancio §2.2: τ(u, T, Y) da Nu-correlazione; termine radiativo con T_w come SENS; bias del giunto freddo come SENS | Usa la storia temporale della LES; separa filtro fisico ed errore | Simultaneità con la PIV **non documentata**: da dichiarare come configurazione ipotetica. τ della sonda reale da combustore IGN. Finestra di 0,5 s |
| **S2. Statistica di traversata** (profili medi su pochi punti) | Ciò che VT, Delft ed EM2C pubblicano davvero: valori d'equilibrio o medie | Media temporale della LES nel punto, più bias radiativo e media spaziale su d | Fedele ai casi documentati | Informazione sulla condizione, non sulla realizzazione. 0,5 s per la media: convergenza da verificare. Hardware diverso nelle fonti |
| **S3. Filo fine compensato** (fluttuazioni filtrate) | Configurazione con LDV (Heitor; Ferrão) | S1 con τ piccolo, compensazione e banda limitata; rumore amplificato come SENS | Segue la realizzazione | Parametri solo da AB o TRA dal NIST. Fragilità in fiamma (rotture VT) |
| **S4. Temperatura di parete** | Fosfori o TC di superficie (step 1) | Non generabile dalla slice: nessuna parete identificata | Pratica diffusa nei casi letti | Non supportata dai dati RealPDEBench verificati. Lo diventerebbe con geometria e condizioni di parete. Temperatura di parete ≠ gas adiacente |
| **S5. Termometria ottica puntuale o di linea** (Raman/CARS) | Statistiche o serie a kHz da un'altra campagna | Campionamento con volume di 0,5 mm e incertezza dichiarata | Supporto noto | Campagne separate dalla PIV nei casi letti; non "lenta e sparsa" |

[I] Per il confronto con la sola velocità, S1 e S2 sono le più vicine all'incarico ("misure lente e sparse"). S2 è documentata. S1 richiede di dichiarare la simultaneità come ipotesi.

---

## 9. Lacune e verifiche mancanti

- **Testi integrali non accessibili da questa sessione:**
  - PoliMi (Olivani 2007, Coghe 2004). Solero 2017 è letto dall'agente dataset ma non da me: dipendenza dichiarata, §3.6;
  - database TNF HYLON Phase 2 (PDF e profili): letto dall'agente dataset, non da me (§3.6bis);
  - UMD (Archer 2005);
  - Heitor 1985/1987, Ferrão & Heitor 1998, Yanagi & Mimura 1981;
  - Tagawa (due termocoppie), Hung/McLoone, Petit 1982, Lenz & Günther, Heitor & Moreira 1993;
  - Jiménez 2022, Hindasageri 2013.

  Host bloccati: [registro §5](step2_modelli_fonti_e_ricerche.md).
- **Valori mancanti anche nei casi letti:**
  - diametro della sonda tipo B (VT) e della sonda Delft JSC;
  - T_w di Delft JHC;
  - lunghezza di percorso WSGG;
  - appendice supplementare B.1 di Fuel 2025: sostituita dalla tesi, che riporta la stessa procedura (App. C.2.1).
- **Effetto catalitico in NH₃/CH₄** e **deposito della semina PIV** sulle sonde: [M].
- **Tolleranze ASTM/IEC** e deriva dei tipi S/B ad alta temperatura: non lette.
- **Sensore virtuale:** nessun precedente in combustori a gas (§10).
- **Waterloo:** simultaneità termocoppia–LDA verificata sul README MaCFP. Il metodo di compensazione adottato e lo stato (compensato o no) delle T_rms rilasciate non sono univoci nei testi letti.
- **Review non lette:** Brübach 2013, Abram 2018 (PECS, ScienceDirect 403).

---

## 10. Sensore virtuale nelle simulazioni e alternative (integrazione SA3)

### 10.1 Il sensore virtuale esiste come pratica numerica, nel campo degli incendi

Tre implementazioni documentate. Le hanno lette i subagenti; i punti decisivi di FDS e di Hubbard 2022 li ho riverificati sui testi.

**FDS, dispositivo THERMOCOUPLE** [D, User Guide, §"Thermocouples"; codice `dump.f90`/`func.f90`; tag FDS-6.11.2 e master, TS]:

- equazione, verificata sul sorgente LaTeX: (D/6)·ρ·c·dT_TC/dt = ε(U/4 − σT_TC⁴) + h(T_g − T_TC), con U intensità radiativa integrata;
- h = k·Nu/D, con Nu = 2 + 0,6 Re^½ Pr^⅓; nel codice Pr è fissato a 0,7 (verificato);
- default: D = 1 mm, ε = 0,85, ρ e c di un tipo K;
- aggiornato a ogni passo temporale, a partire da una temperatura iniziale dichiarata;
- nessuna conduzione nei fili, nessun effetto sul flusso, nessuna media spaziale ammessa.

Nel Validation Guide FDS il pool fire di Waterloo è modellato con bead da 100 µm, *«to match the experiments»*.

**Sierra/Fuego** [D, SAND2022-12437, §2.15, OSTI 1888153, TS dal subagente]:

- Churchill–Bernstein con la componente di velocità **perpendicolare** alla termocoppia;
- irraggiamento direzionale incidente;
- ipotesi dichiarate: *«one-way coupling»*; sonda *«entirely within a single computational cell»*; conduzione trascurata. Per maggior dettaglio *«meshing the thermocouple itself may be required»*.

**Hubbard et al. 2022** (LES Fuego del pool fire di Waterloo; [10.1115/1.4054204](https://doi.org/10.1115/1.4054204), [OSTI 1887423](https://www.osti.gov/servlets/purl/1887423), §3.3, TS; passaggi verificati):

- bilancio con irraggiamento scambiato con il **gas locale**, non con le pareti;
- bead da 90 µm; ε = 1 assunta; proprietà del Pt;
- velocità come rms delle componenti radiale e assiale;
- integrazione a ogni passo LES (~2,5·10⁻⁴ s);
- risultato, verificato da me (§3.3, p. 19 del manoscritto OSTI): *«Without these corrections, simulated Trms values would over-predict experimental values by approximately 100 K in regions above 6 cm in height»*.
  - **Contesto:** T_rms sulla **linea centrale del plume** (Fig. 5), LES Fuego contro i dati Waterloo, sopra 6 cm dalla superficie della vasca.
  - Non è un valore generale né un errore della termocoppia.
  - Effetto piccolo sulla media, grande su fluttuazioni e correlazioni u′T′.
  - **Cautela [I]:**
    - Hubbard stesso cita un'incertezza del 25–30 % sulle fluttuazioni sperimentali.
    - Il README Waterloo dice che la temperatura è stata *«digitally compensated using several different techniques»* per formare le correlazioni, e che *«no in situ compensation was applied to the data here»*. Dai testi letti **non è univoco** se le T_rms rilasciate siano compensate o grezze.
    - Se fossero compensate, confrontarle con una LES filtrata da un modello di sonda non sarebbe coerente. L'entità dei ~100 K va quindi presa come indicazione dell'ordine dell'effetto in quel caso, non come qualificazione del modello.

**Esito per i combustori a gas: nessun LES o DNS** di combustore a gas che applichi un modello di termocoppia ai segnali simulati, nelle fonti raggiungibili [M].

### 10.2 Conseguenze per un generatore su RealPDEBench [I]

- La forma FDS/Fuego/Hubbard è un **precedente documentato** per l'opzione S1 (§8): primo ordine non lineare con h(u, T, Y) locale, integrato sul passo della simulazione.
  - L'**intervallo di salvataggio** di RealPDEBench (0,25 ms) ha lo stesso ordine del passo LES di Hubbard (~2,5·10⁻⁴ s). Il passo del solutore RealPDEBench non è noto, e Hubbard integra a ogni passo del solutore, non su campi salvati.
  - Il precedente riguarda incendi, con bead fini e velocità di pochi m/s.
- I tre modelli scelgono in modo diverso la velocità (modulo, componente normale, rms di due componenti) e il termine radiativo (U dal solutore, intensità direzionale, T_g locale). Sono **scelte di modello da dichiarare**, non dettagli.
- **L'ambiente radiativo è il punto debole:**
  - nei dati RealPDEBench verificati non sono disponibili né un campo radiativo né pareti identificate;
  - Hubbard usa il gas locale come ambiente radiativo: è la scelta meno esigente, ma elimina lo scambio con pareti più fredde o più calde.
  - Nei combustori letti l'irraggiamento è un termine rilevante ma **non dimostrato dominante**. In VT l'errore radiativo stimato (11,1 K a 1600 K) è minore di quello del giunto di riferimento (−19/−30 K) e dell'isteresi (fino al 10 %). In Delft la correzione complessiva non è separata nei suoi termini nei testi letti. In EM2C e Waterloo è stimato ≤ 5 %.

### 10.3 Alternative: informazione disponibile

**Superficie** [D: Nau 2018, TS; Arndt 2021, PZ; Koren 2018, PZ]:

- fosfori a 1 kHz sul bluff body PRECCINSTA: σ 10–17 K; picco del PVC a 475 Hz di ~7 K; accuratezza 0,5–1 %;
- mappe delle finestre del GTMC: precisione 3 % a 1330 K;
- le pareti (~850–1330 K) sono molto più fredde del gas combusto adiacente;
- per usarle una LES deve contenere le pareti (temperatura imposta o CHT; Koren 2018).
- [I] Con la sezione RealPDEBench, dove non è identificata alcuna parete: non supportata dai dati attualmente verificati.

**Ottiche:**

| Tecnica | Supporto e frequenza | Incertezza | Limiti per noi |
|---|---|---|---|
| fs-CARS (Dennis 2016) | ~600 µm, 5 kHz | ±2 % precisione, ±3 % accuratezza | PIV in campagne precedenti |
| Raman (Weigand 2006, citato) | 0,6 mm, 10 Hz | ~4–5 % | — |
| Two-line OH-PLIF | piano | 4–7 % | Nessun dato senza OH: media distorta |
| TDLAS | integrale di linea | — | Richiede T e X_H₂O 3D lungo il raggio: una sezione 2D non basta |
| PIV termografica | piano, stesse particelle della PIV | — | Limite 900–1000 K con BAM:Eu (Yin 2017). [I] Non raggiunge le temperature della LES (1860–2190 K) |

### 10.4 Assimilazione di dati termici sparsi

- **Incendi:** EnKF con alberi di termocoppie (Lin & Wang, AB); FDS inverso (Jahn, Rein, Torero 2011, AB).
- **LES di combustione:** EnKF di Labahn et al. 2019 assimila **solo velocità** (tomo-PIV, getto non reagente; OSTI 1772036, PZ).
- **Assimilazione di termocoppie in LES di combustione:** non trovata [M].

---

## 11. Confronto con la consegna dataset e stato delle configurazioni

### 11.1 Tabella di confronto

Classi di stato:

- **[SPEC]** sufficientemente documentata per discutere una specifica;
- **[PAR]** parametri essenziali mancanti;
- **[IPO]** ipotesi o sensibilità priva di qualificazione sperimentale.

| Configurazione | Modello disponibile | Parametri documentati | Evidenze nei dataset (agente dataset, `cf318f6`) | Lacune | Fattibilità ancora da verificare | Stato |
|---|---|---|---|---|---|---|
| **Delft JHC** (S, guaina 3 mm, porta su piastra) | Bilancio stazionario convezione + irraggiamento; Churchill–Bernstein; ε(T) | d, tipo, guaina, posizioni, 10 Hz, traversata 0,62 mm/s, media su d | Nessun dato digitale individuato (riferimento documentale) | T_w; τ; incertezza della T corretta; h da PIV non reagente in altra condizione | Generatore della **media** di profilo con ambiente radiativo SENS; nessun dato dinamico | **[SPEC]** per una statistica di traversata; **[PAR]** per la dinamica |
| **Delft JSC** (S con guaina di Pt, camera gemella) | Bilancio con gas e parete; Hilpert; ε = 0,175; WSGG | Tipo, guaina, porte, 200 campioni/2d, T_wall misurata | Nessun dato digitale individuato | Diametro della sonda; frequenza; τ; lunghezza di percorso | Come sopra; geometria jet-stabilized lontana dallo swirl | **[PAR]** (manca d) |
| **VT** (B traversata, liner rimosso) | Solo stime d'errore (Tab. 6-1), nessuna correzione applicata | Posizioni (320 nodi), 2 Hz con std < 0,5 %, isteresi ≤ 10 %, bias del giunto freddo | Nessun dato digitale individuato | Diametro del filo; h, ε, L; hardware non confinato | Lo stato misurato non corrisponde a un apparato confinato: non rappresentabile con post-processing | **[PAR]**; utile come protocollo |
| **VT, bead K da 0,762 mm in esercizio** | τ ~1,5 s dichiarato dal costruttore | Bead, guaine, posizioni qualitative | — | τ in situ; posizioni precise; campionamento | Finestra di 0,5 s paragonabile a τ (§6.1) | **[IPO]** |
| **EM2C** (K 1,5 mm nella ORZ) | Nessuna correzione (≤ 5 %) | d, tipo, accesso da barra, equilibrio dopo ~45 min | Nessun dato digitale individuato | Simultaneità non riportata; posizioni solo grafiche; τ | Richiede una ORZ nel dominio, non identificata nella slice | **[PAR]** |
| **HYLON Phase 2** (doppia giunzione) | Estrapolazione a superficie nulla (Eq. 4, [DS]) in camera; RRE all'uscita (paper) | [DS] 6 quote, supporto ±4 mm, ±10 %; [paper] PIV, Raman, CHT | **Profili TC, Raman, LIP e PIV ispezionati**; 13/16 punti compatibili con il Raman; 3 scarti grandi nella CRZ con cause non isolate | Lega, diametri, letture grezze, rms TC, τ; PDF TNF non letto da me | È l'unico caso con riferimento ottico alla stessa condizione: utile per discutere la plausibilità di medie e supporto. Non dà serie temporali | **[PAR]** per la sonda; **[SPEC]** per discutere posizioni, esclusioni e supporto |
| **PoliMi Solero 2017** (R bead 0,3 mm) | Correzione radiativa con h da velocità PIV/LDA (Morgan 1991) [DS] | [DS] d, tipo, 500 Hz, 5000 campioni | Solo figure, nessun dato digitale | Lettura diretta mancata (host bloccato); ε, T_w, correlazione | Unico caso con campionamento a 500 Hz in un combustore swirl: va verificato se riporta rms o serie | **[PAR]**, dipendente da [DS] |
| **Filo fine compensato, simultaneo a LDV** (Heitor; Waterloo) | Primo ordine con τ(u, T), compensazione | Waterloo: d, 125 Hz, 40 000 campioni, simultaneità (verificati) | Waterloo: statistiche congiunte ⟨u′T′⟩ ispezionate; nessuna serie | Metodo di compensazione; stato delle T_rms; incendio, non combustore | Trasferimento a combustore confinato non qualificato | **[IPO]** per il nostro scenario |
| **NIST pool fire** (filo 50 µm) | Bilancio completo con τ, irraggiamento, conduzione trascurata | d, τ 57 ms (modello), correlazioni | TC/TC_RMS misurati; TG/TG_RMS da modello (verificato su README) | Nessuna misura indipendente del gas per validare il modello | Fonte dei parametri di forma, non di validazione | **[IPO]** come parametri trasferiti |
| **Sensore virtuale S1 su RealPDEBench** | FDS/Fuego/Hubbard (forma) | Nessun parametro della nostra configurazione | — | τ, ambiente radiativo, accesso, storia iniziale, passo del solutore | Tutta da qualificare | **[IPO]** |

### 11.2 Che cosa conferma il confronto

- **Confermato:**
  - nessun dataset aperto ispezionato contiene una termocoppia nel gas e una PIV nella stessa realizzazione in un combustore swirl confinato (conclusione indipendente dell'agente dataset, §7, punto 1);
  - i rig chiave (VT, Delft, PoliMi) non hanno dati digitali aperti individuati;
  - le statistiche rilasciate sono **già corrette o compensate** (HYLON, Waterloo, NIST TG).
- **Aggiunto:** HYLON è la configurazione più vicina per apparato e l'unica con un confronto quantitativo TC–Raman. Non migliora la documentazione del modello della sonda rispetto a Delft.
- **Circoscritto:** gli scarti TC–Raman di HYLON e le differenze 52/150 µm del NIST sono compatibili con più cause. Non vanno usati come misura di un singolo meccanismo.

## 12. Stato

- Il materiale di [step2_handoff/](step2_handoff/) è il passaggio di consegne scritto durante l'interruzione. Resta come traccia; i contenuti sono integrati qui e nel registro.
- **Non toccati:** step 1, roadmap, altre missioni, codice, risultati, `documents_thesis`.
