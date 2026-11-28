# Missione 02 — Decisione R2

**Data:** 28 novembre 2026 · **Branch:** `mission02/piv-execution` · **Consegna esaminata:** `6660adbda3c695f4442c47ee2191890353215fe9`

**Letture:**
- [fase2_report_R2.md](fase2_report_R2.md) e [results/piv/phase2/INDEX.md](../../results/piv/phase2/INDEX.md);
- `configs/observations/piv_observation_spec_v1.yaml`: sha256 del file `9f24afc9…3d47`, ricontrollato. Ho letto le 31 configurazioni, i blocchi `error` e `rejection_sensitivity`;
- `src/combustion_recon/data/piv_observations.py`: rumore, rigetti, sostituzione `substitute_iterative_3x3`;
- figure A (mediana), B, N1;
- tabelle Neal (§6) e Cardiff (§7);
- sorgente dei due test preesistenti falliti.

**Stato:** R2 **chiusa con una correzione additiva della specifica (v1.1)**. Il modello principale e le 31 configurazioni di v1 restano invariati. Si aggiungono quattro sensibilità di sola valutazione, motivate da Neal e dall'XML di Cardiff. Nessun training, inferenza, profilazione o accesso a TEST in questa revisione.

---

## 1. In breve

| Tema | Decisione |
|---|---|
| Coerenza fra specifica, codice e proprietà misurate | **Confermata.** Le 31 configurazioni producono ciò che la specifica dichiara: σ realizzata/attesa 0,997–1,005, autocorrelazione entro 0,004 dall'analitica, kernel tarati, conteggi e finestre come da R1. Le figure non mostrano anomalie del generatore |
| Realismo | I test interni dimostrano l'esecuzione del modello, **non** il suo realismo. Il realismo resta quello dichiarato: classe GTMC, con trasferimenti e ipotesi elencati (§2) |
| Modello di correlazione (D-ρ) | **Principale invariato** (kernel MHG di rumore, L_sr = 1,2 W). Neal non lo conferma, ma il suo processamento e il suo riferimento non permettono di trasferire il risultato (§3). **SENS-RHO0 diventa obbligatoria** in ogni confronto della Fase 3 su A e B-in |
| Distribuzione dell'errore | Principale gaussiana, come ipotesi di modello. **Nuova sensibilità SENS-TAIL** (code pesanti, eccesso di curtosi 2) su A e B-in, solo valutazione (§4) |
| Sostituzione | 3×3 iterativa resta la regola di SENS-REJ-…-sub. **Nuova variante 5×5** (impostazione dichiarata nell'XML del processamento Cardiff), solo valutazione. Cardiff **non** è una validazione della regola (§5) |
| Test preesistenti falliti | Non bloccano F2.7 né la Fase 3. Vanno documentati prima della Fase 4 (§6) |
| Specifica | **v1.1 additiva** (§7). Verifiche da ripetere circoscritte (§8) |
| Via libera | **F2.7 subito.** **Fase 3** dopo v1.1 congelata, test di regressione e controllo dell'hash fra macchine (§8) |

---

## 2. Fedeltà delle osservazioni: che cosa mostrano figure e tabelle

**Coerenza specifica–implementazione.** Verificata sui numeri della consegna e sul codice:
- σ per condizione;
- gaussianità del rumore principale (asimmetria −0,001, curtosi +0,005);
- autocorrelazione del rumore uguale a quella del kernel discreto;
- stazionarietà fino ai bordi;
- conteggi A = 15 624 e B = 1 824;
- finestre B-in, B-down, B-off come in R1;
- assenza di T nella firma;
- pacchetto identico per U-Net e GLU.

Il rumore correlato si ottiene da rumore bianco sulla griglia estesa con `correlate1d` per asse, come dichiarato. La sostituzione 3×3 iterativa corrisponde a `rejection_sensitivity.levels.sub`.

**Figure.**
- **A (condizione mediana):** il filtro è visibile solo nel pannello ingrandito; l'errore domina la differenza osservato − ideale (circa 1,03 σ_u in A).
- **B (zoom sul gradiente più forte):** profilo osservato aderente all'ideale entro ±σ_u.
- **Un'osservazione nuova, rilevante per interpretare la Fase 3.** La differenza filtrato − ideale non è distribuita lungo gli strati di taglio, ma concentrata in **macchie isolate di una o poche celle**, visibili anche come piccoli punti nei campi LES ideali e come contorni "a gradino" nello zoom. È coerente con un passaggio dalla mesh LES alla griglia 128×128 di ordine basso o per celle (metodo non dichiarato: missione 01, Q1) e con lo spettro liscio misurato in F1.2. È un'interpretazione delle figure, non una verifica.
  - Una parte dell'effetto del filtro su questa LES rimuove quindi **artefatti di ricampionamento**, non scale turbolente fisiche.
  - Non cambia il modello di misura: una PIV reale filtrerebbe le piccole scale fisiche, che la LES non contiene. Ma impone una lettura: **l'effetto del filtro misurato nella Fase 3 non è una stima dell'effetto della risposta PIV in una fiamma reale.**

**Giudizio sul realismo.** Le osservazioni rappresentano in modo coerente le proprietà PIV sostenute dalle fonti: campo, griglia, risposta, ampiezza dell'errore fissata dal progetto della misura, dato rilasciato completo. Le proprietà più incerte (correlazione e forma dell'errore, rigetti) sono già coperte o, con v1.1, saranno coperte da sensibilità. Non vedo un motivo fisico per cambiare il modello principale.

## 3. Modello di correlazione (D-ρ)

**Che cosa mostra Neal** (§6 del report, figura N1).
- I residui MS − HDR sono **anisotropi**: lungo x, che è la direzione del flusso, ρ(0,5 W) = 0,41–0,45; lungo y 0,13–0,18 (16 px, 50 %).
- Il modello dà 0,68 al primo ritardo in entrambe le direzioni; a ritardo W le misure (0,18–0,26 lungo x, da −0,11 a +0,01 lungo y) racchiudono il valore del modello (0,15).

**Perché il risultato non si trasferisce automaticamente al modello principale.**
1. **Processamento diverso.** OpenPIV a passata singola con finestre quadrate. Il modello di riferimento descrive DaVis multi-passata con finestre gaussiane (WIE17), che è l'algoritmo del GTMC. Le due catene hanno risposta e correlazione del rumore diverse per costruzione.
2. **Il residuo non è il solo errore della misura MS.** Contiene l'errore e la risposta dell'HDR, che è una media su 5 coppie, outlier interpolati e ricampionamento a 1 px, più il disallineamento temporale in un getto pulsante.
   - L'anisotropia, con correlazione maggiore lungo il flusso, è tipica di un contributo **convettivo** del disallineamento temporale, non della sovrapposizione delle finestre, che è isotropa.
   - Lungo y la correlazione misurata a 0,5 W (0,13–0,18) è più bassa anche di quella attesa dalla sola sovrapposizione al 50 % di una finestra quadrata a passata singola (0,5 per un top-hat). Il residuo contiene quindi una parte importante **non correlata**, non attribuibile alla sovrapposizione.
3. **Getto freddo non reagente:** nessuna trasferibilità alla fiamma (P11, roadmap §2.7 punto 4).

**Che cosa non ignoriamo.** Una PIV reale elaborata in un altro modo può avere una correlazione al primo ritardo molto più bassa del modello. È un'evidenza sperimentale, anche se di un altro processamento.

**Decisione.**
- Modello principale invariato: kernel di rumore MHG, L_sr = 1,2 W, coerente con la sorgente del processamento GTMC.
- **SENS-RHO0** (errore indipendente) diventa una sensibilità **obbligatoria**, riportata accanto al principale in ogni confronto della Fase 3 su A e B-in. I casi misurati da Neal (correlazione al primo ritardo fra 0,13 e 0,45) stanno fra SENS-RHO0 e il principale. La coppia delimita il comportamento senza introdurre un terzo kernel tarato su un processamento che non è il nostro.
- **Non aggiungo un kernel "anisotropo Neal":** l'anisotropia osservata non è attribuibile alla misura (punto 2).
- Il criterio D-ρ della roadmap ("rivedere il kernel se incompatibile") è chiuso così: incompatibilità al primo ritardo, con un processamento diverso dal riferimento → nessuna revisione del principale, sensibilità obbligatoria.

## 4. Distribuzione dell'errore

**Evidenza.** Dopo l'esclusione degli outlier (5 MAD), i residui Neal hanno eccesso di curtosi 1,6–2,8. Valgono gli stessi limiti del §3: processamento a passata singola senza validazione, riferimento con errore proprio. La gaussianità del principale è un'ipotesi di modello e lo resta. Le code pesanti di Neal non si trasferiscono come proprietà della PIV GTMC validata, ma indicano che una distribuzione a code pesanti è sperimentalmente plausibile.

**Decisione: nuova sensibilità SENS-TAIL** (solo valutazione, A e B-in).

| Voce | Definizione |
|---|---|
| Costruzione | Si parte dal campo gaussiano correlato **standardizzato** della configurazione PIV corrispondente (stessa realizzazione, stessi nodi). Si applica nodo per nodo la trasformazione monotona z → F_t⁻¹(Φ(z); ν), poi si riscala a varianza unitaria (divisione per √(ν/(ν−2))) e si moltiplica per σ_u,c |
| Parametro | ν = 7, cioè eccesso di curtosi marginale 6/(ν−4) = 2: valore centrale dell'intervallo Neal 1,6–2,8. **Non calibrato** per la fiamma |
| Che cosa conserva | σ_u, ordine dei valori (correlazione di rango) e struttura spaziale. La correlazione di Pearson cambia leggermente e va misurata e riportata (§8) |
| Che cosa non è | Un modello degli outlier non validati: i vettori spuri sono rappresentati dalle sensibilità SENS-REJ |

## 5. Sostituzione

**Evidenza.**
- La regola 3×3 iterativa è una pratica documentata: interpolazione 3×3 nella microPIV TUM; buchi di un solo vettore interpolati nel GTMC.
- L'XML del processamento Cardiff **dichiara** una media mobile su un'area 5×5, con 3 iterazioni e `substitute = True` (scheda PIVD:S2D-01). È documentazione del software, indipendente dal decoder.
- Il confronto quantitativo con i valori sostituiti reali (5×5 più vicina della 3×3, ma nessuna media semplice riproduce i valori reali) dipende invece dal **decoder non ufficiale, non validato contro la statistica DynamicStudio**: il confronto non è stato eseguibile. **Non è una validazione** di nessuna delle due regole.

**Decisione.**
- SENS-REJ-…-sub continua a usare la **3×3 iterativa** (nessun cambiamento in v1).
- Si aggiungono **SENS-REJ-iso-sub5-A** e **SENS-REJ-clu-sub5-A**: stessi rigetti (stesse posizioni) di `iso-sub` e `clu-sub`, sostituiti con la **media iterativa dei vicini noti in un'area 5×5**. Le passate si ripetono finché tutti i rigettati sono riempiti, come la 3×3; si registra se servono più di 3 passate.
- La coppia 3×3/5×5 delimita la scala di lisciatura dei valori sostituiti fra due pratiche documentate.
- Le frazioni Cardiff (29,8 % di codice 16 ecc.) restano una **descrizione di quel dataset isotermo**. Non entrano in nessun parametro.
- La validazione del decoder contro `StatisticsVec#0` resta aperta. **Non è richiesta per proseguire**, perché nessun parametro della specifica dipende dal decoder.

## 6. Test preesistenti falliti e problemi residui

| Problema | Esame | Blocca? |
|---|---|---|
| `test_opus_c149_cache_contract::test_trainer_passes_contract_into_every_dataset` | È un controllo **sul testo sorgente**: si aspetta 5 occorrenze di `**cache_kwargs` in `scripts/16_train_sparse_scientific.py`, che ne contiene 6 (verificato). La sesta è stata introdotta dal commit dei specialisti `06ee61f`, che aggiunge un dataset. Non è un errore del generatore PIV | **No** per F2.7 e Fase 3. **Prima della Fase 4**, Sonnet verifica che la sesta occorrenza passi il contratto della cache al dataset aggiunto e documenta l'esito. Se è corretto, aggiorna l'attesa del test in un commit separato e motivato |
| `test_sparse_eval_masks::test_baseline_consolidation_join_5pct` | Confronta un join storico del Mese 2 con artefatti locali; fallisce anche sul commit `d4fecb2`. La Fase 3 usa invece il **gate di replay E0** contro i CSV per frame della 00, che verifica direttamente valutazione e maschere storiche | **No.** Sonnet riporta il messaggio d'errore nel rapporto della Fase 3 |
| Hash delle osservazioni in float64 (ultima cifra può cambiare fra macchine) | La Fase 3 girerà sul cluster, la qualifica è stata fatta in locale | Controllo circoscritto prima della Fase 3 (§8, punto 4) |
| Allineamento HDR–coppia di Neal (+1 file) | Influenza poco i risultati; riguarda solo l'interpretazione del §3 | No |
| Decoder Cardiff non validato; maschera non verificata | Nessun parametro ne dipende (§5) | No |
| Orientamento *probabile*; domanda agli autori aperta | Coperto da SENS-SWAP | No |

## 7. Specifica v1.1 (correzione tracciabile)

**Natura:** additiva. Nessun valore di v1 cambia. `spec_version: 1.1`, nuovo file `configs/observations/piv_observation_spec_v1_1.yaml` con sha256 proprio. **v1 resta nel repository** e non si sovrascrive.

**Delta rispetto a v1:**

| Voce | Contenuto |
|---|---|
| `configs` | + `SENS-TAIL-A`, `SENS-TAIL-B-in` (`noise: corr_t`, `t_nu: 7`, gli altri campi come `PIV-A` e `PIV-B-in`); + `SENS-REJ-iso-sub5-A`, `SENS-REJ-clu-sub5-A` (come `iso-sub` e `clu-sub`, con `substitution: mean5x5_iterative`) |
| `error` | + blocco `tail_sensitivity`: costruzione del §4, ν = 7, motivazione (Neal, non calibrato), classificazione *sensibilità* |
| `rejection_sensitivity` | + `substitution_rules`: `mean3x3_iterative` (principale per `sub`, TUM/GTMC) e `mean5x5_iterative` (variante, area dichiarata nell'XML Cardiff, non validata) |
| `parameter_classification` | + le tre nuove voci; aggiornamento della voce "correlazione dell'errore" con l'esito di D-ρ (Neal: non confermata al primo ritardo con un processamento diverso; principale invariato; SENS-RHO0 obbligatoria) |
| `roadmap` / `provenance` | Riferimento a `decisione_R2.md`; SHA del codice che scrive v1.1; sha256 di v1 come genitore |
| Seed | Nessun nuovo stream: SENS-TAIL riusa la realizzazione standardizzata della configurazione PIV; sub5 riusa i rigetti di `iso` e `clu` |

**Le 31 configurazioni di v1 devono dare in v1.1 pacchetti identici** (stessi nodi, stessi valori, stesso hash nello stesso ambiente).

## 8. Che cosa Sonnet deve fare, e in quale ordine

1. **F2.7 — Profilazione GPU: autorizzata subito.** Non dipende dalle sensibilità nuove; usa PIV-A, PIV-B-in e IDEAL-A di v1, invariati in v1.1.
2. **F2.8 — Implementazione di v1.1** (CPU):
   - scrivere v1.1 con lo script della specifica, senza modificare v1;
   - estendere il generatore con `corr_t` e `mean5x5_iterative`, senza cambiare il comportamento delle configurazioni esistenti.
3. **F2.9 — Verifiche da ripetere** (circoscritte; nessun F2.6 completo):
   - **regressione:** per le 31 configurazioni v1, pacchetti v1.1 bit-identici a v1 sugli stessi input (frame, ruolo, replica), nello stesso ambiente;
   - **test nuovi:**
     - SENS-TAIL: varianza unitaria entro 1 %, eccesso di curtosi marginale 2 ± 0,2 su un campione grande, correlazione di rango identica alla configurazione PIV, stessi nodi;
     - sub5: stesse posizioni rigettate di `sub`, ogni valore sostituito uguale alla media dei vicini noti 5×5 della passata corrente, numero di passate registrato;
     - `pytest tests/test_piv_observations.py` tutto verde;
   - **qualifica supplementare** sui 760 frame `train_fit`:
     - SENS-TAIL: σ realizzata/attesa, asimmetria, curtosi, frazione con |z| > 3, autocorrelazione di Pearson per asse (ritardi 1–4) confrontata con PIV-A e PIV-B-in;
     - sub5: RMS(sostituito − non rigettato) per regione, accanto alla 3×3;
     - una figura per ciascuna;
     - aggiunta al rapporto come appendice R2 (`fase2_report_R2.md`, §5.9), senza riscrivere il resto.
4. **Controllo dell'hash fra macchine** (prima della Fase 3): sul cluster, generare PIV-A, PIV-B-in, IDEAL-A e FILT-A per 3 condizioni × 2 frame × 1 replica e confrontarli con gli stessi pacchetti generati in locale. **Criterio:** nodi identici e differenza massima dei valori ≤ 10⁻⁹ m/s. Se il criterio è soddisfatto, la Fase 3 usa confronti a tolleranza, non l'uguaglianza dell'hash.
5. **Fase 3: autorizzata dopo i punti 2–4**, secondo la roadmap. Specificazioni:
   - **SENS-RHO0** riportata obbligatoriamente accanto al principale (A e B-in);
   - SENS-TAIL e le due `sub5` si aggiungono all'elenco delle sensibilità di valutazione;
   - i risultati sul filtro (E1 − E0v, E4 − E3a) si leggono con la cautela del §2: una parte del filtro rimuove artefatti di esportazione della LES.
6. **Prima della Fase 4:** documentazione del primo test fallito (§6).

Se una verifica del punto 3 fallisce, Sonnet si ferma e la riporta. Non adatta i parametri per farla passare.

## 9. Incertezze residue

- **Correlazione dell'errore:** il principale viene da metrologia sintetica DaVis. L'unica verifica sperimentale disponibile (Neal) usa un altro processamento e non lo conferma al primo ritardo. Coperta da SENS-RHO0.
- **Forma dell'errore:** gaussiana per ipotesi; SENS-TAIL per code pesanti non calibrate.
- **Rigetti e sostituzioni in fiamma:** frequenza e morfologia ignote. Il dato rilasciato completo non ne dimostra l'assenza. SENS-REJ (3×3 e 5×5) ne esplora alcune conseguenze.
- **Contenuto della LES:** campo liscio a 2–3 mm, con artefatti di sottocampionamento. L'effetto del filtro non è trasferibile a una fiamma reale.
- **Orientamento e assegnazione delle coordinate:** *probabili*. La domanda agli autori resta utile.
