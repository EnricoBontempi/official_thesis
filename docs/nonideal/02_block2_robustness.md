# Blocco II — robustezza dei 12 modelli congelati a osservazioni di velocità degradate

**Data:** 24 novembre 2026 · **Specifica:** `configs/nonideal/degradation_spec_v1_12models.yaml` · **Codice eseguito:** `12ceee6064b228e0014d1e7fecd7797d3b2d5dd4`
**Pacchetto (indice):** [results/nonideal/block2/package/INDEX.md](../../results/nonideal/block2/package/INDEX.md) · **CSV per frame (pesante, sul cluster):** `/home/bontempi/official_thesis_nonideal_b2_pkg/package/tables/per_frame_all.csv.gz` (SHA256 `cb96687c…070b3`); output per modello in `/home/bontempi/official_thesis_nonideal_b2_eval/<run_id>/`.

**Natura dello studio.** Sensibilità descrittiva di reti allenate su osservazioni ideali a degradazioni **sintetiche** (σ = 0.5 e 1.5 m/s; lacune a dischi al 25 % dell'area): scenari motivati, non una validazione PIV né misure del nostro combustore. VAL123 è già stata usata per selezione e sviluppo: non è conferma indipendente. Nessun test di significatività, nessuna soglia di successo. Tre seed e tre repliche di maschera non sono campioni indipendenti di un training.

## 1. Che cosa è stato eseguito e verificato
- 12 checkpoint `best` (U-Net/GLU × p_on/p_off × seed 0/1/2), 6 celle, 3 repliche, seen220 + VAL123: **74 088** righe scientifiche, 0 chiavi duplicate. In più **6 174** righe di replay tecnico (GLU, R0 a batching S7), contate a parte.
- Job: array 30388340 (12/12 COMPLETED, 5.6–10.2 min per task, L40S), verifica 30388341, aggregazione 30388342 (exit 0). Nessun `CAMPAIGN_HALT`; nessun errore nei log. I BLOCK_REPORT dei job smoke 30388027/28 sono previsti e irrilevanti.
- **Gate R0 (≤ 0.05 K macro per run/ruolo/replica contro S7): 12/12 passati.** Scarto macro massimo: 0 K (U-Net, bit-identico), 0.0009–0.0028 K (GLU). Per frame: U-Net ≤ 2·10⁻⁵ K; GLU media dei massimi 0.008–0.012 K, massimo per frame 0.045–0.094 K, tutti i 343 frame appaiati per ruolo/replica. Il confronto per frame supera 0.04 K in 5 GLU su 6: conferma che 0.04 K non era un limite universale; il criterio che decide è il macro.
- GLU batch 1 contro batching S7: massimo per frame 0.045–0.094 K (sensibilità numerica alla forma del batch, già documentata).
- p_off: i canali p della U-Net sono esattamente zero; il GLU p_off ignora p (variazione 0.0). p_on: p modifica la predizione.
- Ricalcolo indipendente dei macro (da `per_frame_all.csv.gz`, codice separato dall'aggregatore): massimo scarto 1.4·10⁻¹⁴ K su 144 macro (12 modelli × 2 ruoli × 6 celle).
- Strutture verificate dai controlli: R0/N1/N2 senza siti rimossi; G1, N1G1 e G1-eq con lo stesso numero di siti rimossi per frame/replica; N1G1 con gli stessi superstiti di G1; stessi siti originari per tutte le celle. G1: area effettiva 25.0–26.0 %, 167–243 siti rimossi su 819 (media 207), 0 rifiuti.

## 2. Risultati (RMSE macro: frame → condizione → media uniforme → repliche)
**seen220** — RMSE macro in K, media sui 3 seed (e tra parentesi Δ medio rispetto al proprio R0, e Δ%)

| modello | R0 | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|---|
| U-Net p_on | 42.32 | 42.62 (+0.30; +0.7%) | 44.91 (+2.59; +6.1%) | 47.74 (+5.42; +12.8%) | 43.81 (+1.48; +3.5%) | 48.14 (+5.82; +13.7%) |
| U-Net p_off | 44.77 | 45.09 (+0.32; +0.7%) | 47.92 (+3.15; +7.0%) | 50.98 (+6.21; +13.9%) | 46.88 (+2.11; +4.7%) | 51.45 (+6.68; +14.9%) |
| GLU p_on | 50.07 | 50.33 (+0.26; +0.5%) | 53.36 (+3.30; +6.6%) | 57.60 (+7.53; +15.0%) | 52.35 (+2.29; +4.6%) | 57.85 (+7.78; +15.5%) |
| GLU p_off | 53.34 | 53.65 (+0.30; +0.6%) | 58.08 (+4.74; +8.9%) | 61.47 (+8.12; +15.2%) | 55.84 (+2.50; +4.7%) | 61.74 (+8.40; +15.8%) |

**VAL123** — RMSE macro in K, media sui 3 seed (e tra parentesi Δ medio rispetto al proprio R0, e Δ%)

| modello | R0 | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|---|
| U-Net p_on | 39.73 | 39.89 (+0.17; +0.4%) | 41.06 (+1.34; +3.4%) | 43.84 (+4.12; +10.5%) | 41.74 (+2.02; +5.1%) | 44.09 (+4.36; +11.1%) |
| U-Net p_off | 40.59 | 41.12 (+0.52; +1.3%) | 43.97 (+3.38; +8.3%) | 46.20 (+5.61; +13.8%) | 44.19 (+3.60; +8.9%) | 46.63 (+6.04; +14.9%) |
| GLU p_on | 42.73 | 42.91 (+0.18; +0.4%) | 45.67 (+2.94; +6.9%) | 47.07 (+4.34; +10.1%) | 44.02 (+1.29; +3.0%) | 47.30 (+4.57; +10.7%) |
| GLU p_off | 44.49 | 44.59 (+0.10; +0.2%) | 48.63 (+4.15; +9.3%) | 50.04 (+5.55; +12.5%) | 46.00 (+1.51; +3.4%) | 50.38 (+5.89; +13.3%) |
Tutti i 12 modelli peggiorano in tutte le celle, su entrambi i ruoli, a livello macro (3/3 seed). A livello di singola condizione (460 combinazioni modello × condizione × cella × ruolo) 9 migliorano, tutte di poco (N1: 7, N2: 1, G1-eq: 1).

### 2.1 Rumore, lacune, combinazione
- **N1 (0.5 m/s):** +0.1…+0.5 K (0.2–1.3 %). Trascurabile in tutti i modelli; per valore assoluto non cambia nulla di apprezzabile.
- **N2 (1.5 m/s, stessa realizzazione ×3):** +1.3…+4.7 K (3.4–9.3 %). Triplicare σ moltiplica l'effetto di circa 8–40 volte (rapporto N2/N1 dei Δ medi): la risposta è fortemente non lineare, quindi N1 è lontano dal regime in cui il rumore conta.
- **G1 (lacune a dischi, ~207 siti persi):** +4.1…+8.1 K (10–15 %), la perdita più grande in ogni modello.
- **N1G1:** +4.4…+8.4 K; ≈ G1 + 0.2–0.5 K. Interazione `N1G1 − N1 − G1 + R0` tra −0.09 e +0.23 K: effetti **additivi**, il rumore a 0.5 m/s non aggrava le lacune.
- **G1-eq (stesso numero di siti persi, casuali):** +1.3…+3.6 K (3–9 %).

### 2.2 U-Net contro GLU (deterioramento, non errore assoluto)
Il GLU parte più alto (R0: 42.7–53.3 K contro 39.7–44.8 K) e si deteriora di più in valore assoluto su seen (G1: +7.5/+8.1 K contro +5.4/+6.2 K; N2: +3.3/+4.7 contro +2.6/+3.2). Relativamente le differenze sono modeste e non coerenti fra i ruoli: G1 seen 15.0/15.2 % (GLU) contro 12.8/13.9 % (U-Net); G1 VAL 10.1/12.5 % (GLU) contro 10.5/13.8 % (U-Net). Per N2 il GLU è relativamente peggiore (6.6–9.3 % contro 3.4–8.3 %). Non si dice quindi che un modello sia più robusto perché parte da un errore alto o basso: con questi dati la differenza di robustezza fra famiglie è piccola e dipende da ruolo e scenario.

### 2.3 p_on contro p_off
- A R0, p_on è migliore di p_off: −3.3 K (GLU seen), −2.5 (U-Net seen), −1.8 (GLU VAL), −0.9 (U-Net VAL).
- Deterioramento (p_on − p_off, media dei tre confronti per indice di seed, descrittivo: i seed hanno inizializzazioni diverse): G1 −0.6…−1.5 K, N1G1 −0.6…−1.7, N2 −0.6…−1.7 (GLU seen −1.5, U-Net VAL −1.7), G1-eq −0.2…−1.6, N1 ≈ 0. **Quindi conoscere phi e NH₃ attenua il peggioramento** in media, di circa 0.6–1.7 K (media sui seed), ma **non in modo uniforme**: per G1/N1G1 su seen il massimo dei tre confronti è positivo (+0.6…+0.8 K), cioè in qualche coppia di seed p_on peggiora di più. Sull'attenuazione di N1 non c'è nulla da dire (effetto nullo).

### 2.4 G1 contro G1-eq: contiguità o numero di siti?
G1 − G1-eq è positivo in tutti i 12 modelli su entrambi i ruoli: +2.0…+5.6 K. G1-eq riproduce solo il 26–37 % del deterioramento di G1 per i GLU e per la U-Net su seen; per la U-Net su VAL il 44–75 %. In sintesi:
- Per GLU e per la U-Net su seen, circa **due terzi** del danno delle lacune dipende dalla **contiguità** (zone senza osservazioni), un terzo dal solo numero di siti persi.
- Per la U-Net su VAL il numero di siti pesa di più (circa metà; per p_off 64 %), dominato dalla condizione `40NH3_0.9`, dove G1-eq (+9.3 K) è quasi uguale a G1 (+9.8 K): lì conta soprattutto la perdita di densità, coerente con la sensibilità della U-Net alla statistica del canale di validità già ipotizzata nel Mese 2. Sulla U-Net VAL G1 produce inoltre uno spostamento di livello (bias +Δ: −8.6 K per p_off, −0.5 K per p_on): non è decomposto ulteriormente.
- L'RMSE dentro le regioni G1 è più alta che fuori (GLU seen p_on 65.3 contro 54.0 K), ma per la U-Net VAL il bias è uguale dentro e fuori (−7…−17 K): l'errore è globale, non confinato alle lacune.

### 2.5 Livello contro struttura
Rumore a 0.5 m/s: ≤ 0.5 K. Rumore a 1.5 m/s: 1.3–4.7 K, dello stesso ordine della rimozione casuale del 25 % dei siti (1.3–3.6 K) ma inferiore alle lacune contigue (4.1–8.1 K). La struttura (contiguità) pesa più del livello di rumore testato.

### 2.6 seen contro VAL e dispersione dei seed
- seen220 si deteriora generalmente di più di VAL123 per G1 (GLU 7.5–8.1 contro 4.3–5.6 K) e per N2 della U-Net p_on (2.6 contro 1.3 K); eccezione: G1-eq U-Net p_off (2.1 contro 3.6 K) per la condizione sopra. seen e VAL sono diversi per condizioni e numero di frame (20 contro 3 condizioni): il confronto è descrittivo.
- Dispersione fra i tre seed del Δ: N1 ≤ 0.2 K; G1 fino a 2.7 K (GLU p_on seen 5.84–8.49; GLU p_off VAL 4.49–6.86; U-Net p_on seen 4.86–6.46); N2 fino a 1.3 K (GLU p_on seen 2.73–4.05). Variazioni fra seed paragonabili alle differenze fra p_on e p_off: nessuna conclusione di ordinamento fra modelli dai singoli seed.
- Condizioni più colpite da G1 su seen: `80NH3_1.05`, `40NH3_1.2`, `80NH3_1.15` (+11…+12 K in media); meno colpite `0NH3_0.9`, `20NH3_0.85`, `0NH3_1` (+2.4…+3.7 K). Su VAL, `40NH3_0.9` è la più sensibile (U-Net p_off G1 +9.8 K, N2 GLU p_off +7.4 K).
- Getti freddi: con G1 l'RMSE nei getti (T < 1800 K) cresce di +14.9…+22.5 K su seen e +8.3…+14.9 K su VAL (medie sui seed), molto più dell'RMSE della zona calda (+1.5…+4.6 K) e del picco (+2.2…+8.3 K); IoU della zona calda −0.008…−0.088. Il danno delle lacune si concentra nei getti.

## 3. Quali configurazioni meritano considerazione per il retraining (descrittivo, non decisione)
Nessuna soglia automatica: elenco ragionato per Enrico.
- **Più rilevanti:** lacune contigue (G1 e N1G1): effetto più grande, uniforme su tutti i 12 modelli e su entrambi i ruoli (+10–16 %), in gran parte dovuto alla contiguità e non al solo numero di siti; per i GLU l'effetto assoluto è il più alto. Il retraining con lacune contigue (generatore in un namespace TRAIN diverso) è l'ipotesi con più materiale; il trasferimento a una geometria di lacuna non vista (EngineBench-like) resta una domanda distinta.
- **Da considerare con cautela:** rumore a 1.5 m/s (N2): effetto modesto (3–9 %) ma non lineare e più marcato nei GLU e nei p_off; è un'intensità ipotetica (classe C), quindi motivo debole da solo.
- **Reggono già bene:** rumore a 0.5 m/s (N1, ≤ 1.3 %), nessun margine di recupero; rimozione casuale dei siti (G1-eq, 3–9 %); la combinazione N1G1 non aggiunge nulla oltre G1. I modelli p_on perdono in media meno dei p_off nelle lacune e in N2, quindi la conoscenza di phi/NH₃ è un fattore utile ma non sufficiente a risolvere il problema delle lacune.
- **Attenzione specifica:** la U-Net p_off su VAL (`40NH3_0.9`) è sensibile anche alla sola perdita di siti: un eventuale retraining dovrebbe distinguere un effetto di densità da uno di geometria (il controllo G1-eq già lo permette).
- **Da risolvere prima di qualunque retraining:** gestione del numero variabile di token nel GLU (troncamento al minimo del batch non deciso); decisione di Enrico su famiglie, celle e fonte della distribuzione delle lacune.

## 4. Limiti
- Degradazioni sintetiche di classe B/C (σ e geometria sono default di scenario); nessuna PIV né incertezza qualificata del combustore.
- Il target è la T numerica LES; VAL123 usata per selezione; 3 seed, 3 repliche (maschere/rumore/regione variano insieme), 20 + 3 condizioni; seed di famiglie e p diversi non sono appaiati in senso causale.
- I modelli sono allenati al 5 % casuale: G1 misura anche una condizione fuori distribuzione.
- Metriche regionali (dentro/fuori G1) non hanno un R0 corrispondente nella stessa area: non sono deterioramenti.
- Non sono stati modellati media spaziale, campionamento regolare, lacune dipendenti da T, spuri, T osservata.
- Nessun TEST, nessun training, A/B/C sospese.
