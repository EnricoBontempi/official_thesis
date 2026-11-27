# Blocco III — i 30 specialisti p_on: recupero, costo sul pulito e trasferimento

**Cosa è:** lettura verificata del pacchetto numerico dei 30 training specialistici (U-Net e GLU × solo p_on × seed 0/1/2 × regimi N1, N2, G1, G1-eq, N1G1), 27 novembre 2026. **Cosa non è:** una validazione PIV, un esperimento nuovo, o una decisione sulla missione 02. Scenari sintetici di classe B/C, non misure del combustore.

I numeri canonici stanno nel pacchetto ([results/nonideal/specialists/package/](../../results/nonideal/specialists/package/INDEX.md)); questo documento li legge e non li sostituisce. Le tabelle derivate dalla lettura (nessuna nuova valutazione) sono in [results/nonideal/specialists/reading/](../../results/nonideal/specialists/reading/).

## 1. Convenzione di segno e protocollo

- **Guadagno** = RMSE dell'originale − RMSE dello specialista, in K, stesso scenario e stesse osservazioni (stessi siti sopravvissuti, verificati per hash). **Positivo: lo specialista sbaglia meno.**
- **Costo sul pulito** = RMSE specialista − RMSE originale su R0. **Positivo: lo specialista è peggiore sul pulito.** Per costruzione, il guadagno su R0 è l'opposto del costo.
- Originali: i modelli p_on del Blocco II, stessa famiglia e stesso seed, non riaddestrati. Specialisti: da zero, un solo regime ciascuno, p_on, nessuna T in input.
- Checkpoint **best** = primo minimo assoluto della macro-RMSE sulla VAL **pulita** (invariato); **last** = ultima epoca, valutato come da protocollo. La metrica specialistica è solo diagnostica e non ha mai selezionato nulla. Nessun altro checkpoint è stato guardato.
- **seen_train_eval** = frame separati (220) nelle condizioni TRAIN: le condizioni di training, non i frame con cui il modello è stato allenato. **VAL** = 123 frame in tre condizioni, non usate per il training dello specialista. Le due letture sono riportate separate.
- Ogni cella è la media sulle tre repliche di maschera (stesse osservazioni, non training indipendenti). Le celle per seed mostrano **media (seed 0 / seed 1 / seed 2)**.

## 2. Verifica numerica della consegna

Controllo di sola lettura, senza inferenza: [scripts/71_nonideal_specialist_delivery_check.py](../../scripts/71_nonideal_specialist_delivery_check.py), esito in [delivery_check.md](../../results/nonideal/specialists/delivery_check/delivery_check.md). Tutti i controlli passano:

- 30 run attesi, tutti validi, nessun problema; SHA di training `b6fc9fef…` e di post-processing `b1c09550…` coerenti fra freeze, provenance delle 60 valutazioni e documenti.
- **60/60** valutazioni best/last: `gate.json` `all_pass` (5/5 confronti entro 0,05 K con il loop di training, massimo 0,0024 K), `COMPLETE.json` con 6 174 righe, pesi/epoca/regime uguali al freeze, pesi non modificati. Un solo hash della matrice di osservazioni, della cache e della specifica in tutte le 60.
- **720** righe in `per_run_scenario.csv` (30 × 2 × 6 × 2), chiavi uniche, valori finiti; guadagno, delta e costo sul pulito ricalcolati; colonne baseline identiche tra regimi e checkpoint per ciascun (famiglia, seed, cella, ruolo): 72 gruppi.
- Le 720 RMSE sono state **ricalcolate dai CSV per frame** (frame → condizioni → repliche) e coincidono con la tabella a 2·10⁻¹⁴ K.
- Hash dei 12 file del pacchetto uguali al manifest; hash del CSV per frame delle baseline uguale a quello atteso (`cb96687c…`).

## 3. Stato finale dei training

- 30/30 fermati dalla regola operativa di plateau (miglioramento relativo < 10⁻³ per 15 valutazioni); 0 `budget_exhausted`. Il plateau è una convenzione operativa, non una certificazione di convergenza.
- Epoche completate: U-Net 640–760 (mediana 690), GLU 680–1070 (mediana 890). Epoca del best: U-Net 70–310, GLU 110–560. In tutti i run il best arriva molto prima dello stop (390–570 epoche prima).
- Un solo run ha usato la continuazione 1000→1200: NI-glu-on-N1-s2 (job 30647863_0), concluso per plateau a 1070 epoche, best all'epoca 560 con VAL pulita 45,011 K. Nel complesso 34 segnali di preemption; riprese registrate nel ledger (non bitwise: vedi §7).

## 4. Risultati

### 4.1 Recupero sul proprio regime (guadagno sullo scenario di training)

Per ordine di grandezza, il degrado che gli originali subiscono (VAL, media sui seed, K): N1 0,2; G1-eq 1,3–2,0; N2 1,3–2,9; G1 4,1–4,3; N1G1 4,4–4,6 (da `baseline_deterioration.csv`). Un guadagno ha senso solo rispetto a questi valori e alla dispersione fra seed.

**Best** — guadagno medio (s0/s1/s2), K:

| famiglia | ruolo | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|---|
| U-Net | seen | -1.83 (+2.4/-7.1/-0.8) | -1.44 (+2.7/-5.5/-1.5) | +0.73 (+7.4/-2.9/-2.3) | +0.32 (+5.2/-3.7/-0.6) | +3.00 (+7.0/+1.3/+0.6) |
| U-Net | VAL | -0.85 (-0.8/-0.7/-1.0) | -2.65 (-2.3/-1.8/-3.8) | -1.88 (+0.5/-1.4/-4.8) | -0.42 (+2.1/-1.8/-1.5) | +0.29 (+3.2/-1.2/-1.2) |
| GLU | seen | +0.35 (+4.8/-4.1/+0.3) | -2.02 (-0.2/-6.9/+1.1) | -1.58 (+7.7/-1.6/-10.9) | +0.22 (+2.3/-0.2/-1.4) | -5.41 (+5.3/-21.6/+0.0) |
| GLU | VAL | -1.26 (+1.5/-3.9/-1.5) | +0.10 (-2.7/+0.3/+2.7) | -2.18 (+2.6/-7.0/-2.2) | -1.90 (-1.3/-4.2/-0.1) | -2.81 (-0.8/-9.5/+1.9) |

**Last** — guadagno medio (s0/s1/s2), K:

| famiglia | ruolo | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|---|
| U-Net | seen | +0.11 (+3.0/-0.8/-1.8) | -1.38 (+1.9/-2.9/-3.2) | +2.20 (+6.7/-0.7/+0.6) | -0.01 (+4.2/-1.9/-2.3) | +1.59 (+5.9/-0.0/-1.1) |
| U-Net | VAL | -2.25 (-2.2/-2.7/-1.9) | -5.34 (-5.1/-6.3/-4.5) | -4.51 (-1.9/-5.6/-6.1) | -2.00 (-1.1/-2.3/-2.6) | -1.97 (-0.7/-3.0/-2.2) |
| GLU | seen | +3.73 (+6.8/+0.5/+3.9) | +3.24 (+3.4/+2.1/+4.2) | +6.40 (+10.3/+3.0/+5.9) | +4.02 (+6.9/+1.4/+3.8) | +3.82 (+7.0/-2.0/+6.5) |
| GLU | VAL | -2.65 (-0.6/-5.2/-2.2) | -0.92 (-3.5/-1.4/+2.1) | -4.74 (+0.8/-10.1/-4.9) | -3.74 (-3.4/-6.3/-1.5) | -4.58 (-2.4/-11.9/+0.6) |

- **VAL, best:** negativo in 8 celle su 10 sulla media; i guadagni medi positivi sono solo U-Net N1G1 (+0,29) e GLU N2 (+0,10), entrambi più piccoli della dispersione fra seed. Contando i singoli run, 8 su 30 hanno guadagno positivo sullo scenario proprio.
- **seen, best:** misto. Positivi U-Net su G1 (+0,73), G1-eq (+0,32), N1G1 (+3,00, 3 seed su 3 positivi) e GLU N1/G1-eq (+0,35/+0,22); 14 run su 30 positivi.
- **Last:** su seen il GLU migliora in tutti e cinque i regimi (+3,2…+6,4, di solito 3 seed su 3); la U-Net poco. Su **VAL** il last è peggiore del best in tutte le 10 celle e solo 3 run su 30 hanno guadagno positivo.

### 4.2 Costo sul pulito R0 (specialista − originale)

**Best**, K:

| famiglia | ruolo | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|---|
| U-Net | seen | +1.83 (-2.4/+7.1/+0.8) | +2.64 (-1.4/+6.4/+2.9) | +2.96 (-2.5/+5.7/+5.6) | +0.51 (-3.7/+4.0/+1.3) | +1.21 (-1.7/+2.3/+3.1) |
| U-Net | VAL | +0.86 (+0.9/+0.6/+1.1) | +3.61 (+3.5/+2.3/+5.0) | +3.90 (+3.2/+2.0/+6.5) | +0.53 (-1.5/+1.2/+1.9) | +1.85 (+0.2/+2.2/+3.2) |
| GLU | seen | -0.31 (-4.8/+4.2/-0.3) | +4.37 (+2.0/+10.0/+1.2) | +4.98 (-2.8/+3.6/+14.1) | +0.51 (-1.3/+0.4/+2.4) | +8.53 (-0.7/+22.8/+3.5) |
| GLU | VAL | +1.32 (-1.4/+3.9/+1.5) | +2.48 (+4.8/+3.1/-0.5) | +4.09 (+1.1/+7.5/+3.6) | +2.35 (+2.1/+4.5/+0.4) | +4.83 (+4.3/+9.5/+0.7) |

**Last**, K:

| famiglia | ruolo | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|---|
| U-Net | seen | -0.01 (-2.8/+0.9/+1.9) | +3.08 (-0.0/+4.4/+4.9) | +2.05 (-1.6/+4.5/+3.2) | +1.01 (-2.7/+2.6/+3.1) | +3.02 (-0.3/+4.0/+5.3) |
| U-Net | VAL | +2.32 (+2.4/+2.7/+1.9) | +6.60 (+6.6/+7.4/+5.8) | +7.05 (+6.4/+6.9/+7.8) | +2.71 (+2.1/+2.8/+3.2) | +4.52 (+4.8/+4.4/+4.4) |
| GLU | seen | -3.69 (-6.8/-0.4/-3.9) | -0.86 (-1.7/+1.0/-1.9) | -2.81 (-5.3/-1.0/-2.2) | -3.17 (-5.6/-1.2/-2.8) | -0.45 (-2.2/+3.4/-2.5) |
| GLU | VAL | +2.69 (+0.7/+5.2/+2.2) | +3.56 (+5.6/+4.9/+0.1) | +7.34 (+2.6/+10.8/+8.6) | +4.23 (+4.2/+6.6/+1.9) | +7.05 (+6.2/+12.5/+2.4) |

Sul best, VAL, il costo medio è positivo in tutte le 10 celle (0,5–4,8 K). Il costo del last su VAL è più alto (2,3–7,3 K). Su seen il segno non è uniforme; il costo del last del GLU su seen è negativo, in contrasto con VAL.

### 4.3 Trasferimento agli altri scenari (guadagno medio su 3 seed, best)

Le righe sono il regime di addestramento, le colonne lo scenario valutato; la colonna R0 è il guadagno sul pulito (= −costo).

VAL:
U-Net, best, VAL (guadagno medio sui 3 seed, K)

| addestrato su \ valutato su | R0 | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|---|
| N1 | -0.9 | -0.9 | -0.7 | -1.3 | -1.0 | -1.3 |
| N2 | -3.6 | -3.5 | -2.6 | -3.4 | -3.6 | -3.2 |
| G1 | -3.9 | -3.8 | -3.4 | -1.9 | -3.2 | -1.8 |
| G1-eq | -0.5 | -0.5 | -0.1 | +0.6 | -0.4 | +0.7 |
| N1G1 | -1.9 | -1.7 | -1.1 | +0.1 | -1.1 | +0.3 |

GLU, best, VAL (guadagno medio sui 3 seed, K)

| addestrato su \ valutato su | R0 | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|---|
| N1 | -1.3 | -1.3 | -0.2 | -0.9 | -1.1 | -0.9 |
| N2 | -2.5 | -2.3 | +0.1 | -1.8 | -2.5 | -1.6 |
| G1 | -4.1 | -4.1 | -3.6 | -2.2 | -3.4 | -2.2 |
| G1-eq | -2.3 | -2.4 | -2.9 | -0.9 | -1.9 | -1.0 |
| N1G1 | -4.8 | -4.9 | -3.9 | -2.8 | -4.3 | -2.8 |

seen:
U-Net, best, seen (guadagno medio sui 3 seed, K)

| addestrato su \ valutato su | R0 | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|---|
| N1 | -1.8 | -1.8 | -1.6 | -2.4 | -2.0 | -2.3 |
| N2 | -2.6 | -2.5 | -1.4 | -2.4 | -2.5 | -2.2 |
| G1 | -3.0 | -2.8 | -2.1 | +0.7 | -1.8 | +0.9 |
| G1-eq | -0.5 | -0.5 | -0.1 | +2.1 | +0.3 | +2.1 |
| N1G1 | -1.2 | -1.1 | -0.2 | +2.8 | -0.1 | +3.0 |

GLU, best, seen (guadagno medio sui 3 seed, K)

| addestrato su \ valutato su | R0 | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|---|
| N1 | +0.3 | +0.4 | +0.9 | +0.8 | +0.6 | +0.8 |
| N2 | -4.4 | -4.2 | -2.0 | -4.3 | -4.3 | -4.1 |
| G1 | -5.0 | -4.9 | -4.1 | -1.6 | -3.9 | -1.5 |
| G1-eq | -0.5 | -0.6 | -0.9 | +1.2 | +0.2 | +1.1 |
| N1G1 | -8.5 | -8.5 | -7.3 | -5.5 | -7.6 | -5.4 |

- Le matrici sono dominate dalla **riga** (il regime di addestramento), non dalla colonna: il segno e l'ordine di grandezza di una riga sono simili su tutti gli scenari, R0 compreso. Lo specialista non si comporta in modo selettivo sullo scenario su cui è allenato.
- **U-Net, seen:** i regimi con lacune (G1-eq, N1G1, G1) migliorano sugli scenari con lacune a dischi G1 e N1G1 (+0,7…+3,0) e non sugli altri. Su VAL questo si riduce a +0,1…+0,7 (G1-eq e N1G1) e non è maggiore della dispersione fra seed.
- **N1 e N2:** su VAL nessun trasferimento utile e perdita sul pulito; su seen solo il GLU-N1 è marginalmente positivo (+0,3…+0,9).

### 4.4 Dispersione fra seed e best/last

- I tre seed sono visibili nelle celle (§4.1–4.2). Le differenze fra seed sono dello stesso ordine o più grandi dei guadagni medi: ad esempio GLU G1 su VAL, best: +2,6 / −7,0 / −2,2 K; GLU N1G1 su VAL: −0,8 / −9,5 / +1,9 K. Per la U-Net il seed 0 tende ad avere guadagni migliori degli altri due (VAL, best, N1G1: +3,2 contro −1,2 e −1,2 per i seed 1 e 2; G1-eq: +2,1 contro −1,8 e −1,5).
- Le baseline stesse differiscono per seed più dei guadagni tipici (RMSE R0 della VAL: U-Net 38,7/42,2/38,2 K; GLU 43,2/41,5/43,5 K; seen: U-Net 45,4/40,7/40,8 K). Il seed 0 della U-Net usa una baseline di lignaggio diverso (`O-C149-on-s0`, le altre `S7-C149-*`, come dichiarato nel contratto). Non l'ho verificato come causa dell'effetto: è solo un'ipotesi da tenere a mente.
- **La dispersione fra i tre seed è variabilità fra training e non una incertezza fisica calibrata.** Con tre seed non si stimano intervalli di confidenza.
- **Best contro last:** il last migliora sistematicamente seen e peggiora VAL e pulito, soprattutto nel GLU. La scelta del best (clean VAL) è quella prevista dal protocollo e non è stata cambiata dopo aver visto i risultati.

## 5. Cosa si può e non si può concludere

- **Nei dati, per questi regimi, questo protocollo di selezione e queste famiglie**, il ri-addestramento specialistico non mostra un recupero consistente su condizioni nuove (VAL), e in media peggiora il pulito in tutte le celle (U-Net e GLU, best). Non dimostra l'inutilità generale del retraining: un'altra selezione, altri regimi, un training misto o più dati potrebbero comportarsi diversamente (non testati qui, non autorizzati).
- **Il miglioramento su seen con il peggioramento su VAL (soprattutto last del GLU) non dimostra da solo sovradattamento.** Differenzia due insiemi di condizioni (TRAIN vs VAL), con maschere e valori diversi; le ipotesi plausibili — adattamento alle condizioni di training, differenza di difficoltà tra condizioni, effetto della selezione sulla VAL pulita, deriva numerica del batching GLU — non sono separate da questi dati. Il fatto che i last siano più «specializzati» sul seen e peggiori sul resto è coerente con la prima ipotesi, ma non la prova.
- **VAL non è una conferma indipendente:** è già stata usata per selezionare il best (VAL123 pulita) e per lo sviluppo del protocollo. Non è stato toccato il TEST.
- Le differenze sono confrontabili con la dispersione fra seed delle baseline (1–4 K): guadagni di entità minore non vanno interpretati.

## 6. Limiti del GLU

- Gli specialisti GLU sono allenati con microbatch 1 e accumulo a batch effettivo 8; gli originali con batch 8. Il confronto specialista-originale del GLU **mescola** effetto del regime, differenza numerica di batching e variabilità fra traiettorie. I test CPU provano la matematica del gradiente accumulato, non l'equivalenza delle traiettorie su GPU; lo smoke ha mostrato una differenza di identità superiore al rumore fra run identici (documento [04_specialist_smoke_criteria_and_budget_policy.md](04_specialist_smoke_criteria_and_budget_policy.md)). Il confronto è quindi **più pulito per la U-Net che per il GLU**. Un controllo con tre GLU puliti a microbatch 1 non è stato fatto e non è autorizzato.
- Le riprese dopo preemption non sono bitwise (nel ledger: fino a 9 tentativi e 8 segnali di preemption per un singolo run GLU, 34 segnali in totale). Il rumore può spostare di un'epoca la decisione di plateau, non misurato su modelli allenati.

## 7. Grafici e campi spaziali

**Prodotto, verificato.** Il job grafico originale (30685595, 2 s, exit 128) è fallito perché lo script ricavava il checkout dal percorso della copia che Slurm esegue dalla sua directory di spool: `git rev-parse HEAD` falliva prima di qualunque codice grafico. Il launcher ora richiede un checkout esplicito (`NI_GRAPHICS_ROOT`) e lo SHA atteso (`EXPECT_GRAPHICS_SHA`), con directory di lavoro e log espliciti ([scripts/engaging/nonideal_graphics_submit.sh](../../scripts/engaging/nonideal_graphics_submit.sh), test in `tests/test_nonideal_graphics_launcher.py`). Il tentativo fallito e quello riuscito sono tracciati in [graphics_hook_v2.json](../../results/nonideal/specialists/graphics_hook_v2.json); il file originale `graphics_hook.json` e i manifest scientifici non sono stati modificati.

Il job 30711670 (checkout `c5963373`, 105 s) è `COMPLETED 0:0` e `GRAPHICS_OK`; non mi sono fermato a Slurm: `COMPLETE.json` `ok`, 30 run, frame 1000; **179 file** del `figure_manifest.json` presenti con hash uguali (nessuno mancante), hash di freeze, tabella del pacchetto e baseline in `sources` uguali a quelli verificati sopra; i 30 `fields_manifest.json` hanno frame 1000, replica 0, ruolo VAL, 3 condizioni, 6 scenari e pesi uguali al freeze. Le cifre di `NOTE.md` (guadagno sul proprio regime) coincidono con le tabelle di §4.1.

- Nel repository: [results/nonideal/specialists/graphics/](../../results/nonideal/specialists/graphics/INDEX.md): 16 PNG di matrici (assoluti e differenze TRAIN × EVAL, seen/VAL, U-Net/GLU), dispersione dei tre seed, regioni (getti/zona calda/picco), `NOTE.md`, manifest. **Attenzione al segno:** le figure `matrix_delta_*` usano **specialista − baseline** (positivo = specialista peggiore, rosso), cioè l'opposto del guadagno di questo documento.
- Sul cluster (non pubblicati, 83 MB): `/home/bontempi/official_thesis_nonideal_spec_pkg/graphics/` con 30 schede PDF/PNG per run (best, VAL, frame 1000, replica 0, sei scenari × tre condizioni) e le schede di variabilità per famiglia e regime (media dei tre campi predetti, deviazione standard fra i tre training, errore del campo medio).

**Contratto dei campi spaziali (verificato nei file).**
- Specialisti: `fields_val_frame1000_best.npz`, best, **frame 1000**, replica 0, ruolo VAL, condizioni `0NH3_1.05`, `20NH3_1.1`, `40NH3_0.9`, 6 scenari, errore firmato = predizione − verità.
- Baseline (12 originali del Blocco II, `fields_figure_frames.npz`): 150 chiavi su 6 scenari; VAL a **frame 0** (stesse tre condizioni), seen a **frame 1600** (condizioni `0NH3_0.85`, `80NH3_1.2`); niente `error_signed_K`, `original_sites` né manifest; verificato sui sei modelli p_on.
- **I frame non coincidono (0 contro 1000 su VAL; il seen delle baseline non ha corrispettivo specialistico).** Quindi **non esiste alcun pannello campo-per-campo baseline contro specialista** e non ne è stato disegnato nessuno: sarebbero frame diversi presentati come appaiati. Mancano: i confronti spaziali appaiati su tutti e sei gli scenari e le tre condizioni VAL al frame 1000, e quelli seen. Completarli richiede una nuova inferenza degli originali al frame 1000 (elenco in [BASELINE_FIELDS_NEEDED.md](../../results/nonideal/specialists/graphics/BASELINE_FIELDS_NEEDED.md)): non fatta, non autorizzata.
- Non c'è conflitto con i confronti numerici: quelli usano i CSV per frame, con gli stessi frame, condizioni e siti sopravvissuti (hash uguali) per baseline e specialista.

## 8. Cosa può servire alla missione 02 (senza decidere o lanciare nulla)

- Il rumore a 0,5 m/s e le lacune a dischi non sono problemi equivalenti: il degrado (originali, VAL) è 0,2 K per N1 e circa 4 K per G1/N1G1. Per gli specialisti questo significa che N1 non lascia spazio di recupero misurabile.
- Come riferimento di entità, la scala di confronto più onesta è la dispersione fra seed (1–4 K) e il costo sul pulito (0,5–7 K): qualunque effetto di un modello di misura PIV più piccolo di questi va letto con la stessa cautela.
- Il quadro di partenza per la missione 02 è quello degli **originali** (Blocco II) più la matrice di trasferimento qui sopra: gli specialisti **non sono risultati un sostituto affidabile** degli originali con questo protocollo. Ogni eventuale retraining per la missione 02 andrebbe deciso dopo il modello di misura, come già previsto dalla sua roadmap, con gli stessi limiti (VAL già usata, tre seed, batching GLU).
- Le matrici sopra sono già sufficienti per discutere; i pannelli spaziali sono un complemento, non una condizione.

## 9. Non fatto

Nessun training, nessuna nuova valutazione numerica, nessun accesso al TEST, nessuna modifica di protocollo, soglie, selezione, baseline o metriche; nessun merge. Non sono stati riletti altri checkpoint oltre ai best/last del protocollo.
