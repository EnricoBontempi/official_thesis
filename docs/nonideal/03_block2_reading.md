# Blocco II — lettura per la discussione

Stress test sintetici dei 12 modelli già valutati (U-Net/GLU × con o senza φ e NH3 × seed 0/1/2), su seen220 e VAL123. Nessuna temperatura in ingresso, nessuna termocoppia, nessuna correzione a posteriori. La temperatura LES è solo il riferimento. Non è una validazione PIV. VAL123 è già servita a selezionare i checkpoint; seen220 non è un insieme di condizioni nuove.

I numeri sotto sono ricalcolati dalle tabelle aggregate del pacchetto, non da una nuova valutazione. Il report [02_block2_robustness.md](02_block2_robustness.md) resta la fonte della campagna. Questa nota aggiunge la decomposizione bias / centered-RMSE, il confronto fra lacune contigue e siti tolti a caso, il danno regionale e i casi lontani dalla media. Figure e indice: [results/nonideal/block2/reading/INDEX.md](../../results/nonideal/block2/reading/INDEX.md).

## Come sono stati controllati i numeri

**Ricalcolato** da `absolute_seed_summary.csv`, `delta_vs_R0_per_model.csv`, `delta_vs_R0_by_condition.csv`, `structure_contrasts.csv`, `regional_g1.csv` e `pon_minus_poff.csv`. Le 48 RMSE medie del report (4 gruppi × 6 celle × 2 ruoli) coincidono entro 0,02 K. A livello famiglia × φ/NH3, dopo la media sui seed e sulle repliche di maschera, i contrasti condizione × cella × ruolo sono 460; 9 migliorano rispetto al proprio caso pulito (7 con rumore 0,5 m/s, 1 con rumore 1,5 m/s, 1 con rimozione casuale), tutti di poco.

**Soltanto letto**, non rieseguito: `VERIFY_OK.json` (12/12, nessun problema), `COMPLETE.json`, il conteggio di 74 088 righe e il ricalcolo per-frame citato nel report. Il CSV per frame e i campi `fields_figure_frames.npz` non sono in questo repository.

Ogni punto delle figure nuove è un seed di training, già mediato sulle tre repliche di maschera. Le repliche muovono insieme maschera, rumore e regione: non sono training indipendenti. Non ci sono test di significatività. I frame sono correlati.

Definizioni, già nel codice di valutazione. Il bias è l'errore medio con segno, predizione meno verità. La centered-RMSE è la RMSE dopo aver tolto quel bias. Sul singolo frame vale RMSE² = bias² + centered-RMSE². Sulle medie di frame, condizioni e repliche l'identità non torna: lo scarto fra la RMSE media e la radice di bias² + centered-RMSE² è di diversi kelvin già nel caso pulito. Le due serie non si sommano e non sono una percentuale della RMSE. I getti sono i voxel con temperatura vera sotto 1800 K; la zona calda è T ≥ 2100 K; il picco è T ≥ 2200 K.

## Cosa regge

Il rumore di 0,5 m/s sposta la RMSE macro di +0,1…+0,5 K (0,2–1,3 % del proprio caso pulito). Tutti i 12 modelli peggiorano comunque su tutte le celle, a livello macro: il minimo sui 120 contrasti modello × ruolo × cella è +0,02 K. L'effetto sulla RMSE, in questi scenari, è piccolo. Un effetto piccolo su questa metrica e su questi σ non è un margine garantito contro un altro modello di rumore, contro un errore di livello, o contro una PIV reale.

Aggiungere 0,5 m/s sopra le lacune contigue aumenta la RMSE di circa +0,2…+0,5 K rispetto alle sole lacune. L'interazione `N1G1 − N1 − G1 + R0`, come media sui tre seed, sta fra −0,09 e +0,23 K. I singoli seed arrivano a −0,18 e +0,35 K. In questi cinque scenari l'interazione è piccola rispetto ai +4…+8 K delle lacune. Non dimostra che rumore e lacune si sommino per altri livelli o altre geometrie.

## Cosa peggiora

Le lacune contigue sono, in ogni gruppo, l'aumento di RMSE più grande: +4,1…+8,1 K sul macro (10–15 % del proprio caso pulito). Il rumore a 1,5 m/s sta in mezzo: +1,3…+4,7 K (3–9 %). Il rapporto fra questi due aumenti medi, N2/N1, va da 6,4 a 41 sui otto gruppi. Il minimo è la U-Net senza φ e NH3 su VAL (3,38 / 0,52). Triplicare σ non triplica l'effetto. Il limite inferiore 8 citato nel report non include quel gruppo.

Il danno delle lacune non è uniforme sul campo. Sui getti freddi l'aumento medio di RMSE, per gruppo e ruolo, è +8…+22 K (i singoli seed su seen arrivano a +13…+26 K). Sulla zona calda è +1,5…+4,6 K; sul picco +2,2…+8,3 K. L'IoU della zona calda scende di circa 0,01–0,09 sulle medie di gruppo. La rete sbaglia soprattutto la regione fredda, non solo il nucleo caldo.

Dentro la maschera dei dischi la RMSE è più alta che fuori (GLU seen, con φ e NH3: 65 K contro 54 K, media sui seed). Per la U-Net su VAL il bias dentro e fuori è simile sullo stesso seed, fra circa −5 e −18 K a seconda di seed e di φ/NH3: lo spostamento di livello attraversa il campo, non resta nel buco.

## Famiglie e conoscenza di φ e NH3

Sull'errore assoluto la U-Net resta sotto il GLU in tutte le celle medie del report. A R0, seen: 42,3–44,8 K contro 50,1–53,3 K. Dopo le lacune contigue: 47,7–51,0 K contro 57,6–61,5 K.

Sul deterioramento il GLU perde più kelvin su seen (lacune +7,5/+8,1 K contro +5,4/+6,2 K). In percentuale il divario è più piccolo e non ha lo stesso segno su tutti i ruoli. «Più robusto» va detto due volte: la U-Net resta più accurata in valore assoluto; l'aumento rispetto al proprio caso pulito è più contenuto per la U-Net su seen, e meno ordinato su VAL. La dispersione fra i tre seed del deterioramento sulle lacune arriva a circa 2–3 K, dello stesso ordine della differenza fra con e senza φ/NH3. Un seed solo non ordina i modelli.

A R0, avere φ e NH3 abbassa la RMSE di circa 0,9–3,3 K. Attenua anche il peggioramento medio, di circa 0,6–1,7 K su lacune, combinazione e rumore a 1,5 m/s. L'abbinamento fra seed con lo stesso indice è solo descrittivo: le inizializzazioni sono diverse. Su seen, per le lacune, qualche coppia va nel verso opposto, fino a circa +0,8 K. φ e NH3 non vengono degradati. Non chiudono il problema delle lacune.

## Numero di siti e geometria

I siti di partenza sono 819. Le lacune contigue ne tolgono, secondo la tabella aggregata, fra 167 e 243 (media circa 207); l'area effettiva dei dischi è 25,0–26,0 %. G1-eq toglie lo stesso numero, in posizioni casuali. Questi estremi sono letti da `sites_and_area.csv`, non ricalcolati frame per frame.

Il rapporto fra l'aumento di RMSE di G1-eq e quello di G1, modello per modello, è 0,26–0,37 per i GLU e per la U-Net su seen, e 0,44–0,75 per la U-Net su VAL. Quando il rapporto è vicino a un terzo, togliere gli stessi siti a caso non riproduce il danno dei dischi. Il complemento non è una quota causale della sola contiguità: le due maschere non colpiscono gli stessi posti, e il rapporto cambia con famiglia, ruolo e condizione. Su VAL, per la U-Net senza φ e NH3, G1-eq arriva a circa tre quarti di G1 perché una condizione domina (sotto).

## Livello e struttura

Due distinzioni diverse.

**Rumore contro geometria delle osservazioni.** 0,5 m/s è quasi invisibile sulla RMSE. 1,5 m/s è dello stesso ordine della rimozione casuale di un quarto dei siti (+1,3…+3,6 K) e resta sotto le lacune contigue. In questo insieme, la geometria dei buchi pesa più del rumore provato. Vale per la RMSE macro. Non vale per ogni metrica: per il GLU senza φ e NH3, su VAL, il rumore a 1,5 m/s sposta il bias di −10,7 K in media sui seed, mentre la centered-RMSE cresce di +2,9 K.

**Bias contro centered-RMSE.** Su seen, per tutti i gruppi, il bias si sposta di meno di 1 K e la centered-RMSE segue l'aumento della RMSE. Su VAL il quadro si apre. Medie sui tre seed, in kelvin:

| gruppo | ruolo | cella | ΔRMSE | Δbias | Δ centered-RMSE |
|---|---|---|---:|---:|---:|
| U-Net, con φ e NH3 | seen | G1 | +5,4 | −0,5 | +5,5 |
| U-Net, senza | seen | G1 | +6,2 | −0,5 | +5,3 |
| GLU, con | seen | G1 | +7,5 | +0,7 | +7,5 |
| GLU, senza | seen | G1 | +8,1 | +0,3 | +7,8 |
| U-Net, con | VAL | G1 | +4,1 | −0,5 | +3,7 |
| U-Net, senza | VAL | G1 | +5,6 | −8,6 | +3,6 |
| GLU, con | VAL | G1 | +4,3 | +0,4 | +4,4 |
| GLU, senza | VAL | G1 | +5,6 | −2,2 | +4,8 |
| U-Net, senza | VAL | N2 | +3,4 | −3,8 | +2,3 |
| GLU, senza | VAL | N2 | +4,1 | −10,7 | +2,9 |

Su VAL, la U-Net senza φ e NH3 diventa più fredda quando perde siti, anche se i siti sono tolti a caso (Δbias di G1-eq: −7,5 K). Il GLU senza φ e NH3 diventa più freddo col rumore a 1,5 m/s, più che con i dischi. Avere φ e NH3, in questi modelli, tiene il bias vicino a zero anche su VAL. Il caso pulito della U-Net su VAL è già freddo (bias fra circa −6 e −11 K): la degradazione si somma a un livello già spostato.

## Casi lontani dalla media

Su seen, le lacune contigue colpiscono soprattutto `80NH3_1.15`, `40NH3_1.2` e `80NH3_1.05` (aumento medio di gruppo intorno a +10…+13 K) e molto meno `0NH3_0.9`, `20NH3_0.85` e `0NH3_1` (circa +0,8…+4 K a seconda della famiglia). I tre seed di una stessa condizione possono distare di alcuni kelvin.

Su VAL la media è tirata da `40NH3_0.9`. Per la U-Net senza φ e NH3, l'aumento di RMSE è +9,8 K con i dischi e +9,3 K togliendo gli stessi siti a caso. Il bias, media sui seed, è circa −18 K con i dischi e circa −20 K con la rimozione casuale; il seed 0 arriva a circa −24 e −26 K. Su quella condizione conta la perdita di siti, non la forma dei dischi. Le altre due condizioni VAL restano molto più basse (G1-eq sotto +1,2 K per la stessa U-Net). Per il GLU senza φ e NH3, il rumore a 1,5 m/s su `40NH3_0.9` vale +7,4 K di RMSE e circa −20 K di bias.

Un miglioramento locale esiste ed è piccolo: la U-Net senza φ e NH3, su seen, con rimozione casuale, migliora di −1,3 K su `20NH3_0.85` (media sui seed). Non è un recupero.

## Figure

Già nel pacchetto, e sufficienti per la domanda che pongono:

- `fig1_macro_rmse_by_cell.png` — RMSE assoluta nei sei scenari, seen e VAL separati, tre seed visibili.
- `fig2_deterioration_vs_R0.png` — aumento rispetto al proprio caso pulito, tre seed visibili.
- `fig5_fields_*.png` — quattro modelli a seed 0, tre frame VAL fissi. Temperatura vera e modulo dell'errore su una scala comune 0–400 K; in ciano i siti tolti, in bianco il contorno dei dischi. Non mostrano la temperatura predetta come pannello a sé. I campi salvati non sono nel repository: non sono state fatte nuove mappe.

Nuove, nella cartella di lettura:

- `fig_g1_vs_g1eq.png` — dischi contro siti a caso, e il rapporto descrittivo fra i due aumenti.
- `fig_bias_centered.png` — variazione di bias e di centered-RMSE.
- `fig_regional_jets_hot.png` — getti, zona calda e picco, per rumore, siti a caso e dischi.
- `fig_conditions.png` — le 20 condizioni seen sotto i dischi, e le tre condizioni VAL con dischi e siti a caso, RMSE e bias.

`fig3` e `fig4` del pacchetto restano. La prima non ha legenda; la seconda usa etichette contratte. Per la discussione si usano le figure nuove.

## Limiti

Degradazioni sintetiche: σ e geometria dei dischi sono default di scenario, non misure di questo combustore. I modelli sono allenati con il 5 % di siti casuali: i dischi sono anche una condizione fuori da quella distribuzione. Tre seed e tre repliche. seen220 e VAL123 hanno condizioni e numeri di frame diversi; il confronto fra i due ruoli è descrittivo. Le metriche dentro/fuori dai dischi non hanno un caso pulito sulla stessa area, quindi non sono deterioramenti. Non sono stati provati media spaziale, campionamento regolare, lacune dipendenti dalla temperatura, spuri, temperatura osservata. Nessun TEST. Nessun nuovo training in questa lettura.
