# Bridge O2 — percorso A38 seed 0 nuovo contro S5 eseguito

Tre controlli distinti. Il replay di inferenza, quando arriverà, non sostituisce il controllo sull'ordine di training.

## Verificato

- Codice di inizializzazione invariato da `38ac7fe`: `unet.py` e `set_seed` non sono nel diff. L'inizializzazione usa un `Generator` dedicato. S5 non ha salvato l'hash dei pesi iniziali, quindi l'identità bit-a-bit con quel run non è disponibile.
- Maschere TRAIN: `sites_from_training_mask` è invariato e dipende da seme, epoca, simulazione e frame, non dall'ordine di presentazione. Il test sintetico lo conferma. Non esiste un dump storico delle maschere TRAIN: il controllo è di codice e di funzione, non di un file S5.
- Maschere VAL persistite: manifest `2e2d8ff6edabf62ad56702a155140d064c0b4014d06ee96debf52811efa5c75e`, lo stesso registrato in `run_identity.json` dei quattro S5 e in `runs_provenance.json`.
- Loss, optimizer, warmup, LR, scheduler: stessi campi nei YAML S5 e Opus (AdamW, lr `1e-3`, weight decay `1e-4`, clip `1`, warmup 950 update, ReduceLROnPlateau min, factor `0.5`, patience 5 eval, threshold rel `0.001`, min lr `1e-5`). Il codice di optimizer e plateau non è nel diff da `38ac7fe`.
- Precisione sul run eseguito: `progress.json` dei quattro S5 registra `allow_tf32=false` su matmul CUDA e cuDNN. Il percorso Opus impone lo stesso flag, con AMP e compile spenti.
- Identità dei checkpoint S5, ricalcolata in lettura il 2026-11-19, coincide con `runs_provenance.json` per `best.pt` e `last.pt` dei quattro run. Git eseguito `38ac7fe40178982ac1bd33c5d5602a5e7d928d3b`. Cache aggregate `710bab909a0efb6cc1b4b4a0d7f7dcc4889472b7adc8076abb94fd529576ab5d`. Stats `678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e`.
- Insieme dei frame per epoca scientifica A38: entrambi i percorsi presentano ciascuno dei 760 indici una volta (batch 8, 95 update, nessun resto).

## Differente

- Ordine dei frame. S5 ha `presentations_per_scientific_epoch` assente e ha usato `DataLoader(shuffle=True)` con seme `shuffle_seed + 100003 * (epoch + 1)`. Il percorso A38 nuovo imposta 760 presentazioni e usa `ScientificPresentationStream`, cioè un `randperm` diretto sullo stesso seme.
- Il DataLoader non è quel `randperm`. Prima della permutazione estrae un intero dal generatore (`_base_seed`). Confronto sintetico, torch locale 2.15 e torch Engaging 2.8.0+cu128: all'epoca 0 il DataLoader inizia con `[193, 743, 248, 313, 2, 620]` e il `randperm` con `[26, 184, 591, 326, 460, 473]`. Alle epoche 0, 1 e 59 tutti i 95 batch differiscono; l'insieme degli indici resta lo stesso.
- Il test precedente che dichiarava l'equivalenza con il DataLoader confrontava lo stream con il `randperm` grezzo. Il nome e la docstring ora lo dicono. L'ordine eseguito da S5 non è stato quello.

Questa differenza cambia la traiettoria SGD. Non è correggibile sui checkpoint già scritti.

## Mancante

- Hash dei pesi iniziali S5.
- Dump delle maschere TRAIN epoca per epoca.
- Esito numerico del replay di inferenza. Job **29670888** sottomesso, non letto. Confronta i due `best.pt` U-Net al 5% con il macro RMSE VAL storico, tolleranza 0.05 K, identità in `expected_unet_best_5pct.json`. Non misura l'ordine di training.
- CSV `train_fit` dell'epoca di arresto S5 (usato da R0 come assenza, non come valore).

## Conseguenza per O5

Il confronto seed 0 fra il pool di S5 (A38, runner storico) e i pool B19/C149 del runner nuovo mescola dati e ordine di presentazione. I sei run Opus fra loro condividono lo stream e restano confrontabili; l'A38 Opus è seed 1, non il seed 0 di S5.

Non si può riallineare S5 senza un training. Servirebbero due riferimenti A38 seed 0, p_on e p_off, nel runner nuovo. Non sono stati lanciati.

O5 non è preparata e non è sottomessa.
