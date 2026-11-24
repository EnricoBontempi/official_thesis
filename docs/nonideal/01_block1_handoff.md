# Handoff Blocco I → Blocco II

**Data:** 24 novembre 2026 · **Branch:** `nonideal/block1-degradation-spec` (da `8d72a81`)
**Stato:** specifica scritta e testata su CPU; revisione di Enrico da registrare. Nessun job, training o forward di modelli eseguito.

## Che cosa c'è

| Cosa | Dove |
|---|---|
| Specifica parsabile (sei celle, G1, seed, batching, modelli) | `configs/nonideal/degradation_spec_v1.yaml` |
| Operatori e validazione della specifica | `src/combustion_recon/data/nonideal_observations.py` |
| Test CPU | `tests/test_nonideal_observations.py`, `tests/test_nonideal_glu_support.py` |
| Qualificazione G1 su TRAIN | `scripts/58_nonideal_qualify_g1.py` → `results/nonideal/block1/g1_train_qualification.json` |
| Fonti, orientamento, limiti | `docs/nonideal/00_ricognizione_osservazioni.md` §9 |

## Contratto per il Blocco II

1. Ordine: campo fisico → siti storici 5 % (repliche 0/1/2) → rumore in m/s su tutti i siti originari → invalidazione → z-score congelato (`value_norm + noise_ms/std`) → input. Media spaziale disattivata.
2. Per ogni frame/replica: stessi siti e stesse realizzazioni per tutti i modelli e seed. Una replica cambia insieme siti, rumore e regione G1.
3. N1 e N2 condividono la realizzazione standardizzata (N2 = 3×N1): confronto appaiato di livello, non campioni indipendenti. N1 e N1G1 hanno rumore identico ai superstiti; G1 e N1G1 la stessa regione; G1-eq rimuove lo stesso numero di siti.
4. U-Net: `to_unet_grid` dà 0 e Mv=0 ai siti rimossi. GLU: token assenti (`to_glu_tokens`). Nessun riempimento.
5. GLU: R0/N1/N2 a N costante (819) → batch di 8 come in S7. G1/N1G1/G1-eq a N variabile → batch omogenei, in pratica 1 (`glu_batching_policy`). Il riferimento appaiato è il replay R0 **con lo stesso batching**: serve quindi un replay aggiuntivo di R0 a batch 1 per il GLU. Le discrepanze di replay si diagnosticano; 0.04 K non è una soglia.
6. Report: grandezze, dispersione, regioni, singole condizioni; nessuna soglia automatica; Enrico decide il Blocco III.

## Da fare nel Blocco II (non fatto qui)

- II.0: esistenza e hash dei sei checkpoint e degli input sul cluster (non verificati).
- Punto di ingresso nell'evaluator S7 che ricostruisca il pacchetto storico e chiami `degrade_observation`; il test di identità bit-a-bit **contro le uscite S7** richiede il cluster (qui è verificata l'identità del pacchetto di osservazione, non delle predizioni).
- Memorizzare per riga: hash della specifica, del checkpoint, della maschera, SHA del codice, `meta` dell'operatore (n_removed, area effettiva, tentativi).
- Frame delle figure: regola nella specifica (frame con indice minore di ogni condizione, replica 0).

## Ipotesi e limiti da tenere visibili

- σ = 0.5/1.5 m/s, area 25 % e diametri 3–10 mm sono default di scenario, non misure del nostro combustore.
- G1 è uno stress test sintetico (classe C); bordi ritagliati, quindi copertura minore ai margini (angoli ≈ 7 % contro ≈ 25 % medio).
- Convenzioni di i/j/k non dichiarate: irrilevanti per operatori isotropi.
- VAL123 già usata per selezione e sviluppo: studio descrittivo.
- Non è una validazione PIV realistica.

## Aggiornamento Blocco II (24 novembre 2026)
Ampliamento a 12 modelli (p_on e p_off, seed 0/1/2): revisione `degradation_spec_v1_12models.yaml`; batching uniforme per famiglia (GLU batch 1 anche per R0, replay tecnico a batching S7); gate R0 ≤ 0.05 K dal riferimento S7. Codice e test CPU pronti; smoke GPU e campagna da eseguire: vedi [02_block2_runbook.md](02_block2_runbook.md).
