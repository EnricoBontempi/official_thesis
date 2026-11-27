# Chiusura dello studio sulle degradazioni

**Qui finiamo ciò che sta già girando, senza allargare la campagna.**

- Verifichiamo i 30 training, le eventuali continuazioni, le valutazioni e i grafici.
- Confrontiamo modelli originali e specialisti su tutti e sei gli scenari.
- Guardiamo recupero sul degrado, perdita sul pulito e trasferimento agli altri degradi.
- Scriviamo il documento conclusivo con figure, numeri e limiti, compresa la differenza di batching GLU.
- Pubblichiamo i risultati e aggiorniamo lo stato della roadmap precedente.

**Consegna finale:** studio concluso e leggibile, oppure elenco preciso delle parti mancanti. I risultati già pubblicati restano nei loro percorsi attuali: questa cartella li collega, senza duplicarli.

---

Stato al 25 novembre 2026: testo descrittivo approvato, roadmap specifica ancora da costruire nella chat di brainstorming. Questo README non autorizza nuove esecuzioni. Per contesto e coordinamento: [indice delle missioni](../README.md).

---

**Aggiornamento del 27 novembre 2026 (nessuna nuova esecuzione autorizzata).** Consegna sul branch `nonideal/specialist-results`, distinta per livello:

| Livello | Stato | Dove |
|---|---|---|
| Campagna numerica (30 training, 60 valutazioni, 720 righe) | **verificata** | [controllo di consegna](../../results/nonideal/specialists/delivery_check/delivery_check.md) |
| Grafici | **disponibili** (job 30711670; schede complete nella release tecnica) | [indice grafici](../../results/nonideal/specialists/graphics/INDEX.md) |
| Rapporto | **consegnato** | [docs/nonideal/05_specialist_results.md](../../docs/nonideal/05_specialist_results.md): Parte 1 per le persone, Parte 2 = appendice tecnica con rendiconto computazionale |
| Dati completi | CSV/JSON nel repository, per frame nella [release](https://github.com/EnricoBontempi/official_thesis/releases/tag/nonideal-specialists-data-v1) | [indice della consegna](../../results/nonideal/specialists/INDEX.md) |

**Lacune residue:** pannelli spaziali appaiati baseline–specialista (frame 0/1600 contro 1000: servirebbe una nuova inferenza degli originali, non autorizzata); latenza d'inferenza non misurata; il confronto GLU resta confuso dal batching (confondente noto del disegno, non un errore da correggere). Correzione dei grafici: [04_specialist_graphics.md](../../docs/nonideal/04_specialist_graphics.md). Lo studio non è dichiarato chiuso finché Enrico non decide su queste lacune.
