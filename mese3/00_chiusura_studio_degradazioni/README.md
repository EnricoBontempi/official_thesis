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

**Aggiornamento del 27 novembre 2026 (nessuna nuova esecuzione autorizzata).** Consegna numerica e grafica dei 30 specialisti verificata e pubblicata sul branch `nonideal/specialist-results`:

- Sintesi: [docs/nonideal/05_specialist_results.md](../../docs/nonideal/05_specialist_results.md) (recupero, costo su R0, trasferimento, seen e VAL separati, tre seed, best/last, stato dei training, limiti del batching GLU, indicazioni per la missione 02).
- Risultati: [results/nonideal/specialists/](../../results/nonideal/specialists/) (tabelle aggregate, figure di riepilogo, controllo di consegna, manifest). Il pacchetto completo resta sul cluster.
- Correzione dei grafici: [docs/nonideal/04_specialist_graphics.md](../../docs/nonideal/04_specialist_graphics.md).
- **Ancora mancante:** i pannelli spaziali appaiati baseline-specialista (frame 0/1600 contro 1000), che richiederebbero una nuova inferenza degli originali; il confronto GLU resta confuso dal batching. Lo studio non è dichiarato chiuso finché Enrico non decide su queste parti.
