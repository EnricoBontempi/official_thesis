# Mese 3 — Missioni e chat di brainstorming

**Una roadmap operativa per ogni sottocartella e un semplice indice generale.** Non serve una seconda `roadmap_mese3.md` che ripeta le altre. L’indice dice che cosa contiene ogni filone, a che punto è e da quali altri dipende.

Questa struttura è stata approvata da Enrico il 25 novembre 2026. Per ora contiene soltanto i testi descrittivi concordati: le roadmap specifiche saranno costruite e discusse nelle rispettive chat di brainstorming. La creazione delle cartelle non autorizza nuove campagne.

Sono previste **sei cartelle**: chiusura attuale, matrice comune e quattro filoni.

## [00_chiusura_studio_degradazioni/](00_chiusura_studio_degradazioni/)

**Qui finiamo ciò che sta già girando, senza allargare la campagna.**

- Verifichiamo i 30 training, le eventuali continuazioni, le valutazioni e i grafici.
- Confrontiamo modelli originali e specialisti su tutti e sei gli scenari.
- Guardiamo recupero sul degrado, perdita sul pulito e trasferimento agli altri degradi.
- Scriviamo il documento conclusivo con figure, numeri e limiti, compresa la differenza di batching GLU.
- Pubblichiamo i risultati e aggiorniamo lo stato della roadmap precedente.

**Consegna finale:** studio concluso e leggibile, oppure elenco preciso delle parti mancanti. I risultati già pubblicati restano nei loro percorsi attuali: questa cartella li collega, senza duplicarli.

## [01_observation_state_matrix/](01_observation_state_matrix/)

**Qui definiamo quali problemi sperimentali possiamo sensatamente affrontare.**

Costruiamo la tabella richiesta da Linzheng: osservazione disponibile → limiti dello strumento → stato desiderato → possibile validazione.

In questa cartella mettiamo anche:

- la verifica comune di ciò che contiene davvero RealPDEBench, inclusa la disponibilità temporale;
- le schede dei due paper di Jason, spiegando problema, misure, target, validazione e limiti;
- le domande ancora da chiarire con esperti o autori.

**Consegna finale:** una matrice documentata che viene aggiornata dagli altri filoni. Una prima versione serve subito; non deve bloccare tutte le ricognizioni finché ogni casella è completa.

## [02_virtual_piv/](02_virtual_piv/)

**Qui costruiamo e studiamo il problema di ricostruzione da osservazioni di velocità motivate dalla PIV.**

L’ordine è:

1. Leggere Sciacchitano e cercare letteratura e dati sperimentali pertinenti.
2. Identificare disposizione dei vettori, risoluzione spaziale, zone mancanti e incertezza.
3. Proporre un modello virtuale realizzabile con i nostri dati, con parametri documentati e ipotesi separati.
4. Discutere e fissare quel modello di misura.
5. Valutare U-Net e GLU esistenti; poi decidere gli eventuali retraining necessari.

**Consegna finale:** un esperimento di ricostruzione legato a un accesso sperimentale preciso. Non soltanto altre intensità di rumore scelte arbitrariamente.

## [03_dmf_gen/](03_dmf_gen/)

**Qui capiamo se un metodo generativo porta qualcosa rispetto alle reti attuali.**

Prima studiamo teoria, paper e codice, producendo anche una spiegazione comprensibile per te. Poi definiamo un eventuale pilot che risponda a due domande distinte:

- Ricostruisce meglio, a quale costo?
- Produce un’incertezza utile e verificabile quando le osservazioni non bastano?

Il confronto deve usare le stesse osservazioni e gli stessi riferimenti dei metodi deterministici. **Non basta mostrare un alone di varianza e chiamarlo UQ.**

**Consegna finale:** valutazione di fattibilità e, se decidiamo di procedere, confronto sperimentale controllato. Non assumiamo già che debba sostituire U-Net o GLU.

## [04_thermal_fusion/](04_thermal_fusion/)

**Qui studiamo come combinare velocità e poche misure termiche realistiche.**

L’ordine è:

1. Cercare esempi pertinenti di sensori termici: posizioni, risposta temporale, incertezza e vincoli fisici.
2. Verificare che il dataset permetta di simulare quei sensori.
3. Scegliere uno scenario concreto, senza inventare che siano disponibili esattamente quattro o nove termocoppie.
4. Costruire baseline semplici di fusione e correzione.
5. Valutare se serve un meccanismo appreso e quale informazione aggiunge.

**Qui conserviamo le tue idee su fusione delle feature, rappresentazioni condivise/specifiche e contrastive learning**, ma scegliamo il metodo dopo aver definito il problema.

**Consegna finale:** capire se e come misure termiche lente e sparse migliorano la ricostruzione rispetto alla sola velocità. DMF-Gen potrà eventualmente essere uno dei metodi, condividendo lo stesso scenario.

## [05_optical_diagnostics/](05_optical_diagnostics/)

**Qui facciamo la ricognizione ottica chiesta da Linzheng, senza aprire subito una campagna.**

Studiamo OH-PLIF, OH* chemiluminescenza e le altre diagnostiche pertinenti, distinguendo segnali misurati e variabili numeriche. Cerchiamo problemi utili, dati disponibili e possibilità di validazione.

La scheda comune del paper di Jason viene riutilizzata; qui ne approfondiamo le implicazioni diagnostiche.

**Consegna finale:** una proposta motivata su quali problemi ottici meritino eventualmente di essere affrontati dopo. Anche concludere che i dati attuali non bastano è un risultato utile.

---

## Come lavoriamo

Ogni cartella avrà la propria roadmap, con obiettivo, passi, risultati attesi e punti in cui discutere prima di passare all’esecuzione. Questo README è soltanto la mappa del lavoro, comprese le attività parallelizzabili.

- **Quartier generale:** la conversazione principale con Enrico tiene insieme priorità, risorse, decisioni scientifiche e questioni manageriali.
- **Chat specifiche:** una per sottocartella, dedicate al brainstorming e alla costruzione della relativa roadmap. Nessuna decisione su un filone autorizza automaticamente attività negli altri.
- **Parallelismo:** ricognizioni PIV, termica, generativa e ottica possono procedere insieme alla prima versione della matrice. Gli esperimenti comparativi richiederanno un contratto di osservazione condiviso. La fattibilità temporale del dataset è un prerequisito per simulare sensori termici lenti.
- **Output esistenti:** codice, configurazioni e risultati restano nei percorsi canonici della repository; qui li colleghiamo, senza copiarli.
- **Roadmap precedenti:** A/B/C rimangono sospese. Lo studio dei 30 specialisti resta in corso fino a verifica della consegna.

Aggiorniamo il [diario canonico](../docs/RESEARCH_LOG.md), senza crearne un altro: messaggi di Linzheng e Jason, indicazioni ricevute, nostre interpretazioni e decisioni effettivamente approvate. Registriamo che continuiamo sulla combustione turbolenta, esploriamo i dati motore in parallelo e concordiamo con Deng l’eventuale contatto con Franzelli.

La cartella `00` è un ponte con il lavoro precedente: metterla qui serve a gestirne la chiusura, non a confonderla con i nuovi esperimenti.

## Contesto e fonti iniziali

- Codice ed esperimenti: [official_thesis](https://github.com/EnricoBontempi/official_thesis).
- Report per i supervisori: [documents_thesis](https://github.com/EnricoBontempi/documents_thesis). Non viene modificata in questa attività.
- Indicazioni dei supervisori e decisioni del 25 novembre: voce dedicata nel diario canonico.
- Sciacchitano (2019), *Uncertainty quantification in particle image velocimetry*: https://doi.org/10.1088/1361-6501/ab1db8
- Nie et al. (2024), *Reconstructing temperature fields from OH distribution and soot volume fraction in turbulent flames using an artificial neural network*: https://doi.org/10.1016/j.combustflame.2023.113182
- Liu et al. (2025), *Neural network-based 3D reconstruction of temperature and velocity for turbulent flames from 2D measurements*: https://doi.org/10.1016/j.combustflame.2025.114454
- DMF-Gen, paper e codice indicati da Linzheng: https://doi.org/10.48550/arXiv.2610.34709 — https://github.com/cosmos2w/DMF-Gen

I riferimenti sono punti di partenza; la struttura non attesta una revisione completa delle fonti o un audit del codice DMF-Gen.
