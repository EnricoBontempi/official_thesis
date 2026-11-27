# Roadmap 04 — Misure di temperatura e fusione termica

**Missione:** `mese3/04_thermal_fusion/`  
**Stato:** approvata da Enrico il 26 novembre 2026 (America/New_York); pubblicazione autorizzata. Nessuna attività esecutiva avviata dalla pubblicazione.

## Obiettivo

Capire come viene misurata la temperatura negli esperimenti di combustione con accesso ottico, soprattutto quelli che utilizzano PIV: quali strumenti vengono impiegati, dove vengono posizionati, perché e quali dati producono.

Su questa base, studiare caratteristiche e problemi delle misure termiche, cercare dataset sperimentali e verificare come rappresentare osservazioni realistiche nei nostri dati numerici. Successivamente, proporre esperimenti per valutare che cosa aggiunge la temperatura alla ricostruzione da velocità.

RealPDEBench è il riferimento numerico attualmente utilizzato, ma non limita la ricerca né costituisce una scelta definitiva del dataset.

## Organizzazione

Questa chat coordina la missione: prepariamo gli incarichi, discutiamo le consegne degli agenti e concordiamo il passo successivo.

**Pubblicare la roadmap non avvia automaticamente gli agenti o gli esperimenti.** Gli incarichi saranno preparati e avviati separatamente.

## Step 1 — Capire come si misura la temperatura in laboratorio

### Incarico a un unico agente

Svolgere una ricerca ampia per rispondere precisamente:

> **Negli esperimenti di combustione con accesso ottico, in particolare quelli che utilizzano PIV, si misura la temperatura? Quale temperatura? Come, con quali strumenti, dove vengono collocati e perché?**

Includere combustori, bruciatori e fiamme da laboratorio, motori ottici e altri apparati pertinenti, senza imporre una divisione preventiva fra categorie.

L’agente può utilizzare liberamente letteratura, riviste, siti di laboratori, documentazione tecnica, Wikipedia, forum e altre risorse per orientarsi. Le conclusioni decisive devono essere ricondotte a fonti verificabili, privilegiando descrizioni sperimentali originali.

La ricerca deve chiarire:

- scopo delle misure termiche;
- distinzione fra temperatura del gas, delle pareti, all’ingresso e allo scarico;
- strumenti utilizzati, comprese termocoppie, altre sonde e tecniche ottiche;
- posizioni e motivazioni del posizionamento;
- vincoli di accesso, intrusività e compatibilità con le altre diagnostiche;
- natura locale, distribuita o integrata della misura;
- informazione istantanea, filtrata, mediata o risolta in fase;
- relazione con la PIV: simultaneità, sincronizzazione o campagne separate;
- casi in cui la misura termica è dichiaratamente assente e informazioni semplicemente non riportate.

Non assumere che accesso ottico significhi utilizzo della PIV, né che la temperatura venga necessariamente misurata con termocoppie.

### Consegna e discussione

Un rapporto comprensibile, sostenuto da casi concreti e riferimenti precisi, che risponda alla domanda iniziale e identifichi gli strumenti e le configurazioni da approfondire.

**Prima dello step 2 discutiamo il rapporto qui:** quanto emerso determinerà quali dati e quali problemi cercare.

## Step 2 — Studiare problemi delle misure e dataset sperimentali

Dopo la discussione dello step 1, preparare **due incarichi paralleli**, riferiti agli strumenti e alle configurazioni emersi.

### Agente A — Letteratura sulle caratteristiche e sui problemi dei dati termici

Approfondire come vengono acquisiti e processati quei dati e quali limitazioni presentano.

Indagare, dove pertinenti:

- rumore, bias, outlier e misure mancanti;
- risposta temporale, ritardi e filtraggio;
- risoluzione spaziale e volume di misura;
- calibrazione, correzioni e incertezza;
- dipendenza dei problemi dalle condizioni operative e dalla posizione;
- effetti del processamento sui dati finali.

Questi fenomeni sono domande da verificare, non caratteristiche da attribuire automaticamente a ogni strumento.

### Agente B — Ricerca dei dataset termici

Cercare in modo ampio dataset sperimentali pertinenti alle misure identificate nello step 1.

Registrare apparato, condizioni, strumenti, posizioni, grandezze disponibili, coordinate, unità, tempi, processamento, incertezza, accessibilità e licenza. Cercare anche eventuali misure di velocità associate.

Distinguere dati citati negli articoli, dati dichiarati disponibili e dati effettivamente accessibili. Ispezionare campioni secondo i limiti concordati nell’incarico.

### Confronto e verifica

Al ritorno dei due agenti, confrontare quanto descritto in letteratura con il contenuto dei dataset.

Verificare quali problemi:

- sono osservabili e quantificabili nei dati;
- sono documentati ma già corretti o rimossi;
- non sono riscontrati nel campione esaminato;
- non sono verificabili con il materiale disponibile.

La variabilità fisica non misura da sola il rumore; l’assenza di lacune nei dati processati non dimostra che non vi siano stati rigetti.

**Consegna:** quadro verificato delle misure, dei problemi e dei dataset, con esempi, parametri documentati e lacune esplicite. Discussione qui prima dello step 3.

## Step 3 — Rappresentare misure termiche realistiche e proporre gli esperimenti

Usare le evidenze precedenti per definire come trasformare i campi numerici di temperatura in osservazioni coerenti con una configurazione sperimentale documentata.

### 3A — Fattibilità e specifica delle osservazioni

Verificare geometria, risoluzione, variabili disponibili, continuità temporale, durata delle sequenze e associazione campo–tempo, riutilizzando l’inventario della missione 01.

Definire, secondo lo strumento scelto:

- posizioni e copertura;
- supporto spaziale della misura;
- risposta temporale e frequenza di acquisizione;
- inizializzazione e storia necessaria;
- errori, incertezza e dati mancanti;
- sincronizzazione con la velocità;
- approssimazioni e fenomeni non rappresentabili.

Ogni parametro deve avere una provenienza esplicita. Nessun numero universale di sensori o posizionamento prefissato.

**Discutere e approvare la specifica prima dell’implementazione.** La temperatura numerica di riferimento resta distinta dalle osservazioni termiche fornite alla rete.

### 3B — Proposta degli esperimenti e successiva esecuzione

Proporre confronti per misurare il contributo delle osservazioni termiche rispetto alla sola velocità, mantenendo controllate le altre condizioni.

Partire da riferimenti semplici di correzione e fusione; valutare successivamente se servano metodi appresi. Architettura, training e budget verranno concordati sulla base dello scenario.

**I training con temperatura partiranno soltanto dopo la conclusione della missione 02**, così da aggiungere T separatamente a un problema con osservazioni di velocità già definite e valutate.

Scouting, verifiche e preparazione della specifica possono procedere nel frattempo.

## Coordinamento con le altre missioni

- **01:** riutilizzare l’inventario comune e fornire le evidenze termiche per aggiornare la matrice.
- **02:** condividere riferimenti sperimentali e coordinare osservazioni di velocità e temperatura.
- **03 e 05:** coordinare eventuali sovrapposizioni metodologiche o bibliografiche.
- **Quartier generale:** riportare le decisioni che coinvolgono più missioni.

Questa missione si occupa delle misure termiche, dei dati e della loro eventuale fusione con la velocità; le scelte complessive della tesi restano nelle sedi dedicate.
