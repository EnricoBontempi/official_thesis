# Handoff — 30 training specialistici (Blocco III autorizzato)

**Data:** 24 novembre 2026 · **Branch:** `nonideal/specialist-retraining` (da `d4f2e3a`) · **Stato:** codice, configurazioni e test CPU pronti; smoke GPU e submission descritti nel diario.

## Contratto (congelato prima dei run)
- **Matrice:** 2 famiglie (U-Net masked, GLU adattato) × p_on × seed 0/1/2 × regimi N1, N2, G1, G1-eq, N1G1 = 30 training da zero. Un solo regime per modello, nessuna percentuale di campioni puliti, nessun training misto. Le baseline pulite (R0) sono i modelli originali, **non** riallenati: manifest in `configs/nonideal/specialist_baselines_v1.json`.
- **Regimi:** operatori e livelli della specifica `degradation_spec_v1_12models.yaml` (invariati). Applicati dopo il campionamento TRAIN 5 % storico (maschere dipendenti dall'epoca, invariate).
- **Realizzazioni:** namespace TRAIN (seed 20261201) con chiave (sim, frame, epoca di maschera) — nessun seed di modello né famiglia nella chiave, quindi stessi pacchetti fra famiglie e seed per lo stesso campione; namespace di SELEZIONE (20261202) per la metrica VAL specialistica; la valutazione usa 20261125 e non viene mai usata per allenare. N1/N2 condividono il rumore standardizzato (N2 = 3×N1); G1, N1G1 e G1-eq condividono la regione (G1-eq rimuove lo stesso numero di siti, uniformemente).
- **Ricetta:** S7 invariata (C149, normalizzatore fit-only760, batch effettivo 8, 760 presentazioni, 95 update/epoca, AdamW 1e-3, wd 1e-4, clip 1, warmup 950, fp32, nessun AMP/TF32/compile, early stop 200/15/0.001, eval ogni 10).
- **GLU, token variabili:** microbatch 1 + accumulo dei gradienti a batch effettivo 8 per **tutti** i cinque specialisti GLU (loss per microbatch scalata 1/8; clip e optimizer step una volta per batch effettivo; contatori di update/warmup/scheduler per batch effettivo). Nessun troncamento al minimo, nessun padding, nessun riempimento. Differenza dal GLU originale (batch 8 in un forward): solo ordine di somma numerico, verificato su perdita, norma del gradiente, gradienti per parametro e aggiornamento. La U-Net resta a batch 8.
- **Selezione:** primo minimo assoluto della macro RMSE VAL **pulita** al 5 % (3 repliche), come S7; scheduler ed early stop guidati dalla stessa metrica; si confrontano augmentation specialistiche con la selezione originale, **non** il checkpoint ottimo per scenario. Metrica VAL dello scenario specialistico registrata, mai usata per decidere. `last` conservato.
- **Budget:** tranche iniziale 1000 epoche; un'unica continuazione a 1200 (stesse regole e stato); oltre, `budget_exhausted`, nessuna estensione.
- **Risorse:** 4 GPU `mit_normal_gpu` (account/QoS `mit_amf_advanced_gpu`) + 4 GPU `mit_preemptable` (`mit_general`, QoS `normal`); L40S; nessuna duplicazione di un run sui due pool; ripresa solo dal proprio `last.pt`.
- **Catena (autonoma):** array di training (due pool) → stage B (classifica; sottomette l'unica continuazione ai run a 1000 senza plateau, sullo stesso pool) → stage C (valida e congela: stati scientifici dichiarati, identità, hash; sottomette la valutazione) → valutazione best e last, sei scenari, gate R0 contro la metrica del loop → aggregazione (rifiuta matrici incomplete, scrive BLOCK_REPORT).
- **Stop:** nessun training aggiuntivo, misto o nuova configurazione dopo i risultati; TEST chiuso; A/B/C sospese.

## Output
`<dest>/FINAL_SUMMARY.md` oppure `<dest>/BLOCK_REPORT.md`; pacchetto `<dest>/package/INDEX.md`; freeze `<dest>/freeze.json`; valutazioni `<dest>/eval/<run_id>/{best,last}/`.
