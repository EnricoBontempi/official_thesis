# Revisione scientifica S7 (temperature-free C149) e disegno degli esperimenti con osservazioni di temperatura (Step 2.7-A / 2.7-B)

- **Autore nota:** Revisore scientifico (Ruolo: analisi e raccomandazione; nessuna esecuzione, nessuna modifica a codice/roadmap/risultati)
- **Tesi:** Enrico Bontempi (MIT / Deng Lab) — *Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion*
- **Data revisione:** 21 novembre 2026
- **Branch / base della nota:** `review/s7-and-temperature-study-opus`, diramato da `mission/dense-unet` (`5f8842fb4042ca9f28589446241f7d97e09690ab`)
- **Stato dei job S7 al momento della revisione:** valutazioni finali sottomesse il 21 novembre ~15:15 EST (`29984795`, `29984793`, `29984812`) — **in corso / esito non verificato**. Nessun vincitore S7 dichiarato in questa nota; nessuna esecuzione scientifica effettuata.

---

## 0. Fonti, branch, commit SHA e riscontri verificati

### 0.1 Mappa delle fonti consultate (tutte verificate via Git / filesystem locale)

| Fonte | Branch / Ref | Commit SHA verificato | Ruolo / note |
|---|---|---|---|
| Piano approvato corrente (`docs/proposal.md`, `roadmapsparse.md`, `roadmap_month2.md`, `docs/month2/00_context_and_decisions.md`, `analisi tripla revisore s7.md`) | `mission/dense-unet` | `5f8842fb4042ca9f28589446241f7d97e09690ab` | Unica fonte normativa approvata |
| Baseline Mese 2, adattamento GLU, screening storico S5/S6 (`02_baseline_validation.md`, `04_glu_release_adaptation.md`, `05_sparse_reconstruction.md`) | `mission/dense-unet` | `5f8842fb4042ca9f28589446241f7d97e09690ab` | Riferimenti storici A38 (`760` frame) |
| Report 06 (`docs/month2/06_opus_data_and_training_variability.md`) | `review/opus-06-report` | `4f0e2637758de0766d0bb0891fe8ff97eee978c9` | Verificato contro CSV sorgente O5 |
| Pacchetto O5 (`global_by_setting.csv` e tabelle) | `opus/o5-eval-attempt2` | `5373aa3c7633585edcee3d879b3c92bbb4748c11` | Sorgente numerica primaria U-Net B19/A38/C149 |
| Report 07 (`docs/month2/07_astra_level_structure_and_anchors.md`) | `review/astra-07-report` | `82bff35726fb70af9987081802ece3ef718e42bd` | Revisione del coordinatore (con §9); tip branch `54d8df95844984d285e2433c0fbaf1f17fc78190` controllato per diff |
| Pacchetto Astra A0–A3 (`replay_k0_verification*`) | `mission/astra-level` | `86a925af61accdef0b235ac084827cf639c1040c` | Verifica replay S5/S6 U-Net e GLU |
| Report 08 (`docs/month2/08_astra_median9_second_pass_analysis.md`) | `review/astra-08-median9` | `7cb7bfdc6430a2731c924a31d90897a3aa52a89f` | Analisi `mean9`/`median9` su T9 storico |
| Pacchetto Astra second-pass (`astra_level.py`, tabelle) | `astra/median9-second-pass` | `04e3b68addc891034f0840c2cbaa141d5ef8c4ac` | Codice e tabelle sorgente `median9` |
| Proposta 09 (`docs/month2/09_astra_opus_synthesis_and_s7_proposal.md`) | `review/astra-opus-s7-proposal` | `9daf1585ba2bd9d264372e62565606a800fbefe5` | **Consultiva, NON approvata** (Enrico ha incluso GLU); discrepanze numeriche verificate sotto |
| Campagna S7 training + continuazione `glu-on-s1` | `s7/c149-free-campaign` | `f792b73388d43f008e8979f1aad0d1fa961cb559` (train), `501d41a0be44709c69cf320737bd8d263fdb5216` (ext) | U-Net `O-C149-on/off-s0` riusati da `d6332000491ba0e8983b9e7fa09b34badb191ea4` |
| Evaluator S7 pubblicato e matrice 12 run | `s7/c149-free-campaign` | `acc8e2b4071578a584accfbde8c3c1a2cb5e2bf1` | `scripts/38_s7_eval.py`, `month2_s7_c149_eval_matrix.json`, `s7_eval_plan.md` |
| Ultimo handoff remoto S7 (`git fetch --all` verificato) | `origin/s7/c149-free-campaign` | `4c61db5181d2ee109397df4fce6c735aa4bb838b` | `s7_eval_handoff.md` (aggiornamento 21 nov ~15:15 EST); nessun commit remoto successivo |

**Codice ispezionato direttamente:** `src/combustion_recon/models/{glu_reconstructor.py,unet.py,baselines.py,month2_baseline_adapters.py}`, `external/GLU_Reconstruction/src/models.py`, `src/combustion_recon/data/{masks.py,month2_observations.py}`, `src/combustion_recon/evaluation/{metrics.py,month2_evaluator.py,s7_eval_plan.py,astra_level.py}`, `src/combustion_recon/training/{sparse_scientific_loop.py,sparse_dataset.py,sparse_view.py}`, `scripts/{15_gappypod_nested_coverage_eval.py,17_s6_sparse_eval_and_swap.py,38_s7_eval.py,39_s7_write_eval_matrix.py}`, `configs/models/glu_minimal.yaml`, `configs/experiments/{month2_sparse_protocol.yaml,month2_baselines.yaml,month2_s7_c149_eval_matrix.json}`, `data/processed/month2_dense_unet_cache_c149/geometry.npz` (solo coordinate spaziali `x, y`, nessun dato `T` caricato).

### 0.2 Discrepanze numeriche confermate nella Proposta 09 (da NON propagare)

Il controllo incrociato tra `06_opus_data_and_training_variability.md` (`4f0e2637`), il CSV sorgente `global_by_setting.csv` (`5373aa3c`) e `09_astra_opus_synthesis_and_s7_proposal.md` (`9daf1585`) conferma che Report 06 combacia con i CSV sorgente, mentre Proposta 09 contiene errori di trascrizione da non copiare:

- `O-A38-off-s0 / s1` VAL `last`: Proposta 09 riporta `49.84 / 49.53 K`; valore sorgente = **`49.48 / 49.17 K`**.
- `O-A38-on / off-s0` `seen best_seen`: Proposta 09 riporta `51.33 / 57.99 K`; valore sorgente = **`50.85 / 56.59 K`**.
- `O-C149-on / off-s0` `seen`: Proposta 09 riporta `best_seen 39.79 / 42.68; last 37.91 / 40.03 K`; valore sorgente = **`best_seen 39.31 / 41.28; last 40.75 / 42.88 K`** (`best 45.42 / 41.99 K`).
- `O-C149-on-s0` VAL `20%`: Proposta 09 riporta `39.21 K`; valore sorgente = **`40.01 K` (`best`), `37.93 K` (`best_seen`), `38.28 K` (`last`)**.
- Walltime Slurm: Proposta 09 indicava `8h`, mentre `mit_normal_gpu` ha `MaxTime = 6h` (verificato in `s7_training_handoff.md`).

---

## 1. Revisione del confronto S7 temperature-free (C149, K=0)

### 1.1 Giudizio complessivo sul protocollo e sulle consegne previste

Il disegno sperimentale della campagna S7 (`U-Net masked` e `GLU adattato deterministico` × `p_on/p_off` × seed `0/1/2` su `C149 = 2980` frame, velocità TRAIN `5%`, `K=0`, normalizzatore congelato fit-only 760, `ScientificPresentationStream` da 95 update/epoca, stesso scheduler e stessa regola di early-stop) e l'evaluator pubblicato (`scripts/38_s7_eval.py`, SHA `acc8e2b4`) **sono strutturalmente adeguati** per un confronto difendibile tra famiglie e tra `p_on/p_off`, **a condizione che vengano chiusi i cinque punti materiali elencati in §1.2** prima di aggregare gli esiti e formulare la scelta del seguito.

Due aspetti positivi meritano menzione esplicita:
1. **Uniformità dell'evaluator:** rivalutare anche i due run U-Net `O-C149-on/off-s0` con `38_s7_eval.py` (invece di riciclare i CSV O5) garantisce uno stesso SHA evaluator per tutti i 12 run. È una lieve deviazione dalla dicitura di `roadmapsparse.md` (*«senza inferenza duplicata»*), scientificamente preferibile purché si registri anche la parità contro O5 (`5373aa3c`).
2. **Separazione tra metriche del loop e valutazione finale:** `month2_s7_c149_eval_matrix.json` marca esplicitamente `loop_metrics_are_not_s7_eval: true` e `loop_metric_source: "training_loop_not_s7_eval"`. Nessuna scelta va tratta da tali numeri.

### 1.2 Classificazione delle lacune: bloccanti, limiti accettabili e pulizia opzionale

#### A. Errori / lacune materiali **BLOCCANTI** prima dell'analisi finale e della scelta del seguito
1. **Completamento verificato delle valutazioni finali (12/12 run):** attendere chiusura Slurm (`COMPLETED 0:0`), readback completo (`81` CSV per run, o `54` con forward condiviso a `weights_sha256` identico su `glu-off-s0` e `glu-on-s2`), `PARITY_OK` del job `29984793` (`glu-on-s1`) e presenza di `COMPLETE.json` (zero `GATE_FAILED.json`) su tutti i 12 run di `29984795` e `29984812`.
2. **Controllo mirato di invarianza al batch su `AdaptedDeterministicGLU` (GPU):** vedi §1.3. Lo scarto osservato nello smoke GLU VAL (`0.025–0.040 K`) non prova contaminazione tra esempi, ma il solo `PARITY_OK` a batch identici (`0.0 K`) **non** dimostra l'assenza di dipendenza impropria tra esempi del batch. Prima di considerare il contratto di inferenza GLU chiuso va eseguito il test mirato di §1.3.3.
3. **Congelamento scritto della regola di lettura/selezione prima di leggere gli esiti finali:** committare `configs/experiments/month2_s7_c149_protocol.yaml` (con la regola definita in §1.4) *prima* di aggregare le tabelle finali S7, come già previsto da `roadmapsparse.md` §S7.
4. **Completamento dei tre controlli comparativi non ancora sottomessi:**
   - **Snapshot-swap `best@5%` replica 0 (seen/VAL) sui 12 run S7:** *rilievo di codice verificato* — `scripts/17_s6_sparse_eval_and_swap.py` codifica `model_seed=0` in modo fisso alle righe `239`, `390` e `416`. Il nuovo wrapper S7 guidato dalla matrice **deve** propagare il vero `model_seed` (`0/1/2`) e `weights_sha256` nei tag/CSV e nella provenienza, evitando collisioni o etichette errate.
   - **Residuo swap `GappyPOD p_on/off` S6 (A38):** da chiudere etichettando esplicitamente pool `A38` e SHA dell'oggetto fit.
   - **Baseline fittate su `C149` (`2980` frame):** `GlobalMean`, `ParamMean`, `GappyPOD R=32` (`p_on` e `p_off`), fittate su `C149` con artefatti versionati, mantenendo congelati gli iperparametri `λ` del Mese 2 (`p_off: 1e-3`, `p_on: 1e-2`) senza ri-tuning silenzioso su VAL; le baseline `A38` restano separate come riferimento storico.
5. **Coerenza sulle metriche regionali in presenza di maschere vuote:** in `src/combustion_recon/evaluation/metrics.py` (`_safe_iou_dice`), quando `union == 0` (sia target sia predizione vuoti nella regione) viene restituito `1.0`, mentre `src/combustion_recon/evaluation/astra_level.py` restituisce `NaN` con conteggi espliciti. Nell'aggregazione S7 occorre riportare obbligatoriamente `n_valid_frames`, `n_empty_true`, `n_empty_pred` e `n_both_empty` per ogni regione (`Hot`, `Peak`, `Jet`, `NonHot`) e documentare la convenzione di `metrics.py` per evitare confronti disallineati con Astra.

#### B. Limiti strutturali **ACCETTABILI** (da dichiarare esplicitamente, NON bloccanti)
- **Tre seed (`0, 1, 2`) a `train_mask_seed = 20261118` fisso:** misurano stabilità rispetto a inizializzazione e ordine di presentazione condizionatamente a un unico stream di maschere TRAIN, non l'intera distribuzione di training o di mascheratura.
- **Ruolo di `VAL123` (3 condizioni):** ha già guidato selezione checkpoint, `ReduceLROnPlateau`, early-stop e (storicamente) l'idea `median9`; è evidenza di sviluppo, non validazione indipendente. `TEST` resta bloccato.
- **Normalizzatore congelato su `760` frame (`A38`) anche per `C149`:** scelta intenzionale per isolare pool/famiglia/seed e riusare `O-C149-on/off-s0` senza cambio di preprocessing.
- **Coperture `1%` e `20%`:** valutano generalizzazione fuori densità da modello addestrato al solo `5%`, non prestazioni ottimali per ciascuna densità.
- **Continuazione `S7-C149-glu-on-s1` (SHA `501d41a`, epoche `1000→1080`):** stessa ricetta e stessa regola di early-stop, prima tranche preservata byte-per-byte, `best.pt` invariato a epoca `630`. Documentata e accettabile.

#### C. Pulizia **OPZIONALE** (non bloccante)
- Estrarre da Slurm `sacct` e `progress.json` i tempi reali di training (wall-clock e GPU-h) per tutti i 12 run da includere nelle run card S7.
- Aggiungere un micro-benchmark uniforme di latenza a `batch_size=1` al `5%` (e VRAM picco) per U-Net e GLU (attualmente `38_s7_eval.py` registra tempi per setting a `batch_size=8` e VRAM picco, ma non latenza singola istantanea per entrambe le famiglie).

---

### 1.3 Valutazione tecnica dell'anomalia GLU VAL nello smoke (batch / numerica fp32)

#### 1.3.1 Evidenza osservata negli handoff
- **Parity completo al 5% (`29968643`, 11 run, stessi batch del loop `batch_size=8`):** `11/11 PARITY_OK`, scarto macro massimo **`0.0 K`** e scarto per-frame massimo **`0.0 K`** (bit-identico) su `train_fit 760`, `seen 220`, `VAL 123 × 3` repliche, per entrambe le famiglie.
- **Smoke su sottoinsieme (`29967397`, `--smoke-frames 6`):** sul medesimo sottoinsieme rispetto al loop/parity, U-Net resta `0 / ≤ 9e-6 K`, GLU `train/seen` resta `≤ 1e-5 K`, mentre **GLU VAL** raggiunge **`0.025–0.040 K`** su singoli frame rispetto al loop (`2.5e-2 K` rispetto al parity, media `2e-3 K`).

#### 1.3.2 Audit statico del codice (`glu_reconstructor.py`, `external/GLU_Reconstruction/src/models.py`, `38_s7_eval.py`)
L'ispezione riga per riga **non rileva alcun operatore architetturale che mescoli esempi lungo la dimensione batch `B`**:
- **Zero `BatchNorm`** in tutto `AdaptedDeterministicGLU` e nei moduli upstream utilizzati: le sole normalizzazioni attive sono `RMSNorm` e `LayerNorm` lungo l'ultima dimensione `D = 32` (`external/GLU_Reconstruction/src/models.py:203, 235–237, 302, 309, 1706–1708`).
- **Geometria fissa di dominio:** `domain_center` e `domain_L` sono buffer persistenti ricavati da `geometry.npz`; la vecchia `compute_isotropic_geometry` dipendente dal batch/call-site è stata rimossa e solleva `RuntimeError` (`src/combustion_recon/models/glu_reconstructor.py:873–876`).
- **Valutazione e numerica:** `38_s7_eval.py` imposta `model.eval()` (riga `520`), `allow_tf32 = False` su matmul e cuDNN (righe `434–435`), `dropout = 0.0`, `_attach_p` indicizza `ds.p_norm[int(meta["cache_index"])]` (non la posizione nel sottoinsieme), e tutte le trasformazioni (`_sort_sensors_by_site`, `CrossAttention`, `nn.TransformerEncoder`, `cdist`, `topk_distance_then_site_index`, `_topk_aggregate`, `SpatiallyAwareCLSFusion`, `head`) sono matematicamente indipendenti per ciascun elemento `b` del batch.

**Perché lo smoke ha colpito `val_eval` in modo diverso da `train_fit` e `seen_train_eval`?**
In `scripts/38_s7_eval.py:231` (`_smoke_positions`):
- `train_fit` e `seen_train_eval` usano `keep_frame_1000=False` → **6 frame** (1 solo batch `B=6`, replica 0: 6 istanze totali).
- `val_eval` usa `keep_frame_1000=True` → ai 6 indici equidistanti (`[0, 24, 49, 73, 98, 122]`) si aggiungono i 3 indici di frame 1000 (`20, 61, 102`) → **9 frame** → Batch 0 (`B=8`) + **Batch 1 (`B=1`, indice 122)**, valutati su 3 repliche = **27 istanze** (contro i 15 batch `B=8` + 1 batch `B=3` del loop).

**Due meccanismi GPU/fp32 plausibili (senza contaminazione architetturale):**
1. **Ipotesi primaria — inversione di parità a 1 ULP in `torch.cdist` sul bordo `top_k = 16` (`glu_reconstructor.py:348, 371`):** `geometry.npz` definisce una griglia cartesiana regolare `128×128` (`dx`, `dy` costanti), generando frequenti **parità geometriche esatte** di distanza euclidea `(±Δx, ±Δy)` tra query e sensori. `torch.cdist(Y, sensor_coords)` usa il default `compute_mode='use_mm_for_euclid_dist_if_necessary'` (GEMM cuBLAS batched del tipo `sqrt(||y||² + ||s||² − 2 y sᵀ)`). Al variare della dimensione del batch (`B=1` vs `B=3` vs `B=8`), cuBLAS può selezionare un kernel diverso con arrotondamenti fp32 di ~1 ULP (`~1e-7`), rompendo una parità esatta *prima* che `topk_distance_then_site_index` applichi il tie-break per `site_index`. Se due sensori equidistanti cadono a cavallo del rango `k=16` vs `17` vicino a un forte gradiente di temperatura (`~1000 K`, `σ_T = 331.7 K`), lo scambio di un singolo vicino produce un salto discreto sul pixel e uno scarto di RMSE per-frame di `0.01–0.04 K` su pochi frame, mantenendo la media bassa (`~2e-3 K`).
2. **Ipotesi secondaria — fast-path / dispatch kernel di `nn.TransformerEncoder` in `eval()` (`external/GLU_Reconstruction/src/models.py:220, 249–253`):** `build_transformer_stack` usa `nn.TransformerEncoder` con `enable_nested_tensor=True` (default), `norm_first=False`, `dropout=0.0`, 4 layer × 2 stack (`65` token latenti e `819` token sensore). Il cambio di kernel SDPA/cuBLAS tra `B=1` e `B=8` introduce differenze fp32 amplificate attraverso 8 layer e `σ_T = 331.7 K` (di solito `1e-5–1e-3 K`, ma può anche interagire con la sensibilità del decoder).

**Perché il solo PASS macro (`≤ 0.05 K`) non basta a chiudere la questione:**
Un eventuale bug di dipendenza fra esempi dentro un fast-path PyTorch/CUDA potrebbe compensarsi nella media macro o restare sotto `0.05 K`, violando però l'indipendenza del singolo frame in inferenza (critica a `batch_size=1` o su lotti riordinati). Non bisogna né diagnosticare contaminazione senza prova, né liquidarla col solo gate macro.

#### 1.3.3 Controllo mirato da affidare all'esecutore (economico, su 1 L40S, senza rifare la campagna)
Sui medesimi checkpoint GLU (`glu-on-s0` e `glu-off-s0`, `best.pt`) e sui frame VAL dello smoke che hanno mostrato scarti `0.025–0.040 K` al `5%`:

1. **Test A — Invarianza ai compagni di batch a `B=8` fisso (test decisivo anti-contaminazione):** valutare lo stesso frame `x` in posizione `0` dentro tre batch di uguale lunghezza `B=8` e stesso `N=819`:
   - `Batch_self = [x, x, x, x, x, x, x, x]`
   - `Batch_real = [x, y_1, ..., y_7]` (7 frame VAL reali)
   - `Batch_extreme = [x, z_1, ..., z_7]` (7 frame sintetici con velocità azzerate o moltiplicate per `10×` e `p` estremi)
   - **Esito atteso (assenza di contaminazione):** `max_abs(pred_K(x)) ≤ 1e-5 K` (atteso `0.0 K`) tra tutti e tre i casi. Qualunque scarto `> 1e-4 K` a `B=8` fisso indica una **dipendenza funzionale impropria tra esempi del batch** (**BLOCCANTE** per GLU; verificare disabilitando `enable_nested_tensor=False` / fast-path `TransformerEncoder`).
2. **Test B — Dipendenza da `B` e isolamento della causa (`B ∈ {1, 3, 6, 8}` sul solo frame `x` ripetuto/ritagliato):** registrare `max_abs(pred_K)`, `Δ RMSE_frame`, e il conteggio di mismatch su `top_idx` (`decoder._record_top_idx = True`), confrontando il `cdist` di default con `torch.cdist(..., compute_mode='donot_use_mm_for_euclid_dist')` (e, opzionalmente, CPU fp32).
   - Se **Test A = PASS (`0.0 K`)** e **Test B** mostra che lo scarto `0.025–0.040 K` compare solo al cambio di `B` (es. per tie-flip su `top_idx` eliminato da `donot_use_mm_for_euclid_dist`, o per puro dispatch GEMM/SDPA): la dipendenza tra esempi è esclusa, le valutazioni S7 a `B=8` sono **valide**, lo scarto è documentato come sensibilità numerica fp32 al `batch_size`, e per il codice futuro (Step 2.7-B / deployment) si raccomanda `compute_mode='donot_use_mm_for_euclid_dist'`.

---

### 1.4 Regola pre-specificata di lettura e selezione per S7 (da congelare prima di leggere gli esiti)

Da registrare in `configs/experiments/month2_s7_c149_protocol.yaml` **prima** dell'aggregazione S7. **Non introduce alcuna soglia fisica arbitraria in kelvin per forzare un vincitore.**

1. **Unità di lettura obbligatoria (mai scegliere il seed migliore):** ogni configurazione `(famiglia, p)` è sempre descritta dall'insieme completo dei **3 seed (`s0, s1, s2`)**: valori individuali, media aritmetica sui 3 seed e intervallo `[min, max]`.
2. **Endpoint primario e assi co-primari (con `seen220` e `VAL123` rigorosamente separati):**
   - **Primario:** `VAL123 @ 5%` sul checkpoint `best` (media sulle 3 repliche di maschera della media uniforme tra le 3 condizioni VAL di `RMSE [K]`).
   - **Co-primari strutturali e di livello (stesso `best@5%`, `VAL123` e `seen220` separati):** `centered-RMSE`, `|bias|`, `bias`, `MAE`. La `centered-RMSE` ha peso critico nella scelta del backbone per lo Step 2.7, perché le osservazioni `T` aggiunte in 2.7-A recuperano già gran parte del livello (`|bias|`), mentre la qualità strutturale resta in capo al backbone.
   - **Condizioni singole e regioni:** tabella per singola condizione (`0NH3_1.05`, `20NH3_1.1`, `40NH3_0.9` su VAL; tutte le 20 su `seen220`, evidenziando la condizione peggiore), metriche regionali (`Hot ≥ 2100 K` RMSE e IoU, `Peak ≥ 2200 K` RMSE, `Jet < 1800 K` RMSE, `NonHot` RMSE, sempre con `n_valid_frames` e conteggi di maschere vuote).
   - **Generalizzazione, stabilità e controlli:** coperture fuori densità `1%` e `20%` (`best`), deriva `best → last` e diagnostica `best_seen_diagnostic`, `snapshot-swap` (`aligned` vs `swap` vs `donor_ref`), confronto con baseline `C149` (`GlobalMean`, `ParamMean`, `GappyPOD R=32`), e costo computazionale reale (GPU-h training da `sacct`, tempo eval, VRAM picco, numero parametri).
3. **Contrasti appaiati a parità di seed `s ∈ {0, 1, 2}`:**
   - Effetto famiglia: `Δ_arch(s, p) = metric(U-Net, p, s) − metric(GLU, p, s)` (separatamente per `p_on` e `p_off`).
   - Effetto condizionamento (appaiato sulla stessa inizializzazione entro famiglia): `Δ_p(s, fam) = metric(fam, p_on, s) − metric(fam, p_off, s)`.
4. **Regola decisionale: scegliere una sola famiglia vs mantenere entrambe in Step 2.7-B:**
   - **Si seleziona una sola famiglia `fam*` per Step 2.7-B** *solo se* valgono **congiuntamente** tutte e quattro le condizioni seguenti:
     1. *Dominanza primaria e strutturale coerente sui seed:* `fam*` ha media 3-seed migliore su **entrambi** `VAL123 @ 5% RMSE` e `VAL123 @ 5% centered-RMSE` (nel regime `p` primario, senza inversione nell'altro `p`), con **segno concorde in 3/3 seed appaiati** e **intervalli `[min, max]` sui 3 seed disgiunti** almeno su `VAL 5% RMSE` oppure su `VAL 5% centered-RMSE`, e senza sconfitta netta `3/3` su `seen220 @ 5%`.
     2. *Assenza di fallimento localizzato:* `fam*` non presenta un collasso sistematico sui 3 seed su una singola condizione VAL, né su `Hot IoU`, `Jet RMSE`, `1%` o `20%` che l'altra famiglia eviti in modo consistente `3/3`.
     3. *Superamento dei controlli:* `fam*` batte chiaramente `GappyPOD C149` e supera lo snapshot-swap (`swap ≫ aligned`).
     4. *Non inferiorità di costo:* `fam*` ha costo computazionale (GPU-h / VRAM) inferiore o comparabile all'altra famiglia.
   - **Si mantengono ENTRAMBE le famiglie in Step 2.7-B** se si verifica **almeno una** delle seguenti condizioni:
     1. Gli intervalli `[min, max]` dei 3 seed delle due famiglie si sovrappongono su `VAL 5% RMSE` o su `VAL 5% centered-RMSE` (inversione di ranking fra seed).
     2. Emerge un **compromesso strutturale o di regime** coerente sui 3 seed (es. una famiglia vince su `VAL RMSE` / `|bias|` o `1%`, mentre l'altra vince `3/3` su `VAL centered-RMSE`, `Jet RMSE` o `seen220`).
     3. La famiglia più costosa (`GLU`) mostra un vantaggio riproducibile sui 3 seed in accuratezza o struttura, mentre la più economica (`U-Net`) resta competitiva a costo molto minore (frontiera di Pareto accuratezza–costo).
5. **Lettura di `p_on` vs `p_off`:** i due regimi rappresentano due scenari applicativi distinti (parametri operativi `NH3/phi` noti vs ignoti). Valutare `Δ_p(s, fam)` separatamente su `VAL 5%`, `seen 5%`, `20%` (dove in Opus `p_off` degradava) e `|bias|` vs `centered-RMSE`; non scartare silenziosamente `p_off` (vedi matrice 2.7-B in §3.5).

### 1.5 Esiti da attendere prima di qualsiasi scelta S7
Non scegliere alcun vincitore ora. Attendere:
1. `PARITY_OK` del job `29984793` (`glu-on-s1`) e `COMPLETE.json` su **tutti i 12 run** (`29984795` + `29984812`);
2. Esito del **Test A / Test B** di invarianza al batch GLU (§1.3.3);
3. Commit del protocollo congelato `configs/experiments/month2_s7_c149_protocol.yaml`;
4. Output verificati di: (a) snapshot-swap 12 run, (b) residuo swap `GappyPOD` S6, (c) baseline `C149` (`GlobalMean`, `ParamMean`, `GappyPOD R=32 p_on/off`);
5. Pacchetto aggregato finale (tabelle macro/per-condizione/seed/regionali/costi + figure su tutte le VAL) e report `docs/month2/10_s7_c149_model_comparison.md`.

---

## 2. Specifica raccomandata per lo sweep K (Step 2.7-A)

Obiettivo: quantificare, sulle **predizioni congelate** dei 12 run S7 e sulle baseline abbinate, la curva errore vs numero di osservazioni ideali `K ∈ {0, 1, 4, 9, 16, 32}`, senza tuning di layout su VAL e senza sweep di stimatori. Rumore e geometrie di termocoppie reali restano fuori da questo blocco.

### 2.1 Siti annidati puramente geometrici (`t_anchors_nested_geom_v1`) e decisione su `T9`

**Decisione esplicita sul riuso di `T9` storico (`{32, 64, 95}²`):** **RIUSARE esattamente `T9` come prefisso `K = 9`**.
- *Motivazione scientifica:* (1) mantiene continuità esatta (`0.000 K` a `K=9`) con i Report Astra 07 e 08; (2) impedisce qualunque selezione post-hoc o sospetto di evitamento guidato da VAL dei due siti sul bordo getto destro `(32, 95)` e `(95, 95)` evidenziati nel Report 08.
- *Distinzione rispetto al codice baseline storico:* `src/combustion_recon/data/month2_observations.py` (`nested_temperature_anchor_permutation`, riga `150`) generava una permutazione casuale diversa per ogni frame/seed (motivo dello scarto storico `70.60 K` vs `76.24 K` tra Report 02 e Astra `T9`). Qui si raccomanda un **layout spaziale fisso e condiviso tra tutti i frame, condizioni e repliche**, coerente con sonde fisse nel dominio e direttamente collegato a `T9`.

**Costruzione deterministica a zero uso di target/errori/VAL (`balanced_fps_interior_16_111`):**
- Dominio candidato: griglia interna `[16, 111]²` (margine uniforme di `16` pixel dai bordi, pari a metà passo di `T9`, evitando sonde sul pixel di contorno `0` o `127`).
- Criterio di selezione (Farthest-Point Sampling bilanciato in coordinate di griglia pixel `(r, c)`): a ogni passo si aggiunge il punto che massimizza la tupla `(d_min, d_2nd_min, d_mean, -r, -c)` rispetto ai punti già selezionati (il tie-break su seconda distanza minima e media elimina il clustering in alto a sinistra del tie-break lessicografico puro).
- **Fase 1 (`K = 1..9`):** seme al centro del dominio `(64, 64)`, FPS bilanciato ristretto a `T9 = {32, 64, 95}²` → garantisce `K=1 = {(64, 64)}` e `K=9 = T9` esattamente.
- **Fase 2 (`K = 10..32`):** prosecuzione dello stesso FPS bilanciato senza restrizioni su `[16, 111]²` → completa naturalmente la griglia duale sfalsata `{16, 48, 80, 111}²` più 7 punti mediani perimetrali a passo `16` px.

**Manifest esatto verificato (`1 ⊂ 4 ⊂ 9 ⊂ 16 ⊂ 32`):**

| Prefisso | Nuovi siti `(row, col)` aggiunti (ordine esatto) | `cover_max` [px] | `cover_mean` [px] | `min_pair` [px] | Quadranti `(TL,TR,BL,BR)` |
|---:|---|---:|---:|---:|---|
| `K = 1` | `(64, 64)` | `90.51` | `48.97` | — | `(0, 0, 0, 1)` |
| `K = 4` | `(32, 32), (32, 95), (95, 32)` | `89.10` | `28.79` | `44.55` | `(1, 1, 1, 1)` |
| `K = 9` (`= T9`) | `(95, 95), (32, 64), (64, 32), (64, 95), (95, 64)` | `45.25` | `18.90` | `31.00` | `(1, 2, 2, 4)` |
| `K = 16` | `(16, 16), (111, 111), (16, 111), (111, 16), (16, 48), (111, 48), (48, 111)` | `35.78` | `13.60` | `22.63` | `(3, 4, 4, 5)` |
| `K = 32` | `(48, 16), (48, 48), (80, 111), (111, 80), (16, 80), (80, 16), (48, 80), (80, 48), (80, 80), (16, 32), (111, 32), (32, 111), (32, 16), (64, 111), (111, 64), (16, 64)` | `22.63` | `9.41` | `16.00` | `(7, 8, 7, 10)` |

*Nota di trasparenza su `K = 4`:* qualunque sottoinsieme di 4 punti di una griglia `3×3` che contenga il centro `(64, 64)` lascia inevitabilmente scoperto il quarto angolo `(95, 95)` fino a `K=5`; il set `K=4` risultante ha comunque esattamente 1 sonda per quadrante (`(1, 1, 1, 1)`) e include in modo non selezionato 1 sito della colonna `95`. Il manifest è serializzato in JSON con SHA256 congelato prima di ogni valutazione.

### 2.2 Stimatori ammessi sulle predizioni congelate (nessuno sweep di stimatori)

Sulla **stessa predizione base congelata** `T_pred` in kelvin (`best@5%` dei 12 run S7, 3 repliche di maschera velocità al `5%`, `seen220` e `VAL123` separati), definiti i residui alle `K` ancore `r_k = T_true(s_k) − T_pred(s_k)` (`k = 1..K`):

- **`free` (`K = 0` e controllo a ogni `K`):** `δ = 0`.
- **`AO_K` (`tau = 1`):** `δ_AO(K) = (∑_{k=1}^K r_k) / (K + 1)` (`= 0` a `K = 0`; identità algebrica `δ_AO(K) = [K/(K+1)] · δ_mean(K)`).
- **`mean_K` (media senza shrink):** `δ_mean(K) = (1/K) ∑_{k=1}^K r_k` per `K ≥ 1`; `δ_mean(0) = 0`.
- **`median_K` (mediana ordinaria senza shrink):** `δ_median(0) = 0`; per `K` dispari (`1, 9`), statistica d'ordine centrale `r_((K+1)/2)`; per `K` pari (`4, 16, 32`), media aritmetica dei due valori centrali `(r_(K/2) + r_(K/2 + 1)) / 2` (`numpy.median`). Identità di controllo: a `K = 1`, `median_1 == mean_1` bit-a-bit (e `AO_1 = 0.5 · mean_1`).
- **`oracle_offset` (solo diagnostico, mai nei ranking deployabili):** `δ_oracle = mean_{Ω_all}(T_true − T_pred)` (usa l'intera `T_true`).

Nessuna variante extra (niente medie troncate, Huber, pesi spaziali o tuning di `τ`).

### 2.3 Baseline abbinate a dati e informazione dichiarati

Stesse 3 repliche velocità `5%`, stesso manifest `t_anchors_nested_geom_v1` (`K ∈ {0, 1, 4, 9, 16, 32}`), `seen220` e `VAL123` separati:

| Gruppo | Modello | Pool fit dichiarato | Info ingresso | Trattamenti `K` |
|---|---|---|---|---|
| **Primario C149** | `GlobalMean` | `C149` (`2980`) | `p_off`, `0%` vel, `K` `T` | `free`, `AO_K`, `mean_K`, `median_K` |
| **Primario C149** | `ParamMean` | `C149` (`2980`) | `p_on`, `0%` vel, `K` `T` | `free`, `AO_K`, `mean_K`, `median_K` |
| **Primario C149** | `GappyPOD R=32` (`p_on/off`) | `C149` (`2980`) | `p_on/off`, `5%` vel, `K` `T` | `free`, `AO_K`, `mean_K`, `median_K` (`λ` congelati Mese 2: `p_off 1e-3`, `p_on 1e-2`) |
| **Primario C149** | `GappyPOD + T-block` (`p_on/off`) | `C149` (`2980`) | `p_on/off`, `5%` vel, `K` `T` | MAP congiunto (`w_T = 0.5` congelato da `02_baseline_validation.md`; verifica `T-block(0) == free`) |
| **Riferimento storico** | `GappyPOD R=32` / `+T-block` | `A38` (`760`) | `p_on/off`, `5%` vel, `K` `T` | Colonna separata ponte verso Report 02/07/08 (mai mescolata con C149) |
| **Controllo debole separato** | `T_only_mean(K)`, `T_only_median(K)` | nessuno | `0%` vel, `p_off`, `K` `T` | Campo costante (`K ≥ 1`, tabella separata) |

### 2.4 Metriche e gestione rigorosa del set off-anchor al variare di `K`

**Il problema del set off-anchor variabile:** al crescere di `K` (`0 → 32`), l'insieme off-anchor `Ω_off(K) = Ω \ S_K` cambia cardinalità (`16384 → 16352`) e rimuove proprio specifici pixel (inclusi siti ad alto gradiente su `col=95`). Confrontare `off-anchor(K=0)` con `off-anchor(K=32)` mescola il miglioramento del modello con il cambio del dominio di valutazione — distorsione ancora più grave nello Step 2.7-B, dove le reti con `T` in ingresso possono azzerare l'errore alle ancore.

**Soluzione adottata (4 domini spaziali espliciti e dichiarati a ogni `K`):**
1. **`Ω_common = Ω \ S_32` (`16352` pixel fissi per TUTTI i `K ∈ {0, 1, 4, 9, 16, 32}`):** **dominio primario per le curve in funzione di `K` e per confrontare 2.7-A con 2.7-B.** Supporto di pixel 100% invariante lungo tutto lo sweep e sempre disgiunto da tutte le ancore osservate.
2. **`Ω_all` (campo intero, `16384` px):** metrica globale standard della tesi (coincide con S7 `free` a `K = 0`).
3. **`Ω_off(K) = Ω \ S_K` (`16384 − K` px):** errore strettamente fuori ancora a `K` fissato (valido per confronti *orizzontali* tra metodi allo stesso `K`).
4. **`S_K` (alle ancore, `K` px per `K ≥ 1`):** diagnostica di fedeltà/overfitting locale alle ancore.

Su `Ω_common`, `Ω_all` e `Ω_off(K)`: `RMSE`, `MAE`, `bias`, `|bias|`, `centered-RMSE`; su `S_K`: `RMSE` e `MAE`; inoltre `offset_error = δ_applied − δ_oracle`, `|offset_error|`, metriche regionali (`Hot`, `Peak`, `Jet`, `NonHot` con `n_valid_frames` e conteggi vuoti), e diagnostica di peggioramento (`%` condizioni e frame peggiorati rispetto a `free`, peggioramento massimo in K).

**Gate e criteri di completamento 2.7-A (anche con esito negativo):**
- *Gate tecnici obbligatori:* (i) parità `K = 0` contro i CSV finali S7 entro `0.05 K` macro (`0.0 K` atteso, assenza riferimento = FAIL); (ii) invarianza della `centered-RMSE` su `Ω_all` e su `Ω_common` sotto tutti gli shift scalari (`free`, `AO_K`, `mean_K`, `median_K`, `oracle` entro `1e-5 K`); (iii) `median_1 == mean_1` e `GappyPOD+T-block(K=0) == GappyPOD_free`.
- *Completamento valido anche con esito negativo/saturo:* lo sweep è concluso quando tutte le tabelle e curve sono verificate su tutti i 12 run S7 + baseline C149, anche qualora `median_K` saturi già a `K = 9`, `mean_K` superi `median_K` a `K = 16/32`, o i guadagni su `seen` risultino nulli/leggermente negativi dove il bias `free` è già piccolo.

---

## 3. Disegno minimo informativo dei training con `T` in ingresso (Step 2.7-B)

### 3.1 Domanda scientifica
Le correzioni scalari di 2.7-A lasciano invariata la `centered-RMSE`. La domanda di 2.7-B è se addestrare la rete ricevendo direttamente le stesse `K` osservazioni `(x_k, y_k, T_k)` migliori la **struttura spaziale** (`centered-RMSE`, `Jet RMSE`, `Hot IoU`, errore su `Ω_common`) oltre `free + median_K`, oltre `GappyPOD+T-block` e oltre una baseline spaziale economica sui residui. **Un esito negativo (nessun guadagno strutturale riproducibile) è un risultato scientifico pienamente ammissibile e va previsto esplicitamente.**

### 3.2 Rappresentazione di valori `T`, coordinate e validità (zero leakage)

Stesso normalizzatore `T` fit-only congelato sui `760` frame (`train_norm_stats_month2_fit.json`), stessi `5%` velocità, stessi `K` e siti in valutazione (`t_anchors_nested_geom_v1`). I pixel non osservati hanno `T_obs_norm = 0.0` e `M_T = 0` (nessuna interpolazione della `T` nascosta).

1. **U-Net (`SparseUNetAssisted`, 9 canali in ingresso, `b=32`):** estensione minima del modello S7 a 7 canali `[vi_obs_norm, vj_obs_norm, M_v, x_scaled, y_scaled, nh3_norm, phi_norm]` aggiungendo esattamente 2 canali `[T_obs_norm_filled0, M_T]`, con `M_T ∈ {0, 1}` e `∑ M_T = K` (schema già previsto in `month2_observations.py`). Le coordinate sono già presenti su tutta la griglia (`x_scaled, y_scaled`); `p_off` azzera `nh3_norm, phi_norm` esattamente come in S7.
2. **GLU (`AdaptedGLUAssisted`, solo se GLU è selezionato o mantenuto con U-Net):** rispetta sia il contratto no-padding di GLU sia il vincolo di `roadmap_month2.md` §2.7-B (*«non scartare le ancore nel k-NN delle velocità; usare token con tipo di variabile»*):
   - *Encoder:* token velocità (`N_v = 819`) = `pos_embed(q_v) + val_v([vi_obs_norm, vj_obs_norm]) + type_emb[0] + W_p p`; token temperatura (`N_T = K`) = `pos_embed(q_T) + val_T([T_obs_norm]) + type_emb[1] + W_p p` (stesse coordinate isotrope normalizzate di dominio `q = 2(r−c)/L`, `W_p p = 0` per `p_off`). La cross-attention verso i latenti globali riceve la concatenazione `[vel_tokens; T_tokens]` (a `K = 0` si riduce esattamente a `[vel_tokens]`, zero padding).
   - *Decoder:* i soli `N_v = 819` token velocità restano nel ramo `top_k = 16` di `ThesisSoftDomainAdaptiveReconstructor` (le ancore `T` non possono mai estromettere vicini di velocità); si aggiunge un ramo locale dedicato alle ancore `T` (aggregazione pesata sulla distanza tramite `cdist(..., compute_mode='donot_use_mm_for_euclid_dist')` su `k_T = min(K, 8)` o attenzione morbida su tutti i `K ≤ 32` token `T`, sommato residuamente a `local_pre` prima di `SpatiallyAwareCLSFusion`, identicamente `0` quando `K = 0`).
   - *Batch omogenei in `K`:* un unico `K` condiviso per tutti gli 8 esempi del batch in training e in valutazione → zero padding per GLU e batching identico per U-Net.

**Test di contratto obbligatori prima di ogni training 2.7-B:** (i) mutare `T` dove `M_T = 0` o `vi/vj` dove `M_v = 0` → output bit-identico (`0.0`); (ii) mutare `T` dove `M_T = 1` (`K ≥ 1`) → output cambia; (iii) `K = 0` gira senza `NaN` e ignora `T_true`; (iv) `p_off` bit-identico al variare di `p`, `p_on` sensibile; (v) per GLU: invarianza ai compagni di batch (`Test A = 0.0 K`) e invarianza del `top_idx` velocità rispetto a `K`.

### 3.3 Training unico a `K` variabile, distribuzione `TRAIN-K` e criterio di selezione

Per valutare l'intero sweep `{0, 1, 4, 9, 16, 32}` senza un training separato per ogni `K`:
- **Distribuzione `TRAIN-K` pre-dichiarata:** per ciascun batch (8 frame), campionare un `K` omogeneo in modo **uniforme** `P_train(K) = 1/6` su `{0, 1, 4, 9, 16, 32}` tramite generatore deterministico persistito nel checkpoint (riproducibilità esatta del resume). Scelta priva di pesi accordati su VAL, che espone equamente il modello al regime `K = 0`, ai regimi poveri (`1, 4`) e a quelli ricchi (`9, 16, 32`).
- **Siti `T` in TRAIN vs EVAL (scelta anti-memorizzazione):**
  - *In VAL / `seen` (eval):* esclusivamente il manifest fisso `t_anchors_nested_geom_v1` di 2.7-A (`1 ⊂ 4 ⊂ 9 ⊂ 16 ⊂ 32`), con le stesse 3 repliche di velocità `5%`.
  - *In TRAIN:* a ogni frame con `K > 0`, campionare `K` siti distinti in modo uniforme casuale sulla griglia `128×128` (con `train_anchor_mask_seed` deterministico ricampionato ogni epoca scientifica, analogo a `train_mask_seed` delle velocità). *Perché non fissare i 32 siti in TRAIN:* fissarli porterebbe la rete a memorizzare 32 pixel specifici anziché apprendere l'assimilazione spaziale generale di `(x_k, y_k, T_k)`, introducendo inoltre un vantaggio asimmetrico di memorizzazione rispetto a `median_K` / `GappyPOD+T-block` / RBF sullo stesso manifest. *(La specializzazione a 32 porte fisse è un problema ingegneristico diverso, eventualmente valutabile solo in seguito.)*
- **Criterio di selezione checkpoint pre-dichiarato (prima del training):**
  - A ogni epoca di valutazione (ogni `10` epoche scientifiche; da profilare il costo eval 6× su VAL123 prima del freeze):
    `J_select = (1/6) ∑_{K ∈ {0, 1, 4, 9, 16, 32}} VAL123_5pct_OmegaCommon_RMSE(K)`
    (media uniforme sui 6 `K` della RMSE media per condizione su `VAL123 @ 5%` calcolata sul dominio invariante **`Ω_common = Ω \ S_32`**, mediata sulle 3 repliche).
  - *Perché `Ω_common` nella selezione:* impedisce a selezione e early-stop di premiare una rete che si limita a copiare i `K` pixel di ingresso (`at-anchor → 0 K`) senza migliorare i `16352` pixel nascosti.
  - Si salvano comunque separatamente le curve per ogni singolo `K` (in particolare `K = 0`), `last.pt` e `best_seen_diagnostic` (calcolato sulla stessa media 6-`K` su `seen220` replica 0, solo diagnostico).

### 3.4 Training da zero vs fine-tuning, preservazione di `K=0` e baseline spaziale economica

- **Scelta motivata: TRAINING DA ZERO** con la stessa ricetta C149 di S7 (`95` update/epoca, AdamW `lr=1e-3`, `wd=1e-4`, warmup `950`, batch `8`, `fp32` no-TF32, `train_mask_seed=20261118`, `model_seed = shuffle_seed ∈ {0, 1, 2}`).
  - *Motivazioni:* (1) parità architetturale tra U-Net (innesto `7→9` su `conv1`) e GLU (nuovo branch token `T` + decoder locale `T`), senza confondere differenze di famiglia con scelte ad hoc di warm-start/LR; (2) co-adattamento pulito fin dai primi layer tra velocità e `T`; (3) i checkpoint S7 `free` restano **intatti al 100%** come controlli diretti `free(K=0)` e `free + median_K`, rendendo l'effetto dell'ingresso `T` pulitamente interpretabile a parità di ricetta e seed.
  - *Gestione di `K = 0`:* non assumere né dichiarare che `assisted(K=0)` equivalga a `free(K=0)`. Riportare esplicitamente `Δ_K0 = assisted(K=0) − original_free(K=0)` sui 3 seed; se il multi-task `K` degrada `K = 0`, documentarlo onestamente (in uso operativo a `K = 0` si invoca il checkpoint `free` originale). Un fine-tuning o testa residua su backbone `free` congelato resta eventualmente un follow-up separato *solo* se il training da zero dovesse fallire o degradare gravemente `K = 0`.
- **Baseline spaziale economica pertinente (controllo strutturale obbligatorio):** **`free + median_K + RBF_resid`**.
  - Sulla predizione congelata `T_free` (e su `ParamMean` / `GlobalMean`), per `K ≥ 4` si interpolano i residui centrati alle ancore `e_k = r_k − δ_median(K)` mediante kernel ridge gaussiano isotropo in coordinate normalizzate `q ∈ [-1, 1]²`:
    `T_hat(q) = T_free(q) + δ_median(K) + ∑_{k=1}^K w_k exp(−||q − q_k||² / (2 ℓ²))`, con `(Φ + α I) w = e` (per `K = 0` è `T_free`; per `K = 1` è `T_free + r_1`).
  - Gli iperparametri `(ℓ, α)` sono scelti **senza toccare VAL** (griglia fissa piccola `3×3` selezionata via Leave-One-Condition-Out sulle sole 20 condizioni `TRAIN` `reference760`, oppure scala geometrica fissata a priori + regolarizzazione su TRAIN). Costo: pochi secondi su CPU (`K ≤ 32`).
  - *Valore scientifico:* verifica direttamente se la rete addestrata con `T` in ingresso apprenda accoppiamento non lineare velocità–temperatura oltre una semplice interpolazione liscia dei `K` residui di temperatura sopra la predizione `free`.

### 3.5 Matrice minima raccomandata dei nuovi training assisted (2.7-B)

Tutti i run condividono: pool `C149 (2980)`, normalizzatore `760` congelato, velocità TRAIN `5%` (`train_mask_seed=20261118`), `K ~ Unif({0, 1, 4, 9, 16, 32})`, valutazione completa su `seen220` e `VAL123` (`5%`, 3 repliche, manifest `t_anchors_nested_geom_v1`, tutti i 6 `K`).

| Scenario post-S7 | Fase | Configurazioni | Seed | Nuovi training | Scopo e criterio di arresto |
|---|---|---|---:|---:|---|
| **Caso A: 1 sola famiglia scelta (`fam*`)** | **Unica** | `fam* × {p_on, p_off}` | `0, 1, 2` | **6** | Specchio esatto dei 6 run free S7 di `fam*`; misura sui 3 seed sia il guadagno strutturale (`centered-RMSE`, `Ω_common`) sia se `T` in ingresso rende `p` ridondante. *(Opzione ultra-snella se budget stretto: prima 3 run sul solo `p` primario; altri 3 sull'altro `p` solo se emerge guadagno su `Ω_common`/`centered-RMSE`.)* |
| **Caso B: entrambe le famiglie mantenute** | **Stadio B1** | `{U-Net, GLU} × p_primario` | `0, 1, 2` | **6** | Confronta le due famiglie su 3 seed contro `free+median_K` e `RBF_resid` nel regime `p` primario (fissato dalla lettura S7 prima del lancio). **Stop se nessuna batte `median_K`/`RBF_resid` strutturalmente** (risparmia 6 run). |
| **Caso B (condizionale)** | **Stadio B2** | Famiglia/e sopravvissute × `p_secondario` | `0, 1, 2` | **0, 3 o 6** | **0** se esito negativo; **3** se domina 1 sola famiglia (totale 9); **6** se entrambe restano competitive/Pareto (totale 12, specchio completo di S7). |

### 3.6 Criteri di completamento di Step 2.7-B (incluso esito negativo)
Step 2.7-B è completato quando:
1. Tutti i run della matrice autorizzata superano i test di contratto/isolamento (§3.2), chiudono con la regola di early-stop congelata e superano la valutazione completa sui 6 `K` (`seen220` e `VAL123`, 3 repliche) con readback verificato.
2. Il confronto appaiato sui 3 seed è tabulato su `Ω_common`, `Ω_all`, `Ω_off(K)`, `S_K`, `centered-RMSE`, `|bias|` e metriche regionali contro: (a) `original_free(K=0)`, (b) `free + median_K`, (c) `free + median_K + RBF_resid`, (d) `GappyPOD+T-block C149`.
3. **Esiti negativi validi (chiusura pulita senza inseguire ritocchi ad hoc):**
   - *Nessun guadagno strutturale:* la rete assisted abbatte l'errore alle ancore `S_K` ma non migliora in modo consistente sui 3 seed `Ω_common RMSE` o `centered-RMSE` rispetto a `free + median_K` o `RBF_resid` → conclusione solida: le osservazioni sparse di `T` correggono essenzialmente livello e residuo liscio; il post-processing su modello `free` congelato è preferibile o Pareto-ottimale.
   - *Degradazione a `K = 0`:* l'assisted migliora a `K = 16/32` ma perde a `K = 0` → si documenta il compromesso multi-`K` e si preserva il checkpoint `free` originale per `K = 0`.

---

## 4. Costi misurati vs incognite esplicite e piano operativo (preparabile ora vs post-S7)

### 4.1 Costi misurati (solo da artefatti verificati) e incognite esplicite

| Voce | Valore misurato / fonte verificata | Stato |
|---|---|---|
| Training `SparseUNet` C149 (`O-C149-on/off-s0`, L40S) | `3h 45m` (`660 ep`) e `4h 00m` (`710 ep`) ≈ `~20.3–20.4 s/ep`; VRAM `0.49–0.60 GB` (`reused_unet_verification.json`, `05_sparse_reconstruction.md`) | **Misurato** |
| Training `GLU` storico S5 A38 (L40S, vecchio DataLoader) | `2h 20m` (`890 ep`) e `2h 35m` (`980 ep`); VRAM `4.3 GB` (`05_sparse_reconstruction.md`, `s7_training_handoff.md`) | **Misurato (A38 storico)** |
| Estensione `S7-C149-glu-on-s1` (`ep 1000→1080`, L40S) | `36 min` per `80` epoche scientifiche + `8` eval ≈ `~27 s/ep` (`s7_eval_plan.md`) | **Misurato (parziale)** |
| Parity eval S7 (`38_s7_eval.py`, 15 forward, L40S) | `~540–570 s` per run (U-Net ≈ GLU, CPU-bound su `train_fit`); VRAM `0.17 GB` (U-Net) / `0.51 GB` (GLU) (`s7_eval_plan.md`) | **Misurato** |
| Eval full S7 (81 setting, `29984795`/`29984812`) | Estrapolazione da parity: `~85–95 min/run` (`56–67 min` se forward condiviso) (`s7_eval_plan.md`) | **Stima estrapolata (in corso)** |
| Sweep post-hoc tipo Astra `median9` (L40S / CPU) | `~2 min/ckpt` L40S (`29686178_0..3`) + `29 s` baseline (`29686179`) (`08_astra_median9_second_pass_analysis.md`) | **Misurato (per 1 K, A38)** |
| Baseline suite Mese 2 A38 (CPU, include HGB/OraclePOD) | `1918 s` (`~32 min`), peak RSS `10.4 GiB` (`02_baseline_validation.md`) | **Misurato (A38)** |
| **Walltime totale dei 10 nuovi training S7 C149 (4 U-Net + 6 GLU)** | Non committato nei file Git correnti | **Incognita esplicita** (estrarre da `sacct`/`progress.json`) |
| **Latenza `batch_size=1` al `5%` (U-Net vs GLU)** | Non profilata in modo uniforme per entrambe le famiglie | **Incognita esplicita** |
| **Fit + sweep 6-`K` `GappyPOD` su `C149` (`2980` frame, CPU)** | Non ancora eseguito (`3.92×` matrice SVD di A38) | **Incognita esplicita** |
| **Durata epoca, costo eval 6-`K` su VAL123 ed epoche di early-stop in 2.7-B (U-Net e GLU assisted)** | Non ancora profilati (soprattutto il ramo decoder `T` di GLU e il costo eval `6×` ogni 10 epoche) | **Incognita esplicita** (obbligatorio smoke di profiling 2 ep + 1 eval prima di fissare walltime Slurm) |

### 4.2 Separazione netta: preparabile ORA vs decisioni che richiedono S7 finale

| Preparabile ORA (nessuna dipendenza dall'esito S7) | Richiede S7 finale completo e verificato (`COMPLETE.json` 12/12 + controlli) |
|---|---|
| 1. Eseguire **Test A / Test B** di invarianza al batch GLU su 1 L40S (§1.3.3, secondi, sui checkpoint `glu-on/off-s0` già esistenti). | 1. Lettura delle tabelle finali S7 e applicazione della regola congelata (§1.4) per decidere **1 famiglia vs 2 famiglie** e `p` primario per 2.7-B. |
| 2. Scrivere e congelare in Git `configs/experiments/month2_s7_c149_protocol.yaml` (regola di selezione §1.4) *prima* di leggere i CSV finali S7. | 2. Esecuzione dello sweep 2.7-A sui 12 checkpoint S7 finali `best@5%` e verifica del gate di parità `K = 0` contro i CSV S7. |
| 3. Preparare/testare il wrapper S7 di **snapshot-swap** (correggendo il `model_seed=0` fisso di `scripts/17`), la pipeline **baseline C149** (`GlobalMean`, `ParamMean`, `GappyPOD R=32`), e chiudere il residuo swap `GappyPOD` S6. | 3. Profiling GPU (smoke 2 ep + 1 eval) e lancio dei training assisted 2.7-B secondo la matrice scelta (Caso A o Caso B). |
| 4. Congelare il manifest geometrico `t_anchors_nested_geom_v1` (§2.1), la maschera `Ω_common` (`16352` px), il codice dello sweep 2.7-A (`AO_K`, `mean_K`, `median_K`, `GappyPOD+T-block`, `RBF_resid`) e i relativi test unitari CPU. | 4. Decisione su eventuale Stadio B2 (nel Caso B) e qualunque decisione futura sul lock di `TEST` (resta non autorizzato). |
| 5. Implementare e testare su CPU i contratti di isolamento 2.7-B (canali `T` U-Net; token `T` + decoder locale `T` GLU solo se si vuole ridurre lead time; sampler batch `K` omogeneo) **senza lanciare training GPU**. | |

---

## 5. Limiti di interpretazione espliciti

1. **Tre seed ≠ distribuzione generale:** 3 seed (`0/1/2`) a `train_mask_seed` fisso descrivono stabilità nel campione rispetto a inizializzazione e ordine di presentazione, non una stima statistica della varianza di popolazione né della sensibilità alla mascheratura TRAIN.
2. **Tre maschere ≠ tre training:** le 3 repliche di maschera in valutazione stimano la variabilità di campionamento spaziale a pesi fissi, non sostituiscono repliche di training.
3. **Contaminazione storica di `VAL123`:** le 3 condizioni VAL hanno guidato selezione `best`, `ReduceLROnPlateau`, early-stop e la formulazione di `median9`. I risultati su VAL restano evidenza di sviluppo.
4. **Stato di `TEST`:** `TEST` non è autorizzato in questa fase; inoltre le condizioni TEST del Mese 1 erano già state esaminate in precedenza e non costituiscono una verifica cieca immacolata.
5. **Confronti tra pool diversi:** confrontare reti `C149` con baseline `A38` confonde effetto dei dati ed effetto del modello; le baseline fittate su `C149` sono obbligatorie per isolare il solo modello.
6. **Nessuna pretesa universale su non linearità o reti neurali:** ogni vantaggio osservato vale nel perimetro specifico di dataset, griglia `128×128`, osservazioni ideali `5%` e protocollo valutato; un esito competitivo di `GappyPOD+T-block` o `RBF_resid` è scientificamente informativo quanto un successo neurale.
7. **Osservazioni numeriche ideali ≠ termocoppie reali:** i `K` valori di `T` in ingresso sono campioni puntuali esatti dal campo DNS/numerico; questo studio **non** valida termocoppie fisiche (assenti rumore di misura, inerzia termica, bias radiativo, intrusività fluidodinamica e vincoli di montaggio).
