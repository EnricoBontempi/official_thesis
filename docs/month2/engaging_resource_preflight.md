# Engaging — ricognizione risorse (preflight sola lettura)

**Data/ora ricognizione:** 2026-11-17, circa 16:28–16:31 America/New_York (UTC−05:00)  
**Host login osservato:** `login007` (ORCD Engaging via SSH host `engaging`)  
**Ambito:** fattibilità cluster per `official_thesis` / diagnostico U-Net. **U4 non autorizzato; nessun job, sbatch/salloc/srun, trasferimento o modifica remota in questa attività.**  
**Progetto HONF:** citato solo come occupazione osservata degli stessi account utente; nessun accesso ai suoi processi/nodi.

---

## 1. Accesso

| Controllo | Esito |
|---|---|
| SSH non interattivo | OK con `ssh -F ~/.ssh/config engaging` (BatchMode). Il client di default senza `-F` fallisce su file di config di sistema. |
| Credenziali / chiavi | Non stampate. Autenticazione già configurata; nessuna interazione aggiuntiva richiesta in questa sessione. |
| Utente remoto | `bontempi` |

---

## 2. Account, partizioni, QoS e significato di “slot”

### 2.1 Associazioni verificate (`sacctmgr`)

| Account | Partition | QoS consentiti |
|---|---|---|
| `mit_amf_advanced_gpu` | `mit_normal_gpu` | `mit_amf_advanced_gpu`, `normal` |
| `mit_amf_advanced_cpu` | `mit_normal` | `mit_amf_advanced_cpu`, `normal` |
| `mit_general` | *(nessuna partition fissa)* | `normal` (MaxSubmit associazione 500) |

### 2.2 Corrispondenza “4 normal + 4 preemptable”

La descrizione utente **non** equivale a “8 GPU idle sul cluster”. Corrisponde a **limiti MaxTRESPU per utente** su QoS distinti:

| Etichetta utente | QoS verificato | MaxTRESPU GPU (per utente) | Note |
|---|---|---|---|
| “normal” (AMF GPU) | `mit_amf_advanced_gpu` | **`gres/gpu=4`** (+ cpu=64, mem=1T) | MaxWall QoS 2-00:00:00; flag `PartitionTimeLimit`, `OverPartQOS`, `DenyOnLimit`. Sulla partition `mit_normal_gpu` il **MaxTime di partition è 6:00:00**. |
| “preemptable” | `mit_preemptable` | **`gres/gpu=4`** (+ cpu=1024, mem=4T) | MaxSubmitPU=256. |
| *(non confondere)* | `mit_normal_gpu` (QoS di default della partition) | **`gres/gpu=2`** | Solo 2 GPU se si usa questo QoS senza AMF advanced. |
| QoS `normal` | `normal` | *(vuoto / non impostato in MaxTRESPU)* | Usato dai job HONF osservati su partition preemptable (vedi §3). |

**Conclusione sulla nomenclatura:** gli “slot” riferiti sono **cap di concorrenza GPU per utente/QoS**, non GPU libere del cluster e non necessariamente “job slots” generici.

---

## 3. Occupazione personale osservata (incl. HONF)

`squeue -u bontempi` al momento della ricognizione:

| Stato | N | Account | Partition | QoS | GRES | Nomi job (pattern) |
|---|---:|---|---|---|---|---|
| RUNNING | 2 | `mit_general` | `mit_preemptable` | `normal` | `gres/gpu:h100:1` | `dbg4b-*` |
| PENDING | 14 | `mit_general` | `mit_preemptable` | `normal` | `gres/gpu:h100:1` | `dbg4b-*` |
| Qualsiasi su AMF / `mit_normal_gpu` | **0** | — | — | — | — | — |

- Tutti i 16 job personali sono coerenti con il checkout HONF `ModularDT-next-dbg4b-*` (solo identificazione da nome; **nessuna modifica/cancellazione/sospensione**).
- Questi job **non** usano il QoS `mit_preemptable` né l’account `mit_amf_advanced_gpu`: quindi **non consumano** i MaxTRESPU `gres/gpu=4` di quei due QoS.
- Headroom teorico per `official_thesis` sotto AMF normal GPU: **4/4 GPU MaxTRESPU libere** (nessun job AMF osservato).
- Headroom teorico sotto QoS `mit_preemptable`: **4/4 libere** rispetto a quel QoS (HONF non lo sta usando).
- Occupazione HONF su `mit_general`+`normal`+preemptable: 2 GPU H100 in run + 14 in coda; il QoS `normal` non espone un MaxTRESPU GPU numerico in `sacctmgr` — il vincolo pratico lì è coda/priorità/cluster, non il “4 preemptable” AMF-style.

**Nessun login su compute node; nessun `scancel`/`scontrol hold`.**

---

## 4. GPU, walltime, preemption (verificato vs non disponibile)

### 4.1 Partition candidate

| Partition | MaxTime | PreemptMode partition | TRES GPU dichiarati (totali partition) |
|---|---|---|---|
| `mit_normal_gpu` | **6:00:00** | **OFF** | b200=96, h200=104, h100=**4**, l40s=212 (totale gres/gpu=416 su 79 nodi) |
| `mit_preemptable` | **2-00:00:00** | **REQUEUE** | a100, b200, h100, h200, l4, l40s, rtx_a6000, rtx_pro_6000, … (totale gres/gpu=1262 su 677 nodi) |

Memoria nodo: esempi osservati ~1–2 TiB RAM (`RealMemory` ~1.0–2.0e6 MiB a seconda del tipo); **non** è una garanzia di allocazione mem-per-job senza richiesta esplicita.

### 4.2 Politica preemption (cluster config, verificata)

Da `scontrol show config`:

- `PreemptType=preempt/partition_prio`
- `PreemptMode=REQUEUE` (globale; la partition normal_gpu sovrascrive a OFF)
- `PreemptParameters=preempt_youngest_first`
- `JobRequeue=1`, `MaxBatchRequeue=5`
- `PreemptExemptTime=00:00:00`

**Non disponibile / non verificato qui:** tempi medi di coda, probabilità di preemption per un job specifico, fairshare dettagliato, garanzia di tipo GPU al submit.

### 4.3 Disponibilità cluster ≠ quota personale

Snapshot `sinfo` (stati nodo, non GPU libere contate per-Gres fine):

- `mit_normal_gpu`: quasi tutti i nodi GPU in `mixed`/`alloc`; **pochissimi (o nessun) nodo GPU pienamente idle** al momento dello snapshot. H100 sulla normal_gpu risultano **molto scarsi** (ordine di 4 GPU totali in partition).
- `mit_preemptable`: molti nodi `mixed`/`alloc`; qualche idle senza GRES; tipi H100/H200/A100/L40S presenti ma condivisi.

Quindi: anche con 4/4 MaxTRESPU AMF liberi, **l’attesa in coda può essere non nulla e non prevedibile** da questa sola ricognizione.

---

## 5. Candidati per una singola U-Net (1 GPU)

Ordine consigliato **solo come fattibilità scheduler**, non come decisione di budget U4 (budget/architettura non toccati):

1. **Primario — non preemptable:**  
   `-A mit_amf_advanced_gpu -p mit_normal_gpu -q mit_amf_advanced_gpu`  
   - 1× GPU; preferire **H200 / L40S / B200** (H100 quasi assenti su questa partition).  
   - Wall ≤ **6 h** (limite partition).  
   - Preemption: OFF sulla partition.  
   - Quota personale: 4 GPU MaxTRESPU, attualmente inutilizzate da job AMF.

2. **Alternativa — preemptable con QoS dedicato:**  
   `-p mit_preemptable -q mit_preemptable` (account da confermare al submit: tipicamente allineato alle policy ORCD; le associazioni esplicite GPU AMF sono sulla normal_gpu).  
   - MaxTRESPU 4 GPU su quel QoS, al momento non consumati dai job HONF osservati.  
   - Wall fino a 2 giorni; **REQUEUE** possibile.  
   - Coda già affollata da altri carichi (inclusi i pending HONF su QoS diverso).

3. **Da evitare per tesi se si vuole isolamento da HONF:** stesso schema `mit_general` + QoS `normal` su preemptable — condivide coda/pattern con HONF e non usa i cap “4+4” AMF/preemptable QoS.

---

## 6. Software e storage (sola lettura)

### 6.1 Checkout `official_thesis`

- **Assente** nei path noti e in ricerca shallow (`find` maxdepth 3–4 sotto `$HOME`, `~/projects`, `~/orcd/pool`, `~/orcd/scratch`).
- Sotto `~/projects` risultano solo tree **ModularDT / HONF** (incluso `ModularDT-next-dbg4b-be98485`). **Nessuno SHA di `official_thesis` sul cluster.**

### 6.2 Python / PyTorch

| Risorsa | Osservato |
|---|---|
| `/usr/bin/python3` | 3.6.8 (login; non adatto al stack tesi) |
| `~/envs/honf` | Conda env **HONF**: Python 3.11.18, **torch 2.8.0+cu128**, CUDA build 12.8; `torch.cuda.is_available()` = **False** sul login (atteso). ~7 GiB. **Progetto separato — non modificato.** |
| Moduli CUDA | `cuda/12.9.1`, `12`/`13` vari; default `cuda/13.1.0`; `cudnn/9.8.0.87-cuda12` |
| Modulo `pytorch` | **Non trovato** (`module spider pytorch` vuoto) |
| `miniforge` | Moduli community disponibili (es. 23.11 → 26.7) — installazione **non** eseguita |
| Env dedicato `official_thesis` | **Non trovato** |

Verifica CUDA runtime su GPU = **pendente** (richiede allocazione; vietata in questo incarico).

### 6.3 Quote e spazio

Da `~/orcd/.quota` (aggiornato 2026-11-17 16:19 EST) e `df`/`quota`:

| Spazio | Uso | Limite | Note |
|---|---:|---:|---|
| HOME (`/home`) | ~68.6 GiB | 200 GiB | NFS; adatto a codice/env piccoli, non a cache/raw grandi |
| POOL (`~/orcd/pool` → `/orcd/pool/005/bontempi`) | **0** | **1024 GiB** | Vuoto; candidato cache/checkpoint |
| SCRATCH (`~/orcd/scratch` → `/orcd/scratch/orcd/008/bontempi`) | **0** | **1024 GiB** | Vuoto; candidato scratch training |

Nessuna scansione ricorsiva indiscriminata oltre listing shallow e `du -sh` sui mount utente.

---

## 7. Quattro livelli di “disponibilità” (distinti)

| Livello | Stato al 2026-11-17 ~16:30 ET |
|---|---|
| **Quota/limite teorico** | AMF GPU QoS: 4 GPU concurrent; preemptable QoS: 4 GPU; normal_gpu QoS di default: 2 GPU; wall normal_gpu 6 h / preemptable 2 d. |
| **Slot personali utilizzabili ora** | AMF `mit_amf_advanced_gpu`: **4 liberi** (0 job). QoS `mit_preemptable`: **4 liberi** rispetto a quel QoS. HONF occupa 2 H100 running (+14 pending) sotto `mit_general`/`normal`, **senza** consumare i due cap da 4 sopra. |
| **GPU libere sul cluster** | Normal_gpu satura/mixed; H100 normal_gpu scarso. Preemptable ha capacità aggregata maggiore ma nodi prevalentemente mixed/alloc. **Non** “8 GPU idle riservate”. |
| **Attesa in coda** | **Non prevedibile** da questa ricognizione; pending HONF e carico cluster già evidenti su preemptable. |

---

## 8. Cosa manca per smoke GPU e profiling breve

Per uno smoke (import torch + `cuda.is_available` + matmul piccolo) e un profiling U-Net breve su 1 GPU, **dopo autorizzazione separata**, servirebbero ancora:

1. **Checkout** `official_thesis` su Engaging (path preferibile sotto POOL o HOME) e SHA allineato al branch di lavoro.
2. **Ambiente Python dedicato** (miniforge/venv) con dipendenze tesi — oppure policy esplicita se riusare stack esistenti; **non** assunto il riuso di `~/envs/honf`.
3. **Dati/cache** Month-2 sul cluster (raw Arrow e/o cache densa ~277 MiB+): trasferimento non eseguito; POOL/SCRATCH vuoti e sufficienti in quota.
4. **Allocazione autorizzata** (`salloc`/`sbatch` 1 GPU) — **non** fatta qui; senza di essa lo smoke CUDA resta pendente.
5. **Scelta hardware** (H200 vs L40S vs B200 vs H100) e walltime entro 6 h se si resta su `mit_normal_gpu`, da discutere col budget U4 (fuori scope).
6. Script/moduli di lancio e conferma account/QoS al submit (AMF vs preemptable).

---

## 9. Incertezze e impedimenti

- Quale account è ammesso con `-q mit_preemptable` per questo utente non è stato forzato con un submit di prova (vietato); le associazioni GPU esplicite sono su `mit_amf_advanced_gpu` ↔ `mit_normal_gpu`.
- I job HONF con QoS `normal` su preemptable potrebbero avere altri limiti non visibili in MaxTRESPU QoS; non investigati oltre `sacctmgr`/`squeue`.
- Contatori “GPU idle” per tipo non sono stati derivati da un inventario fine GresAlloc; solo stati nodo `sinfo`.
- Nessun checkout/env/dati `official_thesis` sul cluster → **non ready-to-run**.
- Login node senza GPU utile per CUDA → smoke GPU bloccato fino ad allocazione autorizzata.
- U4 / budget / schedule / scelta hardware: **ancora in discussione; training non autorizzato.**

---

## 10. Comandi di sola lettura usati (riepilogo)

`ssh -F ~/.ssh/config engaging`; `sacctmgr show assoc/qos/user`; `scontrol show partition|config|node`; `squeue -u $USER`; `sinfo`; `quota` / `df` / `cat ~/orcd/.quota`; listing shallow e `find -maxdepth`; `module avail|spider`; lettura versioni Python/torch **senza** job GPU.

**Non usati:** `sbatch`, `salloc`, `srun`, `scancel`, login compute, install, git clone remoto, rsync/scp di dati.
