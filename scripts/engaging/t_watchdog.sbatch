#!/bin/bash
# Autonomous continuation helper for the assisted trainings (CPU only, light). Slurm holds a batch job that was requeued more than
# MaxBatchRequeue (=5) times (preemptions + wall-time continuations). This loop releases such held t_assist tasks so they resume from
# their own last.pt via the normal launcher; it NEVER submits, resubmits, requeues errors or touches run directories.
#SBATCH --job-name=t_watchdog
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=3-00:00:00
set -uo pipefail
LOG="${T_WATCHDOG_LOG:-$HOME/official_thesis_t_assisted_runs/slurm/watchdog.log}"
mkdir -p "$(dirname "$LOG")"
while true; do
  mapfile -t HELD < <(squeue -u "$USER" -h -r -o '%A|%j|%T|%r' | grep 't_assist' | grep -E 'PENDING\|.*Held' | cut -d'|' -f1)
  for j in "${HELD[@]:-}"; do
    [[ -n "$j" ]] || continue
    echo "[$(date -Is)] releasing held t_assist job $j: $(scontrol show job "$j" 2>/dev/null | grep -oE 'Reason=[^ ]+|Restarts=[0-9]+' | tr '\n' ' ')" >> "$LOG"
    scontrol release "$j" >> "$LOG" 2>&1
  done
  # stop when no t_assist job is left
  squeue -u "$USER" -h -r -o '%j' | grep -q 't_assist' || { echo "[$(date -Is)] no t_assist job left: exiting" >> "$LOG"; exit 0; }
  sleep 600
done
