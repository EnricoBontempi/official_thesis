#!/bin/bash
# Submit smoke, then the scientific sweep, inside Cursor's share of the common budget:
# at most one mit_normal_gpu / mit_amf_advanced_gpu L40S, and no preemptable job.
# CPU jobs are not part of that GPU count. The scientific GPU array is afterok on the
# smoke array and uses concurrency 1, so the two never run together.
set -euo pipefail

TIP="${1:?checkout}"
SHA="${2:?sha}"
cd "$TIP"
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || { echo "REFUSE: HEAD != $SHA" >&2; exit 5; }
[[ -f docs/month2/temperature_study_handoff.md ]] || { echo "REFUSE: Sonnet handoff is not in this checkout" >&2; exit 2; }

BUSY="$(squeue -u "$USER" -h -t R,PD -o '%P %a' | awk '$1=="mit_normal_gpu" && $2=="mit_amf_advanced_gpu" { n++ } END { print n+0 }')"
if (( BUSY + 1 > 4 )); then
  echo "REFUSE: $BUSY normal Advanced GPU jobs already queued or running; this sweep asks for 1 (ceiling 4)" >&2
  exit 7
fi

SMOKE_OUT="$HOME/official_thesis_k_sweep_smoke_v2"
FULL_OUT="$HOME/official_thesis_k_sweep_v1"
BASE_SMOKE="$HOME/official_thesis_k_sweep_baselines_smoke_v2"
BASE_FULL="$HOME/official_thesis_k_sweep_baselines_v1"
AGG_OUT="$HOME/official_thesis_k_sweep_package_v1"
mkdir -p "$SMOKE_OUT/slurm" "$FULL_OUT/slurm" "$BASE_SMOKE/slurm" "$BASE_FULL/slurm" "$AGG_OUT/slurm"

GPU_SMOKE=$(sbatch --parsable --account mit_amf_advanced_gpu --partition mit_normal_gpu --qos mit_amf_advanced_gpu \
  --time 00:40:00 --array=0-1%1 --output "$SMOKE_OUT/slurm/neural-%A_%a.out" --error "$SMOKE_OUT/slurm/neural-%A_%a.err" \
  --export=ALL,EXPECT_SHA="$SHA",K_TIP="$TIP",K_OUT="$SMOKE_OUT",K_MODE=smoke,K_RUN_IDS="S7-C149-unet-on-s1|S7-C149-glu-on-s0" \
  "$TIP/scripts/engaging/temperature_k_neural.sbatch")

CPU_SMOKE=$(sbatch --parsable --account mit_amf_advanced_cpu --partition mit_normal --qos mit_amf_advanced_cpu \
  --time 00:30:00 --output "$BASE_SMOKE/slurm/base-%j.out" --error "$BASE_SMOKE/slurm/base-%j.err" \
  --export=ALL,EXPECT_SHA="$SHA",K_TIP="$TIP",K_OUT="$BASE_SMOKE",K_MODE=smoke \
  "$TIP/scripts/engaging/temperature_k_cpu.sbatch")

GPU_FULL=$(sbatch --parsable --dependency="afterok:${GPU_SMOKE}" --kill-on-invalid-dep=yes \
  --account mit_amf_advanced_gpu --partition mit_normal_gpu --qos mit_amf_advanced_gpu \
  --time 03:00:00 --array=0-11%1 --output "$FULL_OUT/slurm/neural-%A_%a.out" --error "$FULL_OUT/slurm/neural-%A_%a.err" \
  --export=ALL,EXPECT_SHA="$SHA",K_TIP="$TIP",K_OUT="$FULL_OUT",K_MODE=full,K_RUN_IDS= \
  "$TIP/scripts/engaging/temperature_k_neural.sbatch")

CPU_FULL=$(sbatch --parsable --dependency="afterok:${CPU_SMOKE}" --kill-on-invalid-dep=yes \
  --account mit_amf_advanced_cpu --partition mit_normal --qos mit_amf_advanced_cpu \
  --time 02:00:00 --output "$BASE_FULL/slurm/base-%j.out" --error "$BASE_FULL/slurm/base-%j.err" \
  --export=ALL,EXPECT_SHA="$SHA",K_TIP="$TIP",K_OUT="$BASE_FULL",K_MODE=full \
  "$TIP/scripts/engaging/temperature_k_cpu.sbatch")

AGG=$(sbatch --parsable --dependency="afterok:${GPU_FULL}:${CPU_FULL}" --kill-on-invalid-dep=yes \
  --account mit_amf_advanced_cpu --partition mit_normal --qos mit_amf_advanced_cpu \
  --time 00:40:00 --output "$AGG_OUT/slurm/agg-%j.out" --error "$AGG_OUT/slurm/agg-%j.err" \
  --export=ALL,EXPECT_SHA="$SHA",K_FORWARD_SHA="$SHA",K_TIP="$TIP",K_NEURAL="$FULL_OUT",K_BASE="$BASE_FULL",K_DEST="$AGG_OUT" \
  "$TIP/scripts/engaging/temperature_k_aggregate.sbatch")

echo "gpu_smoke=$GPU_SMOKE partition=mit_normal_gpu account=mit_amf_advanced_gpu qos=mit_amf_advanced_gpu gres=gpu:l40s:1 array=0-1%1 out=$SMOKE_OUT"
echo "cpu_smoke=$CPU_SMOKE partition=mit_normal account=mit_amf_advanced_cpu qos=mit_amf_advanced_cpu out=$BASE_SMOKE"
echo "gpu_full=$GPU_FULL dependency=afterok:$GPU_SMOKE partition=mit_normal_gpu account=mit_amf_advanced_gpu qos=mit_amf_advanced_gpu gres=gpu:l40s:1 array=0-11%1 out=$FULL_OUT"
echo "cpu_full=$CPU_FULL dependency=afterok:$CPU_SMOKE partition=mit_normal account=mit_amf_advanced_cpu qos=mit_amf_advanced_cpu out=$BASE_FULL"
echo "aggregate=$AGG dependency=afterok:$GPU_FULL:$CPU_FULL out=$AGG_OUT"
