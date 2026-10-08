#!/usr/bin/env bash
# Usage: bash run_families_seed.sh <seed> <arm>
#   arm: none | trex
set -Eeuo pipefail

SEED="${1:?Usage: $0 <seed> <arm>}"
ARM="${2:?Usage: $0 <seed> <arm>}"

case "$ARM" in
  none|trex) ;;
  *) echo "ERROR: arm must be 'none' or 'trex'" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

: "${HEAT1D_DATA_DIR:=$HOME/Heat1D}"
: "${HEAT2D_DATA_DIR:=$HOME/Heat2D}"
: "${BEAM2D_DATA_DIR:=$HOME/Beam 2D}"

datadir() {
  case "$1" in
    heat1d) printf '%s\n' "$HEAT1D_DATA_DIR" ;;
    heat2d) printf '%s\n' "$HEAT2D_DATA_DIR" ;;
    beam2d) printf '%s\n' "$BEAM2D_DATA_DIR" ;;
  esac
}

for f in trex_warmstart_campaign.py nisq.py nisq_core.py qem_utils.py; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 2; }
done

for fam in heat1d heat2d beam2d; do
  d="$(datadir "$fam")"
  for dof in 8 16 32; do
    [[ -f "$d/${fam}_${dof}DOF_K.npz" && -f "$d/${fam}_${dof}DOF_f.npz" ]] || \
      { echo "ERROR: missing $fam K/f DOF=$dof in $d" >&2; exit 2; }
  done
done

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/${ARM}_seed${SEED}_${STAMP}"
mkdir -p "$OUT/logs"

if [[ "$ARM" == "none" ]]; then
  ARM_ARGS=(--mode noisy_none --mitigation none --no-dd --no-gate-twirl --no-meas-twirl)
else
  ARM_ARGS=(--mode noisy_trex --mitigation trex --no-dd --no-gate-twirl --meas-twirl --n-cal-trex 24)
fi

common=(
  --dof-target 32 --depth 5
  --pauli-atol 0 --cost-type local
  --noise-model fake-athens --require-cluster-files
  --shots 1024
  --lr-levels 0.005 0.012 0.010
  --iters-levels 180 180 250
  --spsa-avg 8 --spsa-c0 0.20 --spsa-gamma 0.101
  --max-step 0.03 --jitter 0.02
  --coarse-init lbfgs --lbfgs-restarts 4 --lbfgs-maxiter 500
  --diag-every 5
  --seed "$SEED" --theta-seed "$SEED"
)

echo "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
echo "  SEED=$SEED  ARM=$ARM"
echo "  OUT=$OUT"
echo "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"

for fam in heat1d heat2d beam2d; do
  d="$(datadir "$fam")"
  python trex_warmstart_campaign.py --family "$fam" "${common[@]}" "${ARM_ARGS[@]}" \
    --datadir "$d" --dry-run \
    > "$OUT/logs/preflight_${fam}.out" 2>&1
  grep -q 'NOISE MODEL: fake-athens' "$OUT/logs/preflight_${fam}.out" || \
    { echo "ERROR: FakeAthens preflight failed for $fam" >&2; exit 2; }
done
echo "Preflight OK."

pids=()
for fam in heat1d heat2d beam2d; do
  d="$(datadir "$fam")"
  mkdir -p "$OUT/$fam"
  (
    env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=5 OMP_PROC_BIND=spread \
        OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      python trex_warmstart_campaign.py \
        --family "$fam" "${common[@]}" "${ARM_ARGS[@]}" \
        --datadir "$d" \
        --json-out    "$OUT/$fam/warmstart_seed${SEED}.json" \
        --metrics-csv "$OUT/$fam/metrics_seed${SEED}.csv" \
        --checkpoint-dir "$OUT/$fam/checkpoints" --checkpoint-every 25 \
        --archive-npz "$OUT/$fam/archive_seed${SEED}.npz" \
        --theta-out   "$OUT/$fam/theta_dof32_seed${SEED}.npy" \
      > "$OUT/logs/${fam}_${ARM}_seed${SEED}.out" 2>&1
  ) &
  pids+=("$!")
done

status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done

if [[ "$status" -ne 0 ]]; then
  echo "ERROR: partial data in $OUT" >&2
  exit "$status"
fi

echo "DONE: $OUT"
