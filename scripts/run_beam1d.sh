#!/usr/bin/env bash
# Beam1D FULL rerun: DOF=8, 16, 32 (internal cascade).
# Usage: bash run_beam1d_full.sh [seed]
set -Eeuo pipefail

SEED="${1:-8103}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNNER="beam1d_definitive_fakeathens.py"
RUN_DIR="beam1d_full_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
echo "  Beam1D FULL rerun (DOF=8  16  32)"
echo "  SEED    = $SEED"
echo "  RUN_DIR = $RUN_DIR"
echo "  iters   = 180 180 250"
echo "  threads = 10 per arm (20 total)"
echo "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"

for required in "$RUNNER" beam1d_amg_hook_v2.py beam1d32_athens_local_campaign.py nisq.py nisq_core.py; do
  [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 2; }
done

if pgrep -u "$USER" -f "[b]eam1d_definitive_fakeathens.py.*--seed ${SEED}" >/dev/null; then
  echo "ERROR: Beam1D solver with seed ${SEED} already running." >&2
  pgrep -af "beam1d_definitive_fakeathens.*--seed ${SEED}" | grep -v pgrep
  exit 3
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/none" "$RUN_DIR/trex"

{
  echo "{"
  echo "  \"timestamp\": \"$(date -Iseconds)\","
  echo "  \"hostname\": \"$(hostname)\","
  echo "  \"seed\": $SEED,"
  echo "  \"dof_target\": 32,"
  echo "  \"iters_levels\": [180, 180, 250],"
  echo "  \"threads_per_arm\": 10,"
  echo "  \"run_dir\": \"$RUN_DIR\","
  echo "  \"script_hashes\": {"
  for f in "$RUNNER" beam1d_amg_hook_v2.py nisq.py nisq_core.py beam1d32_athens_local_campaign.py; do
    [[ -f "$f" ]] && echo "    \"$f\": \"$(md5sum "$f" | awk '{print $1}')\","
  done
  echo "    \"_end\": \"_\""
  echo "  },"
  echo "  \"environment\": {"
  echo "    \"python\": \"$(python --version 2>&1 | awk '{print $2}')\","
  echo "    \"conda_env\": \"${CONDA_DEFAULT_ENV:-none}\","
  echo "    \"nproc\": $(nproc),"
  python - <<'PYENV' 2>/dev/null | sed 's/^/    /'
import json
out = {}
for m in ["numpy","scipy","qiskit","qiskit_aer","mitiq","pandas","matplotlib"]:
    try:
        mod = __import__(m)
        out[m] = getattr(mod, "__version__", "unknown")
    except Exception:
        out[m] = "not installed"
print(json.dumps(out, indent=2))
PYENV
  echo "    \"_end\": \"_\""
  echo "  }"
  echo "}"
} > "$RUN_DIR/manifest.json" 2>/dev/null || echo "WARN: manifest failed"

common=(
  --family beam1d --dof-target 32 --depth 5
  --pauli-atol 0 --cost-type local
  --noise-model fake-athens
  --datadir "$ROOT_DIR" --require-cluster-files
  --shots 1024
  --lr-levels 0.005 0.012 0.010
  --iters-levels 180 180 250
  --spsa-avg 8 --spsa-c0 0.20 --spsa-gamma 0.101
  --max-step 0.03 --jitter 0.02
  --coarse-init lbfgs --lbfgs-restarts 4 --lbfgs-maxiter 500
  --no-dd --no-gate-twirl
  --diag-every 5
)

echo "Preflight..."
python "$RUNNER" "${common[@]}" --mode noisy_trex --mitigation trex \
  --n-cal-trex 24 --meas-twirl --seed "$SEED" --theta-seed "$SEED" --dry-run \
  > "$RUN_DIR/logs/preflight.out" 2>&1
cat "$RUN_DIR/logs/preflight.out"
echo

echo "Launching paired none/trex (10 threads each, ETA ~28h)..."
pids=()
for arm in none trex; do
  if [[ "$arm" == "none" ]]; then
    mode_args=(--mode noisy_none --mitigation none --no-meas-twirl)
  else
    mode_args=(--mode noisy_trex --mitigation trex --n-cal-trex 24 --meas-twirl)
  fi
  (
    env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=5 OMP_PROC_BIND=spread \
        OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      python "$RUNNER" "${common[@]}" "${mode_args[@]}" \
        --seed "$SEED" --theta-seed "$SEED" \
        --json-out     "$RUN_DIR/$arm/warmstart_seed${SEED}.json" \
        --metrics-csv  "$RUN_DIR/$arm/metrics_seed${SEED}.csv" \
        --archive-npz  "$RUN_DIR/$arm/archive_seed${SEED}.npz" \
        --theta-out    "$RUN_DIR/$arm/theta_dof32_seed${SEED}.npy" \
      > "$RUN_DIR/logs/${arm}_warmstart_seed${SEED}.out" 2>&1
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

if [[ "$status" -ne 0 ]]; then
  echo "ERROR: a paired trajectory failed." >&2
  exit "$status"
fi

echo "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
echo "  DONE: $ROOT_DIR/$RUN_DIR"
echo "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
