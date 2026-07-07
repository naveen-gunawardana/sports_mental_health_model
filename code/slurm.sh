#!/bin/bash
#SBATCH --mail-user=babak.hemmatian@stonybrook.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=96:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL

set -euo pipefail

# Activate the project's ISAAC conda env on whatever node Slurm picked.
#
# defq is heterogeneous: h100/orion can see the shared miniforge3 base on
# /shared, but the quadro*/tesla* nodes do NOT mount /shared at all, so no
# `conda` is reachable there and the old "module load / source /shared" path
# hard-failed the instant a task landed on one of them. The env itself lives
# under $HOME (~/.conda/envs/ISAAC), which IS mounted on every node and is
# self-contained, so we can always activate it -- with `conda` when a base is
# reachable, or directly off $HOME when it is not.
#
# Order: (1) module load conda, (2) source the shared base, then EITHER
# `conda activate ISAAC` if we now have conda, OR a direct $HOME activation
# fallback (prepend the env bin to PATH + run its activate.d hooks). $HOME is
# the only path guaranteed visible cluster-wide, so the fallback is what makes
# placement on quadro*/tesla* work instead of failing.
ISAAC_ENV="${HOME}/.conda/envs/ISAAC"

if command -v module >/dev/null 2>&1; then
    module load conda >/dev/null 2>&1 || true
fi
if ! command -v conda >/dev/null 2>&1; then
    if [[ -f /shared/software/miniforge3/etc/profile.d/conda.sh ]]; then
        # shellcheck disable=SC1091
        source /shared/software/miniforge3/etc/profile.d/conda.sh
    fi
fi

# Conda's activation hooks (and the env's activate.d scripts) reference some
# unset shell variables (e.g. ADDR2LINE in binutils' hook), which trips
# 'set -u'. Disable nounset around activation, then restore it.
set +u
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    # Pop any conda envs inherited from the submitter shell (--export=ALL can
    # propagate CONDA_DEFAULT_ENV/CONDA_PREFIX). Without this, 'conda activate
    # ISAAC' short-circuits as a no-op when ISAAC is already marked active, and
    # ISAAC/bin can end up behind miniforge3/bin in PATH -> wrong python.
    while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
        conda deactivate
    done
    conda activate ISAAC
elif [[ -x "${ISAAC_ENV}/bin/python" ]]; then
    # No conda base reachable on this node (e.g. /shared not mounted on
    # quadro*/tesla*). The ISAAC env under $HOME is self-contained, so put it
    # on PATH directly and run its activate.d hooks for any lib/CUDA env vars.
    echo "[slurm.sh] conda base unreachable on $(hostname); activating ISAAC directly from ${ISAAC_ENV}" >&2
    export CONDA_PREFIX="${ISAAC_ENV}"
    export PATH="${ISAAC_ENV}/bin:${PATH}"
    if [[ -d "${ISAAC_ENV}/etc/conda/activate.d" ]]; then
        for _f in "${ISAAC_ENV}"/etc/conda/activate.d/*.sh; do
            [[ -r "${_f}" ]] && source "${_f}"
        done
        unset _f
    fi
else
    echo "[slurm.sh] ERROR: no conda base reachable AND ${ISAAC_ENV}/bin/python missing on $(hostname)." >&2
    exit 1
fi
set -u

# Guard against a silently-wrong python (e.g. a system python on PATH): the
# active interpreter must be the ISAAC env's. Fail loudly here rather than
# deep inside label_location.py with a confusing ImportError.
ACTIVE_PY="$(command -v python || true)"
case "${ACTIVE_PY}" in
    "${ISAAC_ENV}/bin/"*) : ;;
    */ISAAC/bin/*) : ;;
    *) echo "[slurm.sh] ERROR: active python is '${ACTIVE_PY}', not the ISAAC env on $(hostname)." >&2; exit 1 ;;
esac

export PYTHONUNBUFFERED=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configure GPUs if allocated by Slurm
if [[ -n "${SLURM_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_GPUS_ON_NODE:-0}"
  echo "[slurm.sh] GPU allocation detected: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

requires_years=("filter_keywords" "filter_language" "filter_relevance" "filter_keywords_adv" "filter_sample" "label_moralization" "label_sentiment" "label_generalization" "label_emotion" "label_location" "organize_types" "organize_anonymize")
requires_batch=("filter_relevance" "label_moralization" "label_generalization" "label_emotion" "label_sentiment" "label_location")

in_array() { local needle="$1"; shift; for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done; return 1; }

build_task_label() {
  if [[ -z "${years:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    return 0
  fi

  python - "${years}" "${files_per_job:-1}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import sys

years = sys.argv[1]
files_per_job = max(int(sys.argv[2]), 1)
task_id = int(sys.argv[3])

# Mirror utils.parse_range: accept comma-separated combinations of single
# years and contiguous ranges (e.g. "2019", "2019-2023", "2007,2009,2011-2017"),
# returning a sorted/deduplicated list. Keeping this in lock-step with
# parse_range is what guarantees the array-index -> month mapping stays aligned
# with the file_list each python script computes internally.
parsed_years = set()
for tok in years.split(","):
    tok = tok.strip()
    if not tok:
        continue
    if "-" in tok:
        s, e = map(int, tok.split("-", 1))
    else:
        s = e = int(tok)
    parsed_years.update(range(s, e + 1))
parsed_years = sorted(parsed_years)

months = [f"{y:04d}-{m:02d}" for y in parsed_years for m in range(1, 13)]

start_idx = task_id * files_per_job
end_idx = min(start_idx + files_per_job, len(months))
chunk = months[start_idx:end_idx]

if not chunk:
    print(f"task{task_id}")
elif len(chunk) == 1:
    print(chunk[0])
else:
    print(f"{chunk[0]}_to_{chunk[-1]}")
PY
}

# Base args
ARGS=( "./code/${resource}.py" "-r" "${resource}" "-t" "${type}" )

if [[ -n "${group:-}" ]]; then
  ARGS+=( "-g" "${group}" )
fi
if [[ -n "${sample:-}" ]]; then
  ARGS+=( "-c" "${sample}" )
fi

if [[ -n "${target:-}" ]]; then
  ARGS+=( "-S" "${target}" )
fi

if [[ -n "${num_annotators:-}" ]]; then
  ARGS+=( "-n" "${num_annotators}" )
fi

if [[ -n "${perc_overlap:-}" ]]; then
  ARGS+=( "-p" "${perc_overlap}" )
fi

if [[ -n "${stratify:-}" ]]; then
  ARGS+=( "--stratify" "${stratify}" )
fi

# Forward optional input/output overrides
if [[ -n "${input:-}" ]]; then
  ARGS+=( "-i" "${input}" )
fi
if [[ -n "${input_2:-}" ]]; then
  ARGS+=( "-2" "${input_2}" )
fi
if [[ -n "${output:-}" ]]; then
  ARGS+=( "-o" "${output}" )
fi

# Only pass --array if Slurm provided it
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ARGS+=( "--array" "${SLURM_ARRAY_TASK_ID}" )
fi

# Forward the optional spread schedule (file_list-index permutation). When set,
# the python script uses the array slot to index this list instead of file_list.
if [[ -n "${array_order:-}" ]]; then
  ARGS+=( "--array-order" "${array_order}" )
fi

# Conditionally add --years (and enforce if required)
if in_array "${resource}" "${requires_years[@]}"; then
  if [[ -z "${years:-}" ]]; then
    echo "ERROR: --years is required for resource '${resource}'" >&2
    exit 2
  fi
  ARGS+=( "-y" "${years}" )
fi

# Conditionally add --batchsize (and enforce positive integer)
if in_array "${resource}" "${requires_batch[@]}"; then
  if [[ -z "${batchsize:-}" ]]; then
    echo "ERROR: --batchsize is required for resource '${resource}'" >&2
    exit 2
  fi
  if ! [[ "${batchsize}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --batchsize must be a positive integer" >&2
    exit 2
  fi
  ARGS+=( "-b" "${batchsize}" )
fi

# Pass files-per-job if set
if [[ -n "${files_per_job:-}" ]]; then
  ARGS+=( "--files-per-job" "${files_per_job}" )
fi

# Forward location-labeling sampling controls when present
if [[ -n "${maxitems:-}" ]]; then
  ARGS+=( "--maxitems" "${maxitems}" )
fi
if [[ -n "${maxfiles:-}" ]]; then
  ARGS+=( "--maxfiles" "${maxfiles}" )
fi
if [[ -n "${maxradius:-}" ]]; then
  ARGS+=( "--maxradius" "${maxradius}" )
fi

# Update the visible Slurm job name for array tasks so squeue reflects the concrete month span.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  name_parts=("${resource}" "${type}")
  if [[ -n "${group:-}" ]]; then
    name_parts+=("${group}")
  fi
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    task_label="$(build_task_label)"
    if [[ -n "${task_label}" ]]; then
      name_parts+=("${task_label}")
    fi
  elif [[ -n "${years:-}" ]]; then
    name_parts+=("${years}")
  fi

  job_name="$(IFS=__ ; echo "${name_parts[*]}")"
  # NOTE: The scontrol command was hanging on the particular cluster we used, hence the commenting out.
  # scontrol update JobId="${SLURM_JOB_ID}" JobName="${job_name}" >/dev/null 2>&1 || true
fi

echo "Running: python ${ARGS[*]}"

# Run python, mirroring stderr into a temp file so we can post-mortem on
# transient node-side CUDA/NVML driver failures. If detected (and we haven't
# already requeued too many times), call `scontrol requeue` so SLURM puts
# the job back in the queue, almost certainly landing it on a different
# node. afterok dependents stay PD while the requeued attempt runs, instead
# of getting cascade-cancelled by --kill-on-invalid-dep=yes.
#
# Only the known transient patterns trigger a requeue; real code failures
# fall through with the original exit code so they surface as FAILED.
ERR_TMP="$(mktemp -t isaac_slurm_stderr.XXXXXX)"
# shellcheck disable=SC2064
trap "rm -f '${ERR_TMP}'" EXIT

set +e
python "${ARGS[@]}" 2> >(tee -a "${ERR_TMP}" >&2)
PY_RC=$?
set -e

TRANSIENT_RE="Can't initialize NVML|NVML_SUCCESS == DriverAPI|INTERNAL ASSERT FAILED.*CUDACachingAllocator|CUDA error: no CUDA-capable device"
MAX_REQUEUES=2
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"

if [[ "${PY_RC}" -ne 0 ]] \
   && [[ "${RESTART_COUNT}" -lt "${MAX_REQUEUES}" ]] \
   && [[ -n "${SLURM_JOB_ID:-}" ]] \
   && grep -qE "${TRANSIENT_RE}" "${ERR_TMP}" 2>/dev/null; then
  NODE="${SLURMD_NODENAME:-unknown}"
  echo "[slurm.sh] Detected transient CUDA/NVML failure on node ${NODE} (restart=${RESTART_COUNT}/${MAX_REQUEUES}); requeueing ${SLURM_JOB_ID}" >&2
  if scontrol requeue "${SLURM_JOB_ID}"; then
    # scontrol requeue sends SIGTERM shortly after returning; the lines below
    # may not execute. Exit 0 just in case it doesn't, so we don't trip
    # --kill-on-invalid-dep=yes between the requeue call and the SIGTERM.
    sleep 5
    exit 0
  else
    echo "[slurm.sh] scontrol requeue failed; surfacing original exit code ${PY_RC}" >&2
  fi
fi

exit "${PY_RC}"
