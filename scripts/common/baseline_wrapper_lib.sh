#!/usr/bin/env bash

# main.py and every baseline use this one persistent scratch tree. Raw/processed
# datasets are shared; method-specific derived artifacts live below cache/.
DEFAULT_SUPPORT_ROOT="${SUPPORT_GRAPH_STORAGE_ROOT:-./results}"
DEFAULT_DATA_ROOT="${SUPPORT_GRAPH_DATA_DIR:-${SCAFFOLD_DATA_ROOT:-./data}}"
DEFAULT_CACHE_ROOT="${SUPPORT_GRAPH_CACHE_DIR:-${DEFAULT_SUPPORT_ROOT}/cache}"
DEFAULT_RESULTS_ROOT="${SUPPORT_GRAPH_RESULTS_DIR:-${DEFAULT_SUPPORT_ROOT}/results/support_graph_baselines}"
DEFAULT_LOG_ROOT="${SUPPORT_GRAPH_LOG_DIR:-${DEFAULT_SUPPORT_ROOT}/logs/support_graph_baselines}"
DEFAULT_PYTHON="python"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPPORT_GRAPH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET="Cora"
DATA_ROOT="${DEFAULT_DATA_ROOT}"
CACHE_ROOT="${DEFAULT_CACHE_ROOT}"
KEPT_RATIO="0.20"
SMOKE_TEST=0
DEVICE="${DEVICE:-0}"
EPOCHS_OVERRIDE=""
RUNS_OVERRIDE=""
SEED_OVERRIDE=""
INIT_SUPPORT_OVERRIDE=""
RESPARSIFY_EVERY_OVERRIDE=""
RESPARSIFY_RESEED_OVERRIDE=""
OUTPUT_ROOT=""
PRUNE_ROUNDS_OVERRIDE=""
TUNEDGNN_HIDDEN_CHANNELS=""
TUNEDGNN_NUM_LAYERS=""
TUNEDGNN_HEADS=""
TUNEDGNN_LEARNING_RATE=""
TUNEDGNN_WEIGHT_DECAY=""
TUNEDGNN_DROPOUT=""
TUNEDGNN_INPUT_DROPOUT=""
TUNEDGNN_METRIC=""
TUNEDGNN_PRE_LINEAR=""
TUNEDGNN_RESIDUAL_CONNECTIONS=""
TUNEDGNN_LAYER_NORM=""
TUNEDGNN_BATCH_NORM=""
TUNEDGNN_JUMPING_KNOWLEDGE=""
TUNEDGNN_EVAL_EVERY=""
TUNEDGNN_EVAL_START_EPOCH=""
TUNEDGNN_LOG_EVERY=""
TUNEDGNN_PRESET_PROFILE="${TUNEDGNN_PRESET_PROFILE:-original}"
PARTITION_MODE=""
PARTITION_COUNT=""
PARTITION_ID="0"
PARTITION_MAX_NODES=""
PARTITION_EDGE_SAMPLE_SIZE=""
FORCE_SINGLE_LABEL=0
DATASET_LOAD_ONLY=0
DATASET_LOAD_HOMOPHILY=0
EXTRA_ARGS=()

if [[ -x "${DEFAULT_PYTHON}" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

usage_common() {
    cat <<'USAGE'
Common options:
  --dataset NAME          Dataset name, e.g. Cora, Reddit, OGB-products.
  --data-root PATH        Dataset root (same tree main.py uses). Default: ./data
  --cache-root PATH       Method cache root. Default: ./results/cache
  --kept-ratio FLOAT      Kept edge ratio for sparse methods. Default: 0.20
  --smoke-test            Use very small epoch/iteration counts.
  --device ID             CUDA device id. Default: 0
  --python PATH           Python executable.
  --epochs N              Override training epochs.
  --runs N                Override method-internal run count.
  --seed N                Override random seed when the method supports it.
  --init-support NAME     Initial-support tree for SCAFFOLD variants.
                          One of {mst, maxst, fast-mst, fast-maxst, randst,
                          fast-randst, glst, slst, randspt, llst}.
                          Default: maxst.
  --resparsify-every K    Rebuild the sparse graph every K training epochs
                          (SCAFFOLD variants). Default: 1 (every scheduled
                          epoch); use 0 to precompute once.
  --resparsify-reseed BOOL  When resparsifying, reseed the sparsifier RNG each
                            rebuild. Default: true.
  --results-root PATH     Root for method result artifacts.
  --prune-rounds N        Override pruning rounds for methods that expose them.
  --hidden-channels N     tunedGNN comparison-backbone hidden width.
  --num-layers N          tunedGNN comparison-backbone message-passing depth.
  --heads N               tunedGNN comparison-backbone head/group count.
  --learning-rate FLOAT   tunedGNN comparison-backbone learning rate.
  --weight-decay FLOAT    tunedGNN comparison-backbone weight decay.
  --dropout FLOAT         tunedGNN comparison-backbone dropout.
  --input-dropout FLOAT   tunedGNN comparison-backbone input dropout.
  --metric NAME           tunedGNN comparison metric: acc or rocauc.
  --pre-linear            Enable tunedGNN's pre-linear input projection.
  --residual-connections  Enable tunedGNN residual connections.
  --layer-norm            Enable tunedGNN layer normalization.
  --batch-norm            Enable tunedGNN batch normalization.
  --jumping-knowledge     Enable tunedGNN jumping knowledge.
  --preset-profile NAME  tunedGNN contract profile: original or paper.
  --partition-mode MODE   none, auto, or metis for smoke subgraph partitioning.
  --partition-count N     METIS partition count.
  --partition-id N        Partition id to run. Default: 0.
  --partition-max-nodes N Cap smoke partition size after METIS selection.
  --partition-edge-sample-size N  Sample this many edges for large METIS partitioning.
  --force-single-label    Convert multi-label targets to argmax labels for smoke compatibility.
  --dataset-load-only     Load/verify the dataset and exit before method transforms or training.
  --dataset-load-homophily  Also compute exact node/edge homophily during a load-only check.
USAGE
}

parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dataset)
                DATASET="$2"; shift 2 ;;
            --data-root|--data_root)
                DATA_ROOT="$2"; shift 2 ;;
            --cache-root|--cache_root)
                CACHE_ROOT="$2"; shift 2 ;;
            --kept-ratio|--kept_ratio)
                KEPT_RATIO="$2"; shift 2 ;;
            --smoke-test)
                SMOKE_TEST=1; shift ;;
            --no-smoke-test)
                SMOKE_TEST=0; shift ;;
            --device|--gpu)
                DEVICE="$2"; shift 2 ;;
            --python)
                PYTHON_BIN="$2"; shift 2 ;;
            --epochs)
                EPOCHS_OVERRIDE="$2"; shift 2 ;;
            --runs)
                RUNS_OVERRIDE="$2"; shift 2 ;;
            --seed)
                SEED_OVERRIDE="$2"; shift 2 ;;
            --init-support|--init_support|--joint-init-support|--joint_init_support)
                INIT_SUPPORT_OVERRIDE="$2"; shift 2 ;;
            --resparsify-every|--resparsify_every|--scaffold-resparsify-every|--scaffold_resparsify_every)
                RESPARSIFY_EVERY_OVERRIDE="$2"; shift 2 ;;
            --resparsify-reseed|--resparsify_reseed|--scaffold-resparsify-reseed|--scaffold_resparsify_reseed)
                RESPARSIFY_RESEED_OVERRIDE="$2"; shift 2 ;;
            --results-root|--results_root)
                OUTPUT_ROOT="$2"; shift 2 ;;
            --prune-rounds|--prune_rounds)
                PRUNE_ROUNDS_OVERRIDE="$2"; shift 2 ;;
            --hidden-channels|--hidden_channels)
                TUNEDGNN_HIDDEN_CHANNELS="$2"; shift 2 ;;
            --num-layers|--num_layers|--local-layers|--local_layers)
                TUNEDGNN_NUM_LAYERS="$2"; shift 2 ;;
            --heads|--n-heads|--n_heads)
                TUNEDGNN_HEADS="$2"; shift 2 ;;
            --learning-rate|--learning_rate|--lr)
                TUNEDGNN_LEARNING_RATE="$2"; shift 2 ;;
            --weight-decay|--weight_decay)
                TUNEDGNN_WEIGHT_DECAY="$2"; shift 2 ;;
            --dropout)
                TUNEDGNN_DROPOUT="$2"; shift 2 ;;
            --input-dropout|--input_dropout|--in-dropout|--in_dropout)
                TUNEDGNN_INPUT_DROPOUT="$2"; shift 2 ;;
            --metric)
                TUNEDGNN_METRIC="$2"; shift 2 ;;
            --pre-linear|--pre_linear)
                TUNEDGNN_PRE_LINEAR=1; shift ;;
            --no-pre-linear|--no_pre_linear)
                TUNEDGNN_PRE_LINEAR=0; shift ;;
            --residual-connections|--residual_connections|--res)
                TUNEDGNN_RESIDUAL_CONNECTIONS=1; shift ;;
            --no-residual-connections|--no_residual_connections|--no-res)
                TUNEDGNN_RESIDUAL_CONNECTIONS=0; shift ;;
            --layer-norm|--layer_norm|--ln)
                TUNEDGNN_LAYER_NORM=1; shift ;;
            --no-layer-norm|--no_layer_norm|--no-ln)
                TUNEDGNN_LAYER_NORM=0; shift ;;
            --batch-norm|--batch_norm|--bn)
                TUNEDGNN_BATCH_NORM=1; shift ;;
            --no-batch-norm|--no_batch_norm|--no-bn)
                TUNEDGNN_BATCH_NORM=0; shift ;;
            --jumping-knowledge|--jumping_knowledge|--jk)
                TUNEDGNN_JUMPING_KNOWLEDGE=1; shift ;;
            --no-jumping-knowledge|--no_jumping_knowledge|--no-jk)
                TUNEDGNN_JUMPING_KNOWLEDGE=0; shift ;;
            --eval-every|--eval_every|--eval-step|--eval_step)
                TUNEDGNN_EVAL_EVERY="$2"; shift 2 ;;
            --eval-start-epoch|--eval_start_epoch)
                TUNEDGNN_EVAL_START_EPOCH="$2"; shift 2 ;;
            --log-every|--log_every|--display-step|--display_step)
                TUNEDGNN_LOG_EVERY="$2"; shift 2 ;;
            --preset-profile)
                TUNEDGNN_PRESET_PROFILE="$2"; shift 2 ;;
            --partition-mode|--partition_mode)
                PARTITION_MODE="$2"; shift 2 ;;
            --partition-count|--partition_count)
                PARTITION_COUNT="$2"; shift 2 ;;
            --partition-id|--partition_id)
                PARTITION_ID="$2"; shift 2 ;;
            --partition-max-nodes|--partition_max_nodes)
                PARTITION_MAX_NODES="$2"; shift 2 ;;
            --partition-edge-sample-size|--partition_edge_sample_size)
                PARTITION_EDGE_SAMPLE_SIZE="$2"; shift 2 ;;
            --force-single-label|--force_single_label)
                FORCE_SINGLE_LABEL=1; shift ;;
            --dataset-load-only|--dataset_load_only|--load-only)
                DATASET_LOAD_ONLY=1; shift ;;
            --dataset-load-homophily|--dataset_load_homophily)
                DATASET_LOAD_HOMOPHILY=1; shift ;;
            --help|-h)
                usage_common; exit 0 ;;
            *)
                EXTRA_ARGS+=("$1"); shift ;;
        esac
    done
}

print_tunedgnn_contract() {
    printf '[TunedGNNContract] dataset=%s preset_profile=%s seed=%s runs=%s epochs=%s split_protocol=%s hidden_channels=%s num_layers=%s heads=%s lr=%s weight_decay=%s dropout=%s input_dropout=%s metric=%s pre_linear=%s residual=%s layer_norm=%s batch_norm=%s jumping_knowledge=%s eval_every=%s eval_start_epoch=%s log_every=%s\n' \
        "${DATASET}" "${TUNEDGNN_PRESET_PROFILE}" "${SEED_OVERRIDE:-42}" "${RUNS_OVERRIDE:-1}" \
        "${EPOCHS_OVERRIDE:-method-default}" "${SCAFFOLD_SPLIT_PROTOCOL:-tunedgnn}" \
        "${TUNEDGNN_HIDDEN_CHANNELS:-method-default}" \
        "${TUNEDGNN_NUM_LAYERS:-method-default}" \
        "${TUNEDGNN_HEADS:-method-default}" \
        "${TUNEDGNN_LEARNING_RATE:-method-default}" \
        "${TUNEDGNN_WEIGHT_DECAY:-method-default}" \
        "${TUNEDGNN_DROPOUT:-method-default}" \
        "${TUNEDGNN_INPUT_DROPOUT:-method-default}" \
        "${TUNEDGNN_METRIC:-method-default}" \
        "${TUNEDGNN_PRE_LINEAR:-method-default}" \
        "${TUNEDGNN_RESIDUAL_CONNECTIONS:-method-default}" \
        "${TUNEDGNN_LAYER_NORM:-method-default}" \
        "${TUNEDGNN_BATCH_NORM:-method-default}" \
        "${TUNEDGNN_JUMPING_KNOWLEDGE:-method-default}" \
        "${TUNEDGNN_EVAL_EVERY:-method-default}" \
        "${TUNEDGNN_EVAL_START_EPOCH:-method-default}" \
        "${TUNEDGNN_LOG_EVERY:-method-default}"
}

# Memory-safe neighbor-sampling defaults for the scalable MoG/AdaGLT/
# Unified-LTH adapters.  These methods score every sampled edge, so their peak
# memory follows the complete L-hop expansion rather than just the seed batch.
# In particular, 15,10,... at five/seven layers exhausted an 80-GB H100 even
# with seed batches of 512/128.  Keep the tunedGNN backbone and optimizer
# unchanged, but use a bounded sampling workload that also permits several
# independent baselines to share one H100.
scalable_batch_size_for_layers() {
    local layers="$1"
    if ((layers >= 7)); then
        printf '128'
    elif ((layers >= 5)); then
        printf '512'
    elif ((layers >= 3)); then
        printf '1024'
    else
        printf '2048'
    fi
}

scalable_eval_batch_size_for_layers() {
    local layers="$1"
    if ((layers >= 7)); then
        printf '512'
    elif ((layers >= 5)); then
        printf '1024'
    elif ((layers >= 3)); then
        printf '2048'
    else
        printf '4096'
    fi
}

scalable_fanouts_for_layers() {
    local layers="$1"
    local default_first_fanout default_later_fanout
    if ((layers >= 7)); then
        default_first_fanout=5
        default_later_fanout=2
    elif ((layers >= 5)); then
        default_first_fanout=5
        default_later_fanout=3
    elif ((layers >= 3)); then
        default_first_fanout=10
        default_later_fanout=5
    else
        default_first_fanout=15
        default_later_fanout=10
    fi
    local first_fanout="${BASELINE_SCALABLE_FIRST_FANOUT:-${default_first_fanout}}"
    local later_fanout="${BASELINE_SCALABLE_LATER_FANOUT:-${default_later_fanout}}"
    local values=()
    local layer
    for ((layer = 0; layer < layers; layer++)); do
        if ((layer == 0)); then
            values+=("${first_fanout}")
        else
            values+=("${later_fanout}")
        fi
    done
    (IFS=,; printf '%s' "${values[*]}")
}

# Evaluation visits every node in the official split.  A separate bounded
# fanout prevents Products/Pokec evaluation from expanding billions of sampled
# neighbors before the first reportable epoch.  This changes only the sampling
# workload; the evaluated node set and official split remain complete.
scalable_eval_fanouts_for_layers() {
    local layers="$1"
    local default_first_fanout default_later_fanout
    if ((layers >= 5)); then
        default_first_fanout=5
        default_later_fanout=2
    elif ((layers >= 3)); then
        default_first_fanout=10
        default_later_fanout=3
    else
        default_first_fanout=15
        default_later_fanout=10
    fi
    local first_fanout="${BASELINE_SCALABLE_EVAL_FIRST_FANOUT:-${default_first_fanout}}"
    local later_fanout="${BASELINE_SCALABLE_EVAL_LATER_FANOUT:-${default_later_fanout}}"
    local values=()
    local layer
    for ((layer = 0; layer < layers; layer++)); do
        if ((layer == 0)); then
            values+=("${first_fanout}")
        else
            values+=("${later_fanout}")
        fi
    done
    (IFS=,; printf '%s' "${values[*]}")
}

load_tunedgnn_preset_contract() {
    # Resolve the tunedGNN architecture contract from this repository's own
    # table (configs/tunedgnn_presets.py) instead of
    # requiring an Benchmark checkout. Values supplied explicitly on the command
    # line stay authoritative: only unset variables are filled in here.
    local emitted
    if ! emitted="$(
        cd "${SUPPORT_GRAPH_ROOT}" && \
        "${PYTHON_BIN}" -m configs.tunedgnn_contract_env \
            --dataset "${DATASET}" \
            --method "${BASELINE_METHOD_NAME:-}" \
            --profile "${TUNEDGNN_PRESET_PROFILE}" 2>/dev/null
    )"; then
        return 1
    fi

    local line variable value
    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        variable="${line%%=*}"
        value="${line#*=}"
        # Only fill gaps; an explicit --hidden-channels style flag wins.
        if [[ -z "${!variable-}" ]]; then
            eval "${variable}=${value}"
        fi
    done <<< "${emitted}"
    return 0
}

require_tunedgnn_preset_contract() {
    # Karate is only a plumbing smoke test and has no tunedGNN publication
    # preset. Every real comparison dataset must receive the complete resolved
    # contract. Failing here prevents a wrapper from silently falling back to
    # its method-local architecture.
    if [[ "${SMOKE_TEST}" -eq 1 ]]; then
        return 0
    fi
    if [[ "${TUNEDGNN_PRESET_PROFILE}" != "original" && "${TUNEDGNN_PRESET_PROFILE}" != "paper" ]]; then
        printf 'Invalid tunedGNN preset profile: %s\n' "${TUNEDGNN_PRESET_PROFILE}" >&2
        return 2
    fi

    load_tunedgnn_preset_contract || true

    local missing=()
    local variable
    for variable in \
        TUNEDGNN_HIDDEN_CHANNELS \
        TUNEDGNN_NUM_LAYERS \
        TUNEDGNN_HEADS \
        TUNEDGNN_LEARNING_RATE \
        TUNEDGNN_WEIGHT_DECAY \
        TUNEDGNN_DROPOUT \
        TUNEDGNN_INPUT_DROPOUT \
        TUNEDGNN_METRIC \
        TUNEDGNN_PRE_LINEAR \
        TUNEDGNN_RESIDUAL_CONNECTIONS \
        TUNEDGNN_LAYER_NORM \
        TUNEDGNN_BATCH_NORM \
        TUNEDGNN_JUMPING_KNOWLEDGE
    do
        if [[ -z "${!variable}" ]]; then
            missing+=("${variable}")
        fi
    done

    if [[ "${#missing[@]}" -gt 0 ]]; then
        printf 'Missing tunedGNN preset contract for dataset %s: %s\n' \
            "${DATASET}" "${missing[*]}" >&2
        printf 'No preset for this dataset/model in configs/tunedgnn_presets.py; refusing method-local fallback.\n' >&2
        return 2
    fi
}

setup_common_env() {
    if ! awk -v ratio="${KEPT_RATIO}" \
        'BEGIN { exit !(ratio > 0.0 && ratio <= 1.0) }'; then
        echo "--kept-ratio must be in (0, 1], got: ${KEPT_RATIO}" >&2
        exit 2
    fi
    if [[ -z "${OUTPUT_ROOT}" ]]; then
        OUTPUT_ROOT="${DEFAULT_RESULTS_ROOT}"
    fi
    mkdir -p "${DATA_ROOT}" "${CACHE_ROOT}" "${DEFAULT_LOG_ROOT}" \
        "${OUTPUT_ROOT}" \
        "${CACHE_ROOT}/shared/matplotlib" \
        "${CACHE_ROOT}/shared/xdg"
    export SUPPORT_GRAPH_ROOT
    export SUPPORT_GRAPH_STORAGE_ROOT="${DEFAULT_SUPPORT_ROOT}"
    export SUPPORT_GRAPH_DATA_DIR="${DATA_ROOT}"
    export SUPPORT_GRAPH_CACHE_DIR="${CACHE_ROOT}"
    export SUPPORT_GRAPH_RESULTS_DIR="${OUTPUT_ROOT}"
    export BASELINE_DATA_ROOT="${DATA_ROOT}"
    export RESULTS_ROOT="${OUTPUT_ROOT}"
    # Stamp every per-run record with the metric the preset contract asked for,
    # so accuracy cells can never be harvested into a ROC-AUC column unnoticed.
    if [[ -n "${TUNEDGNN_METRIC}" ]]; then
        export BASELINE_METRIC="${TUNEDGNN_METRIC}"
    fi
    if [[ -n "${SEED_OVERRIDE}" ]]; then
        export SCAFFOLD_DATASET_SEED="${SEED_OVERRIDE}"
        export BASELINE_SEED="${SEED_OVERRIDE}"
    fi
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    if [[ -n "${PARTITION_MODE}" ]]; then
        export BASELINE_PARTITION_MODE="${PARTITION_MODE}"
    fi
    if [[ -n "${PARTITION_COUNT}" ]]; then
        export BASELINE_PARTITION_COUNT="${PARTITION_COUNT}"
    fi
    export BASELINE_PARTITION_ID="${PARTITION_ID}"
    if [[ -n "${PARTITION_MAX_NODES}" ]]; then
        export BASELINE_PARTITION_MAX_NODES="${PARTITION_MAX_NODES}"
    fi
    if [[ -n "${PARTITION_EDGE_SAMPLE_SIZE}" ]]; then
        export BASELINE_PARTITION_EDGE_SAMPLE_SIZE="${PARTITION_EDGE_SAMPLE_SIZE}"
    fi
    export BASELINE_PARTITION_CACHE_ROOT="${CACHE_ROOT}"
    export BASELINE_FORCE_SINGLE_LABEL="${FORCE_SINGLE_LABEL}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/shared/matplotlib}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/shared/xdg}"
    export PYTHONPATH="${SUPPORT_GRAPH_ROOT}:${SUPPORT_GRAPH_ROOT}:${SUPPORT_GRAPH_ROOT}/scripts:${PYTHONPATH:-}"

    if [[ "${DATASET_LOAD_ONLY}" -eq 1 ]]; then
        local load_args=(
            --method "${BASELINE_METHOD_NAME:-shared}"
            --dataset "${DATASET}"
            --data-dir "${DATA_ROOT}"
            --cache-dir "${CACHE_ROOT}"
        )
        if [[ "${DATASET_LOAD_HOMOPHILY}" -eq 1 ]]; then
            load_args+=(--homophily)
        fi
        "${PYTHON_BIN}" "${SUPPORT_GRAPH_ROOT}/scripts/common/smoke_method_dataset_load.py" "${load_args[@]}"
        exit $?
    fi
}

cuda_device_arg() {
    if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
    then
        printf 'cuda:%s' "${DEVICE}"
    else
        printf 'cpu'
    fi
}
