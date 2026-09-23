#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPTS_ROOT}/common/baseline_wrapper_lib.sh"
parse_common_args "$@"
export BASELINE_METHOD_NAME="dspar"
setup_common_env
require_tunedgnn_preset_contract

DSPAR_ROOT="${SUPPORT_GRAPH_ROOT}/RelatedMethods/DSpar_tmlr-main"
BENCH_DIR="${DSPAR_ROOT}/mem_speed_bench"
CONF="non_ogb_datasets/conf/gcn.yaml"
if [[ "${SMOKE_TEST}" -eq 1 ]]; then
    CONF="non_ogb_datasets/conf/gcn_smoke.yaml"
fi
RUNS=1
if [[ -n "${RUNS_OVERRIDE}" ]]; then
    RUNS="${RUNS_OVERRIDE}"
fi
EPOCH_ARGS=()
if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
    EPOCH_ARGS=(--epochs "${EPOCHS_OVERRIDE}")
fi
SEED_ARGS=()
if [[ -n "${SEED_OVERRIDE}" ]]; then
    SEED_ARGS=(--seed "${SEED_OVERRIDE}")
fi
TUNED_ARGS=()
[[ -n "${TUNEDGNN_HIDDEN_CHANNELS}" ]] && TUNED_ARGS+=(--hidden_channels "${TUNEDGNN_HIDDEN_CHANNELS}")
[[ -n "${TUNEDGNN_NUM_LAYERS}" ]] && TUNED_ARGS+=(--num_layers "${TUNEDGNN_NUM_LAYERS}")
[[ -n "${TUNEDGNN_LEARNING_RATE}" ]] && TUNED_ARGS+=(--lr "${TUNEDGNN_LEARNING_RATE}")
[[ -n "${TUNEDGNN_WEIGHT_DECAY}" ]] && TUNED_ARGS+=(--weight_decay "${TUNEDGNN_WEIGHT_DECAY}")
[[ -n "${TUNEDGNN_DROPOUT}" ]] && TUNED_ARGS+=(--dropout "${TUNEDGNN_DROPOUT}")
[[ -n "${TUNEDGNN_INPUT_DROPOUT}" ]] && TUNED_ARGS+=(--input_dropout "${TUNEDGNN_INPUT_DROPOUT}")
[[ -n "${TUNEDGNN_METRIC}" ]] && TUNED_ARGS+=(--metric "${TUNEDGNN_METRIC}")
[[ -n "${TUNEDGNN_PRE_LINEAR}" ]] && TUNED_ARGS+=(--pre_linear "${TUNEDGNN_PRE_LINEAR}")
[[ -n "${TUNEDGNN_RESIDUAL_CONNECTIONS}" ]] && TUNED_ARGS+=(--residual "${TUNEDGNN_RESIDUAL_CONNECTIONS}")
[[ -n "${TUNEDGNN_LAYER_NORM}" ]] && TUNED_ARGS+=(--layer_norm "${TUNEDGNN_LAYER_NORM}")
[[ -n "${TUNEDGNN_BATCH_NORM}" ]] && TUNED_ARGS+=(--batch_norm "${TUNEDGNN_BATCH_NORM}")
[[ -n "${TUNEDGNN_JUMPING_KNOWLEDGE}" ]] && TUNED_ARGS+=(--jumping_knowledge "${TUNEDGNN_JUMPING_KNOWLEDGE}")
print_tunedgnn_contract

canonical_dataset_key() {
    printf '%s' "$1" | tr '[:upper:]_' '[:lower:]-'
}

dspar_partition_count() {
    case "$(canonical_dataset_key "$1")" in
        cora) return 1 ;;
        reddit) printf '1500' ;;
        ogb-products|ogbn-products) printf '1500' ;;
        ogb-arxiv|ogbn-arxiv) printf '500' ;;
        ogb-protein|ogb-proteins|ogbn-protein|ogbn-proteins) printf '1000' ;;
        pokec) printf '8000' ;;
        *) printf '64' ;;
    esac
}

dataset_key="$(canonical_dataset_key "${DATASET}")"
BACKBONE_ARGS=()
case "${dataset_key}" in
    cora|citeseer|pubmed|amazon-computer|amazon-photo|coauthor-cs|coauthor-physics|wikics|squirrel|chameleon|roman-empire|amazon-ratings|minesweeper|questions)
        BACKBONE_ARGS=(--tunedgnn_medium_backbone)
        ;;
esac
if [[ "${SMOKE_TEST}" -eq 1 && "${dataset_key}" != "cora" ]]; then
    export BASELINE_PARTITION_MODE="${BASELINE_PARTITION_MODE:-sample}"
    export BASELINE_PARTITION_COUNT="${BASELINE_PARTITION_COUNT:-$(dspar_partition_count "${DATASET}")}"
    export BASELINE_PARTITION_MAX_NODES="${BASELINE_PARTITION_MAX_NODES:-12000}"
    export BASELINE_PARTITION_EDGE_SAMPLE_SIZE="${BASELINE_PARTITION_EDGE_SAMPLE_SIZE:-5000000}"
    export BASELINE_PARTITION_CACHE_ROOT="${BASELINE_PARTITION_CACHE_ROOT:-${CACHE_ROOT}}"
fi

export PYTHONPATH="${DSPAR_ROOT}/src:${BENCH_DIR}:${BENCH_DIR}/non_ogb_datasets:${PYTHONPATH}"

cd "${BENCH_DIR}"
exec "${PYTHON_BIN}" non_ogb_datasets/train_full_batch.py \
    --conf "${CONF}" \
    --dataset "${DATASET}" \
    --root "${DATA_ROOT}" \
    --runs "${RUNS}" \
    "${EPOCH_ARGS[@]}" \
    "${SEED_ARGS[@]}" \
    --gpu "${DEVICE}" \
    --spec_sparsify \
    --kept_ratio "${KEPT_RATIO}" \
    "${BACKBONE_ARGS[@]}" \
    "${TUNED_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
