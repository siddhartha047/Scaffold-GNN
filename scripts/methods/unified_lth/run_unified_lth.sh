#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPTS_ROOT}/common/baseline_wrapper_lib.sh"
parse_common_args "$@"
export BASELINE_METHOD_NAME="unified_lth"
setup_common_env
require_tunedgnn_preset_contract

ULTH_DIR="${SUPPORT_GRAPH_ROOT}/RelatedMethods/Unified-LTH-GNN-main/NodeClassification"
DEVICE_ARG="$(cuda_device_arg)"
TOTAL_EPOCH=300
MASK_EPOCH=200
# The reference Unified-LTH GCN IMP performs two learn/prune/retrain rounds.
PRUNE_ROUNDS=2
# The shared target ratio is an edge budget. Model-weight pruning remains an
# independent optional control so it cannot silently consume the graph budget.
WEIGHT_KEPT_RATIO=1.0
if [[ "${SMOKE_TEST}" -eq 1 ]]; then
    TOTAL_EPOCH=1
    MASK_EPOCH="${EPOCHS_OVERRIDE:-1}"
    PRUNE_ROUNDS=1
fi
if [[ -n "${PRUNE_ROUNDS_OVERRIDE}" ]]; then
    PRUNE_ROUNDS="${PRUNE_ROUNDS_OVERRIDE}"
fi
RUNS="${RUNS_OVERRIDE:-1}"
SEED="${SEED_OVERRIDE:-42}"

if [[ "${PRUNE_ROUNDS}" -lt 1 ]]; then
    echo "--prune-rounds must be positive" >&2
    exit 2
fi
SEED_ARGS=()
if [[ -n "${SEED_OVERRIDE}" ]]; then
    SEED_ARGS=(--seed "${SEED_OVERRIDE}")
fi
TUNED_ARGS=()
[[ -n "${TUNEDGNN_HIDDEN_CHANNELS}" ]] && TUNED_ARGS+=(--hidden_channels "${TUNEDGNN_HIDDEN_CHANNELS}")
[[ -n "${TUNEDGNN_NUM_LAYERS}" ]] && TUNED_ARGS+=(--num_layers "${TUNEDGNN_NUM_LAYERS}")
[[ -n "${TUNEDGNN_LEARNING_RATE}" ]] && TUNED_ARGS+=(--lr "${TUNEDGNN_LEARNING_RATE}")
[[ -n "${TUNEDGNN_WEIGHT_DECAY}" ]] && TUNED_ARGS+=(--weight-decay "${TUNEDGNN_WEIGHT_DECAY}")
[[ -n "${TUNEDGNN_DROPOUT}" ]] && TUNED_ARGS+=(--dropout "${TUNEDGNN_DROPOUT}")
[[ -n "${TUNEDGNN_INPUT_DROPOUT}" ]] && TUNED_ARGS+=(--input_dropout "${TUNEDGNN_INPUT_DROPOUT}")
[[ -n "${TUNEDGNN_METRIC}" ]] && TUNED_ARGS+=(--metric "${TUNEDGNN_METRIC}")
[[ -n "${TUNEDGNN_PRE_LINEAR}" ]] && TUNED_ARGS+=(--pre_linear "${TUNEDGNN_PRE_LINEAR}")
[[ -n "${TUNEDGNN_RESIDUAL_CONNECTIONS}" ]] && TUNED_ARGS+=(--residual "${TUNEDGNN_RESIDUAL_CONNECTIONS}")
[[ -n "${TUNEDGNN_LAYER_NORM}" ]] && TUNED_ARGS+=(--layer_norm "${TUNEDGNN_LAYER_NORM}")
[[ -n "${TUNEDGNN_BATCH_NORM}" ]] && TUNED_ARGS+=(--batch_norm "${TUNEDGNN_BATCH_NORM}")
[[ -n "${TUNEDGNN_JUMPING_KNOWLEDGE}" ]] && TUNED_ARGS+=(--jumping_knowledge "${TUNEDGNN_JUMPING_KNOWLEDGE}")
print_tunedgnn_contract

DATASET_KEY="$(printf '%s' "${DATASET}" | tr '[:upper:]_' '[:lower:]-')"
case "${DATASET_KEY}" in
    coauthor-physics|questions|reddit|reddit2|ogbn-products|ogb-products|products|ogbn-arxiv|ogb-arxiv|arxiv|ogbn-proteins|ogb-proteins|proteins|pokec)
        SCALABLE_BATCH_SIZE="${BASELINE_SCALABLE_BATCH_SIZE:-$(scalable_batch_size_for_layers "${TUNEDGNN_NUM_LAYERS}")}"
        SCALABLE_EVAL_BATCH_SIZE="${BASELINE_SCALABLE_EVAL_BATCH_SIZE:-$(scalable_eval_batch_size_for_layers "${TUNEDGNN_NUM_LAYERS}")}"
        SCALABLE_FANOUTS="${BASELINE_SCALABLE_FANOUTS:-$(scalable_fanouts_for_layers "${TUNEDGNN_NUM_LAYERS}")}"
        SCALABLE_EVAL_FANOUTS="${BASELINE_SCALABLE_EVAL_FANOUTS:-$(scalable_eval_fanouts_for_layers "${TUNEDGNN_NUM_LAYERS}")}"
        exec "${PYTHON_BIN}" "${SCRIPTS_ROOT}/methods/scalable_sparse_node.py" \
            --method unified-lth \
            --display-step "${TUNEDGNN_LOG_EVERY:-10}" \
            --dataset "${DATASET}" \
            --data-root "${DATA_ROOT}" \
            --cache-root "${CACHE_ROOT}" \
            --device "${DEVICE}" \
            --epochs "${EPOCHS_OVERRIDE:-$TOTAL_EPOCH}" \
            --runs "${RUNS}" \
            --seed "${SEED}" \
            --kept-ratio "${KEPT_RATIO}" \
            --fanouts "${SCALABLE_FANOUTS}" \
            --eval-fanouts "${SCALABLE_EVAL_FANOUTS}" \
            --batch-size "${SCALABLE_BATCH_SIZE}" \
            --eval-batch-size "${SCALABLE_EVAL_BATCH_SIZE}" \
            --num-workers "${BASELINE_SCALABLE_NUM_WORKERS:-2}" \
            "${TUNED_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"
        ;;
esac

if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
    # Every fixed ticket receives the complete comparison-backbone training
    # budget. Mask-search epochs are separate method overhead.
    TOTAL_EPOCH="${EPOCHS_OVERRIDE}"
fi
printf '[RuntimeBudget] rounds=%s mask_epochs_per_round=%s fixed_ticket_epochs_per_round=%s final_ticket_epochs=%s\n' \
    "${PRUNE_ROUNDS}" "${MASK_EPOCH}" "${TOTAL_EPOCH}" "${TOTAL_EPOCH}"

cd "${ULTH_DIR}"
exec "${PYTHON_BIN}" main_pruning_imp.py \
    --dataset "${DATASET}" \
    --data_root "${DATA_ROOT}" \
    --device "${DEVICE_ARG}" \
    --total_epoch "${TOTAL_EPOCH}" \
    --mask_epoch "${MASK_EPOCH}" \
    --prune_rounds "${PRUNE_ROUNDS}" \
    --runs "${RUNS}" \
    --target_kept_ratio "${KEPT_RATIO}" \
    --weight_kept_ratio "${WEIGHT_KEPT_RATIO}" \
    "${SEED_ARGS[@]}" \
    "${TUNED_ARGS[@]}" \
    --init_soft_mask_type all_one \
    "${EXTRA_ARGS[@]}"
