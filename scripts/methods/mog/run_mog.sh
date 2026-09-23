#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPTS_ROOT}/common/baseline_wrapper_lib.sh"
parse_common_args "$@"
export BASELINE_METHOD_NAME="mog"
setup_common_env
require_tunedgnn_preset_contract

MOG_DIR="${SUPPORT_GRAPH_ROOT}/RelatedMethods/MoG-main/citation"
EPOCHS=200
RUNS=1
if [[ "${SMOKE_TEST}" -eq 1 ]]; then
    EPOCHS=1
fi
if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
    EPOCHS="${EPOCHS_OVERRIDE}"
fi
if [[ -n "${RUNS_OVERRIDE}" ]]; then
    RUNS="${RUNS_OVERRIDE}"
fi
SEED="${SEED_OVERRIDE:-42}"
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

DATASET_KEY="$(printf '%s' "${DATASET}" | tr '[:upper:]_' '[:lower:]-')"
case "${DATASET_KEY}" in
    reddit|reddit2|ogbn-products|ogb-products|products|ogbn-arxiv|ogb-arxiv|arxiv|ogbn-proteins|ogb-proteins|proteins|pokec)
        # Native MoG learns one mask on each induced graph and reuses it across
        # GNN layers.  Its OGBN-Proteins implementation uses random node
        # partitions, as do tunedGNN Products and Pokec.  The former neighbor
        # path turned Pokec's three tunedGNN partitions into ~6,378 optimizer
        # batches per epoch and was therefore not the intended contract.
        MOG_TRAIN_PARTS=10
        MOG_EVAL_PARTS=10
        MOG_SOURCE_PROFILE="ogbn-arxiv"
        case "${DATASET_KEY}" in
            pokec)
                MOG_TRAIN_PARTS=3
                MOG_EVAL_PARTS=3
                ;;
            ogbn-arxiv|ogb-arxiv|arxiv)
                # The native Arxiv program is full graph.
                MOG_TRAIN_PARTS=1
                MOG_EVAL_PARTS=1
                ;;
            ogbn-proteins|ogb-proteins|proteins)
                MOG_SOURCE_PROFILE="ogbn-proteins"
                # Native MoG defaults: ten train and five fixed eval parts.
                MOG_TRAIN_PARTS=10
                MOG_EVAL_PARTS=5
                ;;
        esac
        MOG_TRAIN_PARTS="${BASELINE_MOG_TRAIN_PARTS:-${MOG_TRAIN_PARTS}}"
        MOG_EVAL_PARTS="${BASELINE_MOG_EVAL_PARTS:-${MOG_EVAL_PARTS}}"
        MOG_WORKERS="${BASELINE_SCALABLE_NUM_WORKERS:-2}"
        if ((MOG_WORKERS > MOG_TRAIN_PARTS)); then
            MOG_WORKERS="${MOG_TRAIN_PARTS}"
        fi
        # Use tunedGNN's own per-dataset epoch budget, which the caller
        # already supplies from Configuration.py DATASET_OVERRIDES (Reddit and
        # Products 300, Proteins 100, Arxiv and Pokec 2000). MoG therefore
        # trains on exactly the same schedule as every other method it is
        # compared against.
        MOG_EPOCHS="${BASELINE_MOG_EPOCHS:-${EPOCHS}}"
        # The shared contract evaluates every 9 epochs from epoch 1001, which
        # fires only 111 times on the 2000-epoch runs and never at all on the
        # 100-300 epoch ones. Target ~20 evaluations per run instead, keeping
        # the 250-epoch cadence for the full-length 2000-epoch runs.
        if ((MOG_EPOCHS >= 2000)); then
            MOG_EVAL_EVERY_DEFAULT=250
        else
            MOG_EVAL_EVERY_DEFAULT=$((MOG_EPOCHS / 20))
            ((MOG_EVAL_EVERY_DEFAULT < 1)) && MOG_EVAL_EVERY_DEFAULT=1
        fi
        MOG_EVAL_EVERY="${BASELINE_MOG_EVAL_EVERY:-${MOG_EVAL_EVERY_DEFAULT}}"
        MOG_EVAL_START_EPOCH="${BASELINE_MOG_EVAL_START_EPOCH:-${MOG_EVAL_EVERY}}"
        echo "[MoGSource] dataset=${DATASET_KEY} source=${SUPPORT_GRAPH_ROOT}/RelatedMethods/MoG-main/${MOG_SOURCE_PROFILE//-/_} adapter=random-node-partition train_parts=${MOG_TRAIN_PARTS} eval_parts=${MOG_EVAL_PARTS} epochs=${MOG_EPOCHS}/${EPOCHS} eval_every=${MOG_EVAL_EVERY} eval_start=${MOG_EVAL_START_EPOCH}"
        exec "${PYTHON_BIN}" "${SCRIPTS_ROOT}/methods/scalable_sparse_node.py" \
            --method mog \
            --mog-source-profile "${MOG_SOURCE_PROFILE}" \
            --dataset "${DATASET}" \
            --data-root "${DATA_ROOT}" \
            --cache-root "${CACHE_ROOT}" \
            --device "${DEVICE}" \
            --epochs "${MOG_EPOCHS}" \
            --runs "${RUNS}" \
            --seed "${SEED}" \
            --kept-ratio "${KEPT_RATIO}" \
            --loader-mode random-node \
            --train-parts "${MOG_TRAIN_PARTS}" \
            --eval-parts "${MOG_EVAL_PARTS}" \
            --num-workers "${MOG_WORKERS}" \
            --eval-step "${MOG_EVAL_EVERY}" \
            --eval-start-epoch "${MOG_EVAL_START_EPOCH}" \
            --display-step "${TUNEDGNN_LOG_EVERY:-10}" \
            "${TUNED_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"
        ;;
esac

cd "${MOG_DIR}"
exec "${PYTHON_BIN}" main.py \
    --dataset "${DATASET}" \
    --data_root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --runs "${RUNS}" \
    "${SEED_ARGS[@]}" \
    --log_steps 1 \
    --k_list "${KEPT_RATIO}" "${KEPT_RATIO}" "${KEPT_RATIO}" \
    --expert_select 3 \
    "${TUNED_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
