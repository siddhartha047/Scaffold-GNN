#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPTS_ROOT}/common/baseline_wrapper_lib.sh"
parse_common_args "$@"
# TunedGNN is an unsparsified baseline. Accept and discard the common wrapper's
# legacy ratio option so it cannot affect validation, execution, or metadata.
KEPT_RATIO="1.0"
unset BASELINE_KEPT_RATIO
TUNEDGNN_MODEL="${TUNEDGNN_MODEL:-gcn}"
if [[ "${TUNEDGNN_MODEL}" != "gcn" && "${TUNEDGNN_MODEL}" != "sage" ]]; then
    echo "TUNEDGNN_MODEL must be gcn or sage, got: ${TUNEDGNN_MODEL}" >&2
    exit 2
fi
if [[ "${TUNEDGNN_MODEL}" == "sage" ]]; then
    export BASELINE_METHOD_NAME="tunedgnn-graphsage"
else
    export BASELINE_METHOD_NAME="tunedgnn"
fi
setup_common_env
require_tunedgnn_preset_contract
print_tunedgnn_contract
export SCAFFOLD_SPLIT_PROTOCOL="${SCAFFOLD_SPLIT_PROTOCOL:-tunedgnn}"
if [[ -n "${SEED_OVERRIDE}" ]]; then
    export SCAFFOLD_DATASET_SEED="${SEED_OVERRIDE}"
    export BASELINE_SEED="${SEED_OVERRIDE}"
fi

TUNED_MEDIUM_DIR="${SUPPORT_GRAPH_ROOT}/RelatedMethods/tunedGNN-main/medium_graph"
TUNED_LARGE_DIR="${SUPPORT_GRAPH_ROOT}/RelatedMethods/tunedGNN-main/large_graph"
EPOCHS=500
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
SEED_ARGS=()
if [[ -n "${SEED_OVERRIDE}" ]]; then
    SEED_ARGS=(--seed "${SEED_OVERRIDE}")
fi
TUNED_EVAL_EVERY="${BASELINE_TUNEDGNN_EVAL_EVERY:-}"
# display_step used to be pinned to eval_step, so turning evaluation off
# also turned per-epoch logging off and left one "Epoch:" line in the whole
# run. The entrypoint already handles the split -- product.py:265 prints
# "Epoch: N, Loss: ..., Eval: skipped" on display_step alone -- so the two
# are separate knobs here. Default 1 keeps every epoch logged.
TUNED_DISPLAY_EVERY="${BASELINE_TUNEDGNN_DISPLAY_EVERY:-1}"

canonical_dataset_key() {
    printf '%s' "$1" | tr '[:upper:]_' '[:lower:]-'
}

dataset_key="$(canonical_dataset_key "${DATASET}")"
TUNED_DIR="${TUNED_MEDIUM_DIR}"
ENTRYPOINT="main.py"
DATASET_ARG="${DATASET}"
PRODUCTS_ORIGINAL=0
PROTEIN_ORIGINAL=0
MODEL_ARGS=(--display_step 1)

case "${dataset_key}" in
    reddit|reddit2|ogb-products|ogbn-products|products|\
    ogb-arxiv|ogbn-arxiv|arxiv|\
    ogb-protein|ogb-proteins|ogbn-protein|ogbn-proteins|protein|proteins|pokec)
        TUNED_DIR="${TUNED_LARGE_DIR}"
        ENTRYPOINT="main-batch.py"
        MODEL_ARGS=(--display_step 1)
        if [[ "${dataset_key}" != "pokec" ]]; then
            unset BASELINE_PARTITION_MODE
            unset BASELINE_PARTITION_COUNT
            unset BASELINE_PARTITION_ID
            unset BASELINE_PARTITION_MAX_NODES
            unset BASELINE_PARTITION_EDGE_SAMPLE_SIZE
            unset BASELINE_PARTITION_CACHE_ROOT
        fi
        ;;
esac

case "${dataset_key}" in
    reddit|reddit2|ogb-products|ogbn-products|products)
        # Reddit has no original tunedGNN command and intentionally reuses the
        # corresponding OGBN-Products GCN/GraphSAGE model configuration.
        PRODUCTS_ORIGINAL=1
        ENTRYPOINT="product.py"
        case "${dataset_key}" in
            reddit2) DATASET_ARG="reddit2" ;;
            reddit) DATASET_ARG="reddit" ;;
            *) DATASET_ARG="ogbn-products" ;;
        esac
        MODEL_ARGS=(--gnn "${TUNEDGNN_MODEL}")
        if [[ -n "${TUNED_EVAL_EVERY}" ]]; then
            MODEL_ARGS+=(--eval_step "${TUNED_EVAL_EVERY}" --display_step "${TUNED_DISPLAY_EVERY}")
        fi
        ;;
    ogb-arxiv|ogbn-arxiv|arxiv)
        # RelatedMethods/tunedGNN-main/large_graph/arxiv.sh GCN line.
        ENTRYPOINT="main-arxiv.py"
        DATASET_ARG="ogbn-arxiv"
        MODEL_ARGS=(--display_step 1)
        if [[ "${TUNEDGNN_MODEL}" == "sage" ]]; then
            MODEL_ARGS+=(--sage)
        fi
        if [[ -n "${TUNED_EVAL_EVERY}" ]]; then
            MODEL_ARGS+=(--eval_step "${TUNED_EVAL_EVERY}" --display_step "${TUNED_DISPLAY_EVERY}")
        fi
        ;;
    ogb-protein|ogb-proteins|ogbn-protein|ogbn-proteins|protein|proteins)
        # RelatedMethods/tunedGNN-main/large_graph/proteins.sh GCN line:
        # python protein.py --gpu 7 --mpnn gcn --n-layers 3 --n-epochs 100
        PROTEIN_ORIGINAL=1
        ENTRYPOINT="protein.py"
        DATASET_ARG="ogbn-proteins"
        MODEL_ARGS=(
            --gpu "${DEVICE}"
            --mpnn "${TUNEDGNN_MODEL}"
            --n-epochs "${EPOCHS}"
            --n-runs "${RUNS}"
            --n-heads "${TUNEDGNN_HEADS}"
        )
        if [[ -n "${TUNED_EVAL_EVERY}" ]]; then
            MODEL_ARGS+=(--eval-every "${TUNED_EVAL_EVERY}" --log-every "${TUNED_EVAL_EVERY}")
        fi
        ;;
    pokec)
        # RelatedMethods/tunedGNN-main/large_graph/pokec.sh GCN line.
        DATASET_ARG="pokec"
        MODEL_ARGS=(
            --batch_size 550000
            --eval_step 9
            --display_step 9
        )
        if [[ "${TUNEDGNN_MODEL}" == "sage" ]]; then
            MODEL_ARGS+=(--sage)
        fi
        if [[ -n "${TUNED_EVAL_EVERY}" ]]; then
            MODEL_ARGS+=(--eval_step "${TUNED_EVAL_EVERY}" --display_step "${TUNED_DISPLAY_EVERY}")
        fi
        if [[ "${SMOKE_TEST}" -ne 1 ]]; then
            MODEL_ARGS+=(--eval_epoch 1000)
        fi
        ;;
esac

if [[ "${PROTEIN_ORIGINAL}" -eq 1 ]]; then
    MODEL_ARGS+=(
        --n-hidden "${TUNEDGNN_HIDDEN_CHANNELS}"
        --n-layers "${TUNEDGNN_NUM_LAYERS}"
        --lr "${TUNEDGNN_LEARNING_RATE}"
        --wd "${TUNEDGNN_WEIGHT_DECAY}"
        --dropout "${TUNEDGNN_DROPOUT}"
        --input-drop "${TUNEDGNN_INPUT_DROPOUT}"
    )
    [[ "${TUNEDGNN_JUMPING_KNOWLEDGE}" == "1" ]] && MODEL_ARGS+=(--jk)
elif [[ "${PRODUCTS_ORIGINAL}" -eq 1 ]]; then
    MODEL_ARGS+=(
        --hidden_channels "${TUNEDGNN_HIDDEN_CHANNELS}"
        --num_layers "${TUNEDGNN_NUM_LAYERS}"
        --dropout "${TUNEDGNN_DROPOUT}"
        --loader_workers "${BASELINE_TUNEDGNN_LOADER_WORKERS:-0}"
        --eval_num_parts "${BASELINE_TUNEDGNN_EVAL_NUM_PARTS:-1}"
    )
    [[ "${TUNEDGNN_LAYER_NORM}" == "1" ]] && MODEL_ARGS+=(--ln)
    [[ "${TUNEDGNN_RESIDUAL_CONNECTIONS}" == "1" ]] && MODEL_ARGS+=(--res)
    [[ "${TUNEDGNN_JUMPING_KNOWLEDGE}" == "1" ]] && MODEL_ARGS+=(--jk)
elif [[ "${TUNED_DIR}" == "${TUNED_LARGE_DIR}" ]]; then
    MODEL_ARGS+=(
        --hidden_channels "${TUNEDGNN_HIDDEN_CHANNELS}"
        --local_layers "${TUNEDGNN_NUM_LAYERS}"
        --lr "${TUNEDGNN_LEARNING_RATE}"
        --weight_decay "${TUNEDGNN_WEIGHT_DECAY}"
        --dropout "${TUNEDGNN_DROPOUT}"
        --in_dropout "${TUNEDGNN_INPUT_DROPOUT}"
        --metric "${TUNEDGNN_METRIC}"
    )
    [[ "${TUNEDGNN_MODEL}" == "sage" ]] && MODEL_ARGS+=(--sage)
    [[ "${TUNEDGNN_RESIDUAL_CONNECTIONS}" == "1" ]] && MODEL_ARGS+=(--res)
    [[ "${TUNEDGNN_LAYER_NORM}" == "1" ]] && MODEL_ARGS+=(--ln)
    [[ "${TUNEDGNN_BATCH_NORM}" == "1" ]] && MODEL_ARGS+=(--bn)
    [[ "${TUNEDGNN_JUMPING_KNOWLEDGE}" == "1" ]] && MODEL_ARGS+=(--jk)
else
    MODEL_ARGS+=(
        --gnn "${TUNEDGNN_MODEL}"
        --hidden_channels "${TUNEDGNN_HIDDEN_CHANNELS}"
        --local_layers "${TUNEDGNN_NUM_LAYERS}"
        --lr "${TUNEDGNN_LEARNING_RATE}"
        --weight_decay "${TUNEDGNN_WEIGHT_DECAY}"
        --dropout "${TUNEDGNN_DROPOUT}"
        --metric "${TUNEDGNN_METRIC}"
    )
    [[ "${TUNEDGNN_PRE_LINEAR}" == "1" ]] && MODEL_ARGS+=(--pre_linear)
    [[ "${TUNEDGNN_RESIDUAL_CONNECTIONS}" == "1" ]] && MODEL_ARGS+=(--res)
    [[ "${TUNEDGNN_LAYER_NORM}" == "1" ]] && MODEL_ARGS+=(--ln)
    [[ "${TUNEDGNN_BATCH_NORM}" == "1" ]] && MODEL_ARGS+=(--bn)
    [[ "${TUNEDGNN_JUMPING_KNOWLEDGE}" == "1" ]] && MODEL_ARGS+=(--jk)
fi

printf '[TunedGNNLaunch] entrypoint=%s model=%s args=' \
    "${ENTRYPOINT}" "${TUNEDGNN_MODEL}"
printf '%q ' "${MODEL_ARGS[@]}" "${EXTRA_ARGS[@]}"
printf '\n'

cd "${TUNED_DIR}"
if [[ "${PRODUCTS_ORIGINAL}" -eq 1 ]]; then
    product_epochs="${EPOCHS}"
    for run in $(seq 1 "${RUNS}"); do
        run_seed=$((${SEED_OVERRIDE:-42} + run - 1))
        printf '[tunedgnn-products] profile=%s run %s/%s seed=%s\n' \
            "${TUNEDGNN_PRESET_PROFILE}" "${run}" "${RUNS}" "${run_seed}"
        BASELINE_RUN_ID="${run}" BASELINE_SEED="${run_seed}" \
        "${PYTHON_BIN}" "${ENTRYPOINT}" \
            --dataset "${DATASET_ARG}" \
            --device "${DEVICE}" \
            --data_dir "${DATA_ROOT}" \
            --seed "${run_seed}" \
            "${MODEL_ARGS[@]}" \
            --epochs "${product_epochs}" \
            "${EXTRA_ARGS[@]}"
    done
    exit 0
fi

if [[ "${PROTEIN_ORIGINAL}" -eq 1 ]]; then
    # Multiprocessing's AF_UNIX socket path is limited to roughly 108 bytes.
    # A nested method-cache TMPDIR exceeds that once Python appends its
    # generated resource-sharer suffix, leaving the DGL loader with no batches.
    dgl_tmp_root="${BASELINE_TUNEDGNN_DGL_TMPDIR:-${SLURM_TMPDIR:-/tmp}}"
    dgl_tmp_dir="${dgl_tmp_root%/}/tunedgnn_dgl_${USER:-user}_$$"
    mkdir -p "${dgl_tmp_dir}"
    export TMPDIR="${dgl_tmp_dir}"
    export TMP="${dgl_tmp_dir}"
    export TEMP="${dgl_tmp_dir}"
    printf '[tunedgnn-protein] TMPDIR=%s\n' "${TMPDIR}"
    exec "${PYTHON_BIN}" "${ENTRYPOINT}" \
        --data_dir "${DATA_ROOT}" \
        --loader_workers "${BASELINE_TUNEDGNN_LOADER_WORKERS:-1}" \
        "${MODEL_ARGS[@]}" \
        "${SEED_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
fi

exec "${PYTHON_BIN}" "${ENTRYPOINT}" \
    --dataset "${DATASET_ARG}" \
    --data_dir "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --runs "${RUNS}" \
    "${SEED_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
