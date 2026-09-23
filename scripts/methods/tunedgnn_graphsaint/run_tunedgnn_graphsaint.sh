#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPTS_ROOT}/common/baseline_wrapper_lib.sh"
parse_common_args "$@"
KEPT_RATIO="1.0"
unset BASELINE_KEPT_RATIO
export BASELINE_METHOD_NAME="tunedgnn-graphsaint-rw"
setup_common_env
require_tunedgnn_preset_contract

METHOD_DIR="${SUPPORT_GRAPH_ROOT}/RelatedMethods/TunedGNN-GraphSAINT-RW"
# Sampler defaults are delegated to graphsaint_rw.py so this method samples
# exactly like the GraphSAINT-RW baseline it is compared against:
#   - small graphs:  batch=500, walk=2, steps=3, normalized sampling
#   - large graphs:  batch=6000/20000, walk=4, steps=30
EPOCHS=500
RUNS=1
# Sampler cost knobs, env-overridable; 0/-1 still delegate to the defaults in
# graphsaint_rw.py. num_steps is the big one: ONE epoch is num_steps subgraph
# builds plus num_steps forward/backward passes, so at the large-graph default
# of 30 an "epoch" is 30 gradient steps against a full-batch GCN's one.
# sample_coverage>0 buys normalization coefficients with an expensive one-time
# pre-sampling pass; 0 disables it.
NUM_STEPS="${BASELINE_GRAPHSAINT_NUM_STEPS:-0}"
WALK_LENGTH="${BASELINE_GRAPHSAINT_WALK_LENGTH:-0}"
BATCH_SIZE="${BASELINE_GRAPHSAINT_BATCH_SIZE:-0}"
SAMPLE_COVERAGE="${BASELINE_GRAPHSAINT_SAMPLE_COVERAGE:--1}"
SAMPLER_NUM_WORKERS="${BASELINE_GRAPHSAINT_SAMPLER_WORKERS:-0}"
EVAL_BATCH_SIZE=4096
# Honour the same knob the plain GraphSAINT wrapper honours. Hardcoded to 1,
# this was the ONLY method in the end-to-end campaign paying a full evaluation
# pass on every epoch: plain graphsaint with eval off runs 12.1 s/epoch on
# ogbn-products, this variant with eval on ran 227.2 -- an 18.8x gap that is
# evaluation, not training, and it is what put GraphSAINT at 84x in panel (b).
EVAL_STEP="${BASELINE_GRAPHSAINT_EVAL_EVERY:-1}"
if [[ "${SMOKE_TEST}" -eq 1 ]]; then
    EPOCHS=1
    RUNS=1
    NUM_STEPS=2
    SAMPLE_COVERAGE=10
    EVAL_STEP=1
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
export SCAFFOLD_SPLIT_PROTOCOL="${SCAFFOLD_SPLIT_PROTOCOL:-tunedgnn}"

# GraphSAINT's node-normalized loss estimates sum(loss over train)/num_nodes,
# not the per-train-node mean. Adam divides the data gradient out again, but
# weight_decay is added *before* the moment estimates, so the effective
# regularization is the preset weight_decay times num_nodes/num_train.
#
# That factor is ~2 on the datasets with 50/25/25 splits and harmless, but the
# Planetoid splits are 20 labels per class:
#
#   dataset   weight_decay  N/train   effective wd
#   citeseer      0.01        27.7       0.277     -> collapsed to 22.8%
#   pubmed        0.0005     328.6       0.164     -> collapsed to 38.8%
#   cora          0.0005      19.3       0.0097    -> fine, 83.2%
#
# Both collapsed runs pinned train accuracy at exactly 1/num_classes with a
# flat loss: weight decay drove the weights to zero before the data term could
# move them. Since this method exists to inherit the tunedGNN presets, and
# those presets pin weight_decay against a per-train-node mean, train_mean is
# the scale that makes them mean here what they mean for full-graph tunedGNN.
# It is exactly inert wherever weight_decay is 0, and inert on the large graphs
# (use_normalization=auto turns the correction off there, and the unnormalized
# branch already takes a per-train-node mean).
NORMALIZED_LOSS_SCALE="${BASELINE_GRAPHSAINT_LOSS_SCALE:-train_mean}"

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
print_tunedgnn_contract

# Deliberately the same cache root as run_graphsaint.sh. The sampler
# fingerprint covers the graph and every setting that changes the statistics,
# and this method leaves data.edge_index untouched, so a node_norm/edge_norm
# pair already computed for GraphSAINT-RW is reused as-is.
SAMPLER_CACHE="${CACHE_ROOT}/graphsaint/sampler"
mkdir -p "${SAMPLER_CACHE}" 2>/dev/null || true

cd "${METHOD_DIR}"
exec "${PYTHON_BIN}" tunedgnn_graphsaint_rw.py \
    --dataset "${DATASET}" \
    --data_root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --runs "${RUNS}" \
    "${SEED_ARGS[@]}" \
    --batch_size "${BATCH_SIZE}" \
    --walk_length "${WALK_LENGTH}" \
    --num_steps "${NUM_STEPS}" \
    --sample_coverage "${SAMPLE_COVERAGE}" \
    --use_normalization auto \
    --normalized_loss_scale "${NORMALIZED_LOSS_SCALE}" \
    --eval_step "${EVAL_STEP}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --sampler_cache_dir "${SAMPLER_CACHE}" \
    --sampler_num_workers "${SAMPLER_NUM_WORKERS}" \
    "${TUNED_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
