#!/usr/bin/env bash

MODEL_DIR=${1:-${MODEL_DIR}}
TFDS_DATA_DIR=${2:-${TFDS_DATA_DIR}}

if [ -z ${MODEL_DIR} ] || [ -z ${TFDS_DATA_DIR} ]; then
  echo "usage: ./sst2-demo.sh gs://your-bucket/path/to/model_dir gs://your-bucket/path/to/tfds/cache"
  exit 1
fi

T5X_DIR="`python3 -m prompt_tuning.scripts.find_module t5x`/.."
FLAXFORMER_DIR="`python3 -m prompt_tuning.scripts.find_module flaxformer`/.."
PROMPT_DIR="`python3 -m prompt_tuning.scripts.find_module prompt_tuning`/.."
echo "Searching for gin configs in:"
echo "- ${T5X_DIR}"
echo "- ${FLAXFORMER_DIR}"
echo "- ${PROMPT_DIR}"
echo "============================="
# PRETRAINED_MODEL="gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_base/checkpoint_1100000"
# PRETRAINED_MODEL="gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_large/checkpoint_1100000"
PRETRAINED_MODEL="gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_xl/checkpoint_1100000"

python3 -m t5x.train \
  --gin_search_paths="${T5X_DIR},${FLAXFORMER_DIR},${PROMPT_DIR}" \
  --gin_file="prompt_tuning/configs/models/t5_1_1_xl_prompt.gin" \
  --gin_file="prompt_tuning/configs/runs/prompt_finetune.gin" \
  --gin_file="prompt_tuning/configs/melodi/optax_optimizer.gin" \
  --gin_file="../melodi/experimental/gins/tasks/flan.gin" \
  --gin_file="../melodi/experimental/gins/methods/prompt_init/class_labels_small.gin" \
  --gin.FLAN_TASK="'mnli_mismatched_type_0'" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
  --gin.TRAIN_STEPS="1_101_000" \
  --gin.EVAL_PERIOD=50 \
  --gin.EVAL_STEPS=100 \
  --gin.DROPOUT_RATE=0.0 \
  --gin.OPTAX_LEARNING_RATE=1.0 \
  --gin.OPTAX_MOMENTUM=0.0 \
  --gin.OPTAX_MELODI_PATH='"gs://melodi-bucket0/melodi_training/xl-newhyper/task=flan11star_nodropout_20prompts_cut1500/model=multitoken_base_sequence_parampreds/horizon=8/memory=6/bsz=512/lr=3e-4/mse=uniform/1685608673/"' \
  --gin.OPTAX_MELODI_MEMORY=6 \
  --gin.OPTAX_MELODI_MODEL='"base-gradients-multitoken"' \
  --gin.OPTAX_OPTIMIZER='"melodi"' \
  --gin.BATCH_SIZE=128 \
  --gin.Trainer.num_microbatches=32 \
  --gin.PROMPT_LENGTH=20 \
  --gin.RANDOM_SEED=100 \
  --gin.partitioning.PjitPartitioner.model_parallel_submesh="(2,2,1,2)" \
  --gin.train_eval/utils.DatasetConfig.batch_size=64 \
  --gin.infer_eval/utils.DatasetConfig.batch_size=64 \
  --tfds_data_dir=${TFDS_DATA_DIR}
