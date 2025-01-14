#!/usr/bin/env bash

TIME=$(date +%s)
TFDS_DATA_DIR=${1:-${TFDS_DATA_DIR}}

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

FLAN_TASK="mnli_mismatched_type_0"

# melodi
MODEL_DIR="gs://melodi-bucket0/melodi_evaluation/20230724/task=${FLAN_TASK}/model=xl_nodropout_spot/method=melodi_mse_rawnorm-h8-gradients-base_multitoken-flan10star_cut1024_20prompts_10trajs-h8m4-lr1.0-dp0.0-eval50/${TIME}/"
python3 -m t5x.train \
  --gin_search_paths="${T5X_DIR},${FLAXFORMER_DIR},${PROMPT_DIR}" \
  --gin_file="prompt_tuning/configs/models/t5_1_1_xl_prompt.gin" \
  --gin_file="prompt_tuning/configs/runs/prompt_finetune.gin" \
  --gin_file="prompt_tuning/configs/melodi/optax_optimizer.gin" \
  --gin_file="../melodi/experimental/gins/tasks/flan.gin" \
  --gin_file="../melodi/experimental/gins/methods/prompt_init/class_labels_small.gin" \
  --gin.FLAN_TASK="'${FLAN_TASK}'" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
  --gin.TRAIN_STEPS="1_101_000" \
  --gin.EVAL_PERIOD=50 \
  --gin.EVAL_STEPS=100 \
  --gin.DROPOUT_RATE=0.0 \
  --gin.OPTAX_LEARNING_RATE=1.0 \
  --gin.OPTAX_MOMENTUM=0.0 \
  --gin.OPTAX_MELODI_PATH='"gs://melodi-bucket0/melodi_training/20230724/xl-newhyper/task=flan10star_nodropout_20prompts_10trajs_cut1024_parampreds0/model=multitoken_base_sequence_gfirst0_resNone/horizon=8/memory=4/bsz=512/lr=1e-4/mse=rawnorm/1689971879/"' \
  --gin.OPTAX_MELODI_MEMORY=4 \
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

MODEL_DIR="gs://melodi-bucket0/melodi_evaluation/20230724/task=${FLAN_TASK}/model=xl_nodropout_spot/method=melodi_mse_rawnorm-h16-gradients-base_multitoken-flan10star_cut1024_20prompts_10trajs-h16m4-lr1.0-dp0.0-eval50/${TIME}/"
python3 -m t5x.train \
  --gin_search_paths="${T5X_DIR},${FLAXFORMER_DIR},${PROMPT_DIR}" \
  --gin_file="prompt_tuning/configs/models/t5_1_1_xl_prompt.gin" \
  --gin_file="prompt_tuning/configs/runs/prompt_finetune.gin" \
  --gin_file="prompt_tuning/configs/melodi/optax_optimizer.gin" \
  --gin_file="../melodi/experimental/gins/tasks/flan.gin" \
  --gin_file="../melodi/experimental/gins/methods/prompt_init/class_labels_small.gin" \
  --gin.FLAN_TASK="'${FLAN_TASK}'" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
  --gin.TRAIN_STEPS="1_101_000" \
  --gin.EVAL_PERIOD=50 \
  --gin.EVAL_STEPS=100 \
  --gin.DROPOUT_RATE=0.0 \
  --gin.OPTAX_LEARNING_RATE=1.0 \
  --gin.OPTAX_MOMENTUM=0.0 \
  --gin.OPTAX_MELODI_PATH='"gs://melodi-bucket0/melodi_training/20230724/xl-newhyper/task=flan10star_nodropout_20prompts_10trajs_cut1024_parampreds0/model=multitoken_base_sequence_gfirst0_resNone/horizon=16/memory=4/bsz=512/lr=1e-4/mse=rawnorm/1689981563/"' \
  --gin.OPTAX_MELODI_MEMORY=4 \
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

# # precomputed optimizer
# MODEL_DIR="gs://melodi-bucket0/melodi_evaluation/debug/task=${FLAN_TASK}/model=xl_nodropout_spot/method=precomputed-melodi-parameters-gradients-base_mlp_multitoken-flan_mnli_cut1024_20prompts_20trajs-h16m4-lr1.0-dp0.0-eval50/${TIME}/"
# python3 -m t5x.train \
#   --gin_search_paths="${T5X_DIR},${FLAXFORMER_DIR},${PROMPT_DIR}" \
#   --gin_file="prompt_tuning/configs/models/t5_1_1_xl_prompt.gin" \
#   --gin_file="prompt_tuning/configs/runs/prompt_finetune.gin" \
#   --gin_file="prompt_tuning/configs/melodi/optax_optimizer.gin" \
#   --gin_file="../melodi/experimental/gins/tasks/flan.gin" \
#   --gin_file="../melodi/experimental/gins/methods/prompt_init/class_labels_small.gin" \
#   --gin.FLAN_TASK="'${FLAN_TASK}'" \
#   --gin.MODEL_DIR="'${MODEL_DIR}'" \
#   --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
#   --gin.TRAIN_STEPS="1_102_000" \
#   --gin.EVAL_PERIOD=2 \
#   --gin.EVAL_STEPS=100 \
#   --gin.DROPOUT_RATE=0.0 \
#   --gin.OPTAX_LEARNING_RATE=1.0 \
#   --gin.OPTAX_MOMENTUM=0.0 \
#   --gin.OPTAX_MELODI_PATH='"gs://melodi-bucket0/melodi_training/grid-residual-params-grads/xl-newhyper/task=flan_mnli_nodropout_20prompts_20trajs_parampreds_cut1024_parampreds0/model=multitoken_base_sequence_params_grads_resmlp_gfirst0_resNone_segemb0/horizon=16/memory=4/bsz=512/lr=1e-3/mse=uniform/1687942358/inference-melodi-mnli_mismatched-base_sequence-flan_qnli_snli_cut1500-h16m16-uniform_mse-2k-1688077371/train_updates/"' \
#   --gin.OPTAX_MELODI_MEMORY=4 \
#   --gin.OPTAX_MELODI_MODEL='"base-gradients-parameters-mlp-multitoken"' \
#   --gin.OPTAX_OPTIMIZER='"precomputed_optimizer"' \
#   --gin.BATCH_SIZE=128 \
#   --gin.Trainer.num_microbatches=32 \
#   --gin.PROMPT_LENGTH=20 \
#   --gin.RANDOM_SEED=100 \
#   --gin.partitioning.PjitPartitioner.model_parallel_submesh="(2,2,1,2)" \
#   --gin.train_eval/utils.DatasetConfig.batch_size=64 \
#   --gin.infer_eval/utils.DatasetConfig.batch_size=64 \
#   --tfds_data_dir=${TFDS_DATA_DIR}

# # adafactor
# MODEL_DIR="gs://melodi-bucket0/melodi_evaluation/task=${FLAN_TASK}/model=xl_nodropout_spot/method=adafactor-lr0.3-dp0.0-eval50/${TIME}/"
# python3 -m t5x.train \
#   --gin_search_paths="${T5X_DIR},${FLAXFORMER_DIR},${PROMPT_DIR}" \
#   --gin_file="prompt_tuning/configs/models/t5_1_1_xl_prompt.gin" \
#   --gin_file="prompt_tuning/configs/runs/prompt_finetune.gin" \
#   --gin_file="prompt_tuning/configs/melodi/optax_optimizer.gin" \
#   --gin_file="../melodi/experimental/gins/tasks/flan.gin" \
#   --gin_file="../melodi/experimental/gins/methods/prompt_init/class_labels_small.gin" \
#   --gin.FLAN_TASK="'${FLAN_TASK}'" \
#   --gin.MODEL_DIR="'${MODEL_DIR}'" \
#   --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
#   --gin.TRAIN_STEPS="1_102_000" \
#   --gin.EVAL_PERIOD=50 \
#   --gin.EVAL_STEPS=100 \
#   --gin.DROPOUT_RATE=0.0 \
#   --gin.OPTAX_LEARNING_RATE=0.3 \
#   --gin.OPTAX_MOMENTUM=0.0 \
#   --gin.OPTAX_MELODI_PATH='"gs://melodi-bucket0/melodi_training/xl-newhyper/task=flan_qnli_snli_nodropout_20prompts_10trajs_cut1500/model=multitoken_base_sequence/horizon=16/memory=4/bsz=512/lr=1e-3/mse=uniform/1686170598/"' \
#   --gin.OPTAX_MELODI_MEMORY=4 \
#   --gin.OPTAX_MELODI_MODEL='"base-gradients-multitoken"' \
#   --gin.OPTAX_OPTIMIZER='"adafactor"' \
#   --gin.BATCH_SIZE=128 \
#   --gin.Trainer.num_microbatches=32 \
#   --gin.PROMPT_LENGTH=20 \
#   --gin.RANDOM_SEED=100 \
#   --gin.partitioning.PjitPartitioner.model_parallel_submesh="(2,2,1,2)" \
#   --gin.train_eval/utils.DatasetConfig.batch_size=64 \
#   --gin.infer_eval/utils.DatasetConfig.batch_size=64 \
#   --tfds_data_dir=${TFDS_DATA_DIR}
#
#
# # heavyball
# MODEL_DIR="gs://melodi-bucket0/melodi_evaluation/task=${FLAN_TASK}/model=xl_nodropout_spot/method=heavyball-lr0.3-mom0.9-dp0.0-eval50/${TIME}/"
# python3 -m t5x.train \
#   --gin_search_paths="${T5X_DIR},${FLAXFORMER_DIR},${PROMPT_DIR}" \
#   --gin_file="prompt_tuning/configs/models/t5_1_1_xl_prompt.gin" \
#   --gin_file="prompt_tuning/configs/runs/prompt_finetune.gin" \
#   --gin_file="prompt_tuning/configs/melodi/optax_optimizer.gin" \
#   --gin_file="../melodi/experimental/gins/tasks/flan.gin" \
#   --gin_file="../melodi/experimental/gins/methods/prompt_init/class_labels_small.gin" \
#   --gin.FLAN_TASK="'${FLAN_TASK}'" \
#   --gin.MODEL_DIR="'${MODEL_DIR}'" \
#   --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
#   --gin.TRAIN_STEPS="1_102_000" \
#   --gin.EVAL_PERIOD=50 \
#   --gin.EVAL_STEPS=100 \
#   --gin.DROPOUT_RATE=0.0 \
#   --gin.OPTAX_LEARNING_RATE=0.3 \
#   --gin.OPTAX_MOMENTUM=0.9 \
#   --gin.OPTAX_MELODI_PATH='"gs://melodi-bucket0/melodi_training/xl-newhyper/task=flan_qnli_snli_nodropout_20prompts_10trajs_cut1500/model=multitoken_base_sequence/horizon=16/memory=4/bsz=512/lr=1e-3/mse=uniform/1686170598/"' \
#   --gin.OPTAX_MELODI_MEMORY=4 \
#   --gin.OPTAX_MELODI_MODEL='"base-gradients-multitoken"' \
#   --gin.OPTAX_OPTIMIZER='"heavyball"' \
#   --gin.BATCH_SIZE=128 \
#   --gin.Trainer.num_microbatches=32 \
#   --gin.PROMPT_LENGTH=20 \
#   --gin.RANDOM_SEED=100 \
#   --gin.partitioning.PjitPartitioner.model_parallel_submesh="(2,2,1,2)" \
#   --gin.train_eval/utils.DatasetConfig.batch_size=64 \
#   --gin.infer_eval/utils.DatasetConfig.batch_size=64 \
#   --tfds_data_dir=${TFDS_DATA_DIR}
#
#
# # normalized heavyball
# MODEL_DIR="gs://melodi-bucket0/melodi_evaluation/task=${FLAN_TASK}/model=xl_nodropout_spot/method=normalized_heavyball-lr10.0-mom0.9-dp0.0-eval50/${TIME}/"
# python3 -m t5x.train \
#   --gin_search_paths="${T5X_DIR},${FLAXFORMER_DIR},${PROMPT_DIR}" \
#   --gin_file="prompt_tuning/configs/models/t5_1_1_xl_prompt.gin" \
#   --gin_file="prompt_tuning/configs/runs/prompt_finetune.gin" \
#   --gin_file="prompt_tuning/configs/melodi/optax_optimizer.gin" \
#   --gin_file="../melodi/experimental/gins/tasks/flan.gin" \
#   --gin_file="../melodi/experimental/gins/methods/prompt_init/class_labels_small.gin" \
#   --gin.FLAN_TASK="'${FLAN_TASK}'" \
#   --gin.MODEL_DIR="'${MODEL_DIR}'" \
#   --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
#   --gin.TRAIN_STEPS="1_102_000" \
#   --gin.EVAL_PERIOD=50 \
#   --gin.EVAL_STEPS=100 \
#   --gin.DROPOUT_RATE=0.0 \
#   --gin.OPTAX_LEARNING_RATE=10.0 \
#   --gin.OPTAX_MOMENTUM=0.9 \
#   --gin.OPTAX_MELODI_PATH='"gs://melodi-bucket0/melodi_training/xl-newhyper/task=flan_qnli_snli_nodropout_20prompts_10trajs_cut1500/model=multitoken_base_sequence/horizon=16/memory=4/bsz=512/lr=1e-3/mse=uniform/1686170598/"' \
#   --gin.OPTAX_MELODI_MEMORY=4 \
#   --gin.OPTAX_MELODI_MODEL='"base-gradients-multitoken"' \
#   --gin.OPTAX_OPTIMIZER='"normalized_heavyball"' \
#   --gin.BATCH_SIZE=128 \
#   --gin.Trainer.num_microbatches=32 \
#   --gin.PROMPT_LENGTH=20 \
#   --gin.RANDOM_SEED=100 \
#   --gin.partitioning.PjitPartitioner.model_parallel_submesh="(2,2,1,2)" \
#   --gin.train_eval/utils.DatasetConfig.batch_size=64 \
#   --gin.infer_eval/utils.DatasetConfig.batch_size=64 \
#   --tfds_data_dir=${TFDS_DATA_DIR}
