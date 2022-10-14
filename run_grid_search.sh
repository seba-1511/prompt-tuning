
for lr in 1.0 0.5 0.25 0.1 0.05 0.025 0.01 0.005
do
    for optimizer in adafactor.gin heavyball.gin sgd.gin
    do
        MODEL_DIR='gs://melodi-bucket0/models/opt='$optimizer'/lr='$lr
        TFDS_DATA_DIR='/home/seba-1511/tfds_data'

        T5X_DIR="`python3 -m prompt_tuning.scripts.find_module t5x`/.."
        FLAXFORMER_DIR="`python3 -m prompt_tuning.scripts.find_module flaxformer`/.."
        PROMPT_DIR="`python3 -m prompt_tuning.scripts.find_module prompt_tuning`/.."
        echo "Searching for gin configs in:"
        echo "- ${T5X_DIR}"
        echo "- ${FLAXFORMER_DIR}"
        echo "- ${PROMPT_DIR}"
        echo "============================="
        PRETRAINED_MODEL="gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_base/checkpoint_1100000"

        python3 -m t5x.train \
          --gin_search_paths="${T5X_DIR},${FLAXFORMER_DIR},${PROMPT_DIR}" \
          --gin_file="prompt_tuning/configs/models/t5_1_1_base_prompt.gin" \
          --gin_file="prompt_tuning/configs/prompts/from_class_labels.gin" \
          --gin_file="prompt_tuning/configs/runs/prompt_finetune.gin" \
          --gin_file="prompt_tuning/configs/prompt_optimizers/"$optimizer \
          --gin_file="../melodi/experimental/gins/tasks/mnli.gin" \
          --gin.CLASS_LABELS="['positive', 'negative']" \
          --gin.MODEL_DIR="'${MODEL_DIR}'" \
          --gin.MIXTURE_OR_TASK_NAME="'glue_mnli_and_dev_v002'" \
          --gin.TASK_FEATURE_LENGTHS="{'inputs': 1024, 'targets': 32}" \
          --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
          --gin.TRAIN_STEPS="1_150_000" \
          --gin.EVAL_PERIOD=1000 \
          --gin.DROPOUT_RATE=0 \
          --gin.PROMPT_LEARNING_RATE=$lr \
          --tfds_data_dir=${TFDS_DATA_DIR}
    done
done

