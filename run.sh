# gsutil -m rm -rf gs://melodi-bucket0/models/debug || true
TIME=$(date +%s)
MODEL_DIR='./models' \
T5X_DIR='/home/seba-1511/t5x/' \
TFDS_DATA_DIR='/home/seba-1511/tfds_data' \
./prompt_tuning/scripts/sst2-demo.sh gs://melodi-bucket0/melodi_evaluation/task=debug/model=xl_nodropout_spot/method=precomputed-groundtruth-lr1.0-dp0.0/$TIME/ /home/seba-1511/tfds_data
# ./prompt_tuning/scripts/sst2-demo.sh gs://melodi-bucket0/melodi_evaluation/task=mnli/model=xl_nodropout_spot/method=melodi-base-gradients-flan16_nodp-lr1.0-m128h4bsz1024-dp0.0/$TIME/ /home/seba-1511/tfds_data
