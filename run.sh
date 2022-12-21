# gsutil -m rm -rf gs://melodi-bucket0/models/debug || true
TIME=$(date +%s)
MODEL_DIR='./models' \
T5X_DIR='/home/seba-1511/t5x/' \
TFDS_DATA_DIR='/home/seba-1511/tfds_data' \
./prompt_tuning/scripts/sst2-demo.sh gs://melodi-bucket0/melodi_evaluation/task=squad/model=xl_newhyper_spot/method=melodi-sequence-flan68-lr1.0-m128h32-dp0.1/$TIME/ /home/seba-1511/tfds_data
