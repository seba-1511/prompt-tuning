# gsutil -m rm -rf gs://melodi-bucket0/models/debug || true
MODEL_DIR='./models' \
T5X_DIR='/home/seba-1511/t5x/' \
TFDS_DATA_DIR='/home/seba-1511/tfds_data' \
./prompt_tuning/scripts/sst2-demo.sh gs://melodi-bucket0/melodi_evaluation/task=mnli/model=xl_lm100_spot/method=adafactor-lr-0.3 /home/seba-1511/tfds_data
