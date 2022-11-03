# gsutil -m rm -rf gs://melodi-bucket0/models/debug || true
MODEL_DIR='./models' \
T5X_DIR='/home/seba-1511/t5x/' \
TFDS_DATA_DIR='/home/seba-1511/tfds_data' \
./prompt_tuning/scripts/sst2-demo.sh gs://melodi-bucket0/models/melodiball-h32-lr-2.0-dense-matched /home/seba-1511/tfds_data
