# gsutil -m rm -rf gs://melodi-bucket0/models/debug || true
TIME=$(date +%s)
MODEL_DIR='./models' \
T5X_DIR='/home/seba-1511/t5x/' \
TFDS_DATA_DIR='/home/seba-1511/tfds_data' \
./prompt_tuning/scripts/sst2-demo.sh gs://melodi-bucket0/melodi_evaluation/task=flan_debug/model=xl_nodropout_spot/method=melodi-gradients-flan16_nodp-lr0.25-m128h4bsz1024-dp0.0/$TIME/ /home/seba-1511/tfds_data
