# gsutil -m rm -rf gs://melodi-bucket0/models/debug || true
TIME=$(date +%s)
MODEL_DIR='./models' \
T5X_DIR='/home/seba-1511/t5x/' \
TFDS_DATA_DIR='/home/seba-1511/tfds_data' \
./prompt_tuning/scripts/sst2-demo.sh gs://melodi-bucket0/melodi_evaluation/task=hellaswag/model=xl_nodropout_spot/method=melodi-gradients-flan15_nodp_cut-lr1.0-m128h4bsz256-dp0.0/$TIME/ /home/seba-1511/tfds_data
