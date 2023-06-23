tensorboard --logdir gs://melodi-bucket0/melodi_evaluation/task=mnli_mismatched_type_0/model=xl_nodropout_spot/method=adafactor-melodi-switch32-h16-parameters-gradients-base_multitoken-flan10star_cut1024_20prompts_10trajs-h16m4-lr1.0-dp0.0-eval50 --load_fast false --host 0.0.0.0 --port=60066
# tensorboard --logdir gs://melodi-bucket0/melodi_evaluation/ --load_fast false --host 0.0.0.0
# tensorboard --logdir gs://melodi-bucket0/melodi_evaluation/ --load_fast false --samples_per_plugin scalars=999999999 --host 0.0.0.0
# tensorboard --logdir gs://melodi-bucket0/models/ --load_fast false --host 0.0.0.0
# tensorboard --logdir gs://melodi-bucket0/melodi_training/ --load_fast false --host 0.0.0.0
