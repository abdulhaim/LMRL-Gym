export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs2/users/isadorawhite/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
conda activate ./gym_venv
sudo chmod -R 777 .
python -m llm_rl_scripts.twenty_questions.ppo.train_ppo HF gpt2-medium gcs://rl-llm-bench-dataset-internal/twenty-questions/train.json gcs://rl-llm-bench-dataset-internal/twenty-questions/eval.json PARAMS gcs://rl-llm-bench-dataset-internal/twenty-questions/oracle --outputs_path=gcs://rail-tpus-isadora/twenty-questions/PPO/ --eval-every-rounds 1 --data-mesh-shape 4 --model-mesh-shape 1 --eval-at-beginning --n-rounds 100 --use-wandb --wandb-project twenty_questions
