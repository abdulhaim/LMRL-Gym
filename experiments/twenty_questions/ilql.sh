export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs2/users/isadorawhite/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
conda activate ./gym_venv
sudo chmod -R 777 .
python -m llm_rl_scripts.twenty_questions.ilql.train_ilql HF gpt2-medium gcs://rl-llm-bench-dataset-internal/twenty-questions/train.json gcs://rl-llm-bench-dataset-internal/twenty-questions/eval.json gcs://rl-llm-bench-dataset-internal/twenty-questions/oracle --outputs_path=gcs://rail-tpus-isadora/twenty-questions/ilql/ --eval-every-steps 512 --data-mesh-shape 4 --model-mesh-shape 1 --eval-at-beginning --epochs 1 --use-wandb --wandb-project twenty_questions
