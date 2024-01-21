export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadorawhite/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
conda init bash
conda activate ./gym_venv
python -m llm_rl_scripts.twenty_questions.ilql.train_ilql HF gpt2-medium gcs://rl-llm-bench-dataset-internal/twenty-questions/train.json gcs://rl-llm-bench-dataset-internal/twenty-questions/eval.json HF gcs://rl-llm-bench-dataset-internal/twenty-questions/oracle --outputs_path=gcs://rail-tpus-isadora/test-twenty-questions/ilql/ --data-mesh-shape 1 --model-mesh-shape 8 --epochs 1 --use-wandb --wandb-project twenty_questions