export GOOGLE_APPLICATION_CREDENTIALS="INCLUDE CREDENTIALS"
export GCLOUD_PROJECT="INCLUDE GCLOUD PROJECT"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
conda init bash
conda activate LLM_RL

# Include --outputs_path file
python -m llm_rl_scripts.twenty_questions.bc.train_bc HF gpt2 gcs://rl-llm-bench-dataset-internal/twenty-questions/train.json gcs://rl-llm-bench-dataset-internal/twenty-questions/eval.json gcs://rl-llm-bench-dataset-internal/twenty-questions/oracle --data-mesh-shape 4 --model-mesh-shape 2 --epochs 1 --use-wandb --wandb-project twenty_questions
