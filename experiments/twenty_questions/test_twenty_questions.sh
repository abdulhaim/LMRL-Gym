export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadorawhite/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
conda init bash
conda activate ./gym_venv
python -m llm_rl_scripts.twenty_questions.bc.train_bc HF gpt2-medium gcs://rl-llm-bench-dataset-internal/twenty-questions/train.json gcs://rl-llm-bench-dataset-internal/twenty-questions/eval.json gcs://rl-llm-bench-dataset-internal/twenty-questions/oracle --outputs_path=gcs://rail-tpus-isadora/test-twenty-questions/bc/ --data-mesh-shape 4 --model-mesh-shape 2 --epochs 1 --use-wandb --wandb-project twenty_questions
# python llm_rl_scripts/twenty_questions/bc/train_bc.py HF gpt2 gcs://rl-llm-bench-dataset-internal/twenty-questions/train.json gcs://rl-llm-bench-dataset-internal/twenty-questions/eval.json gcs://rl-llm-bench-dataset-internal/twenty-questions/oracle --outputs_path gcs://rail-tpus-isadora/test-car_dealer/bc --data-mesh-shape 4 --model-mesh-shape 2 --epochs 1
