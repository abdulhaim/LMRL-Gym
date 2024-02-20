# conda init bash
# conda activate ./gym_venv 
export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadorawhite/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
export TOKENIZERS_PARALLELISM=false
python -m llm_rl_scripts.maze.bc.fully_observed_bc HF gpt2 gcs://rl-llm-bench-dataset-internal/maze/fully_observed_filtered_maze_data.jsonl --outputs_path=gcs://rail-tpus-isadora/test-fully-observed-maze/filtered_bc/ --data-mesh-shape 4 --model-mesh-shape 2 --epochs 500 --use-wandb --wandb-project fully_observed_100_epochs