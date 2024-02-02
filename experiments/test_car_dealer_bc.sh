# conda init bash
# conda activate ./gym_venv 
export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadorawhite/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
python llm_rl_scripts.car_dealer.bc.train_bc.py HF gpt2 gcs://rl-llm-bench-dataset-internal/maze/fully_observed_filtered_maze_data.jsonl --outputs_path=gcs://rail-tpus-isadora/test-car_dealer/bc --data-mesh-shape 4 --model-mesh-shape 2 --epochs 1