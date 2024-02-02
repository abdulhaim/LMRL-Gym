export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadorawhite/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
conda init bash
conda activate ./gym_venv
sudo chmod -R 777 .
python -m llm_rl_scripts.wordle.ilql.train_ilql_gpt2 HF gpt2 gcs://rl-llm-bench-dataset-internal/wordle/train_data.jsonl gcs://rl-llm-bench-dataset-internal/wordle/eval_data.jsonl ./llm_rl_scripts/wordle/vocab/wordle_official_400.txt --outputs_path=gcs://rail-tpus-isadora/worlde/ilql/ --data-mesh-shape 4 --model-mesh-shape 2 --epochs 1 --use-wandb --wandb-project wordle --exp-name wordle_no_share_weights --train-bsize 4 --grad-accum-steps 32 --eval-every-steps 4096 --policy-bsize 2 --policy-n-rollouts 32 --log-every 256 --eval-at-beginning --save-at-end --epochs 10
