export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadorawhite/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
conda init bash
conda activate ./gym_venv
sudo chmod 777 -R .
python -m llm_rl_scripts.car_dealer.bc.train_bc_gpt2 HF gpt2 gcs://rl-llm-bench-dataset-internal/car-dealer/train.json gcs://rl-llm-bench-dataset-internal/car-dealer/eval.json gcs://rl-llm-bench-dataset-internal/car-dealer/simulator/model --outputs_path=gcs://rail-tpus-isadora/car_dealer/bc/ --data-mesh-shape 4 --model-mesh-shape 2 --epochs 1 --use-wandb --wandb-project car_dealer --exp-name bc --eval-at-beginning --eval-every-steps 4096 --eval-loss-bsize 8 --policy-n-rollouts 16 --policy-bsize 16 --train-bsize 8 --grad-accum-steps 16 --epochs 100
# python llm_rl_scripts/twenty_questions/bc/train_bc.py HF gpt2 gcs://rl-llm-bench-dataset-internal/twenty-questions/train.json gcs://rl-llm-bench-dataset-internal/twenty-questions/eval.json gcs://rl-llm-bench-dataset-internal/twenty-questions/oracle --outputs_path gcs://rail-tpus-isadora/test-car_dealer/bc --data-mesh-shape 4 --model-mesh-shape 2 --epochs 1
