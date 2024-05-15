export GOOGLE_APPLICATION_CREDENTIALS="INCLUDE CREDENTIALS"
export GCLOUD_PROJECT="INCLUDE GCLOUD PROJECT"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
conda init bash
conda activate LLM_RL

# Training BC in LLM_RL
python -m llm_rl_scripts.car_dealer.bc.train_bc HF buyer_bc_gpt2xl_test gpt2-xl --output-path=car_dealer/outputs/ --data-path=car_dealer/data/ --epochs=18 --use-lr-schedule --train-bsize=16 --grad-accum-steps=8 --inference-bsize=32 --num-logs-per-epoch=4 --num-evals-per-epoch=4 --save-best --save-last --model-p-shape=4 --use-wandb

# Training MC in LLM_RL
python3 -m llm_rl_scripts.car_dealer.mc.train_mc_returns HF gpt2-xl train_bc.json eval_bc.json car_dealer/best --outputs_path=gcs://rail-tpus-marwa/car_dealer/mc/ --epochs 1     --train-bsize 128     --grad-accum-steps 1     --eval-loss-bsize 32     --eval-loss-batches 256     --policy-max-output-length 128     --log-every 256     --eval-every-steps 1024     --save-every-steps 1024     --save-at-end     --no-save-train-state     --data-mesh-shape -1     --fsdp-mesh-shape 1     --model-mesh-shape 1     --gradient-checkpointing

# Training ILQL in LLM_RL
python3 -m llm_rl_scripts.car_dealer.ilql.train_ilql HF gpt2-xl /nfs/nfs1/users/marwa/generation-benchmarks/cities_data/guess_my_city_train_bc.json /nfs/nfs1/users/marwa/generation-benchmarks/cities_data/guess_my_city_eval_bc.json  /nfs/nfs1/users/marwa/lmrl-gym-final/JaxSEQ/gs:/rail-tpus-marwa/guess_city/exp.2024-01-16-22-14-31.343.9f6185f4b4bc11ee8b6d872abd62eb10/best --outputs_path=gcs://rail-tpus-marwa/guess_city/ilql/ --epochs 1     --train-bsize 128     --grad-accum-steps 1     --eval-loss-bsize 32     --eval-loss-batches 256     --policy-max-output-length 128     --log-every 256     --eval-every-steps 1024     --save-every-steps 1024     --save-at-end     --no-save-train-state     --data-mesh-shape -1     --fsdp-mesh-shape 1     --model-mesh-shape 1     --gradient-checkpointing

# Training PPO in LLM_RL
python -m llm_rl_scripts.car_dealer.ppo.train_ppo PARAMS seller_bc_gpt2xl_test4_converted/model outputs/car_dealer/buyer_bc_gpt2xl_test4_converted/model  --exp-name ppo_revenue_gpt2xl_test1  --outputs-path gcs://rail-tpus-marwa/car_dealer/  --train-bsize 4   --grad-accum-steps 1000 --log-every 1000  --n-rounds 1000  --epochs 4 --n-rollouts 4000  --gamma 0.99 --rollout-bsize 4 --ppo_data_bsize 4    --eval-every-rounds 1 --weight-decay 0.0 --lr 5e-6 --save-every-rounds 50 --init-kl-coef 0.01   --cliprange-value 0.2 --cliprange 0.2   --value-loss-coef 1.0  --wandb-project car_dealer-ppo  --use-wandb 

# Human Eval in LLM_RL
python -m llm_rl_scripts/guess_city/misc/car_dealer_human_eval.py

# Evaluation any model in LLM_RL
python -m llm_rl_scripts/guess_city/misc/car_dealer_model_eval.py

