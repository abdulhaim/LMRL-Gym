import jax
import openai
import os
import tiktoken
import jax.numpy as jnp
from LLM_RL.environment import TextPolicy, TextHistory, Text
from JaxSeq.utils import load_mesh
from llm_rl_scripts.car_dealer.env.env import CarDealerPolicyEnvironment, BatchedCarDealerPolicyEnvironment
from llm_rl_scripts.car_dealer.env.buyer import BatchedGPT2BuyerPolicy
from JaxSeq.models.gpt2.interface import GPT2Inference
from JaxSeq.models.gpt2.load import ModelLoadMode, load_params
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, uuid_name, jsonl_load, get_weight_decay_mask, create_path, get_enabled_save_path
from LLM_RL.environment import Text, TextEnv, TextHistory, interact_environment
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

def text_history_to_str(text_history: TextHistory) -> str:
    return '\n'.join(map(lambda x: x.text, text_history))

class UserPolicy(TextPolicy):
    def __init__(self):
        super().__init__()

    def act(self, text_history: TextHistory) -> TextHistory:
        print(text_history)
        
        result = input("Respond to the buyer:").strip()
        result+="\n"
        return text_history+(Text(result, True),)            


def eval(env, policy, data_mesh_shape: int=1, fsdp_mesh_shape: int=1, model_mesh_shape: int=-1):
    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    num_samples = 10
    verbose = False
    with mesh:
        
        rewards = []
        conversations = []

        def save_once():
            with open(convert_path(os.path.join(save_dir, 'conversations.json')), 'w') as f:
                json.dump(conversations, f, indent=4)

            stats = {
                "avg_reward": sum(rewards) / len(rewards),
                "rewards": rewards,
            }
            with open(convert_path(os.path.join(save_dir, 'stats.json')), 'w') as f:
                json.dump(stats, f, indent=2)

            print(f"saved to {save_dir}")

            return stats

        for i in range(num_samples):
            if verbose:
                print("=" * 25)
            print(f"sample: {i+1}")
            transitions = interact_environment(env, policy, env_options={"verbose": verbose})

            _, _, final_text_history, _, _ = transitions[-1]
            buyer_info = env.buyer_info
            output = env.output
            lines = create_lines_from_text_history(final_text_history)
            conversation = {
                "buyer_info": buyer_info,
                "lines": lines,
                "output": output,
            }
            
            reward = sum([r for _, _, _, r, _ in transitions])

            conversations.append(conversation)
            rewards.append(reward)

            print(f"reward: {reward}, avg_reward: {sum(rewards) / len(rewards)}")
            if verbose:
                print("=" * 25)

            if save_dir is not None and (i + 1) % save_every == 0:
                save_once()
        
        save_once()
def car_dealer_env(buyer_model_path, data_mesh_shape: int=1, fsdp_mesh_shape: int=1, model_mesh_shape: int=-1):
    prng_key = jax.random.PRNGKey(3)
    prng_key, buyer_inference_prng, buyer_policy_prng = jax.random.split(prng_key, 3)
    buyer_model_mode = ModelLoadMode.PARAMS
    model_load_mode = ModelLoadMode.HF 
    bf16_activations = False
    force_pad_embeddings= False
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))

    buyer_params, buyer_model = load_params(
        model_load_mode=buyer_model_mode, 
        model_load_path=convert_path(buyer_model_path) if model_load_mode != ModelLoadMode.HF else buyer_model_path, 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=buyer_inference_prng, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )
    # env = CarDealerPolicyEnvironment(
    #     buyer=buyer_model,
    #     max_conversation_length=50,
    #     reward_mode="fancy",
    # )
    rollout_bsize = 1
    env = BatchedCarDealerPolicyEnvironment(
        buyer=BatchedGPT2BuyerPolicy(
            inference=GPT2Inference.load_inference(
                params=buyer_params, 
                model=buyer_model, 
                tokenizer=tokenizer, 
            ), 
            prng_key=buyer_policy_prng, 
            generation_config=GenerationConfig(
                do_sample=True, 
                num_beams=1, 
                temperature=None, 
                top_p=None, 
                top_k=None, 
                eos_token_id=tokenizer.encode('\n')[0], 
                pad_token_id=tokenizer.pad_token_id, 
                max_new_tokens=128-1, 
            ), 
            blocking_strategy=BlockingStrategy(
                padding=Padding.LEFT, 
                truncation=Truncation.LEFT, 
                max_length=1024-128, 
            ), 
            out_str_process=lambda x: x.removesuffix('\n')+'\n', 
        ),
        buyer_bsize=rollout_bsize,
        max_conversation_length=50,
        reward_mode="revenue",
    )
    return env


