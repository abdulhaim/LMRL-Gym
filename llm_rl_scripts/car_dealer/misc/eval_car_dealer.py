import contextlib
from typing import Any, Dict, List, Optional
import jax
from algorithms.jax_agent import Inference, JaxAgentPolicy
from algorithms.jax_bc.core import load_bc_inference
from algorithms.jax_frozen_ilql.core import load_frozen_ilql_inference
from algorithms.jax_frozen_ilql.load_frozen_ilql import load_frozen_ilql_default_full_inference
from algorithms.jax_ilql.models import BaseModelState, load_advanced_mlp4
from algorithms.jax_ilql.load_ilql import load_ilql_default_full_inference
from algorithms.jax_ilql.core import load_ilql_inference
from jax_utils.jax_shard import shard_params
from environment import Text, TextHistory, TextTransition, interact_environment
from jax_models.gpt2 import load_gpt2_model
from jax_models.t5 import load_t5_model
import numpy as np
from jax.experimental.maps import Mesh
import dcargs
from functools import partial
from utils.path import convert_path
from utils.randomness import seed_generator
import os
import pickle as pkl
import json
import random
from utils.randomness import seed_context
from environments.car_dealer.data import create_lines_from_text_history
from environments.car_dealer.env import CarDealerPolicyEnvironment 
import tree
from transformers import AutoTokenizer
from collections import defaultdict


def load_bc_policy(
    model_name: str, 
    rng: jax.random.KeyArray, 
    mesh: Mesh,
    checkpoint_path: Optional[str]=None, 
    checkpoint_is_sharded: bool=True, 
    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[Any]=None, 
    do_pjit: bool=True, 
    condition_str: Optional[str]=None, 
    max_sequence_length: int=512,
    max_new_tokens: int=80,
    do_sample: bool=False,
):
    if mesh is None:
        mesh = contextlib.nullcontext()
        data_p_shape = 1
    else:
        data_p_shape = mesh.devices.shape[0]

    if checkpoint_is_sharded and checkpoint_path is not None:
        tail_checkpoint, head_checkpoint = os.path.split(checkpoint_path.strip('/'))
        checkpoint_path = os.path.join(tail_checkpoint, 'shard_%d' % (jax.process_index()), head_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("set pad_token")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    if model_name == 'gpt2-xl' or model_name == 'gpt2-medium':
        model, params, shard_rules = load_gpt2_model(
            model_str=model_name, 
            from_pretrained=True, 
            checkpoint_path=checkpoint_path, 
            use_fp16=jax.default_backend() == 'tpu', 
            tokenizer=tokenizer, 
            gradient_checkpoint=False, 
            seed=0, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )
    else:
        raise NotImplementedError
    
    # shard params and optimizer
    if do_pjit:
        params, param_spec = shard_params(partial(model.init_weights, input_shape=(1, 1)), 
                                          params, shard_rules, mesh)
    else:
        param_spec = None
    
    inference = load_bc_inference(
        model=model, 
        params=params, 
        param_spec=param_spec, 
        tokenizer=tokenizer, 
        do_pjit=do_pjit, 
        loss_fn=None, 
    )

    policy = JaxAgentPolicy(
        inference=inference, 
        tokenizer=tokenizer, 
        rng=rng, 
        max_input_length=max_sequence_length-max_new_tokens, 
        condition_str=condition_str, 
        postproc_f=lambda x: x+'\n' if not x.endswith('\n') else x, 
        data_parallel_mesh_shape=data_p_shape, 
        do_sample=do_sample, 
        num_beams=1, 
        pad_token_id=tokenizer.pad_token_id, 
        eos_token_id=tokenizer.encode('\n')[0], 
        max_new_tokens=max_new_tokens, 
    )

    return policy


def load_frozen_mc_policy(
    model_name: str, 
    rng: jax.random.KeyArray, 
    mesh: Mesh,
    lm_checkpoint_path: Optional[str]=None, 
    value_checkpoint_path: Optional[str]=None, 
    value_checkpoint_idx: Optional[int]=None, 
    checkpoint_is_sharded: bool=True, 
    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[Any]=None, 
    do_pjit: bool=True, 
    condition_str: Optional[str]=None, 
    beta: float=1.0, 
    max_sequence_length: int=512,
    max_new_tokens: int=80,
    do_sample: bool=False,
):
    print("Loading frozen policy.")
    if mesh is None:
        mesh = contextlib.nullcontext()
        data_p_shape = 1
    else:
        data_p_shape = mesh.devices.shape[0]

    if checkpoint_is_sharded and lm_checkpoint_path is not None:
        tail_checkpoint, head_checkpoint = os.path.split(lm_checkpoint_path.strip('/'))
        lm_checkpoint_path = os.path.join(tail_checkpoint, 'shard_%d' % (jax.process_index()), head_checkpoint)

    if checkpoint_is_sharded and value_checkpoint_path is not None:
        value_checkpoint_path = os.path.join(value_checkpoint_path, 'shard_%d' % (jax.process_index()))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("set pad_token")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    if model_name == 'gpt2-xl':
        print(f"Loading gpt2-xl lm model from {lm_checkpoint_path}")
        model, params, shard_rules = load_gpt2_model(
            model_str=model_name, 
            from_pretrained=True, 
            checkpoint_path=lm_checkpoint_path, 
            use_fp16=jax.default_backend() == 'tpu', 
            tokenizer=tokenizer, 
            gradient_checkpoint=False, 
            seed=0, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )
    else:
        raise NotImplementedError
    
    model_config = model.config
    
    q_load_head_fn = partial(load_advanced_mlp4, inner_dim=model_config.hidden_size*4, 
                             dropout=model_config.resid_pdrop, add_state_term=True, shard_params=True, 
                             gcloud_project=gcloud_project, gcloud_token=gcloud_token)
    v_load_head_fn = partial(load_advanced_mlp4, inner_dim=model_config.hidden_size*4, 
                             dropout=model_config.resid_pdrop, add_state_term=False, shard_params=True, 
                             gcloud_project=gcloud_project, gcloud_token=gcloud_token)
    
    value_path_suffix = f'_{value_checkpoint_idx}' if value_checkpoint_idx is not None else ''
    rng, ilql_state_rng = jax.random.split(rng)
    print(f"Loading value model from {value_checkpoint_path} with suffix {value_path_suffix}.")
    ilql_state = load_frozen_ilql_default_full_inference(
        q_load_head_fn=q_load_head_fn, 
        q_head_rng_keys=frozenset(['dropout']), 
        v_load_head_fn=v_load_head_fn, 
        v_head_rng_keys=frozenset(['dropout']), 
        emb_dim=model_config.hidden_size, 
        vocab_size=model_config.vocab_size, 
        rng=ilql_state_rng, 
        mesh=mesh, 
        do_pjit=do_pjit, 
        q1_checkpoint_path=os.path.join(value_checkpoint_path, f'q1_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
        q2_checkpoint_path=os.path.join(value_checkpoint_path, f'q2_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
        v_checkpoint_path=os.path.join(value_checkpoint_path, f'v_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
        target_q1_checkpoint_path=os.path.join(value_checkpoint_path, f'target_q1_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
        target_q2_checkpoint_path=os.path.join(value_checkpoint_path, f'target_q2_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
    )
    
    # shard params
    if do_pjit:
        params, param_spec = shard_params(partial(model.init_weights, input_shape=(1, 1)), params, shard_rules, mesh)
    else:
        param_spec = None

    base_lm_state = BaseModelState(
        model=model, 
        params=params, 
        param_spec=param_spec, 
    )

    inference = load_frozen_ilql_inference(
        base_lm_state=base_lm_state, 
        q1_head_state=ilql_state.q1_head_model_state, 
        q2_head_state=ilql_state.q2_head_model_state, 
        v_head_state=ilql_state.v_head_model_state, 
        target_q1_head_state=ilql_state.target_q1_head_model_state, 
        target_q2_head_state=ilql_state.target_q2_head_model_state, 
        emb_cacher=None, 
        emb_dim=model_config.hidden_size, 
        tokenizer=tokenizer, 
        do_pjit=do_pjit, 
        loss_fn=None, 
        beta=beta, 
        use_pre_ln_state=False, 
    )

    rng, policy_rng = jax.random.split(rng)
    policy = JaxAgentPolicy(
        inference=inference, 
        tokenizer=tokenizer, 
        rng=policy_rng, 
        max_input_length=max_sequence_length-max_new_tokens, 
        condition_str=condition_str, 
        postproc_f=lambda x: x+'\n' if not x.endswith('\n') else x, 
        data_parallel_mesh_shape=data_p_shape, 
        do_sample=do_sample, 
        num_beams=1, 
        pad_token_id=tokenizer.pad_token_id, 
        eos_token_id=tokenizer.encode('\n')[0], 
        max_new_tokens=max_new_tokens, 
    )

    return policy


def load_ilql_policy(
    model_name: str, 
    rng: jax.random.KeyArray, 
    mesh: Mesh,
    lm_checkpoint_path: Optional[str]=None, 
    value_checkpoint_path: Optional[str]=None, 
    value_checkpoint_idx: Optional[int]=None, 
    checkpoint_is_sharded: bool=True, 
    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[Any]=None, 
    do_pjit: bool=True, 
    condition_str: Optional[str]=None, 
    beta: float=1.0, 
    max_sequence_length: int=512,
    max_new_tokens: int=80,
    do_sample: bool=False,
):
    if gcloud_project is not None and gcloud_token is None:
        gcloud_token = os.path.join(os.path.expanduser('~'), f'.config/gcloud/{gcloud_project}.json')
    
    if mesh is None:
        mesh = contextlib.nullcontext()
        data_p_shape = 1
    else:
        data_p_shape = mesh.devices.shape[0]

    from utils.gcs_manager import open_pp as open
    open = partial(open, gcloud_project=gcloud_project, gcloud_token=gcloud_token)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("set pad_token")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    if checkpoint_is_sharded and lm_checkpoint_path is not None:
        tail_checkpoint, head_checkpoint = os.path.split(lm_checkpoint_path.strip('/'))
        lm_checkpoint_path = os.path.join(tail_checkpoint, 'shard_%d' % (jax.process_index()), head_checkpoint)

    if checkpoint_is_sharded and value_checkpoint_path is not None:
        value_checkpoint_path = os.path.join(value_checkpoint_path, 'shard_%d' % (jax.process_index()))
    
    value_idx_suffix = f'_{value_checkpoint_idx}' if value_checkpoint_idx is not None else ''

    if model_name == 'gpt2-xl':
        print(f"Loading gpt2-xl lm model from {lm_checkpoint_path}")
        model, params, shard_rules = load_gpt2_model(
            model_str=model_name, 
            from_pretrained=True, 
            checkpoint_path=lm_checkpoint_path, 
            use_fp16=jax.default_backend() == 'tpu', 
            tokenizer=tokenizer, 
            gradient_checkpoint=False, 
            seed=0, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )
        print(f"Loading value models from {value_checkpoint_path} with suffix {value_idx_suffix}.")
        value_base = load_gpt2_model(
            model_str=model_name, 
            from_pretrained=True, 
            checkpoint_path=os.path.join(value_checkpoint_path, f'value_base{value_idx_suffix}') if value_checkpoint_path is not None else None,
            use_fp16=jax.default_backend() == 'tpu', 
            tokenizer=tokenizer, 
            gradient_checkpoint=False, 
            seed=2,
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )
        target_value_base = load_gpt2_model(
            model_str=model_name, 
            from_pretrained=True, 
            checkpoint_path=os.path.join(value_checkpoint_path, f'target_value_base{value_idx_suffix}') if value_checkpoint_path is not None else None, 
            use_fp16=jax.default_backend() == 'tpu', 
            tokenizer=tokenizer, 
            gradient_checkpoint=False, 
            seed=2, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )
    
    model_config = model.config
    
    q_load_head_fn = partial(load_advanced_mlp4, inner_dim=model_config.hidden_size*4, 
                             dropout=model_config.resid_pdrop, add_state_term=True, shard_params=True, 
                             gcloud_project=gcloud_project, gcloud_token=gcloud_token)
    v_load_head_fn = partial(load_advanced_mlp4, inner_dim=model_config.hidden_size*4, 
                             dropout=model_config.resid_pdrop, add_state_term=False, shard_params=True, 
                             gcloud_project=gcloud_project, gcloud_token=gcloud_token)
    
    rng, ilql_state_rng = jax.random.split(rng)
    print(f"Loading value head from {value_checkpoint_path} with suffix {value_idx_suffix}.")
    ilql_state = load_ilql_default_full_inference(
        value_base=value_base,
        target_value_base_params=target_value_base.params,
        q_load_head_fn=q_load_head_fn, 
        q_head_rng_keys=frozenset(['dropout']), 
        v_load_head_fn=v_load_head_fn, 
        v_head_rng_keys=frozenset(['dropout']), 
        rng=ilql_state_rng, 
        mesh=mesh, 
        do_pjit=do_pjit, 
        q1_checkpoint_path=os.path.join(value_checkpoint_path, f'q1_head{value_idx_suffix}.pkl') if value_checkpoint_path is not None else None, 
        q2_checkpoint_path=os.path.join(value_checkpoint_path, f'q2_head{value_idx_suffix}.pkl') if value_checkpoint_path is not None else None, 
        v_checkpoint_path=os.path.join(value_checkpoint_path, f'v_head{value_idx_suffix}.pkl') if value_checkpoint_path is not None else None, 
        target_q1_checkpoint_path=os.path.join(value_checkpoint_path, f'target_q1_head{value_idx_suffix}.pkl') if value_checkpoint_path is not None else None, 
        target_q2_checkpoint_path=os.path.join(value_checkpoint_path, f'target_q2_head{value_idx_suffix}.pkl') if value_checkpoint_path is not None else None, 
    )    

    # shard params
    if do_pjit:
        params, param_spec = shard_params(partial(model.init_weights, input_shape=(1, 1)), 
                                                                params, shard_rules, mesh)
    else:
        param_spec = None

    base_lm_state = BaseModelState(
        model=model, 
        params=params, 
        param_spec=param_spec, 
    )

    inference = load_ilql_inference(
        pi_beta_state=base_lm_state, 
        value_base_state=ilql_state.value_base_model_state, 
        q1_head_state=ilql_state.q1_head_model_state, 
        q2_head_state=ilql_state.q2_head_model_state, 
        target_value_base_state=ilql_state.target_value_base_model_state, 
        v_head_state=ilql_state.v_head_model_state, 
        target_q1_head_state=ilql_state.target_q1_head_model_state, 
        target_q2_head_state=ilql_state.target_q2_head_model_state, 
        tokenizer=tokenizer, 
        do_pjit=do_pjit, 
        loss_fn=None, 
        beta=beta, 
        use_pre_ln_state=False, 
    )

    rng, policy_rng = jax.random.split(rng)
    policy = JaxAgentPolicy(
        inference=inference, 
        tokenizer=tokenizer, 
        rng=policy_rng, 
        max_input_length=max_sequence_length-max_new_tokens, 
        condition_str=condition_str, 
        postproc_f=lambda x: x+'\n' if not x.endswith('\n') else x, 
        data_parallel_mesh_shape=data_p_shape, 
        do_sample=do_sample, 
        num_beams=1, 
        pad_token_id=tokenizer.pad_token_id, 
        eos_token_id=tokenizer.encode('\n')[0], 
        max_new_tokens=max_new_tokens, 
    )

    return policy


def main(
    exp_name: str, 
    model_name: str, 
    algorithm_name: str,

    /,  # Mark the end of positional arguments.

    reward_mode: str="fancy",

    checkpoint_path: Optional[str]=None, 
    value_checkpoint_path: Optional[str]=None,
    value_checkpoint_idx: Optional[str]=None,
    checkpoint_is_sharded: bool=True, 

    buyer_algorithm_name: str="bc",
    buyer_model_name: str="gpt2-xl", 
    buyer_checkpoint_path: Optional[str]="gcs://rail-tpus-charles-3/ILQL5/outputs/car_dealer/buyer_bc_gpt2xl_test4/model", 
    buyer_value_checkpoint_path: Optional[str]=None,
    buyer_value_checkpoint_idx: Optional[str]=None,
    buyer_checkpoint_is_sharded: bool=True, 
    buyer_max_sequence_length: int=1024,
    buyer_max_new_tokens: int=128,

    output_path: Optional[str]='outputs/evals/car_dealer', 

    max_sequence_length: int=1024,
    max_new_tokens: int=128,
    do_sample: bool=False,
    beta: float=1.0,

    num_samples: int=100,
    save_every: int=10,
    seed: int=0,

    do_pjit: bool=True, 
    model_p_shape: int=1, 
    data_p_shape: int=1, 

    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[str]=None, 

    verbose: bool=False,
):
    if gcloud_project is not None and gcloud_token is None:
        gcloud_token = os.path.join(os.path.expanduser('~'), f'.config/gcloud/{gcloud_project}.json')

    input_args = locals().copy()
    print(input_args)

    from utils.gcs_manager import open_pp as open
    open = partial(open, gcloud_project=gcloud_project, gcloud_token=gcloud_token)

    # mesh definition
    if do_pjit:
        mesh_devices = np.array(jax.devices()).reshape(data_p_shape, model_p_shape)
        print('using mesh shape:', mesh_devices.shape)
        print('full mesh:', mesh_devices)
        mesh = Mesh(mesh_devices, ("dp", "mp"))
    else:
        mesh = contextlib.nullcontext()

    # rng
    rng = jax.random.PRNGKey(seed)

    # load seller model
    if algorithm_name == "bc":
        rng, policy_rng = jax.random.split(rng)
        policy = load_bc_policy(
            model_name=model_name, 
            rng=policy_rng, 
            mesh=mesh,
            checkpoint_path=checkpoint_path, 
            checkpoint_is_sharded=checkpoint_is_sharded, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
            do_pjit=do_pjit, 
            condition_str=None, 
            max_sequence_length=max_sequence_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    elif algorithm_name == "frozen_mc":
        rng, policy_rng = jax.random.split(rng)
        policy = load_frozen_mc_policy(
            model_name=model_name, 
            rng=policy_rng, 
            mesh=mesh,
            lm_checkpoint_path=checkpoint_path,
            value_checkpoint_path=value_checkpoint_path,
            value_checkpoint_idx=value_checkpoint_idx,
            checkpoint_is_sharded=checkpoint_is_sharded, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
            do_pjit=do_pjit, 
            condition_str=None, 
            beta=beta,
            max_sequence_length=max_sequence_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    elif algorithm_name == "ilql":
        rng, policy_rng = jax.random.split(rng)
        policy = load_ilql_policy(
            model_name=model_name, 
            rng=policy_rng, 
            mesh=mesh,
            lm_checkpoint_path=checkpoint_path,
            value_checkpoint_path=value_checkpoint_path,
            value_checkpoint_idx=value_checkpoint_idx,
            checkpoint_is_sharded=checkpoint_is_sharded, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
            do_pjit=do_pjit, 
            condition_str=None, 
            beta=beta,
            max_sequence_length=max_sequence_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    else:
        raise NotImplementedError
    
    # load buyer model
    if buyer_algorithm_name == "bc":
        rng, buyer_policy_rng = jax.random.split(rng)
        buyer_policy = load_bc_policy(
            model_name=buyer_model_name, 
            rng=buyer_policy_rng, 
            mesh=mesh,
            checkpoint_path=buyer_checkpoint_path, 
            checkpoint_is_sharded=buyer_checkpoint_is_sharded, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
            do_pjit=do_pjit, 
            max_sequence_length=buyer_max_sequence_length,
            max_new_tokens=buyer_max_new_tokens,
            do_sample=True,
        )
    else:
        raise NotImplementedError

    # check output directory
    save_dir = None
    if output_path is not None:
        save_dir = convert_path(os.path.join(output_path, exp_name))
        if (not save_dir.startswith('gcs://')) and (not os.path.exists(save_dir)):
            print(f"Output directory {save_dir} does not exist. Making directory...")
            os.makedirs(save_dir)
        
        # copy script to outputs as a cheap form of config logging
        with open(__file__, 'r') as f_local:
            with open(os.path.join(save_dir, 'config.py'), 'w') as f_save:
                f_save.write(f_local.read())
        with open(os.path.join(save_dir, 'input_args.pkl'), 'wb') as f:
            pkl.dump(input_args, f)

    # create environment
    env = CarDealerPolicyEnvironment(
        buyer=buyer_policy,
        max_conversation_length=50,
        reward_mode=reward_mode,
    )

    # Do rollouts
    with mesh:
        random_seed = seed_generator(seed)
        
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
            transitions = interact_environment(env, policy, env_seed=next(random_seed), env_options={"verbose": verbose})

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


if __name__ == "__main__":
    dcargs.cli(main)