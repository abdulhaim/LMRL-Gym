import jax
from typing import Optional
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import jsonl_stream, convert_path, load_mesh, setup_experiment_save
from JaxSeq.models.gpt2.interface import GPT2Inference
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, get_weight_decay_mask, MapIterable, FileOpenIterable
from JaxSeq.models.gpt2.interface import GPT2TrainMask, GPT2InferenceMask
from JaxSeq.models.gpt2.load import load_train_state, ModelLoadMode, load_params
from JaxSeq.data import MaskDataset, MaskIterableDataset
from JaxSeq.train import eval_loss, train_loop
from JaxSeq.optimizers import GPT3Optimizer
from transformers import AutoTokenizer
import jax.numpy as jnp
import os
import optax
import pickle as pkl
from transformers.generation import GenerationConfig
from jaxtyping import PyTree
import re
from LLM_RL.algorithms.ppo.gpt2.interface import GPT2PPOPolicy
from LLM_RL.environment import text_history_to_str, text_env_eval
import json
from llm_rl_scripts.car_dealer.env.buyer import BatchedGPT2BuyerPolicy
from llm_rl_scripts.car_dealer.env.env import CarDealerPolicyEnvironment, BatchedCarDealerPolicyEnvironment
from llm_rl_scripts.car_dealer.env.data import Role, create_trajectories_from_conversations
from IPython import embed
import nltk


def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 
    train_data_path: str, 
    eval_data_path: str,
    oracle_model_path: str,
    /,  # Mark the end of positional arguments.

    exp_name: Optional[str]=None, 
    outputs_path: Optional[str]=None, 

    data_mesh_shape: int=1, 
    fsdp_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 

    use_wandb: bool=False, 
    wandb_project: Optional[str]=None, 

    epochs: int=1, 
    max_steps: Optional[int]=None, 

    weight_decay: float=0.001, 
    init_lr: float=0.0001, 
    end_lr: float=0.0001, 
    lr: float=0.0001, 
    lr_warmup_steps: int=1000, 
    lr_decay_steps: int=1001, # no decay, so just needs to be > warmup steps
    bf16_momentum: bool=False, 
    multiply_by_parameter_scale: bool=True, 

    resid_pdrop: float=0.05, 
    attn_pdrop: float=0.05, 
    embd_pdrop: float=0.05, 

    train_bsize: int=4, 
    grad_accum_steps: Optional[int]=32, 

    gradient_checkpointing: bool=False, 
    gradient_checkpointing_policy: str='nothing_saveable', 

    bf16_activations: bool=False, 

    max_length: int=1024, 

    log_every: int=256, 
    eval_every_steps: Optional[int]=None, 
    eval_every_epochs: Optional[int]=1, 
    eval_at_beginning: bool=False, 
    eval_at_end: bool=True, 
    
    save_every_steps: Optional[int]=None, 
    save_every_epochs: Optional[int]=None, 
    save_at_beginning: bool=False, 
    save_at_end: bool=False, 
    save_best: bool=True, 
    max_checkpoints: Optional[int]=None, 
    save_train_state: bool=True, 
    save_bf16: bool=True, 

    eval_loss_bsize: int=4, 
    eval_loss_batches: int=4, 

    policy_n_rollouts: int=32, 
    policy_bsize: int=1, 
    policy_max_input_length: int=256, 
    policy_max_output_length: int=128, 
    policy_do_sample: bool=True, 
    policy_num_beams: int=1, 
    policy_temperature: Optional[float]=None, 
    policy_top_p: Optional[float]=None, 
    policy_top_k: Optional[int]=None, 

    force_pad_embeddings: bool=False, 

    should_restore_loop_state: bool=False, 
):
    
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    input_args = dict(locals())

    print(input_args)
    print(type(input_args))
    # embed()

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")

    # load data
    with open(convert_path(train_data_path), 'r') as f:
        raw_train = json.load(f)
    with open(convert_path(eval_data_path), 'r') as f:
        raw_eval = json.load(f)

    train_text_trajectories = create_trajectories_from_conversations(raw_train, role=Role.SELLER)
    eval_text_trajectories = create_trajectories_from_conversations(raw_eval, role=Role.SELLER)

    def convert_trajectory_to_masked_text(trajectories):
        for trajectory in trajectories:
            text_history = trajectory.text_history
            lst = []
            for text in text_history:
                item = (text.text, text.is_action)
                lst.append(item)
            yield lst
    # generator = convert_trajectory_to_masked_text(train_text_trajectories)
    # text_history = next(generator)
    # print(text_history)
    # for text in text_history:
    #     encoded = tokenizer.encode(text[0])
    #     total_tokens += len(encoded)
    # print(total_tokens)
    # embed()
    
    # train_text_histories = [convert_trajectory_to_masked_text(text_trajectory) for text_trajectory in train_text_trajectories]
    # eval_text_histories = [convert_trajectory_to_masked_text(text_trajectory) for text_trajectory in eval_text_trajectories]

    train_data = MaskIterableDataset.blocked_from_str_segments_iterable(
        convert_trajectory_to_masked_text(train_text_trajectories), 
        tokenizer, 
        blocking_strategy=BlockingStrategy(
            padding=Padding.RIGHT, 
            truncation=Truncation.LEFT, 
            max_length=max_length, 
        ), 
    )

    eval_data = MaskIterableDataset.blocked_from_str_segments_iterable(
        convert_trajectory_to_masked_text(eval_text_trajectories), 
        tokenizer, 
        blocking_strategy=BlockingStrategy(
            padding=Padding.RIGHT, 
            truncation=Truncation.LEFT, 
            max_length=max_length, 
        ), 
    )

    prng_key = jax.random.PRNGKey(3)
    prng_key, oracle_inference_prng, buyer_policy_prng = jax.random.split(prng_key, 3)
    buyer_params, buyer_model = load_params(
        model_load_mode=ModelLoadMode.PARAMS, 
        model_load_path=convert_path(oracle_model_path) if model_load_mode != ModelLoadMode.HF else oracle_model_path, 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=oracle_inference_prng, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )

    buyer_inference = GPT2Inference.load_inference(
        params=buyer_params,
        model=buyer_model,
        tokenizer=tokenizer,
    )

    buyer_policy = BatchedGPT2BuyerPolicy(
        inference=buyer_inference, 
        prng_key=buyer_policy_prng, 
        generation_config=GenerationConfig(
            do_sample=policy_do_sample, 
            num_beams=policy_num_beams, 
            temperature=policy_temperature, 
            top_p=policy_top_p, 
            top_k=policy_top_k, 
            eos_token_id=tokenizer.encode('\n')[0], 
            pad_token_id=tokenizer.pad_token_id, 
            max_new_tokens=policy_max_output_length, 
        ), 
        blocking_strategy=BlockingStrategy(
            padding=Padding.LEFT, 
            truncation=Truncation.LEFT, 
            max_length=policy_max_input_length, 
        ), 
        out_str_process=lambda x: x.removesuffix('\n')+'\n', 
    )

    model_prng_key = jax.random.PRNGKey(2)
    policy_prng, oracle_prng = jax.random.split(model_prng_key)

    env = BatchedCarDealerPolicyEnvironment(
        buyer=buyer_policy,
        buyer_bsize=policy_bsize, 
    )

    def optim_getter(params: PyTree):
        mask = get_weight_decay_mask((
            "".join([r"\['ln_[0-9]+'\]", re.escape("['bias']")]), 
            "".join([r"\['ln_[0-9]+'\]", re.escape("['scale']")]), 
            re.escape("['ln_f']['bias']"), 
            re.escape("['ln_f']['scale']"), 
            "bias", 
        ))(params)
        
        optimizer_config = GPT3Optimizer(
            init_lr=init_lr, 
            end_lr=end_lr, 
            lr=lr, 
            lr_warmup_steps=lr_warmup_steps, 
            lr_decay_steps=lr_decay_steps, 
            weight_decay=weight_decay, 
            bf16_momentum=bf16_momentum, 
            multiply_by_parameter_scale=multiply_by_parameter_scale, 
        )

        optim, _ = optimizer_config.get_optim(mask)

        if grad_accum_steps is not None:
            return optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)
        return optim

    train_state, model = load_train_state(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(model_load_path) if model_load_mode != ModelLoadMode.HF else model_load_path, 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        optim_getter=optim_getter, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=model_prng_key, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )
    model.config.gradient_checkpointing = gradient_checkpointing
    model.config.gradient_checkpointing_policy = gradient_checkpointing_policy
    model.config.resid_pdrop = resid_pdrop
    model.config.embd_pdrop = embd_pdrop
    model.config.attn_pdrop = attn_pdrop

    loop_state = dict()
    if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE, 
                                                          ModelLoadMode.TRAIN_STATE_PARAMS, 
                                                          ModelLoadMode.PARAMS}):
        with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
            loop_state = pkl.load(f)

    trainer = GPT2TrainMask.load_train(
        train_state=train_state, 
        model=model, 
        tokenizer=tokenizer, 
    )

    inference = GPT2InferenceMask.load_inference(
        params=train_state.params, 
        model=model, 
        tokenizer=tokenizer, 
    )

    save_dir, exp_name = setup_experiment_save(
        exp_name=exp_name, 
        outputs_path=convert_path(outputs_path), 
        input_args=input_args, 
        script__file__=__file__, 
        is_main_process=is_main_process, 
    )

    policy_prng = jax.random.PRNGKey(0)
    def evaluator(inference: GPT2InferenceMask):
        nonlocal policy_prng
        policy_prng, new_key = jax.random.split(policy_prng)
        policy = GPT2PPOPolicy(
            inference=inference, 
            prng_key=new_key, 
            generation_config=GenerationConfig(
                do_sample=policy_do_sample, 
                num_beams=policy_num_beams, 
                temperature=policy_temperature, 
                top_p=policy_top_p, 
                top_k=policy_top_k, 
                eos_token_id=tokenizer.encode('\n')[0], 
                pad_token_id=tokenizer.pad_token_id, 
                max_new_tokens=policy_max_output_length, 
            ), 
            blocking_strategy=BlockingStrategy(
                padding=Padding.LEFT, 
                truncation=Truncation.LEFT, 
                max_length=policy_max_input_length, 
            ), 
            out_str_process=lambda x: x.removesuffix('\n')+'\n', 
        )

        loss_metrics = eval_loss(
            inference=inference, 
            dataset=eval_data, 
            prng_key=None, 
            bsize=eval_loss_bsize, 
            eval_batches=eval_loss_batches, 
        )

        interation_raw_results, interaction_summary_results = text_env_eval(
            env=env, 
            policy=policy, 
            n_rollouts=policy_n_rollouts, 
            bsize=policy_bsize, 
        )

        for item in interation_raw_results:
            print('='*25)
            print(text_history_to_str(item[-1].post_transition_history))
            print('='*25)

        return loss_metrics['loss'], {'loss_metrics': loss_metrics, 'generation_metrics': interaction_summary_results}

    train_prng = jax.random.PRNGKey(1)
    save_dtype = jnp.bfloat16 if save_bf16 else jnp.float32
    trainer, inference = train_loop(
        trainer=trainer, 
        inference=inference, 
        evaluator=evaluator, 
        dataset=train_data, 
        prng_key=train_prng, 
        save_dir=save_dir, 
        epochs=epochs, 
        max_steps=max_steps, 
        bsize=train_bsize, 
        log_every=log_every, 
        eval_every_steps=eval_every_steps, 
        eval_every_epochs=eval_every_epochs, 
        eval_at_beginning=eval_at_beginning, 
        eval_at_end=eval_at_end, 
        save_every_steps=save_every_steps, 
        save_every_epochs=save_every_epochs, 
        save_at_beginning=save_at_beginning, 
        save_at_end=save_at_end, 
        save_best=save_best, 
        max_checkpoints=max_checkpoints, 
        save_train_state=save_train_state, 
        save_dtype=save_dtype, 
        use_wandb=use_wandb, 
        wandb_project=wandb_project, 
        wandb_run_name=exp_name, 
        wandb_config=None, 
        is_main_process=is_main_process, 
        **loop_state, 
    )
if __name__ == "__main__":
    tyro.cli(main)