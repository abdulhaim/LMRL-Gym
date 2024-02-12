from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Any, Iterator
import jax
from functools import partial
from typing import List, Optional, Union, Tuple, Callable, NamedTuple, Dict, Any
from LLM_RL.environment import BatchedTextPolicy, TextHistory, text_history_to_str, Text
from JaxSeq.utils import BlockingStrategy, Padding, Truncation
from JaxSeq.utils import strip_prompt_from_completion
from transformers.generation import GenerationConfig
from JaxSeq.models.gpt2.interface import GPT2Inference
from IPython import embed

class BatchedGPT2BuyerPolicy(BatchedTextPolicy):
    def __init__(
        self, 
        inference: GPT2Inference, 
        prng_key: Optional[jax.random.KeyArray], 
        generation_config: Optional[GenerationConfig]=None, 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        in_str_process: Optional[Callable[[str], str]]=None, 
        out_str_process: Optional[Callable[[str], str]]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        trace: bool=True, 
    ):
        self.inference = inference
        self.prng_key = prng_key
        self.generation_config = generation_config
        self.blocking_strategy = blocking_strategy
        self.in_str_process = in_str_process
        self.out_str_process = out_str_process
        self.input_token_process = input_token_process
        self.target_token_process = target_token_process
        if self.in_str_process is None:
            self.in_str_process = lambda x: x
        if self.out_str_process is None:
            self.out_str_process = lambda x: x
        self.trace = trace
    
    def act(self, text_history: List[Optional[TextHistory]], done: Optional[List[bool]]=None) -> List[Optional[TextHistory]]:
        if done is None:
            done = [False]*len(text_history)
        # force eos_token for done sequences
        eos_token = self.inference.tokenizer.eos_token
        if self.generation_config is not None and self.generation_config.eos_token_id is not None:
            eos_token = self.inference.tokenizer.decode(self.generation_config.eos_token_id)
        if eos_token is None:
            eos_token = self.inference.tokenizer.pad_token
        if eos_token is None:
            eos_token = ''

        # check if text_history is not a list of TextHistory
        # if not all(isinstance(item, tuple) for item in text_history):
        #     text_history = [text_history]
        
        # embed()
        raw_input_strs = [
            eos_token if d else self.in_str_process(text_history_to_str(item)) \
                for item, d in zip(text_history, done)
        ]

        new_key = None
        if self.prng_key is not None:
            self.prng_key, new_key = jax.random.split(self.prng_key)
        model_outputs = self.inference.generate_from_str(
            input_strs=raw_input_strs, 
            prng_key=new_key, 
            blocking_strategy=self.blocking_strategy, 
            generation_config=self.generation_config, 
            input_token_process=self.input_token_process, 
            target_token_process=self.target_token_process, 
            trace=self.trace, 
        )

        raw_output_strs = model_outputs.output_strs
        output_strs = [
            "" if d else self.out_str_process(strip_prompt_from_completion(raw_input_str, raw_output_str)) \
                for raw_input_str, raw_output_str, d in zip(raw_input_strs, raw_output_strs, done)
        ]

        return [
            None if d else text_history_item+(Text(output_str, True),) \
                for text_history_item, output_str, d in zip(text_history, output_strs, done)
        ]