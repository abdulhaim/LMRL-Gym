from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod
import re
import jax
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig
from JaxSeq.models.T5.interface import T5Inference
from JaxSeq.models.T5.load import load_params as t5_load_params, ModelLoadMode as T5ModelLoadMode
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, convert_path, get_dtype
from .data import INVALID_QUESTION


class GuessCityOracle(ABC):
    @abstractmethod
    def generate_answer(self, word: str, question: str, return_full: bool=False) -> str:
        pass


def get_t5_oracle_prompt(word: str, question: str):
    prompt = (
    f"""Answer the question about the city truthfully.
    object: {word}
    question: {question}
    answer: """
    )
    return prompt


class T5Oracle(GuessCityOracle):
    def __init__(
        self,
        prng_key: Optional[jax.random.PRNGKeyArray],
        inference: T5Inference,
        generation_config: GenerationConfig,
        blocking_strategy: BlockingStrategy,
    ):
        self.prng_key = prng_key
        self.inference = inference
        self.generation_config = generation_config
        self.blocking_strategy = blocking_strategy
        self.answer_re_pattern = re.compile(r"(yes|no)")
        
    def generate_answer(self, word: str, question: str, return_full: bool=False) -> str:
        if question == INVALID_QUESTION:
            if return_full:
                return "No.", "No."
            return "No."

        oracle_prompt = get_t5_oracle_prompt(word, question)

        answers = self.inference.generate_from_str(
            input_strs=[oracle_prompt],
            prng_key=self.prng_key,
            generation_config=self.generation_config, 
            blocking_strategy=self.blocking_strategy, 
        )
        answer_full = answers.output_strs[0].strip().lower()
        answer_match = self.answer_re_pattern.match(answer_full)
        if answer_match is not None:
            answer = answer_match[0].capitalize() + "."
        
        if answer_match is None:
            answer = "No."
        
        if return_full:
            return answer, answer_full
        return answer

    @classmethod
    def load_oracle(
        cls,
        mesh: jax.sharding.Mesh,
        prng_key: Optional[jax.random.PRNGKeyArray], 
        model_load_mode: T5ModelLoadMode,
        model_load_path: str,
        tokenizer_name: str="google/flan-t5-xl",
        use_fp16_activations: bool=True,
        use_fp16_params: bool=True,
        fsdp: bool=False,
        max_input_length: int=124,
        max_output_length: int=4,
    ):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        model_dtype = get_dtype(use_fp16=use_fp16_activations)
        params_dtype = get_dtype(use_fp16=use_fp16_params)

        params, model = t5_load_params(
            model_load_mode=model_load_mode,
            model_load_path=convert_path(model_load_path) if model_load_mode != T5ModelLoadMode.HF else model_load_path,
            model_dtype=model_dtype,
            tokenizer=tokenizer,
            mesh=mesh,
            fsdp=fsdp,
            params_dtype=params_dtype,
        )

        inference = T5Inference.load_inference(
            params=params, 
            model=model, 
            tokenizer=tokenizer, 
        )

        generation_config = GenerationConfig(
            do_sample=False, 
            num_beams=1, 
            temperature=None, 
            top_p=None, 
            top_k=None, 
            eos_token_id=tokenizer.encode('\n')[0], 
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=model.config.decoder_start_token_id,
            max_new_tokens=max_output_length, 
        )
        
        blocking_strategy = BlockingStrategy(
            padding=Padding.LEFT, 
            truncation=Truncation.RIGHT, 
            max_length=max_input_length, 
        )
        
        oracle = cls(
            prng_key=prng_key,
            inference=inference,
            generation_config=generation_config,
            blocking_strategy=blocking_strategy,
        )

        return oracle

