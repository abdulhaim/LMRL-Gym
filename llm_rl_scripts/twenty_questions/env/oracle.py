from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import re
import jax
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig
from JaxSeq.models.T5.interface import T5Inference
from JaxSeq.models.T5.load import load_params as t5_load_params, ModelLoadMode as T5ModelLoadMode
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, convert_path, get_dtype
from llm_rl_scripts.twenty_questions.env.data import INVALID_QUESTION, WordVariants


class TwentyQuestionsOracle(ABC):
    @abstractmethod
    def generate_answers(self, words: Union[WordVariants, List[WordVariants]], questions: Union[str, List[str]], return_full: bool=False) -> Union[str, List[str]]:
        pass


def get_t5_oracle_prompt(word: WordVariants, question: str):
    prompt = (
f"""Answer the question about the object truthfully.
object: {word}
question: {question}
answer (yes or no): """
    )
    return prompt


class T5Oracle(TwentyQuestionsOracle):
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
        
    def generate_answers(self, words: Union[WordVariants, List[WordVariants]], questions: Union[str, List[str]], return_full: bool=False) -> Union[str, List[str]]:
        input_is_list = True
        if not isinstance(words, list):
            assert not isinstance(questions, list)
            words = [words]
            questions = [questions]
            input_is_list = False

        assert len(words) == len(questions)

        oracle_prompts = [get_t5_oracle_prompt(word, question) for word, question in zip(words, questions)]

        inference_result = self.inference.generate_from_str(
            input_strs=oracle_prompts,
            prng_key=self.prng_key,
            generation_config=self.generation_config, 
            blocking_strategy=self.blocking_strategy, 
        )
        answers = []
        answers_full = []
        for question, output_str in zip(questions, inference_result.output_strs):
            if question == INVALID_QUESTION:
                answers.append("No.")
                answers_full.append("No.")
                continue

            answer_full = output_str.strip().lower()
            answer_match = self.answer_re_pattern.match(answer_full)
            if answer_match is not None:
                answer = answer_match[0].capitalize() + "."
            
            if answer_match is None:
                answer = "No."
            
            answers.append(answer)
            answers_full.append(answer_full)

        if not input_is_list:
            answers = answers[0]
            answers_full = answers_full[0]

        if return_full:
            return answers, answers_full
        return answers

    @classmethod
    def load_oracle(
        cls,
        mesh: jax.sharding.Mesh,
        prng_key: Optional[jax.random.PRNGKeyArray], 
        model_load_mode: T5ModelLoadMode,
        model_load_path: str,
        tokenizer_name: str="google/flan-t5-xl",
        use_fp16_activations: bool=False,
        use_fp16_params: bool=False,
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
