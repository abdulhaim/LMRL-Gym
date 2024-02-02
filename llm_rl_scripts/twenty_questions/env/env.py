from dataclasses import replace
from typing import Dict, List, Optional, Tuple
import random
from LLM_RL.environment import Text, TextEnv, BatchedTextEnv, TextHistory, TextPolicy, StepResult
from .data import INVALID_QUESTION, INITIAL_STR, WordVariants, create_trajectory_from_history, rtg_to_token_str, token_str_to_rtg
from .oracle import TwentyQuestionsOracle


class TwentyQuestionsPolicyEnvironment(TextEnv):
    def __init__(
        self, 
        oracle: TwentyQuestionsOracle,
        word_list: List[WordVariants],  
        max_conversation_length: int=20,
    ):
        self.oracle = oracle
        self.word_list = word_list
        self.max_conversation_length = max_conversation_length

        self.random = random.Random(None)
        self.count = 0
        self.curr_word: Optional[WordVariants] = None

    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        assert text_history[-1].is_action
        assert self.curr_word is not None, "call env.reset() first."
        self.count+=1
        question = text_history[-1].text.strip()
        answer = self.oracle.generate_answers(self.curr_word, question)
        answer_text = Text(answer + "\n", is_action=False)

        trajectory = create_trajectory_from_history(self.curr_word, text_history + (answer_text,), self.max_conversation_length)
        if self.count == self.max_conversation_length:
            print("The word was", self.curr_word[0])
        return trajectory.text_history, trajectory.reward[-2], trajectory.done
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> TextHistory:
        self.count = 0 
        if self.curr_word is not None: 
            print("The word was ", self.curr_word)
            print("Next word...")
        if seed is not None:
            self.random = random.Random(seed)

        if options is None:
            options = {}
        deterministic = options.get("deterministic", False)

        if deterministic:
            assert seed is not None, "In deterministic mode, the seed specifies which word to use."
            word_ind = seed % len(self.word_list)
            self.curr_word = self.word_list[word_ind]
        else:
            self.curr_word = self.random.choice(self.word_list)

        return (Text(INITIAL_STR, is_action=False),)

    def copy(self):
        return TwentyQuestionsPolicyEnvironment(
            oracle=self.oracle,
            word_list=self.word_list,
            max_conversation_length=self.max_conversation_length,
        )


class BatchedTwentyQuestionsPolicyEnvironment(BatchedTextEnv):
    def __init__(
        self, 
        oracle: TwentyQuestionsOracle,
        word_list: List[WordVariants],  
        max_conversation_length: int=20,
        bsize: Optional[int]=None,
    ):
        self.bsize = bsize
        self.oracle = oracle
        self.word_list = word_list
        self.max_conversation_length = max_conversation_length

        self.randoms = [random.Random(None) for b in range(bsize)]
        self.curr_words: Optional[List[WordVariants]] = None

    def step(self,  text_history_batch: List[Optional[TextHistory]], done_batch: Optional[List[bool]]=None) -> List[Optional[StepResult]]:
        assert self.curr_words is not None, "call env.reset() first."

        if self.bsize is None:
            self.bsize = len(text_history_batch)

        npad = self.bsize - len(text_history_batch)

        questions = [text_history[-1].text.strip() if text_history is not None else INVALID_QUESTION for text_history in text_history_batch]
        answers = self.oracle.generate_answers(self.curr_words + [self.word_list[0]]*npad, questions + [INVALID_QUESTION]*npad)[:self.bsize-npad]

        step_results = []
        for answer, curr_word, text_history in zip(answers, self.curr_words, text_history_batch):
            if text_history is None:
                step_results.append(None)
                continue

            answer_text = Text(answer + "\n", is_action=False)
            trajectory = create_trajectory_from_history(curr_word, text_history + (answer_text,), self.max_conversation_length)

            step_result = (trajectory.text_history, trajectory.reward[-2], trajectory.done)
            step_results.append(step_result)
        
        return step_results

    def reset(self, seed_batch: Optional[List[Optional[int]]]=None, options_batch: Optional[List[Optional[Dict]]]=None) -> TextHistory:
        # No padding for reset
        if seed_batch is None:
            seed_batch = [None for _ in range(self.bsize)]
        
        if options_batch is None:
            options_batch = [{} for _ in range(self.bsize)]

        self.randoms: List[random.Random] = []
        self.curr_words = []
        
        initial_text_history_batch = []

        for i, (seed, options) in enumerate(zip(seed_batch, options_batch)):
            self.randoms.append(random.Random(seed))

            deterministic = options.get("deterministic", False)
            if deterministic:
                assert seed is not None, "In deterministic mode, the seed specifies which word to use."
                word_ind = seed % len(self.word_list)
                self.curr_words.append(self.word_list[word_ind])
            else:
                self.curr_words.append(self.randoms[i].choice(self.word_list))

            initial_text_history_batch.append((Text(INITIAL_STR, is_action=False),))

        return initial_text_history_batch

    def copy(self):
        return BatchedTwentyQuestionsPolicyEnvironment(
            oracle=self.oracle,
            word_list=self.word_list,
            max_conversation_length=self.max_conversation_length,
            bsize=self.bsize,
        )

