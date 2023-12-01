from dataclasses import replace
from typing import Dict, List, Optional, Tuple
import random
from LLM_RL.environment import Text, TextEnv, TextHistory
from .data import INITIAL_STR, WordVariants, create_trajectory_from_history
from .oracle import GuessCityOracle


class GuessCityPolicyEnvironment(TextEnv):
    def __init__(
        self, 
        oracle: GuessCityOracle,
        word_list: List[WordVariants],  
        max_conversation_length: int=20,
    ):
        self.oracle = oracle
        self.word_list = word_list
        self.max_conversation_length = max_conversation_length

        self.random = random.Random(None)

        self.curr_word: Optional[WordVariants] = None

    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        assert text_history[-1].is_action
        assert self.curr_word is not None, "call env.reset() first."

        question = text_history[-1].text.strip()
        answer = self.oracle.generate_answer(self.curr_word, question)
        # print(f"step: question={question}, answer={answer}")
        answer_text = Text(answer + "\n", is_action=False)

        trajectory = create_trajectory_from_history(self.curr_word, text_history + (answer_text,), self.max_conversation_length)

        return trajectory.text_history, trajectory.reward[-2], trajectory.done
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> TextHistory:
        if seed is not None:
            self.random = random.Random(seed)

        deterministic = options.get("deterministic", False)
        if deterministic:
            assert seed is not None, "In deterministic mode, the seed specifies which word to use."
            word_ind = seed % len(self.word_list)
            self.curr_word = self.word_list[word_ind]
        else:
            self.curr_word = self.random.choice(self.word_list)

        # print(f"reset: word={self.curr_word}")
        return (Text(INITIAL_STR, is_action=False),)

    def copy(self):
        return GuessCityPolicyEnvironment(
            oracle=self.oracle,
            word_list=self.word_list,
            max_conversation_length=self.max_conversation_length,
        )

