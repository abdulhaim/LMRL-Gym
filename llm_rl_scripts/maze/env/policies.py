from collections import defaultdict
from typing import Union
import random
from typing import List, Optional
from llm_rl_scripts.wordle.env.utils import Cache
from llm_rl_scripts.wordle.env.game import IDX2CHAR, N_CHARS, Vocabulary, WordleGame
import math
from tqdm.auto import tqdm
from LLM_RL.environment import TextPolicy, TextHistory, Text

def text_history_to_str(text_history: TextHistory) -> str:
    return '\n'.join(map(lambda x: x.text, text_history))

class UserPolicy(TextPolicy):
    def __init__(self):
        super().__init__()

    def act(self, text_history: TextHistory) -> TextHistory:
        print(text_history_to_str(text_history))
        
        result = input("Enter your move: ").strip()
        if result == "w":
            result = "move up\n"
        elif result == "d":
            result = "move right\n"
        elif result == "a":
            result = "move left\n"
        elif result == "s":
            result = "move down\n"
        else:
            result = result + "\n"
        return text_history+(Text(result, True),)
