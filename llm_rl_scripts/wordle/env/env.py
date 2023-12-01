from __future__ import annotations
from typing import Dict, Optional, Tuple
from LLM_RL.environment import TextEnv, TextHistory, Text
from llm_rl_scripts.wordle.env.game import Vocabulary, WordleGame
import random

def reformat_history(text_history: TextHistory) -> TextHistory:
    new_text_history = (Text('Wordle:\n', False),)
    for item in text_history:
        if item.is_action:
            new_text_history += (Text(' '.join(list(item.text))+'\n', True),)
        else:
            if len(item.text) == 0:
                new_text_history += (Text('\n', False),)
            else:
                new_text_history += (Text(' '.join(item.text[1:-1].split('><'))+'\n', False),)
    return new_text_history

def deformat_history(text_history: TextHistory) -> TextHistory:
    new_text_history = tuple()
    for item in text_history[1:]:
        if item.is_action:
            new_text_history += (Text(item.text.strip().replace(' ', ''), True),)
        else:
            new_text_history += (Text('<'+'><'.join(list(item.text.strip().replace(' ', '')))+'>', False),)
    return new_text_history

class ReformatWordleEnvironment(TextEnv):
    def __init__(self, env: WordleEnvironment):
        self.env = env
    
    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        text_history, r, done = self.env.step(deformat_history(text_history))
        return reformat_history(text_history), r, done

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> TextHistory:
        return reformat_history(self.env.reset(seed=seed, options=options))

class WordleEnvironment(TextEnv):
    def __init__(self, vocab: Vocabulary, require_words_in_vocab: bool = True, bad_word_reward: float = -1.0):
        self.vocab = vocab
        self.require_words_in_vocab = require_words_in_vocab
        self.bad_word_reward = bad_word_reward
        self.reset()
    
    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        assert text_history[-1].is_action
        self.state, r, t = self.state.next(text_history[-1].text)
        transition = Text(self.state.transition_sequence()[-1], False)
        return text_history+(transition,), r, t
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> TextHistory:
        self.vocab.rng = random.Random(seed)
        self.state = WordleGame.initialize(self.vocab, require_words_in_vocab=self.require_words_in_vocab, bad_word_reward=self.bad_word_reward)
        return tuple()
