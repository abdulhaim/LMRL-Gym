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
    def __init__(self, vocab: Union[str, Vocabulary], hint_policy: Optional[TextPolicy]=None):
        super().__init__()
        self.vocab = vocab
        if isinstance(self.vocab, str):
            self.vocab = Vocabulary.from_file(self.vocab)
        self.hint_policy = hint_policy

    def act(self, text_history: TextHistory) -> TextHistory:
        game = WordleGame.from_str(text_history_to_str(text_history), self.vocab)
        print(game)
        while True:
            if self.hint_policy is not None:
                want_a_hint = input('hint? ')
                if want_a_hint.lower() == 'y' or want_a_hint.lower() == 'yes':
                    result = self.hint_policy.act(text_history)
                    print()
                    return result
            result = input("Enter a word: ")
            if len(result) != N_CHARS:
                print(f"Please enter a {N_CHARS} letter word.")
            elif (self.vocab is not None) and (result not in self.vocab.all_vocab):
                print('Not a word. Try again.')
            else:
                break
        print()
        return text_history+(Text(result, True),)

class StartWordPolicy(TextPolicy):
    def __init__(self, start_words: Optional[List[str]]=None):
        super().__init__()
        self.start_words = start_words
        if self.start_words is None:
            # "tales" is the optimal word under 10k_words.txt
            # "raise" is the optimal word under wordle_official.txt
            self.start_words = ['opera', 'tears', 'soare', 'roate', 'raise', 'arose', 
                                'earls', 'laser', 'reals', 'aloes', 'reais', 'slate', 
                                'sauce', 'slice', 'shale', 'saute', 'share', 'sooty', 
                                'shine', 'suite', 'crane', 'adieu', 'audio', 'stare', 
                                'roast', 'ratio', 'arise', 'tales']
        self.vocab = Vocabulary(self.start_words, None)
    
    def random_str(self):
        return ''.join([random.choice(IDX2CHAR) for _ in range(N_CHARS)])
    
    def act(self, text_history: TextHistory) -> TextHistory:
        game = WordleGame.from_str(text_history_to_str(text_history), self.vocab)
        filtered_start_words = list(filter(lambda x: x in game.vocab.filtered_vocab, self.start_words))
        if len(filtered_start_words) == 0:
            return text_history+(Text(self.random_str(), True),)
        return text_history+(Text(random.choice(filtered_start_words), True),)

class OptimalPolicy(TextPolicy):
    def __init__(self, vocab: Union[str, Vocabulary], start_word_policy: Optional[TextPolicy]=None, progress_bar: bool=False):
        super().__init__()
        self.vocab = vocab
        if isinstance(self.vocab, str):
            self.vocab = Vocabulary.from_file(self.vocab)
        self.start_word_policy = start_word_policy
        self.progress_bar = progress_bar
        self.cache = Cache()

    def act(self, text_history: TextHistory) -> TextHistory:
        game = WordleGame.from_str(text_history_to_str(text_history), self.vocab)
        if game.state in self.cache:
            return text_history+(Text(random.choice(self.cache[game.state]), True),)
        if len(game.action_history) == 0 and self.start_word_policy is not None:
            return self.start_word_policy.act(text_history)
        best_words = []
        best_info = float('-inf')
        for word in tqdm(game.vocab.filtered_vocab, disable=not self.progress_bar):
            total_entropy = 0.0
            total = 0
            for next_state, state_count in game.all_next(word):
                total_entropy += math.log(next_state.vocab.filtered_vocab_size()) * state_count
                total += state_count
            info_gain = math.log(game.vocab.filtered_vocab_size()) - (total_entropy / total)
            if info_gain > best_info:
                best_words, best_info = [word], info_gain
            elif info_gain == best_info:
                best_words.append(word)
        self.cache[game.state] = best_words
        return text_history+(Text(random.choice(best_words), True),)

class RepeatPolicy(TextPolicy):
    def __init__(self, start_word_policy: Optional[TextPolicy], first_n: Optional[int]):
        super().__init__()
        self.first_n = first_n
        self.start_word_policy = start_word_policy
    
    def act(self, text_history: TextHistory) -> TextHistory:
        game = WordleGame.from_str(text_history_to_str(text_history), None)
        if len(game.action_history) == 0:
            if self.start_word_policy is not None:
                return self.start_word_policy.act(text_history)
            return text_history+(Text(game.vocab.get_random_word_all(), True),)
        if self.first_n is None:
            return text_history+(Text(random.choice(game.action_history), True),)
        return text_history+(Text(random.choice(game.action_history[:self.first_n]), True),)

class RandomMixturePolicy(TextPolicy):
    def __init__(self, prob_smart: float, vocab: Union[str, Vocabulary]):
        super().__init__()
        self.vocab = vocab
        if isinstance(self.vocab, str):
            self.vocab = Vocabulary.from_file(self.vocab)
        self.prob_smart = prob_smart

    def act(self, text_history: TextHistory) -> TextHistory:
        game = WordleGame.from_str(text_history_to_str(text_history), self.vocab)
        if random.random() < self.prob_smart:
            return text_history+(Text(game.vocab.get_random_word_filtered(), True),)
        return text_history+(Text(game.vocab.get_random_word_all(), True),)

class WrongPolicy(TextPolicy):
    def __init__(self, vocab: Union[str, Vocabulary]):
        super().__init__()
        self.vocab = vocab
        if isinstance(self.vocab, str):
            self.vocab = Vocabulary.from_file(self.vocab)
        self.choices = set(self.vocab.all_vocab)

    def act(self, text_history: TextHistory) -> TextHistory:
        game = WordleGame.from_str(text_history_to_str(text_history), self.vocab)
        bad_options = self.choices.difference(game.vocab.filtered_vocab)
        if len(bad_options) == 0:
            return text_history+(Text(game.vocab.get_random_word_all(), True),)
        return text_history+(Text(random.sample(bad_options, 1)[0], True),)

class MixturePolicy(TextPolicy):
    def __init__(self, prob1: float, policy1: TextPolicy, policy2: TextPolicy):
        super().__init__()
        self.prob1 = prob1
        self.policy1 = policy1
        self.policy2 = policy2
    
    def act(self, text_history: TextHistory) -> TextHistory:
        if random.random() < self.prob1:
            return self.policy1.act(text_history)
        return self.policy2.act(text_history)

class MonteCarloPolicy(TextPolicy):
    def __init__(self, n_samples: int, sample_policy: TextPolicy, vocab: Union[str, Vocabulary]):
        super().__init__()
        self.n_samples = n_samples
        self.sample_policy = sample_policy
        self.vocab = vocab
        if isinstance(self.vocab, str):
            self.vocab = Vocabulary.from_file(self.vocab)
    
    def act(self, text_history: TextHistory) -> TextHistory:
        action_scores = defaultdict(list)
        for _ in range(self.n_samples):
            game = WordleGame.from_str(text_history_to_str(text_history), self.vocab)
            total_reward = 0
            while not game.is_terminal():
                word_choice = self.sample_policy.act(curr_obs)
                curr_obs, r, _ = game.next(word_choice)
                total_reward += r
            action_scores[curr_obs.action_history[len(game.action_history)]].append(total_reward)
        return text_history+(Text(max(action_scores.items(), key=lambda x: sum(x[1]) / len(x[1]))[0], True),)
