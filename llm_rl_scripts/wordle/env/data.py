from typing import List, Tuple, Dict, Optional
import random
from llm_rl_scripts.wordle.env.env import WordleEnvironment
from llm_rl_scripts.wordle.env.game import WordleGame, WordleState, Vocabulary
from LLM_RL.environment import TextPolicy, interact_environment, Text
from LLM_RL.environment import TextTrajectory
import random

class PolicyDataGenerator:
    def __init__(
        self, 
        env: WordleEnvironment, 
        policy: TextPolicy, 
        seed: Optional[int]=None, 
    ):
        self.env = env
        self.policy = policy
        self.rng = random.Random(seed)
    
    def __iter__(self):
        return self
    
    def __next__(self) -> TextTrajectory:
        transitions = interact_environment(
            self.env, 
            self.policy, 
            env_seed=self.rng.randint(0, 2**31-1), 
        )[0]
        
        history = transitions[-1].post_transition_history
        rewards = sum([[transition.reward, 0.0] for transition in transitions], [])
        done = transitions[-1].done

        return TextTrajectory(history, rewards, done)

class HumanDataGenerator:
    def __init__(
        self, 
        games: List[Tuple[str, List[str]]], 
        transitions: Dict[str, Dict[str, List[str]]], 
        use_true_word: bool, 
        seed: Optional[int]=None, 
    ):
        self.games = games
        self.transitions = transitions
        self.use_true_word = use_true_word
        self.rng = random.Random(seed)
    
    def __iter__(self):
        return self
    
    def __next__(self) -> TextTrajectory:
        while True:
            true_word, game = self.rng.choice(self.games)
            if self.use_true_word:
                while True:
                    actions = []
                    for transition in game:
                        if transition not in self.transitions[true_word] or len(self.transitions[true_word][transition]) == 0:
                            break
                        actions.append(self.rng.choice(self.transitions[true_word][transition]))
                    if len(actions) == len(game):
                        break
                    else:
                        true_word, game = self.rng.choice(self.games)
            else:
                word_choices = list(self.transitions.keys())
                while True:
                    true_word = self.rng.choice(word_choices)
                    actions = []
                    for transition in game:
                        if transition not in self.transitions[true_word] or len(self.transitions[true_word][transition]) == 0:
                            break
                        actions.append(self.rng.choice(self.transitions[true_word][transition]))
                    if len(actions) == len(game):
                        break
                    else:
                        true_word, game = self.rng.choice(self.games)
            history, rewards, done = [], [], False
            state = WordleState.initial_state()
            vocab = Vocabulary([true_word], state, cache=None, fill_cache=False)
            for i, action in enumerate(actions):
                state = state.transition_state(action, true_word)
                game = WordleGame(state, vocab, actions[:(i+1)])
                history.append(Text(action, is_action=True))
                history.append(Text(game.transition_sequence()[-1], is_action=False))
                rewards.extend([game.reward(), 0.0])
                done = game.is_terminal()
            yield TextTrajectory(history, rewards, done)
