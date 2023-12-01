import chess
from stockfish import Stockfish
from tqdm.auto import tqdm
from LLM_RL.utils import convert_path
import os
import numpy as np
from LLM_RL.environment import BatchedTextPolicy, TextEnv, Text, TextHistory, TextPolicy, interact_environment
from typing import Callable, Dict, List, Optional, Tuple, Union, Iterator
import chess

CHESS_ENGINE_PATH = os.environ.get("CHESS_ENGINE_PATH", convert_path("stockfish/stockfish-ubuntu-20.04-x86-64-avx2"))

def preprocess_move(move: str):
    return " ".join(move) + "\n"

def postprocess_move(move: str):
    return move.replace(" ", "").strip()

def preprocess_state(state: str):
    return " ".join(state) + "\n"

def preprocess_state_og(state: str):
    return " ".join(state)

def postprocess_state(state: str):
    return state.replace("  ", "__temp__").replace(" ", "").replace("__temp__", " ").strip()

class ChessEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self, side="w", fen=True, from_position=None, stockfish_level=3, stockfish_elo=1200, random_opponent=False):
        """Initialize the underlying chess environment.

        Args:
            side (str, optional): String representing which side the agent is playing. Defaults to "w". (for white pieces)
            fen (bool, optional): Whether or not to play with "fen" states or "move history" states. Defaults to True.
            from_position (_type_, optional): Fen string describing the position to start from. Defaults to None.
        """
        if from_position is None:
            self.board = chess.Board() 
            self.starting_position = self.board.fen()
        else:
            self.board = chess.Board(fen=from_position)
            self.starting_position = from_position

        self.fen = fen

        # set the side of the board which this agent is playing
        if side == "w":
            self.side = chess.WHITE
        else:
            self.side = chess.BLACK
        
        self.stockfish_params = {"Threads": 1, "UCI_Elo": stockfish_elo}
        # print(self.stockfish_params)
        self.stockfish = Stockfish(path=CHESS_ENGINE_PATH, parameters=self.stockfish_params)
        self.stockfish.set_fen_position(self.starting_position)
        self.random_opponent = random_opponent
        self.prev_moves = []
    
    def reset(self):
        self.board = chess.Board(fen=self.starting_position)
        # self.move_history = ""
        # starting_position = self.board.fen()

        self.stockfish.set_fen_position(self.starting_position)
        self.prev_moves = []
        return self.starting_position, {}

    def sample_valid_action(self):
        """Sample a random legal move in san notation."""
        legal_move_generator = self.board.legal_moves
        valid_actions : list = list(self.board.legal_moves)
        move = np.random.choice(valid_actions, 1)[0]
        print(move)
        return self.board.san(move)
    
    # def _fake_step(self, text_history: TextHistory) -> TextHistory:
    #     assert text_history[-1].is_action
    #     action = text_history[-1].text
    #     with seed_context(self.random_state):
    #         st, reward, done, opp_mv = self._step(action) # my previous chess environment step function
    #         if self.fen:
    #             new_state = Text(st, False)
    #             # want to have a text trajectory which links to the next text trajectory
    #             # this is more like the state-ful RL policy
    #         else:
    #             new_state = opp_mv["opponent move"]
    #     return text_history+new_state, reward, done

    def step(self, action: str, opponent_move: bool = True):
        """
        Reward is 0 for legal moves, -1 for illegal moves and defeat, 
        1 for victory and 0.5 for stalemate.

        action: a move in san format for the chess environment.
        opponent_move: a boolean which controls whether or not to fetch an opponent move
        from the engine 

        returns: state, reward, done, {}
        state: either in FEN notation or the move history depending on the status of 
        the flag self.fen
        reward: 0 for non-terminal states and draws, 1 for victory, -1 for illegal moves and loss
        """
        # move = self.board.parse_san(action) # throws an error if not a legal move
        self.prev_moves.append(action)
        assert self.board.turn == chess.WHITE
        reward, done = -1, 0
        opponent = None
        try: 
            move : chess.Move = self.board.push_san(action)
            if move == chess.Move.null():
                self.board.pop()
                return self.board.fen(), -1, True, {}
        except:
            assert self.board.turn == chess.WHITE
            pass
        
        else:
            assert self.board.turn == chess.BLACK
            self.stockfish.make_moves_from_current_position([move.uci()])
            if self.board.is_game_over():
                reward = 1 if self.board.is_checkmate() else 0 # it's agent's turn so the agent wins
                done = 1
                opponent = None
            elif opponent_move:
                assert self.board.turn == chess.BLACK
                if self.random_opponent:
                    opponent = self._make_random_move()
                else:
                    opponent = self._make_engine_move()
                assert self.board.turn == chess.WHITE
                reward = -1 if self.board.is_checkmate() else 0
                done = self.board.is_game_over()
            else: # when the game isn't over, but no opponent move needs to be made?
                print(f"{move.uci()} {self.board.fen()}")
                reward, done, opponent = 0, 0, None
        finally:
            state = self._get_state()
            return state, reward, done, {"opponent move": opponent}

    def _get_state(self):
        """
        Return either the FEN state or the move history depending on 
        the initial parameters of the environment.
        """
        if self.fen:
            return self.board.fen()
    
    def get_board(self):
        """
        Boiler plate getter function for getting access to board which we are
        playing on.
        """
        return self.board

    def _make_engine_move(self):
        """
        return: a move in san notation representing the move the opponent made.
        """
        assert self.board.turn == chess.BLACK
        move : str = self.stockfish.get_best_move_time(100)
        self.stockfish.make_moves_from_current_position([move])
        ch_move : chess.Move = chess.Move.from_uci(move)
        san_move = self.board.san(ch_move)
        
        # self._update_move_history(self.board.san(ch_move))
        self.board.push(ch_move)
        assert self.board.turn == chess.WHITE
        return san_move
    
    def _make_random_move(self):
        """
        return: a move in san notation representing the move the opponent made.
        """
        print("random move!")
        assert self.board.turn == chess.BLACK
        legal_moves : list = list(self.board.legal_moves)
        ch_move : chess.Move = np.random.choice(legal_moves, 1)[0]
        self.stockfish.make_moves_from_current_position([ch_move.uci()])
        move = self.board.san(ch_move)
        self.board.push(ch_move)
        assert self.board.turn == chess.WHITE
        return move

    def render(self, mode='human', close=False):
        print(self.board)
        return self.board

class FenChessHistoryEnvSingleTurn(TextEnv):
    def __init__(self, initial_history: TextHistory, max_moves=400, from_position=None):
        super().__init__()
        self.chess_env = ChessEnv(fen=True, from_position=from_position)
        self.from_position = from_position
        self.max_moves = max_moves
        self.from_position = from_position
        self.initial_history = initial_history

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.init_state, _ = self.chess_env.reset()
        self.num_moves_made = 0
        return self.initial_history + (Text(preprocess_state(self.init_state), False),)

    def step(self, text_history: TextHistory):
        assert text_history[-1].is_action
        action = text_history[-1].text
        action = postprocess_move(action)
        st, reward, done, opp_mv = self.chess_env.step(action) 
        new_state = Text(preprocess_state(st), False)
        self.num_moves_made += 1
        if self.num_moves_made > self.max_moves:
            done = 1
        return self.initial_history + (new_state,), reward, done
    
    def copy(self):
        return FenChessHistoryEnvSingleTurn(self.initial_history, self.max_moves, self.from_position)

class FenChessHistoryEnv(TextEnv):
    def __init__(self, max_moves=400, from_position=None, random_opponent=False):
        super().__init__()
        self.chess_env = ChessEnv(fen=True, from_position=from_position, random_opponent=random_opponent)
        self.from_position = from_position
        self.max_moves = max_moves
        self.from_position = from_position
        # self.initial_history = initial_history

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.init_state, _ = self.chess_env.reset()
        self.num_moves_made = 0
        return (Text(preprocess_state_og(self.init_state), False),)

    def step(self, text_history: TextHistory):
        assert text_history[-1].is_action
        action = text_history[-1].text
        action = postprocess_move(action)
        st, reward, done, opp_mv = self.chess_env.step(action) 
        new_state = Text(preprocess_state_og(st), False)
        self.num_moves_made += 1
        if self.num_moves_made > self.max_moves:
            done = 1
        return (new_state,), reward, done
    
    def copy(self):
        return FenChessHistoryEnv( self.max_moves, self.from_position)

def large_piece_random_endgame(pieces:str):
    """Provide a string like 'kQK' to represent a black king, white queen, 
    and white king on the board"""
    board = chess.Board()
    while True:
        board.clear()
        possible_squares = np.arange(0, 64)
        for piece in pieces:
            p = chess.Piece.from_symbol(piece)
            square = np.random.choice(possible_squares)
            board.set_piece_at(square, p)
            possible_squares = possible_squares[possible_squares != square]
        fen = board.fen()
        if board.is_valid() and not board.is_check():
            return fen

def text_env_eval_chess_positions(
    positions: List[str],
    policy: Union[TextPolicy, BatchedTextPolicy], 
    n_rollouts: int, 
    initial_text_history: Optional[TextHistory]=None, # only allow one initial_text_history here
    seed_generator: Optional[Iterator[int]]=None, 
    env_options: Optional[Dict]=None, # only allow one env_options here
    interaction_callback: Optional[Callable[[List[Tuple[TextHistory, TextHistory, TextHistory, float, bool]]], None]]=None, 
    bsize: int=1, 
    verbose: bool=True,
    random_opponent: bool=False,
    max_moves: int=400,
):
    interactions, rs, dones = [], [], []
    victories, percent_illegals, episode_length = [], [], []
    for position in positions:
        env = FenChessHistoryEnv(from_position=position, random_opponent=random_opponent, max_moves=max_moves)
        env_interactions = []
        for _ in tqdm(range((n_rollouts+(bsize-1))//bsize), disable=not verbose):
            actual_bsize = min(n_rollouts-len(env_interactions), bsize)
            npad = bsize - actual_bsize
            interaction_batch = interact_environment(
                env, 
                policy, 
                initial_text_history=initial_text_history, 
                env_seed=[None]*actual_bsize if seed_generator is None else [next(seed_generator) for _ in range(actual_bsize)], 
                env_options=[env_options]*actual_bsize, 
                bsize=actual_bsize,
                npad=npad,
            )
            
            for interaction in interaction_batch:
                env_interactions.append(interaction)
                
                # collect some metrics about how the chess agent did
                rewards = [x.reward for x in interaction]
                victories.append(1 if 1 in rewards else 0)
                num_illegal = sum([1 if x.reward == -1 and i < len(rewards) - 1 else 0 for i, x in enumerate(interaction)])
                percent_illegal = num_illegal / len(rewards) * 100
                percent_illegals.append(percent_illegal)
                episode_length.append(len(rewards))
                
                # collect the rewards and dones
                rs.append(sum(map(lambda x: x.reward, interaction)))
                dones.append(interaction[-1].done)
                if interaction_callback is not None:
                    interaction_callback(interaction)
        interactions.extend(env_interactions)
    
    rs = np.asarray(rs, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    results_summary = dict(
        reward=dict(
            mean=np.mean(rs), 
            std=np.std(rs), 
            min=np.min(rs), 
            max=np.max(rs), 
        ), 
        done=dict(
            mean=np.mean(dones), 
            std=np.std(dones), 
            min=np.min(dones), 
            max=np.max(dones), 
        ), 
        victories=dict(
            mean=np.mean(victories),
            std=np.std(victories),
            min=np.min(victories),
            max=np.max(victories),
        ),
        percent_illegals=dict(
            mean=np.mean(percent_illegals),
            std=np.std(percent_illegals),
            min=np.min(percent_illegals),
            max=np.max(percent_illegals),
        ),
        episode_length=dict(
            mean=np.mean(episode_length),
            std=np.std(episode_length),
            min=np.min(episode_length),
            max=np.max(episode_length),
        ),
    )
    
    return interactions, results_summary