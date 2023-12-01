from typing import Callable, Optional, Dict, List, Tuple
from LLM_RL.environment import Text, TextEnv, TextHistory
import numpy as np
import random
from llm_rl_scripts.maze.env.randomness import RandomState
from IPython import embed

def describe_objects(object: str, relations: List[str]):
    if len(relations) == 0:
        return f"There are no {object}s near you."
    if len(relations) == 1:
        return f"There is a {object} {relations[0]}."
    return f"There are {object}s {', '.join(relations)}."

def describe_observation(maze: np.ndarray, 
                         position: Tuple[int, int], 
                         goal_position: Tuple[int, int], 
                         initial_position: Tuple[int, int]=None,
                         move_history: List[str]=None,
                         ) -> int:
    assert len(maze.shape) == 2

    goal_description = f"The goal is at position {' '.join(str(goal_position[0]))}, {' '.join(str(goal_position[1]))}."
    # if initial_position is not None:
    #     initial_description = f"Your starting position is at position {' '.join(str(initial_position[0]))}, {' '.join(str(initial_position[1]))}."
    delta_descriptions = {"to your right": (0, 1), "to your left": (0, -1), "above you": (-1, 0), "below you": (1, 0)} 
                        #   "to your top left": (-1, -1), "to your top right": (-1, 1), 
                        #   "to your bottom left": (1, -1), "to your bottom right": (1, 1)}
    walls = []
    # goals = []
    for k, (dy, dx) in delta_descriptions.items():
        if maze[position[0]+dy, position[1]+dx] == 1:
            walls.append(k)
        # elif maze[position[0]+dy, position[1]+dx] == 3:
        #     goals.append(k)
    
    wall_description = describe_objects("wall", walls)
    
    # # history description
    # history_description = ""
    # if move_history is not None:
    #     history_description = f"Your move history is {' '.join(move_history)}."

    # goal_location_description = describe_objects("goal", goals)

    # return f"{goal_description} {wall_description} {goal_location_description}\n"
    # if initial_position is not None:
    #     return f"{goal_description} {initial_description} {history_description} {wall_description}\n"
    return f"{goal_description} {wall_description}\n"

def describe_observation_give_position(maze:np.ndarray,
                                       position: Tuple[int, int],
                                       goal_position: Tuple[int, int],
                                       initial_position: Tuple[int, int]=None,
                                       move_history: List[str]=None,
                                       ) -> str:
    goal_description = f"The goal is at position {' '.join(str(goal_position[0]))}, {' '.join(str(goal_position[1]))}."
    curr_position_description = f"Your current position is at position {' '.join(str(position[0]))}, {' '.join(str(position[1]))}."
    delta_descriptions = {"to your right": (0, 1), "to your left": (0, -1), "above you": (-1, 0), "below you": (1, 0)} 

    walls = []
    for k, (dy, dx) in delta_descriptions.items():
        if maze[position[0]+dy, position[1]+dx] == 1:
            walls.append(k)
    
    wall_description = describe_objects("wall", walls)
    
    return f"{goal_description} {curr_position_description} {wall_description}\n"

def describe_observation_only_walls(maze:np.ndarray, 
                                    position: Tuple[int, int],
                                    goal_position: Tuple[int, int]=None,
                                    initial_position: Tuple[int, int]=None,
                                    move_history: List[str]=None,) -> str:
    delta_descriptions = {"to your right": (0, 1), "to your left": (0, -1), "above you": (-1, 0), "below you": (1, 0)} 
    walls = []
    for k, (dy, dx) in delta_descriptions.items():
        if maze[position[0]+dy, position[1]+dx] == 1:
            walls.append(k)
    wall_description = describe_objects("wall", walls)
    return f"{wall_description}\n"

diagonal_actions = {
    'move left\n': (0, -1), 
    'move right\n': (0, 1), 
    'move up\n': (-1, 0), 
    'move down\n': (1, 0), 
    'move top left\n': (-1, -1), 
    'move top right\n': (-1, 1), 
    'move bottom left\n': (1, -1), 
    'move bottom right\n': (1, 1), 
}

manhatten_actions = {
    'move left\n': (0, -1), 
    'move right\n': (0, 1), 
    'move up\n': (-1, 0), 
    'move down\n': (1, 0), 
}

def maze_proposal_function(text_history: TextHistory) -> List[TextHistory]:
    return [text_history+(Text(action, True),) for action in manhatten_actions.keys()]

def update_position(maze: np.ndarray, position: Tuple[int, int], action: str, actions: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    if action in actions and maze[position[0] + actions[action][0], position[1] + actions[action][1]] == 0:
        return (position[0] + actions[action][0], position[1] + actions[action][1])
    return position

def standard_reward(action, goal, position, possible_actions):
    if position[0] == goal[0] and position[1] == goal[1]:
        return 0.0
    elif action not in possible_actions:
        return -4.0
    else:
        return -1.0

def illegal_penalty_reward(action, goal, position, possible_actions):
    if position[0] == goal[0] and position[1] == goal[1]:
        return 1.0
    elif action not in possible_actions:
        return -1.0
    else:
        return 0.0

def illegal_penalty_diff_scale(action, goal, position, possible_actions):
    if position[0] == goal[0] and position[1] == goal[1]:
        return 1.0
    elif action not in possible_actions:
        return -100.0
    else:
        return -1.0

class MazeEnv(TextEnv):
    def __init__(self, maze: np.ndarray, 
                 valid_goals: np.ndarray, 
                 actions: Dict[str, Tuple[int, int]], 
                 max_steps: Optional[int]=None, 
                 display_initial_position: bool=False,
                 describe_function: Callable[[np.ndarray, Tuple[int, int], Tuple[int, int], Optional[Tuple[int, int]], Optional[List[str]]], str]=describe_observation_give_position,
                 reward_function: Callable[[str, Tuple[int, int], Tuple[int, int], Dict[str, Tuple[int, int]]], float]=standard_reward,
                 last_k:int=40,
                 ):
        assert len(maze.shape) == 2
        assert all([maze[goal[0], goal[1]] == 0 for goal in valid_goals])

        self.maze = maze
        self.valid_goals = valid_goals
        self.actions = actions
        self.max_steps = max_steps
        self.display_initial_position = display_initial_position
        self.num_steps = 0
        self.describe_function = describe_function
        self.move_history = []
        self.last_k = last_k
        
        self.reward_function = reward_function
        
        self.random_state = RandomState(None)
        self.reset()
    
    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        assert text_history[-1].is_action
        # embed()
        if self.max_steps is not None and self.num_steps >= self.max_steps:
            return (Text("Failure\n", False),), -1.0, True
        
        action = text_history[-1].text    
        self.position = update_position(self.maze, self.position, action, self.actions)
        
        self.move_history.append(action.replace('\n', ''))
        
        reward = self.reward_function(action, self.goal, self.position, self.actions)
        if self.position[0] == self.goal[0] and self.position[1] == self.goal[1]:
            return (Text("Success\n", False),), reward, True
        
        # move_history = [text_history[i].text for i in range(0, len(text_history), 2) if text_history[i].is_action]
        self.num_steps += 1
        obs_description = self.describe_function(self.maze, self.position, self.goal, self.initial_position, self.move_history)
        if action not in self.actions:
            return (Text(obs_description, False),), reward, False

        new_history = list(text_history) + [Text(obs_description, False)]
        new_history = new_history[max(0, len(new_history)-self.last_k):]
        return tuple(new_history), reward, False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> TextHistory:
        self.random_state.reset(seed)
        self.num_steps = 0

        if options is not None and 'goal' in options:
            self.goal = options['goal']
        else:
            self.goal = random.choice(self.valid_goals).tolist()
        
        positions = np.argwhere(self.maze == 0).tolist()
        positions.remove(self.goal)
        
        if options is not None and 'init_position' in options:
            assert list(options['init_position']) in positions
            self.position = list(options['init_position'])
        else:
            self.position = random.choice(positions)
        
        if self.display_initial_position:
            self.initial_position = self.position.copy()
        else:
            self.initial_position = None
        
        # print('initial position:', self.position)
        obs_description = self.describe_function(self.maze, self.position, self.goal, self.initial_position)

        self.random_state.freeze()
        
        return (Text(obs_description, False),)
