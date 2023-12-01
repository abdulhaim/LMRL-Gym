from typing import Tuple, List, Dict
from llm_rl_scripts.maze.env.env import MazeEnv, describe_observation, describe_observation_give_position, illegal_penalty_reward, illegal_penalty_diff_scale, manhatten_actions, standard_reward, describe_observation_only_walls
from llm_rl_scripts.maze.env.mazes import double_t_maze_optimal_directions, maze2d_umaze, double_t_maze
import numpy as np
from LLM_RL.environment import Text
from collections import deque
from IPython import embed

def setup_maze_env(maze_name, describe_function, reward_function=None, last_k=1, max_steps=100):
    # setup environment
    if maze_name == 'umaze':
        maze = maze2d_umaze()
        valid_goals = np.array([[3, 3]])
        start_position = (3, 1)
    elif maze_name == "double_t_maze":
        maze = double_t_maze()
        valid_goals = np.array([[8, 6]])
        start_position = (1, 1)
    else:
        raise ValueError(f'unknown maze name: {maze_name}')
    
    # valid_goals = np.where(maze == 0)
    # valid_goals = np.array(list(zip(valid_goals[0], valid_goals[1])), dtype=np.int32)
    if describe_function == "describe_observation":
        describe_function = describe_observation
    elif describe_function == "describe_observation_give_position":
        describe_function = describe_observation_give_position
    elif describe_function == "describe_observation_only_walls":
        describe_function = describe_observation_only_walls
    else:
        raise ValueError(f'unknown describe function: {describe_function}')
    
    if reward_function is None or reward_function == "standard_reward":
        reward_function = standard_reward
    elif reward_function == "illegal_penalty_reward":
        reward_function = illegal_penalty_reward
    elif reward_function == "illegal_penalty_diff_scale":
        reward_function = illegal_penalty_diff_scale
    else:
        raise ValueError(f'unknown reward function: {reward_function}')
    
    env = MazeEnv(
        maze=maze, 
        valid_goals=valid_goals, 
        actions=manhatten_actions, 
        max_steps=max_steps, 
        display_initial_position=True,
        describe_function=describe_function,
        reward_function=reward_function,
        last_k=last_k,
    )
    return env

def pick_start_position(maze_name):
    if maze_name == 'umaze':
        return (3, 1)
    elif maze_name == "double_t_maze":
        return (1, 1)
    else:
        raise ValueError(f'unknown maze name: {maze_name}')
    

def compute_move_accuracy(policy, reranker=False):
    maze = double_t_maze()
    goal = (8, 6)
    correct_answers = double_t_maze_optimal_directions()
    positions = np.argwhere(maze == 0).tolist()    # note make sure to set temperature to 0
    num_correct = 0
    for position in positions:
        observation = describe_observation_give_position(maze, position, goal)
        text_history = (Text(observation, False),)
        # embed()
        if reranker:
            output = policy.act(text_history)
            prediction = output[-1].text
        else:
            output = policy.act([text_history], done=[False])
            prediction = output[-1][-1].text
        # output = policy.act(text_history)
        # prediction = output[-1].text
        if position[0] == goal[0] and position[1] == goal[1]:
            continue
        if prediction == correct_answers[tuple(position)]:
            num_correct += 1
            print("correct!", observation, position, prediction, correct_answers[tuple(position)])
        else:
            print("incorrect!", observation, position, prediction, correct_answers[tuple(position)])
    accuracy = num_correct/(len(positions)-1)*100
    return accuracy

def maze_solver(maze: np.ndarray, goal_positions: List[Tuple[int, int]]) -> Dict[Tuple[int, int], str]:
    maze = maze.tolist()
    assert len(maze) > 0 and len(maze[0]) > 0, 'maze must be non-zero in area'
    assert all([maze[goal_pos[0]][goal_pos[1]] == 1 for goal_pos in goal_positions]), 'goal pos must be 1'
    move_mapping = {
        (0, 1): 'move right\n',
        (0, -1): 'move left\n',
        (1, 0): 'move down\n',
        (-1, 0): 'move up\n',
    }
    x_size, y_size = len(maze), len(maze[0])
    out_of_bounds = lambda x, y: x < 0 or x >= x_size or y < 0 or y >= y_size
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    optimal_policy = dict()
    queue = deque(goal_positions)
    seen_pos = set(goal_positions)
    while len(queue) > 0:
        x, y = queue.popleft()
        for dx, dy in directions:
            new_pos = (x+dx, y+dy)
            if new_pos in seen_pos or out_of_bounds(*new_pos) or maze[new_pos[0]][new_pos[1]] == 0:
                continue
            queue.append(new_pos)
            seen_pos.add(new_pos)
            optimal_policy[new_pos] = move_mapping[(-dx, -dy)]
    return optimal_policy
