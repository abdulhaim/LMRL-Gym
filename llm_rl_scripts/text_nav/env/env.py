import re
import sys
import textwrap
from typing import Optional, Tuple
import textworld
from textworld.core import EnvInfos
from textworld.agents import HumanAgent
from LLM_RL.environment import Text, TextEnv, TextHistory
from llm_rl_scripts.text_nav.env.make_game import build_and_compile_game


class TextNavEnv(TextEnv):
    """
    Environment for textual navigation game.
    """

    def __init__(self,
                 display_location: bool = False,
                 display_inventory: bool = False):
        
        self.infos = EnvInfos(description=True,
                              admissible_commands=True,
                              location=display_location,
                              inventory=display_inventory)
        self.reset()

    def _reset(self, seed: Optional[int] = None):
        _, self.game_file = build_and_compile_game(not self.infos.location)
        self.env = textworld.start(self.game_file, self.infos)
        self.display_command_during_render = True

        self.state = self.env.reset()

        redundant = ["examine", "look", "inventory"]
        self.state["admissible_commands"] = list(
            c for c in self.state["admissible_commands"] if not any(a in c for a in redundant))
        self.state.feedback += "\nAdmissible commands: {}\n".format(
            ", ".join(self.state["admissible_commands"]))
        
        self.state.feedback = re.sub("-=.*=-\n", "", self.state.feedback)

    def _step(self, command: str):
        command = command.strip()
        self.state, _, _ = self.env.step(command)

        if self.infos.inventory:
            inventory, _, _ = self.env.step("inventory")
            self.state["inventory"] = inventory.feedback.strip()
            self.state.feedback += "\n{}\n".format(self.state["inventory"])

        redundant = ["examine", "look", "inventory"]
        self.state["admissible_commands"] = list(
            c for c in self.state["admissible_commands"] if not any(a in c for a in redundant))
        self.state.feedback += "\nAdmissible commands: {}\n".format(
            ", ".join(self.state["admissible_commands"]))
        
        self.state.feedback = re.sub("-=.*=-\n", "", self.state.feedback)
    
    def reset(self) -> TextHistory:
        self._reset()
        return tuple(self.state.feedback,)

    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        assert text_history[-1].is_action

        command = text_history[-1].text
        self._step(command)
        return (
            text_history + (Text(command, true), Text(self.state.feedback, False)),
            self.state["score"],
            self.state["done"] 
        )

    def render(self) -> str:
        msg = self.state.feedback.rstrip() + "\n"
        if self.display_command_during_render and self.state.last_command is not None:
            msg = '> ' + self.state.last_command + "\n" + msg

        # Wrap each paragraph.
        paragraphs = msg.split("\n")
        paragraphs = ["\n".join(textwrap.wrap(paragraph, width=80)) for paragraph in paragraphs]
        msg = "\n".join(paragraphs)

        sys.stdout.write(msg + "\n")


def play(env: TextNavEnv, max_nb_steps: int = 20) -> float:
    """Play game as a human agent."""
    agent = HumanAgent()
    agent.reset(env.env)
    env._reset()
    env.render()

    try:
        for _ in range(max_nb_steps):
            command = agent.act(env.state, env.state.score, env.state.done)
            env._step(command)
            env.render()

            if env.state.done:
                break

    except KeyboardInterrupt:
        pass  # Stop the game.
    finally:
        env.close()

    msg = "Done after {} steps. Score {}/{}."
    msg = msg.format(env.state.moves, env.state.score, env.state.max_score)
    print(msg)

    return env.state.score
