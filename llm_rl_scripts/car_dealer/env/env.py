from dataclasses import replace
from typing import Dict, List, Optional, Tuple, Any
import random
from LLM_RL.environment import Text, TextEnv, BatchedTextEnv, TextHistory, TextPolicy, BatchedTextPolicy, StepResult
from llm_rl_scripts.car_dealer.env.data import (
    Role, INITIAL_STR, DEFAULT_BUDGETS, DEFAULT_FEATURES, DEFAULT_BRANDS, DEFAULT_PERSONALITIES, DEFAULT_TYPES, BuyerInfo, ConversationOutput,
    create_lines_from_text_history, create_trajectory_from_conversation, extract_output_from_str, compute_reward
)

EpisodeInfo = Dict[str, Any]

class CarDealerPolicyEnvironment(TextEnv):
    def __init__(
        self, 
        buyer: TextPolicy,
        max_conversation_length: int=50,
        reward_mode: str="fancy",
    ):
        self.buyer = buyer
        self.max_conversation_length = max_conversation_length
        self.reward_mode = reward_mode

        self.random = random.Random(None)
        self.buyer_info: Optional[BuyerInfo] = None
        self.output: Optional[ConversationOutput] = None
        self.verbose = False

    def step(self, text_history: TextHistory) -> StepResult:
        assert text_history[-1].is_action
        assert self.buyer_info is not None, "call env.reset() first."
        if self.verbose:
            print(f"Seller: {repr(text_history[-1].text)}")

        # query the buyer for a new line
        input_conversation = {
            "buyer_info": self.buyer_info,
            "lines": create_lines_from_text_history(text_history),
        }
        input_buyer_text_trajectory = create_trajectory_from_conversation(input_conversation, Role.BUYER)

        output_buyer_text_history = self.buyer.act(input_buyer_text_trajectory.text_history)

        reward = 0.0
        done = False

        # test if the buyer concluded the conversation
        last_buyer_str = output_buyer_text_history[-1].text
        output_text_history = text_history + (Text(last_buyer_str, is_action=False),)
        if self.verbose:
            print(f"Buyer: {repr(last_buyer_str)}")

        output, extracted_last_buyer_str = extract_output_from_str(last_buyer_str)

        if output is not None:
            self.output = output
            reward = compute_reward(self.buyer_info, output, self.reward_mode)
            done = True
            output_text_history = text_history + (Text(extracted_last_buyer_str, is_action=False),)
            if self.verbose:
                print(f"Buyer (extracted): {repr(extracted_last_buyer_str)}")
                print(f"reward: {reward}, output: {output}")
            return output_text_history, reward, done
        
        # now test if max conversation length has been reached but it didn't produce the output yet
        if len(output_text_history) - 1 >= self.max_conversation_length:
            # we need to figure out what the output is, so we prompt the buyer again
            input_last_buyer_str = last_buyer_str
            if input_last_buyer_str.endswith("\n"):
                input_last_buyer_str = input_last_buyer_str[:-1]
            input_last_buyer_str = input_last_buyer_str + "Output: Decision="
            output_buyer_text_history = self.buyer.act(output_buyer_text_history[:-1] + (Text(input_last_buyer_str, is_action=True),))
            if self.verbose:
                print(f"Buyer (last step re-prompt): {repr(output_buyer_text_history[-1].text)}")
            output, extracted_last_buyer_str = extract_output_from_str(output_buyer_text_history[-1].text)
            
            done = True
            if output is not None:
                self.output = output
                reward = compute_reward(self.buyer_info, output, self.reward_mode)
                output_text_history = text_history + (Text(extracted_last_buyer_str, is_action=False),)
                if self.verbose:
                    print(f"Buyer (extracted): {repr(extracted_last_buyer_str)}")
                    print(f"reward: {reward}, output: {output}")
                return output_text_history, reward, done
            # if output is still None, then default into returning 0 reward

        return output_text_history, reward, done
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> TextHistory:
        if seed is not None:
            self.random = random.Random(seed)

        self.buyer_info = {
            "personality": self.random.choice(DEFAULT_PERSONALITIES),
            "preferred_brands": self.random.choice(DEFAULT_BRANDS),
            "preferred_type": self.random.choice(DEFAULT_TYPES),
            "preferred_features": self.random.sample(DEFAULT_FEATURES, k=self.random.randint(1, 4)),
            "budget": self.random.choice(DEFAULT_BUDGETS)
        }
        if options is not None:
            self.verbose = options.get("verbose", False)
        else:
            self.verbose = False
        if self.verbose:
            print("Env reset")
            print("buyer info:")
            for key, value in self.buyer_info.items():
                print(f"    {key}: {value}")

        return (Text(INITIAL_STR, is_action=False),)

    def get_episode_info(self) -> Optional[List[Optional[EpisodeInfo]]]:
        info = {
            "buyer_info": self.buyer_info,
            "output": self.output
        }
        return info

    def copy(self):
        return CarDealerPolicyEnvironment(
            buyer=self.buyer,
            max_conversation_length=self.max_conversation_length,
            reward_mode=self.reward_mode
        )


class BatchedCarDealerPolicyEnvironment(BatchedTextEnv):
    def __init__(
        self, 
        buyer: BatchedTextPolicy,
        buyer_bsize: Optional[int]=None,
        max_conversation_length: int=50,
        reward_mode: str="fancy",
    ):
        self.bsize = buyer_bsize
        self.buyer = buyer
        self.max_conversation_length = max_conversation_length
        self.reward_mode = reward_mode

        self.randoms: List[random.Random] = [random.Random(None) for _ in range(self.bsize)]
        self.buyer_infos: Optional[List[BuyerInfo]] = None
        self.outputs: Optional[List[ConversationOutput]] = None
        self.verbose: bool = False

    def step(self,  text_history_batch: List[Optional[TextHistory]], done: Optional[List[bool]]=None) -> List[Optional[StepResult]]:
        assert self.buyer_infos is not None, "call env.reset() first."
        assert len(text_history_batch) <= self.bsize, f"input batch size {len(text_history_batch)} is larger than buyer's batch size {self.bsize}"

        input_buyer_text_histories = [None for _ in range(self.bsize)]
        input_buyer_dones = [True for _ in range(self.bsize)]
        for i, (buyer_info, text_history, done) in enumerate(zip(self.buyer_infos, text_history_batch, done)):
            if done:
                continue

            if self.verbose:
                print(f"Seller #{i}: {repr(text_history[-1].text)}")

            # query the buyer for a new line
            input_conversation = {
                "buyer_info": buyer_info,
                "lines": create_lines_from_text_history(text_history),
            }
            input_buyer_text_trajectory = create_trajectory_from_conversation(input_conversation, Role.BUYER)

            input_buyer_text_histories[i] = input_buyer_text_trajectory.text_history
            input_buyer_dones[i] = done

        output_buyer_text_histories = self.buyer.act(input_buyer_text_histories, input_buyer_dones)

        step_results = [None for _ in range(len(text_history_batch))]

        final_input_buyer_text_histories = [None for _ in range(self.bsize)]
        final_input_buyer_dones = [True for _ in range(self.bsize)]
        need_final_pass = False
        
        for i, (buyer_info, text_history, output_buyer_text_history, input_buyer_done) in enumerate(
            zip(self.buyer_infos, text_history_batch, output_buyer_text_histories, input_buyer_dones)
        ):
            if input_buyer_done:
                continue

            # test if the buyer concluded the conversation
            last_buyer_str = output_buyer_text_history[-1].text
            output_text_history = text_history + (Text(last_buyer_str, is_action=False),)
            if self.verbose:
                print(f"Buyer {i}: {repr(last_buyer_str)}")

            output, extracted_last_buyer_str = extract_output_from_str(last_buyer_str)

            if output is not None:
                self.outputs[i] = output
                reward = compute_reward(buyer_info, output, self.reward_mode)
                output_text_history = text_history + (Text(extracted_last_buyer_str, is_action=False),)
                if self.verbose:
                    print(f"Buyer #{i} (extracted): {repr(extracted_last_buyer_str)}")
                    print(f"reward #{i}: {reward}, output: {output}")
                step_results[i] = (output_text_history, reward, True)
                continue
        
            # now test if max conversation length has been reached but it didn't produce the output yet
            if len(output_text_history) - 1 >= self.max_conversation_length:
                # we need to figure out what the output is, so we prompt the buyer again
                input_last_buyer_str = last_buyer_str
                if input_last_buyer_str.endswith("\n"):
                    input_last_buyer_str = input_last_buyer_str[:-1]
                input_last_buyer_str = input_last_buyer_str + "Output: Decision="
                final_input_buyer_text_histories[i] = output_buyer_text_history[:-1] + (Text(input_last_buyer_str, is_action=True),)
                final_input_buyer_dones[i] = False
                need_final_pass = True
                continue
            
            # not done yet
            step_results[i] = (output_text_history, 0.0, False)

        if need_final_pass:
            # special case for if max conversation length reached but output not produced
            final_output_buyer_text_histories = self.buyer.act(final_input_buyer_text_histories, final_input_buyer_dones)
            for i, (buyer_info, text_history, final_output_buyer_text_history, final_input_buyer_done) in enumerate(
                zip(self.buyer_infos, text_history_batch, final_output_buyer_text_histories, final_input_buyer_dones)
            ):
                # this means don't use this output, the input should be None for this
                if final_input_buyer_done:
                    continue
            
                if self.verbose:
                    print(f"Buyer #{i} (last step re-prompt): {repr(final_output_buyer_text_history[-1].text)}")
                output, extracted_last_buyer_str = extract_output_from_str(final_output_buyer_text_history[-1].text)
                
                if output is not None:
                    self.outputs[i] = output
                    reward = compute_reward(buyer_info, output, self.reward_mode)
                    output_text_history = text_history + (Text(extracted_last_buyer_str, is_action=False),)
                    if self.verbose:
                        print(f"Buyer #{i} (extracted): {repr(extracted_last_buyer_str)}")
                        print(f"reward #{i}: {reward}, output: {output}")
                    step_results[i] = (output_text_history, reward, True)
                else:
                    # if output is still None, then default into returning 0 reward
                    step_results[i] = (output_text_history, 0.0, True)
        
        return step_results[:len(text_history_batch)]
        
    def reset(self, seed_batch: Optional[List[Optional[int]]]=None, options_batch: Optional[List[Optional[Dict]]]=None) -> TextHistory:
        # No padding for reset
        self.buyer_infos = []
        if seed_batch is None:
            seed_batch = [None for _ in range(self.bsize)]

        self.outputs = [None for _ in range(len(seed_batch))]

        initial_text_history_batch: List[TextHistory] = []

        for i, seed in enumerate(seed_batch):
            self.randoms.append(random.Random(seed))

            buyer_info = {
                "personality": self.randoms[i].choice(DEFAULT_PERSONALITIES),
                "preferred_brands": self.randoms[i].choice(DEFAULT_BRANDS),
                "preferred_type": self.randoms[i].choice(DEFAULT_TYPES),
                "preferred_features": self.randoms[i].sample(DEFAULT_FEATURES, k=self.randoms[i].randint(1, 4)),
                "budget": self.randoms[i].choice(DEFAULT_BUDGETS)
            }
            self.buyer_infos.append(buyer_info)

            if self.verbose:
                print(f"Env reset batch_ind {i}")
                print("buyer info:")
                for key, value in buyer_info.items():
                    print(f"    {key}: {value}")

            initial_text_history_batch.append((Text(INITIAL_STR, is_action=False),))

        return initial_text_history_batch

    def get_episode_info(self) -> Optional[List[Optional[EpisodeInfo]]]:
        infos = []
        for buyer_info, output in zip(self.buyer_infos, self.outputs):
            infos.append({
                "buyer_info": buyer_info,
                "output": output
            })
        return infos

    def copy(self):
        return BatchedCarDealerPolicyEnvironment(
            buyer=self.buyer,
            buyer_bsize=self.bsize,
            max_conversation_length=self.max_conversation_length,
            reward_mode=self.reward_mode,
        )

