from __future__ import annotations
from curses.ascii import isdigit
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import re
from LLM_RL.environment import Text, TextHistory, TextTrajectory
from abc import ABC, abstractmethod
import uuid
import time
import numpy as np
import nltk
import math


class Role(Enum):
    BUYER = "BUYER"
    SELLER = "SELLER"

    def other(self):
        if self == self.BUYER:
            return self.SELLER
        elif self == self.SELLER:
            return self.BUYER
        else:
            raise NotImplementedError

    def __str__(self):
        if self == self.BUYER:
            return "Buyer"
        elif self == self.SELLER:
            return "Seller"
        else:
            raise NotImplementedError


INITIAL_STR = "Start\n"
DEFAULT_BUDGETS = [10000, 30000, 50000, 70000, 90000]
DEFAULT_PERSONALITIES = ['abusive', 'angry', 'insulting', 'polite', 'respectful', 'rude', 'sarcastic', 'talkative', 'toxic', 'uncommunicative']
DEFAULT_BRANDS = ['a Volkswagen', 'a Lexus', 'a Ford', 'a Mazda', 'a Hyundai', 'a Toyota', 'a Mercedes-Benz', 'a BMW', 'an Audi', 'a Subaru', 'a Honda', 'a Porsche', 'a Tesla']
DEFAULT_TYPES = ['luxury', 'sedan', 'convertible', 'truck', 'electric', 'SUV']
DEFAULT_FEATURES = ['backup camera', 'navigation system', 'heated seats', 'leather seats', 'third-row seating', 'blind spot monitoring', 'sunroof', 'Apple CarPlay']


BuyerInfo = Dict[str, Union[str, List[str], int]]
ConversationLines = List[Dict[str, str]]
ConversationOutput = Dict[str, Union[bool, int, Optional[int]]]
Conversation = Dict[str, Union[BuyerInfo, ConversationLines, ConversationOutput]]


def join_consecutive_actions(text_history: TextHistory) -> TextHistory:
    new_text_history: List[Text] = []
    curr_texts: List[Text] = []

    def join_texts(texts: List[Text], is_action: bool) -> Text:
        strs_to_join: List[str] = []
        for text in texts:
            to_join = text.text
            while len(to_join) > 0 and to_join[-1] == "\n":
                to_join = to_join[:-1]
            strs_to_join.append(to_join)
        return Text(" ".join(strs_to_join) + "\n", is_action=is_action)

    for text in text_history:
        if text.is_action:
            curr_texts.append(text)
        else:
            if len(curr_texts) > 0:
                new_text_history.append(join_texts(curr_texts, is_action=True))
                curr_texts = []
            new_text_history.append(text)
    if len(curr_texts) > 0:
        new_text_history.append(join_texts(curr_texts, is_action=True))
        curr_texts = []

    return tuple(new_text_history)


def create_buyer_info_str(buyer_info: BuyerInfo) -> str:
    """
    buyer_info: 
        {
            "personality": str,
            "preferred_brands": str,
            "preferred_type": str,
            "preferred_features": [str, ...],
            "budget": int,
        }
    """
    personality = buyer_info["personality"]
    preferred_brands = buyer_info["preferred_brands"]
    preferred_type = buyer_info["preferred_type"]
    preferred_features = ", ".join(buyer_info["preferred_features"])
    budget = f"${buyer_info['budget']:,}"

    return f"Personality: {personality}. Prefers {preferred_brands} {preferred_type} with {preferred_features}. Budget: {budget}\n"


def create_lines_from_text_history(text_history: TextHistory) -> ConversationLines:
    """
    text_history from environment rollout from the seller's perspective.
    [INITIAL_STR, seller text, buyer text, ..., seller text]

    Returns lines:
    [
        {"role": "Seller", "text": "..."},
        {"role": "Buyer", "text": "..."},
        ...
    ]
    """
    curr_role = Role.SELLER
    lines = []
    for text in text_history[1:]:
        if text.is_action:
            assert curr_role == Role.SELLER
        else:
            assert curr_role == Role.BUYER

        lines.append({
            "role": str(curr_role),
            "text": text.text.strip(),
        })

        curr_role = curr_role.other()
    
    return lines


def compute_reward(buyer_info: BuyerInfo, output: ConversationOutput, reward_mode: str="fancy") -> float:
    """
    buyer_info: 
        {
            "personality": str,
            "preferred_brands": str,
            "preferred_type": str,
            "preferred_features": [str, ...],
            "budget": int,
        }
    "output": 
        {
            "car_bought": bool,
            "msrp": int,
            "buy_price": int | None,
        }
    """
    msrp = output["msrp"]
    car_bought = output["car_bought"]
    budget = buyer_info["budget"]
    buy_price = output["buy_price"] if car_bought else None

    if reward_mode == "fancy":
        # just in case, but msrp shouldn't be 0
        if msrp == 0:
            return 0.0

        if car_bought:
            if buy_price is None:
                return 0.0
            r = buy_price / ((budget + msrp) * 0.5)
        else:
            r =  -(budget - msrp) / msrp
        
        return r

    elif reward_mode == "revenue":
        if car_bought:
            if buy_price is None:
                return 0.0
            else:
                return buy_price / 1000.0  # revenue in the thousands
        else:
            return 0.0

    else:
        raise NotImplementedError


# don't allow MSRP to be 0, so it has to start at 1
OUTPUT_EXTRACTION_PATTERN = re.compile(r"Output: Decision=(Accept|Reject) MSRP=\$([1-9][0-9,]*)( Buy Price=\$([0-9][0-9,]*))?")

def extract_output_from_str(line: str) -> Tuple[Optional[ConversationOutput], Optional[str]]:
    """Given an output string, extract an output if possible and return the line without the output string."""
    output_match = re.search(OUTPUT_EXTRACTION_PATTERN, line)
    if output_match is None:
        return None, line
    
    car_bought = output_match.group(1) == "Accept"
    msrp = int(output_match.group(2).replace(",", ""))
    if car_bought and output_match.group(4) is not None:
        buy_price = int(output_match.group(4).replace(",", ""))
    else:
        buy_price = None
    output = {
        "car_bought": car_bought,
        "msrp": msrp,
        "buy_price": buy_price
    }
    
    output_str_ind = line.find(output_match.group(0))
    extracted_line = line[:output_str_ind] + line[output_str_ind+len(output_match.group(0)):]

    return output, extracted_line


def create_trajectory_from_conversation(conversation: Conversation, role: Role, reward_mode: str="fancy") -> TextTrajectory:
    """Create a TextTrajectory from a conversation.
    conversation: A Conversation is a dictionary with
        {
            "buyer_info": {
                "personality": ...,
                "preferred_brands": ...,
                "preferred_type": ....,
                "preferred_features": ["...", ...],
                "budget": 10000,
            }
            "lines": [
                {"role": "Seller", "text": "..."},
                {"role": "Buyer", "text": "..."},
                ...
            ]
            "output": {
                "car_bought": True or False,
                "msrp": 20000,
                "buy_price": 19000 or None,
            }
        }
    """
    text_history: List[Text] = [Text(INITIAL_STR, is_action=False)]
    
    if role == Role.BUYER:
        buyer_info_text = Text(create_buyer_info_str(conversation["buyer_info"]), is_action=False)
        text_history.append(buyer_info_text)
    
    for line in conversation["lines"]:
        is_action = line["role"] == str(role)        
        line_text = Text(line["text"] + "\n", is_action=is_action)
        text_history.append(line_text)
    
    done = "output" in conversation

    if done and role == Role.BUYER:
        msrp = conversation["output"]["msrp"]
        if conversation["output"]["car_bought"]:
            buy_price = conversation["output"]["buy_price"]
            text_history.append(Text(f"Output: Decision=Accept MSRP=${msrp:,} Buy Price=${buy_price:,}\n", is_action=True))
        else:
            text_history.append(Text(f"Output: Decision=Reject MSRP=${msrp:,}\n", is_action=True))

    text_history = list(join_consecutive_actions(text_history))
    text_history = [replace(text, text=text.text if text.text.endswith("\n") else text.text + "\n") for text in text_history]

    reward = [0.0 for _ in range(len(text_history))]

    if done and role == Role.SELLER:
        r = compute_reward(conversation["buyer_info"], conversation["output"], reward_mode=reward_mode)

        for i in range(len(text_history)-1, -1, -1):
            if text_history[i].is_action:
                reward[i] = r
                break

    return TextTrajectory(tuple(text_history), tuple(reward), done)


def create_trajectories_from_conversations(
    conversations: List[Conversation], 
    role: Role,
    reward_mode: str="fancy",
) -> List[TextTrajectory]:
    """
    conversations: raw data from the json file, which is a list of conversations.
        See `create_trajectory_from_conversation`
    """
    return [
        create_trajectory_from_conversation(conversation, role, reward_mode) 
        for conversation in conversations
    ]
