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


Conversation = Dict[str, Union[str, List[str]]]


@dataclass
class WordVariants:
    words: List[str]
    pos_tags: List[List[Tuple[str, str]]]

    @classmethod
    def from_list(cls, words_list: List[str]):
        pos_tags = [nltk.pos_tag(nltk.word_tokenize(word.lower())) for word in words_list]
        return cls(words=words_list, pos_tags=pos_tags)

    @classmethod
    def from_str(cls, words_str: str):
        words_list = words_str.split(";")
        return cls.from_list(words_list)

    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        assert 0 <= idx < len(self.words), f"Index {idx} out of range"
        return self.words[idx]

    def json(self):
        return self.words.copy()

    def __str__(self):
        return f"({', '.join(self.words)})"
    
    def __repr__(self) -> str:
        return f"WordVariants([{', '.join(self.words)}])"


DEFAULT_OBJECT_DICT = {
    "Sports": ["Basketball", "Football", "Baseball", "Soccer ball", "Golf ball", "Tennis ball", "Volleyball", "Tennis racket", "Baseball bat", "Helmet"],
    "Animals": ["Cat", "Dog", "Horse", "Cow", "Sheep", "Rabbit", "Lion", "Tiger", "Bear", "Elephant"],
    "Fruits": ["Apple", "Banana", "Orange", "Strawberry", "Grape", "Watermelon", "Pineapple", "Mango", "Cantaloupe", "Peach"],
    "Vehicles": ["Car", "Truck", "Motorcycle", "Boat", "Airplane;Plane", "Train", "Bus", "Helicopter", "Scooter", "Ship"],
    "Clothes": ["Shirt", "Pants;Pant;Pair of pants", "Jacket", "Dress", "Skirt", "Belt", "Shoes;Shoe;Pair of shoes", "Boots;Boot;Pair of boots", "Socks;Sock;Pair of socks", "Hat", "Scarf"],
    "Electronics": ["Computer", "Smartphone", "Television;TV", "Headphone;Headphones;Pair of headphones", "Monitor;Computer monitor", "Camera", "Microwave;Microwave oven", "Refrigerator", "Blender", "Computer keyboard;Keyboard"],
    "Musical Instruments": ["Piano", "Guitar", "Drum;Drums", "Violin", "Saxophone", "Flute", "Trumpet", "Clarinet", "Harp", "Trombone"],
    "Furniture": ["Chair", "Table", "Bed", "Desk", "Couch", "Dresser", "Bookcase", "Nightstand", "Mattress", "Pillow"],
    "Office Supplies": ["Pen", "Paper;Piece of paper", "Stapler", "Printer", "Calculator", "Battery;Battery pack;Pack of batteries", "Toothbrush", "Toothpaste", "Pencil", "Sharpie", "Scissors;Pair of scissors", "Key", "Diary", "Calendar"],
    "Vegetables": ["Carrot", "Potato", "Broccoli", "Tomato", "Onion", "Spinach", "Corn", "Peas;Pea", "Celery", "Cucumber"],
    "Art": ["Painting;Canvas painting;Oil painting;Watercolor painting", "Paintbrush", "Canvas;Painting canvas", "Eraser;Pencil eraser", "Marker", "Glue;Glue stick;Bottle of glue", "Sculpture"],
    "Kitchen Tools": ["Knife", "Spoon", "Fork", "Plate", "Bowl", "Cooking pot;Pot", "Pan;Saucepan;Frying pan", "Cup", "Chopstick;Chopsticks;Pair of chopsticks", "Whisk"],
    "Nature": ["Rock", "Tree", "Bush", "Mountain", "Forest", "Ocean", "Sea", "Lake", "River", "Meteorite", "Cactus"],
    "Toys": ["Lego;Lego set", "Doll;Toy doll;Plush doll", "Kite", "Puzzle;Jigsaw puzzle", "Stuffed animal"],
    "Jewelry": ["Earring;Earrings;Pair of earrings", "Necklace", "Bracelet", "Ring", "Brooch", "Hairclip", "Pendant", "Watch", "Locket"],
    "Garden Supplies": ["Gloves;Glove;Pair of gloves", "Shovel", "Rake", "Watering can", "Lawn mower"],
    "Tools": ["Hammer", "Screwdriver", "Wrench", "Saw", "Pliers;plier;Pair of pliers", "Drill"]
}

INVALID_QUESTION = "Is this a valid question?\n"
INITIAL_STR = "Questions:\n"


def get_default_word_list() -> List[WordVariants]:
    word_list = []
    for _, words in DEFAULT_OBJECT_DICT.items():
        word_list.extend(map(lambda x: WordVariants.from_str(x), words))
    return word_list


def create_trajectory_from_history(
    word_var: WordVariants, 
    text_history: TextHistory, 
    max_conversation_len: int=20, 
) -> TextTrajectory:
    """Create a TextTrajectory from a TextHistory
    """
    assert len(text_history) % 2 == 1, "TextHistory should be [initial str, question1, answer1, ..., questionN, answerN]."
    assert all(question_text.is_action for question_text in text_history[1::2]), "All questions should be actions."
    assert all(not answer_text.is_action for answer_text in text_history[0::2]), "All answers should not be actions."
    # subtract 1 because of the starting text, then text_history contains pairs of questions and answers
    conversation_len = (len(text_history) - 1) // 2
    assert conversation_len <= max_conversation_len, f"Conversation is too long {conversation_len}. Max should be {max_conversation_len}."

    reward: List[float] = []
    for text in text_history:
        if text.is_action:
            reward.append(-1.0)
        else:
            reward.append(0.0)

    if len(text_history) < 2:
        done = False
    else:
        last_question = text_history[-2].text.strip()
        last_answer = text_history[-1].text.strip()
        word_guessed = last_answer == "Yes." and is_done(word_var, last_question)

        done = word_guessed or conversation_len == max_conversation_len

        if word_guessed:
            reward[-2] = 0.0

    return TextTrajectory(text_history, tuple(reward), done)


def split_question_answer(line: str) -> Tuple[str, str]:
    *question, answer = line.split("? ")
    assert len(question) > 0, f"Invalid line in conversation: \"{line}\""
    question = "? ".join(question) + "?"
    return question, answer


def create_trajectory_from_conversation(conversation: Conversation, max_conversation_len: int=20, recompute_done: bool=False) -> TextTrajectory:
    """Create a TextTrajectory from a conversation.
    conversation: A Conversation is a dictionary with
        {
            "lines": [
                ...
                Lines of questions and answers in the form "Question? Yes or No."
            ],
            "correct": True or False, whether the conversation ended because the correct object was guessed or not,
            "word": [
                ...
                List of object/word that the asker is guessing. Typically one, but some words have variants.
            ]
        }
    max_conversation_len: How many lines of conversation? Default=20, because the task is called twenty questions.
    recompute_done: If True, recompute if a word is correctly guessed for this conversation.
    """
    if recompute_done:
        raise NotImplementedError("Not yet implemented recompute_done=True.")

    text_history: TextHistory = [Text(INITIAL_STR, is_action=False)]
    reward: List[float] = [0.0]
    done: bool = True

    for line in conversation["lines"]:
        question, answer = split_question_answer(line)
        # The agent is the asker
        text_history.append(Text(question + "\n", is_action=True))
        reward.append(-1.0)
        # the environment is the oracle
        text_history.append(Text(answer + "\n", is_action=False))
        reward.append(0.0)

    # if the conversation stopped early, but not due to guessing the word, then the trajectory is only partial
    if len(conversation["lines"]) < max_conversation_len and not conversation["correct"]:
        done = False

    # if word is guessed, last reward should be 0
    if conversation["correct"]:
        reward[-2] = 0.0

    return TextTrajectory(tuple(text_history), tuple(reward), done)


def create_trajectories_from_conversations(
    conversations: List[Conversation], 
    max_conversation_len: int=20, 
    recompute_done: bool=False,
) -> List[TextTrajectory]:
    """
    conversations: raw data from the json file, which is a list of conversations.
        See `create_trajectory_from_conversation`
    max_conversation_len: How many lines of conversation? Default=20, because the task is called twenty questions.
    recompute_done: If True, recompute if a word is correctly guessed for each conversation.
    """
    return [
        create_trajectory_from_conversation(conversation, max_conversation_len=max_conversation_len, recompute_done=recompute_done) 
        for conversation in conversations
    ]


def conversation_to_str(conversation: Conversation) -> str:
    str_lines = [f"word: {conversation['word']}"]
    for line in conversation["lines"]:
        str_lines.append(line)
    str_lines.append(f"correct: {conversation['correct']}")
    return "\n".join(str_lines)


def create_conversation_from_history(
    word_var: WordVariants,
    text_history: TextHistory,
    max_conversation_len: int=20,
) -> Conversation:
    assert len(text_history) % 2 == 1, "TextHistory should be [empty str, question1, answer1, ..., questionN, answerN]."

    question_texts = text_history[1::2]
    answer_texts = text_history[2::2]
    assert all(question_text.is_action for question_text in question_texts), "All questions should be actions."
    assert all(not answer_text.is_action for answer_text in answer_texts), "All answers should not be actions."
    assert len(question_texts) == len(answer_texts)
    
    # subtract 1 because of the starting text, then text_history contains pairs of questions and answers
    conversation_len = (len(text_history) - 1) // 2
    assert conversation_len <= max_conversation_len, f"Conversation is too long {conversation_len}. Max should be {max_conversation_len}."

    lines = []
    for question_text, answer_text in zip(question_texts, answer_texts):
        question = question_text.text.strip()
        answer = answer_text.text.strip()
        lines.append(question + " " + answer)

    correct = answer_texts[-1].text.strip() == "Yes." and is_done(word_var, question_texts[-1].text.strip())
    word = word_var.json()

    return {
        "lines": lines,
        "correct": correct,
        "word": word,
    }


def create_conversations_from_histories(
    word_vars: List[WordVariants],
    text_histories: List[TextHistory], 
    max_conversation_len: int=20, 
) -> List[Conversation]:
    """
    word_vars: List of WordVariants for each TextHistory.
    text_histories: List of TextHistories.
    max_conversation_len: How many lines of conversation? Default=20, because the task is called twenty questions.
    """
    assert len(word_vars) == len(text_histories)
    print(word_vars)
    return [
        create_conversation_from_history(word_var, text_history, max_conversation_len=max_conversation_len) 
        for word_var, text_history in zip(word_vars, text_histories)
    ]


RTG_TOKEN_STR_PATTERN = re.compile(r"<\|rtg=(-?\d+)\|>")


def rtg_to_token_str(rtg: Union[float, int], max_conversation_len: int=20) -> str:
    if isinstance(rtg, float):
        rtg = round(rtg)
    if rtg > 0:
        rtg = 0
    if rtg < -max_conversation_len:
        rtg = -max_conversation_len
    return f"<|rtg={rtg}|>"


def token_str_to_rtg(x: str, max_conversation_len: int=20) -> float:
    """Matches the value of the rtg token at the start of the input string."""
    rtg_match = re.match(RTG_TOKEN_STR_PATTERN, x)
    assert rtg_match is not None, f"{x} does not contain an rtg token at the start."
    rtg = float(rtg_match.group(1))
    if rtg > 0:
        rtg = 0.0
    if rtg < -max_conversation_len:
        rtg = -float(max_conversation_len)
    return rtg


def get_rtg_token_strs(max_conversation_len: int=20) -> List[str]:
    return [rtg_to_token_str(rtg, max_conversation_len=max_conversation_len) for rtg in range(-max_conversation_len, 1)]


def create_dt_text_history_from_trajectory(text_trajectory: TextTrajectory, max_conversation_len: int=20) -> TextHistory:
    """Create a text history with the DT rtgs included."""
    rtg = sum(text_trajectory.reward)
    new_text_history: TextHistory = []
    for (text, r) in zip(text_trajectory.text_history, text_trajectory.reward):
        if text.is_action:
            rtg -= r
            new_text_history.append(text)
        else:
            new_text = Text(rtg_to_token_str(rtg, max_conversation_len=max_conversation_len) + text.text, is_action=False)
            new_text_history.append(new_text)

    assert math.isclose(rtg, 0.0)

    return tuple(new_text_history)


def asker_postproc(question: str) -> str:
    # print(f"asker_postproc: {repr(question)}")
    question = question.strip()
    # empty question
    if len(question) == 0:
        return INVALID_QUESTION
    
    if question[-1] != "?":
        question += "?"
    question = question[0].upper() + question[1:]

    # typically many duplicate words
    if len(question.split(" ")) > 40:
        return INVALID_QUESTION
    
    # non-questions
    if question.split(" ")[0] not in ["Is", "Does", "Can", "Do", "Are", "Could"]:
        return INVALID_QUESTION
    
    # non-questions again
    if question[-2] == "." and question.split(" ")[-1] != "etc.?":
        return INVALID_QUESTION

    return question + "\n"


def asker_postproc_simple(question: str) -> str:
    # print(f"asker_postproc_simple: {repr(question)}")
    question = question.strip()

    # empty question
    if len(question) == 0:
        return "?\n"
    
    if question[-1] != "?":
        question += "?"

    return question + "\n"


def asker_postproc_filter_repeats(question: str) -> str:
    # print(f"asker_postproc_simple: {repr(question)}")
    question = question.strip()

    # empty question
    if len(question) == 0:
        return "?\n"
    
    # usually repeatting words
    question_words = question.split(" ")
    if len(question_words) > 50:
        question = " ".join(question_words[:50])

    if question[-1] != "?":
        question += "?"

    return question + "\n"


def is_done(word_var: WordVariants, question: str):
    # cut out punctuations at the end
    while len(question) > 0 and not question[-1].isalpha():
        question = question[:-1]

    if len(question) == 0:
        return False

    question_pos = nltk.pos_tag(nltk.word_tokenize(question.lower()))
    
    # ignore these nouns when checking for extra words
    ignores = {"object", "something", "type", "kind"}
    for pos_list in word_var.pos_tags:
        for w, _ in pos_list:
            ignores.add(w)

    # check for extra words
    for q_i in range(len(question_pos)):
        q_i_word, q_i_pos = question_pos[q_i]
        # check if the current word is a noun that shouldn't be ignored
        if q_i_pos[:2] == "NN" and q_i_word not in ignores:
            # if it's a counter word that comes before "of", also ignore it
            if q_i + 1 < len(question_pos) and question_pos[q_i + 1][0] == "of":
                continue
            # extra word found
            return False

    # check for the actual word at the end of the question
    for word_pos in word_var.pos_tags:
        if len(word_pos) > len(question_pos):
            continue
        
        all_same = True
        for (var_i_word, _), (q_i_word, _) in zip(word_pos, question_pos[-len(word_pos):]):
            if var_i_word != q_i_word:
                all_same = False
                break
        if all_same:
            return True
    
    return False
