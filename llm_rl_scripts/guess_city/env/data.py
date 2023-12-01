from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import re
from LLM_RL.environment import Text, TextHistory, TextTrajectory
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


INVALID_QUESTION = "Is this a valid question?\n"
INITIAL_STR = "Questions:\n"


def get_default_word_list() -> List[WordVariants]:
    city_names = {}
    word_list = []
    with open('global_cities.txt') as f:
        lines = f.readlines()
        for i in range(2, len(lines)):
            line = lines[i]
            line_list = line.split(";")
            print(line_list)
            population = line_list[3][:-1]
            country = line_list[2].capitalize()[1:]
            city = line_list[1]
            city = city.capitalize()[1:]
            word_list.append(city + "," + country)

    retun word_list

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

