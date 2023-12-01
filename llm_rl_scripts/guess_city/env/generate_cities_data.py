
import random
from typing import Any, Optional, List, Tuple, Callable
import tyro
import jax
import openai
import json
import time
from global_cities_dict import names_dict

key = "OPEN_AI_KEY"
openai.api_key = key

# Sampling Global Cities 
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

def get_gpt3_asker_prompt(word: str, conversation: List[str], maxlen=20, prob=0.1):
    convolen = len(conversation)

    prompt = "You are playing a game where you must guess where someone is from. You can ask 20 questions to determine where they are from. You should ask open ended questions that are diverse. You cannot ask them for the name of the city or country."
     
    prompt += "Each turn, you can ask an open ended question and the person will answer. You cannot ask them for the name of the city or country." #  If you ask a yes/no question, the person will give you a hint or further explanation. 


    if convolen == 0:
        prompt += "You are smart, so you will ask the question that will narrow down the possible cities the person is from as much as possible. "
        prompt +="Don't get stuck on one idea and try to branch out if you get stuck."
   
    elif convolen == maxlen - 1:
        prompt += "You have already asked {convolen} questions, so this is your final guess."

    else:
        prompt += "You have already asked {convolen} questions. "
        prompt += "You are smart, so you will ask the question that will narrow down the possible cities as much as possible. "
        prompt += "Don't get stuck on one idea and try to branch out if you get stuck."
        
    prompt += "\n\n"
    
    if convolen > 0:
        prompt += f"Here are the questions you've asked and their corresponding answers:\n"
        prompt += "\n".join(conversation)
        prompt += "\n\n"
        prompt += f"Here are the questions you've asked and their corresponding answers:\n"


    if len(conversation) == 0:
        prompt += "Generate the first question you will ask to determine the city the person is from."
    elif len(conversation) == maxlen - 1:
        prompt += "Based on what you know about the city so far, generate your final guess of what the city is. Only guess one city. \n\nIs the city"
    else:
        prompt += "Based on what you know about the city so far, generate the next question you will ask to determine the city. "
        if random.random() < prob:
            prompt += f"You think the city might be {word}, so ask a very general question relavent to {word} but don't directly say {word}."
        else:
            prompt += "If you think you know what the city is, you can directly guess the city."

    return prompt


def get_oracle_prompt(word: str, question: str,  conversation: List[str], maxlen=20):
    prompt = (
        f"""You are a question answering oracle. You will answer each question about a given city. You cannot reveal the name of the city or the name of the country in your answer except if the question correctly guesses the city. Here's a few examples:
        example 1:
        city: kuala lampur, malaysia.  
        question: What kind of food do you have in your city?
        answer: It is home to some of the best street food in the world, with dishes like nasi lemak and roti canai being popular among locals and tourists alike.
        example 2:
        city: dallas, usa
        question: What is the weather like in your city?
        answer: It depends on the season. In the winter, temperatures can drop below freezing, while in the summer temperatures can reach over 100 degrees Fahrenheit.
        example 3:
        city: lagos, nigeria
        question: Are there any bodies of water surrounding your city?
        answer: Yes, it is surrounded by several bodies of water including the Atlantic Ocean and a river.
        example 4:
        city: {word}
        question: {question}
        answer:""")

    return prompt

def load_gpt3_asker(
    max_output_length: int=128,
    temperature: float=0.7,
    guide_prob: float=0.0,
    max_conversation_len: int=20,
    verbose: bool=False,
) -> Callable[[str, List[str]], str]:

    def generate_question(word: str, conversation: List[str]) -> str:
        asker_prompt = get_gpt3_asker_prompt(word=word, conversation=conversation, maxlen=max_conversation_len, prob=guide_prob)

        try:
            prompt = [
            {"role": "system", "content": ""},
            {"role": "user", "content": asker_prompt},
            ]

            question = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages=prompt,
                                                    temperature=temperature,
                                                    max_tokens=2048)
        except (openai.APIError, openai.OpenAIError) as e:
            print(e)
            print("sleeping for 10 seconds.")
            time.sleep(10)
            return None

        time.sleep(0.5)
        question = question.choices[0].message.content
        if question[-1] != "?":
            question += "?"
        question = question.split("?")[0] + "?"

        if len(conversation) == max_conversation_len - 1:
            question = "Is the city you are from " + question
        else:
            question = question[0].upper() + question[1:]

        if verbose:
            print(question)
        
        return question

    print("Finished loading gpt3 asker.")
    return generate_question


def load_gpt3_oracle(
    max_output_length: int=128,
    temperature: float=0.7,
    guide_prob: float=0.0,
    max_conversation_len: int=20,
    verbose: bool=False,
) -> Callable[[str, List[str]], str]:

    def generate_answer(word: str, question: str, conversation: List[str]) -> str:
        oracle_prompt = get_oracle_prompt(word, question, conversation=conversation, maxlen=max_conversation_len)

        try:
            prompt = [
            {"role": "system", "content": ""},
            {"role": "user", "content": oracle_prompt},
            ]

            answer = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=prompt,
                                                  temperature=temperature,
                                                  max_tokens=2048)
    
        except (openai.APIError, openai.OpenAIError) as e:
            print(e)
            print("sleeping for 10 seconds.")
            time.sleep(10)
            return None

        answer = answer.choices[0].message.content

        print('answer: ', answer)
        if verbose:
            print(answer)

        return answer

    print("Finished loading gpt3 oracle!!!!")
    return generate_answer

def is_done(word_var, question):
    question = question.lower()
    word_var = word_var.lower()
    word_var = word_var.split(",")[0]
    find_word = question.find(word_var.lower())
    return find_word !=-1

def main(
    asker_model: str="gpt3",
    oracle_model: str="gpt3",

    asker_checkpoint_path: Optional[str]=None,
    oracle_checkpoint_path: Optional[str]=None,
    asker_gradient_checkpoint: bool=False,
    oracle_gradient_checkpoint: bool=False,

    do_pjit: bool=True,
    model_p_shape: int=4,
    data_p_shape: int=1,

    asker_max_input_length: int=512,
    asker_max_output_length: int=128,
    asker_guide_prob: float=0.0,
    asker_gpt3_temperature: float=0.7,

    oracle_max_input_length: int=384,
    oracle_max_output_length: int=128,
    
    num_samples: int=300,
    max_conversation_len: int=20,
    output_path: str="cities_data_",
    seed: int=5,

    verbose: bool=False,

    gcloud_project: Optional[str]="rail-tpus",
    gcloud_token: Optional[str]=None,
):
    
    # seed
    output_path += str(seed) + ".json"

    random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    # load askers
    if asker_model == "gpt3":
        generate_question = load_gpt3_asker(
            max_output_length=asker_max_output_length,
            temperature=asker_gpt3_temperature,
            guide_prob=asker_guide_prob,
            max_conversation_len=max_conversation_len,
            verbose=verbose,
        )
    elif asker_model == "google/flan-t5-xl":
        rng, new_rng = jax.random.split(rng)
        generate_question = load_flan_t5_xl_asker(
            mesh=mesh,
            rng=new_rng,
            model_name=asker_model,
            checkpoint_path=asker_checkpoint_path,
            max_input_length=asker_max_input_length,
            max_output_length=asker_max_output_length,
            gradient_checkpoint=asker_gradient_checkpoint,
            do_pjit=do_pjit,
            gcloud_project=gcloud_project,
            gcloud_token=gcloud_token,
            verbose=verbose,
        )
    else:
        raise NotImplementedError(f"Asker model {asker_model} is not implemented.")

    # load oracles
    if oracle_model == "google/flan-t5-xxl":
        rng, new_rng = jax.random.split(rng)
        generate_answer = load_flan_t5_xxl_oracle(
            mesh=mesh,
            rng=new_rng,
            model_name=oracle_model,
            checkpoint_path=oracle_checkpoint_path,
            max_input_length=oracle_max_input_length,
            max_output_length=oracle_max_output_length,
            gradient_checkpoint=oracle_gradient_checkpoint,
            do_pjit=do_pjit,
            gcloud_project=gcloud_project,
            gcloud_token=gcloud_token,
            verbose=verbose,
        )
    elif oracle_model == "gpt3":
        generate_answer = load_gpt3_oracle(
            max_output_length=asker_max_output_length,
            temperature=asker_gpt3_temperature,
            guide_prob=asker_guide_prob,
            max_conversation_len=max_conversation_len,
            verbose=verbose,
        )
    else:
        raise NotImplementedError(f"Oracle model {oracle_model} is not implemented.")

    print(f"total words: {len(word_list)}")


    print("==========================================================================")

    conversations = []

    convo_lens = []
    convo_dones = []

    for i in range(num_samples):
        word_var: WordVariants = random.choice(word_list)
        print("word", word_var)
        print()

        conversation = []
        done = False
        while len(conversation) < max_conversation_len and not done:
            # ask question
            question = generate_question(word=word_var, conversation=conversation)
            if question is None:
                continue

            # get answer
            print("question: ", question)
            answer = generate_answer(word=word_var, question=question, conversation=conversation)

            if answer is None:
                continue

            # post proc
            conversation.append(question + " " + answer)

            # check done
            if is_done(word_var, question):
                done = True
                break

        convo_lens.append(len(conversation))
        convo_dones.append(done)

        conversations.append({
            "conversation": conversation,
            "done": done,
            "word": word_var,
        })

        print()
        print(f"convo #{i+1}")
        print(f"word {word_var}")
        print(f"length {convo_lens}")
        print(f"done {convo_dones}")

        print(f"average length {sum(convo_lens) / len(convo_lens)}")
        print(f"num dones {sum(convo_dones)} / {len(convo_lens)}")

        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump(conversations, f, indent=4)

        print("===================================================================================================")
        

if __name__ == "__main__":
    tyro.cli(main)