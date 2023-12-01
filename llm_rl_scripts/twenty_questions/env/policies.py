import jax
import openai
import os
from llm_rl_scripts.twenty_questions.env.env import TwentyQuestionsPolicyEnvironment
from llm_rl_scripts.twenty_questions.env.env import TwentyQuestionsPolicyEnvironment
from llm_rl_scripts.twenty_questions.env.oracle import T5Oracle
from llm_rl_scripts.twenty_questions.env.oracle import T5ModelLoadMode as T5OracleModelLoadMode
from llm_rl_scripts.twenty_questions.env.data import create_trajectories_from_conversations, asker_postproc, asker_postproc_simple, asker_postproc_filter_repeats, get_default_word_list
from LLM_RL.environment import TextPolicy, TextHistory, Text
from JaxSeq.utils import load_mesh
import tiktoken

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = "You are an expert player for 20 questions. You only respond in json."

MAIN_PROMPT = """\
Welcome to the game of Twenty Questions! Your objective is to guess what the object is within twenty questions. At every turn, you will have the oppurnity to ask a yes/no question, and receive an answer from the oracle. You can ask tweny questions but must ask as few questions as possible. 
```
""".strip()

TOKENIZER = tiktoken.encoding_for_model("gpt-4")
INPUT_TOKEN_COUNT = 0
OUTPUT_TOKEN_COUNT = 0

def compute_cost(input_token_count: int, output_token_count: int) -> float:
    return ((0.03 * input_token_count) / 1000) + ((0.06 * output_token_count) / 1000)


def text_history_to_str(text_history: TextHistory) -> str:
    return '\n'.join(map(lambda x: x.text, text_history))

class GPT4TwentyQuestionsPolicy(TextPolicy):
    def __init__(self, prompt):
        self.prompt = prompt

    def act(self, text_history: TextHistory) -> TextHistory:
        global INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT
        content = ""
        prompt = self.prompt.replace('{{game_content}}', game_content)
        INPUT_TOKEN_COUNT += len(TOKENIZER.encode(prompt))
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
            except openai.OpenAIError as e:
                print(e)
                time.sleep(10)
                continue
            break
        response_text = response.choices[0].message.content
        OUTPUT_TOKEN_COUNT += len(TOKENIZER.encode(response_text))
        response_json = json.loads(response_text)
        print(f"total cost: {compute_cost(INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT)}; total input tokens: {INPUT_TOKEN_COUNT}; total output tokens: {OUTPUT_TOKEN_COUNT}")
        return text_history+(Text(response_json['guess'].strip(), True),)

class UserPolicy(TextPolicy):
    def __init__(self):
        super().__init__()

    def act(self, text_history: TextHistory) -> TextHistory:
        print(text_history_to_str(text_history))
        
        result = input("Enter your question: ").strip()
        result+="\n"
        return text_history+(Text(result, True),)

def twenty_questions_env(data_mesh_shape: int=1, fsdp_mesh_shape: int=1, model_mesh_shape: int=-1):
    oracle_model_path = "gcs://rail-tpus-charles-3/JaxSeq/outputs/twenty_questions/flan-t5-xl_oracle_lr1e-5_test1_converted/model"
    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))

    model_prng_key = jax.random.PRNGKey(2)
    policy_prng, oracle_prng = jax.random.split(model_prng_key)

    env = TwentyQuestionsPolicyEnvironment(
        oracle=T5Oracle.load_oracle(
            mesh=mesh,
            prng_key=oracle_prng,
            model_load_mode=T5OracleModelLoadMode.PARAMS,
            model_load_path=oracle_model_path,
            use_fp16_activations=False,
            use_fp16_params=False,
            max_input_length=124,
            max_output_length=4,
        ),
        word_list=get_default_word_list(),
        max_conversation_length=20,
    )
    return env

