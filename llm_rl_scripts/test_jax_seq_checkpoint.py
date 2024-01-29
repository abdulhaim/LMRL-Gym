from llm_rl_scripts.guess_city.env.oracle import T5Oracle, get_t5_oracle_prompt
from IPython import embed
from JaxSeq.models.T5.load import ModelLoadMode as T5ModelLoadMode
from JaxSeq.utils import load_mesh
import jax

words = ["sydney, australia"]
questions = ["What is the traditional cuisine like where you are from?"]

oracle = T5Oracle.load_oracle(
    mesh=load_mesh((1, 1, -1), ('dp', 'fsdp', 'mp')),
    prng_key=jax.random.PRNGKey(0),
    model_load_mode=T5ModelLoadMode.PARAMS,
    model_load_path="gcs://rl-llm-bench-dataset-internal/guess-my-city/simulator/model",
    max_output_length=128,
    )
embed()
print(oracle.generate_answer(words, questions, return_full=True))