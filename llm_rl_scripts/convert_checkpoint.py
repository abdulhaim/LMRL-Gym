from llm_rl_scripts.guess_city.env.oracle import T5Oracle, get_t5_oracle_prompt
from JaxSeq.models.T5.load import load_params as t5_load_params, ModelLoadMode as T5ModelLoadMode
from JaxSeq.utils import jsonl_stream, convert_path, load_mesh, get_dtype
from IPython import embed
from transformers import T5ForConditionalGeneration
from collections import OrderedDict
import json
from JaxSeq.bucket_manager import open_with_bucket as open
from transformers import T5Config

def convert_checkpoint(
    model_load_mode: T5ModelLoadMode,
    model_load_path: str,
):
    mesh = load_mesh((1, 1, -1), ('dp', 'fsdp', 'mp'))
    params, model = t5_load_params(
        model_load_mode=model_load_mode,
        model_load_path=model_load_path if model_load_mode != T5ModelLoadMode.HF else model_load_path,
        tokenizer="google/flan-t5-xl",
        mesh=mesh,
        model_dtype=get_dtype(use_fp16=True),
        params_dtype=get_dtype(use_fp16=True)
    )
    config_file = model_load_path + "/config.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    t5_config = T5Config(**config)
    embed()
    t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", config=t5_config)
    state_dict_keys = t5_model.state_dict().keys()

    new_state_dict = convert_params_to_state_dict(params, t5_model, state_dict_keys)
    print(new_state_dict.keys())
    
    embed()
    new_t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", state_dict=new_state_dict, config=t5_config)
    
    

model_load_mode = T5ModelLoadMode.PARAMS
model_load_path = "gcs://rl-llm-bench-dataset-internal/guess-my-city/simulator/model"

def convert_params_to_state_dict(params, t5_model, state_dict_keys):
    new_state_dict = OrderedDict()
    for key in state_dict_keys:
        if key == "shared.weight": # slightly different
            new_key = "shared.embedding"
        elif key == "lm_head.weight":
            new_key = "lm_head.kernel"
        levels = new_key.split(".")
        curr_params = params
        for level in levels:
            curr_params = curr_params[level]
        new_state_dict[key] = curr_params
        # assert new_state_dict[key].shape == t5_model.state_dict()[key].shape
    return new_state_dict


convert_checkpoint(model_load_mode, model_load_path)
