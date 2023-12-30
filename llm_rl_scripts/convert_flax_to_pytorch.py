from pickle import UnpicklingError
from charset_normalizer import from_bytes
from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model, load_flax_weights_in_pytorch_model
from transformers import T5ForConditionalGeneration, FlaxT5ForConditionalGeneration
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.models.T5.load import load_params as t5_load_params, ModelLoadMode as T5ModelLoadMode
from JaxSeq.utils import jsonl_stream, convert_path, load_mesh, get_dtype
import jax
from IPython import embed
import os

def load_t5_pytorch_model(flax_checkpoint_path):
    pt_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", architectures=["T5ForConditionalGeneration"])

    params, model = t5_load_params(
            model_load_mode=T5ModelLoadMode.PARAMS,
            model_load_path=flax_checkpoint_path,
            tokenizer="google/flan-t5-xl",
            mesh=load_mesh((1, 1, -1), ('dp', 'fsdp', 'mp')),
            model_dtype=get_dtype(use_fp16=True),
            params_dtype=get_dtype(use_fp16=True)
        )
    start_indices = (0, 0)
    params["lm_head"]["kernel"] = jax.lax.dynamic_slice(params["lm_head"]["kernel"], start_indices, pt_model.state_dict()["lm_head.weight"].T.shape)
    params["shared"]["embedding"] = jax.lax.dynamic_slice(params["shared"]["embedding"], start_indices, pt_model.state_dict()["shared.weight"].shape)

    print("encoder size: ", pt_model.state_dict()["encoder.embed_tokens.weight"].shape)
    print("decoder size: ", pt_model.state_dict()["decoder.embed_tokens.weight"].shape)

    print("shared embedding size: ", params["shared"]["embedding"].shape)
    print("lm_head kernel size: ", params["lm_head"]["kernel"].shape)

    new_model = load_flax_weights_in_pytorch_model(pt_model, params)

    new_model["encoder.embed_tokens.weight"] = new_model["shared.weight"].copy()
    new_model["decoder.embed_tokens.weight"] = new_model["shared.weight"].copy()
    return new_model

model_load_path = "gcs://rl-llm-bench-dataset-internal/guess-my-city/simulator/model/"
save_path = os.getcwd() + "/outputs/guess_city_pytorch_model.pt"
new_model = load_t5_pytorch_model(model_load_path)
embed()
new_model.save_pretrained(save_path)
# loaded_model = load_flax_checkpoint_in_pytorch_model(model, "gcs://rl-llm-bench-dataset-internal/guess-my-city/simulator/model/")
