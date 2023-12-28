from pickle import UnpicklingError
from charset_normalizer import from_bytes
from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model, load_flax_weights_in_pytorch_model
from transformers import T5ForConditionalGeneration, FlaxT5ForConditionalGeneration
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.models.T5.load import load_params as t5_load_params, ModelLoadMode as T5ModelLoadMode
from JaxSeq.utils import jsonl_stream, convert_path, load_mesh, get_dtype

pt_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", 
                                                    #   vocab_size=32768, 
                                                    #   ignore_mismatched_sizes=True
                                                      )
flax_checkpoint_path = "gcs://rl-llm-bench-dataset-internal/guess-my-city/simulator/model/"
# import correct flax class
# flax_cls = getattr(transformers, "Flax" + model.__class__.__name__)
params, model = t5_load_params(
        model_load_mode=T5ModelLoadMode.PARAMS,
        model_load_path=flax_checkpoint_path,
        tokenizer="google/flan-t5-xl",
        mesh=load_mesh((1, 1, -1), ('dp', 'fsdp', 'mp')),
        model_dtype=get_dtype(use_fp16=True),
        params_dtype=get_dtype(use_fp16=True)
    )

load_flax_weights_in_pytorch_model(pt_model, params)
# loaded_model = load_flax_checkpoint_in_pytorch_model(model, "gcs://rl-llm-bench-dataset-internal/guess-my-city/simulator/model/")
