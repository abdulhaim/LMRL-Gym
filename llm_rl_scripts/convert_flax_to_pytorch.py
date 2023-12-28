from pickle import UnpicklingError
from charset_normalizer import from_bytes
from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model, load_flax_weights_in_pytorch_model
from transformers import T5ForConditionalGeneration, FlaxT5ForConditionalGeneration
from JaxSeq.bucket_manager import open_with_bucket as open

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
flax_checkpoint_path = "gcs://rl-llm-bench-dataset-internal/guess-my-city/simulator/model/"
# import correct flax class
# flax_cls = getattr(transformers, "Flax" + model.__class__.__name__)

# load flax weight dict
with open(flax_checkpoint_path, "rb") as state_f:
    try:
        flax_state_dict = from_bytes(FlaxT5ForConditionalGeneration, state_f.read())
    except UnpicklingError:
        raise EnvironmentError(f"Unable to convert {flax_checkpoint_path} to Flax deserializable object. ")

load_flax_weights_in_pytorch_model(model, flax_state_dict)
# loaded_model = load_flax_checkpoint_in_pytorch_model(model, "gcs://rl-llm-bench-dataset-internal/guess-my-city/simulator/model/")
