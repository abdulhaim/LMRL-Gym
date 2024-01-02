from pickle import UnpicklingError
from charset_normalizer import from_bytes
from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model, load_flax_weights_in_pytorch_model
from transformers import T5ForConditionalGeneration, FlaxT5ForConditionalGeneration, AutoTokenizer
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.models.T5.load import load_params as t5_load_params, ModelLoadMode as T5ModelLoadMode
from JaxSeq.models.T5.interface import T5Inference
from JaxSeq.utils import jsonl_stream, convert_path, load_mesh, get_dtype
from JaxSeq.train import eval_loss
from IPython import embed
import os
import jax
import json
from torch.nn import CrossEntropyLoss
from JaxSeq.data import MaskIterableDataset
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, get_weight_decay_mask, MapIterable, FileOpenIterable
from transformers.generation import GenerationConfig

def load_t5_pytorch_model(params):
    pt_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", architectures=["T5ForConditionalGeneration"])

    # params, model = t5_load_params(
    #         model_load_mode=T5ModelLoadMode.PARAMS,
    #         model_load_path=flax_checkpoint_path,
    #         tokenizer="google/flan-t5-xl",
    #         mesh=load_mesh((1, 1, -1), ('dp', 'fsdp', 'mp')),
    #         model_dtype=get_dtype(use_fp16=True),
    #         params_dtype=get_dtype(use_fp16=True)
    #     )
    start_indices = (0, 0)
    params["lm_head"]["kernel"] = jax.lax.dynamic_slice(params["lm_head"]["kernel"], start_indices, pt_model.state_dict()["lm_head.weight"].T.shape)
    params["shared"]["embedding"] = jax.lax.dynamic_slice(params["shared"]["embedding"], start_indices, pt_model.state_dict()["shared.weight"].shape)

    print("encoder size: ", pt_model.state_dict()["encoder.embed_tokens.weight"].shape)
    print("decoder size: ", pt_model.state_dict()["decoder.embed_tokens.weight"].shape)

    print("shared embedding size: ", params["shared"]["embedding"].shape)
    print("lm_head kernel size: ", params["lm_head"]["kernel"].shape)

    new_model = load_flax_weights_in_pytorch_model(pt_model, params)

    new_model.state_dict()["encoder.embed_tokens.weight"] = new_model.state_dict()["shared.weight"]
    new_model.state_dict()["decoder.embed_tokens.weight"] = new_model.state_dict()["shared.weight"]
    return new_model

model_load_path = "gcs://rl-llm-bench-dataset-internal/guess-my-city/simulator/model/"
eval_data_path = "gcs://rl-llm-bench-dataset-internal/guess-my-city/eval.json"

def pytorch_model_generation(sentence_fragment, model: T5ForConditionalGeneration):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    input_ids = tokenizer(sentence_fragment, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=100, num_beams=1, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def flax_model_generation(sentence_fragment, model: FlaxT5ForConditionalGeneration, params):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    inference = T5Inference.load_inference(
            params=params, 
            model=model, 
            tokenizer=tokenizer, 
        )
    
    generation_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=100,
    )
    generation = inference.generate_from_str([sentence_fragment], 
                                             generation_config=generation_config,)
    return generation.output_strs


    # input_ids = tokenizer(sentence_fragment, return_tensors="np")["input_ids"]
    # outputs = model.generate(input_ids, max_length=100, num_beams=1, do_sample=False).sequences
    # return tokenizer.decode(outputs[0])

sentence_fragment = "I am from "

flax_params, flax_model = t5_load_params(
            model_load_mode=T5ModelLoadMode.PARAMS,
            model_load_path=model_load_path,
            tokenizer="google/flan-t5-xl",
            mesh=load_mesh((1, 1, -1), ('dp', 'fsdp', 'mp')),
            model_dtype=get_dtype(use_fp16=True),
            params_dtype=get_dtype(use_fp16=True)
        )

pt_model = load_t5_pytorch_model(flax_params)

embed()

pytorch_completion = pytorch_model_generation(sentence_fragment, pt_model)
flax_completion = flax_model_generation(sentence_fragment, flax_model)


# save_path = os.getcwd() + "/outputs/guess_city_pytorch_model.pt"
# new_model = load_t5_pytorch_model(model_load_path)
# new_model.save_pretrained(save_path)

def data_load_iterable(eval_data_path):
    with open(eval_data_path, "r") as f:
        eval_data = json.loads(f.read())
    for obj in eval_data:
        for phrase in obj["conversation"]:
            yield (phrase, 1.0)

iterable = data_load_iterable(eval_data_path)
for i in range(10):
    segment = next(iterable)
    print(segment[0])

def compute_jax_seq_eval_loss(eval_data_path, flax_checkpoint_path, bsize=8, eval_batches=4):
    params, model = t5_load_params(
            model_load_mode=T5ModelLoadMode.PARAMS,
            model_load_path=flax_checkpoint_path,
            tokenizer="google/flan-t5-xl",
            mesh=load_mesh((1, 1, -1), ('dp', 'fsdp', 'mp')),
            model_dtype=get_dtype(use_fp16=True),
            params_dtype=get_dtype(use_fp16=True)
        )
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

    iterable = data_load_iterable(eval_data_path)

    eval_data = MaskIterableDataset.blocked_from_str_segments_iterable(
        MapIterable(lambda x: x['sequence'], FileOpenIterable(eval_data_path, 'r', pipe=jsonl_stream)), 
        tokenizer, 
        blocking_strategy=BlockingStrategy(
            padding=Padding.RIGHT, 
            truncation=Truncation.LEFT, 
            max_length=2048 , 
        ), 
    )
    
    inference = T5Inference.load_inference(
            params=params, 
            model=model, 
            tokenizer=tokenizer, 
        )
    
    loss_metrics = eval_loss(
            inference=inference, 
            dataset=eval_data, 
            prng_key=None, 
            bsize=8, 
            eval_batches=4, 
        )
    return loss_metrics 

def compute_pytorch_eval_loss(eval_data_path, flax_checkpoint_path, num_examples=32):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    pt_model = load_t5_pytorch_model(flax_checkpoint_path)

    iterable = data_load_iterable(eval_data_path)
    total_loss = 0
    for i in range(num_examples):
        text = next(iterable)[0]
        loss = compute_loss(tokenizer, pt_model, text)
        total_loss += loss
    loss = total_loss / num_examples

    return loss

def compute_loss(tokenizer, model, text):
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Forward pass to get logits (predictions) from the model
    logits = model(input_ids).logits

    # Flatten logits and labels
    logits_flat = logits.view(-1, logits.shape[-1])
    labels_flat = input_ids.view(-1)

    # Compute cross-entropy loss
    criterion = CrossEntropyLoss()
    loss = criterion(logits_flat, labels_flat)

    return loss.item()

# jax_seq_eval_loss = compute_jax_seq_eval_loss(eval_data_path, model_load_path)
pytorch_eval_loss = compute_pytorch_eval_loss(eval_data_path, model_load_path)
# print("jax_seq_eval_loss: ", jax_seq_eval_loss)
print("pytorch_eval_loss: ", pytorch_eval_loss)