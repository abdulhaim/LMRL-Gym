from typing import List, Tuple, Union
import contextlib
from typing import Optional
import jax
import numpy as np
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as PS
from functools import partial
import os
import tyro
from transformers import AutoTokenizer

from jax_models.gpt2 import load_gpt2_model

from transformers.modeling_flax_utils import FlaxPreTrainedModel
from jaxtyping import PyTree
import re
import msgpack
from jaxtyping import PyTree
import flax
from flax.serialization import to_bytes
import jax
from typing import Optional


# Adapted from: https://github.com/young-geng/EasyLM/blob/main/EasyLM/checkpoint.py

""" 
Custom msgpack checkpointer that saves large train states by serializing
    and saving tensors one by one in a streaming fashion. Avoids running
    out of memory or local TPU disk with default flax checkpointer. The
        checkpointer saves the train state in an asynchronous manner to avoid
    timing out on JAX barriers in multi-host training.
"""

def save_pytree(
    open,
    tree: PyTree, 
    path: str, 
) -> None:
    tree = flax.serialization.to_state_dict(tree)
    packer = msgpack.Packer()
    flattend_tree = flax.traverse_util.flatten_dict(tree, keep_empty_nodes=True)
    with open(path, 'wb') as f:
        for key, value in flattend_tree.items():
            f.write(packer.pack((key, to_bytes(jax.device_get(value)))))


def main(
    model_name: str="gpt2-medium",
    model_path: str="gcs://rail-tpus-charles-3/ILQL5/outputs/twenty_questions/bc_gpt2med_test8/shard_0/model",
    save_dir: str="gcs://rail-tpus-charles-3/ILQL5/outputs/twenty_questions/bc_gpt2med_test8_converted",

    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[str]=None, 

):
    if gcloud_project is not None and gcloud_token is None:
        gcloud_token = os.path.join(os.path.expanduser('~'), f'.config/gcloud/{gcloud_project}.json')

    from utils.gcs_manager import open_pp as open
    open = partial(open, gcloud_project=gcloud_project, gcloud_token=gcloud_token)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("set pad_token")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    model, params, shard_rules = load_gpt2_model(
        model_str=model_name, 
        from_pretrained=True, 
        checkpoint_path=model_path, 
        use_fp16=jax.default_backend() == 'tpu', 
        tokenizer=tokenizer, 
        gradient_checkpoint=True, 
        seed=0, 
        gcloud_project=gcloud_project, 
        gcloud_token=gcloud_token, 
    )

    import pdb
    pdb.set_trace()

    with open(os.path.join(save_dir, 'model', 'config.json'), 'w') as f:
        f.write(model.config.to_json_string())

    save_pytree(
        open=open,
        tree=params, 
        path=os.path.join(save_dir, 'model', 'params.msgpack'), 
    )

if __name__ == "__main__":
    tyro.cli(main)