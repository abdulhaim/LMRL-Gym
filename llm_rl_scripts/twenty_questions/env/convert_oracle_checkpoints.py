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
from transformers import T5Tokenizer
from jax_utils.jax_shard import shard_params
from jax_models.t5 import load_t5_model
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
    save_dir: str="gcs://rail-tpus-charles-3/JaxSeq/outputs/twenty_questions/flan-t5-xl_oracle_lr1e-5_test1_converted",
    eval_env_oracle_path: str="gcs://rail-tpus-charles-3/JaxSeq/outputs/twenty_questions/flan-t5-xl_oracle_lr1e-5_test1/model_2.pkl",

    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[str]=None, 

):
    if gcloud_project is not None and gcloud_token is None:
        gcloud_token = os.path.join(os.path.expanduser('~'), f'.config/gcloud/{gcloud_project}.json')

    from utils.gcs_manager import open_pp as open
    open = partial(open, gcloud_project=gcloud_project, gcloud_token=gcloud_token)

    # mesh definition
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model, params, shard_rules = load_t5_model(
        model_str="google/flan-t5-xl", 
        from_pretrained=True, 
        checkpoint_path=eval_env_oracle_path, 
        use_fp16=jax.default_backend() == 'tpu', 
        tokenizer=tokenizer, 
        gradient_checkpoint=False, 
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