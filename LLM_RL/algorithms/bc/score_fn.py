from typing import List, Optional
from algorithms.jax_bc.core import BCInference
from environment import TextHistory
from transformers.tokenization_utils import PreTrainedTokenizer
import jax.numpy as jnp
import numpy as np
from jax_utils.data import block_sequences
from token_history import text_history_to_token_history
import jax

def build_bc_score_fn(
    inference: BCInference, 
    tokenizer: PreTrainedTokenizer, 
    max_length: int, 
    bsize: int, 
):
    
    def score_fn(text_histories: List[TextHistory]) -> List[float]:
        assert all([text_history[-1].is_action for text_history in text_histories])

        prev_token_histories = []
        token_histories = []
        for text_history in text_histories:
            prev_token_histories.append(text_history_to_token_history(text_history[:-1], tokenizer))
            token_histories.append(text_history_to_token_history(text_history, tokenizer))
        
        # truncate to end and pad tokens
        tokens = np.stack([np.concatenate((token_history.tokens[-max_length:], np.full((max_length-min(token_history.tokens.shape[0], max_length),), tokenizer.pad_token_id)), axis=0) for token_history in token_histories], axis=0)
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        
        all_logprobs = []
        
        for i in range(0, len(text_histories), bsize):
            batch = tokens[i:i+bsize, :]

            prefix_len = jnp.asarray([prev_token_histories[i+x].tokens.shape[0] for x in range(batch.shape[0])], dtype=jnp.int32)
            attention_mask = (batch != tokenizer.pad_token_id).astype(np.float32)

            action_logprobs = jnp.empty(prefix_len.shape, dtype=jnp.float32)

            logprobs = jax.nn.log_softmax(inference.get_logits_from_tokens(batch), axis=-1)
            action_logits = jnp.take_along_axis(logprobs[:, :-1], batch[:, 1:][..., None], axis=2).squeeze(2)
            for x in range(len(prefix_len)):
                action_logprobs = action_logprobs.at[x].set((action_logits[x] * attention_mask[x, 1:])[(prefix_len[x]-1):].sum(axis=0))

            all_logprobs.extend(jax.device_get(action_logprobs).tolist())
        
        return all_logprobs

    return score_fn