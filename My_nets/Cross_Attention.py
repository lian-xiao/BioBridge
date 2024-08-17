import math

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from transformers.activations import ACT2FN

from MolFormer.rotate_attention.rotary import RotaryEmbedding, apply_rotary_pos_emb, apply_rotary_pos_emb_single
from transformers.modeling_utils import apply_chunking_to_forward


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = weight_norm(nn.Linear(config.hidden_size, config.intermediate_size))
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = weight_norm(nn.Linear(config.intermediate_size, config.hidden_size))
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = weight_norm(nn.Linear(config.hidden_size, self.all_head_size))
        self.key = weight_norm(nn.Linear(config.encoder_width, self.all_head_size))
        self.value = weight_norm(nn.Linear(config.encoder_width, self.all_head_size))
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.rotaryemb = RotaryEmbedding(self.num_attention_heads)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.intermediate = BertIntermediate(config)
        self.seq_len_dim = 1
        self.output = BertOutput(config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states,
                encoder_hidden_states,
                encoder_attention_mask=None,
                output_attentions=False,
                pos_emd=False):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        if pos_emd:
            cos, sin = self.rotaryemb(mixed_query_layer)
            query_layer = apply_rotary_pos_emb_single(query_layer.permute(0, 2, 3, 1), cos, sin)
            cos, sin = self.rotaryemb(key_layer, 2)
            key_layer = apply_rotary_pos_emb_single(key_layer.permute(0, 2, 3, 1), cos, sin)
            query_layer = query_layer.permute(0, 3, 1, 2)
            key_layer = key_layer.permute(0, 3, 1, 2)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs_dropped = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        attention_output = outputs[0]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        outputs = outputs + (past_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
