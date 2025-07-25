#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Common types."""
import enum
from typing import Any, Sequence

import numpy as np

import jax.numpy as jnp

Config = Any

Array = jnp.ndarray
PRNGKey = jnp.ndarray
DType = jnp.dtype
Shape = Sequence[int]

AxisNames = tuple[str, ...]
AxisIdxes = tuple[int, ...]

BATCH = "activation_batch"
LENGTH = "activation_length"
PREFILL_LENGTH = "prefill_activation_length"
Q_LENGTH = "activation_q_length"
KV_LENGTH = "activation_kv_length"
EMBED = "activation_embed"
HEAD = "activation_heads"
PREFILL_KV_BATCH = "activation_prefill_kv_batch"
KV_BATCH = "activation_kv_batch"
KV_HEAD = "activation_kv_heads"
KV_HEAD_DIM = "activation_kv_head_dim"
D_KV = "activation_kv"
DECODE_BATCH = "decode_batch"
DECODE_LENGTH = "decode_length"
CACHE_BATCH_PREFILL = "cache_batch_prefill"
CACHE_BATCH = "cache_batch"
CACHE_SEQUENCE = "cache_sequence"
CACHE_HEADS = "cache_heads"
CACHE_HEADS_NONE = "cache_heads_none"
CACHE_KV = "cache_kv"
CACHE_SCALE_BATCH = "cache_scale_batch"
CACHE_SCALE_SEQUENCE = "cache_scale_sequence"
CACHE_SCALE_HEADS = "cache_scale_heads"
CACHE_SCALE_KV = "cache_scale_kv"

MODEL_MODE_AUTOREGRESSIVE = "autoregressive"
MODEL_MODE_PREFILL = "prefill"
MODEL_MODE_TRAIN = "train"
MODEL_MODE_INSERT = "insert"

DECODING_ACTIVE_SEQUENCE_INDICATOR = 1

# A large negative mask value is used for masking to ensure that the
# softmax function assigns an extremely low probability to the masked positions.
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


class DecoderBlockType(enum.Enum):
  """Decoder block types."""

  DEFAULT = "default"
  LLAMA2 = "llama2"
  MISTRAL = "mistral"
  MIXTRAL = "mixtral"
  DEEPSEEK = "deepseek"
  GEMMA = "gemma"
  GEMMA2 = "gemma2"
  GEMMA3 = "gemma3"
  GPT3 = "gpt3"
  SIMPLE = "simple"
  SIMPLE_MLP = "simple_mlp"
  LLAMA4 = "llama4"
