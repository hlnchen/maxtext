"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

from typing import Sequence
import datetime
import functools
import os
import queue
import sys
import time

from absl import app

import numpy as np

import pathwaysutils  # pylint: disable=unused-import

import tensorflow as tf

from jax import random
from jax.experimental import checkify
from jax.sharding import Mesh
import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

import orbax.checkpoint
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager

import grain.python as grain

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from MaxText import checkpointing
from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import optimizers
from MaxText import profiler
from MaxText import pyconfig
from MaxText.gcp_workload_monitor import GCPWorkloadMonitor
from MaxText.input_pipeline.input_pipeline_interface import create_data_iterator
from MaxText.layers import quantizations
from MaxText.layers.models import Transformer
from MaxText.metric_logger import MetricLogger
from MaxText.utils import gcs_utils
from MaxText.utils.goodput_utils import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
)
from MaxText.vertex_tensorboard import VertexTensorboardManager
# Placeholder: internal

# pylint: disable=too-many-positional-arguments

EPS = 1e-8
_DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3


def validate_train_config(config):
  """Validates the configuration is set correctly for 'train.py'."""

  assert config.run_name, "Erroring out, need a real run_name"
  if config.dataset_path and not config.dataset_path.startswith("gs://"):
    max_logging.log("WARNING: 'dataset_path' might be pointing your local file system")
  if not config.base_output_directory.startswith("gs://"):
    max_logging.log("WARNING: 'base_output_directory' might be pointing your local file system")
  assert config.steps > 0, "You must set steps or learning_rate_schedule_steps to a positive integer."

  if config.quantization in ("fp8", "nanoo_fp8"):
    # pylint: disable=line-too-long
    assert config.gradient_accumulation_steps == 1, (
        "fp8 can't be used with gradient_accumulation_steps right now. Please use other quantization or set "
        "gradient_accumulation_steps to 1"
    )

  # Check if GPU Flash Attention is being used with sequence packing
  if config.attention == "cudnn_flash_te" and config.packing and config.dataset_type != "synthetic":
    raise ValueError(
        "cudnn_flash_te only supports BSHD format. The THD (seq packing) support is going to be available in "
        "Transformer Engine 2.0 release. "
        "Please disable sequence packing (set packing=False) or use a different attention mechanism. "
        "With synthetic data, the format is not important as packing is not applied."
    )


def get_first_step(state):
  return int(state.step)


def load_next_batch(train_iter, example_batch, config):
  """Loads the next batch. Can keep reusing the same batch for performance reasons"""

  if config.reuse_example_batch and example_batch is not None:
    return example_batch
  else:
    return next(train_iter)


def record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr, per_device_tokens):
  """Records scalar metrics to be written to tensorboard"""
  metrics["scalar"].update({"perf/step_time_seconds": step_time_delta.total_seconds()})
  metrics["scalar"].update({"perf/per_device_tflops": per_device_tflops})
  metrics["scalar"].update({"perf/per_device_tflops_per_sec": per_device_tflops / step_time_delta.total_seconds()})
  metrics["scalar"].update({"perf/per_device_tokens": per_device_tokens})
  metrics["scalar"].update({"perf/per_device_tokens_per_sec": per_device_tokens / step_time_delta.total_seconds()})
  metrics["scalar"].update({"learning/current_learning_rate": lr})


def save_checkpoint(
    checkpoint_manager,
    step,
    state,
    dataset_type="tfds",
    data_iterator=None,
    config=None,
    force=False,
) -> bool:
  """Wrapper for saving checkpoint."""
  if config and config.enable_checkpointing:
    if (
        force
        or (step % config.checkpoint_period == 0)
        or (config.enable_emergency_checkpoint and step % config.local_checkpoint_period == 0)
    ):
      blocking_until_ready_start = time.time()
      max_logging.log(f"Waiting for step {step} to finish before checkpoint...")
      # We block here on the step finishing so that our checkpointing metrics
      # measure only checkpointing time, not training time.
      jax.block_until_ready(state)
      max_logging.log(
          f"Waited {time.time() - blocking_until_ready_start} seconds for step "
          f"{step} to finish before starting checkpointing."
      )

  # specify chunk_byte_size to force orbax to control maximum file size in checkpoint
  chunk_byte_size = _DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
  if config:
    chunk_byte_size = config.checkpoint_storage_target_data_file_size_bytes
  save_args = jax.tree.map(lambda _: orbax.checkpoint.SaveArgs(chunk_byte_size=chunk_byte_size), state)

  if isinstance(
      checkpoint_manager,
      (
          emergency_checkpoint_manager.CheckpointManager,
          emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager,
      ),
  ):
    checkpointing.replicator_error_handler(config)
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            state=orbax.checkpoint.args.PyTreeSave(
                item=state,
                save_args=save_args,
                ocdbt_target_data_file_size=chunk_byte_size,
            )
        ),
        force=force,
    )

  if dataset_type == "grain":
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            ),
            iter=grain.PyGrainCheckpointSave(data_iterator.local_iterator),
        ),
        force=force,
    )
  else:
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            )
        ),
        force=force,
    )


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------


def record_activation_metrics(output_metrics, intermediate_outputs, config):
  """Adds the activation metrics to the metrics dict"""

  if config.scan_layers:
    metrics_dict = intermediate_outputs["intermediates"]["decoder"]["decoder"]

    for layer_num in range(config.num_decoder_layers):
      output_metrics["scalar"][f"activ_fraction_zero/layer_{layer_num:03d}"] = metrics_dict["activation_fraction_zero"][0][
          layer_num
      ]
      output_metrics["scalar"][f"activ_mean/layer_{layer_num:03d}"] = metrics_dict["activation_mean"][0][layer_num]
      output_metrics["scalar"][f"activ_stdev/layer_{layer_num:03d}"] = metrics_dict["activation_stdev"][0][layer_num]
  else:
    for layer_num in range(config.num_decoder_layers):
      layer = intermediate_outputs["intermediates"]["decoder"][f"layers_{layer_num}"]
      output_metrics["scalar"][f"activ_fraction_zero/layer_{layer_num:03d}"] = layer["activation_fraction_zero"][0]
      output_metrics["scalar"][f"activ_mean/layer_{layer_num:03d}"] = layer["activation_mean"][0]
      output_metrics["scalar"][f"activ_stdev/layer_{layer_num:03d}"] = layer["activation_stdev"][0]


def _split_dpo_state(state):
  reference_params = state.params["reference_params"]
  new_state = state.replace(params={k: v for k, v in state.params.items() if k != "reference_params"})
  return new_state, reference_params


def _merge_dpo_state(state, reference_params):
  return state.replace(params=dict(state.params, reference_params=reference_params))


def dpo_loss_fn(model, config, data, dropout_rng, params, reference_params, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A nn.Module
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout
    params: Model params
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, total_loss, and total_weights
  """
  # inputs, targets, segments, positions = apply_args
  rng1, aqt_rng = jax.random.split(dropout_rng)

  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]

  # for DPO we don't support packed sequence (they shouldn't be present in the first place)
  data["chosen_segmentation"] = (data["chosen_segmentation"] == 1).astype(jnp.int32)
  data["rejected_segmentation"] = (data["rejected_segmentation"] == 1).astype(jnp.int32)
  data["chosen_position"] = data["chosen_position"] * (data["chosen_segmentation"] == 1)
  data["rejected_position"] = data["rejected_position"] * (data["rejected_segmentation"] == 1)

  # concatenated model and reference model forward pass
  inputs = jnp.concatenate([data["chosen"], data["rejected"]], 0)
  inputs_position = jnp.concatenate([data["chosen_position"], data["rejected_position"]], 0)
  inputs_segmentation = jnp.concatenate([data["chosen_segmentation"], data["rejected_segmentation"]], 0)

  logits, intermediate_outputs = model.apply(
      params,
      inputs,
      inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=config.enable_dropout if is_train else False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable="intermediates",
  )
  ref_logits = model.apply(
      {"params": reference_params},
      inputs,
      inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=False,
      rngs={"dropout": rng1, "params": aqt_rng},
  )
  ref_logits = jax.lax.stop_gradient(ref_logits)

  # extract token ids, segmentation and logits for chosen and rejected sequences
  chosen_ids = data["chosen"][..., 1:]
  rejected_ids = data["rejected"][..., 1:]
  chosen_segmentation = data["chosen_segmentation"][..., 1:]
  rejected_segmentation = data["rejected_segmentation"][..., 1:]
  n_logits = logits.shape[-3] // 2  # [B, S, E] - [batch, sequence, embedding/vocab]
  chosen_logits, rejected_logits = logits[:n_logits, :, :], logits[n_logits:, :, :]  # [B, S, E], [B, S, E]
  # ^ [B, S, E], [B, S, E]
  chosen_ref_logits, rejected_ref_logits = ref_logits[:n_logits, :, :], ref_logits[n_logits:, :, :]

  # common subsequence and padding mask
  common_prefix_mask = jnp.cumsum(chosen_ids != rejected_ids, axis=-1) == 0  # [B, S]
  valid_seq_mask = (chosen_segmentation != 0) & (rejected_segmentation != 0) & ~common_prefix_mask  # [B, S]

  # compute logratios from the sequence-reduced observed token log-probability
  chosen_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(chosen_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
  )[..., 0]
  chosen_logps = jnp.sum(chosen_logps_seq * valid_seq_mask, axis=-1)  # [B]
  chosen_ref_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(chosen_ref_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
  )[..., 0]
  chosen_ref_logps = jnp.sum(chosen_ref_logps_seq * valid_seq_mask, axis=-1)  # [B]
  chosen_logratios = chosen_logps - chosen_ref_logps  # [B]

  rejected_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(rejected_logits[..., :-1, :], axis=-1), rejected_ids[..., None], axis=-1
  )[..., 0]
  rejected_logps = jnp.sum(rejected_logps_seq * valid_seq_mask, axis=-1)  # [B]
  rejected_ref_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(rejected_ref_logits[..., :-1, :], axis=-1), rejected_ids[..., None], axis=-1
  )[..., 0]
  rejected_ref_logps = jnp.sum(rejected_ref_logps_seq * valid_seq_mask, axis=-1)  # [B]
  rejected_logratios = rejected_logps - rejected_ref_logps  # [B]

  # DPO loss from chosen and rejected logratios
  LABEL_SMOOTHING, BETA = config.dpo_label_smoothing, config.dpo_beta
  logratios_delta = BETA * (chosen_logratios - rejected_logratios)  # [B]
  losses = (  # [B]
      -jax.nn.log_sigmoid(BETA * logratios_delta) * (1 - LABEL_SMOOTHING)
      - jax.nn.log_sigmoid(-BETA * logratios_delta) * LABEL_SMOOTHING
  )
  total_loss, total_weights = jnp.mean(losses), losses.shape[0]
  loss = total_loss

  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss
  reward_accuracy = jnp.mean(chosen_logratios > rejected_logratios)
  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "reward_accuracy": reward_accuracy,
  }
  return loss, aux


def loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A nn.Module
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout
    params: Model params
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, total_loss, and total_weights
  """
  # inputs, targets, segments, positions = apply_args
  rng1, aqt_rng = jax.random.split(dropout_rng)

  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]
  else:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_eval_on, :]

  logits, intermediate_outputs = model.apply(
      params,
      data["inputs"],
      data["inputs_position"],
      decoder_segment_ids=data["inputs_segmentation"],
      encoder_images=data["images"] if config.use_multimodal else None,
      enable_dropout=config.enable_dropout if is_train else False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable="intermediates",
  )

  # N = 200
  # jax.debug.print("inputs: {x}", x=data["inputs"][0,:N])
  # # gather the logits with next token
  # _logits = logits[0,:N,:]
  # _next_token = data["inputs"][0,1:N+1]
  # print(f"logits.shape: {_logits.shape}, logits.dtype: {_logits.dtype}, logits[10,_next_token[10]]: {_logits[10,_next_token[10]].dtype}")
  # jax.debug.print("next_token: {x}", x=_next_token)
  # _logits_on_next_token = jnp.take_along_axis(_logits, _next_token[:, None], axis=-1).squeeze(-1)
  # # _logits_on_next_token_2 = _logits[jnp.arange(N), _next_token]
  # jax.debug.print("logits_on_next_token: {x}, {y}", x=_logits_on_next_token, y=_logits_on_next_token.dtype)
  # jax.debug.print("logits_on_next_token[9]: {x}, {y}", x=_logits[9,_next_token[9]], y=_logits_on_next_token[9])

  # logits_sum = jax.scipy.special.logsumexp(_logits, axis=-1, keepdims=False)
  # log_softmax = _logits_on_next_token - logits_sum
  # jax.debug.print("log softmax: {x}", x=log_softmax)

  # # raise RuntimeError("===========Stop here===========")


  one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
  xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
  # Mask out paddings at the end of each example.
  xent = xent * (data["targets_segmentation"] != 0)
  total_loss = jnp.sum(xent)
  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  loss = total_loss / (total_weights + EPS)
  # get moe load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss
  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
  }
  return loss, aux


def train_step(model, config, state_mesh_shardings, state, data, dropout_rng):
  """

  Args:
    model: A nn.Module
    state: A pytree of the current state of the model
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout

  Returns:
    new_state: Same format as state.
    metrics: Dictionary of model metrics such as loss, training rate, etc.
    rng2: A new rng key that can be used in future calls.

  """
  reference_params, reference_params_sharding, extra_dpo_args, _loss_fn = [], [], [], loss_fn
  if config.use_dpo:
    state, reference_params = _split_dpo_state(state)
    state_mesh_shardings, reference_params_sharding = _split_dpo_state(state_mesh_shardings)
    extra_dpo_args = [reference_params]
    _loss_fn = dpo_loss_fn

  if config.gradient_accumulation_steps > 1:

    def accumulate_gradient(acc_grad_and_loss, data):
      grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
      (_, aux), cur_batch_gradient = grad_func(
          model, config, data, dropout_rng, state.params, *extra_dpo_args, is_train=True
      )
      acc_grad_and_loss["loss"] += aux["total_loss"]
      acc_grad_and_loss["moe_lb_loss"] += aux["moe_lb_loss"]
      acc_grad_and_loss["grad"] = jax.tree_util.tree_map(
          lambda x, y: x * aux["total_weights"] + y, cur_batch_gradient, acc_grad_and_loss["grad"]
      )
      acc_grad_and_loss["total_weights"] += aux["total_weights"]
      return acc_grad_and_loss, aux

    def reshape_to_microbatch_accumulations(batch_arr):
      """Reshape global batch to microbatches, assuming batch axis is leading."""
      microbatches = config.gradient_accumulation_steps
      microbatch_shape = (microbatches, batch_arr.shape[0] // microbatches) + batch_arr.shape[1:]
      return jnp.reshape(batch_arr, microbatch_shape)

    data = jax.tree_util.tree_map(reshape_to_microbatch_accumulations, data)
    init_grad = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    init_grad_and_loss = {"loss": 0.0, "grad": init_grad, "total_weights": 0, "moe_lb_loss": 0.0}

    grad_and_loss, aux = jax.lax.scan(
        accumulate_gradient, init_grad_and_loss, data, length=config.gradient_accumulation_steps
    )
    loss = (
        grad_and_loss["loss"] / grad_and_loss["total_weights"]
        + grad_and_loss["moe_lb_loss"] / config.gradient_accumulation_steps
    )
    raw_grads = jax.tree_util.tree_map(lambda arr: arr / grad_and_loss["total_weights"], grad_and_loss["grad"])
    aux = jax.tree.map(lambda x: jnp.sum(x, axis=0), aux)  # pytype: disable=module-attr
  else:
    if config.optimizer_memory_host_offload:
      if config.use_dpo:
        reference_params = jax.device_put(reference_params, max_utils.with_memory_kind(reference_params_sharding, "device"))
        extra_dpo_args = [reference_params]
    grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
    (loss, aux), raw_grads = grad_func(model, config, data, dropout_rng, state.params, *extra_dpo_args, is_train=True)
  intermediate_outputs = aux["intermediate_outputs"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]

  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, state, config.gradient_clipping_threshold)
  else:
    grads = raw_grads
  if config.optimizer_memory_host_offload:
    state = state.replace(
        opt_state=jax.device_put(
            state.opt_state,
            jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="device"), state_mesh_shardings.opt_state),
        )
    )
  # Move all parameters to device before optimizer update
  if config.parameter_memory_host_offload:
    max_logging.log("\nMoving all parameters to device before optimizer update")

    def move(path, value):
      max_logging.log(f"train.py: Moving f{path} to device")
      return value.with_memory_kind(kind="device")

    state = state.replace(
        params=jax.device_put(
            state.params,
            jax.tree_util.tree_map_with_path(move, state_mesh_shardings.params),
        )
    )
  new_state = state.apply_gradients(grads=grads)

  scalar_metrics = {
      "learning/loss": loss,
      "learning/moe_lb_loss": moe_lb_loss,
      "learning/total_weights": total_weights,
  }
  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
    scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(new_state.params)
  if config.use_dpo:
    scalar_metrics["learning/dpo_reward_accuracy"] = aux["reward_accuracy"]
  metrics = {
      "scalar": scalar_metrics,
      "scalars": {},
  }

  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

  if config.use_dpo:
    new_state = _merge_dpo_state(new_state, reference_params)

  return new_state, metrics


def eval_step(model, config, state, data, dropout_rng):
  """eval_step no backprop and new state compared with train_step."""

  reference_params, extra_dpo_args, _loss_fn = [], [], loss_fn
  if config.use_dpo:
    state, reference_params = _split_dpo_state(state)
    extra_dpo_args = [reference_params]
    _loss_fn = dpo_loss_fn

  eval_loss_fn = functools.partial(_loss_fn, model, config, data, dropout_rng, is_train=False)
  loss, aux = eval_loss_fn(state.params, *extra_dpo_args)
  total_loss = aux["total_loss"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  metrics = {
      "scalar": {
          "evaluation/loss": loss,
          "evaluation/total_loss": total_loss,
          "evaluation/total_weights": total_weights,
          "evaluation/moe_lb_loss": moe_lb_loss,
      },
  }
  if config.use_dpo:
    metrics["scalar"]["evaluation/dpo_reward_accuracy"] = aux["reward_accuracy"]

  return metrics


def check_example_batch(config, example_batch):
  if config.max_checkify:
    jittable_f = checkify.checkify(lambda x: checkify.check(jnp.any(x > -1), "Batch contains bad synthetic data!"))
    # Check if inputs in batch contains bad synthetic data.
    # pylint: disable=not-callable
    err, _ = jax.jit(jittable_f)(example_batch["inputs"][: config.global_batch_size_to_train_on, :])
    err.throw()


def setup_mesh_and_model(config, devices=None):
  """Set up the mesh and the model for training

  Args:
    config
    devices

  Returns:
    init_rng: RNG key
    writer: Summary writer for tensorboard
    wandb_run: Weights & Biases run
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    tx:
  """

  init_rng = random.PRNGKey(config.init_weights_seed)
  if config.enable_wandb:
    wandb_run = max_utils.initialize_wandb_run(config)
    writer = None
  else:
    wandb_run = None
    writer = max_utils.initialize_summary_writer(config.tensorboard_dir, config.run_name)

  # Mesh definition
  devices_array = maxtext_utils.create_device_mesh(config, devices)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Model and Optimizer definition
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh, quant=quant)
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)
  logger = checkpointing.setup_checkpoint_logger(config)
  if config.enable_emergency_checkpoint:
    if config.use_replicator_service:
      checkpoint_manager = checkpointing.create_orbax_emergency_replicator_checkpoint_manager(
          config.local_checkpoint_directory,
          config.local_checkpoint_period,
          mesh,
      )
    else:
      abstract_state, _, _ = maxtext_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
      checkpoint_manager = checkpointing.create_orbax_emergency_checkpoint_manager(
          config.local_checkpoint_directory,
          config.checkpoint_dir,
          mesh,
          abstract_state,
          config.local_checkpoint_period,
          config.checkpoint_period,
          logger,
      )
  else:
    # TODO(b/368121306): Remove this once zarr3 support is plumbed on the backend
    use_ocdbt = config.checkpoint_storage_use_ocdbt
    use_zarr3 = config.checkpoint_storage_use_zarr3
    if config.enable_single_controller:
      use_ocdbt, use_zarr3 = False, False
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        config.checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
        config.dataset_type,
        logger,
        use_ocdbt,
        use_zarr3,
    )

  return init_rng, writer, wandb_run, checkpoint_manager, mesh, model, learning_rate_schedule, tx


def setup_train_loop(config, recorder):
  """Set up prerequisites for the training loop -
      checkpoint_manager, PRNG keys, Mesh, Model and optimizer.
      Set up data iterator and tokenizer, initialize the model.

  Args:
    config
    recorder

  Returns:
    init_rng:
    wandb_run: Weights & Biases run
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    data_iterator:
    state: the initialized train state
  """

  with maybe_record_goodput(recorder, GoodputEvent.TPU_INIT):
    init_rng, writer, wandb_run, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(config)

  with maybe_record_goodput(recorder, GoodputEvent.TRAINING_PREPARATION):
    data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
    context_parallel_size = mesh.shape["context"]
    # Check if context parallelism is being used with sequence packing
    if context_parallel_size > 1 and config.packing and config.dataset_type != "synthetic":
      raise ValueError(
          "Context parallelism cannot be used with sequence packing except for synthetic data where packing is not applied. "
          "Either disable sequence packing (set packing=False) or disable context parallelism. "
          "Context parallelism with packing support will be added soon."
      )

    # Apply reordering wrapper to data iterators if context parallelism is enabled
    with mesh:
      if context_parallel_size > 1 and config.context_parallel_load_balance:
        data_iterator = map(max_utils.get_reorder_callable(context_parallel_size), data_iterator)
        if eval_data_iterator:
          eval_data_iterator = map(max_utils.get_reorder_callable(context_parallel_size), eval_data_iterator)

    state, _, state_mesh_shardings, data_iterator = maxtext_utils.setup_training_state(
        model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
    )

    if not config.using_pipeline_parallelism:
      # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
      maxtext_utils.assert_params_sufficiently_sharded(state.params, mesh, config.sharding_tolerance)

    if config.use_dpo:
      abstract_state, _, _ = maxtext_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
      max_logging.log(f"Restoring reference parameters for DPO from '{os.path.join(str(config.checkpoint_dir), str(0))}'")
      try:
        step0_restored, _ = checkpointing.load_state_if_possible(
            checkpoint_manager,
            data_iterator,
            load_parameters_from_path="",
            load_full_state_from_path="",
            checkpoint_storage_concurrent_gb=config.checkpoint_storage_concurrent_gb,
            abstract_unboxed_pre_state=abstract_state,
            enable_single_replica_ckpt_restoring=False,
            dataset_type=config.dataset_type,
            step=0,
            use_ocdbt=config.checkpoint_storage_use_ocdbt,
            use_zarr3=config.checkpoint_storage_use_zarr3,
        )
      except FileNotFoundError:
        step0_restored = None
      if step0_restored is not None:
        reference_params = step0_restored["items"].params["params"]
        state = _merge_dpo_state(state, reference_params)
      else:
        max_logging.log(
            f"Could not restore reference parameters for DPO from '{os.path.join(str(config.checkpoint_dir), str(0))}'"
        )

  return (
      init_rng,
      writer,
      wandb_run,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  )


def train_loop(config, recorder, state=None):
  """Main Training loop."""
  (
      init_rng,
      writer,
      wandb_run,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  ) = setup_train_loop(config, recorder)

  if config.use_dpo:
    if "reference_params" not in state.params:
      reference_params = jax.tree.map(jnp.copy, state.params["params"])
      state = _merge_dpo_state(state, reference_params)
    state_mesh_shardings = _merge_dpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])

  # pylint: disable=line-too-long
  (
      functional_train,
      in_shard_train,
      out_shard_train,
      static_argnums_train,
      donate_argnums_train,
  ) = maxtext_utils.get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, config)

  if eval_data_iterator:
    # pylint: disable=line-too-long
    (
        functional_eval,
        in_shard_eval,
        out_shard_eval,
        static_argnums_eval,
        donate_argnums_eval,
    ) = maxtext_utils.get_functional_eval_with_signature(eval_step, mesh, state_mesh_shardings, model, config)

  num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
  per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config)
  per_device_tokens = maxtext_utils.calculate_tokens_training_per_device(config)

  # Write train config params, num model params, and XLA flags to logging systems
  if config.enable_tensorboard:
    max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], writer)
    maxtext_utils.add_config_to_summary_writer(config, writer)
  
  # if config.enable_wandb:
  #   max_utils.add_text_to_wandb("num_model_parameters", str(num_model_parameters), wandb_run)
  #   max_utils.add_text_to_wandb("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], wandb_run)
  #   maxtext_utils.add_config_to_wandb(config, wandb_run)

  # Define the compilation of functional_train, either by loading the compiled version or wrapping a new one in a jit
  if config.compiled_trainstep_file != "":
    print("Loading the compiled function...", flush=True)
    # Need to pass train signature and state to determine i/o shapes of train_state for now.
    p_train_step = maxtext_utils.load_compiled(config, functional_train, state)
    # TODO: p_eval_step is not yet supported in load_compiled
    p_eval_step = None
    print("Loaded compiled function!", flush=True)
  else:
    p_train_step = jax.jit(
        functional_train,
        in_shardings=in_shard_train,
        out_shardings=out_shard_train,
        static_argnums=static_argnums_train,
        donate_argnums=donate_argnums_train,
    )

    if eval_data_iterator:
      p_eval_step = jax.jit(
          functional_eval,
          in_shardings=in_shard_eval,
          out_shardings=out_shard_eval,
          static_argnums=static_argnums_eval,
          donate_argnums=donate_argnums_eval,
      )
    else:
      p_eval_step = None

  running_gcs_metrics = [] if config.gcs_metrics else None
  metrics = None

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  first_profiling_step = prof.start_initial_profile_step
  if config.profiler != "" and first_profiling_step >= config.steps:
    raise ValueError("Profiling requested but initial profiling step set past training final step")
  last_profiling_step = prof.finished_initial_profile_step

  example_batch = None
  last_step_completion = datetime.datetime.now()

  performance_metric_queue = None
  if config.report_heartbeat_metric_for_gcp_monitoring or config.report_performance_metric_for_gcp_monitoring:
    gcp_workload_monitor = GCPWorkloadMonitor(config.run_name)
    if config.report_heartbeat_metric_for_gcp_monitoring:
      gcp_workload_monitor.start_heartbeat_reporting_thread(config.heartbeat_reporting_interval_in_seconds)
    if config.report_performance_metric_for_gcp_monitoring:
      performance_metric_queue = queue.Queue()
      gcp_workload_monitor.start_performance_reporting_thread(performance_metric_queue)

  metric_logger = MetricLogger(writer, wandb_run, config)
  input_data_shardings = maxtext_utils.get_input_data_sharding(config, mesh)
  for step in np.arange(start_step, config.steps):
    if step == first_profiling_step or prof.should_activate_periodic_profile(step):
      optional_postfix = f"step_{step}" if config.profile_periodically_period > 0 else ""
      prof.activate(blocking_object=state, optional_postfix=optional_postfix)

    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      with maybe_record_goodput(recorder, GoodputEvent.DATA_LOADING):
        try:
          example_batch = load_next_batch(data_iterator, example_batch, config)
          # Reshard data from loaded sharding to performant activation sharding
          example_batch = jax.lax.with_sharding_constraint(example_batch, input_data_shardings)
        except Exception as e:  # pylint: disable=broad-except
          max_logging.log(f"load_next_batch failed, you may have run out of data. Error message: {e}")
          break

      check_example_batch(config, example_batch=example_batch)
      # pylint: disable=not-callable
      nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
      with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          state, metrics = p_train_step(state, example_batch, nextrng)

    step_time_delta = datetime.datetime.now() - last_step_completion
    last_step_completion = datetime.datetime.now()
    record_scalar_metrics(metrics, step_time_delta, per_device_tflops, learning_rate_schedule(step), per_device_tokens)
    if performance_metric_queue:
      performance_metric_queue.put(step_time_delta.total_seconds())

    if checkpoint_manager is not None:
      state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
      if save_checkpoint(checkpoint_manager, int(step), state_to_save, config.dataset_type, data_iterator, config):
        checkpointing.print_save_message(step, config.async_checkpointing)

      # Upon preemption, exit when and only when all ongoing saves are complete.
      if checkpoint_manager.reached_preemption(step):
        checkpoint_manager.wait_until_finished()
        sys.exit()

    metric_logger.write_metrics(running_gcs_metrics, metrics, step)

    if config.dump_hlo and step == (config.dump_step if config.dump_step >= 0 else start_step):
      jax.block_until_ready(state)  # Ensure compilation has finished.
      gcs_utils.upload_dump(
          config.dump_hlo_local_dir,
          config.dump_hlo_gcs_dir,
          module_name=config.dump_hlo_module_name,
          delete_local_after=config.dump_hlo_delete_local_after,
          all_host_upload=config.dump_hlo_upload_all,
      )

    if config.eval_interval > 0 and step > start_step and (step + 1) % config.eval_interval == 0:
      assert eval_data_iterator
      cumulative_eval_metrics = {
          "scalar": {
              "eval/total_loss": 0.0,
              "eval/total_weights": 0.0,
              "eval/avg_loss": 0.0,
              "eval/moe_lb_loss": 0.0,
          }
      }
      eval_dpo_reward_accuracy = 0.0
      eval_step_count = 0
      # pylint: disable=not-callable
      for eval_batch in eval_data_iterator:
        if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
          break
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          eval_metrics = p_eval_step(state, eval_batch, nextrng)
        cumulative_eval_metrics["scalar"]["eval/total_loss"] += float(eval_metrics["scalar"]["evaluation/total_loss"])
        cumulative_eval_metrics["scalar"]["eval/total_weights"] += float(eval_metrics["scalar"]["evaluation/total_weights"])
        cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] += float(eval_metrics["scalar"]["evaluation/moe_lb_loss"])
        eval_dpo_reward_accuracy += float(eval_metrics["scalar"].get("evaluation/dpo_reward_accuracy", 0.0))  # for dpo only
        max_logging.log(f"Completed eval step {eval_step_count}")
        eval_step_count += 1
      eval_loss = cumulative_eval_metrics["scalar"]["eval/total_loss"] / (
          cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS
      )
      cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
      cumulative_eval_metrics["scalar"]["eval/avg_moe_lb_loss"] = (
          cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
      )
      if config.use_dpo:
        cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] = eval_dpo_reward_accuracy / eval_step_count
      metric_logger.write_metrics(running_gcs_metrics, cumulative_eval_metrics, step, is_training=False)
      max_logging.log(
          f"average loss after {step=}: {eval_step_count=}, {eval_loss=},"
          f" total_weights={cumulative_eval_metrics['scalar']['eval/total_weights']}"
      )
      if eval_loss <= config.target_eval_loss:
        max_logging.log(f"Early stop and exit loop after reaching {config.target_eval_loss=}")
        prof.deactivate()
        break

    if step == last_profiling_step or prof.should_deactivate_periodic_profile(step):
      prof.deactivate(blocking_object=state)

    if step == start_step:
      max_utils.print_mem_stats("After params initialized")

  if checkpoint_manager is not None:
    if ((int(state.step) - 1) % config.checkpoint_period != 0) and (int(state.step) != 0):
      try:
        state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
        if save_checkpoint(
            checkpoint_manager,
            int(state.step) - 1,
            state_to_save,
            config.dataset_type,
            data_iterator,
            config,
            force=True,
        ):
          checkpointing.print_save_message(int(state.step) - 1, config.async_checkpointing)
      except Exception:  # pylint: disable=broad-except
        max_logging.log(f"Checkpoint is already saved for step {int(state.step)-1}.")

    checkpoint_manager.wait_until_finished()
  metric_logger.write_metrics(running_gcs_metrics, metrics, config.steps - 1)  # final step metrics
  max_utils.close_summary_writer(writer)
  max_utils.close_wandb_run(wandb_run)

  if example_batch:
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      # pytype: disable=attribute-error
      compiled = p_train_step.lower(state, example_batch, nextrng).compile()
      compiled_stats = compiled.memory_analysis()
      if compiled_stats is not None:
        max_logging.log(
            f"Output size: {compiled_stats.output_size_in_bytes}, "
            f"temp size: {compiled_stats.temp_size_in_bytes}, "
            f"argument size: {compiled_stats.argument_size_in_bytes}, "
            f"host temp size: {compiled_stats.host_temp_size_in_bytes}, in bytes."
        )
  return state


def main(argv: Sequence[str]) -> None:
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  # Goodput configurations
  maybe_monitor_goodput(config)
  recorder = create_goodput_recorder(config)

  # Stack traces configurations
  debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=config.collect_stack_trace,
          stack_trace_to_cloud=config.stack_trace_to_cloud,
          stack_trace_interval_seconds=config.stack_trace_interval_seconds,
      )
  )
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)

  with diagnostic.diagnose(diagnostic_config):
    with maybe_record_goodput(recorder, GoodputEvent.JOB):
      train_loop(config, recorder)


if __name__ == "__main__":
  app.run(main)
