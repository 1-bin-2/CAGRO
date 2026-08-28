# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.utils.data
import transformers
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
import datasets
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    LogitsProcessor,
    LogitsProcessorList,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, is_deepspeed_available
from transformers.utils import is_peft_available, is_rich_available,  is_datasets_available
from transformers.trainer_utils import seed_worker
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
# from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.models import create_reference_model, unwrap_model_for_generation
from trl.trainer.utils import prepare_deepspeed
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
try:
    from trl.trainer.utils import print_prompt_completions_sample
except ImportError:
    def print_prompt_completions_sample(prompts, completions, rewards, step, **kwargs):
        pass
from trl import GRPOTrainer
from trl.import_utils import is_deepspeed_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings
import torch.distributed as dist

try:
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
except Exception:
    deepspeed = None
    ZeroParamStatus = None
if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule
from open_r1.cagro import (
    CAGROGateConfig,
    compute_cagro_gate,
    find_unique_nonempty_tag_span,
    map_character_span_to_token_mask,
)
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]




def _z3_params_to_fetch(param_list):
    if ZeroParamStatus is None:
        return []
    return [
        param
        for param in param_list
        if hasattr(param, "ds_id") and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


class RefModelEMACallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer._maybe_update_ema_ref_model()
        return control

def find_mask_between_patterns_1d(input_tensor: torch.Tensor, 
                                  start_pattern_list: list, 
                                  end_pattern_list: list) -> torch.Tensor:
    """
    (Helper function - same as before)
    Finds the mask for a single 1D tensor.
    """
    assert input_tensor.ndim == 1, "Input tensor must be 1-dimensional"
    
    device = input_tensor.device
    dtype = input_tensor.dtype # Use input tensor's dtype

    # Ensure patterns are tensors on the correct device and dtype
    start_pattern = torch.tensor(start_pattern_list, dtype=dtype, device=device)
    end_pattern = torch.tensor(end_pattern_list, dtype=dtype, device=device)

    n = input_tensor.shape[0]
    len_start = len(start_pattern)
    len_end = len(end_pattern)

    start_idx = -1
    end_idx = -1

    # --- Find start_pattern index ---
    if n >= len_start:
        start_windows = input_tensor.unfold(0, len_start, 1)
        start_matches = (start_windows == start_pattern).all(dim=1)
        start_indices = start_matches.nonzero(as_tuple=True)[0]
        if start_indices.numel() > 0:
            start_idx = start_indices[0].item() # Assume first match
        else:
            # Indicate pattern not found for this row
            return torch.zeros_like(input_tensor, dtype=torch.long, device=device) 
            # raise ValueError("Start pattern not found in the tensor.") # Original behavior
    else:
        return torch.zeros_like(input_tensor, dtype=torch.long, device=device) # Too short

    # --- Find end_pattern index ---
    if n >= len_end:
        # Search *after* the start pattern to ensure correct order if multiple end patterns exist
        # Although problem states "only one region", this adds robustness
        search_area_end = input_tensor[start_idx + len_start:] 
        if search_area_end.numel() >= len_end:
            end_windows = search_area_end.unfold(0, len_end, 1)
            end_matches = (end_windows == end_pattern).all(dim=1)
            end_indices = end_matches.nonzero(as_tuple=True)[0]
            if end_indices.numel() > 0:
                 # Index relative to the start of search_area_end, need to add offset
                relative_end_idx = end_indices[0].item()
                end_idx = start_idx + len_start + relative_end_idx 
            else:
                 # End pattern not found *after* start pattern
                return torch.zeros_like(input_tensor, dtype=torch.long, device=device)
        else:
            # Not enough elements after start pattern to contain end pattern
             return torch.zeros_like(input_tensor, dtype=torch.long, device=device)
    else:
       return torch.zeros_like(input_tensor, dtype=torch.long, device=device) # Too short

    # --- Calculate mask region ---
    mask_start = start_idx + len_start
    mask_end = end_idx # end_idx is the *start* of the end pattern

    # mask_start = start_idx 
    # mask_end = end_idx + len_end 

    # --- Create and fill mask ---
    mask = torch.zeros_like(input_tensor, dtype=torch.long, device=device)

    # if mask_start < mask_end:
    #     mask[mask_start:-1] = 1
    if mask_start < mask_end:
        mask[:mask_end] = 1

    # if mask_start < mask_end:
    #     mask[mask_start:mask_end] = 1
    # else: patterns adjacent or end before start, mask remains zero, no warning needed here

    return mask



def generate_2d_mask(input_tensor_2d: torch.Tensor, 
                       start_pattern_list: list, 
                       end_pattern_list: list) -> torch.Tensor:
    """
    Generates a 2D mask by applying the 1D pattern finding logic to each row.

    Args:
        input_tensor_2d: The input 2D PyTorch Tensor (Batch x SequenceLength).
        start_pattern_list: The start pattern list.
        end_pattern_list: The end pattern list.

    Returns:
        A 2D mask tensor of the same shape as input_tensor_2d (dtype=torch.long),
        where each row's mask is generated based on the patterns found in that row.
        Rows where patterns are not found (or order is wrong) will have a mask of all zeros.
    """
    assert input_tensor_2d.ndim == 2, "Input tensor must be 2-dimensional"
    
    num_rows = input_tensor_2d.shape[0]
    if num_rows == 0:
        return torch.empty_like(input_tensor_2d, dtype=torch.long) # Handle empty input

    row_masks = []
    for i in range(num_rows):
        current_row = input_tensor_2d[i]
        # Call the 1D function for the current row
        # Modify 1D function to return zeros instead of raising error if pattern not found
        mask_1d = find_mask_between_patterns_1d(current_row, start_pattern_list, end_pattern_list)
        row_masks.append(mask_1d)

    # Stack the generated 1D masks along the batch dimension (dim=0)
    mask_2d = torch.stack(row_masks, dim=0)
    
    return mask_2d
class ForcePrefixLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_token_ids, prompt_length):
        self.prefix_token_ids = prefix_token_ids
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        gen_len = input_ids.shape[1] - self.prompt_length
        if 0 <= gen_len < len(self.prefix_token_ids):
            forced_id = self.prefix_token_ids[gen_len]
            scores[:, :] = -float("inf")
            scores[:, forced_id] = 0
        return scores
    
class SuppressMultimodalTokensProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        token_names = [
            "<|VIDEO|>",
            "<|AUDIO|>",
            "<|IMAGE|>",
            "<|video_pad|>",
            "<|image_pad|>",
            "<|audio_bos|>",
            "<|audio_eos|>",
            "<|vision_start|>",
            "<|vision_end|>",
        ]

        ids = []
        for token in token_names:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(encoded[0])

        self.suppress_token_ids = sorted(set(ids))

    def __call__(self, input_ids, scores):
        if self.suppress_token_ids:
            scores[:, self.suppress_token_ids] = -float("inf")
        return scores
class FiniteLogitsProcessor(LogitsProcessor):
    def __init__(self, trainer_step=-1, prompt_length=0):
        self.trainer_step = trainer_step
        self.prompt_length = prompt_length
        self.call_count = 0

    def __call__(self, input_ids, scores):
        self.call_count += 1

        rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else 0
        )

        # 仅打印张量形状，不调用 .item()，不会强制GPU同步。
        if rank == 0 and (
            self.call_count <= 3 or self.call_count % 16 == 0
        ):
            print(
                "[GEN_HEARTBEAT] "
                f"step={self.trainer_step} "
                f"call={self.call_count} "
                f"gen_len={input_ids.shape[1] - self.prompt_length} "
                f"batch={input_ids.shape[0]}",
                flush=True,
            )

        # 只修复真正的NaN/Inf，不改变正常有限logits。
        return torch.nan_to_num(
            scores,
            nan=-30.0,
            posinf=30.0,
            neginf=-30.0,
        )

class ForceTagTransitionProcessor(LogitsProcessor):
    def __init__(self, tokenizer, prompt_length, max_context_tokens=96, max_think_tokens=96):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.max_context_tokens = max_context_tokens
        self.max_think_tokens = max_think_tokens
        self.context_open = tokenizer.encode("<context>", add_special_tokens=False)
        self.context_close = tokenizer.encode("</context>", add_special_tokens=False)
        self.think_open = tokenizer.encode("<think>", add_special_tokens=False)
        self.think_close = tokenizer.encode("</think>", add_special_tokens=False)
        self.answer_open = tokenizer.encode("<answer>", add_special_tokens=False)
        self.context_to_think = tokenizer.encode("</context><think>", add_special_tokens=False)
        self.think_to_answer = tokenizer.encode("</think><answer>", add_special_tokens=False)
        self.think_open_tail = tokenizer.encode("think>", add_special_tokens=False)
        self.answer_open_tail = tokenizer.encode("answer>", add_special_tokens=False)

    def _decode(self, token_ids):
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _content_token_count(self, token_ids, marker):
        """Count actual generated tokens after a decoded marker, without re-tokenizing text."""
        for prefix_len in range(1, len(token_ids) + 1):
            if marker in self._decode(token_ids[:prefix_len]):
                return len(token_ids) - prefix_len
        return 0

    @staticmethod
    def _matching_suffix_prefix_length(gen, forced_sequence):
        matched_prefix = 0
        max_prefix = min(len(gen), len(forced_sequence) - 1)
        for prefix_len in range(max_prefix, 0, -1):
            if gen[-prefix_len:] == forced_sequence[:prefix_len]:
                matched_prefix = prefix_len
                break
        return matched_prefix

    @classmethod
    def _force_sequence_from_suffix(cls, gen, scores_row, forced_sequence):
        """Force a multi-token transition without assuming standalone tag tokenization."""
        if not forced_sequence:
            return False
        matched_prefix = cls._matching_suffix_prefix_length(gen, forced_sequence)
        scores_row[:] = -float("inf")
        scores_row[forced_sequence[matched_prefix]] = 0
        return True

    @staticmethod
    def _find(seq, pattern, start=0):
        if not pattern:
            return None
        for i in range(start, len(seq) - len(pattern) + 1):
            if seq[i:i + len(pattern)] == pattern:
                return i
        return None

    def _force_sequence(self, gen, scores_row, trigger_pos, forced_sequence):
        generated = len(gen) - trigger_pos
        if 0 <= generated < len(forced_sequence):
            scores_row[:] = -float("inf")
            scores_row[forced_sequence[generated]] = 0
            return True
        return False

    def __call__(self, input_ids, scores):
        for row_idx in range(input_ids.size(0)):
            gen = input_ids[row_idx, self.prompt_length:].tolist()
            text = self._decode(gen)
            context_opened = "<context>" in text
            context_closed = "</context>" in text
            think_opened = "<think>" in text
            think_closed = "</think>" in text
            answer_opened = "<answer>" in text

            if not context_opened or answer_opened:
                continue

            if think_opened:
                if think_closed:
                    if self._matching_suffix_prefix_length(gen, self.think_to_answer) > 0:
                        self._force_sequence_from_suffix(gen, scores[row_idx], self.think_to_answer)
                    elif text.endswith("<") or self._matching_suffix_prefix_length(gen, self.answer_open_tail) > 0:
                        self._force_sequence_from_suffix(gen, scores[row_idx], self.answer_open_tail)
                    else:
                        self._force_sequence_from_suffix(gen, scores[row_idx], self.answer_open)
                elif self._content_token_count(gen, "<think>") >= self.max_think_tokens:
                    self._force_sequence_from_suffix(gen, scores[row_idx], self.think_to_answer)
                continue

            if context_closed:
                if self._matching_suffix_prefix_length(gen, self.context_to_think) > 0:
                    self._force_sequence_from_suffix(gen, scores[row_idx], self.context_to_think)
                elif text.endswith("<") or self._matching_suffix_prefix_length(gen, self.think_open_tail) > 0:
                    self._force_sequence_from_suffix(gen, scores[row_idx], self.think_open_tail)
                else:
                    self._force_sequence_from_suffix(gen, scores[row_idx], self.think_open)
            elif self._content_token_count(gen, "<context>") >= self.max_context_tokens:
                self._force_sequence_from_suffix(gen, scores[row_idx], self.context_to_think)

        return scores
        
class StopAfterFirstAnswerProcessor(LogitsProcessor):
    def __init__(self, tokenizer, prompt_length, eos_token_id):
        self.prompt_length = prompt_length

        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None

        self.eos_token_id = eos_token_id

        self.answer_open = [
            ids
            for ids in (
                tokenizer.encode(marker, add_special_tokens=False)
                for marker in (
                    "<answer>",
                    " <answer>",
                    "\n<answer>",
                    "\n <answer>",
                )
            )
            if ids
        ]

        self.answer_close = [
            ids
            for ids in (
                tokenizer.encode(marker, add_special_tokens=False)
                for marker in (
                    "</answer>",
                    " </answer>",
                    "\n</answer>",
                    "\n </answer>",
                )
            )
            if ids
        ]

        self._pattern_cache = {}

    def _patterns_for(self, generated):
        key = (
            generated.device.type,
            generated.device.index,
            generated.dtype,
        )

        cached = self._pattern_cache.get(key)

        if cached is None:
            cached = (
                [
                    torch.tensor(
                        pattern,
                        dtype=generated.dtype,
                        device=generated.device,
                    )
                    for pattern in self.answer_open
                ],
                [
                    torch.tensor(
                        pattern,
                        dtype=generated.dtype,
                        device=generated.device,
                    )
                    for pattern in self.answer_close
                ],
            )
            self._pattern_cache[key] = cached

        return cached

    @staticmethod
    def _contains_any(generated, patterns):
        found = torch.zeros(
            generated.size(0),
            dtype=torch.bool,
            device=generated.device,
        )

        for pattern in patterns:
            width = pattern.numel()

            if generated.size(1) >= width:
                windows = generated.unfold(1, width, 1)
                found |= (
                    windows
                    .eq(pattern.view(1, 1, -1))
                    .all(dim=-1)
                    .any(dim=-1)
                )

        return found

    @staticmethod
    def _ends_with_any(generated, patterns):
        found = torch.zeros(
            generated.size(0),
            dtype=torch.bool,
            device=generated.device,
        )

        for pattern in patterns:
            width = pattern.numel()

            if generated.size(1) >= width:
                found |= (
                    generated[:, -width:]
                    .eq(pattern.view(1, -1))
                    .all(dim=-1)
                )

        return found

    def __call__(self, input_ids, scores):
        if self.eos_token_id is None:
            return scores

        generated = input_ids[:, self.prompt_length:]
        open_patterns, close_patterns = self._patterns_for(generated)

        done = self._contains_any(generated, open_patterns)
        done &= self._ends_with_any(generated, close_patterns)

        scores[done, :] = -float("inf")
        scores[done, self.eos_token_id] = 0.0

        return scores

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count



class VLMGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = True,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module
        require_full_parameter_training = os.environ.get("REQUIRE_FULL_PARAMETER_TRAINING", "0") == "1"
        if require_full_parameter_training and peft_config is not None:
            raise ValueError(
                "REQUIRE_FULL_PARAMETER_TRAINING=1 but a PEFT configuration was supplied; "
                "the formal run must not use LoRA/PEFT."
            )

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # model_init_kwargs["enable_audio_output"] = False
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )
        #     # Disable caching if gradient checkpointing is enabled (not supported)
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        # model = model.thinker # for qwen-omni

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # ZeRO-3 partitions parameters during model initialization. In that state
        # p.numel() can be zero while p.ds_numel contains the real parameter count.
        def _effective_numel(param):
            ds_numel = getattr(param, "ds_numel", None)
            return int(ds_numel) if ds_numel is not None else int(param.numel())

        trainable_param_tensors = [
            param for param in model.parameters() if param.requires_grad
        ]
        total_model_params = sum(
            _effective_numel(param) for param in model.parameters()
        )
        total_trainable_params = sum(
            _effective_numel(param) for param in trainable_param_tensors
        )
        lora_params = sum(
            _effective_numel(param)
            for name, param in model.named_parameters()
            if "lora_" in name.lower()
        )

        if require_full_parameter_training:
            if is_peft_model(model) or lora_params:
                raise RuntimeError(
                    "Formal run requested full-parameter training but the model contains PEFT/LoRA parameters."
                )
            if not trainable_param_tensors or total_trainable_params <= 0:
                raise RuntimeError(
                    "Formal full-parameter run has no trainable parameters."
                )
        print(
            "[TRAINING_MODE] full_parameter_nonvision="
            f"{not is_peft_model(model)} trainable={total_trainable_params:,} "
            f"total={total_model_params:,} lora_params={lora_params:,} "
            f"vision_frozen={bool(freeze_vision_modules)}",
            flush=True,
        )

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # CAGRO uses the policy at the start of RL as a permanently frozen
        # reference, even when the GRPO KL coefficient is zero (paper Sec. 3.2).
        need_reference_model = bool(
            args.beta > 0
            or getattr(args, "use_cagro", False)
            or getattr(args, "ema_ref_model", False)
        )
        if need_reference_model and (is_deepspeed_zero3_enabled() or is_deepspeed_available()):
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
            # self.ref_model = self.ref_model.thinker # for qwen-omni
        elif need_reference_model and peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
            # self.ref_model = self.ref_model.thinker # for qwen-omni
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            for processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    setattr(processing_class, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id
        # print(processing_class.tokenizer)
        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)
        if self.ref_model is not None:
            self.ref_model.eval()
            for ref_param in self.ref_model.parameters():
                ref_param.requires_grad_(False)


        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        self.use_cagro = args.use_cagro
        self.cagro_gate_config = CAGROGateConfig(
            answer_threshold=args.answer_validity_threshold,
            context_threshold=args.context_validity_threshold,
            reasoning_threshold=args.reasoning_validity_threshold,
            support_upper_bound=args.support_upper_bound,
            support_tolerance=args.support_tolerance,
            bonus_coefficient=args.bonus_coefficient,
        )

        def reward_role(reward_func):
            name = getattr(reward_func, "__name__", "").lower()
            if name == "format_reward":
                return "format"
            if name == "accuracy_reward":
                return "answer"
            if "context_reward" in name:
                return "context"
            if "reasoning_reward" in name and "consistency" not in name:
                return "reasoning"
            return None

        self.cagro_reward_indices = {}
        unsupported_cagro_rewards = []
        for index, reward_func in enumerate(self.reward_funcs):
            role = reward_role(reward_func)
            if role is None:
                unsupported_cagro_rewards.append(getattr(reward_func, "__name__", repr(reward_func)))
                continue
            if role in self.cagro_reward_indices:
                raise ValueError(f"CAGRO received more than one {role} reward")
            self.cagro_reward_indices[role] = index

        if self.use_cagro:
            missing_roles = {
                "format", "answer", "context", "reasoning"
            } - set(self.cagro_reward_indices)
            if missing_roles:
                raise ValueError(f"CAGRO is missing required reward signals: {sorted(missing_roles)}")
            if unsupported_cagro_rewards:
                raise ValueError(
                    "CAGRO accepts exactly four base signals and no independent consistency reward; "
                    f"unsupported rewards: {unsupported_cagro_rewards}"
                )
            if not args.scale_rewards:
                raise ValueError("CAGRO requires independent group standardization; set scale_rewards=true")
            if args.markov_reward:
                raise ValueError("CAGRO uses four independent base streams; markov_reward must be false")
            if args.sync_ref_model or args.ema_ref_model:
                raise ValueError("CAGRO requires a fixed RL-start reference; EMA/sync reference updates are invalid")

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        elif self.use_cagro:
            paper_weights = {
                "format": 0.2,
                "answer": 0.7,
                "context": 0.2,
                "reasoning": 0.2,
            }
            weights = [0.0] * len(reward_funcs)
            for role, index in self.cagro_reward_indices.items():
                weights[index] = paper_weights[role]
            self.reward_weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)


        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        if self.max_prompt_length is not None:
            warnings.warn(
                "max_prompt_length is ignored by the Qwen-Omni CAGRO path to avoid "
                "splitting multimodal placeholder sequences. Pre-filter overlength examples instead."
            )
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.markov_reward = args.markov_reward
        self.ema_ref_model = args.ema_ref_model
        self.ema_ref_model_decay = args.ema_ref_model_decay
        self.ema_ref_model_update_steps = max(1, args.ema_ref_model_update_steps)
        self.ema_ref_state = None
     
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            pad_token_id=processing_class.tokenizer.pad_token_id,
            bos_token_id=processing_class.tokenizer.bos_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            remove_invalid_values=False,  # debug: inspect raw NaN/Inf before sanitizing
            renormalize_logits=True,
            repetition_penalty=self.repetition_penalty,
            cache_implementation=args.cache_implementation,
        )
        if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
            print(222, self.vlm_module.get_eos_token_id(processing_class))
        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon


        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        if not hasattr(model, "warnings_issued") or model.warnings_issued is None:
            model.warnings_issued = {}
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        if args.ema_ref_model and not is_peft_model(model):
            callbacks = list(callbacks) if callbacks is not None else []
            callbacks.append(RefModelEMACallback(self))
        elif args.ema_ref_model and is_peft_model(model):
            warnings.warn(
                "EMA reference updates are disabled for PEFT training; the frozen base reference remains active "
                "for ordinary GRPO confidence scoring."
            )
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        # if self.num_generations not in possible_values:
        #     raise ValueError(
        #         f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
        #         f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
        #         f"batch size, the valid values for the number of generations are: {possible_values}."
        #     )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                # The CAGRO reference is frozen during training, but ZeRO-3
                # still needs a DeepSpeed engine to gather its partitioned
                # parameters during forward. If every reference parameter is
                # marked frozen before initialization, TRL/DeepSpeed creates
                # an empty optimizer partition. Conversely, preparing it as a
                # bare Accelerate model leaves 1-D partition shards in the
                # embedding layer and fails on the first reference forward.
                deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
                zero_stage = getattr(deepspeed_plugin, "zero_stage", 0) if deepspeed_plugin is not None else 0
                if zero_stage == 3:
                    ref_params = list(self.ref_model.parameters())
                    ref_requires_grad = [param.requires_grad for param in ref_params]
                    for param in ref_params:
                        param.requires_grad_(True)
                    try:
                        self.ref_model = prepare_deepspeed(
                            self.ref_model,
                            self.args.per_device_train_batch_size,
                            fp16=getattr(self.args, "fp16", False),
                            bf16=getattr(self.args, "bf16", False),
                        )
                    finally:
                        for param, requires_grad in zip(self.ref_model.parameters(), ref_requires_grad):
                            param.requires_grad_(requires_grad)
                    self.ref_model.eval()
                else:
                    self.ref_model = prepare_deepspeed(
                        self.ref_model,
                        self.args.per_device_train_batch_size,
                        fp16=getattr(self.args, "fp16", False),
                        bf16=getattr(self.args, "bf16", False),
                    )
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

            if self.ema_ref_model and self.ref_model is None:
                self._init_ema_ref_state()


    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enable gradient checkpointing with the requested checkpoint implementation."""
        model.config.use_cache = False

        checkpoint_kwargs = dict(args.gradient_checkpointing_kwargs or {})
        checkpoint_kwargs.setdefault("use_reentrant", False)

        target_model = model.base_model if is_peft_model(model) else model

        try:
            target_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=checkpoint_kwargs
            )
        except TypeError:
            # Compatibility fallback for older Transformers model classes.
            target_model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        except Exception:
            # InternVL compatibility path retained from the original trainer.
            model.language_model.config.use_cache = False
            model.vision_model.gradient_checkpointing = True
            model.vision_model.encoder.gradient_checkpointing = True
            model.language_model._set_gradient_checkpointing()
            args.gradient_checkpointing = False

        if checkpoint_kwargs.get("use_reentrant", False):
            model.enable_input_require_grads()

        return model

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        # Reference scoring is called from _prepare_inputs before Trainer enters
        # its usual mixed-precision context.  Keep every multimodal forward in
        # the configured precision so FlashAttention-2 receives matching q/k/v
        # and rotary dtypes.
        autocast_dtype = (
            torch.bfloat16 if getattr(self.args, "bf16", False)
            else torch.float16 if getattr(self.args, "fp16", False)
            else None
        )
        if input_ids.device.type == "cuda" and autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    **custom_multimodal_inputs,
                ).logits
        else:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **custom_multimodal_inputs,
            ).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            row_logps = []
            chunk_size = max(1, int(os.environ.get("GRPO_LOGPROB_TOKEN_CHUNK_SIZE", "64")))

            for start in range(0, logits_row.size(0), chunk_size):
                end = min(start + chunk_size, logits_row.size(0))
                log_probs = logits_row[start:end].log_softmax(dim=-1)
                token_log_prob = torch.gather(
                    log_probs,
                    dim=1,
                    index=input_ids_row[start:end].unsqueeze(1),
                ).squeeze(1)
                row_logps.append(token_log_prob)

            per_token_logps.append(torch.cat(row_logps, dim=0))
        return torch.stack(per_token_logps)

    def _get_tokenizer(self):
        return getattr(self.processing_class, "tokenizer", self.processing_class)

    def _encode_marker(self, marker: str) -> list[int]:
        tokenizer = self._get_tokenizer()
        return tokenizer.encode(marker, add_special_tokens=False)

    @staticmethod
    def _find_subsequence(input_tensor: torch.Tensor, pattern: list[int], start: int = 0) -> Optional[int]:
        if len(pattern) == 0 or input_tensor.numel() - start < len(pattern):
            return None
        pattern_tensor = torch.tensor(pattern, dtype=input_tensor.dtype, device=input_tensor.device)
        windows = input_tensor[start:].unfold(0, len(pattern), 1)
        matches = (windows == pattern_tensor).all(dim=1)
        indices = matches.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return None
        return start + indices[0].item()

    def _decoded_prefixes(self, token_ids: list[int]) -> list[str]:
        """Decode actual generated prefixes so BPE merges across XML-like tags stay observable."""
        tokenizer = self._get_tokenizer()
        return [
            tokenizer.decode(
                token_ids[:prefix_len],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for prefix_len in range(len(token_ids) + 1)
        ]

    def _get_semantic_span_masks(
        self,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build mutually disjoint masks for tag bodies, excluding all tags.

        A uniquely parseable local span may retain its base supervision even
        when the full response fails the format reward. Any ambiguous,
        overlapping, empty, or unmappable span fails closed instead of falling
        back to a prefix or the complete response.
        """

        masks = {
            tag: torch.zeros_like(completion_ids, dtype=torch.float32)
            for tag in ("context", "think", "answer")
        }
        answer_span_valid = torch.zeros(
            completion_ids.size(0), dtype=torch.bool, device=completion_ids.device
        )
        answer_open_unique = torch.zeros_like(answer_span_valid)
        answer_close_unique = torch.zeros_like(answer_span_valid)
        for row_idx, row in enumerate(completion_ids):
            valid_len = int(completion_mask[row_idx].sum().item())
            token_ids = row[:valid_len].tolist()
            prefixes = self._decoded_prefixes(token_ids)
            text = prefixes[-1]
            answer_open_unique[row_idx] = text.count("<answer>") == 1
            answer_close_unique[row_idx] = text.count("</answer>") == 1
            spans = {
                tag: find_unique_nonempty_tag_span(text, tag)
                for tag in masks
            }
            present_spans = [span for span in spans.values() if span is not None]
            overlaps = any(
                left.start < right.end and right.start < left.end
                for index, left in enumerate(present_spans)
                for right in present_spans[index + 1 :]
            )
            if overlaps:
                continue

            for tag, char_span in spans.items():
                if char_span is None:
                    continue
                mapped = map_character_span_to_token_mask(prefixes, char_span)
                if mapped is None:
                    continue
                masks[tag][row_idx, :valid_len] = torch.tensor(
                    mapped, dtype=masks[tag].dtype, device=masks[tag].device
                )
                if tag == "answer":
                    answer_span_valid[row_idx] = True

        combined = masks["context"] + masks["think"] + masks["answer"]
        if torch.any(combined > 1):
            raise RuntimeError("CAGRO semantic token masks must be mutually disjoint")
        return masks, answer_span_valid, answer_open_unique, answer_close_unique

    def _prepare_inputs(self, inputs):
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
                # print(inputs)
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

        # # Simple pass-through, just like original
        # return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        # Generate the G samples from one multimodal prompt at a time. Passing
        # G copies of video/audio tensors to generate() multiplies the visual
        # attention memory by G and was the direct cause of the previous OOM.
        # num_return_sequences preserves the same GRPO group while reusing the
        # encoded multimodal prompt during sampling.
        base_inputs = inputs
        if len(base_inputs) != 1:
            raise ValueError(
                "Qwen-Omni CAGRO requires per_device_train_batch_size=1 so that "
                "multimodal prompts can be sampled without duplicating video/audio tensors."
            )
        inputs = [example for example in base_inputs for _ in range(self.num_generations)]

        prompts = [x["prompt"] for x in inputs]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, base_inputs)
        
        # use_audio_in_video = False #inputs[0].get("use_audio_in_video", False)
        use_audio_in_video = any(example.get("use_audio_in_video", False) for example in inputs)
        # print(prompts_text)
        images, videos, audios = [], [], []


        for each in base_inputs:
            if each["images"] is not None:
                images.extend(each["images"])
            if each["audios"] is not None:
                audios.extend(each["audios"])
            if each["videos"] is not None:
                videos.extend(each["videos"])
        if len(images) == 0: images = None
        if len(audios) == 0: audios = None
        if len(videos) == 0: videos = None

        # Qwen2.5-Omni FlashAttention generation requires genuinely left-padded
        # decoder inputs.  Passing padding_side only as a processor kwarg is not
        # sufficient in transformers 4.52; set the tokenizer before encoding.
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        tokenizer.padding_side = "left"

        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            audios,
            videos,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            use_audio_in_video=use_audio_in_video,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        # This custom generation/scoring path runs before Trainer enters its
        # mixed-precision context.  FlashAttention-2 requires visual/audio
        # activations and rotary cos/sin tensors to have the same dtype, so
        # align only floating multimodal features with the configured model
        # precision.  Integer masks, lengths, and token ids stay unchanged.
        target_mm_dtype = (
            torch.bfloat16 if getattr(self.args, "bf16", False)
            else torch.float16 if getattr(self.args, "fp16", False)
            else None
        )
        if target_mm_dtype is not None:
            for mm_key in ("pixel_values", "pixel_values_videos", "input_features"):
                mm_value = prompt_inputs.get(mm_key)
                if isinstance(mm_value, torch.Tensor) and torch.is_floating_point(mm_value):
                    prompt_inputs[mm_key] = mm_value.to(dtype=target_mm_dtype)

        prompt_inputs["use_audio_in_video"] = use_audio_in_video

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]


        # Generate completions
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            was_training = unwrapped_model.training
            gc_enabled = getattr(unwrapped_model, "is_gradient_checkpointing", False)
            old_use_cache = getattr(unwrapped_model.config, "use_cache", None)
            old_gen_use_cache = getattr(self.generation_config, "use_cache", None)
            old_num_return_sequences = getattr(self.generation_config, "num_return_sequences", None)

            try:
                unwrapped_model.eval()

                if gc_enabled and hasattr(unwrapped_model, "gradient_checkpointing_disable"):
                    unwrapped_model.gradient_checkpointing_disable()

                if old_use_cache is not None:
                    unwrapped_model.config.use_cache = True

                self.generation_config.use_cache = True
                generation_chunk_size = max(
                    1,
                    min(
                        self.num_generations,
                        int(os.environ.get("GRPO_GENERATION_CHUNK_SIZE", "1")),
                    ),
                )
                prompt_length_for_generation = prompt_ids.size(1)

                logits_processor = LogitsProcessorList([
                    FiniteLogitsProcessor(trainer_step=int(self.state.global_step), prompt_length=prompt_length_for_generation),
                    SuppressMultimodalTokensProcessor(tokenizer),
                    StopAfterFirstAnswerProcessor(
                        tokenizer,
                        prompt_length_for_generation,
                        self.generation_config.eos_token_id,
                    ),
                ])

                generation_inputs = {
                    k: v for k, v in prompt_inputs.items()
                    if k not in self.vlm_module.get_non_generate_params()
                }
                generated_chunks = []
                synced_generation = (
                    dist.is_available()
                    and dist.is_initialized()
                    and dist.get_world_size() > 1
                )
                for start in range(0, self.num_generations, generation_chunk_size):
                    count = min(generation_chunk_size, self.num_generations - start)
                    self.generation_config.num_return_sequences = count
                    if target_mm_dtype is not None:
                        with torch.autocast(device_type="cuda", dtype=target_mm_dtype):
                            generated_chunk = unwrapped_model.generate(
                                **generation_inputs,
                                generation_config=self.generation_config,
                                logits_processor=logits_processor,
                                synced_gpus=synced_generation,
                            )
                    else:
                        generated_chunk = unwrapped_model.generate(
                            **generation_inputs,
                            generation_config=self.generation_config,
                            logits_processor=logits_processor,
                            synced_gpus=synced_generation,
                        )
                    generated_chunks.append(generated_chunk)
                    # Release the previous generation workspace before drawing the
                    # next sample; this is memory management only and preserves G.
                # Sampling is performed in bounded chunks to reuse the encoded
                # prompt without expanding all G multimodal workspaces at once.
                # Chunks can hit EOS at different positions, so `generate()`
                # may return different sequence lengths. Normalize before stacking;
                # the later EOS-derived completion_mask still excludes all
                # right-padding from scoring.
                pad_token_id = getattr(self.processing_class, "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.generation_config.eos_token_id
                if isinstance(pad_token_id, (list, tuple)):
                    pad_token_id = pad_token_id[0]
                max_generated_length = max(chunk.size(1) for chunk in generated_chunks)
                padded_chunks = []
                for chunk in generated_chunks:
                    if chunk.size(1) < max_generated_length:
                        pad = torch.full(
                            (chunk.size(0), max_generated_length - chunk.size(1)),
                            pad_token_id,
                            dtype=chunk.dtype,
                            device=chunk.device,
                        )
                        chunk = torch.cat([chunk, pad], dim=1)
                    padded_chunks.append(chunk)
                generate_returned_result = torch.cat(padded_chunks, dim=0)

            finally:
                if old_use_cache is not None:
                    unwrapped_model.config.use_cache = old_use_cache

                if old_gen_use_cache is not None:
                    self.generation_config.use_cache = old_gen_use_cache
                if old_num_return_sequences is None:
                    self.generation_config.num_return_sequences = 1
                else:
                    self.generation_config.num_return_sequences = old_num_return_sequences

                if gc_enabled and hasattr(unwrapped_model, "gradient_checkpointing_enable"):
                    unwrapped_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

                if was_training:
                    unwrapped_model.train()
            prompt_length = prompt_ids.size(1)
            if not self.vlm_module.is_embeds_input():
                prompt_completion_ids = generate_returned_result
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                completion_ids = generate_returned_result
                prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

            # The processor saw one prompt, while the generated batch contains
            # G completions. Repeat masks and multimodal tensors in group order
            # for the policy/reference scoring forward passes.
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        def repeat_group_tensor(value, repetitions=None):
            repetitions = self.num_generations if repetitions is None else repetitions
            if value is None or not isinstance(value, torch.Tensor) or value.ndim == 0:
                return value
            return value.repeat((repetitions,) + (1,) * (value.ndim - 1))

        multimodal_inputs_base = {
            k: prompt_inputs[k] if k in prompt_inputs else None
            for k in multimodal_keywords
        }
        multimodal_inputs = {
            k: repeat_group_tensor(prompt_inputs[k]) if k in prompt_inputs else None
            for k in multimodal_keywords
        }

        # Score the G completions in small multimodal chunks. Generation is
        # already chunked above; without this second chunking the reference
        # log-probability forward pass recreates the same 7B audio attention OOM.
        reference_score_chunk_size = max(
            1,
            min(
                self.num_generations,
                int(
                    os.environ.get(
                        "GRPO_REFERENCE_SCORE_CHUNK_SIZE",
                        os.environ.get("GRPO_MULTIMODAL_SCORE_CHUNK_SIZE", "1"),
                    )
                ),
            ),
        )

        def get_chunked_logps(model):
            chunks = []
            for start in range(0, self.num_generations, reference_score_chunk_size):
                count = min(reference_score_chunk_size, self.num_generations - start)
                chunk_mm = {
                    k: repeat_group_tensor(prompt_inputs[k], count) if k in prompt_inputs else None
                    for k in multimodal_keywords
                }
                chunk_logps = self._get_per_token_logps(
                    model,
                    prompt_completion_ids[start : start + count],
                    attention_mask[start : start + count],
                    **chunk_mm,
                )
                chunks.append(chunk_logps)
                del chunk_mm, chunk_logps
            return torch.cat(chunks, dim=0)
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = get_chunked_logps(self.model)
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0 and not self.use_cagro:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = get_chunked_logps(self.ref_model)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            elif self.ema_ref_model and self.ema_ref_state is not None:
                ref_per_token_logps = self._get_per_token_logps_with_ema_ref(
                    self.model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:] 
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = get_chunked_logps(self.model)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        cagro_ref_answer_support = None
        cagro_answer_valid = None
        cagro_answer_token_count = None
        cagro_answer_open_unique = None
        cagro_answer_close_unique = None
        cagro_semantic_masks = None
        if self.use_cagro:
            if ref_per_token_logps is None:
                raise ValueError("CAGRO requires frozen-reference log probabilities")
            if ref_per_token_logps.shape != completion_ids.shape:
                raise RuntimeError(
                    "Reference log-probabilities are not aligned with completion tokens: "
                    f"{tuple(ref_per_token_logps.shape)} != {tuple(completion_ids.shape)}"
                )
            (
                cagro_semantic_masks,
                answer_span_valid,
                cagro_answer_open_unique,
                cagro_answer_close_unique,
            ) = self._get_semantic_span_masks(
                completion_ids,
                completion_mask,
            )
            answer_mask = cagro_semantic_masks["answer"]
            answer_mask = answer_mask.to(ref_per_token_logps.device)
            answer_span_valid = answer_span_valid.to(ref_per_token_logps.device)
            cagro_answer_open_unique = cagro_answer_open_unique.to(ref_per_token_logps.device)
            cagro_answer_close_unique = cagro_answer_close_unique.to(ref_per_token_logps.device)
            answer_mask_bool = answer_mask.bool()
            cagro_answer_token_count = answer_mask_bool.sum(dim=1)
            ref_logps_fp32 = ref_per_token_logps.detach().float()
            all_answer_logps_finite = torch.where(
                answer_mask_bool,
                torch.isfinite(ref_logps_fp32),
                torch.ones_like(answer_mask_bool),
            ).all(dim=1)
            masked_ref_logps = ref_logps_fp32.masked_fill(~answer_mask_bool, 0.0)
            avg_ref_logps_answer = masked_ref_logps.sum(dim=1) / cagro_answer_token_count.clamp_min(1)
            cagro_answer_valid = (
                answer_span_valid
                & (cagro_answer_token_count > 0)
                & all_answer_logps_finite
                & torch.isfinite(avg_ref_logps_answer)
            )
            cagro_ref_answer_support = torch.zeros_like(avg_ref_logps_answer)
            cagro_ref_answer_support[cagro_answer_valid] = torch.exp(
                avg_ref_logps_answer[cagro_answer_valid]
            )
        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Debug: dump the first suspicious completion batch from every rank.
        _debug_step = int(self.state.global_step)
        _debug_token_counts = (
            completion_mask.sum(dim=1).detach().cpu().tolist()
        )
        _debug_dump_steps = {
            0, 1, 2, 5, 10, 15, 18, 19, 20, 21, 22, 23, 24, 25
        }
        if (
            os.environ.get("CAGRO_DUMP_COMPLETIONS", "0") == "1"
            and
            _debug_step in _debug_dump_steps
            and _debug_step not in getattr(
                self, "_raw_completion_dumped_steps", set()
            )
        ):
            import json as _debug_json
            import os as _debug_os

            _debug_dir = _debug_os.path.join(
                self.args.output_dir, "debug_completions"
            )
            _debug_os.makedirs(_debug_dir, exist_ok=True)

            _debug_rank = int(self.accelerator.process_index)
            _debug_path = _debug_os.path.join(
                _debug_dir,
                f"raw_step_{_debug_step}_rank_{_debug_rank}.jsonl",
            )

            with open(_debug_path, "w", encoding="utf-8") as _debug_file:
                for _debug_index, _debug_text in enumerate(completions_text):
                    _debug_record = {
                        "step_before_update": _debug_step,
                        "rank": _debug_rank,
                        "index": _debug_index,
                        "token_count": int(
                            _debug_token_counts[_debug_index]
                        ),
                        "max_completion_width": int(
                            completion_ids.shape[1]
                        ),
                        "has_context": (
                            "<context>" in _debug_text
                            and "</context>" in _debug_text
                        ),
                        "has_think": (
                            "<think>" in _debug_text
                            and "</think>" in _debug_text
                        ),
                        "has_answer": (
                            "<answer>" in _debug_text
                            and "</answer>" in _debug_text
                        ),
                        "text": _debug_text,
                        "repr": repr(_debug_text),
                    }
                    _debug_file.write(
                        _debug_json.dumps(
                            _debug_record, ensure_ascii=False
                        )
                        + "\n"
                    )

            if not hasattr(self, "_raw_completion_dumped_steps"):
                self._raw_completion_dumped_steps = set()
            self._raw_completion_dumped_steps.add(_debug_step)
            print(
                f"[debug] raw completions saved to {_debug_path}",
                flush=True,
            )
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
        else:
            completions = completions_text

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        reward_kwargs = {}
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # No need to duplicate prompts as we're not generating multiple completions per prompt
                        # reward_kwargs[key].extend([example[key]] * self.num_generations)
                        reward_kwargs[key].extend([example[key]])
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                if len(output_reward_func) != len(prompts):
                    raise RuntimeError(
                        f"Reward {getattr(reward_func, '__name__', type(reward_func).__name__)} "
                        f"returned {len(output_reward_func)} scores for {len(prompts)} completions."
                    )
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # markov
        if rewards_per_func.size(1) ==4 and self.markov_reward: # format, acc, reason, evi
            print("using markov")
            not_valid_evidence_index = rewards_per_func[:, -1]<=0.4
            not_valid_reason_index = rewards_per_func[:, -2]<=0.4
            rewards_per_func[not_valid_evidence_index, 1] = 0
            rewards_per_func[not_valid_evidence_index, 2] = 0
            rewards_per_func[not_valid_reason_index, 1] = 0

            # not_valid_format_index = rewards_per_func[:, 1]<=0.2
            # rewards_per_func[not_valid_format_index, 0] = 0

        # if rewards_per_func.size(1) ==2: # format, acc, reason, evi
        #     print("using markov")
        # not_valid_format_index = rewards_per_func[:, 1]<=0.2
        
        # rewards_per_func[not_valid_format_index, :] = 0

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )
        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        
        def clean_signal(signal):
            # Evaluator failures and non-finite values are invalid low evidence,
            # never favorable defaults (paper Sec. 3.2).
            return torch.nan_to_num(signal.float(), nan=0.0, posinf=0.0, neginf=0.0)

        def compute_advantage(signal):
            """Standardize one supervision stream within each G-way group."""
            signal = clean_signal(signal)
            grouped = signal.view(-1, self.num_generations)
            grouped_mean = grouped.mean(dim=1)
            grouped_std = grouped.std(dim=1, unbiased=False)
            mean = grouped_mean.repeat_interleave(self.num_generations, dim=0)
            std = grouped_std.repeat_interleave(self.num_generations, dim=0)
            advantage = signal - mean
            if self.args.scale_rewards:
                advantage = advantage / (std + 1e-4)
            return advantage, std

        # Get only the local process slice after globally gathering rewards.
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        cagro_bonus_rewards = None
        cagro_ref_answer_support_gathered = None
        cagro_answer_valid_gathered = None
        cagro_answer_token_count_gathered = None
        cagro_answer_open_unique_gathered = None
        cagro_answer_close_unique_gathered = None
        cagro_task_valid_gathered = None
        cagro_relative_pass_gathered = None
        partial_advantages = []

        if self.use_cagro:
            if cagro_semantic_masks is None:
                raise RuntimeError("CAGRO semantic masks were not computed")

            indices = self.cagro_reward_indices
            format_signal_raw = rewards_per_func[:, indices["format"]]
            answer_signal_raw = rewards_per_func[:, indices["answer"]]
            context_signal_raw = rewards_per_func[:, indices["context"]]
            reasoning_signal_raw = rewards_per_func[:, indices["reasoning"]]

            cagro_ref_answer_support_gathered = self.accelerator.gather(
                cagro_ref_answer_support
            ).to(device)
            cagro_answer_valid_gathered = self.accelerator.gather(
                cagro_answer_valid.to(torch.uint8)
            ).to(device).bool()
            cagro_answer_token_count_gathered = self.accelerator.gather(
                cagro_answer_token_count
            ).to(device)
            cagro_answer_open_unique_gathered = self.accelerator.gather(
                cagro_answer_open_unique.to(torch.uint8)
            ).to(device).bool()
            cagro_answer_close_unique_gathered = self.accelerator.gather(
                cagro_answer_close_unique.to(torch.uint8)
            ).to(device).bool()

            cagro_bonus_rewards = torch.zeros_like(answer_signal_raw, dtype=torch.float32)
            cagro_task_valid_gathered = torch.zeros_like(
                cagro_answer_valid_gathered, dtype=torch.bool
            )
            cagro_relative_pass_gathered = torch.zeros_like(
                cagro_answer_valid_gathered, dtype=torch.bool
            )
            clipped_support = torch.zeros_like(cagro_ref_answer_support_gathered)

            for group_start in range(0, len(answer_signal_raw), self.num_generations):
                group_slice = slice(group_start, group_start + self.num_generations)
                group_support = [
                    float(value) if valid else None
                    for value, valid in zip(
                        cagro_ref_answer_support_gathered[group_slice].detach().cpu().tolist(),
                        cagro_answer_valid_gathered[group_slice].detach().cpu().tolist(),
                    )
                ]
                gate_result = compute_cagro_gate(
                    format_rewards=format_signal_raw[group_slice].detach().cpu().tolist(),
                    answer_rewards=answer_signal_raw[group_slice].detach().cpu().tolist(),
                    context_rewards=context_signal_raw[group_slice].detach().cpu().tolist(),
                    reasoning_rewards=reasoning_signal_raw[group_slice].detach().cpu().tolist(),
                    answer_span_valid=cagro_answer_valid_gathered[group_slice].detach().cpu().tolist(),
                    answer_support=group_support,
                    config=self.cagro_gate_config,
                )
                cagro_bonus_rewards[group_slice] = torch.tensor(
                    gate_result.bonuses, device=device, dtype=torch.float32
                )
                cagro_task_valid_gathered[group_slice] = torch.tensor(
                    gate_result.task_valid, device=device, dtype=torch.bool
                )
                cagro_relative_pass_gathered[group_slice] = torch.tensor(
                    gate_result.relative_pass, device=device, dtype=torch.bool
                )
                clipped_support[group_slice] = torch.tensor(
                    [value if value is not None else 0.0 for value in gate_result.clipped_support],
                    device=device,
                    dtype=torch.float32,
                )

            format_signal = clean_signal(format_signal_raw)
            answer_signal = clean_signal(answer_signal_raw) + cagro_bonus_rewards
            context_signal = clean_signal(context_signal_raw)
            reasoning_signal = clean_signal(reasoning_signal_raw)

            format_advantage, format_std = compute_advantage(format_signal)
            answer_advantage, answer_std = compute_advantage(answer_signal)
            context_advantage, context_std = compute_advantage(context_signal)
            reasoning_advantage, reasoning_std = compute_advantage(reasoning_signal)

            weights = self.reward_weights.to(device)
            advantages = (
                format_advantage * weights[indices["format"]]
            )[process_slice]
            partial_advantages = [
                {
                    "name": "cagro_context",
                    "reward": (
                        context_advantage * weights[indices["context"]]
                    )[process_slice],
                    "mask": cagro_semantic_masks["context"],
                },
                {
                    "name": "cagro_reasoning",
                    "reward": (
                        reasoning_advantage * weights[indices["reasoning"]]
                    )[process_slice],
                    "mask": cagro_semantic_masks["think"],
                },
                {
                    "name": "cagro_answer",
                    "reward": (
                        answer_advantage * weights[indices["answer"]]
                    )[process_slice],
                    "mask": cagro_semantic_masks["answer"],
                },
            ]
            rewards = (
                weights[indices["format"]] * format_signal
                + weights[indices["answer"]] * answer_signal
                + weights[indices["context"]] * context_signal
                + weights[indices["reasoning"]] * reasoning_signal
            )
            std_grouped_rewards = torch.stack(
                [format_std, answer_std, context_std, reasoning_std], dim=0
            ).mean(dim=0)
        else:
            # Preserve ordinary sequence-level GRPO outside the CAGRO profile.
            weights = self.reward_weights.to(device).unsqueeze(0)
            rewards = (clean_signal(rewards_per_func) * weights).sum(dim=1)
            advantages, std_grouped_rewards = compute_advantage(rewards)
            advantages = advantages[process_slice]



        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]


        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        # reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        if self.use_cagro and cagro_bonus_rewards is not None:
            self._metrics[mode]["rewards/cagro_bonus"].append(
                torch.nanmean(cagro_bonus_rewards).item()
            )
            self._metrics[mode]["cagro/answer_support"].append(
                torch.nanmean(cagro_ref_answer_support_gathered).item()
            )
            self._metrics[mode]["cagro/answer_mask_hit_rate"].append(
                cagro_answer_valid_gathered.float().mean().item()
            )
            self._metrics[mode]["cagro/answer_token_count"].append(
                cagro_answer_token_count_gathered.float().mean().item()
            )
            self._metrics[mode]["cagro/answer_open_unique_rate"].append(
                cagro_answer_open_unique_gathered.float().mean().item()
            )
            self._metrics[mode]["cagro/answer_close_unique_rate"].append(
                cagro_answer_close_unique_gathered.float().mean().item()
            )
            self._metrics[mode]["cagro/task_valid_rate"].append(
                cagro_task_valid_gathered.float().mean().item()
            )
            self._metrics[mode]["cagro/relative_pass_rate"].append(
                cagro_relative_pass_gathered.float().mean().item()
            )
            self._metrics[mode]["cagro/bonus_selection_rate"].append(
                cagro_bonus_rewards.gt(0).float().mean().item()
            )
            wrong_bonus = cagro_bonus_rewards.gt(0) & ~cagro_task_valid_gathered
            self._metrics[mode]["cagro/wrong_bonus_rate"].append(
                wrong_bonus.float().mean().item()
            )
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            # prompts_to_log = gather_object(prompts)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = gather_object(rewards.detach().float().cpu().tolist())
            rewards_to_log = [
                float(value)
                for item in rewards_to_log
                for value in (item if isinstance(item, (list, tuple)) else [item])
            ]
            
            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                # if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                #     import pandas as pd

                #     # For logging
                #     table = {
                #         "step": [str(self.state.global_step)] * len(rewards),
                #         "prompt": prompts_to_log,
                #         "completion": completions_to_log,
                #         "reward": rewards.tolist(),
                #     }
                #     df = pd.DataFrame(table)
                #     wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs_base,
            "multimodal_inputs_base": multimodal_inputs_base,
            "partial_advantages": partial_advantages,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")


    
        # Check if we need to generate new completions or use buffered ones
        # if self.state.global_step % self.num_iterations == 0:
        #     inputs = self._generate_and_score_completions(inputs)
        #     self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        # else:
        #     inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        # self._step += 1

        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        multimodal_inputs_base = inputs.get("multimodal_inputs_base")
        
        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Keep all G completions and the exact full-group GRPO objective, but
        # score the trainable policy in small forward chunks. Qwen2.5-Omni's
        # SDPA audio path builds a quadratic workspace over concatenated audio;
        # forwarding all four repeated audio streams at once can request more
        # than one L20 can hold. The chunk log-probabilities are concatenated in
        # their original order before a single loss and backward pass.
        source_multimodal_inputs = multimodal_inputs_base or multimodal_inputs
        policy_score_chunk_size = max(
            1,
            min(
                self.num_generations,
                int(
                    os.environ.get(
                        "GRPO_POLICY_SCORE_CHUNK_SIZE",
                        os.environ.get("GRPO_MULTIMODAL_SCORE_CHUNK_SIZE", "1"),
                    )
                ),
            ),
        )
        policy_logp_chunks = []
        for start in range(0, self.num_generations, policy_score_chunk_size):
            count = min(policy_score_chunk_size, self.num_generations - start)
            policy_inputs = {}
            for key, value in source_multimodal_inputs.items():
                if value is None or not isinstance(value, torch.Tensor) or value.ndim == 0:
                    policy_inputs[key] = value
                else:
                    policy_inputs[key] = value.repeat((count,) + (1,) * (value.ndim - 1))
            policy_logp_chunks.append(
                self._get_per_token_logps(
                    model,
                    input_ids[start : start + count],
                    attention_mask[start : start + count],
                    **policy_inputs,
                )
            )
            del policy_inputs
        per_token_logps = torch.cat(policy_logp_chunks, dim=0)

        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]
        per_token_logps = torch.nan_to_num(
            per_token_logps, nan=-100.0, posinf=0.0, neginf=-100.0
        ).clamp(min=-100.0, max=0.0)

        # Get the advantages from inputs
        advantages = inputs["advantages"]
        partial_advantages = inputs["partial_advantages"]

        token_advantages = advantages.to(device=per_token_logps.device, dtype=per_token_logps.dtype).unsqueeze(1)
        for partial_advantage in partial_advantages:
            partial_reward = partial_advantage["reward"].to(
                device=per_token_logps.device, dtype=per_token_logps.dtype
            )
            partial_mask = partial_advantage["mask"].to(
                device=per_token_logps.device, dtype=per_token_logps.dtype
            )
            token_advantages = token_advantages + partial_reward.unsqueeze(1) * partial_mask



        mode = "eval" if self.control.should_evaluate else "train"
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        old_per_token_logps = torch.nan_to_num(
            old_per_token_logps, nan=-100.0, posinf=0.0, neginf=-100.0
        ).clamp(min=-100.0, max=0.0)

        # Compute the policy ratio and clipped version
        # coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        log_ratio = (per_token_logps - old_per_token_logps).clamp(min=-20.0, max=20.0)
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        # per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        # per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss1 = coef_1 * token_advantages
        per_token_loss2 = coef_2 * token_advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta > 0:
            if self.state.global_step >(self.state.max_steps/2):
                beta = self.beta*0.25 
            else:
                beta = self.beta*(1-0.75*self.state.global_step/(self.state.max_steps/2))
            ref_per_token_logps = inputs["ref_per_token_logps"]
            # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            ref_per_token_logps = torch.nan_to_num(
                ref_per_token_logps, nan=-100.0, posinf=0.0, neginf=-100.0
            ).clamp(min=-100.0, max=0.0)
            kl_delta = (ref_per_token_logps - per_token_logps).clamp(min=-20.0, max=20.0)
            per_token_kl = torch.exp(kl_delta) - kl_delta - 1
            per_token_kl = torch.nan_to_num(
                per_token_kl, nan=0.0, posinf=100.0, neginf=0.0
            ).clamp(min=0.0, max=100.0)
            per_token_loss = per_token_loss + beta * per_token_kl

            # Log KL divergence
     
            # mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            token_counts = completion_mask.sum(dim=1).clamp_min(1)
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / token_counts).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute final loss
        # loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        per_token_loss = torch.nan_to_num(per_token_loss, nan=0.0, posinf=0.0, neginf=0.0)
        token_counts = completion_mask.sum(dim=1).clamp_min(1)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / token_counts).mean()
        if not torch.isfinite(loss):
            loss = per_token_loss.sum() * 0.0
        # Log clip ratio
        # is_clipped = (per_token_loss1 < per_token_loss2).float()
        is_clipped = (per_token_loss2 < per_token_loss1).float()
        # clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = ((is_clipped * completion_mask).sum(dim=1) / token_counts).mean()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss
    
    def _init_ema_ref_state(self):
        model = self.accelerator.unwrap_model(self.model)
        self.ema_ref_state = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad and torch.is_floating_point(param.data)
        }
        if len(self.ema_ref_state) == 0:
            self.ema_ref_state = None

    def _update_ema_ref_state(self):
        if self.ema_ref_state is None:
            return
        decay = self.ema_ref_model_decay
        model = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            for name, param in model.named_parameters():
                ema_param = self.ema_ref_state.get(name)
                if ema_param is None:
                    continue
                ema_param.mul_(decay).add_(param.detach().to(ema_param.device, dtype=ema_param.dtype), alpha=1.0 - decay)

    def _get_per_token_logps_with_ema_ref(self, model, input_ids, attention_mask, **multimodal_inputs):
        unwrapped_model = self.accelerator.unwrap_model(model)
        backups = []
        with torch.no_grad():
            for name, param in unwrapped_model.named_parameters():
                ema_param = self.ema_ref_state.get(name) if self.ema_ref_state is not None else None
                if ema_param is None:
                    continue
                backups.append((param, param.detach().clone()))
                param.copy_(ema_param.to(param.device, dtype=param.dtype))

            try:
                return self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
            finally:
                for param, backup in backups:
                    param.copy_(backup)

    def _maybe_update_ema_ref_model(self):
        # if not self.ema_ref_model or self.ref_model is None:
        if not self.ema_ref_model:
            return
        if self.ema_ref_state is not None:
            self._update_ema_ref_state()
            return
        if self.ref_model is None:
            return
            
        if self.state.global_step == 0 or self.state.global_step % self.ema_ref_model_update_steps != 0:
            return

        decay = self.ema_ref_model_decay
        model = self.accelerator.unwrap_model(self.model)
        ref_model = self.accelerator.unwrap_model(self.ref_model)
        # model_state = model.state_dict()

        model_params = dict(model.named_parameters())
        model_buffers = dict(model.named_buffers())
        with torch.no_grad():
            for name, ref_param in ref_model.named_parameters():
                # model_param = model_state.get(name)
                model_param = model_params.get(name)
                if model_param is None or not torch.is_floating_point(ref_param.data):
                    continue
                ref_param.data.mul_(decay).add_(model_param.detach().to(ref_param.device, dtype=ref_param.dtype), alpha=1.0 - decay)

            for name, ref_buffer in ref_model.named_buffers():
                # model_buffer = model_state.get(name)
                model_buffer = model_buffers.get(name)
                if model_buffer is None:
                    continue
                if torch.is_floating_point(ref_buffer.data):
                    ref_buffer.data.mul_(decay).add_(model_buffer.detach().to(ref_buffer.device, dtype=ref_buffer.dtype), alpha=1.0 - decay)
                else:
                    ref_buffer.data.copy_(model_buffer.detach().to(ref_buffer.device, dtype=ref_buffer.dtype))

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.state.global_step > 0 and self.state.global_step != getattr(self, "_last_ema_ref_step", None):
            self._maybe_update_ema_ref_model()
            self._last_ema_ref_step = self.state.global_step
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self, train_dataset=None, *args, **kwargs) -> Sampler:
        """Sample one unique prompt for each custom G-way generation group.

        ``_generate_and_score_completions`` already expands every sampled
        prompt into ``num_generations`` completions.  Repeating every dataset
        index G additional times here produced G squared completions per prompt,
        four times the intended compute for G=4 and only one quarter of the
        intended prompt diversity.
        """
        effective_prompt_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        if effective_prompt_batch_size < 1:
            raise ValueError(
                "The global effective prompt batch size must be positive: "
                f"effective_prompt_batch_size={effective_prompt_batch_size}."
            )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=1,
            batch_size=effective_prompt_batch_size,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )


    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=1,
            seed=self.args.seed,
        )
