"""Causal Language Model wrapper and dynamic token slice indexing for discrete optimization.

Provides robust handling of Hugging Face causal architectures, chat templates,
embedding projections, vocabulary filtering, and rigorous token slice alignment
avoiding BPE/SentencePiece boundary distortion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.utils.device import clear_device_cache, get_best_device, synchronize_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenSlices:
    """Explicit 1D tensor index slices for causal prompt decomposition.

    Attributes:
        prefix_slice: Tokens prior to mutable coordinates `slice(0, opt_start)`.
        opt_slice: Coordinates undergoing discrete optimization `slice(opt_start, opt_end)`.
        target_slice: Target sequence tokens `slice(target_start, target_end)`.
        loss_slice: Target labels slice for autoregressive loss computation.
        delim_slice: Optional intermediate delimiter tokens between opt and target.
    """

    prefix_slice: slice
    opt_slice: slice
    target_slice: slice
    loss_slice: slice
    delim_slice: Optional[slice] = None

    @property
    def opt_len(self) -> int:
        return self.opt_slice.stop - self.opt_slice.start

    @property
    def target_len(self) -> int:
        return self.target_slice.stop - self.target_slice.start

    def validate(self, total_seq_len: int) -> None:
        """Enforce mathematical and causal alignment invariants."""
        assert self.opt_slice.start >= 0, "opt_slice.start must be non-negative"
        assert self.opt_slice.stop > self.opt_slice.start, "opt_slice must span at least 1 token"
        assert self.target_slice.start >= self.opt_slice.stop, (
            f"target_slice.start ({self.target_slice.start}) must follow "
            f"opt_slice.stop ({self.opt_slice.stop})"
        )
        assert self.target_slice.stop > self.target_slice.start, "target_slice must span >= 1 token"
        assert self.target_slice.stop <= total_seq_len, (
            f"target_slice.stop ({self.target_slice.stop}) exceeds total seq len ({total_seq_len})"
        )
        assert self.loss_slice.start == self.target_slice.start - 1, (
            "loss_slice.start must equal target_slice.start - 1 for causal next-token prediction"
        )
        assert self.loss_slice.stop == self.target_slice.stop - 1, (
            "loss_slice.stop must equal target_slice.stop - 1"
        )


@dataclass
class PromptDecomposition:
    """Deconstructed prompt representation containing full input IDs and aligned slices."""

    input_ids: torch.Tensor  # [1, seq_len]
    slices: TokenSlices
    formatted_prompt: str
    target_text: str
    optim_text: str


class CausalLMModelWrapper:
    """Modular wrapper for Hugging Face Causal Language Models.

    Abstracts embedding retrieval, vocabulary mask generation, chat template mapping,
    and causal autoregressive loss computation.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device is not None else get_best_device()
        self.dtype = dtype or (torch.float16 if self.device.type == "mps" else torch.float32)

        self.model = model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        # Cache embedding matrix reference
        self._embedding_layer = self._resolve_embedding_layer()
        self._vocab_size = self._embedding_layer.weight.size(0)
        self._hidden_dim = self._embedding_layer.weight.size(1)

        # Cache vocabulary masks (e.g. non-ascii tokens)
        self._non_ascii_mask: Optional[torch.Tensor] = None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def _resolve_embedding_layer(self) -> nn.Embedding:
        """Extract input embedding layer across diverse causal LM architectures."""
        if hasattr(self.model, "get_input_embeddings"):
            layer = self.model.get_input_embeddings()
            if isinstance(layer, nn.Embedding):
                return layer

        # Fallback inspection for nested modules
        for attr in ("model.embed_tokens", "transformer.wte", "embeddings.word_embeddings"):
            curr: Any = self.model
            parts = attr.split(".")
            found = True
            for part in parts:
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    found = False
                    break
            if found and isinstance(curr, nn.Embedding):
                return curr

        raise AttributeError(f"Could not resolve input embedding layer from model {type(self.model)}")

    def get_embedding_matrix(self) -> torch.Tensor:
        """Retrieve embedding weight matrix [vocab_size, hidden_dim]."""
        return self._embedding_layer.weight

    def build_non_ascii_mask(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Construct boolean mask marking non-ASCII or special tokens.

        Returns:
            Tensor of shape `[vocab_size]`, where True indicates tokens that should be FILTERED OUT.
        """
        if self._non_ascii_mask is not None:
            return self._non_ascii_mask.to(device or self.device)

        dev = device or self.device
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=dev)

        for token_id in range(self.vocab_size):
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            # Filter non-printable, non-ASCII, or replacement characters
            if not token_str.isascii() or "\ufffd" in token_str:
                mask[token_id] = True

        # Always filter special tokens (BOS, EOS, PAD, UNK, chat delimiters)
        all_special_ids = set(self.tokenizer.all_special_ids)
        for special_id in all_special_ids:
            if 0 <= special_id < self.vocab_size:
                mask[special_id] = True

        self._non_ascii_mask = mask
        return mask

    def build_prompt_slices(
        self,
        context: str,
        optim_str: str,
        target: str,
        use_chat_template: bool = True,
        system_prompt: Optional[str] = None,
    ) -> PromptDecomposition:
        """Construct prompt and dynamically identify exact token slices.

        Args:
            context: Static user instruction or background context.
            optim_str: Initial optimizable text (e.g. '!' * 20).
            target: Target response string the model should conditionally output.
            use_chat_template: Whether to format using tokenizer.apply_chat_template.
            system_prompt: Optional system prompt content.

        Returns:
            PromptDecomposition container with input IDs, exact slices, and string spans.
        """
        if use_chat_template and hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return self._build_slices_chat_template(
                context=context,
                optim_str=optim_str,
                target=target,
                system_prompt=system_prompt,
            )
        else:
            return self._build_slices_raw(
                context=context,
                optim_str=optim_str,
                target=target,
            )

    def _build_slices_chat_template(
        self,
        context: str,
        optim_str: str,
        target: str,
        system_prompt: Optional[str] = None,
    ) -> PromptDecomposition:
        """Deconstruct chat template prompt into verified token slices."""
        # Sentinel strings to locate exact token boundaries without BPE mangling
        sentinel_before = "<<<OPT_START>>>"
        sentinel_after = "<<<OPT_END>>>"

        user_content_with_sentinels = f"{context} {sentinel_before}{optim_str}{sentinel_after}"

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content_with_sentinels})

        # Render prompt up to assistant generation marker
        prompt_with_sentinels = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Replace sentinels with actual strings
        opt_char_start = prompt_with_sentinels.find(sentinel_before)
        assert opt_char_start != -1, "Sentinel marker not found in formatted prompt"

        prefix_str = prompt_with_sentinels[:opt_char_start]
        after_prefix_str = prompt_with_sentinels[opt_char_start + len(sentinel_before):]

        opt_char_end_in_after = after_prefix_str.find(sentinel_after)
        assert opt_char_end_in_after != -1, "Sentinel end marker not found in formatted prompt"

        actual_optim_str = after_prefix_str[:opt_char_end_in_after]
        delim_and_assistant_prompt = after_prefix_str[opt_char_end_in_after + len(sentinel_after):]

        # Full prompt before target
        prompt_str = f"{prefix_str}{actual_optim_str}{delim_and_assistant_prompt}"
        full_text = f"{prompt_str}{target}"

        # Tokenize components
        prefix_ids = self.tokenizer(prefix_str, add_special_tokens=False).input_ids
        # Include BOS if tokenizer prepends BOS by default and prefix doesn't have it
        full_prompt_enc = self.tokenizer(prompt_str, add_special_tokens=True)
        full_prompt_ids = full_prompt_enc.input_ids

        full_text_enc = self.tokenizer(full_text, add_special_tokens=True)
        full_ids = torch.tensor([full_text_enc.input_ids], dtype=torch.long, device=self.device)

        # Locate slices via token sub-sequence matching
        optim_tokens = self.tokenizer(actual_optim_str, add_special_tokens=False).input_ids
        target_tokens = self.tokenizer(target, add_special_tokens=False).input_ids

        # Deterministic boundary matching in full_ids
        full_list = full_ids[0].tolist()
        m = len(optim_tokens)
        t = len(target_tokens)

        # Find optim_tokens in full_list
        opt_start = -1
        for i in range(len(full_list) - m):
            if full_list[i : i + m] == optim_tokens:
                opt_start = i
                break

        # Fallback to prefix length if sub-sequence match shifted slightly due to whitespace
        if opt_start == -1:
            # Measure prefix token length directly
            temp_prefix = self.tokenizer(prefix_str, add_special_tokens=True).input_ids
            opt_start = len(temp_prefix)
            opt_end = opt_start + m
        else:
            opt_end = opt_start + m

        # Find target_tokens at the tail
        target_start = -1
        for j in range(opt_end, len(full_list) - t + 1):
            if full_list[j : j + t] == target_tokens:
                target_start = j
                break

        if target_start == -1:
            target_start = len(full_list) - t
            target_end = len(full_list)
        else:
            target_end = target_start + t

        prefix_slice = slice(0, opt_start)
        opt_slice = slice(opt_start, opt_end)
        target_slice = slice(target_start, target_end)
        loss_slice = slice(target_start - 1, target_end - 1)
        delim_slice = slice(opt_end, target_start) if target_start > opt_end else None

        slices = TokenSlices(
            prefix_slice=prefix_slice,
            opt_slice=opt_slice,
            target_slice=target_slice,
            loss_slice=loss_slice,
            delim_slice=delim_slice,
        )

        slices.validate(total_seq_len=full_ids.size(1))

        return PromptDecomposition(
            input_ids=full_ids,
            slices=slices,
            formatted_prompt=prompt_str,
            target_text=target,
            optim_text=actual_optim_str,
        )

    def _build_slices_raw(
        self,
        context: str,
        optim_str: str,
        target: str,
    ) -> PromptDecomposition:
        """Construct raw causal prompt and token slices without chat formatting."""
        prefix_str = f"{context} " if context else ""
        prompt_str = f"{prefix_str}{optim_str}"
        full_text = f"{prompt_str} {target}"

        full_enc = self.tokenizer(full_text, add_special_tokens=True)
        full_ids = torch.tensor([full_enc.input_ids], dtype=torch.long, device=self.device)
        full_list = full_ids[0].tolist()

        optim_tokens = self.tokenizer(f" {optim_str}" if prefix_str else optim_str, add_special_tokens=False).input_ids
        target_tokens = self.tokenizer(f" {target}", add_special_tokens=False).input_ids

        m = len(optim_tokens)
        t = len(target_tokens)

        # Match optim tokens
        opt_start = -1
        for i in range(len(full_list) - m):
            if full_list[i : i + m] == optim_tokens:
                opt_start = i
                break

        if opt_start == -1:
            prefix_enc = self.tokenizer(prefix_str, add_special_tokens=True).input_ids
            opt_start = len(prefix_enc)
            opt_end = opt_start + m
        else:
            opt_end = opt_start + m

        # Match target tokens
        target_start = -1
        for j in range(opt_end, len(full_list) - t + 1):
            if full_list[j : j + t] == target_tokens:
                target_start = j
                break

        if target_start == -1:
            target_start = len(full_list) - t
            target_end = len(full_list)
        else:
            target_end = target_start + t

        prefix_slice = slice(0, opt_start)
        opt_slice = slice(opt_start, opt_end)
        target_slice = slice(target_start, target_end)
        loss_slice = slice(target_start - 1, target_end - 1)
        delim_slice = slice(opt_end, target_start) if target_start > opt_end else None

        slices = TokenSlices(
            prefix_slice=prefix_slice,
            opt_slice=opt_slice,
            target_slice=target_slice,
            loss_slice=loss_slice,
            delim_slice=delim_slice,
        )

        slices.validate(total_seq_len=full_ids.size(1))

        return PromptDecomposition(
            input_ids=full_ids,
            slices=slices,
            formatted_prompt=prompt_str,
            target_text=target,
            optim_text=optim_str,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """Autoregressively generate continuation tokens to test model response."""
        self.model.eval()
        with torch.no_grad():
            gen_tokens = self.model.generate(
                input_ids.to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            # Decode generated output suffix
            input_len = input_ids.size(1)
            generated_ids = gen_tokens[0, input_len:].tolist()
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
