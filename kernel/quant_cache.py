from transformers.cache_utils import DynamicCache
import torch
from typing import Any, Dict, List, Optional, Tuple
from .packing import triton_quantize_and_pack_along_last_dim, quant_and_pack_vcache

class ValueQuantizedCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()  # Initialize the base class
        # Only quantization factors and full precision cache for values, not keys
        self.value_scales_cache: List[torch.Tensor] = []
        self.value_zeros_cache: List[torch.Tensor] = []
        self.value_full_precision_cache: List[torch.Tensor] = []  # Full precision storage for values

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the cache and quantization factors for a specific layer.

        Returns:
            A tuple containing:
            - key tensor
            - value tensor
            - value scales tensor
            - value zeros tensor
            - value full precision tensor
        """
        if layer_idx < len(self):
            return (
                self.key_cache[layer_idx], 
                self.value_cache[layer_idx],
                self.value_scales_cache[layer_idx],
                self.value_zeros_cache[layer_idx],
                self.value_full_precision_cache[layer_idx]
            )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        value_full_precision: torch.Tensor,
        value_scales: torch.Tensor,
        value_zeros: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new `key_states`, `value_states`, `value_scales`, `value_zeros`, 
        and `value_full_precision` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            layer_idx (`int`): The index of the layer to cache the states for.
            value_scales (`torch.Tensor`): The quantization scales for the value states.
            value_zeros (`torch.Tensor`): The quantization zero points for the value states.
            value_full_precision (`torch.Tensor`): The full precision value states.
            cache_kwargs (`Dict[str, Any]`, `optional`): Additional arguments for the cache subclass.

        Returns:
            A tuple containing the updated key and value states, value scales, value zeros, and value full precision.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        # Update the key and value caches
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.value_scales_cache.append(value_scales)
            self.value_zeros_cache.append(value_zeros)
            self.value_full_precision_cache.append(value_full_precision)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            # Append or update scales, zeros, and full precision values for values only
            self.value_scales_cache[layer_idx] = torch.cat([self.value_scales_cache[layer_idx], value_scales], dim=0)
            self.value_zeros_cache[layer_idx] = torch.cat([self.value_zeros_cache[layer_idx], value_zeros], dim=0)
            self.value_full_precision_cache[layer_idx] = torch.cat(
                [self.value_full_precision_cache[layer_idx], value_full_precision], dim=0
            )

        return (
            self.key_cache[layer_idx], 
            self.value_cache[layer_idx],
            self.value_full_precision_cache[layer_idx],
            self.value_scales_cache[layer_idx],
            self.value_zeros_cache[layer_idx]
        )

    def get_max_length(self) -> Optional[int]:
        raise NotImplementedError("Method `get_max_length` is not implemented for ValueQuantizedCache.")

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("Method `reorder_cache` is not implemented for ValueQuantizedCache.")

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("Method `to_legacy_cache` is not implemented for ValueQuantizedCache.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "ValueQuantizedCache":
        raise NotImplementedError("Method `from_legacy_cache` is not implemented for ValueQuantizedCache.")

class ValueQuantizedCacheV2(DynamicCache):
    def __init__(self, residual_length: int, bits: int) -> None:
        super().__init__()  # Initialize the base class
        # Only quantization factors and full precision cache for values, not keys
        self.value_scales_cache: List[torch.Tensor] = []
        self.value_zeros_cache: List[torch.Tensor] = []
        self.value_full_precision_cache: List[torch.Tensor] = []  # Full precision storage for values
        self.residual_length = residual_length  # Pre-defined residual length for quantization
        self.bits = bits  # Number of bits for quantization
    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the cache and quantization factors for a specific layer.

        Returns:
            A tuple containing:
            - key tensor
            - value tensor
            - value scales tensor
            - value zeros tensor
            - value full precision tensor
        """
        if layer_idx < len(self):
            return (
                self.key_cache[layer_idx], 
                self.value_cache[layer_idx],
                self.value_scales_cache[layer_idx],
                self.value_zeros_cache[layer_idx],
                self.value_full_precision_cache[layer_idx]
            )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(
        self,
        key_states: torch.Tensor,
        value_full_precision: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new `key_states`, `value_states`, `value_full_precision` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            layer_idx (`int`): The index of the layer to cache the states for.
            value_full_precision (`torch.Tensor`): The full precision value states.
            cache_kwargs (`Dict[str, Any]`, `optional`): Additional arguments for the cache subclass.

        Returns:
            A tuple containing the updated key and value states, value scales, value zeros, and value full precision.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        # Update the key cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)

        # Update the full precision cache
        if len(self.value_full_precision_cache) <= layer_idx:
            self.value_full_precision_cache.append(value_full_precision)
        elif self.value_full_precision_cache[layer_idx] is None:
            self.value_full_precision_cache[layer_idx] = value_full_precision
        else:
            self.value_full_precision_cache[layer_idx] = torch.cat(
                [self.value_full_precision_cache[layer_idx], value_full_precision], dim=-2
            )

        #print(value_full_precision.shape)
        
        # Perform quantization if full precision cache exceeds the residual length
        if self.value_full_precision_cache[layer_idx].shape[2] > self.residual_length:
            self.quantize_and_store(layer_idx)

        # Ensure value and quantization caches are updated
        return (
            self.key_cache[layer_idx], 
            self.value_cache[layer_idx],
            self.value_scales_cache[layer_idx],
            self.value_zeros_cache[layer_idx],
            self.value_full_precision_cache[layer_idx]
        )

    def quantize_and_store(self, layer_idx: int) -> None:
        """
        Quantizes the value_full_precision_cache if it exceeds the residual length and stores
        the quantized values, scales, and zeros into the corresponding caches.

        Parameters:
            layer_idx (`int`): The index of the layer to perform quantization on.
        """
        value_full_precision = self.value_full_precision_cache[layer_idx]
        assert len(value_full_precision.shape) == 4, "Value tensor must have 4 dimensions"
        
        current_length = value_full_precision.shape[2]
        # Calculate how much to quantize: leave the remainder (mod residual_length)
        quantize_length = (current_length // self.residual_length) * self.residual_length
        remainder_length = current_length % self.residual_length

        
        # Split the tensor into the part to quantize and the remainder
        to_quantize = value_full_precision[:, :, :quantize_length, :].contiguous()  # Part to be quantized
        if remainder_length > 0:
            remainder = value_full_precision[:, :, quantize_length:, :].contiguous()    # Part to remain in full precision

        # Perform quantization
        #print(value_full_precision.shape[-1])
        #quantized_value, scales, zeros = triton_quantize_and_pack_along_last_dim(to_quantize, value_full_precision.shape[-1], self.bits)
        quantized_value, scales, zeros = quant_and_pack_vcache(to_quantize, value_full_precision.shape[-1], self.bits)
        
        # Store quantized outputs
        if len(self.value_cache) <= layer_idx:
            self.value_cache.append(quantized_value)
            self.value_scales_cache.append(scales)
            self.value_zeros_cache.append(zeros)
        else:
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], quantized_value], dim=-2)
            self.value_scales_cache[layer_idx] = torch.cat([self.value_scales_cache[layer_idx], scales], dim=-2)
            self.value_zeros_cache[layer_idx] = torch.cat([self.value_zeros_cache[layer_idx], zeros], dim=-2)

        # Update the full precision cache with the remainder
        if remainder_length > 0:
            self.value_full_precision_cache[layer_idx] = remainder
        else:
            self.value_full_precision_cache[layer_idx] = None  # Clear the full precision cache
        #self.value_full_precision_cache[layer_idx] = remainder

    # def get_max_length(self) -> Optional[int]:
    #     raise NotImplementedError("Method `get_max_length` is not implemented for ValueQuantizedCache.")

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("Method `reorder_cache` is not implemented for ValueQuantizedCache.")

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("Method `to_legacy_cache` is not implemented for ValueQuantizedCache.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "ValueQuantizedCache":
        raise NotImplementedError("Method `from_legacy_cache` is not implemented for ValueQuantizedCache.")