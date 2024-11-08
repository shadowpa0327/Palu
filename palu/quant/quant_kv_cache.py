from transformers.cache_utils import DynamicCache
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from .q_packing import triton_quantize_and_pack_along_last_dim, quant_and_pack_vcache

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
        #NOTE(brian1009): # Transpose and make it contiguous to match the requirements of Kernel that is going to consume this tensor
        quantized_value = quantized_value.transpose(3, 2).contiguous()
        # Store quantized outputs
        if len(self.value_cache) <= layer_idx:
            self.value_cache.append(quantized_value)
            self.value_scales_cache.append(scales)
            self.value_zeros_cache.append(zeros)
        else:
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], quantized_value], dim=-1)
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
    
class KeyValueQuantizedCacheV2(DynamicCache):
    def __init__(self, residual_length: Union[int, Tuple[int, int]], bits: int) -> None:
        super().__init__()  # Initialize the base class
        # Only quantization factors and full precision cache for values, not keys
        self.key_scales_cache: List[torch.Tensor] = []
        self.key_zeros_cache: List[torch.Tensor] = []
        self.key_full_precision_cache: List[torch.Tensor] = []  # Full precision storage for values
        self.value_scales_cache: List[torch.Tensor] = []
        self.value_zeros_cache: List[torch.Tensor] = []
        self.value_full_precision_cache: List[torch.Tensor] = []  # Full precision storage for values
        # Pre-defined residual length for quantization
        if isinstance(residual_length, int):
            self.residual_length_k = self.residual_length_v = residual_length
        else:
            self.residual_length_k, self.residual_length_v = residual_length

        self.bits = bits  # Number of bits for quantization
    
    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the cache and quantization factors for a specific layer.

        Returns:
            A tuple containing:
            - key tensor
            - key tensor
            - key scales tensor
            - key zeros tensor
            - key full precision tensor
            - value tensor
            - value scales tensor
            - value zeros tensor
            - value full precision tensor
        """
        if layer_idx < len(self):
            return (
                self.key_cache[layer_idx],
                self.key_scales_cache[layer_idx],
                self.key_zeros_cache[layer_idx],
                self.key_full_precision_cache[layer_idx],
                self.value_cache[layer_idx],
                self.value_scales_cache[layer_idx],
                self.value_zeros_cache[layer_idx],
                self.value_full_precision_cache[layer_idx]
            )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(
        self,
        key_full_precision: torch.Tensor,
        value_full_precision: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new `key_states`, `value_states`, `key_full_precision`, `value_full_precision` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            layer_idx (`int`): The index of the layer to cache the states for.
            key_full_precision (`torch.Tensor`): The full precision key states.
            value_full_precision (`torch.Tensor`): The full precision value states.
            cache_kwargs (`Dict[str, Any]`, `optional`): Additional arguments for the cache subclass.

        Returns:
            A tuple containing the updated key and value states, value scales, value zeros, and value full precision.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            # TODO: Check the correctness 
            self.seen_tokens += key_full_precision.shape[-2]

        # Update the full precision cache
        if len(self.key_full_precision_cache) <= layer_idx:
            self.key_full_precision_cache.append(key_full_precision)
        elif self.key_full_precision_cache[layer_idx] is None:
            self.key_full_precision_cache[layer_idx] = key_full_precision
        else:
            self.key_full_precision_cache[layer_idx] = torch.cat(
                [self.key_full_precision_cache[layer_idx], key_full_precision], dim=-2
            )
        if len(self.value_full_precision_cache) <= layer_idx:
            self.value_full_precision_cache.append(value_full_precision)
        elif self.value_full_precision_cache[layer_idx] is None:
            self.value_full_precision_cache[layer_idx] = value_full_precision
        else:
            self.value_full_precision_cache[layer_idx] = torch.cat(
                [self.value_full_precision_cache[layer_idx], value_full_precision], dim=-2
            )
        
        # Perform quantization if full precision cache exceeds the residual length
        if self.key_full_precision_cache[layer_idx].shape[-2] > self.residual_length_k:
            self.quantize_and_store_key(layer_idx)
        if self.value_full_precision_cache[layer_idx].shape[-2] > self.residual_length_v:
            self.quantize_and_store_value(layer_idx)
            
        # Ensure value and quantization caches are updated
        return (
            self.key_cache[layer_idx],
            self.key_scales_cache[layer_idx],
            self.key_zeros_cache[layer_idx],
            self.key_full_precision_cache[layer_idx],
            self.value_cache[layer_idx],
            self.value_scales_cache[layer_idx],
            self.value_zeros_cache[layer_idx],
            self.value_full_precision_cache[layer_idx]
        )

    def quantize_and_store_key(self, layer_idx: int) -> None:
        """
        Quantizes the key_full_precision_cache if it exceeds the residual length and stores
        the quantized values, scales, and zeros into the corresponding caches.

        Parameters:
            layer_idx (`int`): The index of the layer to perform quantization on.
        """
        key_full_precision = self.key_full_precision_cache[layer_idx]
        assert len(key_full_precision.shape) == 4, "Key tensor must have 4 dimensions"
        
        current_length = key_full_precision.shape[-2]
        # Calculate how much to quantize: leave the remainder (mod residual_length)
        quantize_length = (current_length // self.residual_length_k) * self.residual_length_k
        remainder_length = current_length % self.residual_length_k

        
        # Split the tensor into the part to quantize and the remainder
        k_to_quantize = key_full_precision[:, :, :quantize_length, :].contiguous()  # Part to be quantized
        if remainder_length > 0:
            k_remiander = key_full_precision[:, :, quantize_length:, :].contiguous()    # Part to remain in full precision

        # Perform quantization
        # NOTE(max410011): Use `quant_and_pack_vcache` because the key tensor is quantized along the last dimension too
        # k_quantized, k_scales, k_zeros = quant_and_pack_vcache(k_to_quantize, key_full_precision.shape[-1], self.bits)
        k_quantized, k_scales, k_zeros = triton_quantize_and_pack_along_last_dim(k_to_quantize, key_full_precision.shape[-1], self.bits)
        
        # NOTE(brian1009): Transpose and make it contiguous to match the requirements of Kernel that is going to consume this tensor
        # NOTE(max410011): No need for key
        # k_quantized = k_quantized.transpose(3, 2).contiguous().transpose(2, 3)
        
        # Store quantized outputs
        if len(self.value_cache) <= layer_idx:
            self.key_cache.append(k_quantized)
            self.key_scales_cache.append(k_scales)
            self.key_zeros_cache.append(k_zeros)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k_quantized], dim=-2)
            self.key_scales_cache[layer_idx] = torch.cat([self.key_scales_cache[layer_idx], k_scales], dim=-2)
            self.key_zeros_cache[layer_idx] = torch.cat([self.key_zeros_cache[layer_idx], k_zeros], dim=-2)

        # Update the full precision cache with the remainder
        if remainder_length > 0:
            self.key_full_precision_cache[layer_idx] = k_remiander
        else:
            self.key_full_precision_cache[layer_idx] = None  # Clear the full precision cache

    def quantize_and_store_value(self, layer_idx: int) -> None:
        """
        Quantizes the value_full_precision_cache if it exceeds the residual length and stores
        the quantized values, scales, and zeros into the corresponding caches.

        Parameters:
            layer_idx (`int`): The index of the layer to perform quantization on.
        """
        value_full_precision = self.value_full_precision_cache[layer_idx]
        assert len(value_full_precision.shape) == 4, "Value tensor must have 4 dimensions"
        
        current_length = value_full_precision.shape[-2]
        # Calculate how much to quantize: leave the remainder (mod residual_length)
        quantize_length = (current_length // self.residual_length_v) * self.residual_length_v
        remainder_length = current_length % self.residual_length_v

        
        # Split the tensor into the part to quantize and the remainder
        v_to_quantize = value_full_precision[:, :, :quantize_length, :].contiguous()  # Part to be quantized
        if remainder_length > 0:
            v_remainder = value_full_precision[:, :, quantize_length:, :].contiguous()    # Part to remain in full precision

        # Perform quantization
        # NOTE(max410011): Use `quant_and_pack_vcache` because the key tensor is quantized along the last dimension too
        v_quantized, v_scales, v_zeros = quant_and_pack_vcache(v_to_quantize, value_full_precision.shape[-1], self.bits)
        # v_quantized, v_scales, v_zeros = triton_quantize_and_pack_along_last_dim(v_to_quantize, value_full_precision.shape[-1], self.bits)
        
        # NOTE(brian1009): Transpose and make it contiguous to match the requirements of Kernel that is going to consume this tensor
        # NOTE(max410011): No need for key
        # k_quantized = k_quantized.transpose(3, 2).contiguous().transpose(2, 3)
        
        # Store quantized outputs
        if len(self.value_cache) <= layer_idx:
            # FIXME(max410011): Move to above?
            v_quantized = v_quantized.transpose(3, 2).contiguous().transpose(2, 3)
            v_scales = v_scales.transpose(3, 2).contiguous().transpose(2, 3)
            v_zeros = v_zeros.transpose(3, 2).contiguous().transpose(2, 3)

            self.value_cache.append(v_quantized)
            self.value_scales_cache.append(v_scales)
            self.value_zeros_cache.append(v_zeros)
        else:
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v_quantized], dim=-2)
            self.value_scales_cache[layer_idx] = torch.cat([self.value_scales_cache[layer_idx], v_scales], dim=-2)
            self.value_zeros_cache[layer_idx] = torch.cat([self.value_zeros_cache[layer_idx], v_zeros], dim=-2)
            
            # FIXME(max410011): Move to above?
            self.value_cache[layer_idx] = self.value_cache[layer_idx].transpose(3, 2).contiguous().transpose(2, 3)
            self.value_scales_cache[layer_idx] = self.value_scales_cache[layer_idx].transpose(3, 2).contiguous().transpose(2, 3)
            self.value_zeros_cache[layer_idx] = self.value_zeros_cache[layer_idx].transpose(3, 2).contiguous().transpose(2, 3)

        # Update the full precision cache with the remainder
        if remainder_length > 0:
            self.value_full_precision_cache[layer_idx] = v_remainder
        else:
            self.value_full_precision_cache[layer_idx] = None  # Clear the full precision cache


    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        seq_length = 0
        if self.key_cache[layer_idx] is not None:
            seq_length += self.key_cache[layer_idx].shape[-2]
        if self.key_full_precision_cache[layer_idx] is not None:
            seq_length += self.key_full_precision_cache[layer_idx].shape[-2]
        return seq_length

    # def get_max_length(self) -> Optional[int]:
    #     raise NotImplementedError("Method `get_max_length` is not implemented for KeyValueQuantizedCacheV2.")

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("Method `reorder_cache` is not implemented for KeyValueQuantizedCacheV2.")

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("Method `to_legacy_cache` is not implemented for KeyValueQuantizedCacheV2.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "KeyValueQuantizedCacheV2":
        raise NotImplementedError("Method `from_legacy_cache` is not implemented for KeyValueQuantizedCacheV2.")
        
class FastValueDynamicCache(DynamicCache):
    def __init__(self, residual_length: int) -> None:
        super().__init__()  # Initialize the base class
        self.residual_length = residual_length  # Pre-defined residual length for managing the cache

        # Maintain a single cache for keys
        self.key_cache: List[Optional[torch.Tensor]] = []

        # Maintain two caches for values: `Recent` and `Main`
        self.value_recent_cache: List[Optional[torch.Tensor]] = []  # Temporary cache for recent tokens (not yet moved to `Main`)
        self.value_main_cache: List[Optional[torch.Tensor]] = []  # Cumulative cache for storing moved tokens

    def __getitem__(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get the cache for a specific layer.

        Returns:
            A tuple containing:
            - key tensor
            - recent value tensor
            - main value tensor
        """
        if layer_idx < len(self.key_cache):
            return (
                self.key_cache[layer_idx],
                self.value_recent_cache[layer_idx],
                self.value_main_cache[layer_idx]
            )
        else:
            raise KeyError(f"Cache only has {len(self.key_cache)} layers, attempted to access layer with index {layer_idx}")

    def update(
        self,
        key_full_precision: torch.Tensor,
        value_full_precision: torch.Tensor,
        layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Updates the cache with new `key_full_precision`, `value_full_precision` for the layer `layer_idx`.

        If the number of tokens in the recent value cache exceeds the residual length, they are moved to the main value cache.

        Parameters:
            key_full_precision (`torch.Tensor`): The full precision key states.
            value_full_precision (`torch.Tensor`): The full precision value states.
            layer_idx (`int`): The index of the layer to cache the states for.

        Returns:
            A tuple containing the updated key and value states:
            - key tensor
            - recent value tensor
            - main value tensor
        """
        # Initialize the caches for the layer if they don't exist
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_full_precision)
            self.value_recent_cache.append(value_full_precision)
            self.value_main_cache.append(None)  # None placeholder for the main value cache
        else:
            # Update key cache
            if self.key_cache[layer_idx] is None:
                self.key_cache[layer_idx] = key_full_precision
            else:
                # Concatenate the new keys into the key cache
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_full_precision], dim=-2
                )

            # Update recent value cache
            if self.value_recent_cache[layer_idx] is None:
                self.value_recent_cache[layer_idx] = value_full_precision
            else:
                # Concatenate the new values into the recent value cache
                self.value_recent_cache[layer_idx] = torch.cat(
                    [self.value_recent_cache[layer_idx], value_full_precision], dim=-2
                )

        # If the number of tokens in the recent value cache exceeds the residual length, move them to the main cache
        if self.value_recent_cache[layer_idx].shape[-2] >= self.residual_length:
            self.move_to_main_cache(layer_idx)

        # Return the updated caches
        return (
            self.key_cache[layer_idx],
            self.value_main_cache[layer_idx],
            self.value_recent_cache[layer_idx]
        )

    def move_to_main_cache(self, layer_idx: int) -> None:
        """
        Move the recent value cache tokens to the main value cache for the specified layer.

        This method is called when the number of tokens in the recent value cache exceeds the residual length.
        """
        value_recent = self.value_recent_cache[layer_idx]

        if value_recent is None:
            return

        # Calculate how many tokens to move (multiple of residual_length)
        move_length = (value_recent.shape[-2] // self.residual_length) * self.residual_length
        remainder_length = value_recent.shape[-2] % self.residual_length

        if move_length > 0:
            value_to_move = value_recent[:, :, :move_length, :].contiguous()

            # Concatenate the tokens to the main value cache
            if self.value_main_cache[layer_idx] is None:
                self.value_main_cache[layer_idx] = value_to_move
            else:
                self.value_main_cache[layer_idx] = torch.cat(
                    [self.value_main_cache[layer_idx], value_to_move], dim=-2
                )

            # Update the recent value cache with the remainder
            if remainder_length > 0:
                self.value_recent_cache[layer_idx] = value_recent[:, :, move_length:, :].contiguous()
            else:
                # Reset the recent value cache to None if no remainder
                self.value_recent_cache[layer_idx] = None

        # Note: The key cache remains unchanged as we maintain a single key cache

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states."""
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("Method `reorder_cache` is not implemented for FastDynamicCache.")

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("Method `to_legacy_cache` is not implemented for FastDynamicCache.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "FastDynamicCache":
        raise NotImplementedError("Method `from_legacy_cache` is not implemented for FastDynamicCache.")
