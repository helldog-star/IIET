from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache


class DynamicCacheForODEformer(DynamicCache):

    def __init__(self) -> None:
        super().__init__()
        self.key_cache: List[List[torch.Tensor]] = []
        self.value_cache: List[List[torch.Tensor]] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        high_order_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0 and high_order_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append([key_states])
            self.value_cache.append([value_states])
        elif len(self.key_cache[layer_idx]) <= high_order_idx:
            self.key_cache[layer_idx].append(key_states)
            self.value_cache[layer_idx].append(value_states)
        else:
            self.key_cache[layer_idx][high_order_idx] = torch.cat([self.key_cache[layer_idx][high_order_idx], key_states], dim=-2)
            self.value_cache[layer_idx][high_order_idx] = torch.cat([self.value_cache[layer_idx][high_order_idx], value_states], dim=-2)

        return self.key_cache[layer_idx][high_order_idx], self.value_cache[layer_idx][high_order_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx][0].shape[-2]
    

def generate_text(prompt, model_name="/path/to/your/model", max_length=100):

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    past_key_values = DynamicCacheForODEformer()

    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, past_key_values=past_key_values)
        # output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)


    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    prompt = "Once upon a time"
    # your iiet or pcformer model ckpt
    model_name = "/your/path/ckpt"
    generated_text = generate_text(prompt, model_name)
    print("Generated Text:")
    print(generated_text)
