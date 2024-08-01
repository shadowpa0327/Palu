from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from .configuration_palu_llama import PaluLlamaConfig
from .modeling_palu_llama import PaluLlamaForCausalLM

AutoConfig.register("palullama", PaluLlamaConfig)
AutoModelForCausalLM.register(PaluLlamaConfig, PaluLlamaForCausalLM)
AutoTokenizer.register(PaluLlamaConfig, LlamaTokenizer)

