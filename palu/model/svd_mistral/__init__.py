from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from .configuration_palu_mistral import PaluMistralConfig
from .modeling_palu_mistral import PaluMistralForCausalLM

AutoConfig.register("palumistral", PaluMistralConfig)
AutoModelForCausalLM.register(PaluMistralConfig, PaluMistralForCausalLM)
AutoTokenizer.register(PaluMistralConfig, LlamaTokenizer)

