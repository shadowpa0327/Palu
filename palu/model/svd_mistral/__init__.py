from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from .configuration_asvd_mistral import PaluMistralConfig
from .modeling_asvd_mistral import PaluMistralForCausalLM

AutoConfig.register("palumistral", PaluMistralConfig)
AutoModelForCausalLM.register(PaluMistralConfig, PaluMistralForCausalLM)
AutoTokenizer.register(PaluMistralConfig, LlamaTokenizer)

