from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen2Tokenizer
from .configuration_palu_qwen import PaluQwen2Config
from .modeling_palu_qwen import PaluQwen2ForCausalLM

AutoConfig.register("paluqwen2", PaluQwen2Config)
AutoModelForCausalLM.register(PaluQwen2Config, PaluQwen2ForCausalLM)
AutoTokenizer.register(PaluQwen2Config, Qwen2Tokenizer)

