#llama
from .svd_llama import (
    PaluLlamaConfig,
    PaluLlamaForCausalLM
)

#mistral
from .svd_mistral import (
    PaluMistralConfig,
    PaluMistralForCausalLM
)
#modules
from .modules import (
    HeadwiseLowRankModule
)

#TODO Mistral



AVAILABLE_MODELS = {
    'llama': {
        'config': PaluLlamaConfig,
        'ModelForCausalLM': PaluLlamaForCausalLM
    },
    'mistral': {
        'config': PaluMistralConfig,
        'ModelForCausalLM': PaluMistralForCausalLM
    }
}