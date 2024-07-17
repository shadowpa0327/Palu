import argparse
import importlib
import numpy as np
import random, torch
from functools import reduce
from palu.model import HeadwiseLowRankModule
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

# Set seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_model_numel(model):
    param_cnt = 0
    for name, module in model.named_modules():
        if hasattr(module, '_nelement'):
            param_cnt += module._nelement()
    return param_cnt

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**3
    return size_all_mb


def get_module_by_name(module, module_name):
    names = module_name.split(sep='.')
    return reduce(getattr, names, module)



def dump_to_huggingface_repos(model, tokenizer, save_path, args):
    tokenizer.save_pretrained(save_path)
    #model.generation_config = Gene
    #if "vicuna" in model.config._name_or_path.lower():
        #NOTE(brian1009): Ad-hoc fixing the bug in Vicuna
        #model.config.generation_config = GenerationConfig(temperature=1.0, top_p=1.0)
    model.save_pretrained(save_path)
    config = model.config.to_dict()
    config["head_wise_ranks"] = {}
    for name, module in model.named_modules():
        if isinstance(module, HeadwiseLowRankModule):
            config["head_wise_ranks"][name] = module.ranks
    
    if "llama" in model.config._name_or_path.lower() or model.config.model_type == "llama":
        config["model_type"] = "palullama"
        config['architectures'] = ['PaluLlamaForCausalLM']
    elif "mistral" in model.config._name_or_path.lower():
        config["model_type"] = "palumistral"
        config['architectures'] = ['PaluMistralForCausalLM']
    else:
        raise NotImplementedError
            
    config["original_model_name_or_path"] = model.config._name_or_path
    import json

    json.dump(config, open(save_path + "/config.json", "w"), indent=2)
    
    
def load_model_and_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    # Fix the bug in generation configs
    #TODO: Add reference to the issue that also faced this bug
    if "vicuna" in model.config._name_or_path.lower():
        model.generation_config.do_sample = True
        
    return model, tokenizer



def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name_or_path', type=str, help='model to load')
    parser.add_argument('--lt_bits', type=int, help='Bits for low_rank latents. \
                        When lt_bits is less than 16, we quantize the low_rank latents in low_bits', default=16)
    parser.add_argument('--lt_group_size', type=int, help='Group size for low_rank latents', default=0)
    parser.add_argument('--lt_sym', action='store_true', help='Symmetric quantization for low_rank latents')
    parser.add_argument('--lt_clip_ratio', type=float, help='Clip ratio for low_rank latents', default=1.0)
    parser.add_argument('--lt_hadamard', action='store_true', help='Apply Hadamard transform to low_rank latents')
    return parser