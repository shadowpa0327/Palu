from loguru import logger
import torch.nn as nn
import torch
import os
import click
from tqdm import tqdm
from .data_utils import get_calib_data
from .model import HeadwiseLowRankModule

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def get_whiten_scale_matrix(model, tokenizer, args, dev):
    model_id = model.config._name_or_path
    #NOTE (brian1009): Might need to check the random seed, currently we have < 0.1 perplexity difference at Llama2-7B
    calib_loader = get_calib_data(
        "wikitext2", 
        tokenizer, 
        model_id, 
        nsamples=256, 
        seqlen=2048
    )
    cache_file = f"cache/whiten/{model_id.replace('/','_')}_w2_scaling_matrices_fp16.pt"
    os.makedirs("cache/whiten", exist_ok=True)
    """
    cache format:
    [
        {
            "attn.q_proj": torch.Tensor,
            "attn.k_proj": torch.Tensor,
            "attn.v_proj": torch.Tensor,
            "attn.o_proj": torch.Tensor,
            "mlp.gate_proj": torch.Tensor,
            "mlp.up_proj": torch.Tensor,
            "mlp.down_proj": torch.Tensor
        },
        ... (stacked n times, in the order of model layers)
    ]
    """
    logger.info(f"[whiten] Calibration dataset: {args.calib_dataset}", fg="yellow")
    logger.info(f"[whiten] Search cache_file={cache_file}", fg="yellow")
    if os.path.exists(cache_file) and args.use_cache:
        logger.info(f"[whiten] File {cache_file} exist.", fg="green")
        logger.info(f"[whiten] Load scaling diag matrix from cache: {cache_file}", fg="yellow")
        scaling_matrics = torch.load(cache_file, map_location="cpu")


        layers = model.model.layers
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            subset = find_layers(layer) # Collect all linear layers
            for name in subset:
                if name in scaling_matrics[i]:
                    scaling_diag_matrix = scaling_matrics[i][name]
                    subset[name].scaling_diag_matrix = scaling_diag_matrix

        return 
    
    logger.info(f"No cache_file={cache_file}", fg="red")
    logger.info(f"Create whiten scale matrix dict...", fg="yellow")

    # Create Scaling Matrix with low-resource inference
    # Adapted from https://github.com/AIoT-MLSys-Lab/SVD-LLM/blob/main/SVDLLM.py
    # Here, inference are performed in an layer-wise manner.
    use_cache = model.config.use_cache
    model.config.use_cache = False
    #FIXME: This is not a good implementation...
    if "llama" in model_id or "mistral" in model_id or "vicuna" in model_id or "longchat":
        layers = model.model.layers
    elif "opt" in model_id:
        layers = model.model.decoder.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    scaling_matrices = []
    logger.info("[Decomposition] Start to calculate the scaling matrix in layer-wise manner...")
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks, position_ids=position_ids[0].unsqueeze(0))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        layer_scaling_matrices = {}
        for name in subset:
            if not ("k_proj" in name or "v_proj" in name):
                continue
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().cuda()
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                subset[name].scaling_diag_matrix = scaling_diag_matrix
            except Exception as e:
                logger.warning("eigen scaling_diag_matrix is not positive!")
                if torch.isnan(raw_scaling_diag_matrix).any():
                    logger.warning("raw scaling_diag_matrix contains NaN!")
                elif torch.isinf(raw_scaling_diag_matrix).any():
                    logger.warning("raw scaling_diag_matrix contains Inf!")
                if not torch.equal(raw_scaling_diag_matrix, raw_scaling_diag_matrix.T):
                    logger.warning("raw scaling_diag_matrix is not a symmetric matrix!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-3) * torch.eye(raw_scaling_diag_matrix.shape[0]).cuda()
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                if torch.isnan(scaling_diag_matrix).any():
                    logger.warning("scaling_diag_matrix contains NaN!")
                elif torch.isinf(scaling_diag_matrix).any():
                    logger.warning("scaling_diag_matrix contains Inf!")
                del eigenvalues
                subset[name].scaling_diag_matrix = scaling_diag_matrix
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                logger.warning("scaling_diag_matrix is not full rank!")
                reg_inv =  1e-3 * torch.eye(scaling_diag_matrix.shape[0]).cuda() 
                scaling_diag_matrix += reg_inv
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                del reg_inv
            
            del scaling_matrix_inv
            layer_scaling_matrices[name] = scaling_diag_matrix.cpu()
            torch.cuda.empty_cache()
        scaling_matrices.append(layer_scaling_matrices)
        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()
        
    model.config.use_cache = use_cache
    if args.use_cache:
        torch.save(scaling_matrices, cache_file)
        logger.info(f"Save the whiten scale matrix dict to:  {cache_file}")

def compress_model_whiten(model, tokenizer, args, dev, selection_result):
    logger.info("Compressing model with whiten decomposition...")
    # NOTE(brian1009): Prepare whiten scaling matrix
    get_whiten_scale_matrix(model, tokenizer, args, dev)
    # Compress the model
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    logger.info(f"Start decompose the layer with selected ranks... #target layers: {len(selection_result.keys())}")
    for layername, selected_head_rank in tqdm(selection_result.items()):
        logger.debug(f"Decompose {layername} with ranks: {selected_head_rank}")
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
    
        head_wise_svd_linear = HeadwiseLowRankModule.from_linear_whiten(
            raw_linear,
            selected_head_rank
        )
        setattr(info["father"], info["name"],  head_wise_svd_linear)

def compress_model_svd(model, selection_result):
    logger.info("Compressing model with svd decomposition...")
    # Compress the model
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    logger.info(f"Start decompose the layer with selected ranks... #target layers: {len(selection_result.keys())}")
    for layername, selected_head_rank in tqdm(selection_result.items()):
        logger.debug(f"Decompose {layername} with ranks: {selected_head_rank}")
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        print("head-wise svd", layername, raw_linear)
        head_wise_svd_linear = HeadwiseLowRankModule.from_linear(
            raw_linear,
            selected_head_rank
        )
        setattr(info["father"], info["name"],  head_wise_svd_linear)

# Wrapper for different decompose methods
def compress_model(model, tokenizer, args, dev, selection_result):
    if args.decompose_method == "whiten":
        compress_model_whiten(model, tokenizer, args, dev, selection_result)
    elif args.decompose_method == "svd":
        compress_model_svd(model, selection_result)
    else:
        raise ValueError(f"Decomposition method {args.decompose_method} is not supported.")