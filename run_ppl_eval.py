import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os
from utils import load_model_and_tokenizer, add_common_args
from palu.quant_utils import configure_latent_quantizer
from loguru import logger

def get_ppl_eval_loaders(name, tokenizer, seqlen=2048):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    elif "c4" in name:
        # Wrapper for tokenized input IDs
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
                
        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        testenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        testenc = testenc.input_ids[:, :(256 * seqlen)]
        testenc = TokenizerWrapper(testenc)
        return testenc
    else:
        raise NotImplementedError
    

def get_ppl_eval_loaders(name, tokenizer, seqlen=2048):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    elif "c4" in name:
        # Wrapper for tokenized input IDs
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
                
        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        #testenc = tokenizer("\n\n".join(valdata["text"]), return_tensors="pt")
        testenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        testenc = testenc.input_ids[:, :(256 * seqlen)]
        testenc = TokenizerWrapper(testenc)
        return testenc
    else:
        raise NotImplementedError

@torch.no_grad()
def eval_ppl(model, tokenizer, model_name, datasets, seqlen=2048, device="cuda"):
    model = model.to(device)
    if isinstance(device, str):
        device = torch.device(device)

    results = {}

    for dataset in datasets.split(","):
        cache_testloader = (
            f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
        )
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
        else:
            testloader = get_ppl_eval_loaders(dataset, tokenizer)
            torch.save(testloader, cache_testloader)
        
        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()

        nlls = []

        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(
                    device
                )
            outputs = model.model(batch)
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)  # .contiguous()
            shift_logits = logits[:, :-1, :]  # .contiguous()
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
            
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
        model.config.use_cache = use_cache
        results.update({dataset: ppl.item()})

    return results
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument('--datasets', type=str, help='datasets to evaluate', default='wikitext2')
    parser.add_argument('--seqlen', type=int, help='sequence length for ppl evaluation', default=2048)
    parser.add_argument("--device", type=str, help="device to run the model on", default="cuda")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose information or not.")
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    
    configure_latent_quantizer(
        model, n_bits=args.lt_bits, 
        group_size=args.lt_group_size, 
        sym=args.lt_sym, 
        clip_ratio=args.lt_clip_ratio, 
        hadamard=args.lt_hadamard
    )
    
    results = eval_ppl(model, tokenizer, args.model_name_or_path, args.datasets, args.seqlen, args.device)
    for dataset, ppl in results.items():
        print(f"Evaluation result for {dataset}:")
        print(f"Sequence length: {args.seqlen}")
        print(f"PPL: {ppl}")