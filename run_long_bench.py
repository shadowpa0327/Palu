#Adapted from https://github.com/THUDM/LongBench/blob/main/pred.py
import os
from datasets import load_dataset
import torch
from loguru import logger
from tqdm import tqdm
import numpy as np
import random
import argparse
import time
import json
from loguru import logger
from datetime import datetime
os.environ["WANDB_DISABLED"] = "true"

from longbench_utils import scorer, MODEL2MAXLEN, DATASET2PROMPT, DATASET2MAXLEN
from utils import load_model_and_tokenizer, add_common_args
from palu.quant.quant_utils import configure_latent_quantizer
import palu.model
from token_pruning.monkeypatch import replace_llama,replace_mistral
from token_pruning.token_prune_utils import setup_token_sparse_params


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # Copy from KIVI
    if "longchat" in model_name.lower() or "vicuna" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "llama-3-8b-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

@torch.no_grad()
def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, 
             #parameter for token sparsity
             prompt_sparse_method=None, prompt_sparse_ratio=-1, prompt_capacity=-1):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        
        if prompt_sparse_method is not None:
            setup_token_sparse_params(model, context_length, prompt_sparse_method, prompt_sparse_ratio, prompt_capacity)
                
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def main(args):
    model2maxlen = MODEL2MAXLEN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, use_flash_attn2=args.flash2)
    configure_latent_quantizer(
        model, n_bits=args.lt_bits,
        group_size=args.lt_group_size,
        sym=args.lt_sym,
        clip_ratio=args.lt_clip_ratio,
        hadamard=args.lt_hadamard
    )
    #NOTE(brian1009): This is a hack to get the model name
    # We assume the model name is the inside the last part of the path
    # and the Palu's compression information is follow by the model name with a "_"
    # Hence, we split the path by "/" and then keep only the first part by "_"
    # Example: Mistral-7B-Instruct-v0.2_ratio-0.7_gs-4-fisher_uniform
    raw_model_name = args.model_name_or_path.split("/")[-1]
    model_type = args.model_name_or_path.split("/")[-1].split('_')[0]
        
    model.eval()
    if not model_type in model2maxlen:
        raise ValueError(f"Model {model_type} not supported")
    
    max_length = model2maxlen[model_type]
    logger.info(f"Running model: {raw_model_name}")
    logger.info(f"Max length: {max_length}")
    datasets = args.datasets
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = DATASET2PROMPT
    dataset2maxlen = DATASET2MAXLEN
    # predict on each dataset
    if not os.path.exists("Longbench/pred"):
        os.makedirs("Longbench/pred")
    
    results = {}
    
    for dataset in datasets:
        logger.info("Evaluating dataset: {}".format(dataset))
        start_time = time.time()
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_type, args.token_sparse_method, args.prompt_capacity_rate, args.prompt_capacity)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time for dataset {dataset}: {elapsed_time/60} minutes")
        
        # calculate score
        predictions, answers, lengths = [], [], []
        for pred in preds:
            predictions.append(pred["pred"])
            answers.append(pred["answers"])
            if "length" in pred:
                lengths.append(pred["length"])
            all_classes = pred["all_classes"]
        score = scorer(dataset, predictions, answers, all_classes)
        logger.info(f"dataset: {dataset}")
        logger.info(f"score: {score}")

        # Log the results of each datasets
        file_name = f"{raw_model_name}_bits_{args.lt_bits}"
        if args.token_sparse_method is not None:
            file_name += f"_sparse_{args.token_sparse_method}"
            if args.prompt_capacity_rate > 0:
                file_name += f"_ratio_{args.prompt_capacity_rate}"
            else:
                file_name += f"_capacity_{args.prompt_capacity}"
        with open(f"results/Longbench/{file_name}.json", "a") as f:
            data_to_log = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": dataset,
                "score": score,
            }
            json.dump(data_to_log, f)
            f.write("\n")
    
if __name__ == '__main__':
    seed_everything(42)    
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        '--datasets', type=lambda s: [item for item in s.split(',')], 
        default=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"],
        help='The datasets to be evaluated'
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose information or not."
    )
    parser.add_argument(
        "--token_sparse_method",
        default=None,
        type=str,
        help="Token sparsity method to applied."
    )
    parser.add_argument(
        '--prompt_capacity_rate',
        default=-1,
        type=float,
    )
    parser.add_argument(
        '--prompt_capacity',
        default=-1,
        type=int,
    )
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    #Create directory to log evaluation results.
    os.makedirs("results/Longbench", exist_ok=True)
    
    if args.token_sparse_method is not None:
        assert args.prompt_capacity_rate >= 0 and args.prompt_capacity_rate <= 1 or args.prompt_capacity > 0, "Invalid prompt drop rate"
        logger.info(f"Token sparsity method: {args.token_sparse_method}")
        logger.info(f"Token sparsity ratio: {args.prompt_capacity_rate}")
        logger.info(f"Token sparsity capacity: {args.prompt_capacity}")
        replace_llama(args.token_sparse_method)
        replace_mistral(args.token_sparse_method)
    
    main(args)
    
    