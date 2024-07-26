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
from loguru import logger
os.environ["WANDB_DISABLED"] = "true"

from longbench_utils import scorer, MODEL2MAXLEN, DATASET2PROMPT, DATASET2MAXLEN
from utils import load_model_and_tokenizer, add_common_args
from palu.quant_utils import configure_latent_quantizer
import palu.model

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
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
        
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    configure_latent_quantizer(
        model, n_bits=args.lt_bits,
        group_size=args.lt_group_size,
        sym=args.lt_sym,
        clip_ratio=args.lt_clip_ratio,
        hadamard=args.lt_hadamard
    )
    model_name = args.model_name_or_path.split("/")[-1]
    orig_model_name = "Mistral-7B-v0.1" if "mistral" in model_name.lower() else "Llama-2-7b-chat-hf"
    if "mistral" in model_name.lower():
        orig_model_name = "Mistral-7B-v0.1"
    elif "llama" in model_name.lower():
        orig_model_name = "Llama-2-7b-chat-hf"
    elif "vicuna" in model_name.lower():
        orig_model_name = "vicuna-v1.5-7b-16k"
    
    model.eval()
    max_length = model2maxlen[orig_model_name]
    logger.info(f"Running model: {model_name}")
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
        start_time = time.time()
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time for dataset {dataset}: {elapsed_time/60} minutes")
            
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
        results[dataset] = {"score": score}

    os.makedirs("results/Longbench", exist_ok=True)
    with open(f"results/Longbench/{model_name}.txt", "w") as f:
        f.write(str(results))
    
if __name__ == '__main__':
    seed_everything(42)    
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        '--datasets', type=lambda s: [item for item in s.split(',')], 
        default=["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news"],
        help='The datasets to be evaluated'
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose information or not."
    )
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    
    main(args)
    
    