
# Import necessary modules
from utils import load_model_and_tokenizer, add_common_args
import argparse
import torch
import lm_eval
from tqdm import tqdm
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from lm_eval.utils import eval_logger as logger
from palu.quant_utils import configure_latent_quantizer
import os
import json

def run_lm_eval_zero_shot(model, tokenizer, batch_size=64, max_length=4096, task_list=["arc_easy", "hellaswag"], limit=None):
    model.seqlen = max_length
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, add_bos_token=False, batch_size=batch_size)
    # indexes all tasks from the lm_eval/tasks subdirectory.
    # Alternatively, you can set TaskManager(include_path="path/to/my/custom/task/configs")
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    # Setting task_manager to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in lm_eval/tasks.
    # simple_evaluate will instantiate its own task_manager is the it is set to None here.
    logger.info(f"Evaluation, Task(s): {task_list}")
    with torch.no_grad():
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model=lm_obj,
            #model_args= "add_bos_token=True" if model_type == "jamba" else "",
            tasks=task_list,
            task_manager=task_manager,
            log_samples=False,
            limit=limit
        ) 

    res = make_table(results)
    print(res)
    
    return results['results']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        '--tasks', type=lambda s: [item for item in s.split(',')], default=[],
        help='Task to be evaled'
    )
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help='batch size for lm_eval tasks'
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose information or not."
    )
    parser.add_argument(
        "--save_results",
        type=int,
        help="Whether to save the results or not."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the .json results."
    )
    
    args = parser.parse_args()  
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    configure_latent_quantizer(
        model, n_bits=args.lt_bits,
        group_size=args.lt_group_size,
        sym=args.lt_sym,
        clip_ratio=args.lt_clip_ratio,
        hadamard=args.lt_hadamard
    )
    logger.info("Start running lm_eval zero-shot evaluation...")
    res = run_lm_eval_zero_shot(model, tokenizer, args.batch_size, task_list=args.tasks)
    
    # Create directory if it doesn't exist
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    # Save results to JSON file
    model_name = args.model_name_or_path.split("/")[-1]
    output_file = os.path.join(output_dir, f"{model_name}_{args.lt_bits}{'_had' if args.lt_hadamard else ''}.json")
    with open(output_file, "w") as f:
        json.dump(res, f, indent=4)

    print(f"Results saved to {output_file}")
    