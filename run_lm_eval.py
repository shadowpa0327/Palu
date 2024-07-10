
# Import necessary modules
from utils import load_model_and_tokenizer
import argparse
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

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
    print("=== Evaluation, Task(s): ", task_list)
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
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', 
        type=str, 
        help='model to load'
    )
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
    args = parser.parse_args()
    
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    run_lm_eval_zero_shot(model, tokenizer, args.batch_size, task_list=args.tasks)