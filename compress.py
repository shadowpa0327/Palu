import argparse
import torch
import sys
from loguru import logger
from utils import set_seed, dump_to_huggingface_repos, load_model_and_tokenizer
from palu.rank_search import rank_search
from tqdm import tqdm
from palu.decomposition import compress_model_whiten

def compress(args):
    # set seed
    set_seed(args.seed)
    # load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    model.to(torch.device(args.device))
    # Step 1: Perform rank selection to get layer-wise compression rate
    search_results, rank_sum, total_rank = rank_search(model, tokenizer, args)
    # Step 2: Compress models
    compress_model_whiten(model, tokenizer, args, torch.device("cuda"), search_results)
    
    if args.dump_huggingface_model:
        save_folder = f"{args.model_id.split('/')[-1]}_ratio-{args.param_ratio_target}_gs-{args.head_group_size}-{args.search_method}"
        dump_to_huggingface_repos(model, tokenizer, save_folder, args)
        logger.info(f"Huggingface model is saved to {save_folder}", fg="green")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Pretrained model ID"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random Seed"
    )

    parser.add_argument(
        "--dump_huggingface_model", 
        action="store_true",
        help="Whether to dump huggingface model or not."
    )

    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Whether to use cached calibration results or not.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    parser.add_argument(
        "--n_fisher_calib_samples",
        type=int,
        default=32,
        help="Number of samples used for calibration.",
    )
    
    parser.add_argument(
        "--n_whiten_calib_samples",
        type=int,
        default=256,
        help="Number of samples used for calibration.",
    )

    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="Calibration dataset",
    )

    parser.add_argument(
        "--calib_seqlen",
        type=int,
        default=1024,
        help="Sequence length of the calibration dataset."
    )

    parser.add_argument(
        "--head_group_size",
        type=int,
        default=4,
        help="Group size for group-wise decomposition."
    )


    # Rank Search hyper-paramters
    parser.add_argument(
        "--param_ratio_target", 
        type=float,
        default=-1,
        help="Target param ratio"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose information or not."
    )
    
    parser.add_argument(
        "--search_method",
        type=str,
        default="STRS",
        choices=["fisher", "fisher_uniform", "uniform"],
        help="Search method",
    )
    
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    
    compress(args)