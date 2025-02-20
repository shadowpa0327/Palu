import torch
from tqdm import tqdm
import numpy 
import argparse
from utils import load_model_and_tokenizer, add_common_args
import transformers
from loguru import logger

if __name__ == '__main__':
    torch.random.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    numpy.random.seed(1234)
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
    
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    assert model.config.model_type == "palullama", "Only Palu models are supported for this script."
    # for Palu
    if hasattr(model, "prepare_for_palu_inference"):
        model.prepare_for_palu_inference()
    model.eval()
    
    pipeline = transformers.pipeline(
        'text-generation', 
        model=model,
        tokenizer=tokenizer
    )
    output = pipeline(
        ["Please introduce youself (30 words)!\n I am a"],
        max_new_tokens=100,
        do_sample=True,      # Enable sampling
        temperature=0.7,     # Control randomness
        top_k=50,            # Top-k sampling
        top_p=0.9,           # Nucleus sampling
        repetition_penalty=1.2  # Prevent repetition
    )
    logger.info(output[0][0]['generated_text'])