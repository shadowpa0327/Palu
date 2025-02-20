import os
import numpy as np
import torch
from datasets import load_dataset
import random
from loguru import logger


def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
    cache_file = (
        f"cache/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}.pt"
    )
    random.seed(seed)
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        logger.info(f"[Calib data] Load from {cache_file}", fg="yellow")
        return traindataset
    if name == "c4":
        traindata = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="train",
        )
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    torch.save(traindataset, cache_file)
    return traindataset
