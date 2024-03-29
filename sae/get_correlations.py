import os, sys, gc
sys.path.append("mats_sae_training")

import torch as t
from torch import nn, Tensor
from torch.nn import functional as F

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import LMSparseAutoencoderSessionloader

import numpy as np
import einops
import matplotlib.pyplot as plt

import pickle
from dataclasses import dataclass
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple
from functools import partial
from itertools import cycle

from tqdm import tqdm, trange


device = t.device("cuda" if t.cuda.is_available() else "cpu")
MAIN = __name__ == "__main__"

def free_mem(vars):
    for v in vars: del v
    gc.collect()
    t.cuda.empty_cache()

@t.no_grad()
def get_corr(
    tokens: Int[Tensor, "batch seq"],
    model: HookedTransformer,
    autoencoder: SparseAutoencoder,
    layer: int,
    nonzero: bool
) -> Float[Tensor, "n_feature n_feature"]:
    '''
    Compute the correlation between all features.
    If nonzero, we consider when features do (not) fire together.
    If not nonzero, we consider the magnitude of each firing.
    '''
    hook_pt = f"blocks.{layer}.hook_resid_post"
    logits, cache = model.run_with_cache(tokens, names_filter=[hook_pt])
    logits, cache = autoencoder.run_with_cache(cache[hook_pt], names_filter=["hook_hidden_post"])
    acts = cache["hook_hidden_post"].reshape(-1, cache["hook_hidden_post"].shape[-1])
    free_mem([logits, cache])
    c = t.corrcoef((acts.T > 0).float()).cpu() if nonzero else t.corrcoef(acts.T).cpu()
    free_mem([acts])
    return c

datasets_ = ["Elriggs/openwebtext-100k", "NeelNanda/c4-code-20k", "maxtli/OpenWebText-2M"]
dataset = load_dataset(datasets_[2], split="train")
for layer in range(11, 12):
    print(f"\nLayer {layer}")
    REPO_ID = "jbloom/GPT2-Small-SAEs"
    FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model, sparse_autoencoder, activation_store = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
        path = path
    )
    model.eval()
    sparse_autoencoder.eval()
    autoencoder = sparse_autoencoder.autoencoders[0]

    tokenized_data = tokenize_and_concatenate(dataset, model.tokenizer, max_length=128)
    tokenized_data = tokenized_data.shuffle(123456)
    all_tokens = tokenized_data["tokens"][:1050]
    free_mem([tokenized_data])

    if not os.path.exists(f"correlations/layer{layer}.pt"):
        corrs = [
            get_corr(all_tokens[i:i+50], model, autoencoder, layer, False)
            for i in trange(0, 1000, 50)
        ]
        corrs = sum(corrs) / len(corrs)
        t.save(corrs, f"correlations/layer{layer}.pt")
        free_mem([corrs])
    if not os.path.exists(f"boolean_correlations/layer{layer}.pt"):
        corrs = [
            get_corr(all_tokens[i:i+50], model, autoencoder, layer, True)
            for i in trange(0, 1000, 50)
        ]
        corrs = sum(corrs) / len(corrs)
        t.save(corrs, f"boolean_correlations/layer{layer}.pt")
        free_mem([corrs])
    free_mem([model, sparse_autoencoder, activation_store, autoencoder, all_tokens])