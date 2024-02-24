# %%
import torch
from transformer_lens import HookedTransformer
import numpy as np 
from tqdm import tqdm
from fancy_einsum import einsum
from einops import rearrange
import math
from functools import partial
import torch.optim
import time
from encoders import UntiedEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, tuned_lens_hook, LinePlot

# %%

# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 200
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

# inverse probe setting

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model
lr = 1e-3

# # learning hyperparameters
# convergence_tol = 1e-4
# similarity_tol = .05
# lr_act = 1e-4
# lr_feat = 1e-5
# updates_per_batch = 100
# relu = torch.nn.ReLU()
kl_loss = torch.nn.KLDivLoss(reduction="none")

resid_points_filter = lambda layer_no, name: name == f"blocks.{layer_no}.hook_resid_pre"

# %%

tuned_lens_weights = [torch.nn.Parameter(torch.rand(d_model, d_model).to(device)) for _ in range(n_layers)]
tuned_lens_bias = [torch.nn.Parameter(torch.rand(d_model,).to(device)) for _ in range(n_layers)]
tuned_lens_optimizer = torch.optim.SGD([*tuned_lens_weights, *tuned_lens_bias], lr=lr, weight_decay=1e-3)

for param in model.parameters():
    param.requires_grad = False

# %%
    
def get_tuned_lens_loss(batch):
    activation_storage = []

    target_probs = model.run_with_hooks(
            batch,
            fwd_hooks=[
                (partial(resid_points_filter, layer_no), 
                   partial(tuned_lens_hook,
                           activation_storage,
                           tuned_lens_weights[layer_no],
                           tuned_lens_bias[layer_no])
                    ) for layer_no in range(n_layers)
                ]
    )[:,-1].softmax(dim=-1)
    # activation_storage is array of tensors batch_size x seq_len x d_model, len=n_layers
    # (batch_size * n_layers) x seq_len x d_model
    # batch_size x n_layers x d_model
    residual = model.ln_final(torch.stack(activation_storage, dim=1))

    # n_layers x batch_size x d_model
    # residual = model.unembed(residual)[:,-1].softmax(dim=-1).unflatten(0, (batch_size, n_layers)).permute((1,0,2))
    residual = model.unembed(residual).softmax(dim=-1).permute((1,0,2))

    # n_layers x batch_size
    kl_losses = kl_loss(residual.log(), target_probs).sum(dim=-1)

    return kl_losses, activation_storage


# %%

lp = LinePlot(['kl_loss', 'step_size'])
    
i = 0
while i < 3000:
    batch = next(owt_iter)['tokens']
    tuned_lens_optimizer.zero_grad()

    kl_losses, _ = get_tuned_lens_loss(batch)
    loss = kl_losses.sum()
    loss.backward()
    prev_weights = torch.stack(tuned_lens_weights, dim=0).detach()

    tuned_lens_optimizer.step()

    step_sz = (torch.stack(tuned_lens_weights, dim=0)-prev_weights).abs().sum()
    lp.add_entry({"kl_loss": loss.item(), "step_size": step_sz.item()})

    if i % 100 == 0:
        lp.plot()
    
    i += 1

# %%
    
with open("pruning/tuned_lens_weights.pkl", "wb") as f:
    pickle.dump(tuned_lens_weights, f)
with open("pruning/tuned_lens_bias.pkl", "wb") as f:
    pickle.dump(tuned_lens_bias, f)

# %%

with open("pruning/tuned_lens_weights.pkl", "rb") as f:
    tuned_lens_weights = pickle.load(f)
with open("pruning/tuned_lens_bias.pkl", "rb") as f:
    tuned_lens_bias = pickle.load(f)

# %%
# Confidence probes
    
# Question 1. what makes models confident in the beginning?
# Avg contribution from later attention head -- is it smaller in magnitude, systematically?

# Question 2. is there suppression and does it occur in later layer or before?
# If before, we can try to predict it with a probe.

probe_lr = 1e-3
    
probe_direction = torch.nn.Parameter(torch.randn(n_layers, d_model).to(device))
probe_bias = torch.nn.Parameter(torch.randn(n_layers,).to(device))

probe_optimizer = torch.optim.SGD([probe_direction, probe_bias], lr=probe_lr, weight_decay=0)

# %%
torch.autograd.set_detect_anomaly(True)

# %%
lp = LinePlot([f"probe_loss_{i}" for i in range(n_layers)])
i = 0
while i < 1000:
    batch = next(owt_iter)['tokens']

    with torch.no_grad():
        tuned_lens_acc, activation_storage = get_tuned_lens_loss(batch)

    tuned_lens_acc = tuned_lens_acc.detach()
    tuned_lens_acc.requires_grad = True

    activation_storage = torch.stack(activation_storage, dim=0).detach()
    activation_storage.requires_grad = True
    # n_layers x batch_size
    err_estimate = einsum("n_layer d_model, n_layer batch_size d_model -> n_layer batch_size", probe_direction, activation_storage) + probe_bias.unsqueeze(1)

    # loss = (probe_direction - 1).square()
    loss = (err_estimate - tuned_lens_acc).abs()
    loss.sum().backward()

    lp.add_entry({f"probe_loss_{i}": loss[i].mean().item() for i in range(n_layers)})

    probe_optimizer.step()

    print(probe_direction.isnan().sum())

    if i % 100 == 0:
        lp.plot(twinx=False)
    i += 1

# activation_storage = []

# # get the result at unembed
# target_probs = model.run_with_hooks(
#             batch,
#             fwd_hooks=[
#                 (partial(resid_points_filter, layer_no), 
#                    partial(tuned_lens_hook,
#                            activation_storage,
#                            tuned_lens_weights[layer_no],
#                            tuned_lens_bias[layer_no])
#                     ) for layer_no in range(n_layers)
#                 ]
#     )[:,-1].softmax(dim=-1)

# fit beta model?




# %%
