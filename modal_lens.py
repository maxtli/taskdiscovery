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
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from training_utils import load_model_data, save_hook_last_token, LinePlot

# %%
sns.set()
folder="modal_lens/no_decay"
shared_bias = False
# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 200
clip_value = 1e5
device, model, tokenizer, owt_iter = load_model_data(model_name, batch_size)

# inverse probe setting

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_dim = model.cfg.d_head
d_model = model.cfg.d_model
lr = 1e-2

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

prior_bias = [
    model.blocks[i].attn.b_O.clone() for i in range(n_layers)
]

attn_bias = [
    # torch.nn.Parameter(torch.ones((i+1, d_model)).to(device)) for i in range(n_layers)
    torch.nn.Parameter((prior_bias[i] if shared_bias else prior_bias[i].repeat(i+1,1)).to(device)) for i in range(n_layers)
]
lens_optimizer = torch.optim.AdamW(attn_bias, lr=lr, weight_decay=0)
# lens_scheduler = torch.optim.lr_scheduler.StepLR(lens_optimizer, step_size=200, gamma=0.9)

for param in model.parameters():
    param.requires_grad = False

for p in attn_bias:
    p.register_hook(lambda grad: torch.nan_to_num(grad, nan=0, posinf=0, neginf=0))


# %%
    
def get_lens_loss(batch, compare_tuned_lens=False):
    activation_storage = []

    target_probs = model.run_with_hooks(
            batch,
            fwd_hooks=[
                *[(partial(resid_points_filter, layer_no), 
                   partial(save_hook_last_token, activation_storage),
                    ) for layer_no in range(n_layers)],
                ]
    )[:,-1].softmax(dim=-1)

    # activation_storage: batch x d_model (last token only)
    # resid: layer x batch x d_model (note: layer norms and unembeds need 3 tensor dimensions)
    resid = []
    for layer_no in range(n_layers):
        if layer_no > 0:
            resid = torch.cat([resid_mid,activation_storage[layer_no].unsqueeze(0)], dim=0)
        else:
            resid = activation_storage[layer_no].unsqueeze(0)
        if shared_bias:
            attn_bias_layer = attn_bias[layer_no].unsqueeze(0)
        else:
            attn_bias_layer = attn_bias[layer_no]
        resid_mid = resid + attn_bias_layer.unsqueeze(1)
        normalized_resid_mid = model.blocks[layer_no].ln2(resid_mid)
        mlp_out = model.blocks[layer_no].mlp(normalized_resid_mid)
        resid = resid_mid + mlp_out
    
    resid = model.ln_final(resid)
    
    # layer x batch x d_vocab
    logits = model.unembed(resid).softmax(dim=-1)

    kl_losses = kl_loss(logits.log(), target_probs).sum(dim=-1)

    if compare_tuned_lens:
        tuned_lens_resid = einsum("layer result activation, layer batch activation -> layer batch result", torch.stack(tuned_lens_weights, dim=0), torch.stack(activation_storage, dim=0)) + torch.stack(tuned_lens_bias,dim=0).unsqueeze(1)
        tuned_lens_resid = model.ln_final(tuned_lens_resid)
        tuned_lens_logits = model.unembed(tuned_lens_resid).softmax(dim=-1)
        tuned_lens_losses = kl_loss(tuned_lens_logits.log(), target_probs).sum(dim=-1)
        return kl_losses, tuned_lens_losses, activation_storage
    else: 
        return kl_losses, activation_storage

# %%
lp = LinePlot([*[f"kl_loss_{k}" for k in range(n_layers)], 'step_size'])
    
for i in tqdm(range(10000)):
    batch = next(owt_iter)['tokens']
    lens_optimizer.zero_grad()

    kl_losses, _ = get_lens_loss(batch)
    kl_losses = kl_losses.mean(dim=-1)
    loss = torch.clamp(kl_losses.sum(),max=100)
    loss.backward()

    prev_weights = torch.cat(attn_bias, dim=0).detach()

    lens_optimizer.step()

    step_sz = (torch.cat(attn_bias, dim=0)-prev_weights).abs().sum()
    lp.add_entry({"step_size": step_sz.item(), **{f"kl_loss_{k}": kl_losses[k].item() for k in range(n_layers)}})

    # lens_scheduler.step()

    if math.isnan(lp.stat_book["step_size"][-1]):
        break

    if i % 500 == 10:
        lp.plot(subplots=3, save=f"{folder}/train.png", twinx=False, mv=20)
        with open(f"{folder}/modal_lens_weights.pkl", "wb") as f:
            pickle.dump(attn_bias, f)

# %%
            
# modal lens inference
# folder="modal_lens"
tuned_lens_folder = "pruning/tuned_lens"
with open(f"{folder}/modal_lens_weights.pkl", "rb") as f:
    attn_bias = pickle.load(f)

# %%
with open(f"{tuned_lens_folder}/tuned_lens_weights.pkl", "rb") as f:
    tuned_lens_weights = pickle.load(f)
with open(f"{tuned_lens_folder}/tuned_lens_bias.pkl", "rb") as f:
    tuned_lens_bias = pickle.load(f)

all_losses = ([],[])
for i in tqdm(range(100)):
    batch = next(owt_iter)['tokens']

    with torch.no_grad():
        kl_losses, tuned_lens_loss, activation_storage = get_lens_loss(batch, compare_tuned_lens=True)
        kl_losses = torch.nan_to_num(kl_losses, nan=0, posinf=0, neginf=0)
        all_losses[0].append(kl_losses)
        all_losses[1].append(tuned_lens_loss)

# %%
    
import pandas as pd 

# %%
f, axes = plt.subplots((n_layers-1)//3 + 1, 3, figsize=(15,15))
f, axes_log = plt.subplots((n_layers-1)//3 + 1, 3, figsize=(15,15))
modal_lens_loss_pts = pd.DataFrame(torch.cat(all_losses[0], dim=1).cpu().numpy().T)
modal_lens_loss_pts.columns = [f"{x}_modal" for x in modal_lens_loss_pts.columns]
tuned_lens_loss_pts = pd.DataFrame(torch.cat(all_losses[1], dim=1).cpu().numpy().T)
tuned_lens_loss_pts.columns = [f"{x}_tuned" for x in tuned_lens_loss_pts.columns]
df = modal_lens_loss_pts.merge(tuned_lens_loss_pts, left_index=True, right_index=True)

for i in range(n_layers):
    cur_ax = sns.histplot(x=(df[f"{i}_modal"]), y=(df[f"{i}_tuned"]), ax=axes[i // 3, i % 3])
    cur_ax.set_xlim(df[f"{i}_modal"].quantile(.01), df[f"{i}_modal"].quantile(.99))
    cur_ax.set_ylim(df[f"{i}_tuned"].quantile(.01), df[f"{i}_tuned"].quantile(.99))
    min_val = max(cur_ax.get_xlim()[0],cur_ax.get_ylim()[0])
    max_val = min(cur_ax.get_xlim()[1],cur_ax.get_ylim()[1])
    cur_ax.plot([min_val, max_val],[min_val, max_val], color="red", linestyle="-")

    cur_ax = sns.histplot(x=np.log(df[f"{i}_modal"]), y=np.log(df[f"{i}_tuned"]), ax=axes_log[i // 3, i % 3])
    min_val = max(cur_ax.get_xlim()[0],cur_ax.get_ylim()[0])
    max_val = min(cur_ax.get_xlim()[1],cur_ax.get_ylim()[1])
    cur_ax.plot([min_val, max_val],[min_val, max_val], color="red", linestyle="-")
    
# %%

print(modal_lens_loss_pts.mean())
print(tuned_lens_loss_pts.mean())
# # %%
# print(torch.stack(all_losses, dim=1).mean(dim=1))
# %%
