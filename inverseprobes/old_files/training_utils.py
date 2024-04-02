# %%

import torch
from transformer_lens import HookedTransformer
from data import retrieve_owt_data
from itertools import cycle
import torch.optim
from fancy_einsum import einsum
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
def load_model_data(model_name, batch_size=8, ctx_length=25, repeats=True, ds_name=False, device="cuda:0"):
    # device="cpu"
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = model.tokenizer
    try:
        if ds_name:
            owt_loader = retrieve_owt_data(batch_size, ctx_length, tokenizer, ds_name=ds_name)
        else:
            owt_loader = retrieve_owt_data(batch_size, ctx_length, tokenizer)
        if repeats:
            owt_iter = cycle(owt_loader)
        else:
            owt_iter = owt_loader
    except:
        owt_iter = None
    return device, model, tokenizer, owt_iter

# %%

def save_hook_last_token(save_to, act, hook):
    save_to.append(act[:,-1,:])

def ablation_hook_last_token(batch_feature_idx, repl, act, hook):
    # print(act.shape, hook.name)
    # act[:,-1,:] = repl

    # act: batch_size x seq_len x activation_dim
    # repl: batch_size x features_per_batch x activation_dim
    # print(batch_feature_idx[:,0].dtype)
    # act = act.unsqueeze(1).repeat(1,features_per_batch,1,1)[batch_feature_idx[:,0],batch_feature_idx[:,1]]
    act = act[batch_feature_idx]
    # sns.histplot(torch.abs(act[:,-1]-repl).flatten().detach().cpu().numpy())
    # plt.show()
    act[:,-1] = repl
    # returns: (batch_size * features_per_batch) x seq_len x activation_dim
    # act = torch.cat([act,torch.zeros(1,act.shape[1],act.shape[2]).to(device)], dim=0)
    return act
    # return act.repeat(features_per_batch,1,1)
    # pass

def ablation_all_hook_last_token(repl, act, hook):
    # print(act.shape, hook.name)
    # act[:,-1,:] = repl

    # act: batch_size x seq_len x activation_dim
    # repl: batch_size x features_per_batch x activation_dim
    # print(batch_feature_idx[:,0].dtype)
    # act = act.unsqueeze(1).repeat(1,features_per_batch,1,1)[batch_feature_idx[:,0],batch_feature_idx[:,1]]
    # sns.histplot(torch.abs(act[:,-1]-repl).flatten().detach().cpu().numpy())
    # plt.show()
    act[:,-1] = repl
    # returns: (batch_size * features_per_batch) x seq_len x activation_dim
    # act = torch.cat([act,torch.zeros(1,act.shape[1],act.shape[2]).to(device)], dim=0)
    return act
    # return act.repeat(features_per_batch,1,1)
    # pass

def ablation_hook_copy_all_tokens(bsz, n_heads, act, hook):
    # need to repeat this N times for the number of heads.
    act = torch.cat([act,*[act[:bsz] for _ in range(n_heads)]], dim=0)
    return act

def ablation_hook_attention_all_tokens(constants, bsz, activation_storage, attentions, hook):
    n_heads = constants.shape[0]
    start = bsz * n_heads
    for i in range(constants.shape[0]):
        # if attentions.shape[0] > 400:
        # print(start)
        attentions[-start:-start+bsz,:,i] = constants[i].clone()
        start -= bsz
    
    # print(attentions.shape)
    # if attentions.shape[0] > 400:
    #     sns.histplot(attentions[:bsz][attentions[:bsz].abs() > 20].detach().flatten().cpu())
    #     print((attentions[:bsz].abs() > 500).nonzero())
    #     print(attentions[:bsz][(attentions[:bsz].abs() > 500)])
        
    # ignore first token because it is crazy
    with torch.no_grad():
        activation_storage.append(attentions[:bsz,1:].mean(dim=[0,1]))
    return attentions

# attentions: (batch_size + batch_size * n_samples) x seq_len x n_heads x d_model
# constants: n_heads x d_model
# prune mask: (batch_size * n_samples) x n_heads, 0 = prune, 1 = keep
def pruning_hook_attention_all_tokens(constants, prune_mask, bsz, attentions, hook):
    # N by 2. First column = batch item, second column = head idx
    prune_mask = prune_mask.unsqueeze(1).unsqueeze(-1)
    attentions[bsz:] = (1-prune_mask) * constants + prune_mask * attentions[bsz:].clone()

    # prune_idx = prune_mask.clone()
    # attentions[bsz + prune_idx[:,0],:,prune_idx[:,1]] = prune_idx * constants[prune_idx[:,1]]
    return attentions

def tuned_lens_hook(activation_storage, tuned_lens_weights, tuned_lens_bias, act, hook):
    activation_storage.append(einsum("result activation, batch activation -> batch result", tuned_lens_weights, act[:,-1]) + tuned_lens_bias)
    return act

class LinePlot:
    def __init__(self, stat_list, pref_start=100):
        self.stat_list = stat_list
        self.stat_book = {x: [] for x in stat_list}
        self.t = 0
        self.last_tick = 0
        self.early_term_count = 0
        self.pref_start = pref_start
    
    def add_entry(self, entry):
        for k in self.stat_book:
            if k in entry:
                self.stat_book[k].append(entry[k])
            # default behavior is flat line
            elif self.t == 0:
                self.stat_book[k].append(0)
            else:
                self.stat_book[k].append(self.stat_book[k][-1])
        self.t += 1
    
    def stat_sig_growth(self, series, avg_intv=10, comp_intv=200, start_t=0):
        if self.t - start_t <= comp_intv + avg_intv + 1:
            return False
        historical_avg = [np.mean(self.stat_book[series][-i-avg_intv-1:-i-1]) for i in range(comp_intv // 2, comp_intv, (avg_intv // 3))]
        rolling_avg = np.mean(self.stat_book[series][-avg_intv:])

        # decline, growth
        return 1 - rolling_avg / np.quantile(historical_avg, .1), rolling_avg / np.quantile(historical_avg, .9) - 1
        
    def plot(self, series=None, subplots=None, step=1, start=None, end=0, agg='mean', twinx=True, mv=False, save=None):
        if start is None:
            start = self.pref_start
        if series is None:
            series = self.stat_list
        if end <= start:
            end = self.t
            if end <= start:
                start = 0
        t = [i for i in range(start, end, step)]
        ax = None
        (h,l) = ([],[])
        colors = ["green", "blue", "red", "orange"]
        if subplots is not None:
            rows = (len(series)-1) // subplots + 1
            f, axes = plt.subplots(rows, subplots, figsize=(rows * 5, subplots * 5))
            
        for i,s in enumerate(series):
            if agg == 'mean':
                yvals = [np.mean(self.stat_book[s][i:i+step]) for i in range(start, end, step)]
            else:
                yvals = [self.stat_book[s][i] for i in range(start, end, step)]
            if twinx is True:
                params = {"x": t, "y": yvals, "label": s}
                if len(series) <= 4:
                    params["color"] = colors[i]
                if ax is None:
                    ax = sns.lineplot(**params)
                    h, l = ax.get_legend_handles_labels()
                    ax.get_legend().remove()
                    cur_ax = ax
                else:
                    ax2 = sns.lineplot(**params, ax=ax.twinx())
                    ax2.get_legend().remove()
                    h2, l2 = ax2.get_legend_handles_labels()
                    h += h2
                    l += l2
                    cur_ax = ax
            else:
                if subplots is not None:
                    ax = sns.lineplot(x=t, y=yvals, label=s, ax=axes[i // subplots, i % subplots])
                    cur_ax = ax
            if mv:
                mv_series = [np.mean(yvals[i:min(len(yvals),i+mv)]) for i in range(len(yvals))]
                sns.lineplot(x=t, y=mv_series, label=f"{s}_mv_{mv}", ax=cur_ax)
        if h is None:
            plt.legend()
        else:
            plt.legend(h, l)
        plt.tight_layout()

        if save:
            plt.savefig(save)
        plt.show()
        plt.close()

    def export():
        pass

