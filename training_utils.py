from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.optim
from data import retrieve_owt_data
from transformer_lens import HookedTransformer
import pandas as pd


def load_model_data(
    model_name,
    batch_size=8,
    ctx_length=25,
    repeats=True,
    ds_name=False,
    device="cuda:0",
):
    # device="cpu"
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = model.tokenizer
    if ds_name:
        owt_loader = retrieve_owt_data(
            batch_size, ctx_length, tokenizer, ds_name=ds_name
        )
    else:
        owt_loader = retrieve_owt_data(batch_size, ctx_length, tokenizer)
    if repeats:
        owt_iter = cycle(owt_loader)
    else:
        owt_iter = owt_loader
    return device, model, tokenizer, owt_iter


class LinePlot:
    def __init__(self, stat_list):
        self.stat_list = stat_list
        self.stat_book = {x: [] for x in stat_list}
        self.t = 0

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

    def plot(
        self,
        series=None,
        subplots=None,
        step=1,
        start=0,
        end=0,
        agg="mean",
        twinx=True,
        mv=False,
        save=None,
    ):
        if series is None:
            series = self.stat_list
        if end <= start:
            end = self.t
        t = [i for i in range(start, end, step)]
        ax = None
        (h, l) = ([], [])
        colors = ["green", "blue", "red", "orange"]
        if subplots is not None:
            rows = (len(series) - 1) // subplots + 1
            f, axes = plt.subplots(rows, subplots, figsize=(rows * 5, subplots * 5))

        for i, s in enumerate(series):
            if agg == "mean":
                yvals = [
                    np.mean(self.stat_book[s][i : i + step])
                    for i in range(start, end, step)
                ]
            else:
                yvals = [self.stat_book[s][i] for i in range(start, end, step)]
            if twinx is True:
                params = {"x": t, "y": yvals, "label": s}
                if len(self.stat_list) <= 4:
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
                    ax = sns.lineplot(
                        x=t, y=yvals, label=s, ax=axes[i // subplots, i % subplots]
                    )
                    cur_ax = ax
            if mv:
                mv_series = [
                    np.mean(yvals[i : min(len(yvals), i + mv)])
                    for i in range(len(yvals))
                ]
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

def plot_losses(x,y, n_layers):
    f, axes = plt.subplots((n_layers-1)//3 + 1, 3, figsize=(15,15))
    f, axes_log = plt.subplots((n_layers-1)//3 + 1, 3, figsize=(15,15))
    modal_lens_loss_pts = pd.DataFrame(x)
    modal_lens_loss_pts.columns = [f"{x}_modal" for x in modal_lens_loss_pts.columns]
    tuned_lens_loss_pts = pd.DataFrame(y)
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