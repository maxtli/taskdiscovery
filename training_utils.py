from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.optim
from data import retrieve_owt_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer

def scatter(x,y,n_layers = 12):
  fig, axs = plt.subplots(4,3, figsize = (16,10))
  axs = axs.flatten()
  for i in range(n_layers):
    sns.scatterplot(ax = axs[i],x = x[i], y = y[i], alpha = 0.5)

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
        only_mv = False,

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
                if subplots is not None and only_mv != True:
                    ax = sns.lineplot(
                        x=t, y=yvals, label=s, ax=axes[i // subplots, i % subplots]
                    )
                    cur_ax = ax
            if mv:

                mv_series = [
                    np.mean(yvals[max(0, i - mv):])
                    for i in range(len(yvals))
                ]

                if only_mv:
                    sns.lineplot(x=t, y=mv_series, label=f"{s}_mv_{mv}",ax=axes[i // subplots, i % subplots])
                else:
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


def plot_losses(x, y, n_layers):
    f, axes = plt.subplots((n_layers - 1) // 3 + 1, 3, figsize=(15, 15))
    f, axes_log = plt.subplots((n_layers - 1) // 3 + 1, 3, figsize=(15, 15))
    modal_lens_loss_pts = pd.DataFrame(x)
    modal_lens_loss_pts.columns = [f"{x}_modal" for x in modal_lens_loss_pts.columns]
    tuned_lens_loss_pts = pd.DataFrame(y)
    tuned_lens_loss_pts.columns = [f"{x}_tuned" for x in tuned_lens_loss_pts.columns]
    df = modal_lens_loss_pts.merge(
        tuned_lens_loss_pts, left_index=True, right_index=True
    )

    for i in range(n_layers):
        cur_ax = sns.histplot(
            x=(df[f"{i}_modal"]), y=(df[f"{i}_tuned"]), ax=axes[i // 3, i % 3]
        )
        cur_ax.set_xlim(
            df[f"{i}_modal"].quantile(0.01), df[f"{i}_modal"].quantile(0.99)
        )
        cur_ax.set_ylim(
            df[f"{i}_tuned"].quantile(0.01), df[f"{i}_tuned"].quantile(0.99)
        )
        min_val = max(cur_ax.get_xlim()[0], cur_ax.get_ylim()[0])
        max_val = min(cur_ax.get_xlim()[1], cur_ax.get_ylim()[1])
        cur_ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="-")

        cur_ax = sns.histplot(
            x=np.log(df[f"{i}_modal"]),
            y=np.log(df[f"{i}_tuned"]),
            ax=axes_log[i // 3, i % 3],
        )
        min_val = max(cur_ax.get_xlim()[0], cur_ax.get_ylim()[0])
        max_val = min(cur_ax.get_xlim()[1], cur_ax.get_ylim()[1])
        cur_ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="-")


def fit_beta_model(x, y, fit_intercept=True):
    dirs = []
    r2s = []
    preds = []
    y_tests = []
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    linmodel = LinearRegression(fit_intercept=fit_intercept)

    for layer in range(12):
        # Assuming x_train[:, layer] and x_test[:, layer] are 2D
        # Fit Linear Regression model
        x_tr = x_train[:, layer]
        x_ts = x_test[:, layer]

        y_tr = y_train[:, layer]
        y_ts = y_test[:, layer]

        
        lin_reg = linmodel.fit(x_tr, y_tr)
        preds.append(lin_reg.predict(x_ts))
        y_tests.append(y_ts)

        lin_train_r2 = round(lin_reg.score(x_tr, y_tr), 3)
        lin_test_r2 = round(lin_reg.score(x_ts, y_ts), 3)
        dirs.append(torch.Tensor(lin_reg.coef_))
        r2s.append((lin_train_r2,lin_test_r2 ))

        
    return dirs,r2s,preds,y_tests


def show_token_preds(logits, output, tokenizer, sentence):
    d = {}
    print("Sentence:", sentence)
    for i, layer_logits in enumerate(logits):
        if "layer" not in d:
            d["layer"] = []
        d["layer"].append(i)
        top10_vals, top10_indices = torch.topk(layer_logits.squeeze(), 10)
        decoded_tokens = [
            tokenizer.decode([tok])
            for tok in top10_indices.squeeze().cpu().numpy().tolist()
        ]
        j = 1
        for tok, val in zip(decoded_tokens, top10_vals):
            if f"pred {j}" not in d:
                d[f"pred {j}"] = []
            # Append both token and logit as a tuple for hover data
            d[f"pred {j}"].append(f"{tok} ({val.item():.4f})")
            j += 1

    # Final output prediction
    final_top10_vals, final_top10_indices = torch.topk(output.squeeze(), 10)
    final_decoded_tokens = [tokenizer.decode([tok]) for tok in final_top10_indices]
    d["layer"].append("output")

    j = 1
    for tok, val in zip(final_decoded_tokens, final_top10_vals):
        d[f"pred {j}"].append(f"{tok} ({val.item():.4f})")
        j += 1

    # Create a DataFrame
    df = pd.DataFrame(d)
    fig = go.Figure()
    for col_index, column in enumerate(df.columns):
        for row_index, value in enumerate(df[column]):
            fig.add_trace(
                go.Scatter(
                    x=[col_index],
                    y=[row_index],
                    text=[value],
                    mode="markers+text",
                    marker=dict(size=1, opacity=0),
                    textposition="middle center",
                    hoverinfo="text",
                    showlegend=False,
                )
            )

    # Update layout to resemble a table
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            tickmode="array",
            tickvals=list(range(len(df.columns))),
            ticktext=df.columns,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            tickmode="array",
            tickvals=list(range(len(df))),
            ticktext=df.index,
        ),
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.show()
