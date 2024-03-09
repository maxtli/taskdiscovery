from functools import partial

import torch
import torch.nn.functional as F
from fancy_einsum import einsum
from hooks import add_dir, resid_points_filter, save_hook_last_token, tuned_lens_hook


def get_lens_loss(
    batch,
    model,
    n_layers,
    attn_bias,
    batch_size,
    tuned_lens_weights=None,
    tuned_lens_bias=None,
    shared_bias=False,
    compare_tuned_lens=False,
    return_cross_entropy = False,
):
    activation_storage = []

    target_probs = model.run_with_hooks(
        batch,
        fwd_hooks=[
            *[
                (
                    partial(resid_points_filter, layer_no),
                    partial(save_hook_last_token, activation_storage),
                )
                for layer_no in range(n_layers)
            ],
        ],
    )[:, -1].softmax(dim=-1)

    # activation_storage: batch x d_model (last token only)
    # resid: layer x batch x d_model (note: layer norms and unembeds need 3 tensor dimensions)
    resid = []
    for layer_no in range(n_layers):

        if layer_no > 0:
            resid = torch.cat(
                [resid_mid, activation_storage[layer_no].unsqueeze(0)], dim=0
            )

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

    logits = model.unembed(resid)
    loss = F.kl_div(logits.softmax(dim=-1).log(), target_probs, reduction="none").sum(dim=-1)
    

    if compare_tuned_lens:

        tuned_lens_resid = einsum(
            "layer result activation, layer batch activation -> layer batch result",
            torch.stack(tuned_lens_weights, dim=0),
            torch.stack(activation_storage, dim=0),
        ) + torch.stack(tuned_lens_bias, dim=0).unsqueeze(1)
        tuned_lens_resid = model.ln_final(tuned_lens_resid)
        tuned_lens_logits = model.unembed(tuned_lens_resid).softmax(dim=-1)
        tuned_lens_losses = F.kl_div(
            tuned_lens_logits.log(), target_probs, reduction="none"
        ).sum(dim=-1)
        if return_cross_entropy:
            ce = F.cross_entropy(
                logits.reshape(n_layers * batch_size, -1),
                target_probs.repeat(n_layers, 1),
                reduction="none",
            ).reshape(n_layers, batch_size)
            return loss, ce, activation_storage, target_probs

        return loss, tuned_lens_losses, activation_storage, target_probs
    else:
        if return_cross_entropy:

            ce = F.cross_entropy(
                logits.reshape(n_layers * batch_size, -1),
                target_probs.repeat(n_layers, 1),
                reduction="none",
            ).reshape(n_layers, batch_size)
            return loss,ce, activation_storage, target_probs
        return loss, activation_storage, target_probs


def get_tuned_lens_loss(model, batch, tuned_lens_weights, tuned_lens_bias, n_layers):
    activation_storage = []

    target_probs = model.run_with_hooks(
        batch,
        fwd_hooks=[
            (
                partial(resid_points_filter, layer_no),
                partial(
                    tuned_lens_hook,
                    activation_storage,
                    tuned_lens_weights[layer_no],
                    tuned_lens_bias[layer_no],
                ),
            )
            for layer_no in range(n_layers)
        ],
    )[:, -1].softmax(dim=-1)
    # batch_size x n_layers x d_model
    residual = model.ln_final(torch.stack(activation_storage, dim=1))

    # n_layers x batch_size x d_model
    residual = model.unembed(residual).softmax(dim=-1).permute((1, 0, 2))

    # n_layers x batch_size
    kl_losses = F.kl_div(residual.log(), target_probs, reduction="none").sum(dim=-1)

    return kl_losses, activation_storage


def run_modal_lens_with_dir(
    model,
    batch,
    attn_bias,
    alpha_maps,
    dir_maps,
    batch_size,
    n_layers,
    shared_bias=False
):

    activation_storage = []
    dirs_storage = []

    output = model.run_with_hooks(
        batch,
        fwd_hooks=[
            *[
                (
                    partial(resid_points_filter, layer_no),
                    partial(
                        add_dir,
                        dir_maps[layer_no],
                        alpha_maps[layer_no],
                        activation_storage,
                        dirs_storage,
                        batch_size,
                    ),
                )
                for layer_no in range(n_layers)
            ],
        ],
    )[:, -1].softmax(dim=-1)
    resid = []

    for layer_no in range(n_layers):

        if layer_no > 0:
            resid = torch.cat(
                [resid_mid, activation_storage[layer_no].unsqueeze(0)], dim=0
            )

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
    logits = model.unembed(resid)
    return output, logits, activation_storage, dirs_storage
