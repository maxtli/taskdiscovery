import torch
from fancy_einsum import einsum
from torch.nn import functional as F


def save_(save_to, result, hook):
  save_to.append(result)
  return result

def attn_filter(name):
  return name.endswith("attn.hook_result")

def pattern_filter(name):
  return name.endswith("pattern")

def pre_filter(name):
  return name.endswith("resid_pre")


def save_hook_last_token(save_to, act, hook):
    save_to.append(act[:, -1, :])


def ablation_hook_copy_all_tokens(bsz, n_heads, act, hook):
    # need to repeat this N times for the number of heads.
    act = torch.cat([act, *[act[:bsz] for _ in range(n_heads)]], dim=0)
    return act


def resid_points_filter(layer_no, name):
    return name == f"blocks.{layer_no}.hook_resid_pre"


def ablation_hook_copy_all_tokens(bsz, n_heads, act, hook):
    # need to repeat this N times for the number of heads.
    act = torch.cat([act, *[act[:bsz] for _ in range(n_heads)]], dim=0)
    return act


def save_attn_pattern(save_to, pattern, hook):
    save_to.append(pattern)


pattern_hook_names_filter = lambda name: name.endswith("pattern")


def tuned_lens_hook(activation_storage, tuned_lens_weights, tuned_lens_bias, act, hook):
    activation_storage.append(
        einsum(
            "result activation, batch activation -> batch result",
            tuned_lens_weights,
            act[:, -1],
        )
        + tuned_lens_bias
    )
    return act


def tuned_lens_hook_dir(
    activation_storage,
    tuned_lens_weights,
    tuned_lens_bias,
    dir_tensor,
    alpha,
    batch_size,
    act,
    hook,
):

    pure_residul = act[0:batch_size]
    device = pure_residul.device
    # shape (layer_batch, batch, tokens, activation)
    bias = torch.zeros_like(pure_residul).to(device)

    bias[:, -1, :] = alpha * dir_tensor

    bias.requires_grad_(True)

    # shape (layer_batch, batch, tokens, activation)
    added_dir = pure_residul + bias

    # Last batch will be pure act
    new_act = torch.concat([act, added_dir])

    pure_tuned_lens = (
        einsum(
            "result activation, batch activation -> batch result",
            tuned_lens_weights,
            pure_residul[:, -1],
        )
        + tuned_lens_bias
    )
    dir_tuned_lens = (
        einsum(
            "result activation, batch activation -> batch result",
            tuned_lens_weights,
            added_dir[:, -1],
        )
        + tuned_lens_bias
    )

    activation_storage.append((pure_tuned_lens, dir_tuned_lens))

    return new_act


def add_dir(
    dir_map,
    alpha_map,
    activation_storage,
    dirs_storage,
    batch_size,
    act,
    hook
):

    pure_residul = act[0:batch_size]
    dir_tensor = dir_map(pure_residul[:, -1])
    dirs_storage.append(dir_tensor)
    activation_storage.append(pure_residul[:, -1])
    bias = torch.zeros_like(pure_residul).to(pure_residul.device)
   
    alpha = alpha_map(pure_residul[:, -1])

    bias[:, -1, :] = alpha * dir_tensor
    added_dir = pure_residul - bias
    new_act = torch.cat([act, added_dir])
    return new_act
