import sys; sys.path.append("mats_sae_training")
import gc

import torch as t
from sae_training.utils import LMSparseAutoencoderSessionloader
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt


for layer in range(12):
    REPO_ID = "jbloom/GPT2-Small-SAEs"
    FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model, sparse_autoencoder, activation_store = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
        path = path
    )
    model.eval()
    sparse_autoencoder.eval()
    autoencoder = sparse_autoencoder.autoencoders[0]

    FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576_log_feature_sparsity.pt"
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    log_feature_sparsity = t.load(path, map_location=sparse_autoencoder.cfg.device)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(log_feature_sparsity.squeeze(0).cpu(), bins=100, color="red", alpha=0.7, density=True)
    ax.set_title(f"Log-Frequency of Features: Layer {layer}")
    ax.set_xlabel(r"log$_{10}$(frequency)")
    ax.set_ylabel("density")
    plt.tight_layout()
    plt.savefig(f"figures/layer{layer}_logfreq.png", dpi=400)
    plt.close()

    # get cosine sims of features

    p = log_feature_sparsity < -8
    enc_directions = autoencoder.W_enc[:, p]
    ed_normed = enc_directions / enc_directions.norm(dim=0, keepdim=True)
    # pairwise cosine sim -> sample randomly
    cos_sims = (ed_normed.T @ ed_normed).flatten()
    cos_sims_sample = cos_sims[t.randint(0, cos_sims.shape[0], (10000,))]
    rare = cos_sims_sample.cpu().detach().numpy()

    p = log_feature_sparsity >= -8
    enc_directions = autoencoder.W_enc[:, p]
    ed_normed = enc_directions / enc_directions.norm(dim=0, keepdim=True)
    # pairwise cosine sim -> sample randomly
    cos_sims = (ed_normed.T @ ed_normed).flatten()
    cos_sims_sample = cos_sims[t.randint(0, cos_sims.shape[0], (10000,))]
    not_rare = cos_sims_sample.cpu().detach().numpy()


    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(rare, bins=100, color="red", alpha=0.5, density=True, label="rare")
    ax.hist(not_rare, bins=100, color="blue", alpha=0.5, density=True, label="not rare")
    ax.legend()
    ax.set_title(f"Similarity of Random Sample of Encoder Directions: Layer {layer}")
    ax.set_ylabel("Density")
    ax.set_xlabel("Cosine Similarity")
    plt.tight_layout()
    plt.savefig(f"figures/layer{layer}_cossim.png", dpi=400)
    plt.close()

    del model, sparse_autoencoder, activation_store, autoencoder, enc_directions, ed_normed, cos_sims, cos_sims_sample
    gc.collect()
    t.cuda.empty_cache()