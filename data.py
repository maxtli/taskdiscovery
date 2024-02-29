import datasets
from torch.utils.data import DataLoader
from transformer_lens.utils import tokenize_and_concatenate


def retrieve_owt_data(
    batch_size,
    ctx_length,
    tokenizer,
    split="train",
    from_saved=False,
    ds_name="Elriggs/openwebtext-100k",
):
    dataset = datasets.load_dataset(ds_name, split="train")
    tokens_dataset = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=ctx_length,
        column_name="text",
        add_bos_token=True,
        num_proc=4,
    )
    tokens_dataset
    data_loader = DataLoader(
        tokens_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return data_loader
