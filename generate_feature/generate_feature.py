# %%
# %%
from datasets import load_dataset, Dataset
from itertools import chain
from transformers import AutoTokenizer
import re


# Taken from the implementation of the lm_eval library
def wikitext_detokenizer(string: str) -> str:
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def get_wikitext_dataset(
    split: str,
    tokenizer: AutoTokenizer,
    block_size: int = 512,
) -> Dataset:
    assert split in ["train", "eval_train", "valid"]
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    raw_datasets = raw_datasets.map(
        lambda x: {"text": wikitext_detokenizer(x["text"])},
        batched=False,
        desc="Detokenizing Wikitext",
    )

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    if split in ["train", "eval_train"]:
        train_dataset = lm_datasets["train"]
        ds = train_dataset
    else:
        eval_dataset = lm_datasets["validation"]
        ds = eval_dataset

    # Add data_index to the dataset
    ds = ds.map(
        lambda example, idx: {"data_index": idx},
        with_indices=True,
        desc="Adding data index",
    )

    return ds

# %%
from sae_lens import SAE
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
import torch
from transformers import AutoTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

dataset = load_dataset(
    path="NeelNanda/pile-10k",
    split="train",
    streaming=False,
)


# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience.
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
    sae_id="blocks.8.hook_resid_pre",  # won't always be a hook point
    device=device,
)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)


# %%
with torch.no_grad():
    for example in [token_dataset[0]]:
        tokens = example["tokens"]
        tokens = tokens.to(device)
        tokens = tokens.unsqueeze(0)
        _, cache = model.run_with_cache(tokens)
        sae_out = sae.encode(cache[sae.cfg.hook_name])
        # sae out has shape (1, seq_len, num_features) is a matrix of elements containing mostly zeros
        # and a few non-zero values. The non-zero values are the activations of the sae.
