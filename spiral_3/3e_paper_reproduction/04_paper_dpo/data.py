"""Preference data preparation for DPO."""

def load_hh_rlhf(split: str = "train"):
    raise NotImplementedError

def prepare_preference_pairs(dataset, tokenizer, max_length: int = 512):
    raise NotImplementedError
