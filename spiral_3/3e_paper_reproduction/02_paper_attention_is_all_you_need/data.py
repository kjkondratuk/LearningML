"""Data loading and tokenization for machine translation."""

def load_wmt_data(data_dir: str, src_lang: str = "en", tgt_lang: str = "de"):
    raise NotImplementedError

def build_tokenizer(corpus: list[str], vocab_size: int = 32000):
    raise NotImplementedError
