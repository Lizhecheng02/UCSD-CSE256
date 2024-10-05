import pandas as pd
import gc
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer
)
from tqdm import tqdm
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from sentiment_data import read_sentiment_examples


def train_bpe_tokenizer(bpe_vocab_size, lower_case=False):
    examples = read_sentiment_examples("./data/train.txt")
    sentences = [" ".join(ex.words) for ex in examples]
    labels = [ex.label for ex in examples]

    df = pd.DataFrame(columns=["text", "label"])
    df["text"] = sentences
    df["label"] = labels
    print(df.head())

    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFC()] + [normalizers.Lowercase()] if lower_case else []
    )
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(
        vocab_size=bpe_vocab_size,
        special_tokens=special_tokens
    )

    dataset = Dataset.from_pandas(df[["text"]])

    def train_corpus():
        for i in tqdm(range(0, len(dataset), 100)):
            yield dataset[i:i + 100]["text"]

    raw_tokenizer.train_from_iterator(train_corpus(), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    save_directory = "./bpe_trained_tokenizer"
    tokenizer.save_pretrained(save_directory)


# train_bpe_tokenizer(bpe_vocab_size=20000)
# tokenizer = PreTrainedTokenizerFast.from_pretrained("./bpe_trained_tokenizer")
# vocab = tokenizer.get_vocab()
# sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
# for token, id in tqdm(sorted_vocab, total=len(sorted_vocab)):
#     print(f"Token: {token}, ID: {id}")
#     break
# print(sorted_vocab[10])
