import torch
import torch.nn as nn
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset
from bpe_trainer import train_bpe_tokenizer
from transformers import PreTrainedTokenizerFast


class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, embs_path, max_length=64):
        self.examples = read_sentiment_examples(infile)
        self.sentences = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        embs = read_word_embeddings(embs_path)

        sentences_list = []
        for sentence in self.sentences:
            sentence_list = []
            for word in sentence:
                if embs.word_indexer.index_of(word) == -1:
                    sentence_list.append(embs.word_indexer.index_of("UNK"))
                else:
                    sentence_list.append(embs.word_indexer.index_of(word))
            if len(sentence_list) > max_length:
                sentence_list = sentence_list[:max_length]
            while len(sentence_list) < max_length:
                sentence_list.append(embs.word_indexer.index_of("PAD"))
            sentences_list.append(sentence_list)

        self.sentences = torch.tensor(sentences_list, dtype=torch.float32)
        print(self.sentences.shape)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        print(self.labels.shape)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]
    

class SentimentDatasetDAN_BPE(Dataset):
    def __init__(self, infile, bpe_vocab_size, max_length=32):
        self.examples = read_sentiment_examples(infile)
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        train_bpe_tokenizer(bpe_vocab_size=bpe_vocab_size)
        tokenizer = PreTrainedTokenizerFast.from_pretrained("./bpe_trained_tokenizer")

        sentences_list = []
        for sentence in self.sentences:
            tokens = tokenizer.tokenize(sentence)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            while len(token_ids) < max_length:
                token_ids.append(tokenizer.convert_tokens_to_ids("[PAD]"))
            sentences_list.append(token_ids)

        self.sentences = torch.tensor(sentences_list, dtype=torch.float32)
        print(self.sentences.shape)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        print(self.labels.shape)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


class DAN(nn.Module):
    def __init__(
        self,
        use_random_embed=False,
        vocab_size=15000,
        embed_file="./data/glove.6B.50d-relativized.txt",
        freeze_embed=False,
        input_size=50,
        hidden_sizes=[256, 256, 256],
        output_size=64,
        num_classes=1,
        use_dropout=True,
        dropout_rate=0.2
    ):
        super().__init__()
        self.use_random_embed = use_random_embed
        self.vocab_size = vocab_size
        self.embed_file = embed_file
        self.freeze_embed = freeze_embed
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        if self.use_random_embed:
            self.random_embedding = nn.Embedding(self.vocab_size, self.input_size)
            if self.freeze_embed:
                self.random_embedding.weight.requires_grad = False
            else:
                self.random_embedding.weight.requires_grad = True
        else:
            if self.embed_file == "./data/glove.6B.50d-relativized.txt":
                self.input_size = 50
            elif self.embed_file == "./data/glove.6B.300d-relativized.txt":
                self.input_size = 300
            else:
                print("Embedding file not found!!!")
            self.pretrained_embedding = read_word_embeddings(embeddings_file=self.embed_file).get_initialized_embedding_layer(self.freeze_embed)

        layers = []
        self.prev_size = self.input_size
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(self.prev_size, hidden_size))
            layers.append(nn.ReLU())
            if self.use_dropout:
                layers.append(nn.Dropout(self.dropout_rate))
            self.prev_size = hidden_size
        self.hidden_layers = nn.ModuleList(layers)

        self.output_layer = nn.Linear(self.prev_size, self.output_size)
        self.final_layer = nn.Linear(self.output_size, self.num_classes)

    def forward(self, x):
        if self.use_random_embed:
            embeddings = self.random_embedding(x.long())
            x = embeddings.mean(dim=1)
        else:
            embeddings = self.pretrained_embedding(x.long())
            x = embeddings.mean(dim=1)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.final_layer(x)
        return x


if __name__ == "__main__":
    embs = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    print(embs.word_indexer.index_of("good"))
    print(embs.word_indexer.index_of("UNK"))

    dataset = SentimentDatasetDAN(infile="./data/dev.txt")

    model = DAN()
    print(model)
    print(model(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])))
