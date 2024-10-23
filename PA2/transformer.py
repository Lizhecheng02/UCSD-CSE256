import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = (torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1))
        even_positional_embedding = torch.sin(position / denominator)
        odd_positional_embedding = torch.cos(position / denominator)
        stacked = torch.stack([even_positional_embedding, odd_positional_embedding], dim=2)
        positional_embedding = torch.flatten(stacked, start_dim=1, end_dim=2)
        return positional_embedding


def alibi_encoding(seq_len, num_heads):
    distance_matrix = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    alibi = distance_matrix.unsqueeze(0).repeat(num_heads, 1, 1)
    return alibi


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query, mask, alibi_bias=None):
        N = query.shape[0]

        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        values = self.values(value)
        keys = self.keys(key)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        qk = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])

        if mask is not None:
            qk = qk.masked_fill(mask == 0, float("-inf"))

        attention = qk / (self.embed_size ** (1 / 2))
        if alibi_bias is not None:
            attention = attention + alibi_bias
        attention = torch.softmax(attention, dim=3)

        out = torch.einsum("nhqk, nkhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)

        return out, attention


class WindowedSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, window_size):
        super(WindowedSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.window_size = window_size
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query, mask, alibi_bias=None):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        window_size = min(self.window_size, key_len)

        values = self.values(value)
        keys = self.keys(key)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        attention = torch.zeros(N, self.heads, query_len, query_len).to(queries.device)

        for i in range(query_len):
            start_index = max(0, i - window_size // 2)
            end_index = min(key_len, i + window_size // 2 + 1)
            qk_window = torch.einsum("nqhd, nkhd -> nhqk", [queries[:, i:i + 1], keys[:, start_index:end_index]])
            if mask is not None:
                mask_window = mask[:, :, :, start_index:end_index]
                qk_window = qk_window.masked_fill(mask_window == 0, float("-inf"))

            qk_window = qk_window.squeeze(2)
            attention[:, :, i, start_index:end_index] = qk_window

        attention = attention / (self.embed_size ** (1 / 2))
        # print(attention.shape)
        if alibi_bias is not None:
            attention += alibi_bias

        attention = torch.softmax(attention, dim=3)
        out = torch.einsum("nhqk, nkhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)

        return out, attention
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, alibi_bias=None):
        output, attention = self.attention(value, key, query, mask, alibi_bias)
        x = self.dropout(self.norm1(output + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out, attention


class TransformerBlockWindowAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, window_size):
        super(TransformerBlockWindowAttention, self).__init__()
        self.attention = WindowedSelfAttention(embed_size, heads, window_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, alibi_bias=None):
        output, attention = self.attention(value, key, query, mask, alibi_bias)
        x = self.dropout(self.norm1(output + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out, attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def make_mask(self, input_ids):
        N, seq_length = input_ids.shape
        mask = torch.tril(torch.ones((seq_length, seq_length))).expand(N, 1, seq_length, seq_length)
        # print(mask)
        return mask.to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        mask = self.make_mask(x)
        pos_embed = self.position_embedding().unsqueeze(0).expand(N, -1, -1)[:, :seq_length, :]
        out = self.dropout(self.word_embedding(x) + pos_embed)
        for layer in self.layers:
            out, attention = layer(out, out, out, mask)
        out = self.fc_out(out)
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, pad_idx):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def make_mask(self, input_ids):
        src_mask = (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)
        print(src_mask)
        return src_mask.to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        mask = self.make_mask(x)
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        pos_embed = self.position_embedding().unsqueeze(0).expand(N, -1, -1)[:, :seq_length, :]
        # out = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        out = self.dropout((self.word_embedding(x) + pos_embed))
        for layer in self.layers:
             out, attention = layer(out, out, out, mask)
        return out


class ClassificationEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, pad_idx, num_classes):
        super(ClassificationEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(embed_size, num_classes)
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # print(src_mask)
        return src_mask.to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        mask = self.make_src_mask(x)
        pos_embed = self.position_embedding().unsqueeze(0).expand(N, -1, -1)[:, :seq_length, :]
        out = self.dropout((self.word_embedding(x) + pos_embed))
        for layer in self.layers:
             out, attention = layer(out, out, out, mask)
        out = out.mean(dim=1)
        out = self.classification_head(out)
        return out


class ClassificationEncoderWindowAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, pad_idx, num_classes, window_size=16):
        super(ClassificationEncoderWindowAttention, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList([TransformerBlockWindowAttention(embed_size, heads, dropout, forward_expansion, window_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(embed_size, num_classes)
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # print(src_mask)
        return src_mask.to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        mask = self.make_src_mask(x)
        pos_embed = self.position_embedding().unsqueeze(0).expand(N, -1, -1)[:, :seq_length, :]
        out = self.dropout((self.word_embedding(x) + pos_embed))
        for layer in self.layers:
             out, attention = layer(out, out, out, mask)
        out = out.mean(dim=1)
        out = self.classification_head(out)
        return out


class ClassificationEncoderAlibi(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, pad_idx, num_classes):
        super(ClassificationEncoderAlibi, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(embed_size, num_classes)
        self.pad_idx = pad_idx
        self.alibi_encoding = alibi_encoding(seq_len=max_length, num_heads=heads)

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # print(src_mask)
        return src_mask.to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        mask = self.make_src_mask(x)
        out = self.dropout(self.word_embedding(x))
        for layer in self.layers:
             out, attention = layer(out, out, out, mask, alibi_bias=self.alibi_encoding)
        out = out.mean(dim=1)
        out = self.classification_head(out)
        return out


if __name__ == "__main__":
    # encoder_model = Encoder(
    #     vocab_size=1000,
    #     embed_size=64,
    #     num_layers=2,
    #     heads=2,
    #     device="cpu",
    #     forward_expansion=4,
    #     dropout=0.1,
    #     max_length=16,
    #     pad_idx=0
    # )
    # output = encoder_model(torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0], [6, 7, 8, 9, 10, 0, 0, 0]]))
    # print(output.shape)

    # encoder_model = ClassificationEncoder(
    #     vocab_size=1000,
    #     embed_size=64,
    #     num_layers=2,
    #     heads=2,
    #     device="cpu",
    #     forward_expansion=4,
    #     dropout=0.1,
    #     max_length=16,
    #     pad_idx=0,
    #     num_classes=3
    # )
    # output = encoder_model(torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0], [6, 7, 8, 9, 10, 0, 0, 0]]))
    # print(output)
    # print(output.shape)
    # print(torch.max(output.data, 1))

    decoder_model = Decoder(
        vocab_size=1000,
        embed_size=64,
        num_layers=2,
        heads=2,
        device="cpu",
        forward_expansion=4,
        dropout=0.1,
        max_length=16
    )
    output = decoder_model(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
    print(output)
    print(output.shape)