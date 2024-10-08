## **CSE256 HomeWork1**

#### Author: [Zhecheng Li](https://github.com/Lizhecheng02) (PID: A69033467)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#### How to run the code ?

**Answer:** There are details instructions in the ``README.md`` file, the core operation is to slightly modify ``config.yaml`` file and then run ``python main.py``.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#### Part 1

#### 1.1 The design of DAN model

- **I set the ``hidden_sizes`` to be a list, thus we can easily define the number of layers for the MLP architectures in the whole DAN model.**

```bash
self.hidden_layers = nn.ModuleList(layers)
```

- **I changed the final output dimension of the DAN model into **1**, and use the ``sigmoid`` function to get the final label.**

```bash
self.final_layer = nn.Linear(self.output_size, self.num_classes) # Here self.num_classes = 1
correct += ((torch.sigmoid(pred) > 0.5) == y).type(torch.float).sum().item()
```

- **I rewrote the ``SentimentDatasetDAN`` class to load the dataset for training based on pre-trained GloVe embeddings. Here I use the ``UNK`` token for unknown tokens, and use the ``PAD`` token to padding all sentences into the length of ``32`` or ``64`` (could be set in the ``config.yaml`` file), so that we can do the batch training with ``batch_size`` greater than 1.**

```bash
for word in sentence:
	if embs.word_indexer.index_of(word) == -1:
		sentence_list.append(embs.word_indexer.index_of("UNK"))
    else:
        sentence_list.append(embs.word_indexer.index_of(word))
if len(sentence_list) > max_length:
    sentence_list = sentence_list[:max_length]
while len(sentence_list) < max_length:
    sentence_list.append(embs.word_indexer.index_of("PAD"))
```

- **I wrote the code in the DAN model to make users able to choose whether use random initialized embedding or pretrained GloVe embedding, and I also set the parameter to let people decide whether to train the embedding layer.**

```bash
if self.use_random_embed:
    self.random_embedding = nn.Embedding(self.vocab_size, self.input_size)
    self.random_embedding.weight.requires_grad = self.freeze_embed
else:
	self.pretrained_embedding = read_word_embeddings(embeddings_file=self.embed_file).get_initialized_embedding_layer(self.freeze_embed)
```

- **I added the ``dropout ratio`` for the DAN model in order to prevent model from overfitting the training data.**

```bash
if self.use_dropout:
    layers.append(nn.Dropout(self.dropout_rate))
```

#### 1.2 The experimental results with pre-trained GloVe embeddings

In order to find out the relationship between different hyper-parameters and the final accuracy, I've done a lot experiments based on different settings of parameters, the results are listed in the table below. (Note that I trained under different parameters for at most 20 epochs, since the performance of the DAN model usually already converged after training for 20 epochs, learning rate is 1.0e-4 for every experiment).

| Exp Id | max_length |          embed_file           | freeze_embed |  hidden_sizes  | dropout_rate | Best Train Accuracy (epoch) | Best Dev Accuracy (epoch) |
| :----: | :--------: | :---------------------------: | :----------: | :------------: | :----------: | :-------------------------: | :-----------------------: |
|   1    |     32     | glove.6B.50d-relativized.txt  |    False     | [128, 128, 64] |     0.4      |         0.883 (20)          |        0.791 (18)         |
|   2    |     32     | glove.6B.300d-relativized.txt |    False     | [128, 128, 64] |     0.4      |         0.983 (20)          |        0.821 (13)         |
|   3    |     32     | glove.6B.50d-relativized.txt  |     True     | [128, 128, 64] |     0.4      |         0.731 (19)          |        0.741 (18)         |
|   4    |     32     | glove.6B.300d-relativized.txt |     True     | [128, 128, 64] |     0.4      |         0.830 (20)          |        0.790 (18)         |
|   5    |     32     | glove.6B.300d-relativized.txt |    False     |   [128, 64]    |     0.4      |         0.984 (20)          |        0.820 (10)         |
|   6    |     32     | glove.6B.300d-relativized.txt |    False     |   [128, 64]    |     0.0      |         0.989 (20)          |        0.817 (13)         |
|   7    |     64     | glove.6B.300d-relativized.txt |    False     |   [128, 64]    |     0.4      |         0.978 (20)          |        0.813 (10)         |
|   8    |     16     | glove.6B.300d-relativized.txt |    False     |   [128, 64]    |     0.4      |         0.974 (20)          |         0.790 (8)         |

Conclusions from the above experiments:

(1) According to the exp id 5, 7, 8, the only difference between the three experiments is the max_length. From the accuray of the model, we find that setting the max_length to 32 gave us the best result. The reason why ``max_length = 16 ``gave the worst performance could be due to the lose of the sentence information, since we truncated at the 16th token, and many sentences are longer than 16 tokens, which means we drop many semantic information; The reason why max_length = 64 gave us slightly worse accuracy than 32 may caused by the semantic meanings of many "PAD" tokens, since we use the average embedding in the DAN model, to many "PAD" embedding may disrupt the original meaning of the whole sentence.

(2) When we compare the exp id (1, 2) and (3, 4), the only different is the embed_file. We can easily get the conclusion that embedding with higher dimension gave the better results for the final prediction, because when we use the 300d embedding, the accuracy scores are much better than those 50d ones. The reason behind this is that more dimension means the whole embedding can store more semantic information for this single token, thus leads to more accurate representation for this single token.

(3) When we compare the exp id 2 and 5, the only difference is the number of layers for the MLP architecture in the middle of the DAN model, we can find that there is almost no difference on the final accuracy. However, this doesn't mean that the number of layers do not affect the capabilities of the model, that's because due to the complexity of the model and the length of train dataset, the number of parameters for the whole model does not affect a lot. Unfortunately, due to the computational resource limitation, I do not have the chance to set the hidden layers to the 10 layers to see what will happen.

(4) According to exp id 5 and 6, whether using dropout layer in the DAN model does not have significant impact on the final results. Although we can see that the training accuracy without dropout layers is a little bit higher than the one with dropout layer, it cannot show that with dropout layers the results will definitely be better. However, we can tell that dropout layers do help a little bit on preventing overfitting, the small impact may due to the number of total layers in the whole model. If there are more layers with more dropout layers, the impact will be larger.

(5) According to all the experiments, we can find that there are obvious overfitting on the training dataset, because the training accuracy almost reach 1.00 even after 20 epochs training, but the dev accuracy still remains at around 0.810 to 0.820. 

##### But all in all, it's still very easy to reach the 0.77 accuracy by using pre-trained GloVe embeddings.

#### 1.3 How about randomly initialized embeddings ?

In order to show the influence of the pre-trained embedding to the whole prediction model, here we did some experiments when using randomly initialized embeddings. (Here we set the max_length to 32, learning rate to 3e-4 and epochs to 50 since we need to train the embedding from scratch)

| Exp Id | freeze_embed | embed_size |  hidden_sizes  | dropout_rate | Best Train Accuracy (epoch) | Best Dev Accuracy (epoch) |
| :----: | :----------: | :--------: | :------------: | :----------: | :-------------------------: | :-----------------------: |
|   1    |     True     |    128     | [128, 128, 64] |     0.4      |         0.733 (50)          |        0.617 (20)         |
|   2    |     True     |    128     | [128, 128, 64] |     0.0      |         0.861 (50)          |         0.619 (6)         |
|   3    |    False     |    128     | [128, 128, 64] |     0.4      |         0.996 (50)          |        0.769 (34)         |
|   4    |     True     |    256     | [128, 128, 64] |     0.4      |         0.769 (48)          |        0.650 (12)         |
|   5    |     True     |    256     | [128, 128, 64] |     0.0      |         0.878 (50)          |        0.654 (32)         |
|   6    |    False     |    256     | [128, 128, 64] |     0.4      |         0.992 (46)          |        0.790 (34)         |

Conclusions from the above experiments:

(1) When compare the results of exp id 1 and 2, we can find that set the dropout rate indeed helps the model to prevent from overfitting on the training dataset, but it does not generate any benefit on the dev accuracy.

(2) When we compare the exp id 3 with the above table, we can find that the result of training a 128 dimension embedding from scratch is worse than training based on the pre-trained 50 dimension GloVe embedding (0.769 vs. 0.791). This is because the GloVe embedding is pre-trained on large corpus of data, which can represent the word embedding more accurate, only training the embedding based on training dataset is not enough to totally understand the meaning of the word.

(3) When we compare the exp id 3 and 6, we can find that even if we gave the word with random initialized embeddings, with the higher dimension of the embedding, after training for even the same epoch, the best accuracy is much higher with the high dimensional one. This is because higher dimension means more information to represent for this word.

##### It's even very easy to reach the 0.77 accuracy by training a random embedding only base on training data.

#### Part 2
#### 2.1 How to train a bpe tokenizer ?

The exact code is shown in the bpe_trainer.py file.

#### 2.2 The training steps behind bpe trainer ?

- Gather a large corpus of text data relevant.
- Start by creating a vocabulary with all unique characters in the text.
- Convert the text into sequences of character tokens.
- Identify and count all adjacent character pairs in the text, find the most frequent pair in the entire corpus.
- Merge the most common pair of characters into a single token. For example, if "t" and "h" are most frequent, replace "th" with a new token.
- Add the new merged token to your vocabulary.
- Continue merging pairs, updating the vocabulary with each merge, until reaching a specified vocabulary size or number of merges.

#### 2.3 Experimental results for bpe tokenizer.

For the following experiments, we set the learning rate to 2e-4, max_length to 32 and epochs to 50.

| Exp Id | bpe_vocab_size | freeze_embed | embed_size |  hidden_sizes  | dropout_rate | Best Train Accuracy (epoch) | Best Dev Accuracy (epoch) |
| :----: | :------------: | :----------: | :--------: | :------------: | :----------: | :-------------------------: | :-----------------------: |
|   1    |     20000      |    False     |     64     | [128, 128, 64] |     0.4      |                             |                           |
|   2    |     20000      |    False     |    128     | [128, 128, 64] |     0.4      |                             |                           |
|   3    |     20000      |     True     |     64     | [128, 128, 64] |     0.4      |                             |                           |
|   4    |     20000      |     True     |    128     | [128, 128, 64] |     0.4      |                             |                           |
|   5    |                |              |            |                |              |                             |                           |
|   6    |                |              |            |                |              |                             |                           |
|   7    |                |              |            |                |              |                             |                           |

