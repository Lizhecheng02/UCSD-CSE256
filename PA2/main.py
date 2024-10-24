import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import ClassificationEncoder, Decoder, ClassificationEncoderAlibi, ClassificationEncoderWindowAttention
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utilities import Utilities


seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 5e-4  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 50  # How often to evaluate train and test perplexity during training
max_iters = 500  # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we"ll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 100  # Number of iterations to evaluate perplexity on the test set


# classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
# size of 64, hidden size of 50 and output size of 3.
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don"t need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:
            continue
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, attention_matrices = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        # your model should be computing the cross entropy loss
        loss = decoderLMmodel(X, Y)
        losses.append(loss.item())
        # total_loss += loss.item()
        if len(losses) >= eval_iters:
            break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    # Calculate perplexity as exp(mean loss)
    perplexity = torch.exp(mean_loss).item()

    decoderLMmodel.train()
    return perplexity


def compute_perplexity_with_logits_output(decoderLMmodel, data_loader, criterion, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        output = decoderLMmodel(X)
        output = output.view(-1, output.size(-1))
        Y = Y.view(-1)
        loss = criterion(output, Y)
        losses.append(loss.item())
        # total_loss += loss.item()
        if len(losses) >= eval_iters:
            break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    # Calculate perplexity as exp(mean loss)
    perplexity = torch.exp(mean_loss).item()

    decoderLMmodel.train()
    return perplexity


def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts("speechesdataset")
    # create a tokenizer from the data
    tokenizer = SimpleTokenizer(" ".join(texts))
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, "r", encoding="utf-8") as f:
        lmtrainText = f.read()

    inputfile = "speechesdataset/test_LM_hbush.txt"
    with open(inputfile, "r", encoding="utf-8") as f:
        lm_test_hbush_text = f.read()

    inputfile = "speechesdataset/test_LM_obama.txt"
    with open(inputfile, "r", encoding="utf-8") as f:
        lm_test_obama_text = f.read()

    inputfile = "speechesdataset/test_LM_wbush.txt"
    with open(inputfile, "r", encoding="utf-8") as f:
        lm_test_wbush_text = f.read()

    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    test_LM_dataset_hbush = LanguageModelingDataset(tokenizer, lm_test_hbush_text, block_size)
    test_LM_dataset_obama = LanguageModelingDataset(tokenizer, lm_test_obama_text, block_size)
    test_LM_dataset_wbush = LanguageModelingDataset(tokenizer, lm_test_wbush_text, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    test_LM_loader_hbush = DataLoader(test_LM_dataset_hbush, batch_size=batch_size, shuffle=True)
    test_LM_loader_obama = DataLoader(test_LM_dataset_obama, batch_size=batch_size, shuffle=True)
    test_LM_loader_wbush = DataLoader(test_LM_dataset_wbush, batch_size=batch_size, shuffle=True)

    # for the classification  task, you will train for a fixed number of epochs like this:
    classification_encoder = ClassificationEncoder(
        vocab_size=tokenizer.vocab_size,
        embed_size=n_embd,
        num_layers=n_layer,
        heads=n_head,
        device=device,
        forward_expansion=2,
        dropout=0.1,
        max_length=block_size,
        pad_idx=0,
        num_classes=n_output
    )
    total_params = sum(p.numel() for p in classification_encoder.parameters())
    print("The total parameters for encoder classification model is:", total_params)
    optimizer = optim.Adam(classification_encoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # utility = Utilities(tokenizer=tokenizer, model=classification_encoder)
    # utility.sanity_check(sentence="In fact, I will be right there with you.", block_size=12)
    for epoch in range(epochs_CLS):
        print("Epoch:", epoch + 1)
        for xb, yb in tqdm(train_CLS_loader, total=len(train_CLS_loader)):
            xb, yb = xb.to(device), yb.to(device)
            output, attention_matrices = classification_encoder(xb)
            loss = criterion(output, yb)
            optimizer.zero_grad()  
            loss.backward()       
            optimizer.step() 
            # CLS training code here
        print(f"Epoch {epoch + 1} / {epochs_CLS}, Loss: {loss.item()}")
        accuracy = compute_classifier_accuracy(classifier=classification_encoder, data_loader=test_CLS_loader)
        print(f"Epoch {epoch + 1} / {epochs_CLS}, Accuracy: {accuracy: .2f}%")

    utility = Utilities(tokenizer=tokenizer, model=classification_encoder)
    utility.sanity_check(sentence="In fact, I will be right there with you.", block_size=12)

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    decoder_only_model = Decoder(
        vocab_size=tokenizer.vocab_size,
        embed_size=n_embd,
        num_layers=n_layer,
        heads=n_head,
        device=device,
        forward_expansion=2,
        dropout=0.1,
        max_length=block_size
    )
    total_params = sum(p.numel() for p in decoder_only_model.parameters())
    print("The total parameters for decoder only model is:", total_params)
    optimizer = optim.Adam(decoder_only_model.parameters(), lr=learning_rate)
    # for i, (xb, yb) in tqdm(enumerate(train_LM_loader), total=len(train_LM_loader)):
    #     if i >= max_iters:
    #         break
    #     xb, yb = xb.to(device), yb.to(device)
    #     output = decoder_only_model(xb)
    #     output = output.view(-1, output.size(-1))
    #     yb = yb.view(-1)
    #     loss = criterion(output, yb)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if (i + 1) % eval_interval == 0:
    #         print(f"Step {i + 1} / {max_iters}, Loss: {loss.item()}")
    #         print("Evaluating on hbush data ....")
    #         perplexity_hbush = compute_perplexity_with_logits_output(decoder_only_model, test_LM_loader_hbush, criterion, eval_iters)
    #         print(f"Step {i + 1} / {max_iters}, H-Bush Perplexity: {perplexity_hbush: .2f}")
    #         print("Evaluating on obama data ....")
    #         perplexity_obama = compute_perplexity_with_logits_output(decoder_only_model, test_LM_loader_obama, criterion, eval_iters)
    #         print(f"Step {i + 1} / {max_iters}, Obama Perplexity: {perplexity_obama: .2f}")
    #         print("Evaluating on wbush data ....")
    #         perplexity_wbush = compute_perplexity_with_logits_output(decoder_only_model, test_LM_loader_wbush, criterion, eval_iters)
    #         print(f"Step {i + 1} / {max_iters}, W-Bush Perplexity: {perplexity_wbush: .2f}")
    
        # LM training code here


if __name__ == "__main__":
    main()
