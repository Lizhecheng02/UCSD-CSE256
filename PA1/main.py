# models.py
import yaml
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, SentimentDatasetDAN_BPE, DAN
from tqdm import tqdm


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


def train_epoch_dan(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss = 0
    correct = 0
    for batch, (X, y) in tqdm(enumerate(data_loader), total=len(data_loader)):
        X = X.float()
        y = y.float()

        pred = model(X).squeeze(1)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        correct += ((torch.sigmoid(pred) > 0.5) == y).type(torch.float).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


def eval_epoch_dan(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in tqdm(enumerate(data_loader), total=len(data_loader)):
        X = X.float()
        y = y.float()

        pred = model(X).squeeze(1)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += ((torch.sigmoid(pred) > 0.5) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader, lr):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f"Epoch #{epoch + 1}: train accuracy {train_accuracy: .3f}, dev accuracy {test_accuracy: .3f}")
    
    return all_train_accuracy, all_test_accuracy


def experiment_dan(model, train_loader, test_loader, lr, epochs):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(epochs):
        train_accuracy, train_loss = train_epoch_dan(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch_dan(test_loader, model, loss_fn)
        all_test_accuracy.append(test_accuracy)

        if epoch % 1 == 0:
            print(f"Epoch #{epoch + 1}: train accuracy {train_accuracy: .3f}, dev accuracy {test_accuracy: .3f}")
    
    return all_train_accuracy, all_test_accuracy


def main():
    # Set up argument parser
    # parser = argparse.ArgumentParser(description="Run model training based on specified model type")
    # parser.add_argument("--model", type=str, required=True, help="Model type to train (e.g., BOW)")

    # Parse the command-line arguments
    # args = parser.parse_args()

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", default="config.yaml", type=str, help=".yaml file path", required=False)
    args = parser.parse_args()
    config = load_config(args.config)

    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))

    args = parser.parse_args()

    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate NN2
        start_time = time.time()
        print("\n2 layers:")
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader, lr=args.lr)

        # Train and evaluate NN3
        print("\n3 layers:")
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader, lr=args.lr)

        # Plot the training accuracy
        plt.figure(figsize=(9, 5.5))
        plt.plot(nn2_train_accuracy, label="2 layers")
        plt.plot(nn3_train_accuracy, label="3 layers")
        plt.xlabel("Epochs")
        plt.ylabel("Training Accuracy")
        plt.title("Training Accuracy for 2, 3 Layer Networks")
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = "train_accuracy.png"
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(9, 5.5))
        plt.plot(nn2_test_accuracy, label="2 layers")
        plt.plot(nn3_test_accuracy, label="3 layers")
        plt.xlabel("Epochs")
        plt.ylabel("Dev Accuracy")
        plt.title("Dev Accuracy for 2 and 3 Layer Networks")
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = "dev_accuracy.png"
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        start_time = time.time()

        if args.use_bpe_trainer:
            print("Train Tokenizer From Scratch!!!")
            train_data = SentimentDatasetDAN_BPE("data/train.txt", bpe_vocab_size=args.bpe_vocab_size, max_length=args.max_length)
            dev_data = SentimentDatasetDAN_BPE("data/dev.txt", bpe_vocab_size=args.bpe_vocab_size, max_length=args.max_length)
            train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
        else:
            train_data = SentimentDatasetDAN("data/train.txt", embs_path=args.embed_file, max_length=args.max_length)
            dev_data = SentimentDatasetDAN("data/dev.txt", embs_path=args.embed_file, max_length=args.max_length)
            train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        model = DAN(
            use_random_embed=args.use_random_embed,
            vocab_size=args.bpe_vocab_size if args.use_bpe_trainer else args.vocab_size,
            embed_file=args.embed_file,
            freeze_embed=args.freeze_embed,
            input_size=args.input_size,
            hidden_sizes=args.hidden_sizes,
            output_size=args.output_size,
            num_classes=args.num_classes,
            use_dropout=args.use_dropout,
            dropout_rate=args.dropout_rate
        )
        print(model)

        start_time = time.time()

        dan_train_acc, dan_test_acc = experiment_dan(model, train_loader, test_loader, lr=args.lr, epochs=args.epochs)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Model trained in : {elapsed_time} seconds")

        plt.figure(figsize=(9, 5.5))
        plt.plot(dan_train_acc, label="DAN_BPE" if args.use_bpe_trainer else "DAN")
        plt.xlabel("Epochs")
        plt.ylabel("Training Accuracy")
        plt.title("Training Accuracy for DAN BPE Networks" if args.use_bpe_trainer else "Training Accuracy for DAN Networks")
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = "train_accuracy_dan_bpe.png" if args.use_bpe_trainer else "train_accuracy_dan.png"
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(9, 5.5))
        plt.plot(dan_test_acc, label="DAN_BPE" if args.use_bpe_trainer else "DAN")
        plt.xlabel("Epochs")
        plt.ylabel("Dev Accuracy")
        plt.title("Dev Accuracy for DAN BPE Networks" if args.use_bpe_trainer else "Dev Accuracy for DAN Networks")
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = "dev_accuracy_dan_bpe.png" if args.use_bpe_trainer else "dev_accuracy_dan.png"
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")



if __name__ == "__main__":
    main()
