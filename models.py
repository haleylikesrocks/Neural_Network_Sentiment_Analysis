# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random

from torch.utils import data
from sentiment_data import *
from torch.utils.data import Dataset, DataLoader

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], not_preprocessed=False) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words, not_preprocessed) for ex_words in all_ex_words]

class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, model):
        self.network = model

class DANClassifier(torch.nn.Module):
    def __init__(self, embeddings, inp=300, hid=32, out=2):
        super(DANClassifier, self).__init__()
        self.word_embeddings = embeddings
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings.vectors), padding_idx=0)

        self.V = nn.Linear(inp, hid)
        self.g = nn.ReLU()
        self.mid = nn.Linear(hid,hid)
        self.W = nn.Linear(hid, out)

    def preprocess(self, sentence):
        new_sentence =  []
        #index sentence
        for word in sentence:
            index = self.word_embeddings.word_indexer.index_of(word) if self.word_embeddings.word_indexer.index_of(word) != -1 else 0
            new_sentence.append(index)
        #pad with 0
        new_sentence += [0] * (52 - len(new_sentence))

        #translate to embedding
        embeddings = []
        for idx in new_sentence:
            embeddings.append(self.embedding(torch.tensor(idx)).numpy())
        ave_embedding = torch.mean(torch.tensor(embeddings), 0)

        return ave_embedding
            
    def predict(self, sentence):
        x = self.preprocess(sentence)
        return self.forward(x).max(0)[1]
    
    def predict_all(self, all_ex_words: List[List[str]], not_preprocessed=True) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]

    def forward(self, x):
        x = self.V(x.float())
        x = self.g(x)
        x = self.mid(x)
        x = self.g(x)
        x = self.W(x)
        return x

def get_batches(data, batch_size):
    count = 0
    batches = []
    count_down = len(data)
    while count_down > batch_size:
        batches.append(data[count:(count + batch_size)])
        count += batch_size
        count_down -= batch_size
    if count_down > 1:
        batches.append(data[count:])
    
    return batches

def get_labels_and_data(batch):
    labels = []
    data = []
    for datem, label in batch:
        labels.append(label)
        data.append(np.array(datem, dtype=np.float32))
    return torch.tensor(data), torch.tensor(labels)

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # Define hyper parmeters and model
    num_epochs = 8
    initial_learning_rate = 0.01
    batch_size = 32

    # Model specifications
    model = DANClassifier(word_embeddings)
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    loss_funct = torch.nn.CrossEntropyLoss()

    # Preprocess data
    print("Preprocessing the Training data")
    train_data = []
    for item in train_exs: #for testing
        train_data.append((model.preprocess(item.words), item.label))
    
    dev_data = []
    for item in dev_exs:
        dev_data.append((model.preprocess(item.words), item.label))

    for epoch in range(num_epochs):
        # print("entering epoch %i" % epoch)
        # set epoch level varibles
        total_loss = 0.0
        accuracys = []

        #Batch the data
        random.shuffle(train_data)
        batches = get_batches(train_data, batch_size)

        for batch in batches:
            batch_data, batch_label = get_labels_and_data(batch)

            model.zero_grad()
            y_pred = model.forward(batch_data)
            
            # calculate loss and accuracy
            loss = loss_funct(y_pred, batch_label)
            total_loss += loss
            for i in range(len(batch)):
                ret = 1 if y_pred[i].max(0)[1] == batch_label[i] else 0
                accuracys.append(ret)
            
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()

        # Dev Testing
        dev_accuracys = []
        batches = get_batches(dev_data, batch_size)
        for batch in batches:
            batch_data, batch_label = get_labels_and_data(batch)
            y_pred = model.forward(batch_data)
            for i in range(len(batch)):
                ret = 1 if y_pred[i].max(0)[1] == batch_label[i] else 0
                dev_accuracys.append(ret)

        print("Total loss on epoch %i: %f" % (epoch, total_loss))
        print("The traing set accuracy for epoch %i: %f" % (epoch, np.mean(accuracys)))
        print("The dev set accuracy for epoch %i: %f" % (epoch, np.mean(dev_accuracys)))

    return model


class Words(Dataset):
    def __init__(self, examples, word_embeddings: WordEmbeddings):
        self.data = []

        for item in examples:
            #pass to indexer
            new_sentence =  []
            for word in item.words:
                index = word_embeddings.word_indexer.index_of(word) if word_embeddings.word_indexer.index_of(word) != -1 else 0
                new_sentence.append(index)
            # pad with 0
            new_sentence += [0] * (52 - len(new_sentence))
            
            self.data.append((np.array(new_sentence), item.label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(examples, embedder, num_workers=0, batch_size=32):
    dataset = Words(examples, embedder)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)
        


