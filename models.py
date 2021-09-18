# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


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

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


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
    def __init__(self):
        self.network = DANClassifier(1,32,2)

class DANClassifier(torch.nn.Module):
    def __init__(self, inp=1, hid=32, out=2):
        super(DANClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(5000, 40, padding_idx=0)
        self.V = nn.Linear(inp, hid)
        # self.g = nn.Tanh()
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        #take embedding
        #average embedding

        x =  self.embedding(torch.FloatTensor(x))
        return self.log_softmax(self.W(self.g(self.V(x))))

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model

    """
    # Define hyper parmeters and model
    num_epochs = 1
    initial_learning_rate = 0.1
    batch_size = 32

    # Model specifications
    model = DANClassifier()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    loss = torch.nn.NLLLoss() # because of soft max


    for epoch in range(num_epochs):
        #shuffle data
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)

        # set sepoch level varibles
        total_loss = 0.0
        accuracys = []

        for idx in ex_indices:
            #create batch

            #### batch here 
            # for each sample  paddinx* total lengths
            
            ### grab first n-batch size  put into matrix/n dim tensor [batchsize, len_sent (52?)] [32 x 32 x len vocab - how to get?]
            ### pad with np 0 array
            #####
            x = train_exs[idx].words
            y_true = train_exs[idx].label

            # iterrate 
            # use indeter to convert from word to indices - pass to embedder 
            #  word_embeddings.word_indexer
            # nn.em
            # self.embedding = torch.nn.Embedding.from_pretrained - look up on pytorch(from indexer, 52, padding_idx=0, from_pretrained - glove 300-d)
            # do embedding sperately for each item in the batch
            # run on each word and each example - nested for loop

            # run through model
            model.zero_grad()
            y_pred = model.forward(x)
            
            # calculate loss and accuracy
            total_loss += loss(y_pred, y_true)
            accuracys.append(calculate_accuracy(y_pred, y_true))
            
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()

        print("Total loss on epoch %i: %f" % (epoch, total_loss))
        print("The accuracy for epoch %i: %f" % (epoch, np.mean(accuracys)))


def calculate_accuracy(y_predict, y_true):
    #calculates te acurracy of batched samples
    acc = []
    for index in range(len(y_predict)):
        state = 1 if y_predict[index] == y_true[index] else 0
        acc.append(state)
    return np.mean(acc)
        


