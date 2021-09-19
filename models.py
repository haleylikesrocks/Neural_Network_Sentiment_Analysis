# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
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
    def __init__(self, model):
        self.network = model

class DANClassifier(torch.nn.Module):
    def __init__(self, embeddings, inp=1, hid=32, out=2):
        super(DANClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings.vectors), padding_idx=0)
        self.V = nn.Linear(inp, hid)
        # self.g = nn.Tanh()
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        # nn.init.xavier_uniform_(self.V.weight)
        # nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        #take embedding
        #average embedding
        embedded_data = torch.empty((1,300))
        for i, sentence in enumerate(x):
            embbedings = torch.empty((1,300))
            for index, word in enumerate(sentence):
                embedded_word =  self.embedding(word)
                embbedings = torch.cat((embbedings, torch.unsqueeze(embedded_word,0)))
            ave_embed = torch.mean(torch.tensor(embbedings), 0)
            embedded_data = torch.cat((embedded_data, torch.unsqueeze(ave_embed, 0)))  
        return self.log_softmax(self.W(self.g(self.V(torch.tensor(embedded_data)))))

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
    model = DANClassifier(word_embeddings)
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    loss = torch.nn.NLLLoss() # because of soft max

    train_data =  load_data(train_exs, word_embeddings)
    dev_data = load_data(dev_exs, word_embeddings)


    for epoch in range(num_epochs):
        # set epoch level varibles
        total_loss = 0.0
        accuracys = []

        for data, labels in train_data:
            print(torch.tensor(data).shape) #32 x 52

            # print(len(labels))
            # run through model
            model.zero_grad()
            y_pred = model.forward(data)
            
            # calculate loss and accuracy
            total_loss += loss(y_pred, labels)
            accuracys.append(calculate_accuracy(y_pred, labels))
            
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()

        #work trough dev examples

        print("Total loss on epoch %i: %f" % (epoch, total_loss))
        print("The accuracy for epoch %i: %f" % (epoch, np.mean(accuracys)))

    final = NeuralSentimentClassifier(model)


def calculate_accuracy(y_predict, y_true):
    #calculates te acurracy of batched samples
    acc = []
    for index in range(len(y_predict)):
        state = 1 if y_predict[index] == y_true[index] else 0
        acc.append(state)
    return np.mean(acc)


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
        


