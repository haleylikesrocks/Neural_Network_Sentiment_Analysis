# ffnn_example.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random


class FFNN(nn.Module):
    """
    Defines the core neural network for doing multiclass classification over a single datapoint at a time. This consists
    of matrix multiplication, tanh nonlinearity, another matrix multiplication, and then
    a log softmax layer to give the ouputs. Log softmax is numerically more stable. If you take a softmax over
    [-100, 100], you will end up with [0, 1], which if you then take the log of (to compute log likelihood) will
    break.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, inp, hid, out):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(FFNN, self).__init__()
        self.V = nn.Linear(inp, hid)
        # self.g = nn.Tanh()
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
        # Initialize with zeros instead
        # nn.init.zeros_(self.V.weight)
        # nn.init.zeros_(self.W.weight)

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        return self.log_softmax(self.W(self.g(self.V(x))))


def form_input(x) -> torch.Tensor:
    """
    Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
    of data, does some computation, handles batching, etc.

    :param x: a [num_samples x inp] numpy array containing input data
    :return: a [num_samples x inp] Tensor
    """
    return torch.from_numpy(x).float()


# Example of training a feedforward network with one hidden layer to solve XOR.
if __name__=="__main__":
    # MAKE THE DATA
    # Synthetic data for XOR: y = x1 XOR x2
    train_xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    train_ys = np.array([0, 1, 1, 0], dtype=np.float32)
    # Define some constants
    # Inputs are of size 2
    feat_vec_size = 2
    # Let's use 4 hidden units
    embedding_size = 4
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # RUN TRAINING AND TEST
    num_epochs = 100
    ffnn = FFNN(feat_vec_size, embedding_size, num_classes)
    initial_learning_rate = 0.1
    optimizer = optim.Adam(ffnn.parameters(), lr=initial_learning_rate)
    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_xs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x = form_input(train_xs[idx])
            y = train_ys[idx]
            # Build one-hot representation of y. Instead of the label 0 or 1, y_onehot is either [0, 1] or [1, 0]. This
            # way we can take the dot product directly with a probability vector to get class probabilities.
            y_onehot = torch.zeros(num_classes)
            # scatter will write the value of 1 into the position of y_onehot given by y
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            ffnn.zero_grad()
            log_probs = ffnn.forward(x)
            # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
            loss = torch.neg(log_probs).dot(y_onehot)
            total_loss += loss
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    # Evaluate on the train set
    train_correct = 0
    for idx in range(0, len(train_xs)):
        x = form_input(train_xs[idx])
        y = train_ys[idx]
        log_probs = ffnn.forward(x)
        prediction = torch.argmax(log_probs)
        if y == prediction:
            train_correct += 1
        print("Example " + repr(train_xs[idx]) + "; gold = " + repr(train_ys[idx]) + "; pred = " +\
              repr(prediction) + " with probs " + repr(log_probs))
    print(repr(train_correct) + "/" + repr(len(train_ys)) + " correct after training")