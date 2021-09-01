# Neural_Network_Sentiment_Analysis
Assignment 2 of CS388 from UT Austin NLP class

## Q1 (25 points) 
First we start with optimization.py, which defines a quadratic with two variables:
y = (x1 - 1) ** 2 + 8 * (x2 - 1) ** 2

This file contains a manual implementation of SGD for this function.

a) Implement the gradient of the provided quadratic function in quadratic grad. sgd test quadratic
will then call this function inside an SGD loop and show a visualization of the learning process. Note: you
should not use PyTorch for this part!

b) When initializing at the origin, what is the best step size to use? Set your step size so that it gets to
a distance of within 0.1 of the optimum within as few iterations as possible. Several answers are possible.
Hardcode this value into your code.

Exploration (optional) What is the “tipping point” of the step size parameter, where step sizes larger than
that cause SGD to diverge rather than find the optimum? 
=> At lr = 0.15 the program does not converge, but at lr = 0.11 the program converges in 8 epochs (at 0.12 the program converges more slowly)
