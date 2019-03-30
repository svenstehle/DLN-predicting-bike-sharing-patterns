# Predicting bike sharing patterns (time series) from Udacity Deep Learning Nanodegree
Cloned from Udacity. Goal is to design a neural network from scratch to get the basics down.
See **Introduction** in the PDF `Predicting Bike-Sharing Patterns - Udacity.pdf` for more details.

### Problem statement:
Our algorithm should predict the bike sharing patterns (total count of bike riders) for previously unseen data (days) in the future.
This is a regression task with a continuous output.

### Install and use:
* Clone repo
* Setup virtual environment according to pdf and download data `Bike-Sharing-Dataset` from udacity repo specified in pdf
* Follow `Your_first_neural_network.ipynb`
* You will also need `my_answers.py`

### Approach:
* Loading and preparing data is done for us
* Scaling of variables to values between 0 and 1 and appropriate splits into train, test and validation sets is taken care of for us
* Specified MLP architecture etc. in my_answers.py
* Initialized network weights
* Implemented necessary activation functions, forward pass, backpropagation and weight updates
* Set network hyperparameters
* Followed the notebook to train the network with stochastic gradient descent

### Possible improvements:
* Fine tuning: further hyperparameter tuning (hidden units, learning rate) and save the best model based on validation error

### Thoughts and lessons learned:
* Build a MLP from the ground up
* Stochastic gradient descent (SGD)
* Backpropagation

Predictions are pretty accurate for the timespan before Christmas and New Years. We didn't predict the latter days very well, but we did not have data on these special days (holiday period, less commuting, family time...) either, so that's to be expected. All in all we generalized well.

I needed a much higher learning rate as I initially thought because we scaled the learning rate down according to our batch size while applying SGD.
