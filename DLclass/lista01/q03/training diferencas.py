# -*- coding: utf-8 -*-
"""
First assignment - ann training net, delta rule
Class Deep Learning
UFPB

Mar, 31 2018.

Rafael Magalhães
GitHub @rafaelmm
"""

####################################
# IMPORTANT THINGS HERE
#
####################################

from assignmentfunctions import *
import numpy as np
from sklearn.utils import shuffle


# -----------------------------------
# Training Function Stochastic
# (shuffling all dataset on each epoch)
# -----------------------------------
def training_net_delta(net_arc, net_func, w, b, data_in, target, learn_rate,
                       num_epochs, err_max=0.0001):
    """
    This function execute the algorithm of weights adjustes
    following the steps of measure the error and changing the
    w structure by its gradient
    @args
        w - weights structure
        data_in - training dataset
        target - training targets of dataset
        num_epochs - the total overall loopings to adjuste the w
        learn_rate - the coefficient that ponderate the changes in w
        err_max - a specified value for maximum error acepted in training
    """

    # num of layers
    num_layers = np.shape(net_arc)[0]
    # max num of neurons on any layer
    max_neu = np.max(net_arc)
    # num of examples in input dataset
    num_examples = np.shape(data_in)[0]
    # output size (last network layer size)
    out_size = net_arc[len(net_arc)-1]
    # local error for each example (or instantaneous error)
    err_local = np.zeros(shape=(num_examples, 1))
    # The value of output for each neuron for an especific example
    Y = np.zeros(shape=([num_layers, max_neu]))
    # The value of soma(the accumulator vefore the activator function)
    soma = np.zeros(shape=([num_layers, max_neu]))
    # The result of gardient descendent for each neuron
    gradi = np.zeros(shape=([num_layers, max_neu]))
    # auxiliar temporary weights
    oldw = np.copy(w)

    # the vector error to total epohcs (mean squarred error)
    err_vec = np.zeros((num_epochs, 1))

    # Starting the trainning loop
    for ep in range(num_epochs):

        # cleaning local error and mse for each epoch
        err_local = np.zeros(shape=(num_examples, 1))
        ms_error = 0

        # shuffling the data set

        # for each example
        # Stochastic - shuffle in each epoch
        for example in list(shuffle(range(num_examples))):

            # ----------------
            # 1 - Forward Step
            # ----------------

            for layer in range(num_layers):

                if layer == 0:
                    for neuron in range(net_arc[layer]):
                        Y[layer, neuron] = data_in[example, neuron]

                else:

                    for neuron in range(net_arc[layer]):
                        soma[layer, neuron] = np.dot(w[layer-1, neuron, 1:(net_arc[layer-1]+1)], Y[layer-1, 0:net_arc[layer-1]]) + w[layer-1, neuron, 0]*b
                        Y[layer, neuron] = activation_func(net_func[layer],
                             soma[layer, neuron])

            # ---------------------
            # 2 - Error Measurement
            # ---------------------

            # to calculate example squared error
            err_example = np.zeros(shape=(out_size, 1))

            for neuron in range(out_size):
                err_example[neuron] = target[example, neuron] - Y[num_layers-1, neuron]

            # err_example = err_example ** 2

            err_local[example] = np.sum(err_example ** 2) / 2

            # ---------------------
            # 3 - Backpropagation
            # ---------------------

            # Just last and hidden layers (not input layer)
            for layer in range(num_layers-1, 0, -1):

                # if last layer
                if layer == (num_layers - 1):

                    for neuron in range(out_size):

                        gradi[layer, neuron] = err_example[neuron] * deriv_activation_func(net_func[layer], soma[layer, neuron])

                        # bias update
                        deltaw = learn_rate * b * gradi[layer, neuron]
                        aux = w[layer-1, neuron, 0]
                        w[layer-1, neuron, 0] = aux + deltaw
                        oldw[layer-1, neuron, 0] = aux

                        # other weights
                        for weight in range(net_arc[layer-1]):

                            deltaw = learn_rate * gradi[layer, neuron] * Y[layer-1, weight]
                            aux = w[layer-1, neuron, weight+1]
                            w[layer-1, neuron, weight+1] = aux + deltaw
                            oldw[layer-1, neuron, weight+1] = aux

                # if hidden layer (not last)
                else:

                    for neuron in range(net_arc[layer]):

                        soma_gradi = 0

                        # for each neuron on step ahead layer
                        for kneuron in range(net_arc[layer+1]):
                            soma_gradi += gradi[layer+1, kneuron] * oldw[layer, kneuron, neuron+1]

                        gradi[layer, neuron] = soma_gradi * deriv_activation_func(net_func[layer], soma[layer, neuron])

                        # bias update
                        deltaw = learn_rate * b * gradi[layer, neuron]
                        aux = w[layer-1, neuron, 0]
                        w[layer-1, neuron, 0] = aux + deltaw
                        oldw[layer-1, neuron, 0] = aux

                        # other weights
                        for weight in range(net_arc[layer-1]):

                            deltaw = learn_rate * gradi[layer, neuron] * Y[layer-1, weight]
                            aux = w[layer-1, neuron, weight+1]
                            w[layer-1, neuron, weight+1] = aux + deltaw
                            oldw[layer-1, neuron, weight+1] = aux


        # Mean squarred error for this epoch
        ms_error = np.sum(err_local) / num_examples
        print("ms_erro - " + str(ms_error))

        err_vec[ep] = ms_error


    return (w, ep, err_vec)
