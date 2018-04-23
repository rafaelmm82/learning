# -*- coding: utf-8 -*-
"""
First assignment important Functions
Class Deep Learning
UFPB

Mar, 31 2018.

Rafael Magalhães
GitHub @rafaelmm
"""

####################################
# IMPORTANT THINGS HERE
# Necessary and important functions to run ANN models
# Including:
#
#        activation_func(func_type, z):
#        deriv_activation_func(func_type, z):
#        visualizeActivationFunc(func_type, z):
#        visualizeDerivActivationFunc(func_type, z):
#        forward(net_arc, net_func, w, b, X):
#        predict(output, min_basis=0):
#
####################################


import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Plot Error Viewer
# -----------------------------------
def plot_error(error_vector):
    """
    Simple Function to show the progress of the error vector
    """
    plt.figure()
    plt.plot(range(len(error_vector)), error_vector)
    plt.title('Error value evolution')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Error value')
    plt.show()


# -----------------------------------
# Weights Structure Creator
# -----------------------------------
def weights_init(net_arc):
    """
    Function that initialize the weights randomly using the numpy
    library.
    @Param: w - weightsvalues
    """

    num_layers = np.shape(net_arc)[0]
    max_neu = np.max(net_arc)
    w = np.zeros(shape=([num_layers-1, max_neu, max_neu+1]))

    for layer in range(num_layers-1):
        for neuron in range(net_arc[layer+1]):
            for conexion in range(net_arc[layer]+1):
                w[layer][neuron][conexion] = np.random.random() - 0.5

    return w


# -----------------------------------
# Activation Functions
# -----------------------------------
def activation_func(func_type, z):
    """
    Implements the different kind of activation functions including:
        line - linear function
        sigm - sigmoidal
        tanh - hyperbolic tangent
        ptanh - smothly hyperbolic tangent
        relu - Rectfied
        step - Heavside (binary step 0 or 1)
    """
    if func_type == 'line':
        return z

    if func_type == 'sigm':
        return (1 / (1 + np.exp(-z)))

    if func_type == 'tanh':
        return (np.tanh(z))

    if func_type == 'ptanh':
        a = 1.7159
        b = 2/3
        return (a*np.tanh(b*z))

    if func_type == 'relu':
        return np.max(np.array([0, z]))

    if func_type == 'step':
        return (1 if z >= 0.5 else 0)


# -----------------------------------
# Derivated Activation Functions
# -----------------------------------
def deriv_activation_func(func_type, z):
    """
    Implements the different kind of derivated activation functions including:
        line - linear function
        sigm - sigmoidal
        tanh - hyperbolic tangent
        ptanh - smothly hyperbolic tangent
        relu - Rectfied
        step - Heavside (binary step 0 or 1)
    """

    if func_type == 'line':
        return 1

    if func_type == 'sigm':
        return (1 / (1 + np.exp(-z))) - ((1 / (1 + np.exp(-z))) ** 2)

    if func_type == 'tanh':
        return (1/((np.cosh(z) ** 2)))

    if func_type == 'ptanh':
        a = 1.7159
        b = 2/3
        return (a*b*(1-(np.tanh(b*z)**2)))

    if func_type == 'relu':
        return np.max(np.array([0, z]))

    if func_type == 'step':
        return (1 if z >= 0.5 else 0)


# -----------------------------------
# Activation Functions Plotter
# -----------------------------------
def visualizeActivationFunc(func_type, z):
    """
    Makes a plot of the activation function with input z
    """

    fz = []
    for i in range(len(z)):
        fz.append(activation_func(func_type, z[i]))

    plt.figure()
    plt.plot(z, fz)
    plt.xlabel('Input')
    plt.ylabel('Output Values')
    plt.show()


# -----------------------------------
# Derivated Activation Functions Plotter
# -----------------------------------
def visualizeDerivActivationFunc(func_type, z):
    """
    Makes a plot of the activation function with input z
    """

    fz = []
    for i in range(len(z)):
        fz.append(deriv_activation_func(func_type, z[i]))

    plt.figure()
    plt.plot(z, fz)
    plt.xlabel('Input')
    plt.ylabel('Output Values')
    plt.show()


# -----------------------------------
# Forward Step of Neural Net
# -----------------------------------
def forward(net_arc, net_func, w, b, X):
    """
    The forward pathway of the mlp neural net, it calculates the result of the
    structure considering the X input. It passthroug each neuron of each layer.
    """

    num_layers = np.shape(net_arc)[0]
    max_neu = np.max(net_arc)
    Y = np.zeros(shape=([num_layers, max_neu]))

    for layer in range(num_layers):

        if layer == 0:
            for neuron in range(net_arc[layer]):
                Y[layer, neuron] = X[neuron]

        else:

            for neuron in range(net_arc[layer]):
                act_sum = np.dot(w[layer-1, neuron, 1:(net_arc[layer-1]+1)],
                                 Y[layer-1, 0:net_arc[layer-1]]) + \
                                 w[layer-1, neuron, 0]*b
                Y[layer, neuron] = activation_func(net_func[layer], act_sum)

    # returning the output layer, the last one
    return Y[num_layers-1, 0:(net_arc[num_layers-1])]


# -----------------------------------
# Predict Limiar Output
# -----------------------------------
def predict(output, min_basis=0):
    """
    It's just to round prediction of the perceptron to making
    results more conforming with the real target value
    @param:
        output - the value to convert to [0,1] or [-1, 1]
        min_basis - the lowest value (0=default, -1)
                    if 0 the medium value is 0.5
                    if -1 the medium value is 0
    """

    if min_basis == 0:
        y_pred = [1 if x >= 0.5 else 0 for x in output]
    else:
        y_pred = [1 if x >= 0 else -1 for x in output]

    return y_pred


# -----------------------------------
# MSE Function
# -----------------------------------
def MSE(net_arc, net_func, w, b, data_in, target):
    """
    This function execute the algorithm of weights adjustes
    following the steps of measure the error and changing the
    w structure
    @param:
        w - weights structure
        data_in - training dataset
        target - training targets of dataset
        num_epochs - the total overall loopings to adjuste the w
        learn_rate - the coefficient that ponderate the changes in w
    """

    num_examples = np.shape(data_in)[0]
    mserror = 0

    ex_error_track = np.zeros((num_examples, 1))

    for ex in range(num_examples):
        y_pred = forward(net_arc, net_func, w, b, data_in[ex])
        ex_error = target[ex] - np.transpose(y_pred)
        ex_error_track[ex] = np.sum(ex_error ** 2) / 2

    mserror = np.sum(ex_error_track) / num_examples

    return mserror


# -----------------------------------
# Plot Error Total Grid 9
# -----------------------------------
def plot_error_total_6(trt, lrate):
    """
    Function to show the progress of the errors vectors in a 9 grid
    """

    plt.figure(dpi=200)

    for lr in range(6):

        tra_mean = np.mean(trt[lr], axis=0)

        plt.subplot(231 + lr)

        for i in range(9):
            plt.plot(range(len(trt[lr][i])), trt[lr][i], c='k', linewidth=0.5)

        plt.plot(range(len(tra_mean)), tra_mean, c='r', linestyle='dashed',
                 label='Training Error')
        plt.xticks(fontsize=8)
        plt.title('Mean TErr n='+str(lrate[lr]), fontsize=8)
        plt.xlabel('Number of Epochs', fontsize=8)
        plt.ylabel('Error values', fontsize=7)

    plt.tight_layout()
    plt.show()


# -----------------------------------
# Plot Error Total Grid 6 momentum
# -----------------------------------
def plot_error_total_6_mom(trt, lrate):
    """
    Function to show the progress of the errors vectors in a 9 grid
    """

    plt.figure(dpi=200)

    for lr in range(6):

        tra_mean = np.mean(trt[lr], axis=0)

        plt.subplot(231 + lr)

        for i in range(9):
            plt.plot(range(len(trt[lr][i])), trt[lr][i], c='k', linewidth=0.5)

        plt.plot(range(len(tra_mean)), tra_mean, c='r', linestyle='dashed',
                 label='Training Error')
        plt.xticks(fontsize=8)
        plt.title('Mean TErr n=0.6, alfa='+str(lrate[lr]), fontsize=8)
        plt.xlabel('Number of Epochs', fontsize=8)
        plt.ylabel('Error values', fontsize=7)

    plt.tight_layout()
    plt.show()


# -----------------------------------
# Plot Error Total Grid 9
# -----------------------------------
def plot_error_total_6_batch(trt, lrate, bvec):
    """
    Function to show the progress of the errors vectors in a 9 grid
    """
    
    plt.figure(dpi=200)
    count=0

    for bs in range(len(bvec)):
        
        for lr in range(len(lrate)):
    
            tra_mean = np.mean(trt[bs][lr], axis=0)
    
            plt.subplot(231 + count)
            count+=1
            
    
            for i in range(9):
                plt.plot(range(len(trt[bs][lr][i])), trt[bs][lr][i], c='k', linewidth=0.5)
    
            plt.plot(range(len(tra_mean)), tra_mean, c='r', linestyle='dashed',
                     label='Training Error')
            plt.xticks(fontsize=8)
            plt.title('Mean TErr bs='+str(bvec[bs])+', n='+str(lrate[lr]), fontsize=8)
            plt.xlabel('Number of Epochs', fontsize=8)
            plt.ylabel('Error values', fontsize=7)

    plt.tight_layout()
    plt.show()


# -----------------------------------
# Plot MEANS Error Total Grid 2
# -----------------------------------
def plot_the_means(trt, vat):
    """
    Function to show the means for each graph error courve
    """

    tra_mean = np.mean(trt, axis=0)
    val_mean = np.mean(vat, axis=0)

    plt.figure(dpi=200)

    plt.subplot(121)
    for i in range(3):
        plt.plot(range(len(trt[i])), trt[i], c='k', linewidth=0.5)

    plt.plot(range(len(tra_mean)), tra_mean, c='r', linestyle='dashed',
             label='Training Error')
    plt.xticks(fontsize=8)
    plt.title('Mean Training Error', fontsize=8)
    plt.xlabel('Number of Epochs', fontsize=8)
    plt.ylabel('Error values', fontsize=7)

    plt.subplot(122)
    for i in range(3):
        plt.plot(range(len(vat[i])), vat[i], c='k', linewidth=0.5)

    plt.plot(range(len(val_mean)), val_mean, c='b', linestyle='dashed',
             label='Training Error')
    plt.xticks(fontsize=8)
    plt.title('Mean Validation Error', fontsize=8)
    plt.xlabel('Number of Epochs', fontsize=8)
    plt.ylabel('Error values', fontsize=7)

    plt.tight_layout()
    plt.show()


# -----------------------------------
# Plot MEANS Error Total Unique
# -----------------------------------
def plot_the_means_1(trt):
    """
    Function to show the means for each graph error courve
    """

    tra_mean = np.mean(trt, axis=0)

    plt.figure(dpi=200)

    for i in range(9):
        plt.plot(range(len(trt[i])), trt[i], c='k', linewidth=0.5)

    plt.plot(range(len(tra_mean)), tra_mean, c='r', linestyle='dashed',
             label='Training Error')
    plt.xticks(fontsize=8)
    plt.title('Mean Training Error', fontsize=8)
    plt.xlabel('Number of Epochs', fontsize=8)
    plt.ylabel('Error values', fontsize=7)

    plt.tight_layout()
    plt.show()


# -----------------------------------
# Model Plot Viewer
# -----------------------------------
def plot_model_viewer_xor(net_arc, net_func, w, b):
    """
    Function to show in a 3D scatter plot the model classification
    """

    fig = plt.figure()

    bp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
    plt.scatter(bp[:,0], bp[:,1], c='k', s=50, marker='o', alpha=0.5)

    for i in range(4):
        plt.text((bp[i,0]), bp[i,1], 'o(%d, %d)' % (bp[i,0], bp[i,1]), color='black', fontsize=10)

    for x in range(101):
        for y in range(101):
            # point = [(0.05*x),(0.05*y)]
            point = [(0.01*x),(0.01*y)]
            value = predict(forward(net_arc, net_func, w, b, point))
            plt.scatter(point[0], point[1], c='C'+str(value[0]), s=10, marker='o')

    # title and axis labels
    plt.title('Dataset Model Distribution')
    plt.xlabel('Input 1 [A, ]')
    plt.ylabel('Input 2 [ ,B]')

    plt.show()

