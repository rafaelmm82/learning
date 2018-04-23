# -*- coding: utf-8 -*-
"""
First assignment Q03 - Dataset Generator
Class Deep Learning
UFPB

Mar, 31 2018.

Rafael Magalhães
GitHub @rafaelmm
"""

####################################
# IMPORTANT THINGS HERE
#
# There are two dataset generators here
# First - Function XOR, with input and outputs
# Second - Function f(x) = sin(pi*x)/(pi*x), 0 <= x <= 4
#
####################################


import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Dataset Generator - letter a
# -----------------------------------
def dataset_q03_a():
    """
    Generates a dataset that represents the XOR Function

    Because it is to small and an exact funciont there are no need
    to input parameters.

    It returns two tuples data_in and target.

    It could be used, specificaly to this case alos as test data
    """

    data_in = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])

    target = np.array([[0],
                       [1],
                       [1],
                       [0]])

    return (data_in, target)


# -----------------------------------
# Dataset Plot Viewer - letter a
# -----------------------------------
def plot_dataset_q03_a(data_in, predicted, true_value):
    """
    Function to show a simple table comparing the logic funcion XOR
    @Param
        everything numpy array
        data_in     - shape=(4,2)
        predicted   - shape=(4,1)
        true_value  - shape=(4,1)
    """
    precision = 100 - (np.abs(np.subtract(true_value, predicted))*100)

    print("")
    print("=============================================")
    print("| A  B | True Val | Predicted |  Precision  |")
    print("=============================================")
    
    for i in range(4):
        print("| "+str(data_in[i, 0])+"  "+str(data_in[i, 1])+" |    "+str(np.int(true_value[i]))+ "     |"+"   {:.3f}".format(predicted[i, 0]) + "   |   " + "% {: >5.2f}".format(precision[i, 0]) + "   |" )
    print("=============================================")
    print("")


# -----------------------------------
# Dataset Generator - letter b
# -----------------------------------
def dataset_q03_b(n_tra, n_val=0):
    """
    Generates a dataset that represents a sin function.

    @params:
    n_tra - the number of training examples to be generated
    n_val - the number of validation examples in dataset

    returns a tuple of NumPy arrays in the format of
    (training inputs, training targets, validation input, validation target)
    """

    # n_tra = 10
    # n_val = 8
    tra_in = np.array([[(x/n_tra) for x in range(1, 4*n_tra, 4)]])
    pi_x = np.pi * tra_in
    sin_pi_x = np.sin(pi_x)
    tra_out = np.nan_to_num(np.divide(sin_pi_x, pi_x))

    if n_val:
        val_in = np.array([[(x/n_val) for x in range(1, 4*n_val, 4)]])
        vpi_x = np.pi * val_in
        vsin_pi_x = np.sin(vpi_x)
        val_out = np.nan_to_num(np.divide(vsin_pi_x, vpi_x))
    else:
        val_in = np.zeros(shape=(1, 1))
        val_out = np.zeros(shape=(1, 1))

    return (tra_in.T, tra_out.T, val_in.T, val_out.T)


# -----------------------------------
# Dataset Plot Viewer - letter b
# -----------------------------------
def plot_dataset_q03_b(x, y, vi, vo):
    """
    Function to show the function f(x)=y
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.plot(x, y, color='black', linewidth=1)
    ax.scatter(x[:], y[:], color='r', marker='x')
    # ax.set_xlim(1, 6.5)
    # plt.savefig('foo.png')
    plt.title('Traning Dataset')
    plt.xlabel('X')
    plt.ylabel('Y = f(x)')
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.plot(x, y, color='blue', linewidth=1)
    ax.scatter(vi[:], vo[:], color='b', marker='x')
    # ax.set_xlim(1, 6.5)
    # plt.savefig('foo.png')
    plt.title('Validation Dataset')
    plt.xlabel('X')
    plt.ylabel('Y = f(x)')

    plt.show()
    

# -----------------------------------
# Dataset Plot Viewer - letter b
# -----------------------------------
def plot_prediction_q03_b(x, y, pi, po):
    """
    Function to show the function f(x)=y
    """
    pi=vi
    po = data_pred_d
    fig = plt.figure()
    plt.plot(x, y, color='black', linewidth=1)
    plt.plot(pi, po, color='red', linewidth=1, linestyle='dashed')
    plt.title('Real X Predicted')
    plt.xlabel('X')
    plt.ylabel('Y = f(x)')
    plt.show()
    vi


# -----------------------------------
# Dataset Plot Viewer - letter b
# -----------------------------------
def plot_erro_validation(erro, valid):
    """
    Function to show the function f(x)=y
    """
    #erro = tetotal_delta[0]
    #valid = vetotal_delta[0]
    fig = plt.figure()
    plt.plot(range(len(erro)), erro, color='black', linewidth=1)
    plt.plot(range(len(valid)), valid, color='red', linewidth=1, linestyle='dashed')
    plt.title('Erro Training X Erro Validation')
    plt.xlabel('Num Epochs')
    plt.ylabel('Error Value')
    plt.show()

