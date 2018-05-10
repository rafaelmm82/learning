# -*- coding: utf-8 -*-
"""
First assignment Q04 - Dataset Generator
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
# Dataset Generator - Problem 4
# -----------------------------------
def dataset_q04(n_tra, n_val=0, n_tes=0):
    """
    Generates a dataset that represents the XOR Function

    Because it is to small and an exact funciont there are no need
    to input parameters.

    It returns two tuples data_in and target.

    It could be used, specificaly to this case alos as test data
    """

    total = n_tra + n_val + n_tes

    set_in = np.random.rand(total, 2) * 2.5 - 1.25
    allset_out = np.zeros(shape=(total, 9), dtype=int)

    set_p2 = np.power(set_in, 2)
    set_abssum = np.array([np.add(np.abs(set_in[:, 0]),
                                  np.abs(set_in[:, 1]))]).T
    radius = np.sqrt(np.add(set_p2[:, 0], set_p2[:, 1]))
    set_norm = np.array([radius]).T
    set_prod = np.array([np.prod(set_in, axis=1)]).T

    parcel1 = np.append(set_in, set_abssum, axis=1)
    parcel2 = np.append(parcel1, set_prod, axis=1)
    allset_in = np.append(parcel2, set_norm, axis=1)

    for i in range(total):

        if radius[i] > 1:
            allset_out[i, 0] = 1
        else:
            if (allset_in[i, 0] >= 0) and (allset_in[i, 1] >= 0):
                if set_abssum[i] < 1:
                    allset_out[i, 1] = 1
                else:
                    allset_out[i, 5] = 1
            if (allset_in[i, 0] < 0) and (allset_in[i, 1] >= 0):
                if set_abssum[i] < 1:
                    allset_out[i, 2] = 1
                else:
                    allset_out[i, 6] = 1
            if (allset_in[i, 0] < 0) and (allset_in[i, 1] < 0):
                if set_abssum[i] < 1:
                    allset_out[i, 3] = 1
                else:
                    allset_out[i, 7] = 1
            if (allset_in[i, 0] >= 0) and (allset_in[i, 1] < 0):
                if set_abssum[i] < 1:
                    allset_out[i, 4] = 1
                else:
                    allset_out[i, 8] = 1

    tin = allset_in[0:n_tra, :]
    tout = allset_out[0:n_tra, :]
    vin = allset_in[n_tra:(n_tra+n_val), :]
    vout = allset_out[n_tra:(n_tra+n_val), :]
    tesin = allset_in[total-n_tes:total, :]
    tesout = allset_out[total-n_tes:total, :]

    return (tin, tout, vin, vout, tesin, tesout)


# -----------------------------------
# Dataset Plot Viewer - letter a
# -----------------------------------
def plot_dataset_q04(din, dout):

    fig = plt.figure()

    total = len(din)

    for i in range(total):
        value = np.argmax(dout[i])
        # plt.scatter(din[i,0], din[i,1], c='C'+str(value), s=100, marker='o')
        plt.scatter(din[i,0], din[i,1], c='C'+str(value), s=20, marker='x')
        # plt.scatter(din[i,0], din[i,1], c='k', s=20, marker='x')


    plt.title('Dataset Distribution')
    plt.xlabel('Axis 1 [x, ]')
    plt.ylabel('Axis 2 [ ,y]')

    plt.show()


# -----------------------------------
# Model Plot Viewer
# -----------------------------------
def plot_model_viewer_q04(net_arc, net_func, w, b, text):
    """
    Function to show in a 3D scatter plot the model classification
    """

    dti = np.zeros(shape=(10101,5))

    for x in range(101):
        for y in range(101):
            dti[x*100 + y,0:2] = [(-1.25+(0.025*x)),(-1.25+(0.025*y))]

    dti[:,2] = np.array([np.add(np.abs(dti[:, 0]), np.abs(dti[:, 1]))])
    dti[:,3] = np.array([np.prod(dti[:, 0:2], axis=1)])
    dti[:,4] = np.sqrt(np.add(np.power(dti[:, 0], 2), np.power(dti[:, 1], 2)))

    values = np.array([forward(net_arc, net_func, w, b, dti[x]) for x in range(len(dti))])
    classes = np.argmax(values, axis=1)
    nc = ['C'+str(classes[x]) for x in range(len(classes))]

    # Plotting
    fig = plt.figure()
    plt.scatter(dti[:,0], dti[:,1], c=nc, s=20, marker='o')

    # title and axis labels
    plt.title('Dataset Model Viewer'+str(text))
    plt.xlabel('Input A [X, ]')
    plt.ylabel('Input B [ ,Y]')

    plt.show()
