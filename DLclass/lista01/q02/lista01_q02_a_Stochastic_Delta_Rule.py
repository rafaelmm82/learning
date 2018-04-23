# -*- coding: utf-8 -*-
"""
Q02 from First assignment letter (a) app 01
Backpropagation, Stochastic with Delta Rule
Class Deep Learning
UFPB

Mar, 31 2018.

Rafael Magalhães
GitHub @rafaelmm
"""

####################################
# IMPORTANT THINGS HERE
#
#
####################################


import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


# -----------------------------------
# Dataset Generator
# -----------------------------------
def dataset_generator(n_tra, n_val=0, n_tes=0):
    """
    Generates a dataset that represents the ternary message system.

    @params:
    n_tra - the number of training examples to be generated
    n_val - the number of validation examples in dataset
    n_tes - the number of test examples in dataset

    returns a tuple of NumPy arrays in the format of
    (training inputs, training targets, validation input, ...
    validation target, test input, test targets)
    """
    total = n_tra + n_val + n_tes

    # Each example needs to be a randomly vector of three positions
    # with binary (0,1) number. Also, it has additive noise
    # of radialy 0.1 decimals. The target needs to be a eigth position
    # one hot encoded vector with binary (-1 , 1) without noise.
    allset_in = np.random.randint(2, size=(total, 3))
    allset_noise = np.random.rand(total, 3) * 0.2 - 0.1
    allset_target = np.full((total, 8), -1)

    # allset_target adjust bin to one_hot_binary
    for x in range(total):
        # obtaining the position bin to dec
        p = int(''.join(str(y) for y in allset_in[x]), 2)
        allset_target[x, p] = 1

    # adding noise to dataset
    allset_in = np.add(allset_in, allset_noise)

    # scattering the dataset
    tra_in = allset_in[0:n_tra, :]
    tra_out = allset_target[0:n_tra, :]
    val_in = allset_in[n_tra:(n_tra+n_val), :]
    val_out = allset_target[n_tra:(n_tra+n_val), :]
    tes_in = allset_in[(total-n_tes):total, :]
    tes_out = allset_target[(total-n_tes):total, :]

    return (tra_in, tra_out, val_in, val_out, tes_in, tes_out)


# -----------------------------------
# Dataset Plot Viewer
# -----------------------------------
def plot_data(ti, vi, tei):
    """
    Function to show in a 3D scatter plot the distribution of the dataset
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ti[:,0], ti[:,1], ti[:,2], c='b', marker='o')
    ax.scatter(vi[:,0], vi[:,1], vi[:,2], c='r', marker='^')
    ax.scatter(tei[:,0], tei[:,1], tei[:,2], c='g', marker='x')

    # title and axis labels
    ax.set_title('Dataset distribution')
    ax.set_xlabel('Coordenada 1 [x, , ]')
    ax.set_ylabel('Coordenada 2 [ ,y, ]')
    ax.set_zlabel('Coordenada 3 [ , ,z]')

    plt.show()


# -----------------------------------
# Plot Error Viewer
# -----------------------------------
def plot_error(error_vector):
    """
    Function to show the progress of the error vector
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
        return (1 if z > 0.5 else 0)


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
        return (1 if z > 0.5 else 0)


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
def predict(output):
    """
    It's just to round prediction of the perceptron to making
    results more conforming with the real target value
    """
    y_pred = [1 if x >= 0 else -1 for x in output]

    return y_pred


# -----------------------------------
# Training Function Stochastic (shuffling all dataset on each epoch)
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
        # for example in range(num_examples):

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


####################################
# The Main Section of Solution
####################################


# Generating the dataset
(ti, to, vi, vo, tei, teo) = dataset_generator(200, 40, 40)

# Ploting data to graphicaly understanding
plot_data(ti, vi, tei)

# creating the weightings structure
net_arc = np.array([3, 15, 8])
net_func = ['line', 'ptanh', 'ptanh']
w = weights_init(net_arc)
b = 1
learn_rate = 0.2
num_epochs = 50


# testing the forward pass
forward(net_arc, net_func, w, b, ti[0])
to[0]

# training model
(w1, epstop, verro) = training_net_delta(net_arc, net_func, w, b, ti, to, learn_rate, num_epochs)

# testing the forward pass
forward(net_arc, net_func, w1, b, ti[0])
to[0]
predict(forward(net_arc, net_func, w1, b, ti[0])) - to[0]

plot_error(verro[1:50])

ypredok = [predict(forward(net_arc, net_func, w1, b, ti[x])) for x in range(50)]
ypredok = np.array(ypredok)

yin = to[0:50]
sum(sum(yin - ypredok))
#
# y_pred_val = predict(forward(w_up, b, vi[0], 'tanh'))
# print(confusion_matrix(vo[0], y_pred_val))
#
# y_pred_tes = predict(forward(w_up, b, tei[0], 'tanh'))
# print(confusion_matrix(teo[0], y_pred_tes))
#
# y_pred3 = np.array(list(predict(forward(w_up, b, vi[x], 'tanh')) for x in range(40)))
# print('Matriz de Confusão:')
# print(confusion_matrix(vo[0], y_pred3[0]))
#
 print('F1 Score:')
 print(classification_report(yin, ypredok))
 confusion_matrix(yin, ypredok)




xteste = np.array([[0, 1], [2, 3], [4, 5]])
yteste = np.array([[10, 11],[12, 13],[14, 15]])

xnovo, yteste = shuffle(xteste, yteste)
xnovo
yteste
for i in list(shuffle(range(10))):
    print(i)
