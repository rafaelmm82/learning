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

from assignmentfunctions import *
from dataset_q03 import *
from training_net_delta import training_net_delta
from training_net_delta_batch import training_net_delta_batch
from training_net_delta_mom import training_net_delta_mom

import numpy as np
import matplotlib.pyplot as plt


####################################
# The Main Section of Solution
####################################


# Generating the dataset
(data_in, data_out) = dataset_q03_a()

net_arc = [2, 2, 1]
net_func = ['line', 'sigm', 'sigm']
b = 1
learn_rate = 1.5
alfa = 0.5
num_epochsd = 5000
num_epochsb = 15000
num_epochsm = 10000

total_experiments = 9

# The training and validation errors vectors
tetotal_delta = np.zeros(shape=(total_experiments, num_epochsd, 1))
vetotal_delta = np.zeros(shape=(total_experiments, num_epochsd, 1))

tetotal_bacth = np.zeros(shape=(total_experiments, num_epochsb, 1))
vetotal_batch = np.zeros(shape=(total_experiments, num_epochsb, 1))

tetotal_mom = np.zeros(shape=(total_experiments, num_epochsm, 1))
vetotal_mom = np.zeros(shape=(total_experiments, num_epochsm, 1))


# DOING the job
# -------------

# for each experiment
for exp in range(total_experiments):

    # the first randomly weights
    (w, b) = weights_init(in_dimension, out_dimension)

    training_error_mse = np.zeros((num_of_epochs, 1))
    validation_error_mse = np.zeros((num_of_epochs, 1))




# Defining the architectures
wd = weights_init(net_arc)
wb = weights_init(net_arc)
wm = weights_init(net_arc)


(wdelta, epd, err_vec_delta) = training_net_delta(net_arc, net_func,
                                                  np.copy(wd), b, data_in,
                                                  data_out, learn_rate,
                                                  num_epochsd)


num_epochsd = 1
(wdelta, epd, err_vec_delta) = training_net_delta(net_arc, net_func,
                                                  np.copy(wdelta), b, data_in,
                                                  data_out, learn_rate,
                                                  num_epochsd)

(wbatch, epb, err_vec_batch) = training_net_delta_batch(net_arc, net_func,
                                                        np.copy(wb), b,
                                                        data_in, data_out,
                                                        learn_rate,
                                                        num_epochsb,
                                                        batchsize=2)
(wmom, epm, err_vec_mom) = training_net_delta_mom(net_arc, net_func,
                                                  np.copy(wm), b, data_in,
                                                  data_out, learn_rate, alfa,
                                                  num_epochsm)
# wdelta = wd
# wbatch = wb
# wmom = wm

# DELTA
data_pred_d = np.array([forward(net_arc, net_func, wdelta, b,
                                data_in[x]) for x in range(0, 4)])
# Ploting data to graphicaly understanding
plot_dataset_q03_a(data_in, data_pred_d, data_out)


# BATCH
data_pred_b = np.array([forward(net_arc, net_func, wbatch, b, data_in[x])
                        for x in range(0, 4)])
# Ploting data to graphicaly understanding
plot_dataset_q03_a(data_in, data_pred_b, data_out)


# MOM
data_pred_w = np.array([predict(forward(net_arc, net_func, wmom, b,
                                        data_in[x])) for x in range(0, 4)])
# Ploting data to graphicaly understanding
plot_dataset_q03_a(data_in, data_pred_w, data_out)




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
