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


(ti,to,vi,vo)=dataset_q03_b(40, 15)

net_arc = [1, 8, 8, 1]
net_func = ['line', 'tanh', 'tanh', 'tanh']
b = 1
learn_rate = 0.02
alfa = 0.5
total_experiments = 1
num_epochsd = 5000
num_epochsb = 100
num_epochsm = 100

# The training and validation errors vectors
tetotal_delta = np.zeros(shape=(total_experiments, num_epochsd, 1))
vetotal_delta = np.zeros(shape=(total_experiments, num_epochsd, 1))
#data_pred_d = np.zeros(shape=(total_experiments, to.shape[0], 1))


# DOING the job for DELTA
# -------------

# for each experiment
for exp in range(total_experiments):

    # the first randomly weights
    wd = weights_init(net_arc)

    training_error_mse = np.zeros((num_epochsd, 1))
    validation_error_mse = np.zeros((num_epochsd, 1))

    for noe in range(num_epochsd):

        (wd, epd, ervecd) = training_net_delta(net_arc, net_func,
                                               np.copy(wd), b, ti,
                                               to, learn_rate,
                                               num_epochs=1)

        training_error_mse[noe, 0] = MSE(net_arc, net_func, wd, b, ti, to)
        validation_error_mse[noe, 0] = MSE(net_arc, net_func, wd, b, vi, vo)
        #print('noe='+str(noe)+' mse='+str(training_error_mse[noe]))

    tetotal_delta[exp] = training_error_mse
    vetotal_delta[exp] = validation_error_mse
    print(str(training_error_mse[noe])+' exp= '+str(exp))

    #data_pred_d[exp] = np.array([forward(net_arc, net_func, wd, b, vi[x]) for x in range(0, 4)])

    # Ploting data to graphicaly understanding
    # plot_dataset_q03_a(data_in, data_pred_d[lr][exp], data_out)


# Ploting training and validation error history
plot_dataset_q03_b(ti,to, vi, vo)

data_pred_d = np.array([forward(net_arc, net_func, wd, b, vi[x]) for x in range(len(vi))])


# ploting prediction
plot_prediction_q03_b(ti, to, vi, data_pred_d)
plot_erro_validation(tetotal_delta[0,1:500], vetotal_delta[0,1:500])

#for i in range(9):
#    plot_dataset_q03_a(data_in, data_pred_d[5][i], data_out)

plot_error(vetotal_delta[0])

plot_error_total_6(tetotal_delta, lrv)

# Ploting the means of training and validation error
plot_the_means(tetotal_delta, vetotal_delta)

# Ploting the Model View in 2D
plot_model_viewer_xor(net_arc, net_func, wd, b)





(wdelta, epd, err_vec_delta) = training_net_delta(net_arc, net_func, np.copy(wd), b, data_in, data_out, learn_rate, num_epochsd)
(wbatch, epb, err_vec_batch) = training_net_delta_batch(net_arc, net_func, np.copy(wb), b, data_in, data_out, learn_rate, num_epochsb, batchsize=1)
(wmom, epm, err_vec_mom) = training_net_delta_mom(net_arc, net_func, np.copy(wm), b, data_in, data_out, learn_rate, alfa, num_epochsm)


# DELTA
#wdelta= wd

data_pred_d = np.zeros((4,1))
data_pred_d[0] = predict(forward(net_arc, net_func, wdelta, b, data_in[0]))
data_pred_d[1] = predict(forward(net_arc, net_func, wdelta, b, data_in[1]))
data_pred_d[2] = predict(forward(net_arc, net_func, wdelta, b, data_in[2]))
data_pred_d[3] = predict(forward(net_arc, net_func, wdelta, b, data_in[3]))

# Ploting data to graphicaly understanding
plot_dataset_q03_a(data_in, data_pred_d)




