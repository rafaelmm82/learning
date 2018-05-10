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
net_func = ['line', 'relu', 'sigm']
b = 1
#learn_rate = 0.05
alfa = 0.5
num_epochsd = 5000
num_epochsm = 10000

total_experiments = 9
lrv = np.array([0.3])
total_lr = len(lrv)

# The training and validation errors vectors
tetotal_delta = np.zeros(shape=(total_lr, total_experiments, num_epochsd, 1))
#vetotal_delta = np.zeros(shape=(total_experiments, num_epochsd, 1))
data_pred_d = np.zeros(shape=(total_lr, total_experiments, data_out.shape[0], 1))


tetotal_mom = np.zeros(shape=(total_experiments, num_epochsm, 1))
#vetotal_mom = np.zeros(shape=(total_experiments, num_epochsm, 1))


# DOING the job for DELTA
# -------------

# for each learning rate
for lr in range(total_lr):

    learn_rate = lrv[lr]

    # for each experiment
    for exp in range(total_experiments):

        # the first randomly weights
        wd = weights_init(net_arc)

        training_error_mse = np.zeros((num_epochsd, 1))
        validation_error_mse = np.zeros((num_epochsd, 1))

        for noe in range(num_epochsd):

            (wd, epd, ervecd) = training_net_delta(net_arc, net_func,
                                                   np.copy(wd), b, data_in,
                                                   data_out, learn_rate,
                                                   num_epochs=1)

            training_error_mse[noe, 0] = MSE(net_arc, net_func, wd, b, data_in,
                                             data_out)
            # validation_error_mse[noe, 0] = MSE(net_arc, net_func, wd, b, data_in,
            #                                    data_out)

        tetotal_delta[lr][exp] = training_error_mse
        # vetotal[exp] = validation_error_mse

        data_pred_d[lr][exp] = np.array([forward(net_arc, net_func, wd, b,
                                    data_in[x]) for x in range(0, 4)])
        # Ploting data to graphicaly understanding
        # plot_dataset_q03_a(data_in, data_pred_d[lr][exp], data_out)


for i in range(9):
    plot_dataset_q03_a(data_in, data_pred_d[0][i], data_out)

# Ploting training and validation error history
plot_error_total_6(tetotal_delta, lrv)

# Ploting the means of training and validation error
plot_the_means_1(tetotal_delta[0])

# Ploting the Model View in 2D
plot_model_viewer_xor(net_arc, net_func, wd, b)




# Params for Batch

net_arc = [2, 2, 1]
net_func = ['line', 'sigm', 'sigm']
b = 1
total_experiments = 9
num_epochsb = 15000
lrvb = np.array([0.6, 1.0, 1.5])
bvec = np.array([1, 2], dtype=int)
total_lrb = len(lrvb)
total_bs = len(bvec)

tetotal_batch = np.zeros(shape=(total_bs, total_lrb, total_experiments, num_epochsb, 1))
# vetotal_batch = np.zeros(shape=(total_experiments, num_epochsb, 1))
data_pred_b = np.zeros(shape=(total_bs, total_lrb, total_experiments,
                              data_out.shape[0], 1))


# DOING the job for BATCH
# -------------

# for each batch size
for bs in range(total_bs):

    # for each learning rate
    for lr in range(total_lrb):

        learn_rate = lrvb[lr]

        # for each experiment
        for exp in range(total_experiments):

            # the first randomly weights
            wb = weights_init(net_arc)

            training_error_mse = np.zeros((num_epochsb, 1))
            #validation_error_mse = np.zeros((num_epochsb, 1))

            for noe in range(num_epochsb):

                (wb, epb, ervecb) = training_net_delta_batch(net_arc, net_func,
                                                             np.copy(wb), b,
                                                             data_in, data_out,
                                                             learn_rate,
                                                             num_epochs=1,
                                                             batchsize=bvec[bs])

                training_error_mse[noe, 0] = MSE(net_arc, net_func, wb, b, data_in,
                                                 data_out)
                # validation_error_mse[noe, 0] = MSE(net_arc, net_func, wd, b, data_in,
                #                                    data_out)

            tetotal_batch[bs][lr][exp] = training_error_mse
            # vetotal[exp] = validation_error_mse

            data_pred_b[bs][lr][exp] = np.array([forward(net_arc, net_func, wb, b,
                                        data_in[x]) for x in range(0, 4)])
            # Ploting data to graphicaly understanding
            # plot_dataset_q03_a(data_in, data_pred_d[lr][exp], data_out)


for i in range(9):
    plot_dataset_q03_a(data_in, data_pred_b[1][2][i], data_out)

# Ploting training and validation error history
plot_error_total_6_batch(tetotal_batch, lrvb, bvec)

# Ploting the means of training and validation error
plot_the_means_1(tetotal_batch[0][0])

# Ploting the Model View in 2D
plot_model_viewer_xor(net_arc, net_func, wb, b)













(data_in, data_out) = dataset_q03_a()

net_arc = [2, 2, 1]
net_func = ['line', 'sigm', 'sigm']
b = 1
learn_rate = 0.6
#alfa = 0.5
num_epochsm = 10000

total_experiments = 9
lrv = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
total_lr = len(lrv)

# The training and validation errors vectors
tetotal_mom = np.zeros(shape=(total_lr, total_experiments, num_epochsm, 1))
#vetotal_delta = np.zeros(shape=(total_experiments, num_epochsd, 1))
data_pred_m = np.zeros(shape=(total_lr, total_experiments, data_out.shape[0], 1))


#tetotal_mom = np.zeros(shape=(total_experiments, num_epochsm, 1))
#vetotal_mom = np.zeros(shape=(total_experiments, num_epochsm, 1))


# DOING the job for MOMENTUM
# -------------

# for each learning rate
for lr in range(total_lr):

    alfa = lrv[lr]

    # for each experiment
    for exp in range(total_experiments):

        # the first randomly weights
        wm = weights_init(net_arc)

        training_error_mse = np.zeros((num_epochsm, 1))
        #validation_error_mse = np.zeros((num_epochsm, 1))

        for noe in range(num_epochsm):

            (wm, epm, err_vecm) = training_net_delta_mom(net_arc, net_func,
                                                  np.copy(wm), b, data_in,
                                                  data_out, learn_rate, alfa,
                                                  num_epochs=1)

            training_error_mse[noe, 0] = MSE(net_arc, net_func, wm, b, data_in,
                                             data_out)
            # validation_error_mse[noe, 0] = MSE(net_arc, net_func, wd, b, data_in,
            #                                    data_out)

        tetotal_mom[lr][exp] = training_error_mse
        # vetotal[exp] = validation_error_mse
        print('alfa= '+str(alfa)+'  exp= '+str(exp)+'  mse='+str(tetotal_mom[lr][exp][noe]))

        data_pred_m[lr][exp] = np.array([forward(net_arc, net_func, wm, b,
                                    data_in[x]) for x in range(0, 4)])
        # Ploting data to graphicaly understanding
        # plot_dataset_q03_a(data_in, data_pred_d[lr][exp], data_out)


for i in range(9):
    plot_dataset_q03_a(data_in, data_pred_m[5][i], data_out)

# Ploting training and validation error history
plot_error_total_6_mom(tetotal_mom, lrv)

# Ploting the means of training and validation error
plot_the_means_1(tetotal_mom[5])

# Ploting the Model View in 2D
plot_model_viewer_xor(net_arc, net_func, wm, b)



