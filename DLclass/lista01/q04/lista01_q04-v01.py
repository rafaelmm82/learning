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
from dataset_q04 import *
from training_net_delta import training_net_delta
from training_net_delta_batch import training_net_delta_batch
from training_net_delta_mom import training_net_delta_mom

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


####################################
# The Main Section of Solution
####################################

# Generating the dataset

(ti, to, vi, vo, tei, teo) = dataset_q04(2000, 100, 500)

plot_dataset_q04(vi[:,0:2], vo)


net_arc = [5, 6, 9]
net_func = ['line', 'sigm', 'sigm']
b = 1
learn_rate = 0.15
#alfa = 0.5
total_experiments = 1
num_epochsd = 100
num_epochsb = 100
num_epochsm = 100

# The training and validation errors vectors
tetotal_delta = np.zeros(shape=(total_experiments, num_epochsd, 1))
vetotal_delta = np.zeros(shape=(total_experiments, num_epochsd, 1))



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

        if noe%10 == 0:
            plot_model_viewer_q04(net_arc, net_func, wd, b, noe)
        #print('noe='+str(noe)+' mse='+str(training_error_mse[noe]))

    tetotal_delta[exp] = training_error_mse
    vetotal_delta[exp] = validation_error_mse
    print(str(training_error_mse[noe])+' exp= '+str(exp))


# Ploting training and validation error history
plot_model_viewer_q04(net_arc, net_func, wd, b, ' 1234')


# ploting prediction

#for i in range(9):
#    plot_dataset_q03_a(data_in, data_pred_d[5][i], data_out)

plot_error(vetotal_delta[0])
plot_error(tetotal_delta[0])

plot_error_total_6(tetotal_delta, lrv)

# Ploting the means of training and validation error
plot_the_means(tetotal_delta, vetotal_delta)





values = np.array([forward(net_arc, net_func, wd, b, tei[x]) for x in range(len(tei))])
classes = np.argmax(values, axis=1)

rclasses = np.argmax(teo, axis=1)

# Confusion Matrix analysis
print('')
print('[ Confusion Matrix ]')
print('')
print(confusion_matrix(rclasses, classes))
# print(confusion_matrix(np.hstack(vo[:]), np.hstack(prediction[:])))

# F1 Score report
print('')
print('                 ---===[ F1 Score Report ]===---   ')
print(classification_report(rclasses, classes))
print('')
# print(classification_report(np.hstack(vo[:]), np.hstack(prediction[:])))














# Params for Batch

net_arc = [5, 6, 9]
net_func = ['line', 'sigm', 'sigm']
b = 1
#learn_rate = 0.15

total_experiments = 1
num_epochsb = 100
lrvb = np.array([0.15])
bvec = np.array([2000], dtype=int)
total_lrb = len(lrvb)
total_bs = len(bvec)

tetotal_batch = np.zeros(shape=(total_bs, total_lrb, total_experiments, num_epochsb, 1))
vetotal_batch = np.zeros(shape=(total_bs, total_lrb, total_experiments, num_epochsb, 1))


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
            validation_error_mse = np.zeros((num_epochsb, 1))

            for noe in range(num_epochsb):

                (wb, epb, ervecb) = training_net_delta_batch(net_arc, net_func,
                                                             np.copy(wb), b,
                                                             ti, to,
                                                             learn_rate,
                                                             num_epochs=1,
                                                             batchsize=bvec[bs])

                training_error_mse[noe, 0] = MSE(net_arc, net_func, wb, b, ti, to)
                validation_error_mse[noe, 0] = MSE(net_arc, net_func, wb, b, vi, vo)

                if noe%10 == 0:
                    print('bs='+str(bvec[bs])+' lr='+str(lrvb[lr])+' traning mse='+str(training_error_mse[noe])+'  val mse='+str(validation_error_mse[noe]))

            tetotal_batch[bs][lr][exp] = training_error_mse
            vetotal_batch[bs][lr][exp] = validation_error_mse
            #print(' traning mse='+str(training_error_mse[noe])+'  val mse='+str(validation_error_mse[noe]))


            # Ploting data to graphicaly understanding
            # plot_dataset_q03_a(data_in, data_pred_d[lr][exp], data_out)



plot_model_viewer_q04(net_arc, net_func, wb, b, ' 1234')
plot_error(tetotal_batch[0,0,0])
plot_error(vetotal_batch[0,0,0])


# Ploting training and validation error history
plot_error_total_6_batch(tetotal_batch, lrvb, bvec)

# Ploting the means of training and validation error
plot_the_means_1(tetotal_batch[0][0])

# Ploting the Model View in 2D
plot_model_viewer_xor(net_arc, net_func, wb, b)
















# MATRIZ
vi
values=0
values = np.array([forward(net_arc, net_func, wd, b, vi[x]) for x in range(len(vi))])
classes = np.argmax(values, axis=1)

rclasses = np.argmax(vo, axis=1)






# Confusion Matrix analysis
print('')
print('[ Confusion Matrix ]')
print('')
print(confusion_matrix(rclasses, classes))
# print(confusion_matrix(np.hstack(vo[:]), np.hstack(prediction[:])))

# F1 Score report
print('')
print('                 ---===[ F1 Score Report ]===---   ')
print(classification_report(rclasses, classes))
print('')
# print(classification_report(np.hstack(vo[:]), np.hstack(prediction[:])))



















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




