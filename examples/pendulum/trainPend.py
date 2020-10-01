import sys
sys.path.append("../../src")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import datetime
import pandas as pd
import numpy as np
from example_pendulum import get_pendulum_data
from sindy_utils import library_size
from training import train_network

import pickle

def main():
    training_data = get_pendulum_data(100)
    training_data = randomData(training_data)
    validation_data = get_pendulum_data(10)
    validation_data = randomData(validation_data)
    print(training_data['x'].shape)
    params = paramGen(training_data['x'].shape[-1],training_data['x'].shape[0])
    for num in range(10):
        params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
        params['save_name'] = 'result/pendulum_' + str(num)
        tf.compat.v1.reset_default_graph()

        results = train_network(training_data, validation_data, params)
        with open("result/save{}.pickle".format(num),'wb') as file:
            pickle.dump([results,params],file)

        print(results['sindy_coefficients']*params['coefficient_mask'])

def randomData(data):
    newData = data.copy()
    dataSize = len(data['t'])
    perm = np.random.permutation(dataSize)
    for oldI,newI in enumerate(perm):
        newData['t'][newI] = data['t'][oldI]
        for name in ['x','dx','ddx','z','dz']:
            newData[name][newI,:] = data[name][oldI,:]
    return newData

def paramGen(inSize,epSize):
    params = {}

    params['input_dim'] = inSize
    params['latent_dim'] = 1
    params['model_order'] = 2
    params['poly_order'] = 3
    params['include_sine'] = True
    params['library_dim'] = library_size(2*params['latent_dim'], params['poly_order'], params['include_sine'], True)

    # sequential thresholding parameters
    params['sequential_thresholding'] = True
    params['coefficient_threshold'] = 0.1
    params['threshold_frequency'] = 500
    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
    params['coefficient_initialization'] = 'constant'

    # loss function weighting
    params['loss_weight_decoder'] = 1.0
    params['loss_weight_sindy_x'] = 5e-4
    params['loss_weight_sindy_z'] = 5e-5
    params['loss_weight_sindy_regularization'] = 1e-5

    params['activation'] = 'sigmoid'
    params['widths'] = [128,64,32]

    # training parameters
    params['epoch_size'] = epSize
    params['batch_size'] = 1000
    params['learning_rate'] = 1e-4

    params['data_path'] = os.getcwd() + '/'
    params['print_progress'] = False
    params['print_frequency'] = 100

    # training time cutoffs
    params['max_epochs'] = 5001
    params['refinement_epochs'] = 1001
    return params

if __name__ == '__main__':
    main()
