import sys
sys.path.append("../../src")
import os
import datatime
import pandas as pd
import numpy as np
from example_pendulum import get_pendulum_data
from sindy_utils import library_size
from training import train_network,create_feed_dictionary
from autoencoder import full_network
import tensorflow as tf

import pickle

def main():

    with open("result/save{}.pickle".format(sys.argv[1]),'rb') as file:
        output = pickle.load(file)
    params = output[1]
    autoencoder = full_network(params)
