"""
Functions you need to edit in this script - 
    - train_naive()
    - train_dagger()
    - main()
Feel free to define more functions if required.

Usage: train.py [-h] [--mode {naive,dagger}]

optional arguments:
  -h, --help            show this help message and exit
  --mode {naive,dagger}, -m {naive,dagger}
                        Sets the training mode. Default : naive
"""


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", '-m', choices = ['naive', 'dagger'], default='naive', help = "Sets the training mode. Default : naive")
args = parser.parse_args()
MODE = args.mode 

import numpy as np 
import os 
import pickle
import gzip
import matplotlib.pyplot as plt

# Load Dependencies #
import torch
import torch.nn as nn
import tensorflow as tf
import cv2
#####################

from model import ILAgent # Change this to whatever name you've given your model class
from model import preprocessing
from model import output_preprocessing

DISC_ACTION_SPACE = np.array([[-1., 0. , 0.], [-1., 0., 0.8], [-1., 1., 0.], [0., 0., 0.], [0., 0., 0.8], [0., 1., 0.], [1., 0., 0.], [1., 1., 0.]]) #Discretized action space
agent = ILAgent((1, 84, 84), len(DISC_ACTION_SPACE))

def load_data(directory = "./data", val_split = 0.1):
    """
    Loads the data saved after expert runs.
    Input : directory where data.pkl.gzip is located, val_split
    Output : X_train, Y_train, X_val, Y_val (training and validations sets with split determined by `val_split`)
    """
    data_file = os.path.join(directory, 'data.pkl.gzip')
    
    file =  gzip.open(data_file, 'rb')
    data = pickle.load(file)
        
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation sets
    num_samples = len(data["state"])
    val_len = int(val_split*num_samples)
    X_train, y_train = X[:-val_len], y[:-val_len]
    X_val, y_val = X[-val_len:], y[-val_len:]
    return X_train, y_train, X_val, y_val

def compute_loss(agent, inputs, outputs):
  actions = agent.get_action(inputs)
  loss = torch.mean((actions - outputs)**2)
  return loss

def train_naive(): # add arguments as needed
  batch = preprocessing(X_train)
  outputs = output_preprocessing(y_train)
  for i in range(0, num_iterations):
    loss = compute_loss(agent, batch, outputs)
    loss.backward()
    opt.step()
    opt.zero_grad()

def train_dagger(): # add arguments as needed
    """
    Define your training pipeline for naive behavioural cloning. Delete the pass statement once you're done.
    Save your trained model under "agents/dagger".
    This function should return the history of your training and validation metrics.
    """
    pass

def main():
    # Loading the data
    X_train, y_train, X_val, y_val = load_data(directory = "./data", val_split = 0.1) # Feel free to experiment with val_split

    # Apply preprocessing to observations (if any, delete the next two lines otherwise)
    X_train = preprocessing(X_train)
    X_val = preprocessing(X_val)

    # Training
    if MODE == 'naive':
        train_naive(agent, 100, X_train, y_train)
    else:
        # Call train_dagger, save its results in results/dagger.
        pass

if __name__ == "__main__":
    main()
