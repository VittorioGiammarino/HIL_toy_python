#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:53:43 2020

@author: Vittorio Giammarino 
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import Environment as env
import StateSpace as ss
import DynamicProgramming as dp
import Simulation as sim

# %% map generation 
map = env.Generate_world_subgoals_simplified()

# %% Generate State Space
stateSpace=ss.GenerateStateSpace(map)            
K = stateSpace.shape[0];
TERMINAL_STATE_INDEX = ss.TerminalStateIndex(stateSpace,map)
P = dp.ComputeTransitionProbabilityMatrix(stateSpace,map)
G = dp.ComputeStageCosts(stateSpace,map)
[J_opt_vi,u_opt_ind_vi] = dp.ValueIteration(P,G,TERMINAL_STATE_INDEX)

# %% Plot Optimal Solution
env.PlotOptimalSolution(map,stateSpace,u_opt_ind_vi)

# %% Generate Expert's trajectories
T=6000
base=ss.BaseStateIndex(stateSpace,map)
[traj,control,flag]=sim.SampleTrajMDP(P, u_opt_ind_vi, 1000, T, base, TERMINAL_STATE_INDEX)

# %% Simulation
#env.VideoSimulation(map,stateSpace,control[1][:],traj[1][:])

# %% Process Data

Xtr = np.empty((2,0),int)
inputs = np.empty((3,0),int)

for i in range(len(traj)):
    Xtr = np.append(Xtr, [traj[i][:], control[i][:]],axis=1)
    inputs = np.append(inputs, np.transpose(stateSpace[traj[i][:],:]), axis=1) 
    
labels = Xtr[1,:]
TrainingSet = np.transpose(inputs) 
    
# %% NN design

action_space=5

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(3,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(action_space)
    ])

tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True,
                          expand_nested=True)

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
 
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

encoded = tf.keras.utils.to_categorical(labels)

#model.fit(TrainingSet, labels, epochs=10)

model.fit(TrainingSet, encoded, epochs=10)


# %% predictions

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

deterministic_policy = np.empty(0)

predictions = probability_model.predict(stateSpace[:,:])
 
for i in range(stateSpace.shape[0]):    
    deterministic_policy = np.append(deterministic_policy, np.argmax(predictions[i,:]))
    
env.PlotOptimalSolution(map,stateSpace,deterministic_policy)

# %%

T=6000
base=ss.BaseStateIndex(stateSpace,map)
[trajNN,controlNN,flagNN]=sim.StochasticSampleTrajMDP(P, predictions, 1000, T, base, TERMINAL_STATE_INDEX)

