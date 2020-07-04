#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:36:46 2020

@author: vittorio
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb
import numpy as np
import matplotlib.pyplot as plt
import Environment as env
import StateSpace as ss
import DynamicProgramming as dp
import Simulation as sim
import BehavioralCloning as bc
import HierachicalInverseLearning as hil
import concurrent.futures

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
T=1
base=ss.BaseStateIndex(stateSpace,map)
[traj,control,flag]=sim.SampleTrajMDP(P, u_opt_ind_vi, 1000, T, base, TERMINAL_STATE_INDEX)
labels, TrainingSet = bc.ProcessData(traj,control,stateSpace)

# %% Simulation
#env.VideoSimulation(map,stateSpace,control[1][:],traj[1][:])

# %% Triple parameterization
option_space = 3
action_space = 5
termination_space = 2

NN_options = hil.NN_options(option_space)
NN_actions = hil.NN_actions(action_space)
NN_termination = hil.NN_termination(termination_space)

# %% Baum-Welch for provable HIL iteration

ntraj = 10
N = 10
zeta = 0.1
mu = np.ones(option_space)*np.divide(1,option_space)

for n in range(N):
    print('iter', n, '/', N)
    
    # Uncomment for sequential Running
    alpha = hil.Alpha(TrainingSet, labels, option_space, termination_space, mu, zeta, NN_options, NN_actions, NN_termination)
    beta = hil.Beta(TrainingSet, labels, option_space, termination_space, zeta, NN_options, NN_actions, NN_termination)
    gamma = hil.Gamma(TrainingSet, option_space, termination_space, alpha, beta)
    gamma_tilde = hil.GammaTilde(TrainingSet, labels, beta, alpha, 
                                  NN_options, NN_actions, NN_termination, zeta, option_space, termination_space)
    
    # MultiThreading Running
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     f1 = executor.submit(hil.Alpha, TrainingSet, labels, option_space, termination_space, mu, 
    #                           zeta, NN_options, NN_actions, NN_termination)
    #     f2 = executor.submit(hil.Beta, TrainingSet, labels, option_space, termination_space, zeta, 
    #                           NN_options, NN_actions, NN_termination)  
    #     alpha = f1.result()
    #     beta = f2.result()
        
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     f3 = executor.submit(hil.Gamma, TrainingSet, option_space, termination_space, alpha, beta)
    #     f4 = executor.submit(hil.GammaTilde, TrainingSet, labels, beta, alpha, 
    #                           NN_options, NN_actions, NN_termination, zeta, option_space, termination_space)  
    #     gamma = f3.result()
    #     gamma_tilde = f4.result()
        
    print('Expectation done')
    print('Starting maximization step')
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    epochs = 100 #number of iterations for the maximization step

    # Processing termination
    T = len(TrainingSet)
    TrainingSet_reshaped_termination = np.empty((int(3*(T-1)),4))
    gamma_tilde_reshaped = np.empty((int(3*(T-1)),2),dtype='float32')
    j=1
    for i in range(0,3*(T-1),3):
        TrainingSet_reshaped_termination[i,:] = np.append(TrainingSet[j,:], [[0]])
        TrainingSet_reshaped_termination[i+1,:] = np.append(TrainingSet[j,:], [[1]])
        TrainingSet_reshaped_termination[i+2,:] = np.append(TrainingSet[j,:], [[2]])
        gamma_tilde_reshaped[i:i+3,:] = gamma_tilde[:,:,j]
        j+=1

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # # Iterate over the batches of the dataset.
        # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            tape.watch(NN_termination.trainable_weights)
            pi_b = NN_termination(TrainingSet_reshaped_termination,training=True)
            loss_termination = -kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_termination, NN_termination.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, NN_termination.trainable_weights))
        print('termination loss:', float(loss_termination))

    TrainingSet_reshaped_actions = np.empty((int(3*(T)),4))
    labels_reshaped = np.empty((int(3*(T)),1))
    gamma_reshaped = np.empty((int(3*(T)),2),dtype='float32')
    pi_lo_reshaped = np.empty((int(3*(T)),2),dtype='float32')
    j=0
    for i in range(0,3*(T),3):
        TrainingSet_reshaped_actions[i,:] = np.append(TrainingSet[j,:], [[0]])
        TrainingSet_reshaped_actions[i+1,:] = np.append(TrainingSet[j,:], [[1]])
        TrainingSet_reshaped_actions[i+2,:] = np.append(TrainingSet[j,:], [[2]])
        labels_reshaped[i,:] = labels[j]
        labels_reshaped[i+1,:] = labels[j]
        labels_reshaped[i+2,:] = labels[j]
        gamma_reshaped[i:i+3,:] = gamma[:,:,j]
        j+=1
    
    gamma_actions_false = np.empty((int(3*T),action_space))
    for i in range(3*T):
        for j in range(action_space):
            if int(labels_reshaped[i])==j:
                gamma_actions_false[i,j]=gamma_reshaped[i,0]
            else:
                gamma_actions_false[i,j] = 0
            
    gamma_actions_true = np.empty((int(3*T),action_space))
    for i in range(3*T):
        for j in range(action_space):
            if int(labels_reshaped[i])==j:
                gamma_actions_true[i,j]=gamma_reshaped[i,1]
            else:
                gamma_actions_true[i,j] = 0           

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # # Iterate over the batches of the dataset.
        # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            tape.watch(NN_actions.trainable_weights)
            pi_lo = NN_actions(TrainingSet_reshaped_actions,training=True)
            loss_action = -(kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_action, NN_actions.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, NN_actions.trainable_weights))
        print('action loss:', float(loss_action))
         

    gamma_reshaped_options = np.empty((T,option_space),dtype='float32')
    for i in range(T):
        gamma_reshaped_options[i,:] = gamma[:,1,i]
    

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # # Iterate over the batches of the dataset.
        # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            tape.watch(NN_options.trainable_weights)
            pi_hi = NN_options(TrainingSet,training=True)
            loss_options = -kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_options, NN_options.trainable_weights)
        
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, NN_options.trainable_weights))
        print('options loss:', float(loss_options))
        
    print('Maximization done, Total Loss:',float(loss_options+loss_action+loss_termination) )


        

          

    
    
    
    
# %%
# Select optimizer method
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
epochs = 500 #number of iterations for the maximization step

# Processing termination
T = len(TrainingSet)
TrainingSet_reshaped_termination = np.empty((int(3*(T-1)),4))
gamma_tilde_reshaped = np.empty((int(3*(T-1)),2),dtype='float32')
j=1
for i in range(0,3*(T-1),3):
    TrainingSet_reshaped_termination[i,:] = np.append(TrainingSet[j,:], [[0]])
    TrainingSet_reshaped_termination[i+1,:] = np.append(TrainingSet[j,:], [[1]])
    TrainingSet_reshaped_termination[i+2,:] = np.append(TrainingSet[j,:], [[2]])
    gamma_tilde_reshaped[i:i+3,:] = gamma_tilde[:,:,j]
    j+=1

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # # Iterate over the batches of the dataset.
    # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables autodifferentiation.
    with tf.GradientTape() as tape:
        tape.watch(NN_termination.trainable_weights)
        pi_b = NN_termination(TrainingSet_reshaped_termination,training=True)
        loss = -kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss, NN_termination.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, NN_termination.trainable_weights))
    print('termination loss:', float(loss))


# %%

TrainingSet_reshaped_actions = np.empty((int(3*(T)),4))
labels_reshaped = np.empty((int(3*(T)),1))
gamma_reshaped = np.empty((int(3*(T)),2),dtype='float32')
pi_lo_reshaped = np.empty((int(3*(T)),2),dtype='float32')
j=0
for i in range(0,3*(T),3):
    TrainingSet_reshaped_actions[i,:] = np.append(TrainingSet[j,:], [[0]])
    TrainingSet_reshaped_actions[i+1,:] = np.append(TrainingSet[j,:], [[1]])
    TrainingSet_reshaped_actions[i+2,:] = np.append(TrainingSet[j,:], [[2]])
    labels_reshaped[i,:] = labels[j]
    labels_reshaped[i+1,:] = labels[j]
    labels_reshaped[i+2,:] = labels[j]
    gamma_reshaped[i:i+3,:] = gamma[:,:,j]
    j+=1
    
gamma_actions_false = np.empty((int(3*T),action_space))
for i in range(3*T):
    for j in range(action_space):
        if int(labels_reshaped[i])==j:
            gamma_actions_false[i,j]=gamma_reshaped[i,0]
        else:
            gamma_actions_false[i,j] = 0
            
gamma_actions_true = np.empty((int(3*T),action_space))
for i in range(3*T):
    for j in range(action_space):
        if int(labels_reshaped[i])==j:
            gamma_actions_true[i,j]=gamma_reshaped[i,1]
        else:
            gamma_actions_true[i,j] = 0           

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # # Iterate over the batches of the dataset.
    # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables autodifferentiation.
    with tf.GradientTape() as tape:
        tape.watch(NN_actions.trainable_weights)
        pi_lo = NN_actions(TrainingSet_reshaped_actions,training=True)
        loss = -(kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss, NN_actions.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, NN_actions.trainable_weights))
    print('action loss:', float(loss))


# for k in range(3*T):
#     pi_lo_reshaped[k,:] = [pi_lo[k,int(labels_reshaped[k])], pi_lo[k,int(labels_reshaped[k])]]
         
# %%

gamma_reshaped_options = np.empty((T,option_space),dtype='float32')
for i in range(T):
    gamma_reshaped_options[i,:] = gamma[:,1,i]
    

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # # Iterate over the batches of the dataset.
    # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables autodifferentiation.
    with tf.GradientTape() as tape:
        tape.watch(NN_options.trainable_weights)
        pi_hi = NN_options(TrainingSet,training=True)
        loss = -kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss, NN_options.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, NN_options.trainable_weights))
    print('options loss:', float(loss))
        

          

    
    



