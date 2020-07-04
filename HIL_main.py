#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:36:46 2020

@author: vittorio
"""

import tensorflow as tf
from tensorflow import keras
import keras.backend as kb
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
N = 1
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
# %%

model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(4,)),
    keras.layers.Dense(termination_space),
    keras.layers.Softmax()
    ])

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
   



optimizer = keras.optimizers.Adam(learning_rate=1e-3)

epochs = 100
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # # Iterate over the batches of the dataset.
    # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables autodifferentiation.
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_weights)
        pi_b = model(TrainingSet_reshaped_termination,training=True)
        loss = -kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)
        
        
        # pi_b_evaluation = hil.Pi_b_Data_LossFunction(option_space,termination_space, TrainingSet, model)
        # loss = hil.Pi_b_LossFunction(gamma_tilde[:,:,1:], pi_b_evaluation[:,:,1:])
        # logits = model(TrainingSet, training=True)
        # loss_value = loss_fn(labels, logits)



    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss, model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    print(float(loss))






# pi_b_evaluation = hil.Pi_b_Data_LossFunction(option_space,termination_space, TrainingSet, NN_termination)
# Loss_b = hil.Pi_b_LossFunction(gamma_tilde[:,:,1:], pi_b_evaluation[:,:,1:])


         
# %%

inputs = keras.Input(shape=(784,), name="digits")
x1 = keras.layers.Dense(64, activation="relu")(inputs)
x2 = keras.layers.Dense(64, activation="relu")(x1)
outputs = keras.layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

# %%

# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_train, (-1, 784))
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# %%

epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * 64))    
        

          

    
    



