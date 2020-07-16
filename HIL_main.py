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
import HierachicalImitationLearning as hil
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
env.PlotOptimalSolution(map,stateSpace,u_opt_ind_vi, 'Expert_pickup.eps', 'Expert_dropoff.eps')

# %% Generate Expert's trajectories
T=3
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
T = TrainingSet.shape[0]
TrainingSetTermination = hil.TrainingSetTermination(TrainingSet, option_space)
TrainingSetActions, labels_reshaped = hil.TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels)
lambdas = tf.Variable(initial_value=1.*tf.ones((option_space,)), trainable=False)
eta = tf.Variable(initial_value=1., trainable=False)

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
    optimizer = keras.optimizers.Adamax(learning_rate=1e-4)
    epochs = 100 #number of iterations for the maximization step
        
    gamma_tilde_reshaped = hil.GammaTildeReshape(gamma_tilde, option_space)
    gamma_actions_false, gamma_actions_true = hil.GammaReshapeActions(T, option_space, action_space, gamma, labels_reshaped)
    gamma_reshaped_options = hil.GammaReshapeOptions(T, option_space, gamma)
    # loss_termination = hil.OptimizeNNtermination(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, T, optimizer)
    # loss_action = hil.OptimizeNNactions(epochs, TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true, T, optimizer)
    # loss_options = hil.OptimizeNNoptions(epochs, TrainingSet, NN_options, gamma_reshaped_options, T, optimizer)
    loss = hil.OptimizeLossAndRegularizerTot(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                                             TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                                             TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
                                             gamma, option_space, labels)
    
    # loss = hil.OptimizeLoss(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
    #                         TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
    #                         TrainingSet, NN_options, gamma_reshaped_options, T, optimizer)

    print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))

# %% Evaluation 
Trajs=1
base=ss.BaseStateIndex(stateSpace,map)
[trajHIL,controlHIL,OptionsHIL, 
 TerminationHIL, flagHIL]=sim.HierarchicalStochasticSampleTrajMDP(P, stateSpace, NN_options, 
                                                                  NN_actions, NN_termination, mu, 1000, 
                                                                  Trajs, base, TERMINAL_STATE_INDEX, zeta, option_space)
                                                                  
# %%
env.HILVideoSimulation(map,stateSpace,controlHIL[0][:],trajHIL[0][:],OptionsHIL[0][:],"sim_HIL.mp4")

# %%
Pi_HI = NN_options(stateSpace).numpy()    
Pi_Lo_o1 = np.argmax(NN_actions(hil.TrainingSetPiLo(stateSpace,0)).numpy(),1)
Pi_Lo_o2 = np.argmax(NN_actions(hil.TrainingSetPiLo(stateSpace,1)).numpy(),1)
Pi_Lo_o3 = np.argmax(NN_actions(hil.TrainingSetPiLo(stateSpace,2)).numpy(),1)

# %% Understanding Regularization

option_space = 3
action_space = 5
termination_space = 2

NN_options = hil.NN_options(option_space)
NN_actions = hil.NN_actions(action_space)
NN_termination = hil.NN_termination(termination_space)

ntraj = 10
N = 5
zeta = 0.1
mu = np.ones(option_space)*np.divide(1,option_space)
T = TrainingSet.shape[0]
TrainingSetTermination = hil.TrainingSetTermination(TrainingSet, option_space)
TrainingSetActions, labels_reshaped = hil.TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels)
lambdas = tf.Variable(initial_value=10.*tf.ones((option_space,)), trainable=False)
eta = tf.Variable(initial_value=100., trainable=False)
chi = tf.Variable(initial_value=0.1, trainable=False)

for n in range(N):
    print('iter', n, '/', N)

    alpha = hil.Alpha(TrainingSet, labels, option_space, termination_space, mu, zeta, NN_options, NN_actions, NN_termination)
    beta = hil.Beta(TrainingSet, labels, option_space, termination_space, zeta, NN_options, NN_actions, NN_termination)
    gamma = hil.Gamma(TrainingSet, option_space, termination_space, alpha, beta)
    gamma_tilde = hil.GammaTilde(TrainingSet, labels, beta, alpha, 
                                  NN_options, NN_actions, NN_termination, zeta, option_space, termination_space)

    optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
    epochs = 50 #number of iterations for the maximization step
    
    gamma_tilde_reshaped = hil.GammaTildeReshape(gamma_tilde, option_space)
    gamma_actions_false, gamma_actions_true = hil.GammaReshapeActions(T, option_space, action_space, gamma, labels_reshaped)
    gamma_reshaped_options = hil.GammaReshapeOptions(T, option_space, gamma)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        
        with tf.GradientTape() as tape:
            weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
            tape.watch(weights)
            # Regularization 1
            regular_loss = 0
            for i in range(option_space):
                option =kb.reshape(NN_options(TrainingSet)[:,i],(T,1))
                option_concat = kb.concatenate((option,option),1)
                log_gamma = kb.cast(kb.transpose(kb.log(gamma[i,:,:])),'float32' )
                policy_termination = NN_termination(hil.TrainingSetPiLo(TrainingSet,i))
                array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
                for j in range(T):
                    array = array.write(j,NN_actions(hil.TrainingSetPiLo(TrainingSet,i))[j,kb.cast(labels[j],'int32')])
                policy_action = array.stack()
                policy_action_reshaped = kb.reshape(policy_action,(T,1))
                policy_action_final = kb.concatenate((policy_action_reshaped,policy_action_reshaped),1)
                
                regular_loss = regular_loss -kb.sum(policy_action_final*option_concat*policy_termination*log_gamma)/T
        
            # Regularization 2
            ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            for i in range(option_space):
                ta = ta.write(i,kb.sum(-kb.sum(NN_actions(hil.TrainingSetPiLo(TrainingSet,i))*kb.log(
                                NN_actions(hil.TrainingSetPiLo(TrainingSet,i))),1)/T,0))
            responsibilities = ta.stack()
    
            values = kb.sum(lambdas*responsibilities) 
            
            # Regularization 3
            ta_op = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            ta_op = ta_op.write(0,-kb.sum(NN_options(TrainingSet)*kb.log(NN_options(TrainingSet)))/T)
            resp_options = ta_op.stack()
    
            entro_options = chi*resp_options 
            
            pi_b = NN_termination(TrainingSetTermination,training=True)
            pi_lo = NN_actions(TrainingSetActions,training=True)
            pi_hi = NN_options(TrainingSet,training=True)
            
            loss_termination = kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)
            loss_options = kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)
            loss_action = (kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)
        
            loss = -values #eta*regular_loss #-entro_options #-loss_termination - loss_action -loss_options -entro_options -values

            
        grads = tape.gradient(loss,weights)
        #optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
        optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
        #optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
        print('options loss:', float(loss))


# %%
Pi_HI = np.argmax(NN_options(stateSpace).numpy(),1)  
pi_hi = NN_options(stateSpace).numpy()
Pi_Lo_o1 = np.argmax(NN_actions(hil.TrainingSetPiLo(stateSpace,0)).numpy(),1)
Pi_Lo_o2 = np.argmax(NN_actions(hil.TrainingSetPiLo(stateSpace,1)).numpy(),1)
Pi_Lo_o3 = np.argmax(NN_actions(hil.TrainingSetPiLo(stateSpace,2)).numpy(),1)


env.PlotOptimalOptions(map,stateSpace,Pi_HI, 'Figures/pi_hi_pick_up.eps', 'Figures/pi_hi_drop_off.eps')       
env.PlotOptimalSolution(map,stateSpace,Pi_Lo_o1, 'Figures/option1_pickup_reg.eps', 'Figures/option1_dropoff_reg.eps')
env.PlotOptimalSolution(map,stateSpace,Pi_Lo_o2, 'Figures/option2_pickup_reg.eps', 'Figures/option2_dropoff_reg.eps')
env.PlotOptimalSolution(map,stateSpace,Pi_Lo_o3, 'Figures/option3_pickup_reg.eps', 'Figures/option3_dropoff_reg.eps')
    
    



