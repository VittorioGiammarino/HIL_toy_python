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
T = TrainingSet.shape[0]
TrainingSetTermination = hil.TrainingSetTermination(TrainingSet, option_space)
TrainingSetActions, labels_reshaped = hil.TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels)

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
        
    gamma_tilde_reshaped = hil.GammaTildeReshape(gamma_tilde, option_space)
    gamma_actions_false, gamma_actions_true = hil.GammaReshapeActions(T, option_space, action_space, gamma, labels_reshaped)
    gamma_reshaped_options = hil.GammaReshapeOptions(T, option_space, gamma)
    loss_termination = hil.OptimizeNNtermination(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, T, optimizer)
    loss_action = hil.OptimizeNNactions(epochs, TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true, T, optimizer)
    loss_options = hil.OptimizeNNoptions(epochs, TrainingSet, NN_options, gamma_reshaped_options, T, optimizer)

    print('Maximization done, Total Loss:',float(loss_options+loss_action+loss_termination))

# %% Evaluation 

T=1
base=ss.BaseStateIndex(stateSpace,map)
[trajHIL,controlHIL,OptionsHIL, 
 TerminationHIL, flagHIL]=sim.HierarchicalStochasticSampleTrajMDP(P, stateSpace, NN_options, 
                                                                  NN_actions, NN_termination, mu, 1000, 
                                                                  T, base, TERMINAL_STATE_INDEX, zeta, option_space)
# %%
env.VideoSimulation(map,stateSpace,controlHIL[0][:],trajHIL[0][:],"sim_HIL.mp4")
          

    
    
    
    


          

    
    



