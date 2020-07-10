#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:52:47 2020

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
env.PlotOptimalSolution(map,stateSpace,u_opt_ind_vi)

# %% Generate Expert's trajectories
T=1000
base=ss.BaseStateIndex(stateSpace,map)
[traj,control,flag]=sim.SampleTrajMDP(P, u_opt_ind_vi, 1000, T, base, TERMINAL_STATE_INDEX)
labels, TrainingSet = bc.ProcessData(traj,control,stateSpace)

# %% Simulation
#env.VideoSimulation(map,stateSpace,control[1][:],traj[1][:])

# %% Behavioral Cloning vs Baum-Welch algorithm
option_space = 3
action_space = 5
termination_space = 2

N=10 #Iterations
zeta = 0.1 #Failure factor
mu = np.ones(option_space)*np.divide(1,option_space) #initial option probability distribution

ntraj = [1]
average_NN1, success_percentageNN1, average_expert = bc.EvaluationNN1(map, stateSpace, P, traj, control, ntraj)
averageBW, success_percentageBW  = hil.EvaluationBW(map, stateSpace, P, traj, control, ntraj, 
                                                    action_space, option_space, termination_space, 
                                                    N, zeta, mu)

# %% plot of performance 
plt.figure()
plt.subplot(211)
plt.plot(ntraj, average_NN1,'go--', label = 'Neural Network 1')
plt.plot(ntraj, averageBW,'rs--', label = 'Hierarchical Policy')
plt.plot(ntraj, average_expert,'b', label = 'Expert')
plt.ylabel('Average steps to goal')
plt.subplot(212)
plt.plot(ntraj, success_percentageNN1,'go--', label = 'Nerual Network 1')
plt.plot(ntraj, success_percentageBW,'rs--', label = 'Hierarchical Policy')
plt.plot(ntraj, np.ones((len(ntraj))),'b', label='Expert')
plt.xlabel('Number of Trajectories')
plt.ylabel('Percentage of success')
plt.legend(loc='lower right')
plt.savefig('evaluation_BWvsBC.eps', format='eps')
plt.show()

