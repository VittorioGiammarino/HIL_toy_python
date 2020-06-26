#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:53:43 2020

@author: Vittorio Giammarino 
"""
import numpy as np
import matplotlib.pyplot as plt
import Environment as env
import StateSpace as ss
import DynamicProgramming as dp

# =============================================================================
# %% map generation 
# =============================================================================

map = env.Generate_world_subgoals_simplified()

# %% Generate State Space

print('Generate State Space')

stateSpace = np.empty((0,3),int)

for m in range(0,map.shape[0]):
    for n in range(0,map.shape[1]):
        if map[m,n] != ss.TREE:
            stateSpace = np.append(stateSpace, [[m, n, 0], [m, n, 1]], 0)
            
K = stateSpace.shape[0];
TERMINAL_STATE_INDEX = ss.TerminalStateIndex(stateSpace,map)
P = dp.ComputeTransitionProbabilityMatrix(stateSpace,map)

