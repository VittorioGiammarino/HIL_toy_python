#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:18:14 2020

@author: vittorio
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Generate simplified world function
# =============================================================================

def Generate_world_subgoals_simplified():
    mapsize = [10, 11]
    map = np.zeros( (mapsize[0],mapsize[1]) )
    #define obstacles
    map[0:4,5]=1;
    map[6:mapsize[0],5]=1;
    
    #count trees 
    ntrees=0;
    trees = np.empty((0,2),int)
    for i in range(0,mapsize[0]):
        for j in range(0,mapsize[1]):
            if map[i,j]==1:
                trees = np.append(trees, [[j, i]], 0)
                ntrees += 1
        
    #shooters
    nshooters=1;
    shooters = np.array([3, 2])
    map[shooters[1],shooters[0]]=2
    
    #pick up
    pick_up = np.array([7, 1])
    map[pick_up[1],pick_up[0]]=3
    
    #drop off
    drop_off = np.array([1, 8])
    map[drop_off[1],drop_off[0]]=4
    
    #base
    base = np.array([mapsize[1]-2, mapsize[0]-2])
    map[base[1],base[0]]=5
        
    plt.figure()
    plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
    plt.plot([base[0], base[0], base[0]+1, base[0]+1, base[0]],
             [base[1], base[1]+1, base[1]+1, base[1], base[1]],'k-')
    plt.plot([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
             [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'k-')
    plt.plot([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
             [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'k-')
    plt.plot([shooters[0], shooters[0], shooters[0]+1, shooters[0]+1, shooters[0]],
             [shooters[1], shooters[1]+1, shooters[1]+1, shooters[1], shooters[1]],'k-')
        
    for i in range(0,ntrees):
        plt.plot([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
         [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k-')
        
    plt.fill([base[0], base[0], base[0]+1, base[0]+1, base[0]],
             [base[1], base[1]+1, base[1]+1, base[1], base[1]],'r')
    plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
             [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'y')
    plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
             [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'b')
    plt.fill([shooters[0], shooters[0], shooters[0]+1, shooters[0]+1, shooters[0]],
             [shooters[1], shooters[1]+1, shooters[1]+1, shooters[1], shooters[1]],'c')
        
    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
         [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')
        
    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'P')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'D')
    plt.text(shooters[0]+0.5, shooters[1]+0.5, 'S')
    
        
    return map
                
                
                
                
    