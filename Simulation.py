#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:10:39 2020

@author: vittorio
"""
import numpy as np

def SampleTrajMDP(P,u,max_epoch,nTraj,initial_state,terminal_state):
    
    traj = [[None]*1 for _ in range(nTraj)]
    control = [[None]*1 for _ in range(nTraj)]
    flag = np.empty((0,0),int)
    
    for t in range(0,nTraj):
        
        x = np.empty((0,0),int)
        x = np.append(x, initial_state)
        u_tot = np.empty((0,0))
        
        for k in range(0,max_epoch):
            x_k_possible=np.where(P[x[k],:,int(u[x[k]])]!=0)
            prob = P[x[k],x_k_possible[0][:],int(u[x[k]])]
            prob_rescaled = np.divide(prob,np.amin(prob))
            
            for i in range(1,prob_rescaled.shape[0]):
                prob_rescaled[i]=prob_rescaled[i]+prob_rescaled[i-1]
            draw=np.divide(np.random.rand(),np.amin(prob))
            index_x_plus1=np.amin(np.where(draw<prob_rescaled))
            x = np.append(x, x_k_possible[0][index_x_plus1])
            u_tot = np.append(u_tot,u[x[k]])
            
            if x[k+1]==terminal_state:
                u_tot = np.append(u_tot,u[terminal_state])
                break
        
        traj[t][:] = x
        control[t][:]=u_tot
        
        if x[-1]==terminal_state:
            success = 1
        else:
            success = 0
                
        flag = np.append(flag,success)
        
    return traj, control, flag
        
        
                
            
            