#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:07:10 2020

@author: vittorio
"""

import numpy as np
import StateSpace as ss


def ComputeTransitionProbabilityMatrix(stateSpace,map):
    
    action_space=5
    K = stateSpace.shape[0]
    P = np.zeros((K,K,action_space))
    BaseIndex=ss.BaseStateIndex(stateSpace,map)
    i_base = stateSpace[BaseIndex,0]
    j_base = stateSpace[BaseIndex,1]
    PickUpIndex=ss.PickUpStateIndex(stateSpace,map)
    i_pickup = stateSpace[PickUpIndex,0]
    j_pickup = stateSpace[PickUpIndex,1]
    [M,N]=map.shape
    
    for psi in range(0,2):
        for i in range(0,M):
            for j in range(0,N):
                
                array_temp = [i, j, psi]
                k=ss.FindStateIndex(stateSpace,array_temp)
                
                if map[i,j] != ss.TREE:
                    
                    if map[i,j] == ss.TerminalStateIndex(stateSpace,map):
                        dummy=0
                    else: 
                        for u in range(0,action_space):
                            comp_no=1;
                            # east case
                            if j!=N-1:
                                if u == ss.EAST and map[i,j+1]!=ss.TREE:
                                    r=i
                                    s=j+1
                                    comp_no = 0
                            elif j==N-1 and u==ss.EAST:
                                comp_no=1
                            #west case
                            if j!=0:
                                if u==ss.WEST and map[i,j-1]!=ss.TREE:
                                    r=i
                                    s=j-1
                                    comp_no=0
                            elif j==0 and u==ss.WEST:
                                comp_no=1
                            #south case
                            if i!=0:
                                if u==ss.SOUTH and map[i-1,j]!=ss.TREE:
                                    r=i-1
                                    s=j
                                    comp_no=0
                            elif i==0 and u==ss.SOUTH:
                                comp_no=1
                            #north case
                            if i!=M-1:
                                if u==ss.NORTH and map[i+1,j]!=ss.TREE:
                                    r=i+1
                                    s=j
                                    comp_no=0
                            elif i==M-1 and u==ss.NORTH:
                                comp_no=1
                            #hover case
                            if u==ss.HOVER:
                                r=i
                                s=j
                                comp_no=0
                            
                            if comp_no==0:
                                array_temp = [r, s, psi]
                                t=ss.FindStateIndex(stateSpace,array_temp)
                                
                                z=s
                                y=r
                                yy=r+1
                                zz=s+1
                                zzz=s+2
                                yyy=r+2
                                nyy=r-1
                                nyyy=r-2
                                nzz=s-1
                                nzzz=s-2
                                
                                # No wind case
                                #build the probability of being shooted or not
                                
                                p_nh=1
                                
                                if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                    if map[y,z]==ss.SHOOTER:
                                        p=ss.GAMMA
                                        p_nh=p_nh*(1-p)
                                        
                                if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                    if map[y,zz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)
                                        
                                if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                    if map[y,zzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)
                                        
                                if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                    if map[yy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)
                                        
                                if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                    if map[yyy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+2)
                                        p_nh=p_nh*(1-p)
                                        
                                if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                    if map[yy,zz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+2)
                                        p_nh=p_nh*(1-p)
                                        
                                if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                    if map[y,nzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)
                                        
                                if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                    if map[y,nzzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+2)
                                        p_nh=p_nh*(1-p)
                                        
                                if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                    if map[nyy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)
                                        
                                if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                    if map[nyyy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)
                                        
                                if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                    if map[nyy,zz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)
                                        
                                if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                    if map[nyy,nzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)
                                        
                                if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                    if map[yy,nzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)
                                        
                                array_temp = [i_pickup, j_pickup, 0]
                                pickup0=ss.FindStateIndex(stateSpace,array_temp)                                
                                array_temp = [i_pickup, j_pickup, 1]
                                pickup1=ss.FindStateIndex(stateSpace,array_temp)
                                if t == pickup0:
                                    P[k,pickup1,u]=P[k,pickup1,u]+(1-ss.P_WIND)*p_nh
                                    P[k,pickup0,u]=0
                                else:
                                    P[k,t,u] = P[k,t,u]+(1-ss.P_WIND)*p_nh
                                    
                                base0=ss.BaseStateIndex(stateSpace,map)
                                P[k,base0,u]=P[k,base0,u]+(1-ss.P_WIND)*(1-p_nh)
                                
                                # case wind
                                
                                #north wind
                                if s+1>N-1 or map[r,s+1]==ss.TREE:
                                    P[k,base0,u]=P[k,base0,u]+ss.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r, s+1, psi]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    s_nord=s+1
                                    
                                    z=s_nord
                                    y=r
                                    yy=r+1
                                    zz=s_nord+1
                                    zzz=s_nord+2
                                    yyy=r+2
                                    nzzz=s_nord-2
                                    nyy=r-1
                                    nyyy=r-2
                                    nzz=s_nord-1
                                    
                                    #build prob of being shot
                                    p_nh=1;
                                    
                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)                                    
                                    
                                    if t == pickup0:
                                        P[k,pickup1,u]=P[k,pickup1,u]+0.25*ss.P_WIND*p_nh
                                        P[k,t,u]=0
                                    else:
                                        P[k,t,u] = P[k,t,u]+0.25*ss.P_WIND*p_nh #north wind no hit
                                            
                                    P[k,base0,u]=P[k,base0,u]+0.25*ss.P_WIND*(1-p_nh) #north wind hit
                                                                           
                                #South Wind
                                if s-1<0 or map[r,s-1]==ss.TREE:
                                    P[k,base0,u]=P[k,base0,u]+ss.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r, s-1, psi]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    s_south=s-1
                                    
                                    z=s_south
                                    y=r
                                    yy=r+1
                                    zz=s_south+1
                                    zzz=s_south+2
                                    yyy=r+2
                                    nzzz=s_south-2
                                    nyy=r-1
                                    nyyy=r-2
                                    nzz=s_south-1
                                    
                                    #build prob of being shot
                                    p_nh=1;
                                    
                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)                                    
                                    
                                    if t == pickup0:
                                        P[k,pickup1,u]=P[k,pickup1,u]+0.25*ss.P_WIND*p_nh
                                        P[k,t,u]=0
                                    else:
                                        P[k,t,u] = P[k,t,u]+0.25*ss.P_WIND*p_nh #south wind no hit
                                            
                                    P[k,base0,u]=P[k,base0,u]+0.25*ss.P_WIND*(1-p_nh) #south wind hit                                
                                
                                #East Wind
                                if r+1>M-1 or map[r+1,s]==ss.TREE:
                                    P[k,base0,u]=P[k,base0,u]+ss.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r+1, s, psi]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    r_east=r+1
                                    
                                    z=s
                                    y=r_east
                                    yy=r_east+1
                                    zz=s+1
                                    zzz=s+2
                                    yyy=r_east+2
                                    nzzz=s-2
                                    nyy=r_east-1
                                    nyyy=r_east-2
                                    nzz=s-1
                                    
                                    #build prob of being shot
                                    p_nh=1;
                                    
                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)                                    
                                    
                                    if t == pickup0:
                                        P[k,pickup1,u]=P[k,pickup1,u]+0.25*ss.P_WIND*p_nh
                                        P[k,t,u]=0
                                    else:
                                        P[k,t,u] = P[k,t,u]+0.25*ss.P_WIND*p_nh #east wind no hit
                                            
                                    P[k,base0,u]=P[k,base0,u]+0.25*ss.P_WIND*(1-p_nh) #east wind hit                                     
                                                                       
                                #West Wind
                                if r-1<0 or map[r-1,s]==ss.TREE:
                                    P[k,base0,u]=P[k,base0,u]+ss.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r-1, s, psi]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    r_west=r-1
                                    
                                    z=s
                                    y=r_west
                                    yy=r_west+1
                                    zz=s+1
                                    zzz=s+2
                                    yyy=r_west+2
                                    nzzz=s-2
                                    nyy=r_west-1
                                    nyyy=r_west-2
                                    nzz=s-1
                                    
                                    #build prob of being shot
                                    p_nh=1;
                                    
                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)
                                        
                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)                                    
                                    
                                    if t == pickup0:
                                        P[k,pickup1,u]=P[k,pickup1,u]+0.25*ss.P_WIND*p_nh
                                        P[k,t,u]=0
                                    else:
                                        P[k,t,u] = P[k,t,u]+0.25*ss.P_WIND*p_nh #west wind no hit
                                            
                                    P[k,base0,u]=P[k,base0,u]+0.25*ss.P_WIND*(1-p_nh) #west wind hit                                     
                            elif comp_no == 1:
                                base0=ss.BaseStateIndex(stateSpace,map)
                                P[k,base0,u]=1
                                
    k = ss.TerminalStateIndex(stateSpace, map)

    for t in range(0,K):
        if t!=k:
            for u in range(0,action_space):
                P[k,t,u]=0
        else:
            for u in range(0,action_space):
                P[k,t,u]=1
                
    return P
    
                                    
                                        
                            
                    
                
    