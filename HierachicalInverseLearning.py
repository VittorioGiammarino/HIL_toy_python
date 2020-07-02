#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:25:42 2020

@author: vittorio
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import StateSpace as ss
import Simulation as sim


def NN_options(option_space):
    model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(3,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(option_space)
    ])

    tf.keras.utils.plot_model(model, to_file='model_NN_options.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model

def NN_actions(action_space):
    model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(4,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(action_space)
    ])

    tf.keras.utils.plot_model(model, to_file='model_NN_actions.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model

def NN_termination(termination_space):
    model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(4,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(termination_space)
    ])

    tf.keras.utils.plot_model(model, to_file='model_NN_termination.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model

def MakePredictions(model, inputs):
    
    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

    predictions = probability_model.predict(inputs)
    
    return predictions

def Pi_hi(ot, Pi_hi_parameterization, state):
    tf.autograph.experimental.do_not_convert(
    func=MakePredictions)

    Pi_hi = MakePredictions(Pi_hi_parameterization, state)
    o_prob = Pi_hi[0,ot]
    
    return o_prob

def Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space):
    if b == True:
        o_prob_tilde = Pi_hi(ot, Pi_hi_parameterization, state)
    elif ot == ot_past:
        o_prob_tilde = 1-zeta+np.divide(zeta,option_space)
    else:
        o_prob_tilde = np.divide(zeta,option_space)
        
    return o_prob_tilde

def Pi_lo(a, Pi_lo_parameterization, state_and_option):
    Pi_lo = MakePredictions(Pi_lo_parameterization, state_and_option)
    a_prob = Pi_lo[0,int(a)]
    
    return a_prob

def Pi_b(b, Pi_b_parameterization, state_and_option):
    Pi_b = MakePredictions(Pi_b_parameterization, state_and_option)
    if b == True:
        b_prob = Pi_b[0,1]
    else:
        b_prob = Pi_b[0,0]
        
    return b_prob
    
def Pi_combined(ot, ot_past, a, b, Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, state, zeta, option_space):
    Pi_hi_eval = Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space)
    Pi_lo_eval = Pi_lo(a, Pi_lo_parameterization, np.append(state, [[ot]],axis=1))
    Pi_b_eval = Pi_b(b, Pi_b_parameterization, np.append(state, [[ot]],axis=1))
    output = Pi_hi_eval*Pi_lo_eval*Pi_b_eval
    
    return output
    
def ForwardRecursion(alpha_past, a, Pi_hi_parameterization, Pi_lo_parameterization,
                     Pi_b_parameterization, state, zeta, option_space, termination_space):
# =============================================================================
#     alpha is the forward message: alpha.shape()= [option_space, termination_space]
# =============================================================================
    alpha = np.empty((option_space, termination_space))
    for i1 in range(option_space):
        ot = i1
        for i2 in range(termination_space):
            if i2 == 1:
                bt=True
            else:
                bt=False
            
            Pi_comb = np.zeros(option_space)
            for ot_past in range(option_space):
                Pi_comb[ot_past] = Pi_combined(ot, ot_past, a, bt, 
                                               Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, 
                                               state, zeta, option_space)
            alpha[ot,i2] = np.dot(alpha_past[:,0],Pi_comb)+np.dot(alpha_past[:,1],Pi_comb)  
    alpha = np.divide(alpha,np.sum(alpha))
            
    return alpha
    
def ForwardFirstRecursion(mu, a, Pi_hi_parameterization, Pi_lo_parameterization,
                          Pi_b_parameterization, state, zeta, option_space, termination_space):
# =============================================================================
#     alpha is the forward message: alpha.shape()=[option_space, termination_space]
#   mu is the initial distribution over options: mu.shape()=[1,option_space]
# =============================================================================
    alpha = np.empty((option_space, termination_space))
    for i1 in range(option_space):
        ot = i1
        for i2 in range(termination_space):
            if i2 == 1:
                bt=True
            else:
                bt=False
            
            Pi_comb = np.zeros(option_space)
            for ot_past in range(option_space):
                Pi_comb[ot_past] = Pi_combined(ot, ot_past, a, bt, 
                                               Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, 
                                               state, zeta, option_space)
            alpha[ot,i2] = np.dot(mu, Pi_comb[:])    
    alpha = np.divide(alpha, np.sum(alpha))
            
    return alpha

def BackwardRecursion(beta_next, a, Pi_hi_parameterization, Pi_lo_parameterization,
                      Pi_b_parameterization, state, zeta, option_space, termination_space):
# =============================================================================
#     beta is the backward message: beta.shape()= [option_space, termination_space]
# =============================================================================
    beta = np.empty((option_space, termination_space))
    for i1 in range(option_space):
        ot = i1
        for i2 in range(termination_space):
            for i1_next in range(option_space):
                ot_next = i1_next
                for i2_next in range(termination_space):
                    if i2 == 1:
                        b_next=True
                    else:
                        b_next=False
                    beta[i1,i2] = beta[i1,i2] + beta_next[ot_next,i2_next]*Pi_combined(ot_next, ot, a, b_next, 
                                                                                       Pi_hi_parameterization, Pi_lo_parameterization, 
                                                                                       Pi_b_parameterization, state, zeta, option_space)
    beta = np.divide(beta,np.sum(beta))
    
    return beta

def Alpha(TrainingSet, labels, option_space, termination_space, mu, zeta, NN_options, NN_actions, NN_termination):
    alpha = np.empty((option_space,termination_space,len(TrainingSet)))
    for t in range(len(TrainingSet)):
        print('alpha iter', t+1, '/', len(TrainingSet))
        if t ==0:
            state = TrainingSet[t,:].reshape(1,len(TrainingSet[t,:]))
            action = labels[t]
            alpha[:,:,t] = ForwardFirstRecursion(mu, action, NN_options, 
                                                 NN_actions, NN_termination, 
                                                 state, zeta, option_space, termination_space)
        else:
            state = TrainingSet[t,:].reshape(1,len(TrainingSet[t,:]))
            action = labels[t]
            alpha[:,:,t] = ForwardRecursion(alpha[:,:,t-1], action, NN_options, 
                                            NN_actions, NN_termination, 
                                            state, zeta, option_space, termination_space)
           
    return alpha

def Beta(TrainingSet, labels, option_space, termination_space, zeta, NN_options, NN_actions, NN_termination):
    beta = np.empty((option_space,termination_space,len(TrainingSet)))
    beta[:,:,len(TrainingSet)-1] = np.divide(np.ones((option_space,termination_space)),2*option_space)
    
    for t_raw in range(len(TrainingSet)-1):
        t = len(TrainingSet) - (t_raw+1)
        print('beta iter', t_raw+1, '/', len(TrainingSet)-1)
        state = TrainingSet[t,:].reshape(1,len(TrainingSet[t,:]))
        action = labels[t]
        beta[:,:,t-1] = BackwardRecursion(beta[:,:,t], action, NN_options, 
                                        NN_actions, NN_termination, state, zeta, 
                                        option_space, termination_space)
        
    return beta
                    
                    

    
    
    