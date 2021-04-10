# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:20:14 2020

@author: Abdelhamid Bouzid
"""
import torch as T
from deepSarsa import deepSarsa
import numpy as np

class Agent(object):
    def __init__(self, lr, n_actions, input_dim, gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        
        self.lr           = lr
        self.n_actions    = n_actions
        self.input_dim    = input_dim
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.eps_dec      = eps_dec
        self.eps_min      = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        
        self.model            = deepSarsa(self.lr, self.n_actions, self.input_dim)
        
    def choose_action(self, state):
        self.model.eval()
        if np.random.random()> self.epsilon:
            
            state   = T.tensor(state, dtype=T.float).to(self.model.device)
            actions = self.model(state)
            action  = T.argmax(actions).item()
            
        else:
            action = np.random.choice(self.action_space)
        self.model.train()
        return action
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min
                            
    def learn(self, state, action, reward, n_state,n_action):
        
        self.model.optimizer.zero_grad()
        
        state   = T.tensor(state, dtype=T.float).to(self.model.device)
        action  = T.tensor(action).to(self.model.device)
        reward  = T.tensor(reward).to(self.model.device)
        n_state = T.tensor(n_state, dtype=T.float).to(self.model.device)
        n_action  = T.tensor(n_action).to(self.model.device)
        
        Q_next = self.model.forward(n_state)[n_action]
        truth  = reward + self.gamma*Q_next
        
        Q_pred = self.model.forward(state)[action]
        
        loss = self.model.loss(truth, Q_pred).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()
        self.decrement_epsilon()
