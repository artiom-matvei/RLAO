# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
import numpy as np

n_channels_hidden = 4
n_layers = 1
n_filt = 64

class ConvDynamics(nn.Module):
    def __init__(self, xvalid, yvalid, n_history):
        super().__init__()
        
        self.xvalid = xvalid
        self.yvalid = yvalid

        self.n_history = n_history

        self.net = nn.Sequential(
            nn.Conv2d(n_history*2, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            
            nn.Conv2d(n_filt, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, 1, 3, padding=1)
            #nn.Tanh()
        )

        self._hidden = None

    def forward(self, states, actions, history = None):
        if states.ndim == 3:
            states = states.view(1, *states.shape)
        if actions.ndim == 3:
            actions = actions.view(1, *actions.shape)       

        if history is not None:
            if history.ndim == 3:
                history = history.view(1, 1, *actions.shape)
        
            feats = torch.cat([history, states, actions], dim=1)
        else:
            feats = torch.cat([states, actions], dim=1)   

        out = self.net(feats)

        ret = torch.zeros_like(out)
        ret[..., self.xvalid, self.yvalid] = out[..., self.xvalid, self.yvalid]
        
        return ret




class ConvPolicy(nn.Module):
    def __init__(self, xvalid, yvalid, sigma, F, n_history):
        super().__init__()
        self.xvalid = xvalid
        self.yvalid = yvalid

        self.n_history = n_history

        self.register_buffer('F', F.unsqueeze(0))

        self.net = nn.Sequential(
            nn.Conv2d(n_history, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, 1, 3, padding=1),
            #nn.Tanh()
        )

        self.sigma = sigma
    
    def weights_init(self, standart_dev = 0.1, mean_bias = 0):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean = 0, std = standart_dev)
                nn.init.constant_(module.bias, mean_bias)

    def forward(self, state, history = None):
        if state.ndim == 3:
            state = state.view(1, *state.shape)
        if state.ndim == 2:
            state = state.view(1,1, *state.shape)
        if history is not None:
            feats = torch.cat([state, history], dim=1)
        else:
            feats = state      

        out = self.net(feats)

        #torch.cuda.synchronize()
        #t = time.time()
        out = out.clamp(-1, 1)

        ret = torch.zeros_like(out)

        #print(self.xvalid.shape)

        out = out[:, :, self.xvalid, self.yvalid].squeeze(1)
        out = torch.matmul(self.F, out.unsqueeze(2))
        out = out.squeeze(-1).unsqueeze(1)
        
        
        ret[..., self.xvalid, self.yvalid] = out[..., ]
        #torch.cuda.synchronize()
        #print(time.time()-t)
        return ret




class EnsembleDynamics(nn.Module):

    def __init__(self, xvalid, yvalid, n_history, n_models=5):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([])

        for _ in range(n_models):
            self.models.append(ConvDynamics(xvalid,yvalid, n_history))

    def forward(self, states, actions, history = None):
        next_states = []
        #n_particles_per_model = n_samples // self.n_models
        
        for model in self.models:
            next_states.append(model.forward(states, actions, history))
        
        return torch.cat(next_states, dim=1)
        #return torch.mean(torch.cat(next_states, dim=1), dim = 1, keepdim = True)









