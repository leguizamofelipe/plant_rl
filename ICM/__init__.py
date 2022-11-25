import torch
import torch.nn as nn
import os

class ICM(nn.Module):
    def __init__(self, n_obs, n_actions, alpha=1, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        # Set up the inverse model
        self.inverse = nn.Linear(n_obs*2, 256)
        self.pi_logits = nn.Linear(256, n_actions)

        # Set up the forward model
        self.dense1 = nn.Linear(n_obs+1, 256)
        self.new_state = nn.Linear(256, n_obs)

        device = 'cpu'
        self.to(device)

        self.gpu = torch.device(device)

    def forward(self, state, new_state, action):
        # Inverse model forward pass
        inverse_in = torch.cat([state, new_state], dim=1).to(self.gpu)
        inverse = nn.functional.elu(self.inverse(inverse_in))
        pi_logits = self.pi_logits(inverse)

        # Forward model forward pass
        action = action.reshape((action.size()[0], 1))
        forward_input = torch.cat([state, action], dim=1).to(self.gpu)
        dense = nn.functional.elu(self.dense1(forward_input))
        state_ = self.new_state(dense)

        return pi_logits, state_

    def calc_loss(self, state, new_state, action):
        state = torch.tensor(state, dtype=torch.float, device=self.gpu)
        action = torch.tensor(action, dtype=torch.float, device=self.gpu)
        new_state = torch.tensor(new_state, dtype=torch.float, device=self.gpu)

        pi_logits, state = self.forward(state, new_state, action)

        # Find inverse loss
        cross_entropy = nn.CrossEntropyLoss()
        inverse_loss = (1-self.beta)*cross_entropy(pi_logits, action.to(dtype=torch.long))

        # Find forward loss

        mse = nn.MSELoss()
        forward_loss = self.beta*mse(state, new_state)

        intrinsic_reward = self.alpha*((state-new_state).pow(2)).mean(dim=1)
        return intrinsic_reward, inverse_loss, forward_loss
