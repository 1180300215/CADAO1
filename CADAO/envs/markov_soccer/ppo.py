import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self,state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, device,
                    num_steps=100, batch_size=4096, num_agent=2, actor_lr=1e-4, critic_lr=1e-4, c_dim=2, gamma=0.99, 
                    num_update_per_iter=10, clip_param=0.2, max_grad_norm=5.0,
                    ):
        super(PPO, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.c_dim = c_dim
        self.num_steps = num_steps
        self.actor_net = PolicyNet(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.critic_net = ValueNet(self.state_dim, self.hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_update_per_iter = num_update_per_iter
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.buffer = [[] for _ in range(num_agent)]
        self.counter = 0
        self.training_step = 0
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        u = np.zeros(self.action_dim)
        u[action.item()] += 1
        return np.concatenate([u, np.zeros(self.c_dim)]), action.item(), action_prob[:,action.item()].item()
    
    def get_value(self, state):
        state = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()
    
    def save_params(self, path):
        save_dict = {'actor': self.actor_net.state_dict(), 'critic': self.critic_net.state_dict()}
        name = path+'.pt'
        torch.save(save_dict, name, _use_new_zipfile_serialization=False)

    def load_params(self, filename):
        save_dict = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename), map_location=self.device)
        self.actor_net.load_state_dict(save_dict['actor'])
        self.critic_net.load_state_dict(save_dict['critic'])

    def store_transition(self, transition, agent_index):
        self.buffer[agent_index-1].append(transition)
        self.counter += 1
    
    def update(self, agent_index):
        state = torch.tensor(np.array([t.state for t in self.buffer[agent_index-1]]), dtype=torch.float).to(self.device)
        action = torch.tensor(np.array([t.action for t in self.buffer[agent_index-1]]), dtype=torch.long).view(-1,1).to(self.device)
        old_action_prob = torch.tensor(np.array([t.a_prob for t in self.buffer[agent_index-1]]), dtype=torch.float).view(-1,1).to(self.device)
        reward = [t.reward for t in self.buffer[agent_index-1]]

        R = 0
        G = []
        count = 0
        for r in reward[::-1]:
            R = r + self.gamma * R
            count += 1
            G.insert(0, R)
            if count >= self.num_steps:
                R = 0
                count = 0
        G = torch.tensor(np.array(G), dtype=torch.float).to(self.device)
        for _ in range(self.num_update_per_iter):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer[agent_index-1]))), self.batch_size, False):
                s_batch = state[index]
                a_batch = action[index]
                old_action_prob_batch = old_action_prob[index]
                G_batch = G[index].view(-1,1)
                V_batch = self.critic_net(s_batch)
                delta = G_batch - V_batch
                advantage = delta.detach().clone()

                action_prob_batch = self.actor_net(s_batch).gather(1, a_batch) # new policy
                ratio = (action_prob_batch / old_action_prob_batch)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean() # Max->Min desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(G_batch, V_batch)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.training_step += 1

        del self.buffer[agent_index-1][:] # clear experience
