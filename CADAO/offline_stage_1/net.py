import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from offline_stage_1.trajectory_gpt2 import GPT2Model


class GPTEncoder(nn.Module):
    def __init__(
            self,
            conf,
            obs_dim,
            act_dim,
            hidden_size,
            max_ep_len=1000,
            **kwargs
    ):
        super().__init__()
        self.conf = conf
        self.K = conf['K']
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
        self.gpt_encoder = GPT2Model(config)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_obs = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size),
            nn.ELU(),
        )
        self.embed_action = nn.Sequential(
            nn.Linear(self.act_dim, hidden_size),
            nn.ELU(),
        )
        self.embed_rew = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ELU(),
        )
        self.embed_fusion = nn.Linear(3 * hidden_size, hidden_size)
        self.embed_ln = nn.LayerNorm(3 * hidden_size)
    
    def embed_input(self, obs, action, reward, timestep):
        obs_embeddings = self.embed_obs(obs)
        action_embeddings = self.embed_action(action)
        rew_embeddings = self.embed_rew(reward)
        time_embeddings = self.embed_timestep(timestep)
        timestep_m1 = torch.where(timestep > 0, timestep - 1, timestep)
        time_m1_embeddings = self.embed_timestep(timestep_m1)
        
        obs_embeddings = obs_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_m1_embeddings
        rew_embeddings = rew_embeddings + time_embeddings
        
        h = torch.cat((action_embeddings, rew_embeddings, obs_embeddings), dim=-1)
        h = self.embed_ln(h)
        
        return h
    
    def forward(self, obs, action, reward, timestep, attention_mask=None,):
        
        batch_size, seq_length = obs.shape[0], obs.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=obs.device)
        
        h = self.embed_input(obs, action, reward, timestep)
        stacked_inputs = self.embed_fusion(h)
        
        encoder_outputs = self.gpt_encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=attention_mask,
        )
        
        token_embeddings = encoder_outputs['last_hidden_state']
        return token_embeddings
    
    def get_tokens(self, obs, action, reward, timestep, attention_mask=None):
        
        if obs.dim() == 2:
            obs = obs.reshape(1, -1, self.obs_dim)
            action = action.reshape(1, -1, self.act_dim)
            reward = reward.reshape(1, -1, 1)
            timestep = timestep.reshape(1, -1)
        batch_size, seq_length = obs.shape[0], obs.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=obs.device)
        
        if seq_length < self.K:
            obs = torch.cat([torch.zeros((batch_size, self.K - seq_length, self.obs_dim), device=obs.device), obs], dim=1)
            action = torch.cat([torch.ones((batch_size, self.K - seq_length, self.act_dim), device=obs.device) * -10., action], dim=1)
            reward = torch.cat([torch.zeros((batch_size, self.K - seq_length, 1), device=obs.device), reward], dim=1)
            timestep = torch.cat([torch.zeros((batch_size, self.K - seq_length), device=obs.device), timestep], dim=1).to(torch.long)
            attention_mask = torch.cat([torch.zeros((batch_size, self.K - seq_length), dtype=torch.long, device=obs.device), attention_mask], dim=1)
            seq_length = self.K

        h = self.embed_input(obs, action, reward, timestep)
        
        stacked_inputs = self.embed_fusion(h)
        
        encoder_outputs = self.gpt_encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=attention_mask,
        )
        
        token_embeddings = encoder_outputs['last_hidden_state']
        
        return token_embeddings, attention_mask
    
    def load_model(self, param_path, device="cpu"):
        self.load_state_dict(
            torch.load(param_path, map_location=torch.device(device),weights_only=True)
        )


class MLPDecoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim, act_dim, use_tanh):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(obs_dim + latent_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, act_dim)
        if use_tanh:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None

    def forward(self, x, e):
        h0 = torch.cat([x, e], dim=-1)
        h1 = self.ln1(F.relu(self.fc1(h0)))
        if self.tanh is not None:
            return self.tanh(self.fc2(h1))
        else:
            return self.fc2(h1)
    
    def load_model(self, param_path, device="cpu"):
        self.load_state_dict(
            torch.load(param_path, map_location=torch.device(device))
        )
