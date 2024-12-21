import numpy as np
import torch
from torch import nn
import time
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from offline_stage_1.utils import LOG
from offline_stage_1.net import MLPDecoder


class PolicyEmbeddingTrainer:
    def __init__(self,
                encoder,
                batch_size,
                encoder_optimizer,
                encoder_scheduler,
                get_batch_fn,
                loss_gen_fn,
                config,
            ):
        self.encoder = encoder
        self.batch_size = batch_size
        self.encoder_optimizer = encoder_optimizer
        self.encoder_scheduler = encoder_scheduler
        
        self.get_batch_fn = get_batch_fn
        self.loss_gen_fn = loss_gen_fn
        
        self.config = config
        self.init_decoder()
        
        self.avg_pooling = nn.AvgPool1d(kernel_size=self.config["NUM_STEPS"])
        self.temperature = self.config["TEMPERATURE"]
        self.base_temperature = self.config["BASE_TEMPERATURE"]
        self.ALPHA = self.config["ALPHA"]
        self.LAMBDA = self.config["LAMBDA"]
        self.clip_grad = self.config["CLIP_GRAD"]
        
        self.diagnostics = dict()

        self.start_time = time.time()
    
    def init_decoder(self,):
        env_type = self.config["ENV_TYPE"]
        obs_dim = self.config["OBS_DIM"]
        hidden_dim = self.config["HIDDEN_DIM"]
        act_dim = self.config["ACT_DIM"]
        
        if env_type == "PA":
            use_tanh = False
        elif env_type == "MS":
            use_tanh = False
        
        decoder = MLPDecoder(
            obs_dim=obs_dim,
            latent_dim=hidden_dim,
            hidden_dim=hidden_dim,
            act_dim=act_dim,
            use_tanh=use_tanh,
        )
        self.decoder = decoder.to(self.config["DEVICE"])
        self.decoder_optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.config['LEARNING_RATE'],
            weight_decay=self.config['WEIGHT_DECAY'],
        )
        self.decoder_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.decoder_optimizer,
            lambda steps: min((steps + 1) / self.config['WARMUP_STEPS'], 1)
        )
    
    def train(self, num_update):
        train_losses = []
        logs = dict()

        train_start = time.time()

        self.encoder.train()
        self.decoder.train()
        for _ in range(num_update):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            self.encoder_scheduler.step()
            self.decoder_scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['training/total_loss_mean'] = np.mean(train_losses)
        logs['training/total_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]
            LOG.info(f'{k}: {logs[k]}')

        return logs
    
    def train_step(self,):
        
        batch = self.get_batch_fn()
        n_o, a, r, label, timesteps, mask, o_gen, a_gen, mask_gen = batch
        
        hidden_states = self.encoder(
            obs=n_o,
            action=a,
            reward=r,
            timestep=timesteps,
            attention_mask=mask,
        )
        
        hidden_states = self.avg_pooling(hidden_states.permute([0, 2, 1])).squeeze(-1)
        
        # * --------  calculate generative loss  -------- *
        obs_dim = o_gen.shape[-1]
        latent_dim = hidden_states.shape[-1]
        act_dim = a_gen.shape[-1]
        
        latent_gen = torch.unsqueeze(hidden_states, 1).repeat(1, o_gen.shape[1], 1).reshape(-1, latent_dim)
        o_gen = o_gen.reshape(-1, obs_dim)
        a_targ = a_gen.detach().clone().reshape(-1, act_dim)
        
        a_pred = self.decoder.forward(
            o_gen, latent_gen
        )
        a_pred = a_pred[mask_gen.reshape(-1) > 0]
        a_targ = a_targ[mask_gen.reshape(-1) > 0]
        
        loss_gen = self.loss_gen_fn(a_pred, a_targ)
        
        # * --------  calculate discrimitive loss  -------- *
        dis_features = hidden_states
        label = label.contiguous().view(-1, 1)
        if label.shape[0] != self.batch_size:
            raise ValueError('Num of label does not match num of features')
        dis_mask = torch.eq(label, label.T).float().to(dis_features.device)

        anchor_dot_contrast = torch.div(
            torch.matmul(dis_features, dis_features.T),
            self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(dis_mask),
            1,
            torch.arange(self.batch_size).view(-1, 1).to(dis_mask.device),
            0
        )
        dis_mask = dis_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (dis_mask * log_prob).sum(1) / dis_mask.sum(1)

        loss_dis = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss_dis = loss_dis.mean()
        
        if self.ALPHA <= 1e-4:
            loss = loss_dis
        elif self.LAMBDA <= 1e-4:
            loss = loss_gen
        else:
            loss = self.ALPHA * loss_gen + self.LAMBDA * loss_dis
        
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_grad)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip_grad)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        with torch.no_grad():
            self.diagnostics['training/gen_loss'] = loss_gen.detach().clone().cpu().item()
            self.diagnostics['training/dis_loss'] = loss_dis.detach().clone().cpu().item()
            self.diagnostics['training/total_loss'] = loss.detach().clone().cpu().item()
        
        return loss.detach().cpu().item()
    
    def save_model(self, postfix, save_dir):
        encoder_model_name = '/pel_encoder' + postfix
        decoder_model_name = '/pel_decoder' + postfix
        torch.save(self.encoder.state_dict(), save_dir+encoder_model_name)
        LOG.info(f'PEL-Encoder saved to: {save_dir+encoder_model_name}')
        torch.save(self.decoder.state_dict(), save_dir+decoder_model_name)
        LOG.info(f'PEL-Decoder saved to: {save_dir+decoder_model_name}')