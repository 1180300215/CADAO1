import numpy as np
import torch
from torch import nn
from collections import namedtuple
import time
from .timer import Timer
from .utils.arrays import batch_to_device, to_np, to_device, apply_dict
import copy
import os, sys
import pdb
from .utils.diffusion_utils import LOG
from ml_logger import logger
from numpy import *
from copy import deepcopy
RewardBatch = namedtuple('Batch', 'trajectories conditions returns hidden_states hidden_mask loss_masks diffuse_masks')
Batch = namedtuple('Batch', 'trajectories conditions hidden_states')

class EMA():
    '''
        empirical moving average   # 指数移动平均
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class OppoDiffusionTrainer :
    def __init__(self,
                encoder,
                decoder,
                batch_size,
                get_batch_fn,
                encoder_optimizer,
                encoder_scheduler,
                config,
            ):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.batch_size = batch_size
            self.encoder_optimizer = encoder_optimizer
            self.encoder_scheduler = encoder_scheduler


            self.get_batch_fn = get_batch_fn


            self.config = config
            self.clip_grad = self.config["CLIP_GRAD"]
            self.diagnostics = dict()
            self.horizon = self.config["horizon"]
            self.history_horizon = self.config["history_horizon"]
            self.include_returns = self.config["include_returns"]
            
            self.ema = EMA(config["ema_decay"])
            self.ema_model = deepcopy(self.decoder)
            self.update_ema_every = config["update_ema_every"]
            self.save_checkpoints = config["save_checkpoints"]

            self.step_start_ema = config["step_start_ema"]
            self.log_freq = config["log_freq"]
            self.sample_freq = config["sample_freq"]
            self.save_freq = config["save_freq"]
            self.label_freq = config["label_freq"]
            self.save_parallel = config["save_parallel"]
            self.decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config["learning_rate"])
            self.gradient_accumulate_every = config["gradient_accumulate_every"]
            self.bucket = config["bucket"]
            self.reset_parameters()
            self.step = 0

            self.avg_pooling = nn.AvgPool1d(kernel_size=self.config["NUM_STEPS"])

            self.device = config["DEVICE"]

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.decoder.state_dict())
            
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.decoder)


    def train(self, n_train_steps,save_model_dir):

        timer = Timer()
        for step in range(n_train_steps):
            self.encoder.train()
            self.decoder.train()
            for i in range(self.gradient_accumulate_every):
                loss, infos = self.train_step()
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_grad)
            self.encoder_optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip_grad)
            self.decoder_optimizer.step()
            
            self.encoder_scheduler.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save(save_dir=save_model_dir)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                logger.print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['steps'] = self.step
                metrics['loss'] = loss.detach().item()
                logger.log_metrics_summary(metrics, default_stats='mean')

            self.step += 1
        self.save(save_dir=save_model_dir)

    def train_step(self):
        batch = self.get_batch_fn()
        n_o_e, a_e, r_e, timesteps_e, mask_e, o_d, a_d, r_d_s, loss_masks, diffuse_masks = batch
        # o_d torch.Size([640, 20, 10])
        # a_d torch.Size([640, 20, 5])
        # r_d torch.Size([640, 20, 1])
        # r_d_s torch.Size([640,1])

        # o_d_length = len(o_d)   # 640
        # o_d_width = len(o_d[0])  # 20   

        hidden_states ,hidden_mask = self.encoder.get_tokens(
            obs=n_o_e, 
            action=a_e, 
            reward=r_e, 
            timestep=timesteps_e, 
            attention_mask=mask_e,
        )    #torch.Size([640, 100, 8])
        # hidden_states = self.avg_pooling(hidden_states.permute([0, 2, 1])).squeeze(-1)
        #   #torch.Size([640, 8])
        # # 别忘了再改成  torch.tensor()在后面
        # hidden_states = hidden_states.cpu().detach().numpy().reshape(640,1,-1)
        # hidden_states = tile(hidden_states,(1,20,1))   # shape 为 （640，20，8）
        hidden_mask = hidden_mask.int()

        o_d = o_d.cpu().detach().numpy()
        a_d = a_d.cpu().detach().numpy()
        returns = r_d_s.cpu().detach().numpy()

        # o_d = np.concatenate([o_d,hidden_states],axis=-1)
        # # 目前o_d 是array  别忘了改成   torch.tensor()
        if self.history_horizon > 0:
            conditions = self.get_history_conditions(o_d)
        else :
            conditions = self.get_conditions(o_d) 
        # conditions = self.get_conditions(o_d)    # 640个  {}
        trajectories = np.concatenate([a_d,o_d],axis=-1)
        trajectories = torch.tensor(trajectories)
        returns = torch.tensor(returns)

        if self.include_returns:
            batch = RewardBatch(trajectories, conditions, returns, hidden_states, hidden_mask, loss_masks, diffuse_masks)
        else:
            batch = Batch(trajectories,conditions,hidden_states)

        batch = batch_to_device(batch,device=self.device)
        loss, infos = self.decoder.loss(*batch)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        return loss, infos

    def get_history_conditions(self,observations):
        history_conditions = observations[:,:self.history_horizon+1]
        return {"x":torch.tensor(history_conditions)}
     
    def get_conditions(self,observations):
        return {0:torch.tensor(observations[:,0])}
        # batch = len(observations)
        # cond = []
        # for i in range(batch):
        #     cond.append({0:observations[i][0]})

        # return cond            
 

    def save(self,save_dir):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.decoder.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        # savepath = os.path.join(self.bucket, logger.prefix, 'checkpoint')

        # os.makedirs(savepath, exist_ok=True)
        # encoder_model_name = '/encoder_min_np' + ''
        # torch.save(self.encoder.state_dict(),savepath+encoder_model_name) 
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            encoder_model_name = '/encoder_1.2_decoder32_attn' +  f'{self.step}'
            savepath = os.path.join(save_dir, f'state_1.2_decoder32_attn{self.step}.pt')
        else:
            encoder_model_name = 'encoder_1.2'
            savepath = os.path.join(save_dir, 'state_1.2.pt')
        torch.save(self.encoder.state_dict(),save_dir+encoder_model_name)
        logger.print(f'encoder saved to {save_dir}')
        torch.save(data, savepath)   
        logger.print(f'[ utils/training ] Saved model to {save_dir}')


    def load(self):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        self.decoder.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])