import os, sys
from pprint import PrettyPrinter
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import time
from offline_stage_1.net import GPTEncoder
from stage2_test_online.nn_trainer import OppoDiffusionTrainer
from stage2_test_online.utils.diffusion_utils import (
    load_agent_oppo_data,
    # get_batch,
    cal_agent_oppo_obs_mean,
    LOG,
)
import pickle
from stage2_test_online.online_finetune.utils import get_batch_mix_revise_new
from ml_logger import logger
from stage2_test_online.diffuser_models.diffusion import GaussianInvDynDiffusion
from stage2_test_online.diffuser_models.temporal import TemporalUnet
from stage2_test_online.config import Config,get_config_dict
import torch
import numpy as np
import wandb
if Config.RUN_OFFLINE:
    os.environ["WANDB_MODE"] = "offline"

def main():

    env_type = Config.ENV_TYPE
    agent_obs_dim = Config.AGENT_OBS_DIM
    oppo_obs_dim = Config.OPPO_OBS_DIM
    act_dim = Config.ACT_DIM
    num_steps = Config.NUM_STEPS
    obs_normalize = Config.OBS_NORMALIZE
    average_total_obs = Config.AVERAGE_TOTAL_OBS
    
    exp_id = Config.EXP_ID
    log_to_wandb = Config.WANDB
    encoder_path = Config.ENCODER_PARAM_PATH
    device = Config.DEVICE

    batch_size = Config.BATCH_SIZE
    learning_rate = Config.LEARNING_RATE
    warmup_steps = Config.WARMUP_STEPS
    weight_decay = Config.WEIGHT_DECAY
    
    hidden_dim = Config.HIDDEN_DIM
    dropout = Config.DROPOUT
    num_layer = Config.NUM_LAYER
    num_head = Config.NUM_HEAD
    activation_func = Config.ACTIVATION_FUNC

    agent_index = Config.AGENT_INDEX
    oppo_index = Config.OPPO_INDEX
    
    data_path = Config.OFFLINE_DATA_PATH
    save_model_dir = Config.MODEL_DIR
    
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir, exist_ok=False)
    
    seen_oppo_policy = Config.SEEN_OPPO_POLICY
    LOG.info(f'Seen opponent policy list: {seen_oppo_policy}')
    unseen_oppo_policy = Config.UNSEEN_OPPO_POLICY
    LOG.info(f'Unseen opponent policy list: {unseen_oppo_policy}')
    
    CONFIG_DICT = get_config_dict()
    seed = CONFIG_DICT["SEED_RES"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
    offline_data = load_agent_oppo_data(data_path, agent_index, oppo_index, act_dim, CONFIG_DICT)  #  原始数据，待修改

    LOG.info("Finish loading offline dataset.")
#待修改收集    
    with open(('***.pkl'), 'rb') as f:
        online_data = pickle.load(f)
    LOG.info("Finish loading online dataset.")

    if obs_normalize:
        agent_obs_mean_list, agent_obs_std_list, oppo_obs_mean_list, oppo_obs_std_list,agent_obs_max_list, agent_obs_min_list = cal_agent_oppo_obs_mean(offline_data, total=average_total_obs)
        CONFIG_DICT["AGENT_OBS_MEAN"] = agent_obs_mean_list
        CONFIG_DICT["AGENT_OBS_STD"] = agent_obs_std_list
        CONFIG_DICT["OPPO_OBS_MEAN"] = oppo_obs_mean_list
        CONFIG_DICT["OPPO_OBS_STD"] = oppo_obs_std_list
        CONFIG_DICT["AGENT_OBS_MAX"] = agent_obs_max_list
        CONFIG_DICT["AGENT_OBS_MIN"] = agent_obs_min_list
    
    exp_prefix = env_type
    num_oppo_policy = CONFIG_DICT["NUM_OPPO_POLICY"]
    group_name = f'{exp_prefix}-{num_oppo_policy}oppo'
    curtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_prefix = f'{group_name}-{exp_id}-{curtime}'
    LOG.info("--------------------- EXP INFO ---------------------")
    PrettyPrinter().pprint(CONFIG_DICT)
    LOG.info("----------------------------------------------------")
    
    encoder = GPTEncoder(
        conf=CONFIG_DICT,
        obs_dim=oppo_obs_dim,
        act_dim=act_dim,
        hidden_size=hidden_dim,
        max_ep_len=(num_steps+12),
        activation_function=activation_func,
        n_layer=num_layer,
        n_head=num_head,
        n_inner=4 * hidden_dim,
        n_positions=1024,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
        add_cross_attention=False,
    )
    encoder = encoder.to(device=device)
    encoder.load_model(encoder_path,device=device)

    encoder_optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    encoder_scheduler = torch.optim.lr_scheduler.LambdaLR(
        encoder_optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    
    tep_model = TemporalUnet(
        horizon=Config.horizon+Config.history_horizon,   
        transition_dim=Config.observation_dim,    
        cond_dim=Config.observation_dim, 
        dim=Config.dim,  
        dim_mults=Config.dim_mults,  
        returns_condition=Config.returns_condition,  
        transformer_depth=Config.transformer_depth,
        attn_dim=Config.HIDDEN_DIM,
        num_heads_channels=Config.num_heads_channels,
        condition_dropout=Config.condition_dropout, 
        calc_energy=Config.calc_energy,  
    )
    tep_model.to(device=device)



    decoder = GaussianInvDynDiffusion(
        model=tep_model,
        horizon=Config.horizon,  
        history_horizon = Config.history_horizon,
        observation_dim=Config.observation_dim, 
        action_dim=Config.action_dim, 
        n_timesteps=Config.n_diffusion_steps,  
        loss_type=Config.loss_type,  
        clip_denoised=Config.clip_denoised,  
        predict_epsilon=Config.predict_epsilon,  
        hidden_dim=Config.hidden_dim,  
        ## loss weighting
        action_weight=Config.action_weight, 
        loss_discount=Config.loss_discount,   
        loss_weights=Config.loss_weights,  
        returns_condition=Config.returns_condition,   
        condition_guidance_w=Config.condition_guidance_w,   
        ar_inv=Config.ar_inv,  
        train_only_inv=Config.train_only_inv, 
    )
    decoder.to(device=device)

    trainer = OppoDiffusionTrainer(     
        encoder=encoder,
        decoder=decoder,
        batch_size=batch_size,
        get_batch_fn=get_batch_mix_revise_new(offline_data,online_data,CONFIG_DICT),
        encoder_optimizer=encoder_optimizer,
        encoder_scheduler=encoder_scheduler,
        config=CONFIG_DICT,
    )


    
    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project=CONFIG_DICT["PROJECT_NAME"],
            config=CONFIG_DICT,
        )
        save_model_dir += wandb.run.name
        os.mkdir(save_model_dir)
    
    LOG.info("Start diffusion training.")
    n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)

    for i in range(n_epochs):
        logger.print(f'Epoch {i} / {n_epochs} | {logger.prefix}')
        trainer.train(n_train_steps=Config.n_steps_per_epoch,save_model_dir=save_model_dir)
    LOG.info(f"Finish diffusion policy.")


if __name__ == '__main__':
    main()