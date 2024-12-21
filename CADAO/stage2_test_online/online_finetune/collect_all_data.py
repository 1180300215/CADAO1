import argparse
import os, sys
import pickle
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import time
from offline_stage_1.net import GPTEncoder
from stage2_test_online.diffuser_models.diffusion import GaussianInvDynDiffusion
from stage2_test_online.diffuser_models.temporal import TemporalUnet
from stage2_test_online.config import Config,get_config_dict

from stage2_test_online.utils.diffusion_utils import (
    get_env_and_oppo,
    load_agent_oppo_data,
    cal_agent_oppo_obs_mean,
    pad_future,
    LOG,
)
from stage2_test_online.online_finetune.utils import (
    collect_all_episodes,
    load_all_data,
    LOG,
)
import torch
import numpy as np
import wandb
if Config.RUN_OFFLINE:
    os.environ["WANDB_MODE"] = "offline"
from stage2_test_online.normalizers import DatasetNormalizer

def collect_all_data():
    env_type = Config.ENV_TYPE
    agent_obs_dim = Config.AGENT_OBS_DIM
    oppo_obs_dim = Config.OPPO_OBS_DIM
    act_dim = Config.ACT_DIM
    num_steps = Config.NUM_STEPS
    obs_normalize = Config.OBS_NORMALIZE
    average_total_obs = Config.AVERAGE_TOTAL_OBS
    
    exp_id = Config.EXP_ID
    
    log_to_wandb = Config.WANDB
    
    device = Config.DEVICE
    test_mode = 'unseen'
    hidden_dim = Config.HIDDEN_DIM
    dropout = Config.DROPOUT
    num_layer = Config.NUM_LAYER
    num_head = Config.NUM_HEAD
    activation_func = Config.ACTIVATION_FUNC
    action_tanh = Config.ACTION_TANH
    
    agent_index = Config.AGENT_INDEX
    oppo_index = Config.OPPO_INDEX
    
    data_path = Config.OFFLINE_DATA_PATH
    
    seen_oppo_policy = Config.SEEN_OPPO_POLICY
    unseen_oppo_policy = Config.UNSEEN_OPPO_POLICY
    normalizer = Config.normalizer
    CONFIG_DICT = get_config_dict()
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    CONFIG_DICT["SEED_RES"] = seed
    CONFIG_DICT["DEVICE"] = device
    CONFIG_DICT["EVAL_DEVICE"] = device
    
    if test_mode == "seen":
        test_oppo_policy = seen_oppo_policy
    elif test_mode == "unseen":
        test_oppo_policy = unseen_oppo_policy
    elif test_mode == "mix":
        test_oppo_policy = seen_oppo_policy+unseen_oppo_policy
    
    env_and_test_oppo = get_env_and_oppo(CONFIG_DICT, test_oppo_policy)
    
    offline_data = load_agent_oppo_data(data_path, agent_index, oppo_index, act_dim, config_dict=CONFIG_DICT)
    LOG.info("Finish loading offline dataset.")

    offline_data = pad_future(offline_data)

    normalizers = DatasetNormalizer(offline_data,normalizer)

    if obs_normalize:
        agent_obs_mean_list, agent_obs_std_list, oppo_obs_mean_list, oppo_obs_std_list,agent_obs_max_list, agent_obs_min_list = cal_agent_oppo_obs_mean(offline_data, total=average_total_obs)
        CONFIG_DICT["AGENT_OBS_MEAN"] = agent_obs_mean_list
        CONFIG_DICT["AGENT_OBS_STD"] = agent_obs_std_list
        CONFIG_DICT["OPPO_OBS_MEAN"] = oppo_obs_mean_list
        CONFIG_DICT["OPPO_OBS_STD"] = oppo_obs_std_list
        CONFIG_DICT["AGENT_OBS_MAX"] = agent_obs_max_list
        CONFIG_DICT["AGENT_OBS_MIN"] = agent_obs_min_list
    

    with open(f"utility/{test_mode}_oppo_indexes.npy", 'rb') as f:
        test_oppo_indexes = np.load(f)
        CONFIG_DICT["TEST_OPPO_INDEXES"] = test_oppo_indexes
        LOG.info(f"{test_mode}_oppo_indexes: {test_oppo_indexes}")
    
    exp_prefix = env_type
    num_oppo_policy = len(test_oppo_policy)
    group_name = f'{exp_prefix}-{test_mode}-{num_oppo_policy}oppo'
    curtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_prefix = f'{group_name}-{exp_id}-{curtime}'
    
    encoder = GPTEncoder(
        conf=CONFIG_DICT,
        obs_dim=oppo_obs_dim,
        act_dim=act_dim,
        hidden_size=hidden_dim,
        max_ep_len=(num_steps+20),
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
    encoder_path = '***'
    encoder_state_dict = torch.load(encoder_path,map_location=device)
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()



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
        history_horizon=Config.history_horizon,
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

    decoder = decoder.to(device=device)
    decoder_load_path = '***'
    decoder_state_dict = torch.load(decoder_load_path,map_location=device)
    decoder.load_state_dict(decoder_state_dict['ema'])
    decoder.eval()

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project="Collect_2" + f"-{test_mode}",
            config=CONFIG_DICT,
        )
    
    LOG.info("Start testing TAO.")
    LOG.info(f'Testing mode: {test_mode}')
    all_context_window = collect_all_episodes(
        encoder=encoder,
        decoder=decoder,
        env_and_test_oppo=env_and_test_oppo,
        num_test=400,
        switch_interval=40,
        test_oppo_policy=test_oppo_policy,
        normalizers=normalizers,
        config=CONFIG_DICT,
        log_to_wandb=log_to_wandb,
    )

    all_online_data = load_all_data(all_context_window)

    return all_online_data