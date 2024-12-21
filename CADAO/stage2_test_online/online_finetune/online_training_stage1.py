import os, sys
from pprint import PrettyPrinter
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import time
import pickle
from offline_stage_1.config import Config, get_config_dict
from offline_stage_1.net import GPTEncoder
from offline_stage_1.nn_trainer import PolicyEmbeddingTrainer
from offline_stage_1.utils import (
    get_batch_mix,
    load_oppo_data,
    cal_obs_mean,
    get_batch_mix_offline,
    CrossEntropy,
    LOG,
)
import torch
import numpy as np
import wandb
from stage2_test_online.online_finetune.collect_data import collect_data
if Config.RUN_OFFLINE:
    os.environ["WANDB_MODE"] = "offline"


def main():
    env_type = Config.ENV_TYPE  
    obs_dim = Config.OBS_DIM  
    act_dim = Config.ACT_DIM  
    num_steps = Config.NUM_STEPS  

    exp_id = Config.EXP_ID
    seed = Config.SEED_PEL
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_to_wandb = Config.WANDB
    device = Config.DEVICE
    num_iter = Config.NUM_ITER  
    num_update_per_iter = Config.NUM_UPDATE_PER_ITER  
    checkpoint_freq = Config.CHECKPOINT_FREQ  
    batch_size = Config.BATCH_SIZE
    learning_rate = Config.LEARNING_RATE
    obs_normalize = Config.OBS_NORMALIZE  
    average_total_obs = Config.AVERAGE_TOTAL_OBS  
    dropout = Config.DROPOUT
    num_layer = Config.NUM_LAYER  
    num_head = Config.NUM_HEAD  
    activation_func = Config.ACTIVATION_FUNC
    warmup_steps = Config.WARMUP_STEPS  
    weight_decay = Config.WEIGHT_DECAY

    hidden_dim = Config.HIDDEN_DIM

    agent_index = Config.AGENT_INDEX  
    oppo_index = Config.OPPO_INDEX  

    data_path = "../../envs/multiagent_particle_envs/data/offline_dataset_PA_5oppo_10k.pkl"
    save_model_dir = Config.PEL_MODEL_DIR

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir, exist_ok=False)

    CONFIG_DICT = get_config_dict() 

    offline_data = load_oppo_data(data_path, oppo_index, act_dim, CONFIG_DICT) 
    LOG.info("Finish loading offline dataset.")

    if obs_normalize:
        obs_offline_mean_list, obs_offline_std_list = cal_obs_mean(offline_data, total=average_total_obs)
        CONFIG_DICT["offline_OBS_MEAN"] = obs_offline_mean_list
        CONFIG_DICT["offline_OBS_STD"] = obs_offline_std_list
# 待收集 修改
    with open(('***.pkl'), 'rb') as f: 
        online_data = pickle.load(f)
    # online_data = collect_data()
    LOG.info("Finish loading online dataset.")
    if obs_normalize:
        obs_online_mean_list, obs_online_std_list = cal_obs_mean(online_data, total=average_total_obs)
        CONFIG_DICT["online_OBS_MEAN"] = obs_online_mean_list
        CONFIG_DICT["online_OBS_STD"] = obs_online_std_list

    exp_prefix = env_type
    num_oppo_policy = len(offline_data)+len(online_data)
    group_name = f'{exp_prefix}-{num_oppo_policy}oppo'
    curtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_prefix = f'{group_name}-{exp_id}-{curtime}'
    LOG.info("--------------------- EXP INFO ---------------------")
    PrettyPrinter().pprint(CONFIG_DICT)
    LOG.info("----------------------------------------------------")

    encoder = GPTEncoder(
        conf=CONFIG_DICT,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=hidden_dim,
        max_ep_len=(num_steps + 20),
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
    encoder_optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    encoder_scheduler = torch.optim.lr_scheduler.LambdaLR(
        encoder_optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    trainer = PolicyEmbeddingTrainer(
        encoder=encoder,
        batch_size=batch_size,
        encoder_optimizer=encoder_optimizer,
        encoder_scheduler=encoder_scheduler,
        get_batch_fn=get_batch_mix_offline(offline_data, online_data, CONFIG_DICT),
        loss_gen_fn=CrossEntropy,
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

    LOG.info("Start policy embedding learning.")
    for i in range(num_iter):
        LOG.info(f"----------- Iteration [{i}] -----------")
        outputs = trainer.train(
            num_update=num_update_per_iter,
        )

        if i % checkpoint_freq == 0:
            trainer.save_model(
                postfix=f"_iter_{i}",
                save_dir=save_model_dir,
            )
            LOG.info(f"Finish training of iteration [{i}].")

        outputs.update({"global_step": i})

        if log_to_wandb:
            wandb.log(outputs)

    trainer.save_model(
        postfix=f"_iter_{i}",
        save_dir=save_model_dir,
    )
if __name__ == '__main__':
    main()