import numpy as np
from torch import nn
import pickle, torch
from torch.distributions import Categorical
import logging
import time
import os, sys
import random
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import envs.multiagent_particle_envs.multiagent.scenarios as scenarios
from envs.multiagent_particle_envs.multiagent.environment import MultiAgentEnv
from envs.multiagent_particle_envs.opponent_policy import get_all_oppo_policies as get_all_oppo_policies_mpe
from open_spiel.python import rl_environment
from envs.markov_soccer.opponent_policy import get_all_oppo_policies as get_all_oppo_policies_soccer
from envs.markov_soccer.soccer_state import get_two_state
from stage2_test_online.utils.arrays import to_torch
from stage2_test_online.normalizers import DatasetNormalizer
import wandb
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
LOG = logging.getLogger()

def collect_episodes(encoder, decoder, env_and_test_oppo, num_test, switch_interval, test_oppo_policy, normalizers, config, log_to_wandb):
    LOG.info(f'Testing against opponent policies: {test_oppo_policy}')
    LOG.info(f'# of total testing episodes: {num_test}')
    LOG.info(f'# of total testing opponent policies: {num_test // switch_interval}')
    env = env_and_test_oppo["env"]
    agent_obs_dim, oppo_obs_dim, act_dim = config['AGENT_OBS_DIM'], config['OPPO_OBS_DIM'], config['ACT_DIM']
    agent_index, oppo_index = config["AGENT_INDEX"], config["OPPO_INDEX"]
    num_steps = config["NUM_STEPS"]
    env_type = config["ENV_TYPE"]
    c_dim = config["C_DIM"]
    reward_scale = config["REWARD_SCALE"]
    device = config["DEVICE"]
    eval_mode = config['EVAL_MODE']
    test_oppo_indexes = config["TEST_OPPO_INDEXES"]
    ocw_size = config["OCW_SIZE"]
    hidden_dim = config["HIDDEN_DIM"]
    history_horizon = config["history_horizon"]
    encoder.eval()
    decoder.eval()

    returns = []
    cur_test_oppo_index = 0
    oppo_context_window = None
    oppo_part_context_window = None
    true_context_window = None
    for i in range(num_test):
        outputs = dict()
        if i % switch_interval == 0:
            test_start = time.time()
            oppo_id = test_oppo_indexes[cur_test_oppo_index]
            oppo_name = test_oppo_policy[oppo_id]
            if isinstance(oppo_name, tuple):
                oppo_name_ = oppo_name[0]+'_'+oppo_name[-1]
            else:
                oppo_name_ = oppo_name
            oppo_policy = env_and_test_oppo["oppo_policy"][oppo_id]
            if oppo_name in config["SEEN_OPPO_POLICY"]:
                target_rtg =  config["OPPO_TARGET"][oppo_id] / reward_scale *2
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = config['AGENT_OBS_MEAN'][oppo_id], config['AGENT_OBS_STD'][oppo_id]
                    oppo_obs_mean, oppo_obs_std = config['OPPO_OBS_MEAN'][oppo_id], config['OPPO_OBS_STD'][oppo_id]
                    agent_obs_max, agent_obs_min = config['AGENT_OBS_MAX'][oppo_id], config['AGENT_OBS_MIN'][oppo_id]
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            else:
                # target_rtg =  np.max(config["OPPO_TARGET"]) / reward_scale 
                target_rtg = np.mean(config["OPPO_TARGET"]) / reward_scale 
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = np.mean(np.stack(config['AGENT_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['AGENT_OBS_STD'], axis=0), axis=0)
                    oppo_obs_mean, oppo_obs_std = np.mean(np.stack(config['OPPO_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['OPPO_OBS_STD'], axis=0), axis=0)
                    agent_obs_max, agent_obs_min = np.mean(np.stack(config['AGENT_OBS_MAX'], axis=0), axis=0), np.mean(np.stack(config['AGENT_OBS_MIN'], axis=0), axis=0)
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            LOG.info(f'Start testing against opponent policies: {oppo_name_} ...')

        with torch.no_grad():
            ret, oppo_context_window_new, true_context_window = online_episode_get_window(
                env,
                env_type,
                agent_obs_dim,
                oppo_obs_dim,
                act_dim,
                c_dim,
                hidden_dim,
                history_horizon,
                encoder,
                decoder,
                oppo_policy,
                agent_index,
                oppo_index,
                num_steps=num_steps,
                reward_scale=reward_scale,
                returns_rtg = target_rtg,
                eval_mode=eval_mode,
                agent_obs_mean=agent_obs_mean,
                agent_obs_std=agent_obs_std,
                oppo_obs_mean=oppo_obs_mean,
                oppo_obs_std=oppo_obs_std,
                agent_obs_max=agent_obs_max,
                agent_obs_min=agent_obs_min,
                oppo_context_window=oppo_context_window,
                oppo_part_context_window=oppo_part_context_window,
                true_context_window=true_context_window,
                normalizers=normalizers,
                device=device,
                obs_normalize=config['OBS_NORMALIZE'],
                )
            oppo_context_window = oppo_context_window_new
            oppo_part_context_window = oppo_context_window[-ocw_size:]
        returns.append(ret)
        outputs.update({
            'test-epi/global_return': ret,
            f'test-epi/{oppo_name_}_target_{target_rtg:.3f}_return': ret,
            "granularity/num_episode": i,
        })
        if (i+1) % switch_interval == 0:
            test_oppo_log = {
                'test-oppo/oppo_return': np.mean(returns[-switch_interval:]),
                "granularity/num_opponent_policy": cur_test_oppo_index,
                "time/testing": time.time() - test_start,
            }
            outputs.update(test_oppo_log)
            LOG.info(f'Testing result of opponent [{cur_test_oppo_index}]:')
            for k, v in test_oppo_log.items():
                LOG.info(f'{k}: {v}')
            LOG.info('=' * 80)
            cur_test_oppo_index += 1
        if log_to_wandb:
            wandb.log(outputs)
    return_mean = np.mean(returns)
    length = len(true_context_window)
    LOG.info(f'ocw的长度为: {length}')
    LOG.info(f'Average return against all opponent policies: {return_mean}')
    if log_to_wandb:
        wandb.log({'test-epi/global_return_mean': return_mean})
    return true_context_window


def load_online_data(oppo_context_w):
    online_data_list = [[]]
    online_cw_size = len(oppo_context_w)
    for i in range(online_cw_size):
        num_steps = len(oppo_context_w[i][0])
        o_ep = []
        a_ep = []
        r_ep = []
        o_next_ep = []
        for j in range(num_steps):
            o_ep.append(np.array(oppo_context_w[i][3][j].cpu()))
            a_ep.append(np.array(oppo_context_w[i][1][j].cpu()))
            r_ep.append(np.array(oppo_context_w[i][2][j].cpu()))
            o_next_ep.append(np.array(oppo_context_w[i][0][j].cpu()))
        online_data_list[0].append(
            {
                "observations": np.array(o_ep),
                "actions": np.array(a_ep),
                "rewards": np.array(r_ep),
                "next_observations": np.array(o_next_ep),
            }
        )
    return online_data_list


def online_episode_get_window(
        env,
        env_type,
        agent_obs_dim,
        oppo_obs_dim,
        act_dim,
        c_dim,
        hidden_dim,
        history_horizon,
        encoder,
        decoder,
        oppo_policy,
        agent_index,
        oppo_index,
        num_steps=100,
        reward_scale=100.,
        returns_rtg = None,
        eval_mode='normal',
        agent_obs_mean=0.,
        agent_obs_std=1.,
        oppo_obs_mean=0.,
        oppo_obs_std=1.,
        agent_obs_max=1.,
        agent_obs_min=0.,
        oppo_context_window=None,
        oppo_part_context_window=None,
        true_context_window=None,
        normalizers=None,
        device="cuda",
        obs_normalize=True,
):
    encoder.eval()
    encoder.to(device=device)
    K = encoder.K
    decoder.eval()
    decoder.to(device=device)

    # hidden_obs_dim = agent_obs_dim + hidden_dim
    agent_obs_mean = torch.from_numpy(agent_obs_mean).to(device=device)
    agent_obs_std = torch.from_numpy(agent_obs_std).to(device=device)
    oppo_obs_mean = torch.from_numpy(oppo_obs_mean).to(device=device)
    oppo_obs_std = torch.from_numpy(oppo_obs_std).to(device=device)
    agent_obs_max = torch.from_numpy(agent_obs_max).to(device=device)
    agent_obs_min = torch.from_numpy(agent_obs_min).to(device=device)


    if env_type == 'MS':
        time_step = env.reset()
        _, _, rel_state1, rel_state2 = get_two_state(time_step)
        obs_n = [rel_state1, rel_state2]
    else:
        obs_n = env.reset()   
    if eval_mode == 'noise':
        for i in agent_index:
            obs_n[i] = obs_n[i] + np.random.normal(0, 0.1, size=obs_n[i].shape)

    if oppo_part_context_window != None:
        oppo_embeds, oppo_mask = [], []
        for oppo_trajs in oppo_part_context_window:
            n_o_oppo, a_oppo, r_oppo, _, _, timestep_oppo = oppo_trajs
            es = np.random.randint(0, n_o_oppo.shape[0])  
            oppo_embeds_, oppo_mask_ = encoder.get_tokens(
                n_o_oppo[es:es + K].to(device=device, dtype=torch.float32),
                a_oppo[es:es + K].to(device=device, dtype=torch.float32),
                r_oppo[es:es + K].to(device=device, dtype=torch.float32),
                timestep_oppo[es:es + K].to(device=device, dtype=torch.long),
                attention_mask=None,
            )
            oppo_embeds.append(oppo_embeds_)
            oppo_mask.append(oppo_mask_)
        oppo_embeds = torch.cat(oppo_embeds, dim=1).contiguous()
        oppo_mask = torch.cat(oppo_mask, dim=1).contiguous()

    else:
        oppo_embeds, oppo_mask = None, None
    
    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    obs_list_n = [None for _ in oppo_index + agent_index]
    obs_normal_list = [None for _ in oppo_index + agent_index]
    conditions_list = [None for _ in oppo_index + agent_index]
    samples_list = [None for _ in oppo_index + agent_index]
    obs_comb_list = [None for _ in oppo_index + agent_index]
    act_list_n = [None for _ in oppo_index + agent_index]
    r_list_n = [None for _ in oppo_index + agent_index]
    returns_rtg_list = [None for _ in oppo_index + agent_index]
    target_rtg_list_n = [None for _ in oppo_index + agent_index]
    timestep_list_n = [None for _ in oppo_index + agent_index]
    condition_obs_list = [None for _ in oppo_index + agent_index]
    condition_obs_norm_list = [None for _ in oppo_index + agent_index]
    for i in oppo_index + agent_index:
        obs_list_n[i] = torch.from_numpy(obs_n[i]).reshape(1, agent_obs_dim if i in agent_index else oppo_obs_dim).to(
            device=device, dtype=torch.float32)
        act_list_n[i] = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        r_list_n[i] = torch.zeros(0, device=device, dtype=torch.float32)
        ep_return = returns_rtg
        target_rtg_list_n[i] = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        returns_rtg_list[i] = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timestep_list_n[i] = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        if history_horizon > 0:
            condition_obs_list[i] = np.concatenate(
                [
                    np.repeat(
                        obs_n[i].reshape(1,-1)[:1],
                        history_horizon,
                        axis=0,
                    ),
                    obs_n[i].reshape(1,-1)
                ],
                axis=0
            )
            condition_obs_list[i] = torch.tensor(condition_obs_list[i],device=device)       
    episode_return = [0. for _ in oppo_index + agent_index]
    true_steps = 0
    for t in range(num_steps):
        act_n = [None for _ in oppo_index + agent_index]
        for i in agent_index:
            r_list_n[i] = torch.cat([r_list_n[i], torch.zeros(1, device=device)])
            if obs_normalize:
                obs_normal_list[i] = (obs_list_n[i][-1] - agent_obs_min) / (agent_obs_max - agent_obs_min)
                obs_normal_list[i] = obs_normal_list[i].reshape(1,agent_obs_dim)  
                obs_normal_list[i] = obs_normal_list[i].cpu().detach().numpy()
                obs_normal_list[i] = 2 * obs_normal_list[i] - 1                              
                if history_horizon > 0:
                    condition_obs_norm_list[i] = (condition_obs_list[i] - agent_obs_min) / (agent_obs_max - agent_obs_min)
                    condition_obs_norm_list[i] = condition_obs_norm_list[i].cpu().detach().numpy()
                    condition_obs_norm_list[i] = 2 * condition_obs_norm_list[i] - 1
                    # condition_obs_norm_list[i] = condition_obs_list[i].cpu().detach().numpy()
                    # condition_obs_norm_list[i] = normalizers.normalize(condition_obs_norm_list[i],"observations")
                    conditions_list[i] = {"x":to_torch(condition_obs_norm_list[i][-(history_horizon+1):].reshape(1,-1,agent_obs_dim),device=device)}
                else :
                    conditions_list[i] = {0: to_torch(obs_normal_list[i], device=device)}
                samples_list[i] = decoder.conditional_sample(conditions_list[i],returns=target_rtg_list_n[i],cross_attn=oppo_embeds,cross_mask=oppo_mask)
                if history_horizon > 0:
                    obs_comb_list[i] = torch.cat([samples_list[i][:, history_horizon, :], samples_list[i][:, history_horizon+1, :]], dim=-1)
                else:
                    obs_comb_list[i] = torch.cat([samples_list[i][:, 0, :], samples_list[i][:, 1, :]], dim=-1)
                obs_comb_list[i] = obs_comb_list[i].reshape(-1, 2*agent_obs_dim)
                action = decoder.inv_model(obs_comb_list[i])
            else:
                obs_normal_list[i] = obs_list_n[i]
                obs_normal_list[i] = obs_normal_list[i].cpu().detach().numpy()
                # obs_normal_list[i] = np.concatenate([obs_normal_list[i],hidden_states],axis=-1)
                conditions_list[i] = {0: to_torch(obs_normal_list[i], device=device)}
                samples_list[i] = decoder.conditional_sample(conditions_list[i],returns=returns_rtg_list[i],cross_attn=oppo_embeds)
                obs_comb_list[i] = torch.cat([samples_list[i][:, 0, :], samples_list[i][:, 1, :]], dim=-1)
                obs_comb_list[i] = obs_comb_list[i].reshape(-1, 2*agent_obs_dim)
                action = decoder.inv_model(obs_comb_list[i])
            if env_type == "PA":
                action = torch.nn.Softmax(dim=1)(action)  
                action_index = torch.argmax(action[0])
                action = torch.eye(act_dim, dtype=torch.float32)[action_index]
                act = action.detach().clone().cpu().numpy()
                act = np.concatenate([act, np.zeros(c_dim)])
            elif env_type == 'MS':
                action_prob = torch.nn.Softmax(dim=1)(action)
                dist = Categorical(action_prob[0])
                act = dist.sample().detach().clone().cpu().numpy()
                action = torch.eye(act_dim, dtype=torch.float32)[act]
            act_n[i] = act

        for j in oppo_index:
            act_list_n[j] = torch.cat([act_list_n[j], torch.zeros((1, act_dim), device=device)], dim=0)
            r_list_n[j] = torch.cat([r_list_n[j], torch.zeros(1, device=device)])
            if env_type == "PA":
                action = act = oppo_policy.action(obs_n[j])
            elif env_type == 'MS':
                action_prob = oppo_policy(torch.tensor(obs_n[j], dtype=torch.float32, device=device))
                dist = Categorical(action_prob)
                act = dist.sample().detach().clone().cpu().numpy()
                action = np.eye(act_dim, dtype=np.float32)[act]
            act_n[j] = act
            cur_action = torch.from_numpy(action[:act_dim]).to(device=device, dtype=torch.float32).reshape(1, act_dim)
            act_list_n[j][-1] = cur_action
        
        if env_type == 'MS':
            act_n = np.array(act_n)
            time_step = env.step(act_n)
            rew1, rew2 = time_step.rewards[0], time_step.rewards[1]
            reward_n = [rew1, rew2]
            _, _, rel_state1_, rel_state2_ = get_two_state(time_step)
            obs_n = [rel_state1_, rel_state2_]
            done_ = (time_step.last() == True)
            done_n = [done_, done_]
            info_n = {}
        else:
            obs_n, reward_n, done_n, info_n = env.step(act_n)

        for i in oppo_index + agent_index:
            cur_obs = torch.from_numpy(obs_n[i]).to(device=device, dtype=torch.float32).reshape(1,agent_obs_dim if i in agent_index else oppo_obs_dim)
            obs_list_n[i] = torch.cat([obs_list_n[i], cur_obs], dim=0)
            if i in agent_index:
                if history_horizon > 0:
                   condition_obs_list[i] = torch.cat([condition_obs_list[i], cur_obs], dim=0) 
            r_list_n[i][-1] = reward_n[i]

            if eval_mode != 'delayed':
                pred_return = target_rtg_list_n[i] - (reward_n[i] / reward_scale)
            else:
                pred_return = target_rtg_list_n[i][0, -1]
            target_rtg_list_n[i] = pred_return
            timestep_list_n[i] = torch.cat(
                [timestep_list_n[i],
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
            episode_return[i] += reward_n[i]
        true_steps += 1
        if done_n[0] or done_n[1]:
            break
    steps_ = min(num_steps, true_steps)
    for j in oppo_index:
        n_o_oppo_true = obs_list_n[j]
        a_oppo_true = act_list_n[j]
        r_oppo_true = r_list_n[j]
        timestep_oppo_true = timestep_list_n[j]    
    if true_context_window == None:
        true_context_window = [(n_o_oppo_true[1:1 + steps_, :], a_oppo_true[:steps_, :], r_oppo_true[:steps_], n_o_oppo_true[:steps_, :],
                                timestep_oppo_true[0, :steps_], timestep_oppo_true[0, 1:steps_ + 1])]
    else:
        true_context_window.append((n_o_oppo_true[1:1 + steps_, :], a_oppo_true[:steps_, :], r_oppo_true[:steps_], n_o_oppo_true[:steps_, :],
                                   timestep_oppo_true[0, :steps_], timestep_oppo_true[0, 1:steps_ + 1]))    
    for j in oppo_index:
        if obs_normalize:
            n_o_oppo = (obs_list_n[j] - oppo_obs_mean) / oppo_obs_std
        else:
            n_o_oppo = obs_list_n[j]
        a_oppo = act_list_n[j]
        r_oppo = r_list_n[j]
        timestep_oppo = timestep_list_n[j]

    if oppo_context_window == None:
        oppo_context_window = [(n_o_oppo[1:1 + steps_, :], a_oppo[:steps_, :], r_oppo[:steps_], n_o_oppo[:steps_, :],
                                timestep_oppo[0, :steps_], timestep_oppo[0, 1:steps_ + 1])]
    else:
        oppo_context_window.append((n_o_oppo[1:1 + steps_, :], a_oppo[:steps_, :], r_oppo[:steps_], n_o_oppo[:steps_, :],
                                   timestep_oppo[0, :steps_], timestep_oppo[0, 1:steps_ + 1]))

    average_epi_return = np.mean([episode_return[k] for k in agent_index])

    return average_epi_return, oppo_context_window, true_context_window


def collect_all_episodes(encoder, decoder, env_and_test_oppo, num_test, switch_interval, test_oppo_policy, normalizers, config, log_to_wandb):
    LOG.info(f'Testing against opponent policies: {test_oppo_policy}')
    LOG.info(f'# of total testing episodes: {num_test}')
    LOG.info(f'# of total testing opponent policies: {num_test // switch_interval}')
    env = env_and_test_oppo["env"]
    agent_obs_dim, oppo_obs_dim, act_dim = config['AGENT_OBS_DIM'], config['OPPO_OBS_DIM'], config['ACT_DIM']
    agent_index, oppo_index = config["AGENT_INDEX"], config["OPPO_INDEX"]
    num_steps = config["NUM_STEPS"]
    env_type = config["ENV_TYPE"]
    c_dim = config["C_DIM"]
    reward_scale = config["REWARD_SCALE"]
    device = config["DEVICE"]
    eval_mode = config['EVAL_MODE']
    test_oppo_indexes = config["TEST_OPPO_INDEXES"]
    ocw_size = config["OCW_SIZE"]
    hidden_dim = config["HIDDEN_DIM"]
    history_horizon = config["history_horizon"]
    encoder.eval()
    decoder.eval()

    returns = []
    cur_test_oppo_index = 0
    all_context_window = None
    all_part_context_window = None
    true_context_window = None
    for i in range(num_test):
        outputs = dict()
        if i % switch_interval == 0:
            test_start = time.time()
            oppo_id = test_oppo_indexes[cur_test_oppo_index]
            oppo_name = test_oppo_policy[oppo_id]
            if isinstance(oppo_name, tuple):
                oppo_name_ = oppo_name[0]+'_'+oppo_name[-1]
            else:
                oppo_name_ = oppo_name
            oppo_policy = env_and_test_oppo["oppo_policy"][oppo_id]
            if oppo_name in config["SEEN_OPPO_POLICY"]:
                target_rtg =  config["OPPO_TARGET"][oppo_id] / reward_scale 
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = config['AGENT_OBS_MEAN'][oppo_id], config['AGENT_OBS_STD'][oppo_id]
                    oppo_obs_mean, oppo_obs_std = config['OPPO_OBS_MEAN'][oppo_id], config['OPPO_OBS_STD'][oppo_id]
                    agent_obs_max, agent_obs_min = config['AGENT_OBS_MAX'][oppo_id], config['AGENT_OBS_MIN'][oppo_id]
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            else:
                target_rtg =  np.mean(config["OPPO_TARGET"]) / reward_scale *2
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = np.mean(np.stack(config['AGENT_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['AGENT_OBS_STD'], axis=0), axis=0)
                    oppo_obs_mean, oppo_obs_std = np.mean(np.stack(config['OPPO_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['OPPO_OBS_STD'], axis=0), axis=0)
                    agent_obs_max, agent_obs_min = np.mean(np.stack(config['AGENT_OBS_MAX'], axis=0), axis=0), np.mean(np.stack(config['AGENT_OBS_MIN'], axis=0), axis=0)
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            LOG.info(f'Start testing against opponent policies: {oppo_name_} ...')

        with torch.no_grad():
            ret, all_context_window_new, true_context_window = online_get_oppo_agent_window(
                env,
                env_type,
                agent_obs_dim,
                oppo_obs_dim,
                act_dim,
                c_dim,
                hidden_dim,
                history_horizon,
                encoder,
                decoder,
                oppo_policy,
                agent_index,
                oppo_index,
                num_steps=num_steps,
                reward_scale=reward_scale,
                returns_rtg = target_rtg,
                eval_mode=eval_mode,
                agent_obs_mean=agent_obs_mean,
                agent_obs_std=agent_obs_std,
                oppo_obs_mean=oppo_obs_mean,
                oppo_obs_std=oppo_obs_std,
                agent_obs_max=agent_obs_max,
                agent_obs_min=agent_obs_min,
                all_context_window=all_context_window,
                all_part_context_window=all_part_context_window,
                true_context_window=true_context_window,
                normalizers=normalizers,
                device=device,
                obs_normalize=config['OBS_NORMALIZE'],
                )
            all_context_window = all_context_window_new
            all_part_context_window = all_context_window[-ocw_size:]
        returns.append(ret)
        outputs.update({
            'test-epi/global_return': ret,
            f'test-epi/{oppo_name_}_target_{target_rtg:.3f}_return': ret,
            "granularity/num_episode": i,
        })
        if (i+1) % switch_interval == 0:
            test_oppo_log = {
                'test-oppo/oppo_return': np.mean(returns[-switch_interval:]),
                "granularity/num_opponent_policy": cur_test_oppo_index,
                "time/testing": time.time() - test_start,
            }
            outputs.update(test_oppo_log)
            LOG.info(f'Testing result of opponent [{cur_test_oppo_index}]:')
            for k, v in test_oppo_log.items():
                LOG.info(f'{k}: {v}')
            LOG.info('=' * 80)
            cur_test_oppo_index += 1
        if log_to_wandb:
            wandb.log(outputs)
    return_mean = np.mean(returns)
    length = len(true_context_window)
    LOG.info(f'ocw的长度为: {length}')
    LOG.info(f'Average return against all opponent policies: {return_mean}')
    if log_to_wandb:
        wandb.log({'test-epi/global_return_mean': return_mean})
    return true_context_window


def online_get_oppo_agent_window(
        env,
        env_type,
        agent_obs_dim,
        oppo_obs_dim,
        act_dim,
        c_dim,
        hidden_dim,
        history_horizon,
        encoder,
        decoder,
        oppo_policy,
        agent_index,
        oppo_index,
        num_steps=100,
        reward_scale=100.,
        returns_rtg = None,
        eval_mode='normal',
        agent_obs_mean=0.,
        agent_obs_std=1.,
        oppo_obs_mean=0.,
        oppo_obs_std=1.,
        agent_obs_max=1.,
        agent_obs_min=0.,
        all_context_window=None,
        all_part_context_window=None,
        true_context_window=None,
        normalizers=None,
        device="cuda",
        obs_normalize=True,
):
    encoder.eval()
    encoder.to(device=device)
    K = encoder.K
    decoder.eval()
    decoder.to(device=device)

    agent_obs_mean = torch.from_numpy(agent_obs_mean).to(device=device)
    agent_obs_std = torch.from_numpy(agent_obs_std).to(device=device)
    oppo_obs_mean = torch.from_numpy(oppo_obs_mean).to(device=device)
    oppo_obs_std = torch.from_numpy(oppo_obs_std).to(device=device)
    agent_obs_max = torch.from_numpy(agent_obs_max).to(device=device)
    agent_obs_min = torch.from_numpy(agent_obs_min).to(device=device)


    if env_type == 'MS':
        time_step = env.reset()
        _, _, rel_state1, rel_state2 = get_two_state(time_step)
        obs_n = [rel_state1, rel_state2]
    else:
        obs_n = env.reset()   
    if eval_mode == 'noise':
        for i in agent_index:
            obs_n[i] = obs_n[i] + np.random.normal(0, 0.1, size=obs_n[i].shape)

    if all_part_context_window != None:
        oppo_embeds, oppo_mask = [], []
        for oppo_trajs in all_part_context_window:
            _, _, _, _, _, _, n_o_oppo, a_oppo, r_oppo, _, _, timestep_oppo = oppo_trajs
            es = np.random.randint(0, n_o_oppo.shape[0])  
            oppo_embeds_, oppo_mask_ = encoder.get_tokens(
                n_o_oppo[es:es + K].to(device=device, dtype=torch.float32),
                a_oppo[es:es + K].to(device=device, dtype=torch.float32),
                r_oppo[es:es + K].to(device=device, dtype=torch.float32),
                timestep_oppo[es:es + K].to(device=device, dtype=torch.long),
                attention_mask=None,
            )
            oppo_embeds.append(oppo_embeds_)
            oppo_mask.append(oppo_mask_)
        oppo_embeds = torch.cat(oppo_embeds, dim=1).contiguous()
        oppo_mask = torch.cat(oppo_mask, dim=1).contiguous()

    else:
        oppo_embeds, oppo_mask = None, None
    

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    obs_list_n = [None for _ in oppo_index + agent_index]
    obs_normal_list = [None for _ in oppo_index + agent_index]
    conditions_list = [None for _ in oppo_index + agent_index]
    samples_list = [None for _ in oppo_index + agent_index]
    obs_comb_list = [None for _ in oppo_index + agent_index]
    act_list_n = [None for _ in oppo_index + agent_index]
    r_list_n = [None for _ in oppo_index + agent_index]
    returns_rtg_list = [None for _ in oppo_index + agent_index]
    target_rtg_list_n = [None for _ in oppo_index + agent_index]
    timestep_list_n = [None for _ in oppo_index + agent_index]
    condition_obs_list = [None for _ in oppo_index + agent_index]
    condition_obs_norm_list = [None for _ in oppo_index + agent_index]
    for i in oppo_index + agent_index:
        obs_list_n[i] = torch.from_numpy(obs_n[i]).reshape(1, agent_obs_dim if i in agent_index else oppo_obs_dim).to(
            device=device, dtype=torch.float32)
        act_list_n[i] = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        r_list_n[i] = torch.zeros(0, device=device, dtype=torch.float32)
        ep_return = returns_rtg
        target_rtg_list_n[i] = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        returns_rtg_list[i] = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timestep_list_n[i] = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        if history_horizon > 0:
            condition_obs_list[i] = np.concatenate(
                [
                    np.repeat(
                        obs_n[i].reshape(1,-1)[:1],
                        history_horizon,
                        axis=0,
                    ),
                    obs_n[i].reshape(1,-1)
                ],
                axis=0
            )
            condition_obs_list[i] = torch.tensor(condition_obs_list[i],device=device)       
    episode_return = [0. for _ in oppo_index + agent_index]
    true_steps = 0
    for t in range(num_steps):
        act_n = [None for _ in oppo_index + agent_index]
        for i in agent_index:
            act_list_n[i] = torch.cat([act_list_n[i], torch.zeros((1, act_dim), device=device)], dim=0)
            r_list_n[i] = torch.cat([r_list_n[i], torch.zeros(1, device=device)])
            if obs_normalize:
                obs_normal_list[i] = (obs_list_n[i][-1] - agent_obs_min) / (agent_obs_max - agent_obs_min)
                obs_normal_list[i] = obs_normal_list[i].reshape(1,agent_obs_dim)  
                obs_normal_list[i] = obs_normal_list[i].cpu().detach().numpy()
                obs_normal_list[i] = 2 * obs_normal_list[i] - 1   
                if history_horizon > 0:
                    condition_obs_norm_list[i] = (condition_obs_list[i] - agent_obs_min) / (agent_obs_max - agent_obs_min)
                    condition_obs_norm_list[i] = condition_obs_norm_list[i].cpu().detach().numpy()
                    condition_obs_norm_list[i] = 2 * condition_obs_norm_list[i] - 1
                    conditions_list[i] = {"x":to_torch(condition_obs_norm_list[i][-(history_horizon+1):].reshape(1,-1,agent_obs_dim),device=device)}
                else :
                    conditions_list[i] = {0: to_torch(obs_normal_list[i], device=device)}
                samples_list[i] = decoder.conditional_sample(conditions_list[i],returns=target_rtg_list_n[i],cross_attn=oppo_embeds,cross_mask=oppo_mask)
                if history_horizon > 0:
                    obs_comb_list[i] = torch.cat([samples_list[i][:, history_horizon, :], samples_list[i][:, history_horizon+1, :]], dim=-1)
                else:
                    obs_comb_list[i] = torch.cat([samples_list[i][:, 0, :], samples_list[i][:, 1, :]], dim=-1)
                obs_comb_list[i] = obs_comb_list[i].reshape(-1, 2*agent_obs_dim)
                action = decoder.inv_model(obs_comb_list[i])
            else:
                obs_normal_list[i] = obs_list_n[i]
                obs_normal_list[i] = obs_normal_list[i].cpu().detach().numpy()
                conditions_list[i] = {0: to_torch(obs_normal_list[i], device=device)}
                samples_list[i] = decoder.conditional_sample(conditions_list[i],returns=returns_rtg_list[i],cross_attn=oppo_embeds)
                obs_comb_list[i] = torch.cat([samples_list[i][:, 0, :], samples_list[i][:, 1, :]], dim=-1)
                obs_comb_list[i] = obs_comb_list[i].reshape(-1, 2*agent_obs_dim)
                action = decoder.inv_model(obs_comb_list[i])
            if env_type == "PA":
                action = torch.nn.Softmax(dim=1)(action)   
                action_index = torch.argmax(action[0])
                action = torch.eye(act_dim, dtype=torch.float32)[action_index]
                act = action.detach().clone().cpu().numpy()
                act = np.concatenate([act, np.zeros(c_dim)])
            elif env_type == 'MS':
                action_prob = torch.nn.Softmax(dim=1)(action)
                dist = Categorical(action_prob[0])
                act = dist.sample().detach().clone().cpu().numpy()
                action = torch.eye(act_dim, dtype=torch.float32)[act]
            act_n[i] = act
            act_list_n[i][-1] = action

        for j in oppo_index:
            act_list_n[j] = torch.cat([act_list_n[j], torch.zeros((1, act_dim), device=device)], dim=0)
            r_list_n[j] = torch.cat([r_list_n[j], torch.zeros(1, device=device)])
            if env_type == "PA":
                action = act = oppo_policy.action(obs_n[j])
            elif env_type == 'MS':
                action_prob = oppo_policy(torch.tensor(obs_n[j], dtype=torch.float32, device=device))
                dist = Categorical(action_prob)
                act = dist.sample().detach().clone().cpu().numpy()
                action = np.eye(act_dim, dtype=np.float32)[act]
            act_n[j] = act
            cur_action = torch.from_numpy(action[:act_dim]).to(device=device, dtype=torch.float32).reshape(1, act_dim)
            act_list_n[j][-1] = cur_action
        
        if env_type == 'MS':
            act_n = np.array(act_n)
            time_step = env.step(act_n)
            rew1, rew2 = time_step.rewards[0], time_step.rewards[1]
            reward_n = [rew1, rew2]
            _, _, rel_state1_, rel_state2_ = get_two_state(time_step)
            obs_n = [rel_state1_, rel_state2_]
            done_ = (time_step.last() == True)
            done_n = [done_, done_]
            info_n = {}
        else:
            obs_n, reward_n, done_n, info_n = env.step(act_n)

        for i in oppo_index + agent_index:
            cur_obs = torch.from_numpy(obs_n[i]).to(device=device, dtype=torch.float32).reshape(1,agent_obs_dim if i in agent_index else oppo_obs_dim)
            obs_list_n[i] = torch.cat([obs_list_n[i], cur_obs], dim=0)
            if i in agent_index:
                if history_horizon > 0:
                   condition_obs_list[i] = torch.cat([condition_obs_list[i], cur_obs], dim=0) 
            r_list_n[i][-1] = reward_n[i]

            if eval_mode != 'delayed':
                pred_return = target_rtg_list_n[i] - (reward_n[i] / reward_scale)
            else:
                pred_return = target_rtg_list_n[i][0, -1]
            target_rtg_list_n[i] = pred_return
            timestep_list_n[i] = torch.cat(
                [timestep_list_n[i],
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
            episode_return[i] += reward_n[i]
        true_steps += 1
        if done_n[0] or done_n[1]:
            break
    steps_ = min(num_steps, true_steps)    

    for i in agent_index:
        n_o_agent = obs_list_n[i]
        a_agent = act_list_n[i]
        r_agent = r_list_n[i]
        timestep_agent = timestep_list_n[i]

        for j in oppo_index:
            n_o_oppo_true = obs_list_n[j]
            a_oppo_true = act_list_n[j]
            r_oppo_true = r_list_n[j]
            timestep_oppo_true = timestep_list_n[j]   

        if true_context_window == None:
            true_context_window = [(n_o_agent[1:1 + steps_, :],a_agent[:steps_, :],r_agent[:steps_],n_o_agent[:steps_, :],
                                timestep_agent[0, :steps_],timestep_agent[0, 1:steps_ + 1],
                                n_o_oppo_true[1:1 + steps_, :], a_oppo_true[:steps_, :], r_oppo_true[:steps_], n_o_oppo_true[:steps_, :],
                                timestep_oppo_true[0, :steps_], timestep_oppo_true[0, 1:steps_ + 1])]
        else:
            true_context_window.append((n_o_agent[1:1 + steps_, :],a_agent[:steps_, :],r_agent[:steps_],n_o_agent[:steps_, :],
                                timestep_agent[0, :steps_],timestep_agent[0, 1:steps_ + 1],
                                n_o_oppo_true[1:1 + steps_, :], a_oppo_true[:steps_, :], r_oppo_true[:steps_], n_o_oppo_true[:steps_, :],
                                timestep_oppo_true[0, :steps_], timestep_oppo_true[0, 1:steps_ + 1]))

        for j in oppo_index:
            if obs_normalize:
                n_o_oppo = (obs_list_n[j] - oppo_obs_mean) / oppo_obs_std
            else:
                n_o_oppo = obs_list_n[j]
            a_oppo = act_list_n[j]
            r_oppo = r_list_n[j]
            timestep_oppo = timestep_list_n[j]
        
        if all_context_window == None:
            all_context_window = [(n_o_agent[1:1 + steps_, :],a_agent[:steps_, :],r_agent[:steps_],n_o_agent[:steps_, :],
                                timestep_agent[0, :steps_],timestep_agent[0, 1:steps_ + 1],
                                n_o_oppo[1:1 + steps_, :], a_oppo[:steps_, :], r_oppo[:steps_], n_o_oppo[:steps_, :],
                                timestep_oppo[0, :steps_], timestep_oppo[0, 1:steps_ + 1])]
        else:
            all_context_window.append((n_o_agent[1:1 + steps_, :],a_agent[:steps_, :],r_agent[:steps_],n_o_agent[:steps_, :],
                                timestep_agent[0, :steps_],timestep_agent[0, 1:steps_ + 1],
                                n_o_oppo[1:1 + steps_, :], a_oppo[:steps_, :], r_oppo[:steps_], n_o_oppo[:steps_, :],
                                timestep_oppo[0, :steps_], timestep_oppo[0, 1:steps_ + 1]))
    
    average_epi_return = np.mean([episode_return[k] for k in agent_index])
    
    return average_epi_return, all_context_window, true_context_window


def get_batch_mix_revise_new(offline_data, online_data, config_dict):
    agent_obs_dim = config_dict["AGENT_OBS_DIM"]
    oppo_obs_dim = config_dict["OPPO_OBS_DIM"]
    act_dim = config_dict["ACT_DIM"]
    num_offline_oppo_policy = len(offline_data)
    num_online_oppo_policy = len(online_data)
    offline_num_trajs_list = config_dict["NUM_TRAJS"]
    online_num_trajs_list = [len(online_data[0])]
    batch_size = config_dict["BATCH_SIZE"]
    K_encoder = config_dict["K"]
    horizon = config_dict["horizon"]
    num_steps = config_dict["NUM_STEPS"]
    discount = config_dict["discount"]
    history_horizon = config_dict["history_horizon"]
    if config_dict["OBS_NORMALIZE"]:
        agent_obs_mean_list = config_dict["AGENT_OBS_MEAN"]
        agent_obs_std_list = config_dict["AGENT_OBS_STD"]
        oppo_obs_mean_list = config_dict["OPPO_OBS_MEAN"]
        oppo_obs_std_list = config_dict["OPPO_OBS_STD"]
        agent_obs_max_list = config_dict["AGENT_OBS_MAX"]
        agent_obs_min_list = config_dict["AGENT_OBS_MIN"]
    reward_scale = config_dict["REWARD_SCALE"]
    use_padding = config_dict["use_padding"]
    device = config_dict["DEVICE"]
    ocw_size = config_dict["OCW_SIZE"]
    normalizer = config_dict["normalizer"]
    offline_data = pad_future(offline_data)
    online_data = pad_future(online_data)
    normalizers = DatasetNormalizer(offline_data,normalizer)
    discounts = discount ** np.arange(num_steps)[:, None]
    if history_horizon > 0:
        offline_data = pad_offline_history(offline_data,config_dict)
        online_data = pad_online_history(online_data,config_dict)
        horizon = horizon + history_horizon

    target_test = 0.5
    def fn(batch_size=batch_size, max_len_e=K_encoder):
        n_o_e, a_e, r_e, timesteps_e, mask_e = [], [], [], [], []
        o_d, a_d, r_d = [], [], []
        n_o_d = []
        r_d_sum = []
        loss_masks = []
        diffuse_masks = []
        for i in range(num_offline_oppo_policy):
            batch_inds = np.random.choice(
                np.arange(offline_num_trajs_list[i]),
                size=batch_size,
                replace=False,
            )
            for k in range(batch_size):
                traj = offline_data[i][batch_inds[k]]
                max_start = traj[0]['rewards'].shape[0] 

                ds = np.random.randint(0, max_start)
                
                o_d.append(traj[0]['observations'][ds:ds + horizon].reshape(1, -1, agent_obs_dim))
                a_d.append(traj[0]['actions'][ds:ds + horizon].reshape(1, -1, act_dim))
                n_o_d.append(traj[0]['next_observations'][ds:ds+horizon].reshape(1, -1, agent_obs_dim))
                r_d_sum.append(discount_sum(traj[0]['rewards'][ds:],discounts)/reward_scale)       
                a_tlen_d = a_d[-1].shape[1]
                o_tlen_d = o_d[-1].shape[1]
                loss_mask = np.ones((horizon - 1 , 1))
                diffuse_mask = np.ones((horizon,1))
                diffuse_mask[:history_horizon+1] = 0.0
                if ds < history_horizon:
                    loss_mask[:history_horizon - ds] = 0.0
                if a_tlen_d < horizon :
                    loss_mask[a_tlen_d:] = 0.0 
                if o_tlen_d < horizon :
                    diffuse_mask[o_tlen_d:] = 0.0 
                diffuse_masks.append(diffuse_mask)
                loss_masks.append(loss_mask)
                for _ in range(horizon-a_tlen_d):
                    nomove = [0.,0.,0.,0.,1.]
                    a_d[-1] = np.concatenate([a_d[-1] , np.array(nomove).reshape(1,1,-1)], axis=1)
                for _ in range(horizon - o_tlen_d):
                    concat = n_o_d[-1][0][-1].reshape(1,1,-1)
                    o_d[-1] = np.concatenate([o_d[-1] ,concat], axis=1)

                n_o_d[-1] = np.concatenate([n_o_d[-1] , np.zeros((1, horizon - a_tlen_d, agent_obs_dim))], axis=1)
                if config_dict["OBS_NORMALIZE"]:
                    o_d[-1] = (o_d[-1] - agent_obs_min_list[i]) / (agent_obs_max_list[i] - agent_obs_min_list[i])
                    o_d[-1] = 2 * o_d[-1] - 1
                    # o_d[-1] = normalizers.normalize(o_d[-1],"observations")

                oppo_batch_inds = np.random.choice(
                    np.arange(offline_num_trajs_list[i]),
                    size=ocw_size,
                    replace=False,
                )
                n_o_e_, a_e_, r_e_, timesteps_e_, mask_e_ = [], [], [], [], []
                for j in range(ocw_size):
                    oppo_traj = offline_data[i][oppo_batch_inds[j]]
                    es = np.random.randint(0, oppo_traj[1]['rewards'].shape[0])
                    n_o_e_.append(oppo_traj[1]['next_observations'][es:es+max_len_e].reshape(1, -1, oppo_obs_dim))
                    a_e_.append(oppo_traj[1]['actions'][es:es+max_len_e].reshape(1, -1, act_dim))
                    r_e_.append(oppo_traj[1]['rewards'][es:es+max_len_e].reshape(1, -1, 1))
                    timesteps_e_.append(np.arange((es+1), (es+1+n_o_e_[-1].shape[1])).reshape(1, -1))
                    timesteps_e_[-1][timesteps_e_[-1] >= (num_steps+1)] = num_steps                    
                    tlen_e = n_o_e_[-1].shape[1]                    
                    n_o_e_[-1] = np.concatenate([np.zeros((1, max_len_e - tlen_e, oppo_obs_dim)), n_o_e_[-1]], axis=1)
                    if config_dict["OBS_NORMALIZE"]:
                        n_o_e_[-1] = (n_o_e_[-1] - oppo_obs_mean_list[i]) / oppo_obs_std_list[i]
                    a_e_[-1] = np.concatenate([np.ones((1, max_len_e - tlen_e, act_dim)) * -10., a_e_[-1]], axis=1)
                    r_e_[-1] = np.concatenate([np.zeros((1, max_len_e - tlen_e, 1)), r_e_[-1]], axis=1)
                    timesteps_e_[-1] = np.concatenate([np.zeros((1, max_len_e - tlen_e)), timesteps_e_[-1]], axis=1)
                    mask_e_.append(np.concatenate([np.zeros((1, max_len_e - tlen_e)), np.ones((1, tlen_e))], axis=1))
                n_o_e.append(np.concatenate(n_o_e_, axis=1))
                a_e.append(np.concatenate(a_e_, axis=1))
                r_e.append(np.concatenate(r_e_, axis=1))
                timesteps_e.append(np.concatenate(timesteps_e_, axis=1))
                mask_e.append(np.concatenate(mask_e_, axis=1))

        for i in range(num_online_oppo_policy):
            batch_inds = get_batch_inds_good(online_data,target_test)
            for k in range(batch_size):
                traj = online_data[i][batch_inds[k]]
                max_start = traj[0]['rewards'].shape[0] 

                ds = np.random.randint(0, max_start)
                
                o_d.append(traj[0]['observations'][ds:ds + horizon].reshape(1, -1, agent_obs_dim))
                a_d.append(traj[0]['actions'][ds:ds + horizon].reshape(1, -1, act_dim))
                n_o_d.append(traj[0]['next_observations'][ds:ds+horizon].reshape(1, -1, agent_obs_dim))
                r_d_sum.append(discount_sum(traj[0]['rewards'][ds:],discounts)/reward_scale)       
                a_tlen_d = a_d[-1].shape[1]
                o_tlen_d = o_d[-1].shape[1]
                loss_mask = np.ones((horizon - 1 , 1))
                diffuse_mask = np.ones((horizon,1))
                diffuse_mask[:history_horizon+1] = 0.0
                if ds < history_horizon:
                    loss_mask[:history_horizon - ds] = 0.0
                if a_tlen_d < horizon :
                    loss_mask[a_tlen_d:] = 0.0 
                if o_tlen_d < horizon :
                    diffuse_mask[o_tlen_d:] = 0.0 
                diffuse_masks.append(diffuse_mask)
                loss_masks.append(loss_mask)
                for _ in range(horizon-a_tlen_d):
                    nomove = [0.,0.,0.,0.,1.]
                    a_d[-1] = np.concatenate([a_d[-1] , np.array(nomove).reshape(1,1,-1)], axis=1)
                for _ in range(horizon - o_tlen_d):
                    concat = n_o_d[-1][0][-1].reshape(1,1,-1)
                    o_d[-1] = np.concatenate([o_d[-1] ,concat], axis=1)

                n_o_d[-1] = np.concatenate([n_o_d[-1] , np.zeros((1, horizon - a_tlen_d, agent_obs_dim))], axis=1)
                if config_dict["OBS_NORMALIZE"]:
                    # o_d[-1] = (o_d[-1] - agent_obs_mean_list[i]) / agent_obs_std_list[i]
                    o_d[-1] = (o_d[-1] - agent_obs_min_list[i]) / (agent_obs_max_list[i] - agent_obs_min_list[i])
                    o_d[-1] = 2 * o_d[-1] - 1
                    # o_d[-1] = normalizers.normalize(o_d[-1],"observations")

                oppo_batch_inds = np.random.choice(
                    np.arange(online_num_trajs_list[i]),
                    size=ocw_size,
                    replace=False,
                )
                n_o_e_, a_e_, r_e_, timesteps_e_, mask_e_ = [], [], [], [], []
                for j in range(ocw_size):
                    oppo_traj = online_data[i][oppo_batch_inds[j]]
                    es = np.random.randint(0, oppo_traj[1]['rewards'].shape[0])
                    n_o_e_.append(oppo_traj[1]['next_observations'][es:es+max_len_e].reshape(1, -1, oppo_obs_dim))
                    a_e_.append(oppo_traj[1]['actions'][es:es+max_len_e].reshape(1, -1, act_dim))
                    r_e_.append(oppo_traj[1]['rewards'][es:es+max_len_e].reshape(1, -1, 1))
                    timesteps_e_.append(np.arange((es+1), (es+1+n_o_e_[-1].shape[1])).reshape(1, -1))
                    timesteps_e_[-1][timesteps_e_[-1] >= (num_steps+1)] = num_steps                    
                    tlen_e = n_o_e_[-1].shape[1]                    
                    n_o_e_[-1] = np.concatenate([np.zeros((1, max_len_e - tlen_e, oppo_obs_dim)), n_o_e_[-1]], axis=1)
                    if config_dict["OBS_NORMALIZE"]:
                        n_o_e_[-1] = (n_o_e_[-1] - oppo_obs_mean_list[i]) / oppo_obs_std_list[i]
                    a_e_[-1] = np.concatenate([np.ones((1, max_len_e - tlen_e, act_dim)) * -10., a_e_[-1]], axis=1)
                    r_e_[-1] = np.concatenate([np.zeros((1, max_len_e - tlen_e, 1)), r_e_[-1]], axis=1)
                    timesteps_e_[-1] = np.concatenate([np.zeros((1, max_len_e - tlen_e)), timesteps_e_[-1]], axis=1)
                    mask_e_.append(np.concatenate([np.zeros((1, max_len_e - tlen_e)), np.ones((1, tlen_e))], axis=1))
                n_o_e.append(np.concatenate(n_o_e_, axis=1))
                a_e.append(np.concatenate(a_e_, axis=1))
                r_e.append(np.concatenate(r_e_, axis=1))
                timesteps_e.append(np.concatenate(timesteps_e_, axis=1))
                mask_e.append(np.concatenate(mask_e_, axis=1))


        o_d = torch.from_numpy(np.concatenate(o_d, axis=0)).to(dtype=torch.float32, device=device)
        a_d = torch.from_numpy(np.concatenate(a_d, axis=0)).to(dtype=torch.float32, device=device)
        r_d_sum = torch.from_numpy(np.array(r_d_sum).reshape(-1,1)).to(dtype=torch.float32,device=device)
        loss_masks = torch.from_numpy(np.concatenate(loss_masks,axis=0)).to(dtype=torch.float32, device=device)
        diffuse_masks = torch.from_numpy(np.array(diffuse_masks)).to(dtype=torch.float32, device=device)

        n_o_e = torch.from_numpy(np.concatenate(n_o_e, axis=0)).to(dtype=torch.float32, device=device)
        a_e = torch.from_numpy(np.concatenate(a_e, axis=0)).to(dtype=torch.float32, device=device)
        r_e = torch.from_numpy(np.concatenate(r_e, axis=0)).to(dtype=torch.float32, device=device)
        timesteps_e = torch.from_numpy(np.concatenate(timesteps_e, axis=0)).to(dtype=torch.long, device=device)
        mask_e = torch.from_numpy(np.concatenate(mask_e, axis=0)).to(device=device)
        
        return n_o_e, a_e, r_e, timesteps_e, mask_e, o_d, a_d, r_d_sum, loss_masks ,diffuse_masks
    
    return fn


def pad_online_history(online_data,config_dict):
    num_oppo_policy = len(online_data)
    num_trajs_list = [len(online_data[0])]
    history_horizon = config_dict["history_horizon"]
    nomove = [0,0,0,0,1]
    for i in range(num_oppo_policy):
        for j in range(num_trajs_list[i]):
            online_data[i][j][0]['observations'] = np.concatenate(
                [
                    np.repeat(
                        online_data[i][j][0]['observations'][:1],
                        history_horizon,
                        axis = 0,
                    ),
                    online_data[i][j][0]['observations']
                ],
                axis=0              
            )
            online_data[i][j][0]['actions'] = np.concatenate(
                [
                    np.repeat(
                        np.array(nomove).reshape(1,-1)[:1],
                        history_horizon,
                        axis=0,
                    ),
                    online_data[i][j][0]['actions']
                ],
                axis=0
            )
    #理论上rewards  也需要，考虑MS 环境可以不用concat
    return online_data    
    

def pad_offline_history(offline_data,config_dict):
    num_oppo_policy = config_dict["NUM_OPPO_POLICY"]
    num_trajs_list = config_dict["NUM_TRAJS"]
    history_horizon = config_dict["history_horizon"]
    nomove = [0,0,0,0,1]
    for i in range(num_oppo_policy):
        for j in range(num_trajs_list[i]):
            offline_data[i][j][0]['observations'] = np.concatenate(
                [
                    np.repeat(
                        offline_data[i][j][0]['observations'][:1],
                        history_horizon,
                        axis = 0,
                    ),
                    offline_data[i][j][0]['observations']
                ],
                axis=0              
            )
            offline_data[i][j][0]['actions'] = np.concatenate(
                [
                    np.repeat(
                        np.array(nomove).reshape(1,-1)[:1],
                        history_horizon,
                        axis=0,
                    ),
                    offline_data[i][j][0]['actions']
                ],
                axis=0
            )
    #理论上rewards  也需要，考虑MS 环境可以不用concat
    return offline_data


def discount_sum(x,discounts):

    discounts = discounts[:len(x)]
    discounts = discounts.reshape(1,-1)

    s = (discounts * x).sum()

    return s


def pad_future(offline_data):
    num_oppo_policy = len(offline_data)
    for i in range(num_oppo_policy):
        for traj in offline_data[i]:
            if traj[0]["observations"].shape[0] < 100 :
                traj[0]["observations"] = np.concatenate([traj[0]['observations'],traj[0]['next_observations'][-1].reshape(1,-1)],axis = 0)    
    return offline_data


def load_all_data(all_context_window):
    all_data_list = [[]]
    all_data_size = len(all_context_window)
    for i in range(all_data_size):
        num_steps = len(all_context_window[i][0])
        agent_o_ep = []
        agent_a_ep = []
        agent_r_ep = []
        agent_o_next_ep = []
        oppo_o_ep = []
        oppo_a_ep = []
        oppo_r_ep = []
        oppo_o_next_ep = []
        for j in range(num_steps):
            agent_o_ep.append(np.array(all_context_window[i][3][j].cpu()))
            agent_a_ep.append(np.array(all_context_window[i][1][j].cpu()))
            agent_r_ep.append(np.array(all_context_window[i][2][j].cpu()))
            agent_o_next_ep.append(np.array(all_context_window[i][0][j].cpu()))
            oppo_a_ep.append(np.array(all_context_window[i][7][j].cpu()))
            oppo_o_ep.append(np.array(all_context_window[i][9][j].cpu()))
            oppo_r_ep.append(np.array(all_context_window[i][8][j].cpu()))
            oppo_o_next_ep.append(np.array(all_context_window[i][6][j].cpu()))
        all_data_list[0].append([
            {
                "observations": np.array(agent_o_ep),
                "actions": np.array(agent_a_ep),
                "rewards": np.array(agent_r_ep),
                "next_observations": np.array(agent_o_next_ep),
            },
            {
                "observations": np.array(oppo_o_ep),
                "actions": np.array(oppo_a_ep),
                "rewards": np.array(oppo_r_ep),
                "next_observations": np.array(oppo_o_next_ep),
            }
        ])
    return all_data_list


def get_batch_inds_good(online_data, target_test):
    num_all = len(online_data[0])
    sum = np.zeros(num_all)
    list_up = [[]]
    for k in range(num_all):
        for i in range(len(online_data[0][k][0]["rewards"])):
            sum[k] = sum[k] + online_data[0][k][0]["rewards"][i]
    num_up = 0
    num_up_list = []
    for j in range(num_all):
        if(sum[j]>target_test):
            list_up[0].append(online_data[0][j])
            num_up_list.append(j)
            num_up = num_up + 1
    batch_inds_up = np.random.choice(
        np.arange(num_up),
        size=128,
        replace=False,
    )
    batch_inds = []
    for i in range(len(batch_inds_up)):
        batch_inds.append(num_up_list[batch_inds_up[i]])
    random.shuffle(batch_inds)
    return batch_inds


def test_episodes(encoder, decoder, env_and_test_oppo, num_test, switch_interval, test_oppo_policy, normalizers, config, args, log_to_wandb):
    LOG.info(f'Testing against opponent policies: {test_oppo_policy}')
    LOG.info(f'# of total testing episodes: {num_test}')
    LOG.info(f'# of total testing opponent policies: {num_test // switch_interval}')
    env = env_and_test_oppo["env"]
    agent_obs_dim, oppo_obs_dim, act_dim = config['AGENT_OBS_DIM'], config['OPPO_OBS_DIM'], config['ACT_DIM']
    agent_index, oppo_index = config["AGENT_INDEX"], config["OPPO_INDEX"]
    num_steps = config["NUM_STEPS"]
    env_type = config["ENV_TYPE"]
    c_dim = config["C_DIM"]
    reward_scale = config["REWARD_SCALE"]
    device = args.device
    eval_mode = config['EVAL_MODE']
    test_oppo_indexes = config["TEST_OPPO_INDEXES"]
    ocw_size = config["OCW_SIZE"]
    hidden_dim = config["HIDDEN_DIM"]
    history_horizon = config["history_horizon"]


    returns = []
    cur_test_oppo_index = 0
    oppo_context_window = None
    for i in range(num_test):
        outputs = dict()
        if i % switch_interval == 0:
            test_start = time.time()
            oppo_id = test_oppo_indexes[cur_test_oppo_index]
            oppo_name = test_oppo_policy[oppo_id]
            if isinstance(oppo_name, tuple):
                oppo_name_ = oppo_name[0]+'_'+oppo_name[-1]
            else:
                oppo_name_ = oppo_name
            oppo_policy = env_and_test_oppo["oppo_policy"][oppo_id]
            if oppo_name in config["SEEN_OPPO_POLICY"]:
                target_rtg =  config["OPPO_TARGET"][oppo_id] / reward_scale 
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = config['AGENT_OBS_MEAN'][oppo_id], config['AGENT_OBS_STD'][oppo_id]
                    oppo_obs_mean, oppo_obs_std = config['OPPO_OBS_MEAN'][oppo_id], config['OPPO_OBS_STD'][oppo_id]
                    agent_obs_max, agent_obs_min = config['AGENT_OBS_MAX'][oppo_id], config['AGENT_OBS_MIN'][oppo_id]
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            else:
                target_rtg =  np.max(config["OPPO_TARGET"]) / reward_scale 
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = np.mean(np.stack(config['AGENT_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['AGENT_OBS_STD'], axis=0), axis=0)
                    oppo_obs_mean, oppo_obs_std = np.mean(np.stack(config['OPPO_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['OPPO_OBS_STD'], axis=0), axis=0)
                    agent_obs_max, agent_obs_min = np.mean(np.stack(config['AGENT_OBS_MAX'], axis=0), axis=0), np.mean(np.stack(config['AGENT_OBS_MIN'], axis=0), axis=0)
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            encoder_path = ['encoder_path1','encoder_path2','encoder_path3','encoder_path4']
            decoder_path = ['decoder_path1','decoder_path2','decoder_path3','decoder_path4']
            return_eval = [ _ for _ in range(len(encoder_path))]
            for j in range(len(encoder_path)):
                encoder.load_model(encoder_path[j],device=device)
                encoder.eval()
                decoder_state_dict = torch.load(decoder_path[j],map_location=device)
                decoder.load_state_dict(decoder_state_dict['ema'])
                decoder.eval()
                return_all = []
                oppo_evaluate_context_window = None
                for k in range(10):
                    with torch.no_grad():
                        ret, oppo_evaluate_context_window_new = eval_episode_rtg(
                            env,
                            env_type,
                            agent_obs_dim,
                            oppo_obs_dim,
                            act_dim,
                            c_dim,
                            hidden_dim,
                            history_horizon,
                            encoder,
                            decoder,
                            oppo_policy,
                            agent_index,
                            oppo_index,
                            num_steps=num_steps,
                            reward_scale=reward_scale,
                            returns_rtg = target_rtg,
                            eval_mode=eval_mode,
                            agent_obs_mean=agent_obs_mean,
                            agent_obs_std=agent_obs_std,
                            oppo_obs_mean=oppo_obs_mean,
                            oppo_obs_std=oppo_obs_std,
                            agent_obs_max=agent_obs_max,
                            agent_obs_min=agent_obs_min,
                            oppo_context_window=oppo_context_window,
                            normalizers=normalizers,
                            device=device,
                            obs_normalize=config['OBS_NORMALIZE'],
                        )
                        oppo_evaluate_context_window = oppo_evaluate_context_window_new
                        oppo_evaluate_context_window = oppo_evaluate_context_window[-ocw_size:]
                    return_all.append(ret)
                return_eval[j] = np.mean(return_all[-10:])
            eval_index = 0
            max_eval = return_eval[0]
            for j in range(len(encoder_path)):
                if return_eval[j]>max_eval :
                    eval_index = j
                    max_eval= return_eval[j]
            cur_encoder_path = encoder_path[eval_index]
            cur_decoder_path = decoder_path[eval_index]
            LOG.info(f'Start testing against opponent policies: {oppo_name_} ...')

        encoder.load_model(cur_encoder_path,device=device)
        encoder.eval()
        decoder_state_dict = torch.load(cur_decoder_path,map_location=device)
        decoder.load_state_dict(decoder_state_dict['ema'])
        decoder.eval()

        with torch.no_grad():
            ret, oppo_context_window_new = eval_episode_rtg(
                env,
                env_type,
                agent_obs_dim,
                oppo_obs_dim,
                act_dim,
                c_dim,
                hidden_dim,
                history_horizon,
                encoder,
                decoder,
                oppo_policy,
                agent_index,
                oppo_index,
                num_steps=num_steps,
                reward_scale=reward_scale,
                returns_rtg = target_rtg,
                eval_mode=eval_mode,
                agent_obs_mean=agent_obs_mean,
                agent_obs_std=agent_obs_std,
                oppo_obs_mean=oppo_obs_mean,
                oppo_obs_std=oppo_obs_std,
                agent_obs_max=agent_obs_max,
                agent_obs_min=agent_obs_min,
                oppo_context_window=oppo_context_window,
                normalizers=normalizers,
                device=device,
                obs_normalize=config['OBS_NORMALIZE'],
                )
            oppo_context_window = oppo_context_window_new
            oppo_context_window = oppo_context_window[-ocw_size:]
        returns.append(ret)
        outputs.update({
            'test-epi/global_return': ret,
            f'test-epi/{oppo_name_}_target_{target_rtg:.3f}_return': ret,
            "granularity/num_episode": i,
        })
        if (i+1) % switch_interval == 0:
            test_oppo_log = {
                'test-oppo/oppo_return': np.mean(returns[-switch_interval:]),
                "granularity/num_opponent_policy": cur_test_oppo_index,
                "time/testing": time.time() - test_start,
            }
            outputs.update(test_oppo_log)
            LOG.info(f'Testing result of opponent [{cur_test_oppo_index}]:')
            for k, v in test_oppo_log.items():
                LOG.info(f'{k}: {v}')
            LOG.info('=' * 80)
            cur_test_oppo_index += 1
        if log_to_wandb:
            wandb.log(outputs)
    return_mean = np.mean(returns)
    LOG.info(f'Average return against all opponent policies: {return_mean}')
    if log_to_wandb:
        wandb.log({'test-epi/global_return_mean': return_mean})
    

def eval_episode_rtg(
        env,
        env_type,
        agent_obs_dim,
        oppo_obs_dim,
        act_dim,
        c_dim,
        hidden_dim,
        history_horizon,
        encoder,
        decoder,
        oppo_policy,
        agent_index,
        oppo_index,
        num_steps=100,
        reward_scale=100.,
        returns_rtg = None,
        eval_mode='normal',
        agent_obs_mean=0.,
        agent_obs_std=1.,
        oppo_obs_mean=0.,
        oppo_obs_std=1.,
        agent_obs_max=1.,
        agent_obs_min=0.,
        oppo_context_window=None,
        normalizers=None,
        device="cuda",
        obs_normalize=True,
):
    encoder.eval()
    encoder.to(device=device)
    K = encoder.K
    decoder.eval()
    decoder.to(device=device)

    agent_obs_mean = torch.from_numpy(agent_obs_mean).to(device=device)
    agent_obs_std = torch.from_numpy(agent_obs_std).to(device=device)
    oppo_obs_mean = torch.from_numpy(oppo_obs_mean).to(device=device)
    oppo_obs_std = torch.from_numpy(oppo_obs_std).to(device=device)
    agent_obs_max = torch.from_numpy(agent_obs_max).to(device=device)
    agent_obs_min = torch.from_numpy(agent_obs_min).to(device=device)


    if env_type == 'MS':
        time_step = env.reset()
        _, _, rel_state1, rel_state2 = get_two_state(time_step)
        obs_n = [rel_state1, rel_state2]
    else:
        obs_n = env.reset()  
    if eval_mode == 'noise':
        for i in agent_index:
            obs_n[i] = obs_n[i] + np.random.normal(0, 0.1, size=obs_n[i].shape)

    if oppo_context_window != None:
        oppo_embeds, oppo_mask = [], []
        for oppo_trajs in oppo_context_window:
            n_o_oppo, a_oppo, r_oppo, _, _, timestep_oppo = oppo_trajs
            es = np.random.randint(0, n_o_oppo.shape[0])  
            oppo_embeds_, oppo_mask_ = encoder.get_tokens(
                n_o_oppo[es:es + K].to(device=device, dtype=torch.float32),
                a_oppo[es:es + K].to(device=device, dtype=torch.float32),
                r_oppo[es:es + K].to(device=device, dtype=torch.float32),
                timestep_oppo[es:es + K].to(device=device, dtype=torch.long),
                attention_mask=None,
            )
            oppo_embeds.append(oppo_embeds_)
            oppo_mask.append(oppo_mask_)
        oppo_embeds = torch.cat(oppo_embeds, dim=1).contiguous()
        oppo_mask = torch.cat(oppo_mask, dim=1).contiguous()

    else:
        oppo_embeds, oppo_mask = None, None
    
    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    obs_list_n = [None for _ in oppo_index + agent_index]
    obs_normal_list = [None for _ in oppo_index + agent_index]
    conditions_list = [None for _ in oppo_index + agent_index]
    samples_list = [None for _ in oppo_index + agent_index]
    obs_comb_list = [None for _ in oppo_index + agent_index]
    act_list_n = [None for _ in oppo_index + agent_index]
    r_list_n = [None for _ in oppo_index + agent_index]
    returns_rtg_list = [None for _ in oppo_index + agent_index]
    target_rtg_list_n = [None for _ in oppo_index + agent_index]
    timestep_list_n = [None for _ in oppo_index + agent_index]
    condition_obs_list = [None for _ in oppo_index + agent_index]
    condition_obs_norm_list = [None for _ in oppo_index + agent_index]
    for i in oppo_index + agent_index:
        obs_list_n[i] = torch.from_numpy(obs_n[i]).reshape(1, agent_obs_dim if i in agent_index else oppo_obs_dim).to(
            device=device, dtype=torch.float32)
        act_list_n[i] = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        r_list_n[i] = torch.zeros(0, device=device, dtype=torch.float32)
        ep_return = returns_rtg
        target_rtg_list_n[i] = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        returns_rtg_list[i] = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timestep_list_n[i] = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        if history_horizon > 0:
            condition_obs_list[i] = np.concatenate(
                [
                    np.repeat(
                        obs_n[i].reshape(1,-1)[:1],
                        history_horizon,
                        axis=0,
                    ),
                    obs_n[i].reshape(1,-1)
                ],
                axis=0
            )
            condition_obs_list[i] = torch.tensor(condition_obs_list[i],device=device)       
    episode_return = [0. for _ in oppo_index + agent_index]
    true_steps = 0
    for t in range(num_steps):
        act_n = [None for _ in oppo_index + agent_index]
        for i in agent_index:
            r_list_n[i] = torch.cat([r_list_n[i], torch.zeros(1, device=device)])
            if obs_normalize:
                obs_normal_list[i] = (obs_list_n[i] - agent_obs_mean) / agent_obs_std
                obs_normal_list[i] = obs_normal_list[i].cpu().detach().numpy()

                if history_horizon > 0:
                    condition_obs_norm_list[i] = (condition_obs_list[i] - agent_obs_min) / (agent_obs_max - agent_obs_min)
                    condition_obs_norm_list[i] = condition_obs_norm_list[i].cpu().detach().numpy()
                    condition_obs_norm_list[i] = 2 * condition_obs_norm_list[i] - 1
                    # condition_obs_norm_list[i] = condition_obs_list[i].cpu().detach().numpy()
                    # condition_obs_norm_list[i] = normalizers.normalize(condition_obs_norm_list[i],"observations")
                    conditions_list[i] = {"x":to_torch(condition_obs_norm_list[i][-(history_horizon+1):].reshape(1,-1,agent_obs_dim),device=device)}
                else :
                    conditions_list[i] = {0: to_torch(obs_normal_list[i], device=device)}
                samples_list[i] = decoder.conditional_sample(conditions_list[i],returns=target_rtg_list_n[i],cross_attn=oppo_embeds,cross_mask=oppo_mask)
                if history_horizon > 0:
                    obs_comb_list[i] = torch.cat([samples_list[i][:, history_horizon, :], samples_list[i][:, history_horizon+1, :]], dim=-1)
                else:
                    obs_comb_list[i] = torch.cat([samples_list[i][:, 0, :], samples_list[i][:, 1, :]], dim=-1)
                obs_comb_list[i] = obs_comb_list[i].reshape(-1, 2*agent_obs_dim)
                action = decoder.inv_model(obs_comb_list[i])
            else:
                obs_normal_list[i] = obs_list_n[i]
                obs_normal_list[i] = obs_normal_list[i].cpu().detach().numpy()
                # obs_normal_list[i] = np.concatenate([obs_normal_list[i],hidden_states],axis=-1)
                conditions_list[i] = {0: to_torch(obs_normal_list[i], device=device)}
                samples_list[i] = decoder.conditional_sample(conditions_list[i],returns=returns_rtg_list[i],cross_attn=oppo_embeds)
                obs_comb_list[i] = torch.cat([samples_list[i][:, 0, :], samples_list[i][:, 1, :]], dim=-1)
                obs_comb_list[i] = obs_comb_list[i].reshape(-1, 2*agent_obs_dim)
                action = decoder.inv_model(obs_comb_list[i])
            if env_type == "PA":
                action = torch.nn.Softmax(dim=1)(action)   
                action_index = torch.argmax(action[0])
                action = torch.eye(act_dim, dtype=torch.float32)[action_index]
                act = action.detach().clone().cpu().numpy()
                act = np.concatenate([act, np.zeros(c_dim)])
            elif env_type == 'MS':
                action_prob = torch.nn.Softmax(dim=1)(action)
                dist = Categorical(action_prob[0])
                act = dist.sample().detach().clone().cpu().numpy()
                action = torch.eye(act_dim, dtype=torch.float32)[act]
            act_n[i] = act

        for j in oppo_index:
            act_list_n[j] = torch.cat([act_list_n[j], torch.zeros((1, act_dim), device=device)], dim=0)
            r_list_n[j] = torch.cat([r_list_n[j], torch.zeros(1, device=device)])
            if env_type == "PA":
                action = act = oppo_policy.action(obs_n[j])
            elif env_type == 'MS':
                action_prob = oppo_policy(torch.tensor(obs_n[j], dtype=torch.float32, device=device))
                dist = Categorical(action_prob)
                act = dist.sample().detach().clone().cpu().numpy()
                action = np.eye(act_dim, dtype=np.float32)[act]
            act_n[j] = act
            cur_action = torch.from_numpy(action[:act_dim]).to(device=device, dtype=torch.float32).reshape(1, act_dim)
            act_list_n[j][-1] = cur_action
        
        if env_type == 'MS':
            act_n = np.array(act_n)
            time_step = env.step(act_n)
            rew1, rew2 = time_step.rewards[0], time_step.rewards[1]
            reward_n = [rew1, rew2]
            _, _, rel_state1_, rel_state2_ = get_two_state(time_step)
            obs_n = [rel_state1_, rel_state2_]
            done_ = (time_step.last() == True)
            done_n = [done_, done_]
            info_n = {}
        else:
            obs_n, reward_n, done_n, info_n = env.step(act_n)

        for i in oppo_index + agent_index:
            cur_obs = torch.from_numpy(obs_n[i]).to(device=device, dtype=torch.float32).reshape(1,agent_obs_dim if i in agent_index else oppo_obs_dim)
            if i in oppo_index :
                obs_list_n[i] = torch.cat([obs_list_n[i], cur_obs], dim=0)
            else :
                if history_horizon > 0:
                    condition_obs_list[i] = torch.cat([condition_obs_list[i], cur_obs], dim=0)
                else :
                    obs_list_n[i] = cur_obs
            r_list_n[i][-1] = reward_n[i]

            if eval_mode != 'delayed':
                pred_return = target_rtg_list_n[i] - (reward_n[i] / reward_scale)
            else:
                pred_return = target_rtg_list_n[i][0, -1]
            target_rtg_list_n[i] = pred_return
            timestep_list_n[i] = torch.cat(
                [timestep_list_n[i],
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
            episode_return[i] += reward_n[i]
        true_steps += 1
        if done_n[0] or done_n[1]:
            break
        
    for j in oppo_index:
        if obs_normalize:
            n_o_oppo = (obs_list_n[j] - oppo_obs_mean) / oppo_obs_std
        else:
            n_o_oppo = obs_list_n[j]
        a_oppo = act_list_n[j]
        r_oppo = r_list_n[j]
        timestep_oppo = timestep_list_n[j]
    steps_ = min(num_steps, true_steps)
    if oppo_context_window == None:
        oppo_context_window = [(n_o_oppo[1:1 + steps_, :], a_oppo[:steps_, :], r_oppo[:steps_], n_o_oppo[:steps_, :],
                                timestep_oppo[0, :steps_], timestep_oppo[0, 1:steps_ + 1])]
    else:
        oppo_context_window.append((n_o_oppo[1:1 + steps_, :], a_oppo[:steps_, :], r_oppo[:steps_], n_o_oppo[:steps_, :],
                                   timestep_oppo[0, :steps_], timestep_oppo[0, 1:steps_ + 1]))

    average_epi_return = np.mean([episode_return[k] for k in agent_index])

    return average_epi_return, oppo_context_window