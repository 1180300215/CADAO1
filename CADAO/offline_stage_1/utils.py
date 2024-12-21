import numpy as np
import pickle
import torch
import logging


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
LOG = logging.getLogger()


def load_oppo_data(data_path, oppo_index, act_dim, config_dict):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)     
        data_o = data["observations"]
        data_a = data["actions"]
        data_r = data["rewards"]
        data_o_next = data["next_observations"]
    num_oppo_policy = len(data_o)     
    config_dict["NUM_OPPO_POLICY"] = num_oppo_policy
    data_list = [[] for _ in range(num_oppo_policy)]   
    for i in range(num_oppo_policy):
        num_agent_policy = len(data_o[i])   
        for j in range(num_agent_policy):
            num_epis = len(data_o[i][j])   
            for e in range(num_epis):
                num_steps = len(data_o[i][j][e])   
                for oppo in oppo_index:
                    o_ep = []
                    a_ep = []
                    r_ep = []
                    o_next_ep = []
                    for k in range(num_steps):
                        o_ep.append(np.array(data_o[i][j][e][k][oppo]))
                        a_ep.append(np.array(data_a[i][j][e][k][oppo])[:act_dim])
                        r_ep.append(np.array(data_r[i][j][e][k][oppo]))
                        o_next_ep.append(np.array(data_o_next[i][j][e][k][oppo]))
                    data_list[i].append(
                        {
                            "observations": np.array(o_ep),
                            "actions": np.array(a_ep),
                            "rewards": np.array(r_ep),
                            "next_observations": np.array(o_next_ep),
                        }
                    )
    num_oppo_trajs_list = []          
    for i in range(num_oppo_policy):
        num_oppo_trajs_list.append(len(data_list[i]))
    config_dict["NUM_OPPO_TRAJS"] = num_oppo_trajs_list
    return data_list

def cal_obs_mean(trajectories, total=False):
    total_obses = []
    num_oppo_policy = len(trajectories)
    eps = 1e-6
    obs_mean_list = []
    obs_std_list = []
    for i in range(num_oppo_policy):
        obses_i = []
        for traj in trajectories[i]:
            total_obses.append(traj['observations'])
            obses_i.append(traj['observations'])
        obses_i = np.concatenate(obses_i, axis=0)
        obs_mean_list.append(np.mean(obses_i, axis=0))
        obs_std_list.append(np.std(obses_i, axis=0) + eps)
    if total:
        total_obses = np.concatenate(total_obses, axis=0)
        obs_mean_list = [np.mean(total_obses, axis=0) for _ in range(num_oppo_policy)]
        obs_std_list = [np.std(total_obses, axis=0) + eps for _ in range(num_oppo_policy)]
    return obs_mean_list, obs_std_list


def get_batch(offline_data, config_dict):
    obs_dim = config_dict["OBS_DIM"]
    act_dim = config_dict["ACT_DIM"]
    num_oppo_policy = config_dict["NUM_OPPO_POLICY"]
    num_oppo_trajs_list = config_dict["NUM_OPPO_TRAJS"]    
    batch_size = config_dict["BATCH_SIZE"]
    K = config_dict["NUM_STEPS"]  
    if config_dict["OBS_NORMALIZE"]:
        obs_mean_list = config_dict["OBS_MEAN"]
        obs_std_list = config_dict["OBS_STD"]
    device = config_dict["DEVICE"]
    mixed_data = [oppo_data for oppo_data_list in offline_data for oppo_data in oppo_data_list] 
    oppo_labels = np.concatenate(
        [np.ones((num_oppo_trajs_list[i],), dtype=np.int32) * i for i in range(num_oppo_policy)],
        axis=0,
        dtype=np.int32,
    )
    num_total_trajs = sum(num_oppo_trajs_list)
    shuffle_index = np.arange(num_total_trajs)  
    np.random.shuffle(shuffle_index)
    # shuffle the data and labels
    mixed_data = [mixed_data[i] for i in shuffle_index]
    oppo_labels = oppo_labels[shuffle_index]
    all_indexes = [np.argwhere(oppo_labels == i).reshape(-1) for i in range(num_oppo_policy)]
    def fn(batch_size=batch_size, max_len=K):
        n_o, a, r, label, timesteps, mask, o_gen, a_gen, mask_gen = [], [], [], [], [], [], [], [], []
        batch_inds = np.random.choice(
            np.arange(num_total_trajs),
            size=batch_size,
            replace=False,
        )     
        for k in range(batch_size):
            traj = mixed_data[batch_inds[k]]
            l = oppo_labels[batch_inds[k]]
            index_gen = np.random.choice(all_indexes[l])   
            traj_gen = mixed_data[index_gen]
            n_o.append(traj['next_observations'][:max_len].reshape(1, -1, obs_dim))
            a.append(traj['actions'][:max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][:max_len].reshape(1, -1, 1))
            o_gen.append(traj_gen['observations'][:max_len].reshape(1, -1, obs_dim))
            a_gen.append(traj_gen['actions'][:max_len].reshape(1, -1, act_dim))
            timesteps.append(
                np.arange(1, (n_o[-1].shape[1] + 1)).reshape(1, -1)
            )
            timesteps[-1][timesteps[-1] >= (max_len+1)] = max_len
            
            tlen = n_o[-1].shape[1]
            tlen_gen = o_gen[-1].shape[1]
            
            n_o[-1] = np.concatenate([np.zeros((1, max_len - tlen, obs_dim)), n_o[-1]], axis=1)
            o_gen[-1] = np.concatenate([np.zeros((1, max_len - tlen_gen, obs_dim)), o_gen[-1]], axis=1)
            if config_dict["OBS_NORMALIZE"]:
                obs_mean, obs_std = obs_mean_list[l], obs_std_list[l]
                n_o[-1] = (n_o[-1] - obs_mean) / obs_std
                o_gen[-1] = (o_gen[-1] - obs_mean) / obs_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            a_gen[-1] = np.concatenate([np.ones((1, max_len - tlen_gen, act_dim)) * -10., a_gen[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            
            label.append(l)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
            mask_gen.append(np.concatenate([np.zeros((1, max_len - tlen_gen)), np.ones((1, tlen_gen))], axis=1))
        
        n_o = torch.from_numpy(np.concatenate(n_o, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        label = torch.from_numpy(np.array(label)).to(dtype=torch.long, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        o_gen = torch.from_numpy(np.concatenate(o_gen, axis=0)).to(dtype=torch.float32, device=device)
        a_gen = torch.from_numpy(np.concatenate(a_gen, axis=0)).to(dtype=torch.float32, device=device)
        mask_gen = torch.from_numpy(np.concatenate(mask_gen, axis=0)).to(device=device)
        
        return n_o, a, r, label, timesteps, mask, o_gen, a_gen, mask_gen
    
    return fn


def get_batch_mix(offline_data,online_data,config_dict):
    obs_dim = config_dict["OBS_DIM"]
    act_dim = config_dict["ACT_DIM"]
    num_offline_oppo_policy = len(offline_data)
    num_online_oppo_policy = len(online_data)
    num_offline_oppo_trajs_list = config_dict["NUM_OPPO_TRAJS"] 
    num_online_oppo_trajs_list = [len(online_data[0])]
    batch_size = config_dict["BATCH_SIZE"]
    K = config_dict["NUM_STEPS"]  
    if config_dict["OBS_NORMALIZE"]:
        obs_offline_mean_list = config_dict["offline_OBS_MEAN"]
        obs_offline_std_list = config_dict["offline_OBS_STD"]
        obs_online_mean_list = config_dict["online_OBS_MEAN"]
        obs_online_std_list = config_dict["online_OBS_STD"]
    device = config_dict["DEVICE"]
    offline_mixed_data = [oppo_data for oppo_data_list in offline_data for oppo_data in oppo_data_list]  
    offline_oppo_labels = np.concatenate(
        [np.ones((num_offline_oppo_trajs_list[i],), dtype=np.int32) * i for i in range(num_offline_oppo_policy)],
        axis=0,
        dtype=np.int32,
    ) 
    online_mix_data = [oppo_data for oppo_data_list in online_data for oppo_data in oppo_data_list] 
    online_oppo_labels = np.concatenate(
        [np.ones((num_online_oppo_trajs_list[i],), dtype=np.int32) * (i + 5)for i in range(num_online_oppo_policy)],
        axis=0,
        dtype=np.int32,
    ) 
    num_offline_total_trajs = sum(num_offline_oppo_trajs_list)
    num_online_total_trajs = sum(num_online_oppo_trajs_list)
    shuffle_offline_index = np.arange(num_offline_total_trajs) 
    shuffle_online_index = np.arange(num_online_total_trajs)
    np.random.shuffle(shuffle_offline_index)
    np.random.shuffle(shuffle_online_index)
    # shuffle the data and labels(offline and online)
    offline_mixed_data = [offline_mixed_data[i] for i in shuffle_offline_index]
    offline_oppo_labels = offline_oppo_labels[shuffle_offline_index]
    online_mix_data = [online_mix_data[i] for i in shuffle_online_index]
    online_oppo_labels = online_oppo_labels[shuffle_online_index]
    all_offline_indexes = [np.argwhere(offline_oppo_labels == i).reshape(-1) for i in range(num_offline_oppo_policy)]
    all_online_indexes = [np.argwhere(online_oppo_labels == (i+5)).reshape(-1) for i in range(num_online_oppo_policy)]

    def fn(batch_size=batch_size, max_len=K):
        n_o, a, r, label, timesteps, mask, o_gen, a_gen, mask_gen = [], [], [], [], [], [], [], [], []
        batch_offline_inds = np.random.choice(
            np.arange(num_offline_total_trajs),
            size=int(batch_size/2),
            replace=False,
        )  
        for k in range(int(batch_size/2)):
            traj = offline_mixed_data[batch_offline_inds[k]]
            l = offline_oppo_labels[batch_offline_inds[k]]
            index_gen = np.random.choice(all_offline_indexes[l])  
            traj_gen = offline_mixed_data[index_gen]

            n_o.append(traj['next_observations'][:max_len].reshape(1, -1, obs_dim))    
            a.append(traj['actions'][:max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][:max_len].reshape(1, -1, 1))
            o_gen.append(traj_gen['observations'][:max_len].reshape(1, -1, obs_dim))  
            a_gen.append(traj_gen['actions'][:max_len].reshape(1, -1, act_dim))
            timesteps.append(
                np.arange(1, (n_o[-1].shape[1] + 1)).reshape(1, -1)
            )  
            timesteps[-1][timesteps[-1] >= (max_len + 1)] = max_len

            tlen = n_o[-1].shape[1]
            tlen_gen = o_gen[-1].shape[1]

            n_o[-1] = np.concatenate([np.zeros((1, max_len - tlen, obs_dim)), n_o[-1]], axis=1)
            o_gen[-1] = np.concatenate([np.zeros((1, max_len - tlen_gen, obs_dim)), o_gen[-1]], axis=1)
            if config_dict["OBS_NORMALIZE"]:
                obs_mean, obs_std = obs_offline_mean_list[l], obs_offline_std_list[l]
                n_o[-1] = (n_o[-1] - obs_mean) / obs_std
                o_gen[-1] = (o_gen[-1] - obs_mean) / obs_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            a_gen[-1] = np.concatenate([np.ones((1, max_len - tlen_gen, act_dim)) * -10., a_gen[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)

            label.append(l)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
            mask_gen.append(np.concatenate([np.zeros((1, max_len - tlen_gen)), np.ones((1, tlen_gen))], axis=1))
        batch_online_inds = np.random.choice(
            np.arange(num_online_total_trajs),
            size=int(batch_size/2),
            replace=False,
        )
        for s in range(int(batch_size/2)):
            traj = online_mix_data[batch_online_inds[s]]
            l = online_oppo_labels[batch_online_inds[s]]
            index_gen = np.random.choice(all_online_indexes[l-5]) 
            traj_gen = online_mix_data[index_gen]

            n_o.append(traj['next_observations'][:max_len].reshape(1, -1, obs_dim))    
            a.append(traj['actions'][:max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][:max_len].reshape(1, -1, 1))
            o_gen.append(traj_gen['observations'][:max_len].reshape(1, -1, obs_dim))  
            a_gen.append(traj_gen['actions'][:max_len].reshape(1, -1, act_dim))
            timesteps.append(
                np.arange(1, (n_o[-1].shape[1] + 1)).reshape(1, -1)
            )  
            timesteps[-1][timesteps[-1] >= (max_len + 1)] = max_len

            tlen = n_o[-1].shape[1]
            tlen_gen = o_gen[-1].shape[1]

            n_o[-1] = np.concatenate([np.zeros((1, max_len - tlen, obs_dim)), n_o[-1]], axis=1)
            o_gen[-1] = np.concatenate([np.zeros((1, max_len - tlen_gen, obs_dim)), o_gen[-1]], axis=1)
            if config_dict["OBS_NORMALIZE"]:
                obs_mean, obs_std = obs_online_mean_list[l-5], obs_online_std_list[l-5]
                n_o[-1] = (n_o[-1] - obs_mean) / obs_std
                o_gen[-1] = (o_gen[-1] - obs_mean) / obs_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            a_gen[-1] = np.concatenate([np.ones((1, max_len - tlen_gen, act_dim)) * -10., a_gen[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)

            label.append(l)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
            mask_gen.append(np.concatenate([np.zeros((1, max_len - tlen_gen)), np.ones((1, tlen_gen))], axis=1))

        all_shuffle_indexes = np.arange(batch_size)
        np.random.shuffle(all_shuffle_indexes)
        n_o = [n_o[i] for i in all_shuffle_indexes]
        a = [a[i] for i in all_shuffle_indexes]
        r = [r[i] for i in all_shuffle_indexes]
        label = [label[i] for i in all_shuffle_indexes]
        o_gen = [o_gen[i] for i in all_shuffle_indexes]
        a_gen = [a_gen[i] for i in all_shuffle_indexes]

        n_o = torch.from_numpy(np.concatenate(n_o, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        label = torch.from_numpy(np.array(label)).to(dtype=torch.long, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        o_gen = torch.from_numpy(np.concatenate(o_gen, axis=0)).to(dtype=torch.float32, device=device)
        a_gen = torch.from_numpy(np.concatenate(a_gen, axis=0)).to(dtype=torch.float32, device=device)
        mask_gen = torch.from_numpy(np.concatenate(mask_gen, axis=0)).to(device=device)

        return n_o, a, r, label, timesteps, mask, o_gen, a_gen, mask_gen

    return fn


def CrossEntropy(a_predict, a_label):
    ce = torch.nn.CrossEntropyLoss()
    a_label = torch.argmax(a_label, dim=-1)
    loss = ce(a_predict, a_label)
    return loss



def get_batch_mix_offline(offline_data,online_data,config_dict):
    obs_dim = config_dict["OBS_DIM"]
    act_dim = config_dict["ACT_DIM"]
    num_offline_oppo_policy = len(offline_data)
    num_online_oppo_policy = len(online_data)
    num_offline_oppo_trajs_list = config_dict["NUM_OPPO_TRAJS"]  
    num_online_oppo_trajs_list = [len(online_data[0])]
    batch_size = config_dict["BATCH_SIZE"]
    K = config_dict["NUM_STEPS"]  
    if config_dict["OBS_NORMALIZE"]:
        obs_offline_mean_list = config_dict["offline_OBS_MEAN"]
        obs_offline_std_list = config_dict["offline_OBS_STD"]
        obs_online_mean_list = config_dict["online_OBS_MEAN"]
        obs_online_std_list = config_dict["online_OBS_STD"]
    device = config_dict["DEVICE"]
    offline_mixed_data = [oppo_data for oppo_data_list in offline_data for oppo_data in oppo_data_list]  
    offline_oppo_labels = np.concatenate(
        [np.ones((num_offline_oppo_trajs_list[i],), dtype=np.int32) * i for i in range(num_offline_oppo_policy)],
        axis=0,
        dtype=np.int32,
    ) 
    online_mix_data = [oppo_data for oppo_data_list in online_data for oppo_data in oppo_data_list]  
    online_oppo_labels = np.concatenate(
        [np.ones((num_online_oppo_trajs_list[i],), dtype=np.int32) * (i + 5)for i in range(num_online_oppo_policy)],
        axis=0,
        dtype=np.int32,
    )  
    num_offline_total_trajs = sum(num_offline_oppo_trajs_list)
    num_online_total_trajs = sum(num_online_oppo_trajs_list)
    shuffle_offline_index = np.arange(num_offline_total_trajs)  
    shuffle_online_index = np.arange(num_online_total_trajs)
    np.random.shuffle(shuffle_offline_index)
    np.random.shuffle(shuffle_online_index)
    # shuffle the data and labels(offline and online)
    offline_mixed_data = [offline_mixed_data[i] for i in shuffle_offline_index]
    offline_oppo_labels = offline_oppo_labels[shuffle_offline_index]
    online_mix_data = [online_mix_data[i] for i in shuffle_online_index]
    online_oppo_labels = online_oppo_labels[shuffle_online_index]
    all_offline_indexes = [np.argwhere(offline_oppo_labels == i).reshape(-1) for i in range(num_offline_oppo_policy)]
    all_online_indexes = [np.argwhere(online_oppo_labels == (i+5)).reshape(-1) for i in range(num_online_oppo_policy)]

    def fn(batch_size=batch_size, max_len=K):
        n_o, a, r, label, timesteps, mask, o_gen, a_gen, mask_gen = [], [], [], [], [], [], [], [], []
        batch_offline_inds = np.random.choice(
            np.arange(num_offline_total_trajs),
            size=int(batch_size/2),
            replace=False,
        )  
        for k in range(int(batch_size/2)):
            traj = offline_mixed_data[batch_offline_inds[k]]
            l = offline_oppo_labels[batch_offline_inds[k]]
            index_gen = np.random.choice(all_offline_indexes[l]) 
            traj_gen = offline_mixed_data[index_gen]

            n_o.append(traj['next_observations'][:max_len].reshape(1, -1, obs_dim))    
            a.append(traj['actions'][:max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][:max_len].reshape(1, -1, 1))
            o_gen.append(traj_gen['observations'][:max_len].reshape(1, -1, obs_dim))  
            a_gen.append(traj_gen['actions'][:max_len].reshape(1, -1, act_dim))
            timesteps.append(
                np.arange(1, (n_o[-1].shape[1] + 1)).reshape(1, -1)
            ) 
            timesteps[-1][timesteps[-1] >= (max_len + 1)] = max_len

            tlen = n_o[-1].shape[1]
            tlen_gen = o_gen[-1].shape[1]

            n_o[-1] = np.concatenate([np.zeros((1, max_len - tlen, obs_dim)), n_o[-1]], axis=1)
            o_gen[-1] = np.concatenate([np.zeros((1, max_len - tlen_gen, obs_dim)), o_gen[-1]], axis=1)
            if config_dict["OBS_NORMALIZE"]:
                obs_mean, obs_std = obs_offline_mean_list[l], obs_offline_std_list[l]
                n_o[-1] = (n_o[-1] - obs_mean) / obs_std
                o_gen[-1] = (o_gen[-1] - obs_mean) / obs_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            a_gen[-1] = np.concatenate([np.ones((1, max_len - tlen_gen, act_dim)) * -10., a_gen[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)

            label.append(l)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
            mask_gen.append(np.concatenate([np.zeros((1, max_len - tlen_gen)), np.ones((1, tlen_gen))], axis=1))
        batch_online_inds = np.random.choice(
            np.arange(num_online_total_trajs),
            size=int(batch_size/2),
            replace=False,
        )
        for s in range(int(batch_size/2)):
            traj = online_mix_data[batch_online_inds[s]]
            l = online_oppo_labels[batch_online_inds[s]]
            index_gen = np.random.choice(all_online_indexes[l-5])  
            traj_gen = online_mix_data[index_gen]

            n_o.append(traj['next_observations'][:max_len].reshape(1, -1, obs_dim))    
            a.append(traj['actions'][:max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][:max_len].reshape(1, -1, 1))
            o_gen.append(traj_gen['observations'][:max_len].reshape(1, -1, obs_dim))  
            a_gen.append(traj_gen['actions'][:max_len].reshape(1, -1, act_dim))
            timesteps.append(
                np.arange(1, (n_o[-1].shape[1] + 1)).reshape(1, -1)
            ) 
            timesteps[-1][timesteps[-1] >= (max_len + 1)] = max_len

            tlen = n_o[-1].shape[1]
            tlen_gen = o_gen[-1].shape[1]

            n_o[-1] = np.concatenate([np.zeros((1, max_len - tlen, obs_dim)), n_o[-1]], axis=1)
            o_gen[-1] = np.concatenate([np.zeros((1, max_len - tlen_gen, obs_dim)), o_gen[-1]], axis=1)
            if config_dict["OBS_NORMALIZE"]:
                obs_mean, obs_std = obs_offline_mean_list[0], obs_offline_std_list[0]
                n_o[-1] = (n_o[-1] - obs_mean) / obs_std
                o_gen[-1] = (o_gen[-1] - obs_mean) / obs_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            a_gen[-1] = np.concatenate([np.ones((1, max_len - tlen_gen, act_dim)) * -10., a_gen[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)

            label.append(l)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
            mask_gen.append(np.concatenate([np.zeros((1, max_len - tlen_gen)), np.ones((1, tlen_gen))], axis=1))
        all_shuffle_indexes = np.arange(batch_size)
        np.random.shuffle(all_shuffle_indexes)
        n_o = [n_o[i] for i in all_shuffle_indexes]
        a = [a[i] for i in all_shuffle_indexes]
        r = [r[i] for i in all_shuffle_indexes]
        label = [label[i] for i in all_shuffle_indexes]
        o_gen = [o_gen[i] for i in all_shuffle_indexes]
        a_gen = [a_gen[i] for i in all_shuffle_indexes]

        n_o = torch.from_numpy(np.concatenate(n_o, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        label = torch.from_numpy(np.array(label)).to(dtype=torch.long, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        o_gen = torch.from_numpy(np.concatenate(o_gen, axis=0)).to(dtype=torch.float32, device=device)
        a_gen = torch.from_numpy(np.concatenate(a_gen, axis=0)).to(dtype=torch.float32, device=device)
        mask_gen = torch.from_numpy(np.concatenate(mask_gen, axis=0)).to(device=device)

        return n_o, a, r, label, timesteps, mask, o_gen, a_gen, mask_gen

    return fn