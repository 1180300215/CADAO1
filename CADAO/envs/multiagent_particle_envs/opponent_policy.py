import numpy as np
from .ppo import PPO


OPPO_OBS_INDEX = {
    "landmark1": [0, 2],
    "landmark2": [2, 4],
    "agent1": [4, 6],
    "agent2": [6, 8],
}


def get_all_oppo_policies(env, oppo_policy, oppo_index, num_step, device):
    all_oppo_policies = []
    for oppo_id in range(len(oppo_policy)):
        if isinstance(oppo_policy[oppo_id], tuple):
            policy = eval(oppo_policy[oppo_id][0] + '(env,' + str(oppo_index) +', "' + oppo_policy[oppo_id][1] +'", "' + str(device) + '" )')
        else:
            policy = eval(oppo_policy[oppo_id] + "(env," + str(oppo_index) + "," + str(num_step) + ")")
        all_oppo_policies.append(policy)
    return all_oppo_policies


class Policy(object):
    def __init__(self, env, agent_index, num_step=100):
        self.env = env
        self.agent_index = agent_index
        self.num_step = num_step
        self.cur_step = 0
    
    def rollout_one_step(self):
        u_other = np.zeros(self.env.world.dim_p)
        p_force_ = [u_other for _ in range(len(self.env.world.entities))]
        # we rollout each action for 1 step to see the next observation
        all_obs = []
        for i in range(5):
            u_ = np.zeros(self.env.world.dim_p)
            if i == 1: u_[0] = -1.0
            if i == 2: u_[0] = 1.0
            if i == 3: u_[1] = -1.0
            if i == 4: u_[1] = 1.0
            u_ *= 5.0
            p_force_[self.agent_index] = u_
            p_force_ = self.env.world.apply_environment_force(p_force_)
            p_vel_ = self.env.world.agents[self.agent_index].state.p_vel * (1 - self.env.world.damping)
            p_vel_ += (p_force_[self.agent_index] / self.env.world.agents[self.agent_index].mass) * self.env.world.dt
            p_pos_ = self.env.world.agents[self.agent_index].state.p_pos + p_vel_ * self.env.world.dt
            
            entity_pos_ = []
            for entity in self.env.world.landmarks:
                entity_pos_.append(entity.state.p_pos - p_pos_)
            other_pos_ = []
            for other in self.env.world.agents:
                if other is self.env.world.agents[self.agent_index]: continue
                other_pos_.append(other.state.p_pos - p_pos_)
            obs_i = np.concatenate(entity_pos_ + other_pos_)
            all_obs.append(obs_i)
        
        return all_obs
    
    def reset(self):
        raise NotImplementedError()
    
    def action(self, obs):
        raise NotImplementedError()

class StaticPolicy(Policy):
    # NOTE: only used for testing the environment
    def __init__(self, env, agent_index, num_step=100):
        super(StaticPolicy, self).__init__(env, agent_index, num_step)
    def action(self, obs):
        if self.env.discrete_action_input:
            u = 0
        else:
            u = np.zeros(5) # 5-d because of no-move action
            u[0] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

class RandomPolicy(Policy):
    # NOTE: only used for testing the environment
    def __init__(self, env, agent_index, num_step=100):
        super(RandomPolicy, self).__init__(env, agent_index, num_step)
    def action(self, obs):
        a = np.random.randint(0,5)
        if self.env.discrete_action_input:
            u = np.copy(a)
        else:
            u = np.zeros(5) # 5-d because of no-move action
            u[a] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

class FixOnePolicy(Policy):
    def __init__(self, env, agent_index, num_step=100):
        super(FixOnePolicy, self).__init__(env, agent_index, num_step)
        self.reset()
        
    def reset(self):
        self.target = 'landmark' + str(np.random.randint(1,3))
    
    def action(self, obs):
        if self.cur_step == self.num_step:
            self.reset()
            self.cur_step = 0
        target_delta_pos_index = OPPO_OBS_INDEX[self.target]
        
        next_obs = self.rollout_one_step()
        dist = []
        for i in range(5):
            delta_pos = next_obs[i][target_delta_pos_index[0]:target_delta_pos_index[1]]
            dist.append(np.sqrt(np.sum(np.square(delta_pos))))
        a = np.argmin(dist)
        
        if self.env.discrete_action_input:
            u = np.copy(a)
        else:
            u = np.zeros(5) # 5-d because of no-move action
            u[a] += 1.0
        self.cur_step += 1
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

class ChaseOnePolicy(Policy):
    def __init__(self, env, agent_index, num_step=100):
        super(ChaseOnePolicy, self).__init__(env, agent_index, num_step)
        self.reset()
        
    def reset(self):
        self.target = 'agent' + str(np.random.randint(1,3))
        
    def action(self, obs):
        if self.cur_step == self.num_step:
            self.reset()
            self.cur_step = 0
        target_delta_pos_index = OPPO_OBS_INDEX[self.target]
        
        next_obs = self.rollout_one_step()
        dist = []
        for i in range(5):
            delta_pos = next_obs[i][target_delta_pos_index[0]:target_delta_pos_index[1]]
            dist.append(np.sqrt(np.sum(np.square(delta_pos))))
        a = np.argmin(dist)
        
        if self.env.discrete_action_input:
            u = np.copy(a)
        else:
            u = np.zeros(5) # 5-d because of no-move action
            u[a] += 1.0
        self.cur_step += 1
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

class MiddlePolicy(Policy):
    def __init__(self, env, agent_index, num_step=100):
        super(MiddlePolicy, self).__init__(env, agent_index, num_step)
        self.target_1 = 'landmark1'
        self.target_2 = 'landmark2'
    def action(self, obs):
        target_1_delta_pos_index = OPPO_OBS_INDEX[self.target_1]
        target_2_delta_pos_index = OPPO_OBS_INDEX[self.target_2]

        next_obs = self.rollout_one_step()
        dist = []
        for i in range(5):
            delta_1_pos = next_obs[i][target_1_delta_pos_index[0]:target_1_delta_pos_index[1]]
            delta_2_pos = next_obs[i][target_2_delta_pos_index[0]:target_2_delta_pos_index[1]]
            delta_pos = (delta_1_pos+delta_2_pos)/2
            dist.append(np.sqrt(np.sum(np.square(delta_pos))))
        a = np.argmin(dist)
        
        if self.env.discrete_action_input:
            u = np.copy(a)
        else:
            u = np.zeros(5) # 5-d because of no-move action
            u[a] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

class BouncePolicy(Policy):
    def __init__(self, env, agent_index, num_step=100):
        super(BouncePolicy, self).__init__(env, agent_index, num_step)
        self.target = 'landmark1'
    def action(self, obs):
        test_delta_pos_index = OPPO_OBS_INDEX[self.target]
        test_delta_pos = obs[test_delta_pos_index[0]:test_delta_pos_index[1]]
        test_delta_pos_value = np.sqrt(np.sum(np.square(test_delta_pos)))
        if test_delta_pos_value < 1.2 * self.env.world.landmarks[0].size:
            if self.target == 'landmark1':
                self.target = 'landmark2'
            elif self.target == 'landmark2':
                self.target = 'landmark1'
        target_delta_pos_index = OPPO_OBS_INDEX[self.target]
        
        next_obs = self.rollout_one_step()
        dist = []
        for i in range(5):
            delta_pos = next_obs[i][target_delta_pos_index[0]:target_delta_pos_index[1]]
            dist.append(np.sqrt(np.sum(np.square(delta_pos))))
        a = np.argmin(dist)
        
        if self.env.discrete_action_input:
            u = np.copy(a)
        else:
            u = np.zeros(5) # 5-d because of no-move action
            u[a] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

class FixThreePolicy(Policy):
    def __init__(self, env, agent_index, num_step=100):
        super(FixThreePolicy, self).__init__(env, agent_index, num_step)
        self.reset()
        
    def reset(self):
        self.target = np.random.choice([
            "landmark1",
            "landmark2",
            "middle",
        ])
    
    def action(self, obs):
        if self.cur_step == self.num_step:
            self.reset()
            self.cur_step = 0
        target_1_delta_pos_index = OPPO_OBS_INDEX['landmark1']
        target_2_delta_pos_index = OPPO_OBS_INDEX['landmark2']

        next_obs = self.rollout_one_step()
        dist = []
        for i in range(5):
            delta_1_pos = next_obs[i][target_1_delta_pos_index[0]:target_1_delta_pos_index[1]]
            delta_2_pos = next_obs[i][target_2_delta_pos_index[0]:target_2_delta_pos_index[1]]
            if self.target == "landmark1":
                delta_pos = delta_1_pos
            elif self.target == "landmark2":
                delta_pos = delta_2_pos
            elif self.target == "middle":
                delta_pos = (delta_1_pos+delta_2_pos)/2
            dist.append(np.sqrt(np.sum(np.square(delta_pos))))
        a = np.argmin(dist)
        
        if self.env.discrete_action_input:
            u = np.copy(a)
        else:
            u = np.zeros(5) # 5-d because of no-move action
            u[a] += 1.0
        self.cur_step += 1
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

class ChaseBouncePolicy(Policy):
    def __init__(self, env, agent_index, num_step=100):
        super(ChaseBouncePolicy, self).__init__(env, agent_index, num_step)
        self.target = 'agent1'
    def action(self, obs):
        test_delta_pos_index = OPPO_OBS_INDEX[self.target]
        test_delta_pos = -obs[test_delta_pos_index[0]:test_delta_pos_index[1]]
        test_delta_pos_value = np.sqrt(np.sum(np.square(test_delta_pos)))
        if test_delta_pos_value < 2.0 * self.env.world.agents[1].size:
            if self.target == 'agent1':
                self.target = 'agent2'
            elif self.target == 'agent2':
                self.target = 'agent1'
        target_delta_pos_index = OPPO_OBS_INDEX[self.target]
        
        next_obs = self.rollout_one_step()
        dist = []
        for i in range(5):
            delta_pos = next_obs[i][target_delta_pos_index[0]:target_delta_pos_index[1]]
            dist.append(np.sqrt(np.sum(np.square(delta_pos))))
        a = np.argmin(dist)
        
        if self.env.discrete_action_input:
            u = np.copy(a)
        else:
            u = np.zeros(5) # 5-d because of no-move action
            u[a] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

class RLPolicy(Policy):
    def __init__(self, env, agent_index, param_path, device):
        super(RLPolicy, self).__init__(env, agent_index)
        state_dim = env.observation_space[agent_index].shape[0]
        action_dim = env.action_space[env.n-1].n
        hidden_dim = 128
        self.net = PPO(state_dim, hidden_dim, action_dim, device,
                    actor_lr=0., critic_lr=0.,)
        self.net.load_params(param_path)
    
    def action(self, obs):
        act, _, _ = self.net.select_action(obs)
        return act