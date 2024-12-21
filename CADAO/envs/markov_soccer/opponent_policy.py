import os
import numpy as np
import torch
from .ppo import PPO

from .networks import policy

def get_all_oppo_policies(oppo_policy, oppo_index, device):
    all_oppo_policies = []
    for oppo_name in oppo_policy:
        if isinstance(oppo_name, tuple):
            if oppo_name[-1] in ["3", "4"]:
                p = RLPolicy(device, oppo_name[1])
            else:
                p = policy(12,5)
                p.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), oppo_name[1])))
                p = p.to(device)
        elif 'SnatchAttackPolicy' in oppo_name:
            p = SnatchAttackPolicy(pos=oppo_index, device=device)
        elif 'SnatchEvadePolicy' in oppo_name:
            p = SnatchEvadePolicy(pos=oppo_index, device=device)
        elif 'GuardAttackPolicy' in oppo_name:
            p = GuardAttackPolicy(pos=oppo_index, device=device)
        elif 'GuardEvadePolicy' in oppo_name:
            p = GuardEvadePolicy(pos=oppo_index, device=device)
        elif 'ChaseAttackPolicy' in oppo_name:
            p = ChaseAttackPolicy(pos=oppo_index, device=device)
        elif 'ChaseEvadePolicy' in oppo_name:
            p = ChaseEvadePolicy(pos=oppo_index, device=device)
        all_oppo_policies.append(p)
    return all_oppo_policies


class SnatchAttackPolicy:
    def __init__(self, noise=0.1, pos=0, device='cpu'):
        self.noise = noise
        self.pos = pos
        self.device = device
        self.goal1 = np.array([4,1]) if pos == 0 else np.array([0,1])
        self.goal2 = np.array([4,2]) if pos == 0 else np.array([0,2])
        self.goal3 = np.array([0,1]) if pos == 0 else np.array([4,1])
        self.goal4 = np.array([0,2]) if pos == 0 else np.array([4,2])
    
    def __call__(self, rel_state):
        '''
        An offensive policy that takes the relative state as input and outputs the action probability distribution. This policy makes every effort to put the ball into its goal no matter what the opponent does. And we have the noise parameter to randomly choose a different action. We should move legally, which means x and y coordinates should be in [0,4] and [0,3] respectively.
        ::param rel_state: relative state of the game, which is a 12-dim vector. Among them, Ball_me, Ball_opponent, Goal1_me, Goal2_me, Goal1_opponent and Goal2_opponent use 2 dims in sequence. And each 2 dims represent the x and y coordinates respectively.
        ::return: action probability distribution, where the index kUp = 0, kDown = 1, kLeft = 2, kRight = 3, kStand = 4
        '''
        rel_state_ = rel_state.cpu().numpy()
        ball_me = rel_state_[0:2]
        ball_opponent = rel_state_[2:4]
        Goal1_me = rel_state_[4:6]
        Goal2_me = rel_state_[6:8]
        Goal3_opponent = rel_state_[8:10]
        Goal4_opponent = rel_state_[10:12]
        
        my_pos = self.goal1 - Goal1_me
        my_pos_after = np.array([my_pos for _ in range(5)])
        action_mask = np.ones(5)
        my_pos_after[0] += np.array([0,1])
        my_pos_after[1] += np.array([0,-1])
        my_pos_after[2] += np.array([-1,0])
        my_pos_after[3] += np.array([1,0])
        my_pos_after[4] += np.array([0,0])
        for i in range(5):
            if my_pos_after[i][0] < 0 or my_pos_after[i][0] > 4 or my_pos_after[i][1] < 0 or my_pos_after[i][1] > 3:
                action_mask[i] = 0
        
        action_prob = np.zeros(5)
        if ball_me[0] == 0 and ball_me[1] == 0:
            if (Goal1_me[0] == 0 and Goal1_me[1] == 0) or (Goal2_me[0] == 0 and Goal2_me[1] == 0):
                # if we have the ball and reach the goal, we end the game
                if self.pos == 0:
                    action_prob[3] = 1
                else:
                    action_prob[2] = 1
                return torch.tensor(action_prob, dtype=torch.float32, device=self.device)
            # * Attack: if we have the ball, we should move it to the goal
            G1_dist = np.zeros(5)
            G1_dist[0] = np.sqrt(np.sum(np.square(Goal1_me - np.array([0,1]))))
            G1_dist[1] = np.sqrt(np.sum(np.square(Goal1_me - np.array([0,-1]))))
            G1_dist[2] = np.sqrt(np.sum(np.square(Goal1_me - np.array([-1,0]))))
            G1_dist[3] = np.sqrt(np.sum(np.square(Goal1_me - np.array([1,0]))))
            G1_dist[4] = np.sqrt(np.sum(np.square(Goal1_me - np.array([0,0]))))
            G1_dist[action_mask == 0] = np.inf
            G2_dist = np.zeros(5)
            G2_dist[0] = np.sqrt(np.sum(np.square(Goal2_me - np.array([0,1]))))
            G2_dist[1] = np.sqrt(np.sum(np.square(Goal2_me - np.array([0,-1]))))
            G2_dist[2] = np.sqrt(np.sum(np.square(Goal2_me - np.array([-1,0]))))
            G2_dist[3] = np.sqrt(np.sum(np.square(Goal2_me - np.array([1,0]))))
            G2_dist[4] = np.sqrt(np.sum(np.square(Goal2_me - np.array([0,0]))))
            G2_dist[action_mask == 0] = np.inf
            All_dist = np.concatenate((G1_dist, G2_dist))
            a = np.argmin(All_dist)
            if a >= 5: a -= 5
        else:
            # * Snatch: if we don't have the ball, we should move to the ball
            B_dist = np.zeros(5)
            B_dist[0] = np.sqrt(np.sum(np.square(ball_me - np.array([0,1]))))
            B_dist[1] = np.sqrt(np.sum(np.square(ball_me - np.array([0,-1]))))
            B_dist[2] = np.sqrt(np.sum(np.square(ball_me - np.array([-1,0]))))
            B_dist[3] = np.sqrt(np.sum(np.square(ball_me - np.array([1,0]))))
            B_dist[4] = np.sqrt(np.sum(np.square(ball_me - np.array([0,0]))))
            B_dist[action_mask == 0] = np.inf
            a = np.argmin(B_dist)
        
        action_prob[a] = 1 - self.noise
        num_legal_actions = np.sum(action_mask)
        for i in range(5):
            if action_mask[i] == 1 and i != a:
                action_prob[i] = self.noise / (num_legal_actions -1)
        
        return torch.tensor(action_prob, dtype=torch.float32, device=self.device)


class SnatchEvadePolicy:
    def __init__(self, noise=0.1, pos=0, device='cpu'):
        self.noise = noise
        self.pos = pos
        self.device = device
        self.goal1 = np.array([4,1]) if pos == 0 else np.array([0,1])
        self.goal2 = np.array([4,2]) if pos == 0 else np.array([0,2])
        self.goal3 = np.array([0,1]) if pos == 0 else np.array([4,1])
        self.goal4 = np.array([0,2]) if pos == 0 else np.array([4,2])
    
    def __call__(self, rel_state):
        '''
        A defensive policy that takes the relative state as input and outputs the action probability distribution. This policy makes every effort to prevent the opponent from putting the ball into its goal no matter what the opponent does. And we have the noise parameter to randomly choose a different action. We should move legally, which means x and y coordinates should be in [0,4] and [0,3] respectively.
        ::param rel_state: relative state of the game, which is a 12-dim vector. Among them, Ball_me, Ball_opponent, Goal1_me, Goal2_me, Goal1_opponent and Goal2_opponent use 2 dims in sequence. And each 2 dims represent the x and y coordinates respectively.
        ::return: action probability distribution, where the index kUp = 0, kDown = 1, kLeft = 2, kRight = 3, kStand = 4
        '''
        rel_state_ = rel_state.cpu().numpy()
        ball_me = rel_state_[0:2]
        ball_opponent = rel_state_[2:4]
        Goal1_me = rel_state_[4:6]
        Goal2_me = rel_state_[6:8]
        Goal3_opponent = rel_state_[8:10]
        Goal4_opponent = rel_state_[10:12]

        my_pos = self.goal1 - Goal1_me
        my_pos_after = np.array([my_pos for _ in range(5)])
        action_mask = np.ones(5)
        my_pos_after[0] += np.array([0,1])
        my_pos_after[1] += np.array([0,-1])
        my_pos_after[2] += np.array([-1,0])
        my_pos_after[3] += np.array([1,0])
        my_pos_after[4] += np.array([0,0])
        for i in range(5):
            if my_pos_after[i][0] < 0 or my_pos_after[i][0] > 4 or my_pos_after[i][1] < 0 or my_pos_after[i][1] > 3:
                action_mask[i] = 0

        action_prob = np.zeros(5)
        if ball_me[0] == 0 and ball_me[1] == 0:
            if (Goal1_me[0] == 0 and Goal1_me[1] == 0) or (Goal2_me[0] == 0 and Goal2_me[1] == 0):
                # if we have the ball and reach the goal, we end the game
                if self.pos == 0:
                    action_prob[3] = 1
                else:
                    action_prob[2] = 1
                return torch.tensor(action_prob, dtype=torch.float32, device=self.device)
            # * Evade: if we have the ball, we should move it as far away from the opponent as possible.
            O_dist = np.zeros(5)
            O_dist[0] = np.sqrt(np.sum(np.square(ball_opponent + np.array([0,1]))))
            O_dist[1] = np.sqrt(np.sum(np.square(ball_opponent + np.array([0,-1]))))
            O_dist[2] = np.sqrt(np.sum(np.square(ball_opponent + np.array([-1,0]))))
            O_dist[3] = np.sqrt(np.sum(np.square(ball_opponent + np.array([1,0]))))
            O_dist[4] = np.sqrt(np.sum(np.square(ball_opponent + np.array([0,0]))))
            O_dist[action_mask == 0] = -np.inf
            a = np.argmax(O_dist)
        else:
            # * Snatch: if we don't have the ball, we should move to the ball
            B_dist = np.zeros(5)
            B_dist[0] = np.sqrt(np.sum(np.square(ball_me - np.array([0,1]))))
            B_dist[1] = np.sqrt(np.sum(np.square(ball_me - np.array([0,-1]))))
            B_dist[2] = np.sqrt(np.sum(np.square(ball_me - np.array([-1,0]))))
            B_dist[3] = np.sqrt(np.sum(np.square(ball_me - np.array([1,0]))))
            B_dist[4] = np.sqrt(np.sum(np.square(ball_me - np.array([0,0]))))
            B_dist[action_mask == 0] = np.inf
            a = np.argmin(B_dist)
        
        action_prob[a] = 1 - self.noise
        num_legal_actions = np.sum(action_mask)
        for i in range(5):
            if action_mask[i] == 1 and i != a:
                action_prob[i] = self.noise / (num_legal_actions -1)
        
        return torch.tensor(action_prob, dtype=torch.float32, device=self.device)


class GuardAttackPolicy:
    def __init__(self, noise=0.1, pos=0, device='cpu'):
        self.noise = noise
        self.pos = pos
        self.device = device
        self.goal1 = np.array([4,1]) if pos == 0 else np.array([0,1])
        self.goal2 = np.array([4,2]) if pos == 0 else np.array([0,2])
        self.goal3 = np.array([0,1]) if pos == 0 else np.array([4,1])
        self.goal4 = np.array([0,2]) if pos == 0 else np.array([4,2])
    
    def __call__(self, rel_state):
        rel_state_ = rel_state.cpu().numpy()
        ball_me = rel_state_[0:2]
        ball_opponent = rel_state_[2:4]
        Goal1_me = rel_state_[4:6]
        Goal2_me = rel_state_[6:8]
        Goal3_opponent = rel_state_[8:10]
        Goal4_opponent = rel_state_[10:12]
        
        my_pos = self.goal1 - Goal1_me
        Goal3_me = self.goal3 - my_pos
        Goal4_me = self.goal4 - my_pos
        
        my_pos_after = np.array([my_pos for _ in range(5)])
        action_mask = np.ones(5)
        my_pos_after[0] += np.array([0,1])
        my_pos_after[1] += np.array([0,-1])
        my_pos_after[2] += np.array([-1,0])
        my_pos_after[3] += np.array([1,0])
        my_pos_after[4] += np.array([0,0])
        for i in range(5):
            if my_pos_after[i][0] < 0 or my_pos_after[i][0] > 4 or my_pos_after[i][1] < 0 or my_pos_after[i][1] > 3:
                action_mask[i] = 0
        
        action_prob = np.zeros(5)
        if ball_me[0] == 0 and ball_me[1] == 0:
            if (Goal1_me[0] == 0 and Goal1_me[1] == 0) or (Goal2_me[0] == 0 and Goal2_me[1] == 0):
                # if we have the ball and reach the goal, we end the game
                if self.pos == 0:
                    action_prob[3] = 1
                else:
                    action_prob[2] = 1
                return torch.tensor(action_prob, dtype=torch.float32, device=self.device)
            # * Attack: if we have the ball, we should move it to the goal
            G1_dist = np.zeros(5)
            G1_dist[0] = np.sqrt(np.sum(np.square(Goal1_me - np.array([0,1]))))
            G1_dist[1] = np.sqrt(np.sum(np.square(Goal1_me - np.array([0,-1]))))
            G1_dist[2] = np.sqrt(np.sum(np.square(Goal1_me - np.array([-1,0]))))
            G1_dist[3] = np.sqrt(np.sum(np.square(Goal1_me - np.array([1,0]))))
            G1_dist[4] = np.sqrt(np.sum(np.square(Goal1_me - np.array([0,0]))))
            G1_dist[action_mask == 0] = np.inf
            G2_dist = np.zeros(5)
            G2_dist[0] = np.sqrt(np.sum(np.square(Goal2_me - np.array([0,1]))))
            G2_dist[1] = np.sqrt(np.sum(np.square(Goal2_me - np.array([0,-1]))))
            G2_dist[2] = np.sqrt(np.sum(np.square(Goal2_me - np.array([-1,0]))))
            G2_dist[3] = np.sqrt(np.sum(np.square(Goal2_me - np.array([1,0]))))
            G2_dist[4] = np.sqrt(np.sum(np.square(Goal2_me - np.array([0,0]))))
            G2_dist[action_mask == 0] = np.inf
            All_dist = np.concatenate((G1_dist, G2_dist))
            a = np.argmin(All_dist)
            if a >= 5: a -= 5
        else:
            # * Guard: if we don't have the ball, we should guard our goal
            if (Goal3_me[0] == 0 and Goal3_me[1] == 0):
                action_prob[0] = 1
                return torch.tensor(action_prob, dtype=torch.float32, device=self.device)
            elif (Goal4_me[0] == 0 and Goal4_me[1] == 0):
                action_prob[1] = 1
                return torch.tensor(action_prob, dtype=torch.float32, device=self.device)
            G3_dist = np.zeros(5)
            G3_dist[0] = np.sqrt(np.sum(np.square(Goal3_me - np.array([0,1]))))
            G3_dist[1] = np.sqrt(np.sum(np.square(Goal3_me - np.array([0,-1]))))
            G3_dist[2] = np.sqrt(np.sum(np.square(Goal3_me - np.array([-1,0]))))
            G3_dist[3] = np.sqrt(np.sum(np.square(Goal3_me - np.array([1,0]))))
            G3_dist[4] = np.sqrt(np.sum(np.square(Goal3_me - np.array([0,0]))))
            G3_dist[action_mask == 0] = np.inf
            G4_dist = np.zeros(5)
            G4_dist[0] = np.sqrt(np.sum(np.square(Goal4_me - np.array([0,1]))))
            G4_dist[1] = np.sqrt(np.sum(np.square(Goal4_me - np.array([0,-1]))))
            G4_dist[2] = np.sqrt(np.sum(np.square(Goal4_me - np.array([-1,0]))))
            G4_dist[3] = np.sqrt(np.sum(np.square(Goal4_me - np.array([1,0]))))
            G4_dist[4] = np.sqrt(np.sum(np.square(Goal4_me - np.array([0,0]))))
            G4_dist[action_mask == 0] = np.inf
            All_dist = np.concatenate((G3_dist, G4_dist))
            a = np.argmin(All_dist)
            if a >= 5: a -= 5
        
        action_prob[a] = 1 - self.noise
        num_legal_actions = np.sum(action_mask)
        for i in range(5):
            if action_mask[i] == 1 and i != a:
                action_prob[i] = self.noise / (num_legal_actions -1)
        
        return torch.tensor(action_prob, dtype=torch.float32, device=self.device)


class GuardEvadePolicy:
    def __init__(self, noise=0.1, pos=0, device='cpu'):
        self.noise = noise
        self.pos = pos
        self.device = device
        self.goal1 = np.array([4,1]) if pos == 0 else np.array([0,1])
        self.goal2 = np.array([4,2]) if pos == 0 else np.array([0,2])
        self.goal3 = np.array([0,1]) if pos == 0 else np.array([4,1])
        self.goal4 = np.array([0,2]) if pos == 0 else np.array([4,2])
    
    def __call__(self, rel_state):
        rel_state_ = rel_state.cpu().numpy()
        ball_me = rel_state_[0:2]
        ball_opponent = rel_state_[2:4]
        Goal1_me = rel_state_[4:6]
        Goal2_me = rel_state_[6:8]
        Goal3_opponent = rel_state_[8:10]
        Goal4_opponent = rel_state_[10:12]
        
        my_pos = self.goal1 - Goal1_me
        Goal3_me = self.goal3 - my_pos
        Goal4_me = self.goal4 - my_pos
        
        my_pos_after = np.array([my_pos for _ in range(5)])
        action_mask = np.ones(5)
        my_pos_after[0] += np.array([0,1])
        my_pos_after[1] += np.array([0,-1])
        my_pos_after[2] += np.array([-1,0])
        my_pos_after[3] += np.array([1,0])
        my_pos_after[4] += np.array([0,0])
        for i in range(5):
            if my_pos_after[i][0] < 0 or my_pos_after[i][0] > 4 or my_pos_after[i][1] < 0 or my_pos_after[i][1] > 3:
                action_mask[i] = 0
        
        action_prob = np.zeros(5)
        if ball_me[0] == 0 and ball_me[1] == 0:
            if (Goal1_me[0] == 0 and Goal1_me[1] == 0) or (Goal2_me[0] == 0 and Goal2_me[1] == 0):
                # if we have the ball and reach the goal, we end the game
                if self.pos == 0:
                    action_prob[3] = 1
                else:
                    action_prob[2] = 1
                return torch.tensor(action_prob, dtype=torch.float32, device=self.device)
            # * Evade: if we have the ball, we should move it as far away from the opponent as possible.
            O_dist = np.zeros(5)
            O_dist[0] = np.sqrt(np.sum(np.square(ball_opponent + np.array([0,1]))))
            O_dist[1] = np.sqrt(np.sum(np.square(ball_opponent + np.array([0,-1]))))
            O_dist[2] = np.sqrt(np.sum(np.square(ball_opponent + np.array([-1,0]))))
            O_dist[3] = np.sqrt(np.sum(np.square(ball_opponent + np.array([1,0]))))
            O_dist[4] = np.sqrt(np.sum(np.square(ball_opponent + np.array([0,0]))))
            O_dist[action_mask == 0] = -np.inf
            a = np.argmax(O_dist)
        else:
            # * Guard: if we don't have the ball, we should guard our goal
            if (Goal3_me[0] == 0 and Goal3_me[1] == 0):
                action_prob[0] = 1
                return torch.tensor(action_prob, dtype=torch.float32, device=self.device)
            elif (Goal4_me[0] == 0 and Goal4_me[1] == 0):
                action_prob[1] = 1
                return torch.tensor(action_prob, dtype=torch.float32, device=self.device)
            G3_dist = np.zeros(5)
            G3_dist[0] = np.sqrt(np.sum(np.square(Goal3_me - np.array([0,1]))))
            G3_dist[1] = np.sqrt(np.sum(np.square(Goal3_me - np.array([0,-1]))))
            G3_dist[2] = np.sqrt(np.sum(np.square(Goal3_me - np.array([-1,0]))))
            G3_dist[3] = np.sqrt(np.sum(np.square(Goal3_me - np.array([1,0]))))
            G3_dist[4] = np.sqrt(np.sum(np.square(Goal3_me - np.array([0,0]))))
            G3_dist[action_mask == 0] = np.inf
            G4_dist = np.zeros(5)
            G4_dist[0] = np.sqrt(np.sum(np.square(Goal4_me - np.array([0,1]))))
            G4_dist[1] = np.sqrt(np.sum(np.square(Goal4_me - np.array([0,-1]))))
            G4_dist[2] = np.sqrt(np.sum(np.square(Goal4_me - np.array([-1,0]))))
            G4_dist[3] = np.sqrt(np.sum(np.square(Goal4_me - np.array([1,0]))))
            G4_dist[4] = np.sqrt(np.sum(np.square(Goal4_me - np.array([0,0]))))
            G4_dist[action_mask == 0] = np.inf
            All_dist = np.concatenate((G3_dist, G4_dist))
            a = np.argmin(All_dist)
            if a >= 5: a -= 5
        
        action_prob[a] = 1 - self.noise
        num_legal_actions = np.sum(action_mask)
        for i in range(5):
            if action_mask[i] == 1 and i != a:
                action_prob[i] = self.noise / (num_legal_actions -1)
        
        return torch.tensor(action_prob, dtype=torch.float32, device=self.device)


class ChaseAttackPolicy:
    def __init__(self, noise=0.1, pos=0, device='cpu'):
        self.noise = noise
        self.pos = pos
        self.device = device
        self.goal1 = np.array([4,1]) if pos == 0 else np.array([0,1])
        self.goal2 = np.array([4,2]) if pos == 0 else np.array([0,2])
        self.goal3 = np.array([0,1]) if pos == 0 else np.array([4,1])
        self.goal4 = np.array([0,2]) if pos == 0 else np.array([4,2])
    
    def __call__(self, rel_state):
        rel_state_ = rel_state.cpu().numpy()
        ball_me = rel_state_[0:2]
        ball_opponent = rel_state_[2:4]
        Goal1_me = rel_state_[4:6]
        Goal2_me = rel_state_[6:8]
        Goal3_opponent = rel_state_[8:10]
        Goal4_opponent = rel_state_[10:12]

        my_pos = self.goal1 - Goal1_me
        opponent_me = ball_me - ball_opponent
        
        my_pos_after = np.array([my_pos for _ in range(5)])
        action_mask = np.ones(5)
        my_pos_after[0] += np.array([0,1])
        my_pos_after[1] += np.array([0,-1])
        my_pos_after[2] += np.array([-1,0])
        my_pos_after[3] += np.array([1,0])
        my_pos_after[4] += np.array([0,0])
        for i in range(5):
            if my_pos_after[i][0] < 0 or my_pos_after[i][0] > 4 or my_pos_after[i][1] < 0 or my_pos_after[i][1] > 3:
                action_mask[i] = 0

        action_prob = np.zeros(5)
        if ball_me[0] == 0 and ball_me[1] == 0:
            if (Goal1_me[0] == 0 and Goal1_me[1] == 0) or (Goal2_me[0] == 0 and Goal2_me[1] == 0):
                # if we have the ball and reach the goal, we end the game
                if self.pos == 0:
                    action_prob[3] = 1
                else:
                    action_prob[2] = 1
                return torch.tensor(action_prob, dtype=torch.float32, device=self.device)
            # * Attack: if we have the ball, we should move it to the goal
            G1_dist = np.zeros(5)
            G1_dist[0] = np.sqrt(np.sum(np.square(Goal1_me - np.array([0,1]))))
            G1_dist[1] = np.sqrt(np.sum(np.square(Goal1_me - np.array([0,-1]))))
            G1_dist[2] = np.sqrt(np.sum(np.square(Goal1_me - np.array([-1,0]))))
            G1_dist[3] = np.sqrt(np.sum(np.square(Goal1_me - np.array([1,0]))))
            G1_dist[4] = np.sqrt(np.sum(np.square(Goal1_me - np.array([0,0]))))
            G1_dist[action_mask == 0] = np.inf
            G2_dist = np.zeros(5)
            G2_dist[0] = np.sqrt(np.sum(np.square(Goal2_me - np.array([0,1]))))
            G2_dist[1] = np.sqrt(np.sum(np.square(Goal2_me - np.array([0,-1]))))
            G2_dist[2] = np.sqrt(np.sum(np.square(Goal2_me - np.array([-1,0]))))
            G2_dist[3] = np.sqrt(np.sum(np.square(Goal2_me - np.array([1,0]))))
            G2_dist[4] = np.sqrt(np.sum(np.square(Goal2_me - np.array([0,0]))))
            G2_dist[action_mask == 0] = np.inf
            All_dist = np.concatenate((G1_dist, G2_dist))
            a = np.argmin(All_dist)
            if a >= 5: a -= 5
        else:
            # * Chase: if we don't have the ball, we chase the opponent
            O_dist = np.zeros(5)
            O_dist[0] = np.sqrt(np.sum(np.square(opponent_me - np.array([0,1]))))
            O_dist[1] = np.sqrt(np.sum(np.square(opponent_me - np.array([0,-1]))))
            O_dist[2] = np.sqrt(np.sum(np.square(opponent_me - np.array([-1,0]))))
            O_dist[3] = np.sqrt(np.sum(np.square(opponent_me - np.array([1,0]))))
            O_dist[4] = np.sqrt(np.sum(np.square(opponent_me - np.array([0,0]))))
            O_dist[action_mask == 0] = np.inf
            a = np.argmin(O_dist)
        
        action_prob[a] = 1 - self.noise
        num_legal_actions = np.sum(action_mask)
        for i in range(5):
            if action_mask[i] == 1 and i != a:
                action_prob[i] = self.noise / (num_legal_actions -1)
        
        return torch.tensor(action_prob, dtype=torch.float32, device=self.device)


class ChaseEvadePolicy:
    def __init__(self, noise=0.1, pos=0, device='cpu'):
        self.noise = noise
        self.pos = pos
        self.device = device
        self.goal1 = np.array([4,1]) if pos == 0 else np.array([0,1])
        self.goal2 = np.array([4,2]) if pos == 0 else np.array([0,2])
        self.goal3 = np.array([0,1]) if pos == 0 else np.array([4,1])
        self.goal4 = np.array([0,2]) if pos == 0 else np.array([4,2])
    
    def __call__(self, rel_state):
        rel_state_ = rel_state.cpu().numpy()
        ball_me = rel_state_[0:2]
        ball_opponent = rel_state_[2:4]
        Goal1_me = rel_state_[4:6]
        Goal2_me = rel_state_[6:8]
        Goal3_opponent = rel_state_[8:10]
        Goal4_opponent = rel_state_[10:12]

        my_pos = self.goal1 - Goal1_me
        opponent_me = ball_me - ball_opponent
        
        my_pos_after = np.array([my_pos for _ in range(5)])
        action_mask = np.ones(5)
        my_pos_after[0] += np.array([0,1])
        my_pos_after[1] += np.array([0,-1])
        my_pos_after[2] += np.array([-1,0])
        my_pos_after[3] += np.array([1,0])
        my_pos_after[4] += np.array([0,0])
        for i in range(5):
            if my_pos_after[i][0] < 0 or my_pos_after[i][0] > 4 or my_pos_after[i][1] < 0 or my_pos_after[i][1] > 3:
                action_mask[i] = 0

        action_prob = np.zeros(5)
        if ball_me[0] == 0 and ball_me[1] == 0:
            if (Goal1_me[0] == 0 and Goal1_me[1] == 0) or (Goal2_me[0] == 0 and Goal2_me[1] == 0):
                # if we have the ball and reach the goal, we end the game
                if self.pos == 0:
                    action_prob[3] = 1
                else:
                    action_prob[2] = 1
                return torch.tensor(action_prob, dtype=torch.float32, device=self.device)
            # * Evade: if we have the ball, we should move it as far away from the opponent as possible.
            O_dist = np.zeros(5)
            O_dist[0] = np.sqrt(np.sum(np.square(ball_opponent + np.array([0,1]))))
            O_dist[1] = np.sqrt(np.sum(np.square(ball_opponent + np.array([0,-1]))))
            O_dist[2] = np.sqrt(np.sum(np.square(ball_opponent + np.array([-1,0]))))
            O_dist[3] = np.sqrt(np.sum(np.square(ball_opponent + np.array([1,0]))))
            O_dist[4] = np.sqrt(np.sum(np.square(ball_opponent + np.array([0,0]))))
            O_dist[action_mask == 0] = -np.inf
            a = np.argmax(O_dist)
        else:
            # * Chase: if we don't have the ball, we chase the opponent
            O_dist = np.zeros(5)
            O_dist[0] = np.sqrt(np.sum(np.square(opponent_me - np.array([0,1]))))
            O_dist[1] = np.sqrt(np.sum(np.square(opponent_me - np.array([0,-1]))))
            O_dist[2] = np.sqrt(np.sum(np.square(opponent_me - np.array([-1,0]))))
            O_dist[3] = np.sqrt(np.sum(np.square(opponent_me - np.array([1,0]))))
            O_dist[4] = np.sqrt(np.sum(np.square(opponent_me - np.array([0,0]))))
            O_dist[action_mask == 0] = np.inf
            a = np.argmin(O_dist)
        
        action_prob[a] = 1 - self.noise
        num_legal_actions = np.sum(action_mask)
        for i in range(5):
            if action_mask[i] == 1 and i != a:
                action_prob[i] = self.noise / (num_legal_actions -1)
        
        return torch.tensor(action_prob, dtype=torch.float32, device=self.device)


class RLPolicy:
    def __init__(self, device, param_path):
        self.net = PPO(12, 64, 5, device,
                    actor_lr=0., critic_lr=0.,)
        self.net.load_params(param_path)
        self.device = device
    
    def __call__(self, obs):
        obs = obs.cpu().numpy()
        act, _, _ = self.net.select_action(obs)
        return torch.tensor(act, dtype=torch.float32, device=self.device)