import random

import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from rl_utils import con_loss, import_neutron_list
import os
import pickle
import joblib  # para for-loop


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim, bias=False)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim, bias=False)

        # constructure pruning
        self.mask1 = torch.ones_like(self.fc2.weight[0]).to(device)
        self.mask2 = torch.ones_like(self.fc3.weight[0]).to(device)

    def forward(self, x):
        x = torch.mul(F.relu(self.fc1(x)), self.mask1)
        x = torch.mul(F.relu(self.fc2(x)), self.mask2)
        return F.softmax(self.fc3(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, prune_lambda,preset_ratio):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.prune_lambda = prune_lambda  # 剪枝调参的 lambda
        self.preset_ratio = preset_ratio

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    # consructure pruning pruner 4 64 64 2 100-> 90 95 97
    # (0-100,0-80) 25 50 75 90 (0-50), [40 10], [30,20] [90, 70],
    # 80 () (0-100) 100 70 80 (95, 90, 87)
    # 1. DQN DDPG PPO PPO-PRUN(90%)(4)
    # 2.  PP0 PPO-L1 PPO-GROUP
    # 3. 20 40 60 90
    # 4. LAMDA 0.1 0.2
    def pruner(self, model, prune_ratio, device):  #constrcture pruning
        imp_list = torch.Tensor().to(device)
        imp_nur, out_nur = import_neutron_list(model, device)  # get important list
        imp_list = torch.cat((imp_list, imp_nur, out_nur), dim=0)
        lay1_nnz, lay2_nnz = model.mask1.sum().item(), model.mask2.sum().item()
        lay1_total, lay2_total = model.mask1.size().numel(), model.mask2.size().numel()
        lay1_prune_ratio, lay2_prune_ratio = (lay1_total-lay1_nnz)/lay1_total, (lay2_total - lay2_nnz)/lay2_total
        # thresold for remaining 2% neuron of each layer
        if lay1_prune_ratio < 0.97 and lay2_prune_ratio < 0.97:
            threshold = imp_list.quantile(q=prune_ratio)
            model.mask1 = torch.gt(imp_nur, threshold).to(device)
            model.mask2 = torch.gt(out_nur, threshold).to(device)
            lay1_nnz, lay2_nnz = model.mask1.sum().item(), model.mask2.sum().item()
            lay1_prune_ratio, lay2_prune_ratio = (lay1_total - lay1_nnz) / lay1_total, (lay2_total - lay2_nnz) / lay2_total
        if lay1_prune_ratio < 0.97 and lay2_prune_ratio > 0.97:
            # print(f'layer2 neuron is {lay2_nnz} can not pruned, prune layer1')
            threshold = imp_nur.quantile(q=(2 * prune_ratio - lay2_prune_ratio))  # noted that layer2 cannot be pruned
            model.mask1 = torch.gt(imp_nur, threshold).to(device)
            lay1_nnz, lay2_nnz = model.mask1.sum().item(), model.mask2.sum().item()
            lay1_prune_ratio, lay2_prune_ratio = (lay1_total - lay1_nnz) / lay1_total, (lay2_total - lay2_nnz) / lay2_total
        # if lay1_nnz < lay1_total * 0.02 and lay2_nnz > lay2_total * 0.02:
            # print(f'layer1 neuron is {lay1_nnz} can not pruned, prune layer2')
        if lay1_prune_ratio > 0.97 and lay2_prune_ratio < 0.97:
            threshold = out_nur.quantile(q=2 * prune_ratio - lay1_prune_ratio) # noted that layer1 cannot be pruned
            model.mask2 = torch.gt(out_nur, threshold).to(device)
            lay1_nnz, lay2_nnz = model.mask1.sum().item(), model.mask2.sum().item()
            lay1_prune_ratio, lay2_prune_ratio = (lay1_total - lay1_nnz) / lay1_total, (lay2_total - lay2_nnz) / lay2_total
        return lay1_prune_ratio, lay2_prune_ratio  # for print pruning ratio


    def update(self, transition_dict, i_episode, total_episode):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions'])).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8962235
        pops_ratio = self.preset_ratio + (0 - self.preset_ratio) * (1 - (i_episode - 0) / (total_episode * 1))
        # lay1_prune_ratio, lay2_prune_ratio = self.pruner(model=self.actor, prune_ratio= (i_episode/total_episode) * 0.92, device=self.device)
        lay1_prune_ratio, lay2_prune_ratio = self.pruner(model=self.actor, prune_ratio=pops_ratio, device=self.device)

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断

            penalty_loss = con_loss(model=self.actor, device=self.device)  # construture loss
            # if prune_lambda >=1, will be error in quantile < 0
            actor_loss = torch.mean(-torch.min(surr1, surr2)) + self.prune_lambda * penalty_loss  # PPO损失函数 + 结构化稀疏
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        return lay1_prune_ratio, lay2_prune_ratio  # for print pruning ratio


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v1'
run_times = 4
actor_lr = 5e-04
critic_lr = 1e-03
num_episodes = 500   # safe prune ratio < 1 / num_episodes
hidden_dim = 128
gamma = 0.98
lmbda = 0.9
epochs = 10
eps = 0.2
env = gym.make(env_name)


def ppo_run(env_seed,prune_lambda,preset_ratio):
    task = 'ppo_con'
    prune_lambda = prune_lambda
    # torch.cuda.set_device(env_seed % 2)  # 设置GPU运行位置
    torch.cuda.set_device(random.randint(2,6))  # 设置GPU运行位置
    # torch.cuda.set_device(0)  # 设置GPU运行位置
    env.seed(env_seed)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device, prune_lambda,preset_ratio)

    return_list, mean, std = rl_utils.train_on_policy_agent(env, agent, num_episodes,task,env_seed)
    # if not os.path.isdir('Experiment_One/ppo_con'):
    #     os.makedirs('Experiment_One/ppo_con')
    # joblib.dump(return_list, filename=f'Experiment_One/ppo_con/return_lambda_{prune_lambda}_preset_ratio{preset_ratio}_seed{env_seed}.pth')
    # joblib.dump(mean, filename=f'Experiment_One/ppo_con/mean_lambda_{prune_lambda}_preset_ratio{preset_ratio}_seed{env_seed}.pth')
    # joblib.dump(std, filename=f'Experiment_One/ppo_con/std_lambda_{prune_lambda}_preset_ratio{preset_ratio}_seed{env_seed}.pth')
    # means.append(mean)
    # stds.append(std)


if __name__=="__main__":
    # parrel running
    from joblib import Parallel, delayed
    import time
    # con_lamda_list = [1e-8, 5e-8, 5e-7, 1e-7, 1e-6, 5e-6], preset_ratio = 0.96     # ppo_consture lambda settings
    con_lamda_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]  # ppo_consture lambda settings
    preset_ratio_list = [0.5,0.6,0.7,0.8]
    start_time = time.time()
    # Parallel(n_jobs=8)(delayed(ppo_run)(i+10,prune_lambda, preset_ratio) for i in range(run_times) for prune_lambda in con_lamda_list for preset_ratio in preset_ratio_list)
    ppo_run(0,1e-6,0.7)
    end_time = time.time()
    print(f'used_time is {end_time - start_time}')

