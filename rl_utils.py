from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from torch import nn


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] -
              cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes,task,env_seed):
    return_list = []
    mean = []
    std = []
    with tqdm(total=num_episodes, desc='Total_Epi %d' % num_episodes, ncols=150) as pbar:
        for i_episode in range(num_episodes):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [],
                               'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            mean.append(np.mean(return_list))
            std.append(np.std(return_list))
            lay1_prune_ratio, lay2_prune_ratio = agent.update(transition_dict, i_episode, num_episodes - 1)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (i_episode), 'return': '%.3f' % np.mean(return_list[-10:]),
                                  'lay1_ratio': '%.3f' % (lay1_prune_ratio * 100),
                                  'lay2_ratio': '%.3f' % (lay2_prune_ratio * 100),
                                  'total_ratio': '%.3f' % float((lay1_prune_ratio + lay2_prune_ratio) * 100 / 2)})
            pbar.update(1)
            if (i_episode+1) % 125 == 0:
                # torch.save(agent.actor.state_dict(), f'actor_param_{i_episode+1}.pkl') # save state_dict() instead of mask
                # torch.save(agent, f'agent_param_{i_episode+1}.pkl')
                torch.save({'actor.state_dict':agent.actor.state_dict(),
                            'mask1.state_dict':agent.actor.mask1,
                            'mask2.state_dict':agent.actor.mask2},f'Experiment_One/{task}/actor_param_episode{i_episode+1}_{env_seed}.pkl')

    return return_list, mean, std


def train_off_policy_agent(epi_num, env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration{epi_num}/{i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (
                            num_episodes / 10 * i + i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


# =================================constructure pruning ================================================================
# pick important nuron lists
def import_neutron_list(model, device):
    prune_index = 0
    imp_dim0 = torch.Tensor().to(device)
    imp_dim1 = torch.Tensor().to(device)
    for param in model.parameters():
        if prune_index < 2:
            tem_dim1 = torch.norm(param, p=2, dim=1)  # 前神经元输入 4 64 64 2
            imp_dim1 = torch.cat((imp_dim1, tem_dim1), dim=0)
        if 0 < prune_index < 3:
            tem_dim0 = torch.norm(param, p=2, dim=0)  # 后神经元输出
            imp_dim0 = torch.cat((imp_dim0, tem_dim0), dim=0)
        prune_index += 1
    # sum lambda (input weight * output weight)
    # l2 += torch.sum(torch.mul(args.finetune_input_lambda * imp_dim1, args.finetune_output_lambda * imp_dim0))
    # diffrent lambda for sum lambda (input weight * output weight)
    layer1_input, layer2_input = torch.chunk(imp_dim1, chunks=2, dim=0)
    layer1_output, layer2_output = torch.chunk(imp_dim0, chunks=2, dim=0)
    imp_nur = torch.mul(layer1_input, layer1_output)
    out_nur = torch.mul(layer2_input, layer2_output)
    return imp_nur, out_nur


# constructure pruning loss
def con_loss(model, device):
    loss = 0
    imp_nur, out_nur = import_neutron_list(model, device)  # get important list
    loss += torch.sum(imp_nur)
    loss += torch.sum(out_nur)
    return loss


# =================================group lasso pruning ================================================================
# group lasso nurons
def group_neutron_list(model, device):
    prune_index = 0
    gro_nur = torch.Tensor().to(device)
    for param in model.parameters():
        if prune_index < 2:
            tem_dim = torch.norm(param, p=2, dim=1)  # 神经元 4 64 64 2 因为用了norm-2，所以不用abs(weight)
            gro_nur = torch.cat((gro_nur, tem_dim), dim=0)
        prune_index += 1
    imp_nur, out_nur = torch.chunk(gro_nur, chunks=2, dim=0)
    return imp_nur, out_nur


# group lasso pruning loss
def gro_loss(model, device):
    loss = 0
    imp_nur, out_nur = group_neutron_list(model, device)  # get important list
    loss += torch.sum(imp_nur)
    loss += torch.sum(out_nur)
    return loss


# =================================random pruning ================================================================
# group random nurons
def random_neutron_list(model, device):
    prune_index = 0
    gro_nur = torch.Tensor().to(device)
    for param in model.parameters():
        if prune_index < 2:
            tem_dim = torch.norm(param, p=2, dim=1)  # 神经元 4 64 64 2
            gro_nur = torch.cat((gro_nur, tem_dim), dim=0)
        prune_index += 1
    imp_nur, out_nur = torch.chunk(gro_nur, chunks=2, dim=0)
    imp_nur, out_nur = torch.ones_like(imp_nur).to(device), torch.ones_like(out_nur).to(device)
    return imp_nur, out_nur


# =================================l1 lasso pruning ================================================================
# l1 nurons
def l1_weight_list(model, device):
    weight_list = torch.Tensor().to(device)
    # noted that def(lasso_loss): for param in model.parameters() written
    # param.weight can be only inherit nn.Linear; model.parameters means to weights, not need to param.weight
    for param in model.modules():
        if isinstance(param, nn.Linear):
            weight_list = torch.cat((weight_list.view(-1), param.weight.view(-1)))
    return weight_list


# l1 pruning loss
def lasso_loss(model, device):
    loss = 0
    weight_list = l1_weight_list(model, device)  # get important list
    tem = torch.norm(weight_list, p=1, dim=0)
    loss += torch.sum(tem)
    # or loop each layer parameters for l1 loss
    # l1 = 0
    # for param in model.parameters():
    #     tem = torch.norm(param, p=1, dim=1)  # dim=1 or dim=0 are both same
    #     l1 += torch.sum(tem)
    # l1 = l1.item()
    return loss


def neutron_mask_list(masks, device):
    imp_dim0 = torch.Tensor().to(device)
    imp_dim1 = torch.Tensor().to(device)
    for mask_index, mask in enumerate(masks):
        if mask_index < 2:
            tem_dim1 = torch.norm(mask, p=2, dim=1)  # 前神经元输入 4 64 64 2
            imp_dim1 = torch.cat((imp_dim1, tem_dim1), dim=0)
        if 0 < mask_index < 3:
            tem_dim0 = torch.norm(mask, p=2, dim=0)  # 后神经元输出
            imp_dim0 = torch.cat((imp_dim0, tem_dim0), dim=0)
    layer1_input, layer2_input = torch.chunk(imp_dim1, chunks=2, dim=0)
    layer1_output, layer2_output = torch.chunk(imp_dim0, chunks=2, dim=0)
    imp_mask = torch.mul(layer1_input, layer1_output)
    out_mask = torch.mul(layer2_input, layer2_output)
    return imp_mask, out_mask

def train_on_policy_uncon_agent(env, agent, num_episodes,task,env_seed):
    return_list = []
    mean = []
    std = []
    with tqdm(total=int(num_episodes), desc='Epi %d' % int(num_episodes), ncols=180) as pbar:
        for i_episode in range(int(num_episodes)):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [],
                               'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            mean.append(np.mean(return_list))
            std.append(np.std(return_list))
            lay1_prune_ratio, lay2_prune_ratio, fc1_prune_ratio, fc2_prune_ratio, fc3_prune_ratio = agent.update(transition_dict, i_episode, int(num_episodes))
            if (i_episode) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (i_episode), 'return': '%.2f' % np.mean(return_list[-10:]),
                                  'lay1_rat': '%.2f' % (lay1_prune_ratio * 100),
                                  'lay2_rat': '%.2f' % (lay2_prune_ratio * 100),
                                  'fc1_rat': '%.2f' % (fc1_prune_ratio * 100),
                                  'fc2_rat': '%.2f' % (fc2_prune_ratio * 100),
                                  'fc3_rat': '%.2f' % (fc3_prune_ratio * 100),
                                  'total_rat': '%.2f' % float((fc1_prune_ratio + fc2_prune_ratio + fc3_prune_ratio) * 100 / 3)})
                # pbar.set_postfix({f'episode: {(i_episode + 1)}, '
                #                   f'return: {np.mean(return_list[-10:]):.3f}, '
                #                   f'lay1 : {lay1_prune_ratio:.3f}, '
                #                   f'lay2 : {lay2_prune_ratio:.3f}, '
                #                   f'total : {(lay1_prune_ratio+lay2_prune_ratio):.3f}'
                #                   })
            pbar.update(1)
            if (i_episode+1) % 125 == 0:
                torch.save({'actor.state_dict':agent.actor.state_dict(),
                            'mask1.state_dict':agent.actor.mask1,
                            'mask2.state_dict':agent.actor.mask2,
                            'mask3.state_dict':agent.actor.mask3},f'Experiment_One/{task}/actor_param_episode{i_episode+1}_{env_seed}.pkl')

    return return_list, mean, std

# =================================pops pruning ================================================================
# pops nurons
def pops_weight_list(model, device):
    weight_list = torch.Tensor().to(device)
    # noted that def(lasso_loss): for param in model.parameters() written
    # param.weight can be only inherit nn.Linear; model.parameters means to weights, not need to param.weight
    for param in model.modules():
        if isinstance(param, nn.Linear):
            weight_list = torch.cat((weight_list.view(-1), param.weight.view(-1)), dim=0)
    return weight_list