# Author(s):    zhuan
# Date:         2025/2/21
# Time:         11:06

# Your code starts here
# ================================================================
# import gym
# env = gym.make('CartPole-v1',render_mode="human")
# env.reset()
# for _ in range(1000):
#     env.render()
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     print(observation)
# env.close()
# ==================================================================
# from gym import envs
# env_specs = envs.registry.all()
# envs_ids = [env_spec.id for env_spec in env_specs]
# print(envs_ids)
# =====================================================================
# import gym
# import numpy as np
#
#
# class SimpleAgent:
#     def __init__(self, env):
#         pass
#
#     def decide(self, observation):  # 决策
#         position, velocity = observation
#         lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
#                  0.3 * (position + 0.9) ** 4 - 0.008)
#         ub = -0.07 * (position + 0.38) ** 2 + 0.07
#         if lb < velocity < ub:
#             action = 2
#         else:
#             action = 0
#         return action  # 返回动作
#
#     def learn(self, *args):  # 学习
#         pass
#
#
# def play(env, agent, render=False, train=False):
#     episode_reward = 0.  # 记录回合总奖励，初始化为0
#     observation = env.reset()  # 重置游戏环境，开始新回合
#     while True:  # 不断循环，直到回合结束
#         if render:  # 判断是否显示
#             env.render()  # 显示图形界面，图形界面可以用 env.close() 语句关闭
#         action = agent.decide(observation)
#         next_observation, reward, done, _ = env.step(action)  # 执行动作
#         episode_reward += reward  # 收集回合奖励
#         if train:  # 判断是否训练智能体
#             agent.learn(observation, action, reward, done)  # 学习
#         if done:  # 回合结束，跳出循环
#             break
#         observation = next_observation
#     return episode_reward  # 返回回合总奖励
#
#
# env = gym.make('MountainCar-v0')
# env.seed(3)  # 设置随机种子，让结果可复现
# agent = SimpleAgent(env)
# print('观测空间 = {}'.format(env.observation_space))
# print('动作空间 = {}'.format(env.action_space))
# print('观测范围 = {} ~ {}'.format(env.observation_space.low,
#                                   env.observation_space.high))
# print('动作数 = {}'.format(env.action_space.n))
#
# episode_reward = play(env, agent, render=True)
# print('回合奖励 = {}'.format(episode_reward))
#
# episode_rewards = [play(env, agent) for _ in range(100)]
# print('平均回合奖励 = {}'.format(np.mean(episode_rewards)))
# =======================================================================
# import gym
# env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
# env = CliffWalkingWapper(env)
# agent = QLearning(
#     state_dim=env.observation_space.n,
#     action_dim=env.action_space.n,
#     learning_rate=cfg.policy_lr,
#     gamma=cfg.gamma,)
# rewards = []
# ma_rewards = [] # moving average reward
# for i_ep in range(cfg.train_eps): # train_eps: 训练的最大episodes数
#     ep_reward = 0  # 记录每个episode的reward
#     state = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
#     while True:
#         action = agent.choose_action(state)  # 根据算法选择一个动作
#         next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
#         agent.update(state, action, reward, next_state, done)  # Q-learning算法更新
#         state = next_state  # 存储上一个观察值
#         ep_reward += reward
#         if done:
#             break
#     rewards.append(ep_reward)
#     if ma_rewards:
#         ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
#     else:
#         ma_rewards.append(ep_reward)
#     print("Episode:{}/{}: reward:{:.1f}".format(i_ep+1, cfg.train_eps,ep_reward))
# =====================================================================================
import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """ 伯努利多臂老虎机,输入K表示拉杆个数 """
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # 随机生成K个0～1的数,作为拉动每根拉杆的获奖
        # 概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.K = K

    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未
        # 获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 维护一个列表,记录每一步的动作
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔

    def update_regret(self, k):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k

class UCB(Solver):
    """ UCB算法,继承Solver类 """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class ThompsonSampling(Solver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k


if __name__ == '__main__':

    np.random.seed(1)  # 设定随机种子,使实验具有可重复性
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
          (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

    # np.random.seed(1)
    # epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    # epsilon_greedy_solver.run(5000)
    # print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    # plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

    # np.random.seed(0)
    # epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    # epsilon_greedy_solver_list = [
    #     EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
    # ]
    # epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    # for solver in epsilon_greedy_solver_list:
    #     solver.run(5000)
    #
    # plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

    # np.random.seed(1)
    # decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    # decaying_epsilon_greedy_solver.run(5000)
    # print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
    # plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

    # np.random.seed(1)
    # coef = [1,1e-1, 1e-2, 1e-3, 1e-4, 1e-5]  # 控制不确定性比重的系数
    # UCB_solver_list = [UCB(bandit_10_arm, coef[e]) for e in range(len(coef))]
    # UCE_solver_names = ["coef={}".format(coef[e]) for e in range(len(coef))]
    # for solver in UCB_solver_list:
    #     solver.run(5000)
    # plot_results(UCB_solver_list, UCE_solver_names)
    # print('上置信界算法的累积懊悔为：', [UCB_solver_list[e].regret for e in range(len(coef))])

    np.random.seed(1)
    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(5000)
    print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])