# DRL进阶

## TRPO

### 理论流

#### main

1. 背景介绍
2. 推导策略目标
3. 优化近似
4. 共轭梯度求解$x=H^{-1}g$
5. 引入广义优势估计A
6. 引入线性搜索更新$\theta$

#### def

1. 背景：运用策略梯度A+时序差分C时，训练不够稳定上下起伏过大。TRPO（trust region policy optimization）提出一种信任区域策略优化方法
2. 策略目标：<img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311143941640.png" alt="image-20250311143941640" style="zoom:30%;" />

![image-20250311144057804](C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311144057804.png)

新的策略-旧的策略的变式=时序差分优势函数  $E[\sum\gamma^t[A_t]]$

<img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311144913333.png" alt="image-20250311144913333" style="zoom:25%;" />

只要$J(\theta')-J(\theta)>=0$,就能保证策略性能单调递增。
近似一下$J(\theta')$得到$L(\theta')$:

![image-20250311145244334](C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311145244334.png)

用KL散度来衡量两个新旧策略分布的远近程度：

![image-20250311145348251](C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311145348251.png)

3. 优化：尝试泰勒展开一下，并且缩放去掉无关项：

![image-20250311145432414](C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311145432414.png)

整理得：

<img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311145508831.png" alt="image-20250311145508831" style="zoom:25%;" />

其中：

<img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311145645914.png" alt="image-20250311145645914" style="zoom:50%;" />

g其实就是两个策略差对优势A的加权均值

<img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311145811417.png" alt="image-20250311145811417" style="zoom:50%;" />

H其实就是KL散度对策略变量$\theta$的Hessian矩阵，维度为$n\times n$

利用KKT条件得到：

$F(\theta)=g^T(\theta-\theta_k)+\lambda(\frac{1}{2}(\theta-\theta_k)^TH(\theta-\theta_k)-\delta)$

求解得到$\lambda$=<img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311150619435.png" alt="image-20250311150619435" style="zoom:25%;" />

<img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311150655162.png" alt="image-20250311150655162" style="zoom:40%;" />

令$x= H^{-1}g,{得到\theta_{k+1}=\theta_k+\sqrt{\frac{2\delta}{x^THx}}x}$

4. 共轭梯度求解x

![image-20250311153018891](C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311153018891.png)

迭代法求解

<img src="D:\Pinkman\Desktop\projDRL\微信图片_20250313144945.jpg" alt="微信图片_20250313144945" style="zoom:50%;" />

i. $\alpha 推导证明$

直接求解推导可以得到$\alpha$的推导式

<img src="D:\Pinkman\Desktop\projDRL\微信图片_20250313144929.jpg" alt="微信图片_20250313144929" style="zoom:50%;" />

证明$r_k=p_k$

![image-20250311154628140](C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311154628140.png)

证明$r_{k+1}^Tr_k=0$

<img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311154753847.png" alt="image-20250311154753847" style="zoom:33%;" />

ii.$r_{k}的递推式证明略，提示代入x_{k+1}=x_k+\alpha p_k和r_{k+1}=b-Ax_{k+1}$

iii.$\beta$的变式推导

<img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311155150050.png" alt="image-20250311155150050" style="zoom:25%;" />

5. 引入广义优势估计A

**广义优势估计**（Generalized Advantage Estimation，GAE）我们尚未得知如何估计优势函数。接下来我们简单介绍一下 GAE 的做法。根据多步时序差分的思想

![image-20250311152016140](C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311152016140.png)

GAE 将这些不同步数的优势估计进行指数加权平均：

<img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311152057428.png" alt="image-20250311152057428" style="zoom:200%;" />

$\lambda \in[0,1]$是在 GAE 中额外引入的一个超参数。

GAE算法可以转换为秦九韶算法
<img src="D:\Pinkman\Desktop\projDRL\微信图片_20250313145059.jpg" alt="微信图片_20250313145059" style="zoom:50%;" />

5. 引入线性搜索更新$\theta$

   具体来说，就是找到一个最小的非负整数i，使得按照：

<img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250311152705001.png" alt="image-20250311152705001" style="zoom:50%;" />

$\alpha\in[0,1]$是一个决定线性搜索长度的超参数。

### 代码流

#### main

1. 定义超参数，环境，种子，agent
2. 开始训练
3. 收集return_list计算回报平均值画图

```python
def main():
    env_name = None
    args = argparse.Namespace()
    args.cartpole_args = dict(
        num_episodes=500,
        hidden_dim=128,
        gamma=0.98,
        lmbda=0.95,
        critic_lr=1e-2,
        kl_constraint=0.0005,
        alpha=0.5,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu"),
        env_name='CartPole-v0',
        TRPO_type=TRPO,
    )
    args.pendulum_args = dict(
        num_episodes=2000,
        hidden_dim=128,
        gamma=0.9,
        lmbda=0.9,
        critic_lr=1e-2,
        kl_constraint=0.00005,
        alpha=0.5,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu"),
        env_name='Pendulum-v0',
        TRPO_type=TRPOContinuous,
    )
    train_trpo(**args.cartpole_args)
    train_trpo(**args.pendulum_args)
    
def train_trpo(hidden_dim, lmbda, kl_constraint, alpha, critic_lr, gamma, device, num_episodes, env_name,TRPO_type, **kwargs):
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    agent = TRPO_type(hidden_dim, env.observation_space, env.action_space, lmbda,
                      kl_constraint, alpha, critic_lr, gamma, device)
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()
```

#### def

1. agent定义

i. init,take_action,update方法

```python
class TRPOContinuous:
    """ 处理连续动作的TRPO算法 """

    def __init__(self, hidden_dim, state_space, action_space, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device):
        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.alpha = alpha
        self.device = device
        
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return [action.item()]
        
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        old_action_dists = torch.distributions.Normal(mu.detach(),
                                                      std.detach())
        old_log_probs = old_action_dists.log_prob(actions)
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.policy_learn(states, actions, old_action_dists, old_log_probs,
                          advantage)
```

ii. GAE计算方法

```python
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
```

iii. 策略更新方法

```python
    def policy_learn(self, states, actions, old_action_dists, old_log_probs,
                     advantage):
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                   old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        descent_direction = self.conjugate_gradient(obj_grad, states,
                                                    old_action_dists)
        Hd = self.hessian_matrix_vector_product(states, old_action_dists,
                                                descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint /
                              (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_probs,
                                    old_action_dists,
                                    descent_direction * max_coef)
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())
```

计算策略目标

```python
    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs,
                              actor):
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)
```

共轭梯度求解步长向量

```python
    def conjugate_gradient(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists,
                                                    p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x
```

计算Hx，采用隐式计算H的方法，计算复杂度由$O(n^2)$降低为$O(n)$,注意若x与$\theta$相关，则需要$Hx=\nabla(\nabla f\times x)-\nabla x\times \nabla f$

```python
    def hessian_matrix_vector_product(self,
                                      states,
                                      old_action_dists,
                                      vector,
                                      damping=0.1):
        mu, std = self.actor(states)
        new_action_dists = torch.distributions.Normal(mu, std)
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists,
                                                 new_action_dists))
        kl_grad = torch.autograd.grad(kl,
                                      self.actor.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        grad2_vector = torch.cat(
            [grad.contiguous().view(-1) for grad in grad2])
        return grad2_vector + damping * vector
```

线性搜索寻找合适放缩因子

```python
    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor)
        for i in range(15):
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists,
                                                     new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                 old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

```



2. 训练流程

```python
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
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
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list
```

3. 回报平均值计算方法

```python
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
```

关于中间值：窗口为9，计算累积数列，每8个数取均值滑动，500个值中有492个中间值，8个为前四个开始，后四个结束

关于开始值：设置窗口分别为[1 3 5 7], 取前窗口[i]个数取均值作为开始值，例如取1个数均值为开始值1，取3个数均值为开始值2

关于结束值：设置窗口分别为[1 3 5 7], 取后窗口[i]个数取均值作为开始值，例如取1个数均值为结束值1，取3个数均值为结束值2



## PPO

### 理论流

#### main

1. 背景介绍
2. 基于TRPO的策略函数的惩罚改进和截断改进
3. 引入广义优势函数GAE

#### def

1. TRPO计算太繁杂，PPO是一种简化版本

2. TRPO策略函数：
   <img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250312150408044.png" alt="image-20250312150408044" style="zoom:50%;" />

   （1）PPO惩罚：用拉格朗日乘数法直接将 KL 散度的限制放进了目标函数中，这就变成了一个无约束的优化问题，在迭代的过程中不断更新 KL 散度前的系数。

   <img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250312150433835.png" alt="image-20250312150433835" style="zoom:50%;" />
   <img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250312150504155.png" alt="image-20250312150504155" style="zoom:50%;" />
   （2）PPO-截断：更加直接，它在目标函数中进行限制，以保证新的参数和旧的参数的差距不会太大。大量实验表明，PPO-截断总是比 PPO-惩罚表现得更好。
   <img src="C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250312150603948.png" alt="image-20250312150603948" style="zoom:50%;" />
   ![image-20250312150626865](C:\Users\zhuan\AppData\Roaming\Typora\typora-user-images\image-20250312150626865.png)



### 代码流

#### main

1. 定义超参数，环境，种子，agent
2. 开始训练
3. 收集return_list计算回报平均值画图

```python
def main():
    args = argparse.Namespace()
    args.CartPole = dict(
        actor_lr=1e-3,
        critic_lr=1e-2,
        num_episodes=500,
        hidden_dim=128,
        gamma=0.98,
        lmbda=0.95,
        epochs=10,
        eps=0.2,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu"),

        env_name='CartPole-v0',
        PPO_type=PPO,
    )
    args.Pendulum = dict(
        actor_lr=1e-4,
        critic_lr=5e-3,
        num_episodes=2000,
        hidden_dim=128,
        gamma=0.9,
        lmbda=0.9,
        epochs=10,
        eps=0.2,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu"),

        env_name='Pendulum-v0',
        PPO_type=PPOContinuous,
    )
    train_ppo(**args.CartPole)
    train_ppo(**args.Pendulum)
    
def train_ppo(actor_lr,
              critic_lr,
              num_episodes,
              hidden_dim,
              gamma,
              lmbda,
              epochs,
              eps,
              device,
              env_name,
              PPO_type, **kwargs):
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    agent = PPO_type(env.observation_space, hidden_dim, env.action_space, actor_lr, critic_lr, lmbda,
                     epochs, eps, gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()
    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()
```

#### def

1. agent的update方法，是直接对策略函数反向传播流程，进行epochs次反向传播更新

```python
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
```



