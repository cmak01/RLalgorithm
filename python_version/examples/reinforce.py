import sys  
import os  

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 将上级目录加入 path 以便导入 src.grid_world  
sys.path.append("..")  
  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from torch.distributions import Categorical  
import matplotlib.pyplot as plt  
  
from src.grid_world import GridWorld  
from examples.arguments import args  
  
# ===========================  
# 1. 定义策略网络 (Policy Network)  
# ===========================  
class PolicyNet(nn.Module):  
    def __init__(self, state_dim, hidden_dim, action_dim):  
        super(PolicyNet, self).__init__()  
        self.fc1 = nn.Linear(state_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, action_dim)  
  
    def forward(self, x):  
        x = F.relu(self.fc1(x))  
        x = self.fc2(x)  
        return F.softmax(x, dim=1)  
  
# ===========================  
# 2. 定义 REINFORCE 算法  
# ===========================  
class REINFORCE:  
    def __init__(self, state_dim, action_dim, learning_rate=0.01, gamma=0.99):  
        self.policy_net = PolicyNet(state_dim, 128, action_dim)  
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)  
        self.gamma = gamma  
        self.log_probs = []    # 存储对数概率  
        self.rewards = []      # 存储单步奖励  
  
    def select_action(self, state_vec):  
        """  
        根据当前状态选择动作  
        state_vec: 形状为 (1, state_dim) 的 tensor  
        """  
        state_tensor = torch.FloatTensor(state_vec)  
        probs = self.policy_net(state_tensor)  
          
        # 创建类别分布并采样  
        m = Categorical(probs)  
        action_idx = m.sample()  
          
        # 存储该动作的 log_prob 用于后续梯度计算  
        self.log_probs.append(m.log_prob(action_idx))  
          
        return action_idx.item()  
  
    def store_reward(self, reward):  
        self.rewards.append(reward)  
  
    def update(self):  
        """  
        在一个 Episode 结束后更新策略网络  
        """  
        R = 0  
        policy_loss = []  
        returns = []  
  
        # 1. 计算折扣回报 (Discounted Returns)  
        # 从最后一步向前计算  
        for r in self.rewards[::-1]:  
            R = r + self.gamma * R  
            returns.insert(0, R)  
  
        returns = torch.tensor(returns)  
          
        # (可选) 对回报进行标准化，有助于训练稳定  
        if len(returns) > 1:  
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)  
  
        # 2. 计算损失函数: Loss = - sum(log_prob * G_t)  
        for log_prob, G_t in zip(self.log_probs, returns):  
            policy_loss.append(-log_prob * G_t)  
  
        # 3. 反向传播与优化  
        self.optimizer.zero_grad()  
        # 将列表中的 loss 求和（因为是梯度累加）  
        policy_loss = torch.stack(policy_loss).sum()  
        policy_loss.backward()  
        self.optimizer.step()  
  
        # 4. 清空本回合数据  
        self.log_probs = []  
        self.rewards = []  
  
# ===========================  
# 3. 辅助函数  
# ===========================  
def state_to_one_hot(state, env_size):  
    """  
    将 (x, y) 坐标转换为扁平化的 One-Hot 向量  
    """  
    x, y = state  
    # 假设 width = env_size[0]  
    idx = y * env_size[0] + x  
    total_states = env_size[0] * env_size[1]  
      
    vec = np.zeros(total_states)  
    vec[idx] = 1.0  
    return vec[np.newaxis, :] # 返回 (1, total_states)  
  
def get_full_policy_matrix(agent, env):  
    """  
    生成环境所需的策略矩阵 (num_states, num_actions) 用于可视化  
    """  
    total_states = env.env_size[0] * env.env_size[1]  
    policy_matrix = np.zeros((total_states, len(env.action_space)))  
      
    with torch.no_grad():  
        for y in range(env.env_size[1]):  
            for x in range(env.env_size[0]):  
                state = (x, y)  
                idx = y * env.env_size[0] + x  
                state_vec = state_to_one_hot(state, env.env_size)  
                state_tensor = torch.FloatTensor(state_vec)  
                  
                probs = agent.policy_net(state_tensor).numpy()[0]  
                policy_matrix[idx] = probs  
                  
    return policy_matrix  
  
# ===========================  
# 4. 主训练循环  
# ===========================  
if __name__ == "__main__":  
    # 初始化环境  
    # 注意：args 在 arguments.py 中定义，这里直接使用默认参数  
    env = GridWorld()  
      
    state_dim = env.num_states  
    action_dim = len(env.action_space)  
      
    # 初始化 Agent  
    agent = REINFORCE(state_dim, action_dim, learning_rate=0.002, gamma=0.99)  
      
    num_episodes = 5000  
    max_steps = 100  
      
    print(f"Start Training REINFORCE on {env.env_size[0]}x{env.env_size[1]} GridWorld...")  
      
    # 用于记录奖励曲线  
    episode_rewards = []  
  
    for i_episode in range(num_episodes):  
        state, _ = env.reset()  
        ep_reward = 0  
          
        for t in range(max_steps):  
            # 1. 状态预处理  
            state_vec = state_to_one_hot(state, env.env_size)  
              
            # 2. 选择动作 (返回 index)  
            action_idx = agent.select_action(state_vec)  
            action = env.action_space[action_idx] # 映射回 (dx, dy)  
              
            # 3. 执行动作  
            next_state, reward, done, _ = env.step(action)  
              
            # 4. 存储奖励  
            agent.store_reward(reward)  
              
            state = next_state  
            ep_reward += reward  
              
            if done:  
                break  
          
        # 5. 回合结束，更新网络  
        agent.update()  
        episode_rewards.append(ep_reward)  
          
        # 打印日志  
        if (i_episode + 1) % 100 == 0:  
            avg_reward = np.mean(episode_rewards[-100:])  
            print(f"Episode {i_episode+1}/{num_episodes}, Average Reward: {avg_reward:.2f}")  
  
    print("Training finished.")  
  
    # ===========================  
    # 5. 结果可视化  
    # ===========================  
    print("Visualizing Policy...")  
      
    # 重新重置环境用于画图  
    env.reset()  
    env.render(animation_interval=0.01) # 初始化 canvas  
  
    # 1. 获取并添加策略流场 (Policy Matrix)  
    policy_matrix = get_full_policy_matrix(agent, env)  
    env.add_policy(policy_matrix)  
      
    # 2. (可选) 添加简单的状态价值估计  
    # 由于 REINFORCE 是 Policy-based，没有直接学习 Value Function  
    # 这里我们简单用 Policy 概率分布的最大值作为 "置信度" 展示，或者全 0  
    # 为了演示 API，我们生成一个随机数或置信度  
    confidence_values = np.max(policy_matrix, axis=1)  
    env.add_state_values(confidence_values)  
      
    # 3. 演示一次贪婪策略 (Test Run)  
    state, _ = env.reset()  
    print("Testing learned policy...")  
    for t in range(50):  
        env.render(animation_interval=0.3)  
        state_vec = state_to_one_hot(state, env.env_size)  
        with torch.no_grad():  
            probs = agent.policy_net(torch.FloatTensor(state_vec))  
            action_idx = torch.argmax(probs).item() # 贪婪选择概率最大的动作  
          
        action = env.action_space[action_idx]  
        next_state, reward, done, _ = env.step(action)  
        state = next_state  
        if done:  
            print("Target Reached!")  
            break  

    env.render(animation_interval=0.3)          
    # 保持窗口打开  
    plt.ioff()  
    plt.show()  
