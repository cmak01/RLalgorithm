import sys  
import os  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import numpy as np  
import random  
from collections import deque  
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  
# 确保能找到 src 目录  
sys.path.append("..")  
from src.grid_world import GridWorld  
  
# ==================== Hyperparameters ====================  
BATCH_SIZE = 64  
LR = 0.001  
GAMMA = 0.99            # 折扣因子  
EPSILON_START = 1.0     # 初始探索率  
EPSILON_END = 0.01      # 最小探索率  
EPSILON_DECAY = 0.995   # 探索率衰减  
MEMORY_CAPACITY = 10000 # 经验回放池大小  
TARGET_UPDATE = 10      # 目标网络更新频率 (episodes)  
NUM_EPISODES = 500      # 训练总回合数  
HIDDEN_DIM = 128        # 隐藏层神经元数量  
# =========================================================  
  
class ReplayBuffer:  
    def __init__(self, capacity):  
        self.buffer = deque(maxlen=capacity)  
  
    def push(self, state, action, reward, next_state, done):  
        self.buffer.append((state, action, reward, next_state, done))  
  
    def sample(self, batch_size):  
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))  
        return np.stack(state), action, reward, np.stack(next_state), done  
  
    def __len__(self):  
        return len(self.buffer)  
  
class QNetwork(nn.Module):  
    def __init__(self, input_dim, output_dim):  
        super(QNetwork, self).__init__()  
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)  
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)  
        self.fc3 = nn.Linear(HIDDEN_DIM, output_dim)  
  
    def forward(self, x):  
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x))  
        return self.fc3(x)  
  
class DQNAgent:  
    def __init__(self, env):  
        self.env = env  
        self.state_dim = 2 # (x, y)  
        self.action_dim = len(env.action_space)  
        self.action_space = env.action_space # List of tuples [(0,1), ...]  
          
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
          
        # Q Networks  
        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(self.device)  
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)  
        self.target_net.load_state_dict(self.policy_net.state_dict())  
        self.target_net.eval()  
          
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)  
        self.memory = ReplayBuffer(MEMORY_CAPACITY)  
        self.epsilon = EPSILON_START  
  
    def select_action(self, state):  
        # Epsilon-Greedy Action Selection  
        if random.random() > self.epsilon:  
            with torch.no_grad():  
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  
                q_values = self.policy_net(state_tensor)  
                action_idx = q_values.argmax().item()  
        else:  
            action_idx = random.randrange(self.action_dim)  
          
        return action_idx  
  
    def update(self):  
        if len(self.memory) < BATCH_SIZE:  
            return  
  
        # Sample from replay buffer  
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)  
  
        state = torch.FloatTensor(state).to(self.device)  
        next_state = torch.FloatTensor(next_state).to(self.device)  
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)  
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)  
  
        # Compute Q(s_t, a)  
        q_values = self.policy_net(state)  
        q_value = q_values.gather(1, action)  
  
        # Compute V(s_{t+1}) for target  
        # Off-policy: use max Q from target network for next state  
        with torch.no_grad():  
            next_q_values = self.target_net(next_state)  
            next_q_value = next_q_values.max(1)[0].unsqueeze(1)  
            expected_q_value = reward + (1 - done) * GAMMA * next_q_value  
  
        # Loss (MSE)  
        loss = nn.MSELoss()(q_value, expected_q_value)  
  
        self.optimizer.zero_grad()  
        loss.backward()  
        self.optimizer.step()  
  
    def update_epsilon(self):  
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)  
  
def train():  
    env = GridWorld()  
    agent = DQNAgent(env)  
      
    print(f"Start Training DQN on {agent.device}...")  
    print(f"Grid Size: {env.env_size}, Target: {env.target_state}")  
  
    for i_episode in range(NUM_EPISODES):  
        state, _ = env.reset()  
        # 确保 state 是 numpy array 或者 list，如果是 tuple 保持不变，  
        # 但存入 buffer 时我们希望统一格式，这里 state 是 tuple (x, y)  
          
        total_reward = 0  
        done = False  
          
        while not done:  
            # 1. 选择动作 (返回的是 index)  
            action_idx = agent.select_action(state)  
              
            # 2. 将 index 转换为环境需要的 tuple 动作 (例如 (0, 1))  
            action_tuple = agent.action_space[action_idx]  
              
            # 3. 执行动作  
            next_state, reward, done, info = env.step(action_tuple)  
              
            # 4. 存入经验回放  
            agent.memory.push(state, action_idx, reward, next_state, done)  
              
            # 5. 模型学习  
            agent.update()  
              
            state = next_state  
            total_reward += reward  
  
            # 限制单回合最大步数，防止训练初期无限循环  
            if len(env.traj) > 100:   
                break  
  
        # 更新 Target Network  
        if i_episode % TARGET_UPDATE == 0:  
            agent.target_net.load_state_dict(agent.policy_net.state_dict())  
          
        agent.update_epsilon()  
  
        if (i_episode + 1) % 50 == 0:  
            print(f"Episode: {i_episode+1}/{NUM_EPISODES}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")  
  
    print("Training Complete.")  
    return agent, env  
  
def visualize_results(agent, env):  
    """提取 Q 表并使用环境的 add_policy 和 add_state_values 进行可视化"""  
      
    # 初始化矩阵  
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))  
    state_values = np.zeros(env.num_states)  
      
    agent.policy_net.eval()  
      
    # 遍历所有状态，通过网络预测 Q 值  
    for y in range(env.env_size[1]):  
        for x in range(env.env_size[0]):  
            state_idx = y * env.env_size[0] + x  
            state_tuple = (x, y)  
              
            state_tensor = torch.FloatTensor(state_tuple).unsqueeze(0).to(agent.device)  
            with torch.no_grad():  
                q_values = agent.policy_net(state_tensor).cpu().numpy().flatten()  
              
            # 记录该状态的最大 Q 值 (Value)  
            state_values[state_idx] = np.max(q_values)  
              
            # 生成 Policy (Greedy): 将最大 Q 值对应的动作设为 1，其余为 0  
            # 或者使用 Softmax 获得概率分布，这里为了清晰展示使用 Greedy  
            best_action = np.argmax(q_values)  
            policy_matrix[state_idx][best_action] = 1.0  
  
    # 1. 设置渲染器  
    env.reset()  
    env.render(animation_interval=0.01) # 快速初始化画面  
      
    # 2. 添加学到的 Policy 箭头  
    print("Visualizing Policy...")  
    env.add_policy(policy_matrix)  
      
    # 3. 添加学到的 State Values 数值  
    print("Visualizing State Values...")  
    env.add_state_values(state_values, precision=1)  
      
    # 4. 演示一次贪婪策略的行走  
    print("Demonstrating learned path...")  
    state, _ = env.reset()  
    done = False  
    path_len = 0  
    while not done and path_len < 30:  
        env.render(animation_interval=0.3)  
        action_idx = agent.select_action(state) # 此时 epsilon 很小，接近贪婪  
        action_tuple = agent.action_space[action_idx]  
        state, _, done, _ = env.step(action_tuple)  
        path_len += 1  
      
    env.render()  
    plt.show() # 保持窗口打开  
    input("Press Enter to close...")  
  
if __name__ == "__main__":  
    trained_agent, env = train()  
      
    # 为了演示效果，将 epsilon 设为 0 (完全贪婪)  
    trained_agent.epsilon = 0.0  
    visualize_results(trained_agent, env)  
