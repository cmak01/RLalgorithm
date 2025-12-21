import sys    
sys.path.append("..")    
from src.grid_world import GridWorld    
import numpy as np    
import time    
import matplotlib.pyplot as plt

class LinearSarsaAgent:
    def __init__(self, env, alpha=0.01, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = env.action_space
        self.num_actions = len(self.action_space)

        self.feature_dim = 6

        self.weights = np.zeros((self.num_actions, self.feature_dim))

    def get_features(self, state):
        x, y = state
        w, h = self.env.env_size
        norm_x = x / (w - 1)
        norm_y = y / (h - 1)
        features = np.array([
            1.0,
            norm_x,
            norm_y,
            norm_x ** 2,
            norm_y ** 2,
            norm_x * norm_y
        ])
        return features

    def get_q_value(self, state, action_idx):
        features = self.get_features(state)
        return np.dot(self.weights[action_idx], features)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action_idx = np.random.choice(self.num_actions)
        else:
            q_values = [self.get_q_value(state, a_idx) for a_idx in range(self.num_actions)]
            max_q = np.max(q_values)
            action_with_max_q = [i for i, q in enumerate(q_values) if q == max_q]
            action_idx = np.random.choice(action_with_max_q)

        return self.action_space[action_idx], action_idx

    def update(self, state, action_idx, reward, next_state, next_action_idx, done):
        q_current = self.get_q_value(state, action_idx)

        if done:
            target = reward
        else:
            q_next = self.get_q_value(next_state, next_action_idx)
            target = reward + self.gamma * q_next

        td_error = target - q_current
        features = self.get_features(state)
        self.weights[action_idx] += self.alpha * td_error * features

    def get_policy_matrix(self):
        num_states = self.env.num_states
        w, h = self.env.env_size
        policy_matrix = np.zeros((num_states, self.num_actions))

        for i in range(num_states):
            x = i % w
            y = i // w
            state = (x, y)

            q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
            best_action = np.argmax(q_values)
            policy_matrix[i, best_action] = 1.0

        return policy_matrix

    def get_value_matrix(self):
        num_states = self.env.num_states
        w, h = self.env.env_size
        values = np.zeros(num_states)

        for i in range(num_states):
            x = i % w
            y = i // w
            state = (x, y)
            q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
            values[i] = np.max(q_values)

        return values

def run_episode_with_policy(env, policy):
    """使用计算出的策略运行一次环境"""
    state, _ = env.reset()
    width = env.env_size[0]
    done = False
    steps = 0

    print("\n开始使用最优策略演示路径...")
    while not done and steps < 50: # 限制步数防止死循环
        env.render(animation_interval=0.5)

        # 获取当前状态的索引
        s_idx = state[1] * width + state[0]  
          
        # 根据策略选择动作  
        # policy[s_idx] 是一个概率数组，例如 [0, 1, 0, 0, 0]  
        action_idx = np.argmax(policy[s_idx])  
        action = env.action_space[action_idx]  
          
        next_state, reward, done, info = env.step(action)  
        print(f"Step: {steps}, State: {state} -> Action: {action} -> Next: {next_state}")  
          
        state = next_state  
        steps += 1  
      
    if done:  
        print("成功到达目标！")  
    else:  
        print("未能在限制步数内到达目标。")

if __name__ == "__main__":
    env = GridWorld()

    agent = LinearSarsaAgent(env, alpha=0.01, gamma=0.9, epsilon=0.1)

    num_episodes = 5000

    print(f"Start Training Sarsa with Function Approximation for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state, _ = env.reset()
        action, action_idx = agent.choose_action(state)

        done = False
        total_reward = 0

        while not done:
            next_state, reward, done, _ = env.step(action)
            
            next_action, next_action_idx = agent.choose_action(next_state)

            agent.update(state, action_idx, reward, next_state, next_action_idx, done)

            state = next_state
            action = next_action
            action_idx = next_action_idx

            total_reward += reward

        if episode > 0 and episode % 100 == 0:
            agent.epsilon = max(0.01, agent.epsilon * 0.95)
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    print("Training finished.")

    print("Visualizing Policy and Values...")
    policy = agent.get_policy_matrix()
    run_episode_with_policy(env, policy)
    env.add_policy(policy)
    env.add_state_values(agent.get_value_matrix(), precision=2)
    # 渲染静态结果
    print("显示价值函数和策略图 (请查看弹出的窗口)...")
    env.render(animation_interval=2)

    # 保持窗口打开
    print("演示结束。")
    plt.ioff()
    plt.show()