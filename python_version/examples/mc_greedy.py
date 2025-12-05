import sys
sys.path.append("..")
from src.grid_world import GridWorld
import numpy as np
import time
import matplotlib.pyplot as plt

def get_state_index(state, env_size):
    """将坐标 (x, y) 转换为扁平化的索引"""
    return state[1] * env_size[0] + state[0]

def get_state_coord(index, env_size):
    x = index % env_size[0]
    y = index // env_size[0]
    return (x, y)

def mc_greedy(env, gamma=0.9, epsilon=0.1):
    # 1. 初始化环境
    env_size = env.env_size
    num_states = env.num_states
    num_actions = len(env.action_space)

    # 2. 超参数设置
    len_trajectory = int(1e6)    # 轨迹长度足够长

    # 3. 初始化 Q 表, 计数表 N (用于计算平均值), 和策略  
    # Q[s, a]: 状态 s 下采取动作 a 的价值
    Q = np.zeros((num_states, num_actions))
    # Returns[s, a]: 状态 s 下采取动作 a 的 N[s, a] 次累计奖赏的和
    Returns = np.zeros((num_states, num_actions))
    # N[s, a]: 状态 s 下采取动作 a 被访问的次数
    N = np.zeros((num_states, num_actions))

    # 初始化策略为均匀随机策略
    policy = np.ones((num_states, num_actions)) / num_actions

    for _ in range(2):
        # 生成一条足够长的采样轨迹
        state, _ = env.reset()
        trajectory = [] # 存储 (state_idx, action_idx, reward)
        start_time = time.time()
        for _ in range(len_trajectory):
            state_idx = get_state_index(state, env_size)
            # 根据当前策略选择动作 (Sampling from policy)
            action_idx = np.random.choice(num_actions, p=policy[state_idx])
            action = env.action_space[action_idx]
            next_state, reward, _, _ = env.step(action)
            # 记录轨迹
            trajectory.append((state_idx, action_idx, reward))
            state = next_state
        print(f"Time taken: {time.time() - start_time:.4f}s")

        # --- 更新 Q 值和策略 (MC Update) ---  
        g = 0  

        # 逆序遍历轨迹计算回报 (Backward pass)
        for t in range(len(trajectory) - 1, -1, -1):
            s_idx, a_idx, r = trajectory[t]
            g = gamma * g + r
            Returns[s_idx, a_idx] += g
            N[s_idx, a_idx] += 1
            # 策略评估
            Q[s_idx, a_idx] = Returns[s_idx, a_idx] / N[s_idx, a_idx]
            # 策略提高
            q_max_idx = np.argmax(Q[s_idx])
            p = epsilon / num_actions
            for j in range(num_actions):
                policy[s_idx, j] = p
            policy[s_idx, q_max_idx] = 1 - (1 - 1 / num_actions) * epsilon

    return policy         

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
      
    # 最后渲染一帧  
    env.render(animation_interval=1.0)

# Example usage:
if __name__ == "__main__":             
    env = GridWorld()

    optimal_policy = mc_greedy(env)

    # 让智能体按照算出来的策略走一遍
    run_episode_with_policy(env, optimal_policy)

    # Add policy
    env.add_policy(optimal_policy)

    # Render the environment
    env.render(animation_interval=2)
    plt.ioff()
    plt.show()