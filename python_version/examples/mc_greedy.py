import sys  
sys.path.append("..")  
from src.grid_world import GridWorld  
import numpy as np  
import time  
import matplotlib.pyplot as plt  
  
def get_state_index(state, env_size):  
    """将坐标 (x, y) 转换为扁平化的索引"""  
    return state[1] * env_size[0] + state[0]  
  
def mc_greedy(env, gamma=0.9, num_episodes=100000):  
    """  
    First-visit / Every-visit Monte Carlo Control with Epsilon-Greedy  
    """  
    # 1. 初始化环境参数  
    env_size = env.env_size  
    num_states = env.num_states  
    num_actions = len(env.action_space)
    epsilon_start = 1.0  
    epsilon_end = 0.01
    decay_episodes = int(num_episodes * 0.4)
    # 计算每一步减少多少  
    epsilon_decay_step = (epsilon_start - epsilon_end) / decay_episodes
    num_episodes_front = int(num_episodes * 0.2)  
  
    # 2. 初始化 Q 表, 计数表 N, 和策略  
    Q = np.zeros((num_states, num_actions))  
    Returns = np.zeros((num_states, num_actions)) # 累计回报和  
    N = np.zeros((num_states, num_actions))       # 访问次数  
      
    # 初始化策略 (均匀随机)  
    policy = np.ones((num_states, num_actions)) / num_actions  
  
    print(f"开始训练: {num_episodes} 个回合...")  
    start_time = time.time()  

    for i_episode in range(num_episodes):  
        state, _ = env.reset()  
        trajectory = [] # 存储 (state_idx, action_idx, reward)  
          
        # --- 生成一个完整的回合 (Episode Generation) ---     
        while True:
            state_idx = get_state_index(state, env_size)  
              
            # 采样动作  
            action_idx = np.random.choice(num_actions, p=policy[state_idx])  
            action = env.action_space[action_idx]  
              
            next_state, reward, done, _ = env.step(action)  
              
            # 记录: (s, a, r)  
            trajectory.append((state_idx, action_idx, reward))  
              
            if done:  
                break  
            state = next_state  
          
        # --- 策略评估与改进 (Policy Evaluation & Improvement) ---  
        G = 0
        visited_pairs = set()
        # === 线性衰减计算 ===
        if i_episode < num_episodes_front:
            current_epsilon = epsilon_start      
        elif i_episode < num_episodes_front + decay_episodes:  
            current_epsilon = epsilon_start - (i_episode - num_episodes_front) * epsilon_decay_step  
        else:  
            current_epsilon = epsilon_end
        # 逆序回溯计算  
        for t in range(len(trajectory) - 1, -1, -1):  
            s_idx, a_idx, r = trajectory[t]  
            G = gamma * G + r  

            # First-visit MC: 仅在第一次出现 (s, a) 时更新
            if (s_idx, a_idx) not in visited_pairs:
                visited_pairs.add((s_idx, a_idx))
            else:
                continue
              
            Returns[s_idx, a_idx] += G  
            N[s_idx, a_idx] += 1  
            Q[s_idx, a_idx] = Returns[s_idx, a_idx] / N[s_idx, a_idx]  
              
            # --- 策略改进 (即时更新) ---  
            best_a_idx = np.argmax(Q[s_idx])  
              
            # 更新该状态的策略概率
            policy[s_idx] = current_epsilon / num_actions
            policy[s_idx, best_a_idx] += (1 - current_epsilon)  
  
        if (i_episode + 1) % 1000 == 0:  
            print(f"\rEpisode {i_episode + 1}/{num_episodes} done.", end="")  
  
    print(f"\nTime taken: {time.time() - start_time:.4f}s")  
    return policy  
  
def run_episode_with_policy(env, policy):  
    """使用计算出的策略运行一次环境 (转为贪婪策略演示)"""  
    state, _ = env.reset()  
    width = env.env_size[0]  
    done = False  
    steps = 0  
  
    print("\n开始使用最优策略演示路径...")  
    while not done and steps < 50:  
        env.render(animation_interval=0.5)  
  
        s_idx = state[1] * width + state[0]  
          
        # 演示时取概率最大的动作 (Greedy)  
        action_idx = np.argmax(policy[s_idx])  
        action = env.action_space[action_idx]  
          
        next_state, reward, done, info = env.step(action)  
        print(f"Step: {steps}, State: {state} -> Action: {action}, Reward: {reward}")  
          
        state = next_state  
        steps += 1  
      
    # 最后一帧  
    env.render(animation_interval=1.0)  
    if done:  
        print("成功到达目标！")  
    else:  
        print("未能在限制步数内到达目标。")  
  
if __name__ == "__main__":               
    env = GridWorld()  
  
    # 增加 episode 数量以保证覆盖率，调整 epsilon 鼓励探索  
    optimal_policy = mc_greedy(env, gamma=0.9, num_episodes=30000)  
  
    # 为了绘图清晰，我们将策略转化为纯贪婪策略 (Deterministic) 再传给 env 绘图  
    # 这样箭头只会指向概率最大的方向  
    display_policy = np.zeros_like(optimal_policy)  
    for s in range(len(optimal_policy)):  
        best_a = np.argmax(optimal_policy[s])    
        display_policy[s, best_a] = 1.0  
  
    # 演示路径  
    run_episode_with_policy(env, optimal_policy)  
  
    # 在环境上画出策略  
    env.add_policy(display_policy)  
  
    # 保持窗口显示  
    print("Close the plot window to exit.")  
    env.render(animation_interval=2) 
    plt.ioff()  
    plt.show()  
