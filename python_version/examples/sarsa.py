import sys    
sys.path.append("..")    
from src.grid_world import GridWorld    
import numpy as np    
import time    
import matplotlib.pyplot as plt    
    
def get_state_index(state, env_size):    
    """将坐标 (x, y) 转换为扁平化的索引"""    
    return state[1] * env_size[0] + state[0]    
    
def sarsa(env, gamma=0.9, alpha=0.1, num_episodes=100000):    
    """    
    Sarsa Algorithm with Epsilon-Greedy Policy  
    """    
    # 1. 初始化环境参数    
    env_size = env.env_size    
    num_states = env.num_states    
    num_actions = len(env.action_space)  
      
    epsilon_start = 1.0    
    epsilon_end = 0.01  
    decay_episodes = int(num_episodes * 0.3)  
    # 计算每一步减少多少    
    epsilon_decay_step = (epsilon_start - epsilon_end) / decay_episodes    
    
    # 2. 初始化 Q 表和策略    
    Q = np.zeros((num_states, num_actions))    
        
    # 初始化策略 (均匀随机)    
    policy = np.ones((num_states, num_actions)) / num_actions    
    
    print(f"开始训练: {num_episodes} 个回合...")    
    start_time = time.time()    
  
    for i_episode in range(num_episodes):    
        state, _ = env.reset()  
        state_idx = get_state_index(state, env_size)  
          
        # 根据当前策略采样动作 (Epsilon-Greedy)  
        action_idx = np.random.choice(num_actions, p=policy[state_idx])    
        action = env.action_space[action_idx]    
          
        # === 线性衰减 epsilon ===    
        if i_episode < decay_episodes:    
            current_epsilon = epsilon_start - i_episode * epsilon_decay_step    
        else:    
            current_epsilon = epsilon_end    
              
        # --- 生成一个完整的回合 ---       
        while not env._is_done(state):                  
            next_state, reward, done, _ = env.step(action)  
            next_state_idx = get_state_index(next_state, env_size)  
              
            # 采样下一个动作 A' (用于 Sarsa 更新)  
            action_idx1 = np.random.choice(num_actions, p=policy[next_state_idx])  
              
            # === 关键修正 1: 计算 TD Target ===  
            # 如果 done 为 True，则没有未来奖励，target 仅为 reward  
            target = reward + gamma * Q[next_state_idx, action_idx1] * (1 - done)  
              
            # 更新 Q 值  
            # 公式: Q(S,A) = Q(S,A) + alpha * (Target - Q(S,A))  
            Q[state_idx, action_idx] += alpha * (target - Q[state_idx, action_idx])  
              
            # === 关键修正 2: 向量化更新策略 ===  
            best_a_idx = np.argmax(Q[state_idx])  
              
            # 先将所有动作概率设为 epsilon / |A|  
            policy[state_idx] = current_epsilon / num_actions  
            # 将最优动作的概率增加 (1 - epsilon)  
            policy[state_idx, best_a_idx] += (1.0 - current_epsilon)  
              
            # 状态与动作转移  
            state = next_state  
            state_idx = next_state_idx  
            action = env.action_space[action_idx1]  
            action_idx = action_idx1   
              
            # 虽然 while 循环会检查，但显式处理 done 逻辑更清晰  
            if done:  
                break  
    
        if (i_episode + 1) % 1000 == 0: # 减少打印频率，加快训练显示  
            print(f"\rEpisode {i_episode + 1}/{num_episodes} done.", end="")    
    
    print(f"\n训练结束，耗时: {time.time() - start_time:.2f}s")    
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
    
    # 训练  
    optimal_policy = sarsa(env, gamma=0.9, alpha=0.1, num_episodes=10000)  
    
    # 提取纯贪婪策略用于绘图 (Deterministic)  
    display_policy = np.zeros_like(optimal_policy)    
    for s in range(len(optimal_policy)):    
        best_a = np.argmax(optimal_policy[s])      
        display_policy[s, best_a] = 1.0    
    
    # 演示路径    
    run_episode_with_policy(env, optimal_policy)    
    
    # 在环境上画出策略    
    env.add_policy(display_policy)    
    
    print("Close the plot window to exit.")    
    env.render(animation_interval=2)   
    plt.ioff()    
    plt.show()  
