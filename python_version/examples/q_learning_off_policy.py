import sys    
sys.path.append("..")    
from src.grid_world import GridWorld    
import numpy as np    
import time    
import matplotlib.pyplot as plt    
    
def get_state_index(state, env_size):    
    """将坐标 (x, y) 转换为扁平化的索引"""    
    return state[1] * env_size[0] + state[0]

def q_learning_off_policy(env, gamma=0.9, alpha=0.1, epsilon=0.1, num_episodes=10000):
    # 1. 初始化环境参数    
    env_size = env.env_size    
    num_states = env.num_states    
    num_actions = len(env.action_space)      
    
    # 2. 初始化 Q 表  
    # Q(S, A) 矩阵    
    Q = np.zeros((num_states, num_actions))    
        
    # 初始化目标策略 (最终输出的策略)    
    policy_T = np.zeros((num_states, num_actions))    
    
    print(f"开始训练: {num_episodes} 个回合 (Epsilon={epsilon})...")    
    start_time = time.time()

    for i_episode in range(num_episodes):    
        state, _ = env.reset()  
        state_idx = get_state_index(state, env_size)         
              
        # --- 生成一个完整的回合 ---       
        while not env._is_done(state):
            # 这属于 Off-Policy: 行为策略是 e-greedy，但我们更新 Q 值的 Target 是 max Q (greedy)
            if np.random.rand() < epsilon:
                # 随机探索
                action_idx = np.random.randint(num_actions)
            else:
                # 利用当前知识 (如果有多个最大值，随机选一个防止死板)  
                # action_idx = np.argmax(Q[state_idx]) # 简单写法  
                # 健壮写法：处理初始 Q 全为 0 的情况，随机选择
                values = Q[state_idx]
                best_actions = np.flatnonzero(values == values.max())
                action_idx = np.random.choice(best_actions)    
    
            action = env.action_space[action_idx]

            # 执行动作                  
            next_state, reward, done, _ = env.step(action)  
            next_state_idx = get_state_index(next_state, env_size)    
             
            # Q-Learning 核心公式  
            # Target = R + gamma * max_a Q(S', a)  
            best_next_q = np.max(Q[next_state_idx])  
            target = reward + gamma * best_next_q * (1 - done)  
              
            # 更新 Q 值  
            # 公式: Q(S,A) = Q(S,A) + alpha * (Target - Q(S,A))  
            Q[state_idx, action_idx] += alpha * (target - Q[state_idx, action_idx])  
              
            # 状态转移  
            state = next_state  
            state_idx = next_state_idx     
              
            if done:  
                break  
    
        if (i_episode + 1) % 1000 == 0: 
            print(f"\rEpisode {i_episode + 1}/{num_episodes} done.", end="")    
    
    print(f"\n训练结束，耗时: {time.time() - start_time:.2f}s")
    # === 训练结束后统一导出策略 ===  
    # 根据最终收敛的 Q 表，生成贪婪策略 (Target Policy)
    for s in range(num_states):
        best_a_idx = np.argmax(Q[s])
        policy_T[s, best_a_idx] = 1.0
        
    return policy_T

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

if __name__ == "__main__":             
    env = GridWorld()
    
    # 使用较少的回合数即可，因为加入了 epsilon-greedy 策略效率更高  
    # epsilon=0.1 表示 10% 概率随机探索，90% 概率利用 
    optimal_policy = q_learning_off_policy(env, gamma=0.9, alpha=0.1, epsilon=0.1, num_episodes=50000)

    # 让智能体按照算出来的策略走一遍
    run_episode_with_policy(env, optimal_policy)

    # Add policy
    env.add_policy(optimal_policy)

    # 最终渲染  
    print("关闭图形窗口以退出程序。")
    env.render(animation_interval=2)
    plt.ioff()
    plt.show()    