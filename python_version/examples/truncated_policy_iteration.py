import sys
sys.path.append("..")
from src.grid_world import GridWorld
import numpy as np
import time
from matplotlib import pyplot as plt

def get_state_index(x, y, env_size):
    return y * env_size[0] + x

def get_state_coord(index, env_size):
    x = index % env_size[0]
    y = index // env_size[0]
    return (x, y)

def truncated_policy_iteration(env, gamma=0.9, eval_steps=10, max_iterations=100):
    num_actions = len(env.action_space)
    policy_matrix = np.ones((env.num_states, num_actions)) / num_actions
    V = np.zeros(env.num_states)

    iteration = 0
    start_time = time.time()
    while iteration < max_iterations:
        iteration += 1

        for _ in range(eval_steps):
            # new_V = np.copy(V)
            for s_idx in range(env.num_states):
                v_s = 0.0
                state = get_state_coord(s_idx, env.env_size)
                for a_idx, action in enumerate(env.action_space):
                    prob_a = policy_matrix[s_idx, a_idx]
                    if prob_a > 0.0:
                        next_state, reward = env._get_next_state_and_reward(state, action)
                        v_s += prob_a * (reward + gamma * V[get_state_index(next_state[0], next_state[1], env.env_size)])
                # new_V[s_idx] = v_s
                V[s_idx] = v_s
            # V = new_V

        policy_stable = True
        for s_idx in range(env.num_states):
            old_action_idx = np.argmax(policy_matrix[s_idx])
            state = get_state_coord(s_idx, env.env_size)
            q_values = []
            for a_idx, action in enumerate(env.action_space):
                next_state, reward = env._get_next_state_and_reward(state, action)
                q = reward + gamma * V[get_state_index(next_state[0], next_state[1], env.env_size)]
                q_values.append(q)
                policy_matrix[s_idx, a_idx] = 0.0
            q_max_idx = np.argmax(q_values)
            policy_matrix[s_idx, q_max_idx] = 1.0
            if old_action_idx != q_max_idx:
                policy_stable = False
        
        print(f"Iteration {iteration}: Policy Evaluation ({eval_steps} steps) -> Improvement. Stable: {policy_stable}")

        if policy_stable:
            print(f"Converged after {iteration} iterations.") 
            break
    print(f"Time taken: {time.time() - start_time:.4f}s")
    return V, policy_matrix

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
    
    optimal_values, optimal_policy = truncated_policy_iteration(env, gamma=0.9, eval_steps=5, max_iterations=100)

    # 让智能体按照算出来的策略走一遍
    run_episode_with_policy(env, optimal_policy)

    # Add policy
    env.add_policy(optimal_policy)

    # Add state values
    env.add_state_values(optimal_values, precision=2)

    # Render the environment
    env.render(animation_interval=2)
    plt.ioff()
    plt.show()