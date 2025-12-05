import sys
sys.path.append("..")
from src.grid_world import GridWorld
import numpy as np
import matplotlib.pyplot as plt

def value_iteration(env, gamma=0.9, theta=1e-4):
    """
    执行值迭代算法
    :param env: GridWorld 环境实例
    :param gamma: 折扣因子 (0 < gamma <= 1)
    :param theta: 收敛阈值
    :return: 状态价值数组 V，策略矩阵 policy
    """
    # 初始化状态值函数 V，全为 0
    V = np.zeros(env.num_states)
    # 初始化策略矩阵，形状：[状态数，动作数]，存储每个动作的概率
    policy = np.zeros((env.num_states, len(env.action_space)))
    # 获取网格宽度，用于坐标转换
    width = env.env_size[0]

    print("开始值迭代...")
    iteration = 0
    while True:
        delta = 0
        # 遍历每一个状态
        for s_idx in range(env.num_states):
            # 将一维索引转换为 (x, y) 坐标
            x = s_idx % width
            y = s_idx // width
            state = (x, y)

            v_old = V[s_idx]
            #计算该状态下所有动作的 Q 值：Q(s, a) = r + gamma * V(s')
            q_values = []
            for j, action in enumerate(env.action_space):
                # 使用环境内部函数预测下一步状态和奖励 (Model-Based)
                next_state, reward = env._get_next_state_and_reward(state, action)
                # 将下一步状态 (x, y) 转回一维索引
                next_s_idx = next_state[1] * width + next_state[0]
                # Bellman 最优方程
                # 注意：此环境是确定性的，没有概率转移 p
                q_value = reward + gamma * V[next_s_idx]
                q_values.append(q_value)
                # 该状态下的所有动作概率赋 0.0
                policy[s_idx][j] = 0.0
            
            # 找到最大 Q 值对应的动作索引（可能有多个）
            best_action_idx = np.argmax(q_values)
            # 设置确定性策略：最优动作概率为 1
            policy[s_idx][best_action_idx] = 1.0
            # 更新 V(s) 为最大的 Q 值
            V[s_idx] = max(q_values)

            # 记录最大变动量
            delta = max(delta, abs(v_old - V[s_idx]))

        iteration += 1
        if delta < theta:
            print(f"值迭代在第 {iteration} 轮收敛。")
            break

    return V, policy

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
               
    # 执行值迭代
    # gamma: 远期奖励折扣，0.9 表示比较看重未来
    optimal_values, optimal_policy = value_iteration(env, gamma=0.9)
    
    # 让智能体按照算出来的策略走一遍
    run_episode_with_policy(env, optimal_policy)

    # Add policy
    env.add_policy(optimal_policy)

    # Add state values
    env.add_state_values(optimal_values, precision=1)

    # 渲染静态结果
    print("显示价值函数和策略图 (请查看弹出的窗口)...")
    env.render(animation_interval=2)

    # 保持窗口打开
    print("演示结束。")
    plt.ioff()
    plt.show()