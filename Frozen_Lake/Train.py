import numpy as np
import gym
import random
from tqdm import trange

def initialize_Q_table(state_space, action_space):
    """Khởi tạo Q-table với tất cả giá trị bằng 0."""
    return np.zeros((state_space, action_space))

def epsilon_greedy_policy(Qtable, state, epsilon, env):
    """Chọn hành động bằng chính sách epsilon-tham lam."""
    if random.uniform(0, 1) > epsilon:
        # Khai thác: chọn hành động tốt nhất
        action = np.argmax(Qtable[state])
    else:
        # Khám phá: chọn hành động ngẫu nhiên
        action = env.action_space.sample()
    return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable, gamma, learning_rate):
    """Hàm huấn luyện tác nhân."""
    print(f"Bắt đầu huấn luyện cho map {env.spec.id}...")
    for episode in trange(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        
        # Sửa lỗi reset() để lấy state
        state, info = env.reset() # Các phiên bản mới trả về (state, info)
        done = False

        for _ in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon, env)
            
            # --- DÒNG SỬA LỖI CHÍNH ---
            # Nhận 5 giá trị thay vì 4
            new_state, reward, terminated, truncated, info = env.step(action)
            
            # Tập chơi kết thúc nếu terminated hoặc truncated là True
            done = terminated or truncated
            # --------------------------

            # Cập nhật Q-table bằng công thức Bellman
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )
            
            if done:
                break
            state = new_state
            
    print("Huấn luyện hoàn tất!")
    return Qtable

# Các hàm và phần main() khác giữ nguyên

def main():
    """Hàm chính để huấn luyện và lưu Q-tables."""
    # Các siêu tham số chung
    n_training_episodes = 100000
    learning_rate = 0.5
    max_steps = 99
    gamma = 0.95
    max_epsilon = 1.0
    min_epsilon = 0.005
    decay_rate = 0.00005

    # --- Huấn luyện cho map 4x4 ---
    env_4x4 = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    state_space_4x4 = env_4x4.observation_space.n
    action_space_4x4 = env_4x4.action_space.n
    Qtable_4x4 = initialize_Q_table(state_space_4x4, action_space_4x4)
    
    trained_Qtable_4x4 = train(
        n_training_episodes, min_epsilon, max_epsilon, decay_rate, 
        env_4x4, max_steps, Qtable_4x4, gamma, learning_rate
    )
    np.save("q_table_4x4.npy", trained_Qtable_4x4)
    print("Đã lưu Q-table cho map 4x4 vào file 'q_table_4x4.npy'")
    env_4x4.close()

    print("\n" + "="*30 + "\n")

    # --- Huấn luyện cho map 8x8 ---
    env_8x8 = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    state_space_8x8 = env_8x8.observation_space.n
    action_space_8x8 = env_8x8.action_space.n
    Qtable_8x8 = initialize_Q_table(state_space_8x8, action_space_8x8)
    
    trained_Qtable_8x8 = train(
        n_training_episodes, min_epsilon, max_epsilon, decay_rate, 
        env_8x8, max_steps, Qtable_8x8, gamma, learning_rate
    )
    np.save("q_table_8x8.npy", trained_Qtable_8x8)
    print("Đã lưu Q-table cho map 8x8 vào file 'q_table_8x8.npy'")
    env_8x8.close()

if __name__ == "__main__":
    main()