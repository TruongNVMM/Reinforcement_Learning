import pygame
import gym
import numpy as np
import random
import os

# --- Cấu hình Pygame ---
pygame.init()
WIDTH, HEIGHT = 600, 700
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Frozen Lake AI")
FONT = pygame.font.SysFont("consolas", 40)
SMALL_FONT = pygame.font.SysFont("consolas", 20)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (100, 100, 255)

class Button:
    """Lớp để tạo các nút có thể nhấn trong Pygame."""
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text

    def draw(self, screen):
        pygame.draw.rect(screen, BLUE, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 3)
        text_surf = FONT.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

def show_main_menu():
    """Hiển thị menu chính để chọn map."""
    title_text = FONT.render("Chọn Map Frozen Lake", True, WHITE)
    button_4x4 = Button(150, 250, 300, 100, "Map 4x4")
    button_8x8 = Button(150, 400, 300, 100, "Map 8x8")

    while True:
        SCREEN.fill(BLACK)
        SCREEN.blit(title_text, (title_text.get_rect(center=(WIDTH/2, 150))))
        button_4x4.draw(SCREEN)
        button_8x8.draw(SCREEN)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_4x4.is_clicked(event.pos):
                    return "4x4"
                if button_8x8.is_clicked(event.pos):
                    return "8x8"
        
        pygame.display.flip()

def play_game(map_name):
    """Chạy trò chơi với tác nhân AI đã được huấn luyện."""
    q_table_file = f"q_table_{map_name}.npy"
    if not os.path.exists(q_table_file):
        # Hiển thị thông báo lỗi nếu không tìm thấy file Q-table
        error_text = SMALL_FONT.render(f"Lỗi: Không tìm thấy file '{q_table_file}'.", True, WHITE)
        error_text2 = SMALL_FONT.render("Vui lòng chạy 'train_and_save.py' trước.", True, WHITE)
        SCREEN.fill(BLACK)
        SCREEN.blit(error_text, (error_text.get_rect(center=(WIDTH/2, HEIGHT/2 - 20))))
        SCREEN.blit(error_text2, (error_text2.get_rect(center=(WIDTH/2, HEIGHT/2 + 20))))
        pygame.display.flip()
        pygame.time.wait(3000)
        return

    # Tải Q-table và tạo môi trường
    Qtable = np.load(q_table_file)
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=False, render_mode="rgb_array")
    
    # Sửa lỗi reset() để lấy state
    state, info = env.reset(seed=random.randint(0, 500))
    done = False
    
    clock = pygame.time.Clock()
    
    # Biến để lưu trữ reward cuối cùng
    final_reward = 0
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                # pygame.quit()
                return
        
        # Lấy hành động tốt nhất từ Q-table
        action = np.argmax(Qtable[state])
        
        # --- DÒNG SỬA LỖI CHÍNH ---
        # Nhận 5 giá trị thay vì 4
        new_state, reward, terminated, truncated, info = env.step(action)
        
        # Tập chơi kết thúc nếu terminated hoặc truncated là True
        done = terminated or truncated
        # --------------------------
        
        state = new_state
        final_reward = reward # Lưu reward cuối cùng để hiển thị

        # Render môi trường ra màn hình
        frame = env.render()
        game_surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        game_surface = pygame.transform.scale(game_surface, (500, 500))

        SCREEN.fill(BLACK)
        
        # Hiển thị thông tin
        map_info_text = FONT.render(f"Map: {map_name}", True, WHITE)
        SCREEN.blit(map_info_text, (map_info_text.get_rect(center=(WIDTH/2, 50))))
        
        SCREEN.blit(game_surface, (50, 100))
        pygame.display.flip()
        
        # Đợi một chút để người dùng có thể xem
        clock.tick(2) # 2 frames per second
        
    env.close()
    
    # Hiển thị thông báo kết thúc và chờ một chút trước khi quay lại menu
    result_text = FONT.render("Hoàn thành!", True, WHITE if final_reward > 0 else (255, 100, 100))
    SCREEN.blit(result_text, (result_text.get_rect(center=(WIDTH/2, HEIGHT - 50))))
    pygame.display.flip()
    pygame.time.wait(2000)

def main():
    """Vòng lặp chính của ứng dụng."""
    pygame.init()
    while True:
        selected_map = show_main_menu()
        if selected_map is None:
            break
        play_game(selected_map)
    pygame.quit()

if __name__ == "__main__":
    main()