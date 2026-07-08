import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def enhanced_pso(func, dim, bounds, num_particles=20, max_iter=30):
    """
    Thuật toán EPSO - Enhanced Particle Swarm Optimization
    (Đã tinh chỉnh số vòng lặp để phù hợp với việc tìm kiếm đặc trưng)
    """
    w_start = 0.9
    w_end = 0.4
    c1 = 1.5
    c2 = 1.5
    v_max = 0.2 * (bounds[1] - bounds[0])
    
    positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
    velocities = np.random.uniform(-v_max, v_max, (num_particles, dim))
    
    pbest_positions = np.copy(positions)
    pbest_scores = np.array([func(p) for p in positions])
    
    gbest_index = np.argmin(pbest_scores)
    gbest_position = np.copy(pbest_positions[gbest_index])
    gbest_score = pbest_scores[gbest_index]
    
    for iteration in range(max_iter):
        w = w_start - ((w_start - w_end) / max_iter) * iteration
        
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest_positions[i] - positions[i]) + 
                             c2 * r2 * (gbest_position - positions[i]))
            
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], bounds[0], bounds[1])
            
            current_score = func(positions[i])
            
            if np.random.rand() < 0.05:  
                positions[i] = gbest_position + np.random.standard_cauchy(dim)
                positions[i] = np.clip(positions[i], bounds[0], bounds[1])
                current_score = func(positions[i])
                
            if current_score < pbest_scores[i]:
                pbest_positions[i] = np.copy(positions[i])
                pbest_scores[i] = current_score
                
                if current_score < gbest_score:
                    gbest_position = np.copy(positions[i])
                    gbest_score = current_score
                    
    return gbest_position, gbest_score

class EPSOFeatureSelector:
    def __init__(self, dim, num_particles=20, max_iter=30, threshold=0.5):
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.threshold = threshold
        
    def select_features(self, X_train, y_train):
        """
        Dùng EPSO như bộ lọc để chọn đặc trưng quan trọng dựa trên mô hình KNN nhỏ gọn.
        X_train: (N, features)
        y_train: (N,)
        """
        clf = KNeighborsClassifier(n_neighbors=3)
        
        def objective_function(mask):
            # Cắt ngưỡng để lấy danh sách đặc trưng được chọn (0 hoặc 1)
            selected = mask > self.threshold
            num_selected = np.sum(selected)
            
            # Nếu không chọn đặc trưng nào, bị phạt cực nặng
            if num_selected == 0:
                return 9999.0 
            
            # Chỉ lấy các đặc trưng được chọn
            X_subset = X_train[:, selected]
            
            # Đánh giá bằng Cross Validation trên 3 fold (nhanh)
            # Hàm mục tiêu EPSO cần TỐI THIỂU HÓA, nên lấy (1 - accuracy)
            scores = cross_val_score(clf, X_subset, y_train, cv=3, n_jobs=-1)
            error_rate = 1.0 - np.mean(scores)
            
            # Thêm hệ số phạt (penalty) nếu chọn quá nhiều đặc trưng (Khuyến khích số chiều nhỏ)
            alpha = 0.01 
            fitness = error_rate + alpha * (num_selected / self.dim)
            return fitness
            
        bounds = np.array([0, 1])
        print(f"Đang chạy EPSO thu gọn đặc trưng từ {self.dim} chiều...")
        best_mask, best_fitness = enhanced_pso(
            objective_function, 
            dim=self.dim, 
            bounds=bounds, 
            num_particles=self.num_particles, 
            max_iter=self.max_iter
        )
        
        selected_indices = np.where(best_mask > self.threshold)[0]
        print(f"=> EPSO đã giảm chiều từ {self.dim} xuống {len(selected_indices)} đặc trưng (Fitness={best_fitness:.4f})")
        
        return selected_indices
