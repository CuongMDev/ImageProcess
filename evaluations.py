import numpy as np
import cv2

def calculate_vr(original, enhanced):
    """
    Tính tỷ lệ phương sai cục bộ vùng không chứa cạnh (VR).
    """
    orig_f = original.astype(np.float32)
    enh_f = enhanced.astype(np.float32)
    
    # 1. Phát hiện cạnh bằng Canny (theo nguồn [1][2])
    edges = cv2.Canny(original, 100, 200)
    non_edge_mask = (edges == 0)
    
    # 2. Định nghĩa cửa sổ 1x16 (row-wise) để tính phương sai cục bộ
    # Sử dụng bộ lọc trượt để tính trung bình và trung bình bình phương
    window_size = 16
    kernel = np.ones((1, window_size), dtype=np.float32) / window_size
    
    def get_local_variance(img):
        mu = cv2.filter2D(img, -1, kernel)
        mu_sq = cv2.filter2D(img**2, -1, kernel)
        var = mu_sq - mu**2
        return var

    var_orig = get_local_variance(orig_f)
    var_enh = get_local_variance(enh_f)
    
    # 3. Chỉ tính tổng phương sai tại các vùng không phải cạnh
    sum_var_orig = np.sum(var_orig[non_edge_mask])
    sum_var_enh = np.sum(var_enh[non_edge_mask])
    
    # Tránh chia cho 0
    if sum_var_orig == 0: return 1.0
    
    return sum_var_enh / sum_var_orig

def calculate_nar(original, enhancement_func, noise_variances=[3, 4]):
    """
    Tính tỷ lệ khuếch đại nhiễu (NAR).
    enhancement_func: Hàm thực hiện thuật toán cải thiện (ví dụ: Proposed ACE).
    """
    orig_f = original.astype(np.float32)
    enhanced_clean = enhancement_func(original).astype(np.float32)
    nar_list = []
    
    for var in noise_variances:
        # 1. Tạo nhiễu Gaussian
        std = np.sqrt(var)
        noise = np.random.normal(0, std, original.shape).astype(np.float32)
        noisy_input = np.clip(orig_f + noise, 0, 255).astype(np.uint8)
        
        # 2. Cải thiện ảnh nhiễu
        enhanced_noisy = enhancement_func(noisy_input).astype(np.float32)
        
        # 3. Tính phương sai của hiệu số (Noise in the enhanced image)
        # Theo [1]: "variance of the difference in the enhanced image"
        diff_image = enhanced_noisy - enhanced_clean
        actual_noise_var = np.var(diff_image)
        
        # 4. Tỷ lệ NAR cho mức nhiễu này
        nar_list.append(actual_noise_var / var)
        
    return np.mean(nar_list)

