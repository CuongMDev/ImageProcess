import numpy as np
import cv2

def calculate_tenengrad(image):
    """
    Tính Tenengrad (Sharpness) dựa trên độ dốc Sobel.
    """
    if image.ndim != 2:
        raise ValueError("Tenengrad chỉ áp dụng cho ảnh grayscale.")

    img = image.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_sq = gx * gx + gy * gy
    return float(np.mean(grad_mag_sq))

def calculate_eme(image, block_size=(8, 8), eps=1e-6):
    """
    Tính EME (Enhancement Measure Estimation) - đo mức tương phản cục bộ.
    Chia ảnh thành các khối và tính 20*log10(max/min) rồi lấy trung bình.
    """
    if image.ndim != 2:
        raise ValueError("EME chỉ áp dụng cho ảnh grayscale.")

    h, w = image.shape
    bh, bw = block_size
    if bh <= 0 or bw <= 0:
        raise ValueError("block_size phải > 0.")

    # Cắt ảnh để chia hết theo block_size
    h_crop = h - (h % bh)
    w_crop = w - (w % bw)
    img = image[:h_crop, :w_crop].astype(np.float32)

    eme_sum = 0.0
    block_count = 0

    for y in range(0, h_crop, bh):
        for x in range(0, w_crop, bw):
            block = img[y:y + bh, x:x + bw]
            block_max = float(block.max())
            block_min = float(block.min())
            eme_sum += 20.0 * np.log10((block_max + eps) / (block_min + eps))
            block_count += 1

    return eme_sum / block_count if block_count > 0 else 0.0

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

