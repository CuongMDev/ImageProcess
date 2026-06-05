import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter, minimum_filter

def yeni_local_mean(image, alpha=7.0):
    """Tính toán mặt nạ mờ Mu(m,n) bằng bộ lọc YENI [2-5]"""
    img = image.astype(np.float32)
    rows, cols = img.shape
    mu_f = np.zeros_like(img)
    mu_b = np.zeros_like(img)

    # 1. Bộ lọc tiến (Forward) [3, 4]
    mu_f[:, 0] = img[:, 0]
    for n in range(1, cols):
        diff = np.abs(mu_f[:, n-1] - img[:, n])
        lam = (1.0 - diff / 255.0) ** alpha
        mu_f[:, n] = lam * mu_f[:, n-1] + (1.0 - lam) * img[:, n]

    # 2. Bộ lọc lùi (Backward) [5]
    mu_b[:, -1] = img[:, -1]
    for n in range(cols - 2, -1, -1):
        diff = np.abs(mu_b[:, n+1] - img[:, n])
        lam = (1.0 - diff / 255.0) ** alpha
        mu_b[:, n] = lam * mu_b[:, n+1] + (1.0 - lam) * img[:, n]

    # Trung bình cộng để triệt tiêu lệch pha [2, 5]
    return (mu_f + mu_b) / 2.0

def proposed_ace(image, alpha=7.0, a=1, b=7, c=21, K=1.0):
    """Thuật toán ACE đề xuất sử dụng hàm tăng cường Cosin [6-8]"""
    img = image.astype(np.float32)
    mu = yeni_local_mean(img, alpha)
    detail = img - mu
    om = np.abs(detail)
    
    gain = np.zeros_like(om)
    # Đoạn [a, b]: Tăng theo Cosin góc phần tư thứ 3 [8]
    mask_ab = (om >= a) & (om < b)
    gain[mask_ab] = (K / 2.0) * (1 - np.cos(np.pi * (om[mask_ab] - a) / (b - a)))
    
    # Đoạn [b, c]: Giảm theo Cosin góc phần tư thứ 1 [8]
    mask_bc = (om >= b) & (om <= c)
    gain[mask_bc] = (K / 2.0) * (1 + np.cos(np.pi * (om[mask_bc] - b) / (c - b)))
    
    # Công thức cải thiện (10): y = x + g * (x - mu) [7]
    output = img + gain * detail
    return np.clip(output, 0, 255).astype(np.uint8)

def linear_um(image, gain=1.0, sigma=1.0):
    img = image.astype(np.float32)

    # LPF: Gaussian blur 2D (chuẩn paper)
    lpf = gaussian_filter(img, sigma=sigma)

    detail = img - lpf
    output = img + gain * detail

    return np.clip(output, 0, 255).astype(np.uint8)

# LƯU Ý: Các phương pháp Cubic, Rational và OS Laplacian dưới đây được mô phỏng 
# dựa trên đặc điểm mô tả trong tài liệu (vùng [10-12]), vì công thức 
# chi tiết nằm ở các tài liệu tham khảo ngoài.

def cubic_um(image, gain=0.0005, sigma=1.0):
    img = image.astype(np.float32)

    mu = gaussian_filter(img, sigma=sigma)
    detail = img - mu

    # signed cubic (chuẩn paper)
    detail_cubic = np.sign(detail) * (np.abs(detail) ** 3)

    output = img + gain * detail_cubic

    return np.clip(output, 0, 255).astype(np.uint8)

def rational_um(image, gain=1.0, h=0.01, sigma=1.0):
    img = image.astype(np.float32)

    mu = gaussian_filter(img, sigma=sigma)
    detail = img - mu

    output = img + (gain * detail) / (1.0 + h * (detail ** 2))

    return np.clip(output, 0, 255).astype(np.uint8)

def os_laplacian_um(image, gain=0.5, size=3):
    img = image.astype(np.float32)

    # Tính min, med, max độc lập trên toàn bộ ảnh bằng C-backend của scipy
    pmin = minimum_filter(img, size=size)
    pmed = median_filter(img, size=size)
    pmax = maximum_filter(img, size=size)

    # Kết hợp OS smoothing
    os_base = 0.25 * pmin + 0.5 * pmed + 0.25 * pmax

    detail = img - os_base
    output = img + gain * detail

    return np.clip(output, 0, 255).astype(np.uint8)