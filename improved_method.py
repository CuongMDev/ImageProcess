import cv2
from matplotlib import image
import numpy as np

def yeni_local_mean(image, alpha_edge=7.0, alpha_texture=1.5):
    """
    YENI cải tiến:
    - Alpha thích nghi theo mật độ cạnh
    - Giảm khuếch đại texture dày đặc (cỏ, lá cây, tóc...)
    """

    img = image.astype(np.float32)
    rows, cols = img.shape

    # --------------------------------------------------
    # 1. Edge density map
    # --------------------------------------------------
    # Làm mờ nhẹ ảnh trước khi tính đạo hàm để tránh nhiễu hạt (noise) bị hiểu nhầm là texture
    img_smooth = cv2.GaussianBlur(img, (3, 3), 0)
    
    gx = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 1, ksize=3)

    grad = np.sqrt(gx**2 + gy**2)

    # Lọc ngưỡng để tìm các điểm thực sự là cạnh
    edge = (grad > 20).astype(np.float32)

    # Tính mật độ (Density) bằng một kernel đủ rộng
    density = cv2.blur(edge, (9, 9))
    density = np.clip(density, 0, 1) # Đảm bảo density luôn nằm trong khoảng [0, 1]

    # --------------------------------------------------
    # * ĐIỂM CHÍNH LÀ ĐÂY: NỘI SUY ALPHA *
    # Density = 0 (phẳng hoặc 1 nét đơn) -> alpha_map = alpha_edge (Giữ sắc nét)
    # Density = 1 (cỏ, tóc dày đặc)     -> alpha_map = alpha_texture (Ép làm mờ)
    # --------------------------------------------------
    alpha_map = alpha_edge * (1.0 - density) + alpha_texture * density

    # --------------------------------------------------
    # 2. Forward filter
    # --------------------------------------------------
    mu_f = np.zeros_like(img)
    mu_f[:, 0] = img[:, 0]

    for n in range(1, cols):

        diff = np.abs(
            mu_f[:, n - 1] -
            img[:, n]
        )

        alpha_local = alpha_map[:, n]

        lam = (
            1.0 -
            diff / 255.0
        ) ** alpha_local

        mu_f[:, n] = (
            lam * mu_f[:, n - 1]
            +
            (1.0 - lam) * img[:, n]
        )

    # --------------------------------------------------
    # 3. Backward filter
    # --------------------------------------------------
    mu_b = np.zeros_like(img)
    mu_b[:, -1] = img[:, -1]

    for n in range(cols - 2, -1, -1):

        diff = np.abs(
            mu_b[:, n + 1] -
            img[:, n]
        )

        alpha_local = alpha_map[:, n]

        lam = (
            1.0 -
            diff / 255.0
        ) ** alpha_local

        mu_b[:, n] = (
            lam * mu_b[:, n + 1]
            +
            (1.0 - lam) * img[:, n]
        )

    # --------------------------------------------------
    # 4. Average
    # --------------------------------------------------
    mu = (mu_f + mu_b) / 2.0

    return mu

def perfectly_balanced_ace(image, alpha=7.0, a=1, b=7, c=21, K=1.0):
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