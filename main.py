import numpy as np
import cv2

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

def linear_um(image, gain=1.0):
    """Linear Unsharp Masking sử dụng Laplacian filter {-1, 2, -1} [1, 9]"""
    img = image.astype(np.float32)
    # Áp dụng bộ lọc Laplacian 1D theo hàng
    kernel = np.array([[-1, 2, -1]], dtype=np.float32)
    laplacian = cv2.filter2D(img, -1, kernel)
    output = img + gain * laplacian
    return np.clip(output, 0, 255).astype(np.uint8)

# LƯU Ý: Các phương pháp Cubic, Rational và OS Laplacian dưới đây được mô phỏng 
# dựa trên đặc điểm mô tả trong tài liệu (vùng [10-12]), vì công thức 
# chi tiết nằm ở các tài liệu tham khảo ngoài.

def cubic_um(image, gain=0.5):
    """Mô phỏng Cubic UM: Tăng cường dựa trên lũy thừa bậc 3 của chi tiết [10, 11]"""
    img = image.astype(np.float32)
    mu = cv2.GaussianBlur(img, (1, 5), 0) # Lọc thông thấp theo hàng
    detail = img - mu
    output = img + gain * (detail ** 3) / (128**2) # Tăng cường phi tuyến
    return np.clip(output, 0, 255).astype(np.uint8)

def rational_um(image, gain=2.0):
    """Mô phỏng Rational UM: Sử dụng hàm hữu tỉ để giảm o/u shooting [11, 12]"""
    img = image.astype(np.float32)
    mu = cv2.GaussianBlur(img, (1, 5), 0)
    detail = img - mu
    # Hàm hữu tỉ dạng k*d / (1 + h*d^2)
    rational_gain = gain / (1 + (detail/25)**2)
    output = img + rational_gain * detail
    return np.clip(output, 0, 255).astype(np.uint8)

def process_and_compare(path):
    orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if orig is None: return
    
    results = {
        "Original": orig,
        "Proposed ACE (YENI)": proposed_ace(orig),
        "Linear UM": linear_um(orig),
        "Cubic UM": cubic_um(orig),
        "Rational UM": rational_um(orig)
    }
    
    for name, img in results.items():
        cv2.imwrite(f"{name.replace(' ', '_')}.png", img)
    print("Đã xuất các file so sánh.")


def main():
    """Điểm vào chương trình: đọc ảnh đầu vào và xuất các ảnh so sánh."""
    process_and_compare("image.png")


if __name__ == "__main__":
    main()