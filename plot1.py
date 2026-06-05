import numpy as np
import matplotlib.pyplot as plt

# Thiết lập dải tần số omega từ 0 đến pi
omega = np.linspace(0, np.pi, 500)

# 1. Tính đáp ứng tần số của bộ lọc Laplacian {-1, 2, -1}
# Công thức chuẩn hóa để đạt giá trị 1 tại omega = pi
h_laplacian = (1 - np.cos(omega)) / 2

# 2. Tính đáp ứng tần số của bộ lọc thông cao ACE (YENI)
# Sử dụng phương trình (11) từ paper
def ace_high_pass(w, lam):
    numerator = (lam**2 + lam) * (1 - np.cos(w))
    denominator = 1 - 2 * lam * np.cos(w) + lam**2
    return numerator / denominator

# Các giá trị lambda để mô phỏng "Increasing lambda" như trong Figure 1
lambdas = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]

# Vẽ đồ thị
plt.figure(figsize=(8, 6))

# Vẽ các đường ACE với các giá trị lambda khác nhau
for l in lambdas:
    plt.plot(omega, ace_high_pass(omega, l), 'k-', label=f'λ = {l}' if l == lambdas[-1] else "")

# Vẽ đường Laplacian (đường đứt nét) để so sánh
plt.plot(omega, h_laplacian, 'k--', label='Laplacian filter')

# Định dạng đồ thị theo Figure 1 trong paper
plt.title('Figure 1: Adaptive high-pass filters used in proposed ACE method')
plt.xlabel('ω')
plt.ylabel('1 - H(ω)')
plt.xlim(0, np.pi)
plt.ylim(0, 1.1)
plt.grid(True, linestyle=':', alpha=0.6)

# Thêm chú thích và mũi tên chỉ hướng tăng của lambda
plt.annotate('Increasing λ', xy=(2.5, 0.95), xytext=(2.5, 0.7),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.legend(['ACE high-pass filters', 'Laplacian filter'], loc='lower right')
plt.tight_layout()
plt.savefig('plot1.png', dpi=300)
plt.show()