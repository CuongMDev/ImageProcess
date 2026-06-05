import numpy as np
import matplotlib.pyplot as plt

# Các tham số từ paper (Mục 2.2)
a, b, c, K = 1, 7, 21, 1

# Tạo dải giá trị Output Magnitude (OM)
om = np.linspace(0, 25, 500)
gain = np.zeros_like(om)

# Tính toán gain dựa trên các đoạn cosin
# Đoạn [a, b]: Upward shifted cosine trong góc phần tư thứ 3 (tăng từ 0 đến K)
mask_ab = (om >= a) & (om < b)
theta_ab = np.pi + (np.pi / 2) * (om[mask_ab] - a) / (b - a)
gain[mask_ab] = K * (1 + np.cos(theta_ab))

# Đoạn [b, c]: Cosine trong góc phần tư thứ 1 (giảm từ K về 0)
mask_bc = (om >= b) & (om <= c)
theta_bc = (np.pi / 2) * (om[mask_bc] - b) / (c - b)
gain[mask_bc] = K * np.cos(theta_bc)

# Vẽ đồ thị
plt.figure(figsize=(8, 6))
plt.plot(om, gain, 'k-', linewidth=2)

# Thêm các nhãn a, b, c, K theo Figure 2
plt.annotate('a', xy=(a, 0), xytext=(a+0.5, 0.05), arrowprops=dict(arrowstyle='->'))
plt.annotate('b', xy=(b, 0), xytext=(b+0.5, 0.05), arrowprops=dict(arrowstyle='->'))
plt.annotate('c', xy=(c, 0), xytext=(c+0.5, 0.05), arrowprops=dict(arrowstyle='->'))
plt.annotate('K', xy=(b, K), xytext=(b+1, K-0.05), arrowprops=dict(arrowstyle='->'))

plt.axvline(x=b, color='gray', linestyle='--', alpha=0.5)
plt.title('Figure 2: Enhancement gain function')
plt.xlabel('Output magnitude of ACE-high pass filter (OM)')
plt.ylabel('Enhancement Gain (g)')
plt.xlim(0, 25)
plt.ylim(0, 1.1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('plot2.png', dpi=300)
plt.show()