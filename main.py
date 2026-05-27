import cv2
import numpy as np


def normalize(img):
    return np.clip(img, 0, 255).astype(np.uint8)


# ==========================================================
# 1. Linear Laplacian Unsharp Masking
# ==========================================================
def linear_laplacian_um(img, k=1.0):
    img = img.astype(np.float32)

    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)

    out = img - k * lap

    return normalize(out)


# ==========================================================
# 2. Cubic Unsharp Masking
# ==========================================================
def cubic_um(img, k=0.00005):
    img = img.astype(np.float32)

    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)

    out = img + k * (lap ** 3)

    return normalize(out)


# ==========================================================
# 3. Rational Unsharp Masking
# ==========================================================
def rational_um(img, a=2.0, b=0.05):
    img = img.astype(np.float32)

    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)

    enhancement = (a * lap) / (1 + b * np.abs(lap))

    out = img + enhancement

    return normalize(out)


# ==========================================================
# 4. OSLap (Order Statistic Laplacian)
# dùng Median thay cho Mean
# ==========================================================
def oslap(img, k=2.0, kernel_size=5):
    img = img.astype(np.float32)

    median = cv2.medianBlur(img.astype(np.uint8), kernel_size)
    median = median.astype(np.float32)

    os_lap = img - median

    out = img + k * os_lap

    return normalize(out)


# ==========================================================
# 5. Proposed (phiên bản gần với ý tưởng bài báo)
# ==========================================================
def proposed_method(
        img,
        window_size=15,
        gain=4.0):

    img = img.astype(np.float32)

    local_mean = cv2.blur(
        img,
        (window_size, window_size)
    )

    detail = img - local_mean

    enhanced_detail = (
        127.0 *
        np.tanh(gain * detail / 127.0)
    )

    out = local_mean + enhanced_detail

    return normalize(out)


# ==========================================================
# Demo
# ==========================================================
img = cv2.imread(
    "image.png",
    cv2.IMREAD_GRAYSCALE
)

cv2.imwrite(
    "linear.png",
    linear_laplacian_um(img)
)

cv2.imwrite(
    "cubic.png",
    cubic_um(img)
)

cv2.imwrite(
    "rational.png",
    rational_um(img)
)

cv2.imwrite(
    "oslap.png",
    oslap(img)
)

cv2.imwrite(
    "proposed.png",
    proposed_method(img)
)