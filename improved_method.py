import cv2
from matplotlib import image
import numpy as np
from scipy.ndimage import median_filter
from methods import yeni_local_mean

def perfectly_balanced_ace(
    image,
    alpha=7.0,
    gain=2.5,
    median_size=5,
    blur_size=5
):
    img = image.astype(np.float32)

    # --------------------------------------------------
    # 1. YENI local mean
    # --------------------------------------------------
    mu = yeni_local_mean(img, alpha)

    # --------------------------------------------------
    # 2. Edge confidence
    # --------------------------------------------------
    residual = np.abs(img - mu)

    tau = np.mean(residual) + 1e-6
    sigma = np.std(residual) + tau

    confidence = (
        residual / (residual + tau)
    ) * np.exp(-residual / sigma)

    # --------------------------------------------------
    # 3. OS-Laplacian
    # --------------------------------------------------
    med = median_filter(img, size=median_size)

    lap_os = img - med

    # --------------------------------------------------
    # 4. Soft clipping
    # --------------------------------------------------
    T = np.percentile(np.abs(lap_os), 95) + 1e-6

    lap_os = T * np.tanh(lap_os / T)

    # --------------------------------------------------
    # 5. Adaptive detail
    # --------------------------------------------------
    detail = confidence * lap_os

    # --------------------------------------------------
    # 6. Remove local DC bias
    #    (ngăn sáng lan rộng)
    # --------------------------------------------------
    detail -= cv2.blur(
        detail,
        (blur_size, blur_size)
    )

    # --------------------------------------------------
    # 7. Protect highlights/shadows
    # --------------------------------------------------
    room = np.minimum(
        img,
        255.0 - img
    ) / 127.5

    # --------------------------------------------------
    # 8. Enhancement
    # --------------------------------------------------
    output = img + gain * detail * room

    return np.clip(
        output,
        0,
        255
    ).astype(np.uint8)