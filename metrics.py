from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2


def calculate_psnr(real_image, generated_image):
    return psnr(
        real_image, generated_image, data_range=real_image.max() - real_image.min()
    )


def calculate_ssim(original, generated):
    # Ensure images are resized to at least 7x7
    min_size = max(7, max(original.shape[0], original.shape[1]))
    original_resized = cv2.resize(original, (min_size, min_size))
    generated_resized = cv2.resize(generated, (min_size, min_size))

    # Calculate data range
    data_range = original_resized.max() - original_resized.min()

    # Compute SSIM
    score = ssim(
        original_resized,
        generated_resized,
        data_range=data_range,
        win_size=7,
        channel_axis=-1,
    )
    return score


def calculate_sam(real_image, generated_image):
    real_image = real_image.astype(np.float64)
    generated_image = generated_image.astype(np.float64)

    dot_product = np.sum(real_image * generated_image, axis=-1)

    norm_real_image = np.linalg.norm(real_image, axis=-1)
    norm_generated_image = np.linalg.norm(generated_image, axis=-1)

    cos_theta = dot_product / (norm_real_image * norm_generated_image + 1e-8)
    sam = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return np.mean(sam)
