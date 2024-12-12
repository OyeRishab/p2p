import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
from pix import Pix2Pix
from PIL import Image
import cv2


def seamless_clone(input_image, generated_image):
    # Transpose and convert to uint8
    input_image = (input_image[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    generated_image = (generated_image[0].transpose(1, 2, 0) * 255).astype(np.uint8)

    # Create a mask for blending
    mask = np.ones(generated_image.shape[:2], dtype=np.uint8) * 255

    # Compute center point
    height, width = generated_image.shape[:2]
    center = (width // 2, height // 2)

    # Perform seamless cloning
    blended = cv2.seamlessClone(
        src=generated_image,
        dst=input_image,
        mask=mask,
        p=center,
        flags=cv2.NORMAL_CLONE,
    )

    return blended / 255.0


def divide_image_with_overlap(image, patch_size, overlap):
    """
    Divide image into overlapping patches

    Args:
        image (torch.Tensor): Input image tensor
        patch_size (int): Size of each patch
        overlap (int): Overlap between patches

    Returns:
        list: List of overlapping patches
    """
    patches = []
    _, _, h, w = image.shape

    for i in range(0, h - overlap, patch_size - overlap):
        for j in range(0, w - overlap, patch_size - overlap):
            patch = image[:, :, i : min(i + patch_size, h), j : min(j + patch_size, w)]
            patches.append(patch)

    return patches


def stitch_images_with_blend(patches, image_shape, patch_size, overlap):
    """
    Stitch patches with smooth blending

    Args:
        patches (list): List of generated patches
        image_shape (tuple): Shape of original image
        patch_size (int): Size of each patch
        overlap (int): Overlap between patches

    Returns:
        torch.Tensor: Stitched and blended image
    """
    _, _, h, w = image_shape
    stitched_image = torch.zeros((1, 3, h, w))
    weight_map = torch.zeros((1, 3, h, w))

    patch_idx = 0
    for i in range(0, h - overlap, patch_size - overlap):
        for j in range(0, w - overlap, patch_size - overlap):
            patch = patches[patch_idx]
            patch_h, patch_w = patch.shape[2], patch.shape[3]

            # Create a weight map for smooth blending
            patch_weight = torch.ones_like(patch)
            if i > 0:
                patch_weight[:, :, :overlap, :] *= (
                    torch.linspace(0, 1, overlap)
                    .view(1, -1)
                    .repeat(patch.shape[0], patch.shape[1], 1, patch.shape[3])
                )
            if j > 0:
                patch_weight[:, :, :, :overlap] *= (
                    torch.linspace(0, 1, overlap)
                    .view(-1, 1)
                    .repeat(patch.shape[0], patch.shape[1], patch.shape[2], 1)
                )

            # Update stitched image with weighted patch
            stitched_image[:, :, i : i + patch_h, j : j + patch_w] += (
                patch * patch_weight
            )
            weight_map[:, :, i : i + patch_h, j : j + patch_w] += patch_weight

            patch_idx += 1

    # Normalize by weight map
    stitched_image /= weight_map
    return stitched_image


# Main script remains mostly the same, replace divide_image and stitch_images with new functions
patch_size = 64
overlap = 16  # Overlap between patches

# Divide the input image into overlapping patches
patches = divide_image_with_overlap(input_image, patch_size, overlap)

# Generate patches (rest of the generation code remains the same)
generated_patches = []
resize_to_256 = transforms.Resize((256, 256))
resize_to_64 = transforms.Resize((64, 64))

with torch.no_grad():
    for patch in patches:
        resized_patch = resize_to_256(patch)
        generated_patch = model.gen(resized_patch)
        downsized_patch = resize_to_64(generated_patch)
        generated_patches.append(downsized_patch)

# Stitch the generated patches back together with blending
generated_image = stitch_images_with_blend(
    generated_patches, input_image.shape, patch_size, overlap
)

# Convert the tensors to numpy arrays for visualization
input_image_np = scale_and_convert(input_image)
generated_image_np = scale_and_convert(generated_image)

# Optional: Use seamless cloning for further blending
blended_image = seamless_clone(input_image_np, generated_image_np)

# Plot the input and generated images
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

# Plot input image
axes[0].imshow(np.transpose(input_image_np[0], (1, 2, 0)))
axes[0].set_title("Input Image")
axes[0].axis("off")

# Plot generated image
axes[1].imshow(np.transpose(generated_image_np[0], (1, 2, 0)))
axes[1].set_title("Generated Image")
axes[1].axis("off")

# Plot blended image
axes[2].imshow(blended_image)
axes[2].set_title("Blended Image")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("test_output_image.png")
plt.close(fig)
