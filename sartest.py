import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
from pix import Pix2Pix
from split import Sentinel
from torch.utils.data import DataLoader
from PIL import Image, ImageEnhance
import os
import cv2

# Create output directory if it doesn't exist
OUTPUT_DIR = "generated_images"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PARAMS = {
    "netD": "patch",
    "lambda_L1": 100.0,
    "is_CGAN": True,
    "use_upsampling": False,
    "mode": "nearest",
    "c_hid": 64,
    "n_layers": 3,
    "lr": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "batch_size": 32,
    "epochs": 330,
    "seed": 42,
}

SEED = PARAMS["seed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)


def load_model():
    model = Pix2Pix(
        is_train=True,
        netD=PARAMS["netD"],
        lambda_L1=PARAMS["lambda_L1"],
        is_CGAN=PARAMS["is_CGAN"],
        use_upsampling=PARAMS["use_upsampling"],
        mode=PARAMS["mode"],
        c_hid=PARAMS["c_hid"],
        n_layers=PARAMS["n_layers"],
        lr=PARAMS["lr"],
        beta1=PARAMS["beta1"],
        beta2=PARAMS["beta2"],
    )

    gen_ckpt = "/content/drive/MyDrive/pix2pix_gen_220.pth"
    model.gen.load_state_dict(
        torch.load(gen_ckpt, map_location=DEVICE, weights_only=True), strict=False
    )

    disc_ckpt = "/content/drive/MyDrive/pix2pix_disc_220.pth"
    model.disc.load_state_dict(
        torch.load(disc_ckpt, map_location=DEVICE, weights_only=True), strict=False
    )

    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    return model


def apply_desert_filter(image_array):
    """
    Apply a desert-like filter to the image:
    - Increase brightness
    - Add warm tones
    - Increase saturation
    - Adjust contrast
    """
    # Convert to PIL Image
    image = Image.fromarray((image_array * 255).astype(np.uint8))

    # Enhance brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.3)  # Increase brightness by 30%

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)  # Increase contrast by 20%

    # Convert to cv2 format for color adjustments
    image = np.array(image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Increase saturation
    image_hsv[:, :, 1] = image_hsv[:, :, 1] * 1.2  # Increase saturation by 20%
    image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1], 0, 255)

    # Shift hue slightly towards yellow/orange
    image_hsv[:, :, 0] = (image_hsv[:, :, 0] * 0.9 + 20) % 180

    # Convert back to RGB
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    return image / 255.0


def scale_and_convert(tensor):
    tensor = (tensor + 1) / 2
    return tensor.clamp(0, 1).cpu().numpy()


def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def pad_image_to_multiple(image, multiple):
    _, _, h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return torch.nn.functional.pad(image, (0, pad_w, 0, pad_h))


def divide_image(image, patch_size):
    patches = []
    _, _, h, w = image.shape
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[:, :, i : i + patch_size, j : j + patch_size]
            patches.append(patch)
    return patches


def stitch_images(patches, image_shape, patch_size):
    _, _, h, w = image_shape
    stitched_image = torch.zeros((1, 3, h, w), device=patches[0].device)
    idx = 0
    rows = h // patch_size
    cols = w // patch_size

    for i in range(rows):
        for j in range(cols):
            if idx < len(patches):
                curr_patch = patches[idx]
                y_start = i * patch_size
                x_start = j * patch_size
                y_end = min((i + 1) * patch_size, h)
                x_end = min((j + 1) * patch_size, w)

                patch_h = y_end - y_start
                patch_w = x_end - x_start

                stitched_image[:, :, y_start:y_end, x_start:x_end] = curr_patch[
                    :, :, :patch_h, :patch_w
                ]
                idx += 1

    return stitched_image


def save_image(array, filename):
    """Save a numpy array as an image."""
    plt.imsave(filename, array)


def save_image_with_filter(image_array, filename):
    """Save both original and filtered versions of the image."""
    if image_array.ndim == 3:
        filtered_image = apply_desert_filter(image_array)
    else:
        filtered_image = apply_desert_filter(np.transpose(image_array[0], (1, 2, 0)))

    plt.imsave(filename, filtered_image)


def process_image(image_path, model):
    # Define the transform for the input image
    input_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # Load and preprocess the input image
    input_image = load_and_preprocess_image(image_path, input_transform)
    input_image = input_image.to(DEVICE)

    # Pad image if necessary
    input_image = pad_image_to_multiple(input_image, 64)

    # Divide the input image into 64x64 patches
    patch_size = 64
    patches = divide_image(input_image, patch_size)

    # Resize each patch to 256x256, pass through the model, and downsize back to 64x64
    generated_patches = []
    resize_to_256 = transforms.Resize((256, 256))
    resize_to_64 = transforms.Resize((64, 64))

    with torch.no_grad():
        for patch in patches:
            if patch.dim() == 3:
                patch = patch.unsqueeze(0)
            resized_patch = resize_to_256(patch)
            generated_patch = model.gen(resized_patch)
            downsized_patch = resize_to_64(generated_patch)
            generated_patches.append(downsized_patch)

    # Stitch the generated patches back together
    generated_image = stitch_images(generated_patches, input_image.shape, patch_size)

    # Convert the tensors to numpy arrays for visualization
    input_image_np = scale_and_convert(input_image)
    generated_image_np = scale_and_convert(generated_image)

    return input_image_np, generated_image_np


def save_comparison(input_image_np, generated_image_np, output_filename):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot input image
    axes[0].imshow(np.transpose(input_image_np[0], (1, 2, 0)))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Plot generated image without filter
    axes[1].imshow(np.transpose(generated_image_np[0], (1, 2, 0)))
    axes[1].set_title("Generated Image (No Filter)")
    axes[1].axis("off")

    # Apply desert filter and plot
    filtered_image = apply_desert_filter(np.transpose(generated_image_np[0], (1, 2, 0)))
    axes[2].imshow(filtered_image)
    axes[2].set_title("Generated Image (Desert Filter)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_filename}_comparison.png"))
    plt.close(fig)

    # Save individual images
    save_image(
        np.transpose(input_image_np[0], (1, 2, 0)),
        os.path.join(OUTPUT_DIR, f"{output_filename}_input.png"),
    )
    save_image(
        np.transpose(generated_image_np[0], (1, 2, 0)),
        os.path.join(OUTPUT_DIR, f"{output_filename}_generated.png"),
    )
    save_image(
        filtered_image,
        os.path.join(OUTPUT_DIR, f"{output_filename}_generated_desert.png"),
    )


def main():
    # Load the model
    model = load_model()

    # Process single image
    image_path = "test.jpg"  # Change this to your input image path

    try:
        # Process the image
        input_image_np, generated_image_np = process_image(image_path, model)

        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save the results
        save_comparison(input_image_np, generated_image_np, base_filename)

        # Save the filtered version separately
        generated_image_transposed = np.transpose(generated_image_np[0], (1, 2, 0))
        save_image_with_filter(
            generated_image_transposed,
            os.path.join(OUTPUT_DIR, f"{base_filename}_generated_desert_only.png"),
        )

        print(f"Processing complete! Images saved in {OUTPUT_DIR}/")

    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    main()
