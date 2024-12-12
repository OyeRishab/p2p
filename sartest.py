import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
from pix import Pix2Pix
from split import Sentinel
from torch.utils.data import DataLoader
from PIL import Image

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
print("Loaded succesfully!")


def scale_and_convert(tensor):
    tensor = (tensor + 1) / 2
    return tensor.clamp(0, 1).cpu().numpy()


def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


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
    stitched_image = torch.zeros((1, 3, h, w))
    idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            stitched_image[:, :, i : i + patch_size, j : j + patch_size] = patches[idx]
            idx += 1
    return stitched_image


# Define the transform for the input image
input_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ]
)

# Load and preprocess the input image
image_path = "test.jpg"
input_image = load_and_preprocess_image(image_path, input_transform)
input_image = input_image.to(DEVICE)

# Divide the input image into 64x64 patches
patch_size = 64
patches = divide_image(input_image, patch_size)

# Resize each patch to 256x256, pass through the model, and downsize back to 64x64
generated_patches = []
resize_to_256 = transforms.Resize((256, 256))
resize_to_64 = transforms.Resize((64, 64))

with torch.no_grad():
    for patch in patches:
        resized_patch = resize_to_256(patch)
        generated_patch = model.gen(resized_patch)
        downsized_patch = resize_to_64(generated_patch)
        generated_patches.append(downsized_patch)

# Stitch the generated patches back together
generated_image = stitch_images(generated_patches, input_image.shape, patch_size)

# Convert the tensors to numpy arrays for visualization
input_image_np = scale_and_convert(input_image)
generated_image_np = scale_and_convert(generated_image)

# Plot the input and generated images
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

# Plot input image
axes[0].imshow(np.transpose(input_image_np[0], (1, 2, 0)))
axes[0].set_title("Input Image")
axes[0].axis("off")

# Plot generated image
axes[1].imshow(np.transpose(generated_image_np[0], (1, 2, 0)))
axes[1].set_title("Generated Image")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("test_output_image.png")
plt.close(fig)
