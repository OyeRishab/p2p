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

total_params = sum(p.numel() for p in model.gen.parameters())
total_trainable_params = sum(
    p.numel() for p in model.gen.parameters() if p.requires_grad
)
print("Generator:")
print(f"Total params: {total_params}, Total trainable params: {total_trainable_params}")

total_params = sum(p.numel() for p in model.disc.parameters())
total_trainable_params = sum(
    p.numel() for p in model.disc.parameters() if p.requires_grad
)
print("Discriminator:")
print(f"Total params: {total_params}, Total trainable params: {total_trainable_params}")

gen_ckpt = "pix2pix_gen_330.pth"
model.gen.load_state_dict(
    torch.load(gen_ckpt, map_location=DEVICE, weights_only=True), strict=False
)

disc_ckpt = "pix2pix_disc_330.pth"
model.disc.load_state_dict(
    torch.load(disc_ckpt, map_location=DEVICE, weights_only=True), strict=False
)

model.to(DEVICE)
model.eval()
print("Loaded succesfully!")

root_dir = "v_2"
split_save_path = "split.json"
# Load the custom dataset
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ]
)

dataset = Sentinel(
    root_dir=root_dir,
    split_type="test",
    transform=train_transforms,
    split_mode="random",
    split_ratio=(0.8, 0.1, 0.1),
    seed=SEED,
)

dataloader = DataLoader(
    dataset, batch_size=PARAMS["batch_size"], shuffle=True, num_workers=2
)


def scale_and_convert(tensor):
    # Images are in the range of [-1,1]
    # So, scale back from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    return (
        tensor.clamp(0, 1).cpu().numpy()
    )  # Ensures values are within [0, 1] and move to CPU


# Function to visualize the images
def plot_images(num_images, input_image, real_target, generated_target, save_path):
    input_image = input_image[:num_images]
    real_target = real_target[:num_images]
    generated_target = generated_target[:num_images]

    # Scale and convert tensors to numpy
    input_image = scale_and_convert(input_image)
    real_target = scale_and_convert(real_target)
    generated_target = scale_and_convert(generated_target)

    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 3, 9))

    for i in range(num_images):
        # Plot input image
        axes[0, i].imshow(
            np.transpose(input_image[i], (1, 2, 0))
        )  # Convert from CxHxW to HxWxC
        axes[0, i].set_title(f"Input {i+1}")
        axes[0, i].axis("off")

        # Plot real target image
        axes[1, i].imshow(np.transpose(real_target[i], (1, 2, 0)))
        axes[1, i].set_title(f"Real {i+1}")
        axes[1, i].axis("off")

        # Plot generated target image
        axes[2, i].imshow(np.transpose(generated_target[i], (1, 2, 0)))
        axes[2, i].set_title(f"Generated {i+1}")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


real_images, target_images = next(iter(dataloader))
real_images, target_images = real_images.to(DEVICE), target_images.to(DEVICE)

out = model.get_current_visuals(real_images, target_images)
real_images, target_images, generated_images = out["real"], out["target"], out["fake"]

# Save the outputs to a file
plot_images(5, real_images, target_images, generated_images, "output_images.png")


# Function to load and preprocess a single image
def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# Define the transform for the input image
input_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ]
)

# Load and preprocess the input image
image_path = "path_to_your_image.jpg"
input_image = load_and_preprocess_image(image_path, input_transform)
input_image = input_image.to(DEVICE)

# Generate the output using the model
with torch.no_grad():
    generated_image = model.gen(input_image)

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