import torch
import matplotlib.pyplot as plt
import numpy as np
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

# Initialize and load the Pix2Pix model
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
model.gen.load_state_dict(torch.load(gen_ckpt, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

# Define transforms
input_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ]
)


# Function to load and preprocess the image
def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# Function to scale and convert tensor to NumPy array
def scale_and_convert(tensor):
    tensor = tensor.cpu().detach()
    tensor = (tensor + 1) / 2  # Rescale to [0,1]
    np_array = tensor.numpy()
    return np_array


# Load the input SAR image
image_path = "test_sar_image.jpg"
input_image = load_and_preprocess_image(image_path, input_transform).to(DEVICE)

# Generate the output using the model
with torch.no_grad():
    generated_image = model.generate(input_image, is_scaled=True).to(DEVICE)

# Convert input_image and generated_image to appropriate scales
input_image_np = scale_and_convert(input_image)
generated_image_np = scale_and_convert(generated_image)

# Convert input_image to grayscale
grayscale_image = input_image.mean(dim=1, keepdim=True)


# Apply edge detection using Sobel filters
def sobel_edge_detection(image):
    sobel_kernel_x = (
        torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=torch.float32,
            device=image.device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    sobel_kernel_y = (
        torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            dtype=torch.float32,
            device=image.device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    edges_x = torch.nn.functional.conv2d(image, sobel_kernel_x, padding=1)
    edges_y = torch.nn.functional.conv2d(image, sobel_kernel_y, padding=1)
    edge_map = torch.sqrt(edges_x**2 + edges_y**2)
    return edge_map


edge_map = sobel_edge_detection(grayscale_image)

# Normalize edge_map to [0,1]
edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())

# Expand edge_map to 3 channels
edge_map_rgb = edge_map.repeat(1, 3, 1, 1)

# Blend edge_map with generated_image
alpha = 0.5  # Blending factor
blended_image = generated_image * (1 - alpha) + edge_map_rgb * alpha

# Convert blended_image to NumPy array
blended_image_np = scale_and_convert(blended_image)

# Plot and save the resulting blended image
plt.imshow(np.transpose(blended_image_np[0], (1, 2, 0)))
plt.title("Blended Image")
plt.axis("off")
plt.tight_layout()
plt.savefig("blended_output_image.png")
plt.show()
