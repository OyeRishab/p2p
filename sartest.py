import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
import cv2
from PIL import Image

# Assuming these are your custom classes/modules
from pix import Pix2Pix

# Parameters
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

# Setup
SEED = PARAMS["seed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)


class ImageProcessor:
    def __init__(self, gen_ckpt, disc_ckpt):
        """
        Initialize Pix2Pix model and load checkpoints

        Args:
            gen_ckpt (str): Path to generator checkpoint
            disc_ckpt (str): Path to discriminator checkpoint
        """
        self.model = Pix2Pix(
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

        # Load generator and discriminator checkpoints
        self.model.gen.load_state_dict(
            torch.load(gen_ckpt, map_location=DEVICE, weights_only=True), strict=False
        )
        self.model.disc.load_state_dict(
            torch.load(disc_ckpt, map_location=DEVICE, weights_only=True), strict=False
        )

        self.model.to(DEVICE)
        self.model.eval()
        print("Model loaded successfully!")

        # Input transformation
        self.input_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    @staticmethod
    def scale_and_convert(tensor):
        """
        Scale tensor to [0, 1] range and convert to numpy

        Args:
            tensor (torch.Tensor): Input tensor

        Returns:
            numpy.ndarray: Scaled and converted image
        """
        tensor = (tensor + 1) / 2
        return tensor.clamp(0, 1).cpu().numpy()

    def divide_image_with_overlap(self, image, patch_size=64, overlap=16):
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
                patch = image[
                    :, :, i : min(i + patch_size, h), j : min(j + patch_size, w)
                ]
                patches.append(patch)

        return patches

    def stitch_images_with_blend(self, patches, image_shape, patch_size=64, overlap=16):
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

    def seamless_clone(self, input_image, generated_image):
        """
        Use OpenCV's seamless cloning to blend patches smoothly

        Args:
            input_image (numpy.ndarray): Original input image
            generated_image (numpy.ndarray): Generated image with patches

        Returns:
            numpy.ndarray: Smoothly blended image
        """
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

    def process_image(self, image_path, patch_size=64, overlap=16):
        """
        Process an input image through the Pix2Pix model

        Args:
            image_path (str): Path to input image
            patch_size (int): Size of patches
            overlap (int): Overlap between patches

        Returns:
            tuple: Input image, generated image, and blended image
        """
        # Load and preprocess the input image
        input_image = self.load_and_preprocess_image(image_path)
        input_image = input_image.to(DEVICE)

        # Divide the input image into overlapping patches
        patches = self.divide_image_with_overlap(input_image, patch_size, overlap)

        # Generate patches
        generated_patches = []
        resize_to_256 = transforms.Resize((256, 256))
        resize_to_64 = transforms.Resize((64, 64))

        with torch.no_grad():
            for patch in patches:
                resized_patch = resize_to_256(patch)
                generated_patch = self.model.gen(resized_patch)
                downsized_patch = resize_to_64(generated_patch)
                generated_patches.append(downsized_patch)

        # Stitch the generated patches back together with blending
        generated_image = self.stitch_images_with_blend(
            generated_patches, input_image.shape, patch_size, overlap
        )

        # Convert the tensors to numpy arrays for visualization
        input_image_np = self.scale_and_convert(input_image)
        generated_image_np = self.scale_and_convert(generated_image)

        # Optional: Use seamless cloning for further blending
        blended_image = self.seamless_clone(input_image_np, generated_image_np)

        return input_image_np, generated_image_np, blended_image

    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess an image

        Args:
            image_path (str): Path to input image

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = Image.open(image_path).convert("RGB")
        image = self.input_transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image

    def visualize_results(
        self, input_image, generated_image, blended_image, output_path
    ):
        """
        Visualize and save processing results

        Args:
            input_image (numpy.ndarray): Original input image
            generated_image (numpy.ndarray): Generated image
            blended_image (numpy.ndarray): Blended image
            output_path (str): Path to save output image
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot input image
        axes[0].imshow(np.transpose(input_image[0], (1, 2, 0)))
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # Plot generated image
        axes[1].imshow(np.transpose(generated_image[0], (1, 2, 0)))
        axes[1].set_title("Generated Image")
        axes[1].axis("off")

        # Plot blended image
        axes[2].imshow(blended_image)
        axes[2].set_title("Blended Image")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)


def main():
    # Initialize the image processor
    processor = ImageProcessor(
        gen_ckpt="/content/drive/MyDrive/pix2pix_gen_220.pth",
        disc_ckpt="/content/drive/MyDrive/pix2pix_disc_220.pth",
    )

    # Process the image
    input_image, generated_image, blended_image = processor.process_image(
        image_path="test.jpg", patch_size=64, overlap=16
    )

    # Visualize results
    processor.visualize_results(
        input_image, generated_image, blended_image, output_path="test_output_image.png"
    )


if __name__ == "__main__":
    main()
