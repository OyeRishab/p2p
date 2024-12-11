import torch
from pix import Pix2Pix
from split import Sentinel
from torch.utils.data import DataLoader
from torchvision.transforms import v2


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
    "batch_size": 16,
    "epochs": 300,
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
model.to(DEVICE)

model = torch.compile(model)

base_path = "/"
dataset_root_dir = "/v_2"
split_save_path = "/split.json"


train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ]
)

dataset = Sentinel(
    root_dir=dataset_root_dir,
    split_type="train",
    transform=train_transforms,
    split_mode="random",
    split_ratio=(0.9, 0.0, 0.1),
    seed=SEED,
)
dataset.save_split(output_file=split_save_path)

dataloader = DataLoader(
    dataset, batch_size=PARAMS["batch_size"], shuffle=True, num_workers=4
)

# Train the model
num_epochs = PARAMS["epochs"]
len_batch = len(dataloader)
save_freq = 5

for epoch in range(181, num_epochs + 1):
    total_lossD = 0.0
    total_lossG = 0.0
    total_lossG_GAN = 0.0
    total_lossG_L1 = 0.0
    model.train()
    for real_images, target_images in dataloader:
        real_images, target_images = real_images.to(DEVICE), target_images.to(DEVICE)
        losses = model.train_step(real_images, target_images)
        total_lossD += losses["loss_D"]
        total_lossG += losses["loss_G"]
        total_lossG_GAN += losses["loss_G_GAN"]
        total_lossG_L1 += losses["loss_G_L1"]

    # Train
    loss_D = total_lossD / len_batch
    loss_G = total_lossG / len_batch
    loss_G_GAN = total_lossG_GAN / len_batch
    loss_G_L1 = total_lossG_L1 / len_batch

    # Log the losses
    print(f"Epoch [{epoch}/{num_epochs}] - Loss_D: {loss_D:.4f}, Loss_G: {loss_G:.4f}")

    if epoch % save_freq == 0:
        gen_path = base_path + f"pix2pix_gen_{epoch}.pth"
        disc_path = base_path + f"pix2pix_disc_{epoch}.pth"
        model.save_model(gen_path=gen_path, disc_path=disc_path)
        model.save_optimizer(
            gen_opt_path="pix2pix_gen_opt.pth", disc_opt_path="pix2pix_disc_opt.pth"
        )
