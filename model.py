import os
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

model.to(DEVICE)

model = torch.compile(model)

base_path = "/"
dataset_root_dir = "/content/v_2"
split_save_path = "split.json"

train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ]
)

if __name__ == "__main__":
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
        dataset, batch_size=PARAMS["batch_size"], shuffle=True, num_workers=2
    )

    gen_ckpt = base_path + "pix2pix_gen_220.pth"
    disc_ckpt = base_path + "pix2pix_disc_220.pth"

    if os.path.exists(gen_ckpt) and os.path.exists(disc_ckpt):
        model.gen.load_state_dict(
            torch.load(gen_ckpt, map_location=DEVICE, weights_only=True), strict=False
        )
        model.disc.load_state_dict(
            torch.load(disc_ckpt, map_location=DEVICE, weights_only=True), strict=False
        )
        print("Loaded succesfully!")

    load_gen_opt_path = base_path + "pix2pix_gen_opt.pth"
    load_disc_opt_path = base_path + "pix2pix_disc_opt.pth"

    if os.path.exists(load_gen_opt_path) and os.path.exists(load_disc_opt_path):
        model.optimizer_G.load_state_dict(torch.load(load_gen_opt_path))
        model.optimizer_D.load_state_dict(torch.load(load_disc_opt_path))
        print("Loaded saved optimizers.")

    # Train the model
    num_epochs = PARAMS["epochs"]
    len_batch = len(dataloader)
    save_freq = 20

    for epoch in range(0, num_epochs + 1):
        total_lossD = 0.0
        total_lossG = 0.0
        total_lossG_GAN = 0.0
        total_lossG_L1 = 0.0
        model.train()
        for real_images, target_images in dataloader:
            real_images, target_images = real_images.to(DEVICE), target_images.to(
                DEVICE
            )
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
        print(
            f"Epoch [{epoch}/{num_epochs}] - Loss_D: {loss_D:.4f}, Loss_G: {loss_G:.4f}"
        )

        if epoch % save_freq == 0:
            gen_path = base_path + f"pix2pix_gen_{epoch}.pth"
            disc_path = base_path + f"pix2pix_disc_{epoch}.pth"
            model.save_model(gen_path=gen_path, disc_path=disc_path)
            model.save_optimizer(
                gen_opt_path="pix2pix_gen_opt.pth", disc_opt_path="pix2pix_disc_opt.pth"
            )
            print(f"Model saved at epoch {epoch}")
            print("Average Loss_D : ", loss_D)
            print("Average Loss_G : ", loss_G)
