


# %%
if __name__ == "__main__":
    import os
    import shutil
    import tempfile
    import pydicom


    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from monai import transforms
    from monai.apps import MedNISTDataset
    from monai.config import print_config
    from monai.data import CacheDataset, DataLoader, Dataset
    from monai.utils import first, set_determinism
    from torch import nn
    from torch.cuda.amp import GradScaler, autocast
    from tqdm import tqdm

    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    from generative.losses import PatchAdversarialLoss, PerceptualLoss
    from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
    from generative.networks.schedulers import DDPMScheduler

    print_config()

# %%


    # for reproducibility purposes set a seed
    set_determinism(42)

    # %%
    os.environ["MONAI_DATA_DIRECTORY"] = r"C:\Users\u0151811\Documents\PhD\Monai_DATA/"
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    # %%
    class CustomDICOMDataset:
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.class_names = self._find_classes()
            self.data = self._load_dataset()

        def _find_classes(self):
            class_names = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
            return class_names

        def _load_dataset(self):
            data = []  # List to store file paths and class names
            for class_name in self.class_names:
                class_path = os.path.join(self.root_dir, class_name)
                for file in os.listdir(class_path):
                    if file.endswith(".dcm"):
                        file_path = os.path.join(class_path, file)
                        data.append({"image": file_path, "class_name": class_name})  # Store file paths instead of image data
            return data

        def __getitem__(self, index):
            item = self.data[index]
            file_path = item['image']
            class_name = item['class_name']
            return {'image': file_path, 'class_name': class_name}  # Return file paths instead of image data

        def __len__(self):
            return len(self.data)

    # Create a custom dataset
    train_data = CustomDICOMDataset(root_dir)

    train_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "Ax"]

    # %%
    # Define image_size as a single tuple
    image_size = (320, 291, 1)

    # Modify the transformation pipeline
    train_transforms = transforms.Compose([
        transforms.LoadImageD(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.RandAffined(
            keys=["image"],
            rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
            translate_range=[(-1, 1), (-1, 1)],
            scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
            spatial_size=[image_size[0], image_size[1],image_size[2]],  # Adjusted spatial_size
            padding_mode="border",
            prob=0.5,
        ),
        transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
        transforms.Resized(keys=["low_res_image"], spatial_size=(80, 72,1)),
    ])

    # Apply the transformation to the dataset without caching
    train_ds = Dataset(data=train_datalist, transform=train_transforms)

    #%% extra check for dataset
    sample = train_ds[0]
    original_image = sample["image"]
    low_res_image = sample["low_res_image"]
    
    plt.imshow(original_image[0, :, :, :], cmap="gray")
    plt.title("Original Image")
    plt.show()

    plt.imshow(low_res_image[0,:,:,:], cmap="gray")
    plt.title("Low resolution Image")
    plt.show()
    
    #%%
    train_loader = DataLoader(train_ds, batch_size=5, shuffle=True, num_workers=0)
    # # Apply the transformation to the dataset
    # train_ds = CacheDataset(data=train_datalist, transform=train_transforms)
    # train_loader = DataLoader(train_ds, batch_size=5, shuffle=True, num_workers=4, persistent_workers=True)
    # %%
    #Plot 3 examples from the training set
    check_data = first(train_loader)
    fig, ax = plt.subplots(nrows=1, ncols=3)
    for i in range(3):
        ax[i].imshow(check_data["image"][i, 0, :, :], cmap="gray")
        ax[i].axis("off")

    # %%
    # Plot 3 examples from the training set in low resolution
    fig, ax = plt.subplots(nrows=1, ncols=3)
    for i in range(3):
        ax[i].imshow(check_data["low_res_image"][i, 0, :, :], cmap="gray")
        ax[i].axis("off")

    # %% [markdown]
    # ## Create data loader for validation set
    val_data = CustomDICOMDataset(root_dir)
    val_datalist = [{"image": item["image"]} for item in val_data.data if item["class_name"] == "Ax"]

    val_transforms = transforms.Compose(
        [
            transforms.LoadImageD(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(keys=["low_res_image"], spatial_size=(80,72,1)),
        ]
    )
    val_ds = Dataset(data=val_datalist, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=5, shuffle=True, num_workers=0)

    # %% [markdown]
    # ## Define the autoencoder network and training components

    # %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # %%
    autoencoderkl = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(256, 512, 512),
        latent_channels=3,
        num_res_blocks=2,
        norm_num_groups=32,
        attention_levels=(False, False, True),
    )
    autoencoderkl = autoencoderkl.to(device)

    discriminator = PatchDiscriminator(spatial_dims=3, in_channels=1, num_layers_d=3, num_channels=64)
    discriminator = discriminator.to(device)


    # %%
    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="alex")
    perceptual_loss.to(device)
    perceptual_weight = 0.002

    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.005

    optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=5e-5)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # %%
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    # %% [markdown]
    # ## Train Autoencoder

    # %%
    kl_weight = 1e-6
    n_epochs = 75
    val_interval = 10
    autoencoder_warm_up_n_epochs = 10

    for epoch in range(n_epochs):
        autoencoderkl.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            batch_size, extra, height, width, channels = images.size()
            #images = images.view(batch_size * extra, channels, height, width)
            optimizer_g.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                reconstruction, z_mu, z_sigma = autoencoderkl(images)

                recons_loss = F.l1_loss(reconstruction.float(), images.float())
                p_loss = perceptual_loss(reconstruction.float(), images.float())
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)

                if epoch > autoencoder_warm_up_n_epochs:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += adv_weight * generator_loss

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            if epoch > autoencoder_warm_up_n_epochs:
                optimizer_d.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                    loss_d = adv_weight * discriminator_loss

                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()

            epoch_loss += recons_loss.item()
            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

            progress_bar.set_postfix(
                {
                    "recons_loss": epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / (step + 1),
                    "disc_loss": disc_epoch_loss / (step + 1),
                }
            )

        if (epoch + 1) % val_interval == 0:
            autoencoderkl.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(device)
                    reconstruction, z_mu, z_sigma = autoencoderkl(images)
                    recons_loss = F.l1_loss(images.float(), reconstruction.float())
                    val_loss += recons_loss.item()

            val_loss /= val_step
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # ploting reconstruction
            plt.figure(figsize=(2, 2))
            plt.imshow(torch.cat([images[0, 0].cpu(), reconstruction[0, 0].cpu()], dim=1), vmin=0, vmax=1, cmap="gray")
            plt.tight_layout()
            plt.axis("off")
            plt.show()

    progress_bar.close()

    del discriminator
    del perceptual_loss
    torch.cuda.empty_cache()

    # %% [markdown]
    # ## Rescaling factor
    #
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) became crucial in image-to-image translation models (such as the ones used for super-resolution). For this reason, we will compute the component-wise standard deviation to be used as scaling factor.

    # %%
    with torch.no_grad():
        with autocast(enabled=True):
            z = autoencoderkl.encode_stage_2_inputs(check_data["image"].to(device))

    print(f"Scaling factor set to {1/torch.std(z)}")
    scale_factor = 1 / torch.std(z)

    # %% [markdown]
    # ## Train Diffusion Model
    #
    # In order to train the diffusion model to perform super-resolution, we will need to concatenate the latent representation of the high-resolution with the low-resolution image. For this, we create a Diffusion model with `in_channels=4`. Since only the outputted latent representation is interesting, we set `out_channels=3`.

    # %%
    unet = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=3,
        num_res_blocks=2,
        num_channels=(256, 256, 512, 1024),
        attention_levels=(False, False, True, True),
        num_head_channels=(0, 0, 64, 64),
    )
    unet = unet.to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)

    # %% [markdown]
    # As mentioned, we will use the conditioned augmentation (introduced in [2] section 3 and used on Stable Diffusion Upscalers and Imagen Video [3] Section 2.5) as it has been shown critical for cascaded diffusion models, as well for super-resolution tasks. For this, we apply Gaussian noise augmentation to the low-resolution images. We will use a scheduler `low_res_scheduler` to add this noise, with the `t` step defining the signal-to-noise ratio and use the `t` value to condition the diffusion model (inputted using `class_labels` argument).

    # %%
    low_res_scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)

    max_noise_level = 350

    # %%
    optimizer = torch.optim.Adam(unet.parameters(), lr=5e-5)

    scaler_diffusion = GradScaler()

    n_epochs = 200
    val_interval = 20
    epoch_loss_list = []
    val_epoch_loss_list = []

    for epoch in range(n_epochs):
        unet.train()
        autoencoderkl.eval()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            low_res_image = batch["low_res_image"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                with torch.no_grad():
                    latent = autoencoderkl.encode_stage_2_inputs(images) * scale_factor

                # Noise augmentation
                noise = torch.randn_like(latent).to(device)
                low_res_noise = torch.randn_like(low_res_image).to(device)
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (latent.shape[0],), device=latent.device).long()
                low_res_timesteps = torch.randint(
                    0, max_noise_level, (low_res_image.shape[0],), device=low_res_image.device
                ).long()

                noisy_latent = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)
                noisy_low_res_image = scheduler.add_noise(
                    original_samples=low_res_image, noise=low_res_noise, timesteps=low_res_timesteps
                )

                latent_model_input = torch.cat([noisy_latent, noisy_low_res_image], dim=1)

                noise_pred = unet(x=latent_model_input, timesteps=timesteps, class_labels=low_res_timesteps)
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler_diffusion.scale(loss).backward()
            scaler_diffusion.step(optimizer)
            scaler_diffusion.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))

        if (epoch + 1) % val_interval == 0:
            unet.eval()
            val_loss = 0
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)
                low_res_image = batch["low_res_image"].to(device)

                with torch.no_grad():
                    with autocast(enabled=True):
                        latent = autoencoderkl.encode_stage_2_inputs(images) * scale_factor
                        # Noise augmentation
                        noise = torch.randn_like(latent).to(device)
                        low_res_noise = torch.randn_like(low_res_image).to(device)
                        timesteps = torch.randint(
                            0, scheduler.num_train_timesteps, (latent.shape[0],), device=latent.device
                        ).long()
                        low_res_timesteps = torch.randint(
                            0, max_noise_level, (low_res_image.shape[0],), device=low_res_image.device
                        ).long()

                        noisy_latent = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)
                        noisy_low_res_image = scheduler.add_noise(
                            original_samples=low_res_image, noise=low_res_noise, timesteps=low_res_timesteps
                        )

                        latent_model_input = torch.cat([noisy_latent, noisy_low_res_image], dim=1)
                        noise_pred = unet(x=latent_model_input, timesteps=timesteps, class_labels=low_res_timesteps)
                        loss = F.mse_loss(noise_pred.float(), noise.float())

                val_loss += loss.item()
            val_loss /= val_step
            val_epoch_loss_list.append(val_loss)
            print(f"Epoch {epoch} val loss: {val_loss:.4f}")

            # Sampling image during training
            sampling_image = low_res_image[0].unsqueeze(0)
            latents = torch.randn((1, 3, 16, 16)).to(device)
            low_res_noise = torch.randn((1, 1, 16, 16)).to(device)
            noise_level = 20
            noise_level = torch.Tensor((noise_level,)).long().to(device)
            noisy_low_res_image = scheduler.add_noise(
                original_samples=sampling_image,
                noise=low_res_noise,
                timesteps=torch.Tensor((noise_level,)).long().to(device),
            )

            scheduler.set_timesteps(num_inference_steps=1000)
            for t in tqdm(scheduler.timesteps, ncols=110):
                with torch.no_grad():
                    with autocast(enabled=True):
                        latent_model_input = torch.cat([latents, noisy_low_res_image], dim=1)
                        noise_pred = unet(
                            x=latent_model_input, timesteps=torch.Tensor((t,)).to(device), class_labels=noise_level
                        )
                    latents, _ = scheduler.step(noise_pred, t, latents)

            with torch.no_grad():
                decoded = autoencoderkl.decode_stage_2_outputs(latents / scale_factor)

            low_res_bicubic = nn.functional.interpolate(sampling_image, (64, 64), mode="bicubic")
            plt.figure(figsize=(2, 2))
            plt.style.use("default")
            plt.imshow(
                torch.cat([images[0, 0].cpu(), low_res_bicubic[0, 0].cpu(), decoded[0, 0].cpu()], dim=1),
                vmin=0,
                vmax=1,
                cmap="gray",
            )
            plt.tight_layout()
            plt.axis("off")
            plt.show()


    # %% [markdown]
    # ### Plotting sampling example

    # %%
    # Sampling image during training
    unet.eval()
    num_samples = 3
    validation_batch = first(val_loader)

    images = validation_batch["image"].to(device)
    sampling_image = validation_batch["low_res_image"].to(device)[:num_samples]

    # %%
    latents = torch.randn((num_samples, 3, 16, 16)).to(device)
    low_res_noise = torch.randn((num_samples, 1, 16, 16)).to(device)
    noise_level = 10
    noise_level = torch.Tensor((noise_level,)).long().to(device)
    noisy_low_res_image = scheduler.add_noise(
        original_samples=sampling_image, noise=low_res_noise, timesteps=torch.Tensor((noise_level,)).long().to(device)
    )
    scheduler.set_timesteps(num_inference_steps=1000)
    for t in tqdm(scheduler.timesteps, ncols=110):
        with torch.no_grad():
            with autocast(enabled=True):
                latent_model_input = torch.cat([latents, noisy_low_res_image], dim=1)
                noise_pred = unet(x=latent_model_input, timesteps=torch.Tensor((t,)).to(device), class_labels=noise_level)

            # 2. compute previous image: x_t -> x_t-1
            latents, _ = scheduler.step(noise_pred, t, latents)

    with torch.no_grad():
        decoded = autoencoderkl.decode_stage_2_outputs(latents / scale_factor)

    # %%
    low_res_bicubic = nn.functional.interpolate(sampling_image, (64, 64), mode="bicubic")
    fig, axs = plt.subplots(num_samples, 3, figsize=(8, 8))
    axs[0, 0].set_title("Original image")
    axs[0, 1].set_title("Low-resolution Image")
    axs[0, 2].set_title("Outputted image")
    for i in range(0, num_samples):
        axs[i, 0].imshow(images[i, 0].cpu(), vmin=0, vmax=1, cmap="gray")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(low_res_bicubic[i, 0].cpu(), vmin=0, vmax=1, cmap="gray")
        axs[i, 1].axis("off")
        axs[i, 2].imshow(decoded[i, 0].cpu(), vmin=0, vmax=1, cmap="gray")
        axs[i, 2].axis("off")
    plt.tight_layout()

    # %% [markdown]
    # ### Clean-up data directory

    # %%
    if directory is None:
        shutil.rmtree(root_dir)
