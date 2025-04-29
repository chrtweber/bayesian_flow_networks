import os
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


from typing import Callable, Tuple
from torchtyping import TensorType as Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from torch_bfn import ContinuousBFN, LinearNetwork
from torch_bfn.utils import EMA, norm_denorm, str_to_torch_dtype


def load_sachs_dataset() -> (
    Tuple[
        DataLoader, DataLoader, Callable[[Tensor["B", "D"]], Tensor["B", "D"]]
    ]
):
    print("Loading Sachs dataset...")

    # Step 1: Load observations
    X_np = np.load(
        "examples/sachs/continuous/data1.npy"
    )  # or adjust path if needed

    print("Shape of data:", X_np.shape)  # Should be (n_samples, 11)

    # Step 2: Convert to tensor
    X = torch.tensor(X_np, dtype=torch.float32)

    # Step 3: Normalize
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True)
    X_norm = (X - X_mean) / X_std

    # Step 4: Dataset and DataLoaders
    dset = TensorDataset(X_norm)

    train_size = int(0.8 * len(dset))
    val_size = len(dset) - train_size
    train_dset, val_dset = random_split(dset, [train_size, val_size])

    train_loader = DataLoader(train_dset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=128, shuffle=False)

    denorm = lambda x: x * X_std + X_mean  # Denormalization function

    pca = PCA(n_components=2)
    samples_real = X.numpy()
    samples_real_2d = pca.fit_transform(samples_real)

    return (train_loader, val_loader, denorm, X_norm, pca)


def save_checkpoint(model, epoch: int, path: str = "trained_models"):
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, f"bfn_epoch_{epoch}.pt")
    torch.save(model.state_dict(), fpath)
    print(f"Saved checkpoint at: {fpath}")


def load_checkpoint(model, epoch: int, path: str = "trained_models"):
    fpath = os.path.join(path, f"bfn_epoch_{epoch}.pt")
    model.load_state_dict(torch.load(fpath))
    print(f"Loaded checkpoint from: {fpath}")
    return model


def plot_highdim_samples(
    denormed_samples: Tensor["B", "D"],
    pca: PCA,
    fpath: str = "outputs/samples_pca.png",
    pause: float = 0.1,
):
    samples = denormed_samples.numpy()
    samples_2d = pca.fit_transform(samples)
    plt.figure(figsize=(6, 4))
    plt.scatter(samples_2d[:, 0], samples_2d[:, 1], edgecolor="k", alpha=0.5)
    plt.title("BFN Samples (PCA) " + fpath)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.xlim(-400, 600)
    plt.ylim(-100, 200)

    plt.show(block=False)
    plt.pause(pause)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    plt.savefig(fpath)
    plt.close()


def train(
    model: ContinuousBFN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    denorm: Callable[[Tensor["B", "D"]], Tensor["B", "D"]],
    pca: PCA,
    start_epoch: int = 0,
    epochs: int = 100,
    device_str: str = "cpu",
    dtype_str: str = "float32",
):
    device = torch.device(device_str)
    dtype = str_to_torch_dtype(dtype_str)
    ema = EMA(0.9)

    model.to(device, dtype)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ema.register(model)

    for epoch in range(start_epoch, start_epoch + epochs + 1):
        loss = None
        for batch in train_loader:
            X = batch[0].to(device, dtype)
            loss = model.loss(X, sigma_1=0.01).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

        if epoch % 100 == 0:
            assert loss is not None
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            samples = model.sample(1000, sigma_1=0.01, n_timesteps=10)
            plot_highdim_samples(
                denorm(samples.cpu()),
                pca,
                f"outputs/sachs_samples_{epoch}.png",
            )

        if epoch % 1000 == 0:
            save_checkpoint(model, epoch)


if __name__ == "__main__":
    device = "cpu"
    dtype = "float32"

    train_loader, val_loader, denorm, X_norm, pca = load_sachs_dataset()

    net = LinearNetwork(
        dim=11,  # Number of variables in Sachs dataset
        hidden_dims=[512, 512],
        sin_dim=16,
        time_dim=64,
        random_time_emb=False,
        dropout_p=0.0,
    )

    model = ContinuousBFN(
        dim=11,
        net=net,
        device_str=device,
        dtype_str=dtype,
    )

    start_epoch = 5000
    # Load previous checkpoint (if exists)
    load_checkpoint(model, epoch=start_epoch)

    # Plot real Sachs data using existing function
    plot_highdim_samples(
        denorm(X_norm), pca, fpath="outputs/sachs_real_samples.png", pause=3
    )

    train(
        model,
        train_loader,
        val_loader,
        denorm,
        pca,
        start_epoch=start_epoch,
        epochs=2000,
        device_str=device,
        dtype_str=dtype,
    )
