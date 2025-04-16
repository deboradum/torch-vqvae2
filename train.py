import os
import time
import torch
import argparse

import numpy as np
import torch.optim as optim

from PIL import Image

from models.vqvae import VQVAE2, VQVAE2_large
from dataset import get_dataloaders

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def save_snapshot(net, loader, path="results/0/"):
    net.eval()
    os.makedirs(path, exist_ok=True)

    ctr = 0
    for i, (batch, _) in enumerate(loader):
        if i > 4:
            return
        batch = batch.to(device)

        with torch.no_grad():
            outputs = net(batch)
            x_hat, *losses, perplexity = outputs

        batch = batch.cpu()
        x_hat = x_hat.cpu()

        for j in range(batch.size(0)):
            orig = batch[j]
            recon = x_hat[j]

            # Handle single-channel or multi-channel images
            if orig.size(0) == 1:
                orig = orig.squeeze(0)
                recon = recon.squeeze(0)
                combined = np.concatenate([orig.numpy(), recon.numpy()], axis=1)
                img = Image.fromarray(np.clip(combined * 255, 0, 255).astype(np.uint8))
            else:
                orig = orig.permute(1, 2, 0).numpy()
                recon = recon.permute(1, 2, 0).numpy()
                combined = np.concatenate([orig, recon], axis=1)
                img = Image.fromarray(np.clip(combined * 255, 0, 255).astype(np.uint8))

            img.save(os.path.join(path, f"{ctr}.png"))
            ctr += 1


def loss_fn(net, X, metrics):
    outputs = net(X)
    x_hat, *losses, perplexity = outputs
    recon_loss = ((x_hat - X) ** 2).mean() / x_train_var
    loss = recon_loss + sum(losses)

    metrics["total_loss"].append(loss.item())
    metrics["recon_loss"].append(recon_loss.item())
    metrics["perplexity"].append(perplexity.item())

    return loss


def get_avg(values):
    return sum(values) / len(values)


def get_avg_metrics(window, metrics):
    total_loss = get_avg(metrics["total_loss"][-window:-1])
    recon_loss = get_avg(metrics["recon_loss"][-window:-1])
    perplexity = get_avg(metrics["perplexity"][-window:-1])

    return total_loss, recon_loss, perplexity


def train(
    epochs,
    net,
    optimizer,
    train_loader,
    test_loader,
    output_dir,
    log_every=100,
    save_every=1,
):
    s = time.perf_counter()
    for epoch in range(epochs):
        metrics = {"total_loss": [], "recon_loss": [], "perplexity": []}
        net.train(True)
        for i, (X, _) in enumerate(train_loader):
            X = X.to(device)
            # X = X.permute(0, 2, 3, 1)

            optimizer.zero_grad()
            loss = loss_fn(net, X, metrics)
            loss.backward()
            optimizer.step()

            if i % log_every == 0 and i > 0:
                took = time.perf_counter() - s
                time_per_it = took / log_every
                avg_total_loss, avg_recon_loss, avg_perplexity = get_avg_metrics(log_every, metrics)
                print(
                    f"Epoch {epoch}, step {i}/{len(train_loader)} - loss: {avg_total_loss:.5f}, recon_loss: {avg_recon_loss:.5f}, perplexity: {avg_perplexity:.5f}. Took {took:.3f}s ({time_per_it:.3f} s/it)"
                )
                save_snapshot(net, test_loader, path=f"{output_dir}/{epoch}_{i}")
                s = time.perf_counter()

        if epoch % save_every == 0:
            model_path = f"{output_dir}/checkpoint_epoch_{epoch}.npz"
            print(f"Saving model in {model_path}")
            torch.save(net.state_dict(), model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=str, default="small")
    parser.add_argument("dataset", type=str, default="geoguessr")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    print(f"Using device {device}")

    args = parse_args()
    metrics = {"total_loss": [], "recon_loss": [], "perplexity": []}
    if args.size == "small":
        net = VQVAE2(128, 64, 2, 512, 32, 0.25)
    elif args.size == "large":
        net = VQVAE2_large(128, 64, 2, 512, 64, 0.25)
    else:
        raise NotImplementedError("size argument should be 'small' or 'large'")

    net.to(device)
    if args.load_checkpoint:
        net.load_state_dict(torch.load(args.load_checkpoint, weights_only=True))

    optimizer = optim.Adam(net.parameters(), lr=3e-4)

    train_loader, test_loader, x_train_var = get_dataloaders(args.dataset, args.size, batch_size=8)
    train(
        epochs=10,
        net=net,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        output_dir=args.output_dir,
        log_every=args.log_every,
        save_every=args.save_every,
    )
