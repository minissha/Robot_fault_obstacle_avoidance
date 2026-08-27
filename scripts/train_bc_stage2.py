"""
Train a PointCloudBCPolicy on demos collected by collect_demos_stage2.py.

Usage:
    python scripts/train_bc_stage2.py --demos demos/stage2_demos.npz \
        --out checkpoints/bc_stage2.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from policies.bc_policy import PointCloudBCPolicy
from utils.seeding import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=str, required=True)
    parser.add_argument("--out", type=str, default="checkpoints/bc_stage2.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_global_seed(args.seed)

    data = np.load(args.demos, allow_pickle=True)
    obs = torch.as_tensor(data["observations"], dtype=torch.float32)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32)
    n_points = int(data["n_points"])
    point_channels = int(data["point_channels"])
    proprio_dim = int(data["proprio_dim"])
    print(f"Loaded {len(obs)} transitions. obs_dim={obs.shape[1]} "
          f"(n_points={n_points}, point_channels={point_channels}, "
          f"proprio_dim={proprio_dim}), act_dim={actions.shape[1]}")

    expected_dim = n_points * point_channels + n_points + proprio_dim
    assert obs.shape[1] == expected_dim, (
        f"obs_dim mismatch: got {obs.shape[1]}, expected {expected_dim}. "
        "Demos may have been collected with a different PerceptionConfig."
    )

    dataset = TensorDataset(obs, actions)
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = PointCloudBCPolicy(
        n_points=n_points, point_channels=point_channels,
        proprio_dim=proprio_dim, act_dim=actions.shape[1],
        embed_dim=args.embed_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for obs_b, act_b in train_loader:
            optimizer.zero_grad()
            pred = model(obs_b)
            loss = loss_fn(pred, act_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(obs_b)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_b, act_b in val_loader:
                pred = model(obs_b)
                val_loss += loss_fn(pred, act_b).item() * len(obs_b)
        val_loss /= max(n_val, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "n_points": n_points,
                "point_channels": point_channels,
                "proprio_dim": proprio_dim,
                "embed_dim": args.embed_dim,
            }, args.out)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d} | train_loss {train_loss:.5f} | "
                  f"val_loss {val_loss:.5f}")

    print(f"\nBest val_loss={best_val_loss:.5f}. Saved best model to {args.out}")


if __name__ == "__main__":
    main()
