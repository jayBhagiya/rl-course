from __future__ import annotations

import argparse

import torch
from torch import nn

from rl_course.common import emit_json, positive_int


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def target_function(inputs: torch.Tensor) -> torch.Tensor:
    return torch.stack((inputs.square().sum(dim=-1), inputs.sum(dim=-1)), dim=-1)


def train(
    steps: int = 2_000,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, float | int | str]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(seed)
    model = Network().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    test_inputs = torch.rand(256, 3, device=device)
    with torch.no_grad():
        initial_loss = nn.functional.mse_loss(model(test_inputs), target_function(test_inputs)).item()
    model.train()
    for _ in range(steps):
        inputs = torch.rand(batch_size, 3, device=device)
        loss = nn.functional.mse_loss(model(inputs), target_function(inputs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss = nn.functional.mse_loss(model(test_inputs), target_function(test_inputs)).item()
    return {
        "steps": steps,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "initial_mse": initial_loss,
        "final_mse": final_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a small network to two analytic functions.")
    parser.add_argument("--steps", type=positive_int, default=2_000)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    emit_json(train(args.steps, args.batch_size, args.seed, args.device), args.output)


if __name__ == "__main__":
    main()
