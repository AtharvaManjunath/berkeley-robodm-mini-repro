from __future__ import annotations

import numpy as np
import torch

from robot_bench.episode_schema import Episode, flatten_numeric


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def image_arrays(ep: Episode) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for step in ep.steps:
        for value in step.observations.values():
            if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3:
                arrays.append(value)
    return arrays


def dummy_forward(episodes: list[Episode], model: TinyModel, include_backward: bool = False) -> tuple[int, int]:
    features: list[np.ndarray] = []
    frames = 0
    steps = 0
    for ep in episodes:
        img_vals = []
        for arr in image_arrays(ep):
            tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)
            img_vals.extend([float(tensor.mean()), float(tensor.std())])
            frames += 1
        action = np.concatenate([flatten_numeric(step.action) for step in ep.steps])
        state = np.concatenate([flatten_numeric(step.observations.get("state")) for step in ep.steps])
        base = np.asarray(
            [
                np.mean(img_vals) if img_vals else 0.0,
                np.std(img_vals) if img_vals else 0.0,
                float(np.mean(action)) if action.size else 0.0,
                float(np.std(action)) if action.size else 0.0,
                float(np.mean(state)) if state.size else 0.0,
                float(np.std(state)) if state.size else 0.0,
                float(len(ep.steps)),
                float(frames),
            ],
            dtype=np.float32,
        )
        features.append(base)
        steps += len(ep.steps)
    x = torch.from_numpy(np.stack(features))
    loss = model(x).mean()
    if include_backward:
        loss.backward()
        model.zero_grad(set_to_none=True)
    else:
        _ = float(loss.detach().cpu())
    return steps, frames
