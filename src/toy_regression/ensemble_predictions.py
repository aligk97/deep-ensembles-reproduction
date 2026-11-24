from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Prediction:
    mean: torch.Tensor
    total_variance: torch.Tensor
    aleatoric: torch.Tensor | None
    epistemic: torch.Tensor | None
    member_means: torch.Tensor
    member_variances: torch.Tensor | None

    @property
    def std(self):
        return self.total_variance.sqrt()


def aggregate_predictions(member_means, member_variances=None):
    mean = member_means.mean(dim=0)
    epistemic = member_means.var(dim=0, unbiased=False) if len(member_means) > 1 else None
    aleatoric = member_variances.mean(dim=0) if member_variances is not None else None
    if aleatoric is None and epistemic is None:
        raise ValueError("At least two point regressors or one Gaussian member are required")
    if aleatoric is None:
        total = epistemic
    elif epistemic is None:
        total = aleatoric
    else:
        total = aleatoric + epistemic
    return Prediction(mean, total, aleatoric, epistemic, member_means, member_variances)


@torch.inference_mode()
def predict(models, x, gaussian, device):
    outputs = [model(x.to(device)) for model in models]
    if gaussian:
        means, variances = zip(*outputs)
        variances = torch.stack(variances).cpu()
    else:
        means, variances = outputs, None
    result = aggregate_predictions(torch.stack(means).cpu(), variances)
    for name, tensor in vars(result).items():
        if tensor is not None and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite prediction array: {name}")
    return result


def save_predictions(path, prediction, split, condition, seed, save_members):
    values = {
        key: value.numpy() for key, value in vars(prediction).items()
        if value is not None and (save_members or not key.startswith("member_"))
    }
    np.savez_compressed(
        path, **split.arrays(), **values, std=prediction.std.numpy(),
        condition=condition.name, outer_seed=seed,
        aleatoric_available=prediction.aleatoric is not None,
        epistemic_available=prediction.epistemic is not None,
    )
