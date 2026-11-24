"""CMP4509 Deep Ensembles | Toy Regression - Step 1: dataset generation (Windows compatible).

Run (PowerShell, inside venv): python .\01_generate_data.py
Output: outputs/toy_training_data.csv and outputs/toy_training_data.png
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import torch

DATA_SEED = 42
N = 20
TRAIN_X_MIN, TRAIN_X_MAX = -4.0, 4.0  # Our approximate choice; not explicitly given in paper.
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(DATA_SEED)

    x_train = TRAIN_X_MIN + (TRAIN_X_MAX - TRAIN_X_MIN) * torch.rand(
        (N, 1), generator=generator
    )
    noise = 3.0 * torch.randn((N, 1), generator=generator)
    y_train = x_train.pow(3) + noise

    x_plot = torch.linspace(-6.0, 6.0, 400).unsqueeze(1)
    y_true = x_plot.pow(3)

    csv_path = OUTPUT_DIR / "toy_training_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "noise"])
        writer.writerows(zip(x_train.flatten().tolist(),
                             y_train.flatten().tolist(), noise.flatten().tolist()))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_plot.flatten().numpy(), y_true.flatten().numpy(), color="blue", label="Ground Truth: y = x^3")
    ax.scatter(x_train.flatten().numpy(), y_train.flatten().numpy(), color="red", label="Training Data (20 samples)", zorder=3)
    ax.axvline(TRAIN_X_MIN, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(TRAIN_X_MAX, color="gray", linestyle="--", alpha=0.5)
    ax.set(xlabel="x", ylabel="y", title="Toy Regression - Training Dataset")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig_path = OUTPUT_DIR / "toy_training_data.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available (for later training): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU unavailable in PyTorch: later training will use CPU unless CUDA is fixed.")
    print(f"x_train shape: {tuple(x_train.shape)}")
    print(f"y_train shape: {tuple(y_train.shape)}")
    print(f"Data seed: {DATA_SEED}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved PNG: {fig_path}")


if __name__ == "__main__":
    main()
