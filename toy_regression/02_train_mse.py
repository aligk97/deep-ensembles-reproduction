r"""CMP4509 Deep Ensembles | Toy Regression - Step 2: train ONE MSE network.

Run in PowerShell, from toy_regression folder:
    .\.venv\Scripts\python.exe .\02_train_mse.py

Reads:   outputs/toy_training_data.csv (created by 01_generate_data.py)
Creates: outputs/mse_single_model.pt
         outputs/mse_single_prediction.png
         outputs/mse_single_loss.png
         outputs/mse_single_training_log.csv

IMPORTANT: This is an introductory single-network baseline. Figure 1's first
panel is an ENSEMBLE of 5 MSE networks, not this single-network exercise.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")  # Save figures to files, also works without a display.
import matplotlib.pyplot as plt
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA_FILE = OUT / "toy_training_data.csv"

MODEL_SEED = 123
EPOCHS = 3000
LEARNING_RATE = 0.01
HIDDEN_UNITS = 50


class SimpleRegressor(nn.Module):
    """One input, one hidden ReLU layer, and one numeric output."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def read_training_data(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            "First run: .\\.venv\\Scripts\\python.exe .\\01_generate_data.py"
        )

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows or any("x" not in row or "y" not in row for row in rows):
        raise ValueError("CSV file should contain columns named 'x' and 'y'.")
    x = torch.tensor([[float(row["x"])] for row in rows], dtype=torch.float32)
    y = torch.tensor([[float(row["y"])] for row in rows], dtype=torch.float32)
    return x, y


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Same 20 training examples from Step 1; no new dataset is generated.
    x_cpu, y_cpu = read_training_data(DATA_FILE)
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train = x_cpu.to(device), y_cpu.to(device)

    model = SimpleRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training examples: {len(x_train)}")
    print(f"Network: 1 -> {HIDDEN_UNITS} (ReLU) -> 1")
    print(f"Epochs: {EPOCHS}; learning rate: {LEARNING_RATE}; seed: {MODEL_SEED}")
    print("Training ...")

    history: list[tuple[int, float]] = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        prediction = model(x_train)
        loss = criterion(prediction, y_train)
        loss.backward()   # Compute gradients of the error with respect to weights.
        optimizer.step()  # Update weights in the direction that reduces error.

        loss_value = float(loss.detach().cpu())
        history.append((epoch, loss_value))
        if epoch == 1 or epoch % 500 == 0:
            print(f"Epoch {epoch:4d}/{EPOCHS} | MSE: {loss_value:.4f}")

    model.eval()
    with torch.no_grad():
        predicted_train = model(x_train)
        train_rmse = torch.sqrt(criterion(predicted_train, y_train)).item()

        x_plot = torch.linspace(-6, 6, 400, device=device).unsqueeze(1)
        y_pred_plot = model(x_plot).cpu().flatten().numpy()
        y_true_plot = x_plot.cpu().flatten().numpy() ** 3
        x_plot_np = x_plot.cpu().flatten().numpy()

    checkpoint = OUT / "mse_single_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_seed": MODEL_SEED,
            "hidden_units": HIDDEN_UNITS,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "train_rmse": train_rmse,
        },
        checkpoint,
    )

    log_path = OUT / "mse_single_training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "training_mse"])
        writer.writerows(history)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_plot_np, y_true_plot, color="royalblue", linewidth=2, label="Ground truth: y = x^3")
    ax.plot(x_plot_np, y_pred_plot, color="darkorange", linewidth=2, label="Single MSE network prediction")
    ax.scatter(x_cpu.flatten().numpy(), y_cpu.flatten().numpy(), color="crimson", zorder=3,
               label="20 noisy training samples")
    ax.axvline(-4, color="gray", linestyle="--", alpha=0.6)
    ax.axvline(4, color="gray", linestyle="--", alpha=0.6)
    ax.set(xlabel="x", ylabel="y", title="Toy Regression | One Network Trained with MSE")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    prediction_path = OUT / "mse_single_prediction.png"
    fig.savefig(prediction_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([epoch for epoch, _ in history], [value for _, value in history], color="darkorange")
    ax.set(xlabel="Epoch", ylabel="Training MSE", title="Training loss - single MSE network")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    loss_path = OUT / "mse_single_loss.png"
    fig.savefig(loss_path, dpi=180)
    plt.close(fig)

    print(f"Final training RMSE: {train_rmse:.4f} (on the same 20 training points)")
    print("Files saved:")
    for path in [checkpoint, prediction_path, loss_path, log_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
