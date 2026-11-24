r"""CMP4509 Deep Ensembles | Toy Regression | Asama 3: 5 MSE modelinden ensemble.

Windows PowerShell (toy_regression klasorunde):
    .\.venv\Scripts\python.exe .\03_train_mse_ensemble.py

Girdi:
    outputs/toy_training_data.csv    (01_generate_data.py uretir)

Ciktilar (mevcut outputs/ klasorunde):
    mse_ensemble_5_model_01.pt ... mse_ensemble_5_model_05.pt
    mse_ensemble_5_prediction.png       (Figure 1'in ilk paneline benzer)
    mse_ensemble_5_members.png          (bes agin ayri tahminleri)
    mse_ensemble_5_loss.png             (bes agin MSE kayiplari)
    mse_ensemble_5_predictions.csv      (her x noktasinda ortalama ve belirsizlik)
    mse_ensemble_5_training_log.csv     (epoch bazinda egitim kaybi)
    mse_ensemble_5_summary.csv          (model bazinda egitim RMSE'si)
    mse_single_vs_ensemble_5.png        (02 dosyasinin checkpoint'i varsa)

Makaledeki Figure 1 sol panel: 5 agin MSE ile egitilip tahminlerinin
ampirik varyansinin kullanilmasi. Burada 1-50-ReLU-1, Adam 0.01,
3000 epoch ve [-4, 4] egitim araligi bizim sadeleştirmelerimizdir.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Grafikler ekrani acmadan dosyaya kaydedilir.
import matplotlib.pyplot as plt
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA_FILE = OUT / "toy_training_data.csv"
SINGLE_CHECKPOINT = OUT / "mse_single_model.pt"

NUM_MODELS = 5
MODEL_SEEDS = (123, 124, 125, 126, 127)
HIDDEN_UNITS = 50
EPOCHS = 3000
LEARNING_RATE = 0.01
TRAIN_MIN, TRAIN_MAX = -4.0, 4.0


class SimpleRegressor(nn.Module):
    """Onceki dosyadaki mimariyla ayni: 1 -> 50 (ReLU) -> 1."""

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
            f"Egitim verisi bulunamadi: {path}\n"
            "Once 01_generate_data.py dosyasini calistir."
        )

    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 20 or any("x" not in row or "y" not in row for row in rows):
        raise ValueError("20 satirli, 'x' ve 'y' sutunlu CSV bekleniyor.")

    x = torch.tensor([[float(row["x"])] for row in rows], dtype=torch.float32)
    y = torch.tensor([[float(row["y"])] for row in rows], dtype=torch.float32)
    return x, y


def train_member(
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
) -> tuple[SimpleRegressor, list[float], float]:
    """Bir agi bagimsiz sifirdan egitir (baska aglarla agirlik paylasmaz)."""
    torch.manual_seed(seed)
    # CPU'da olusturup daha sonra GPU'ya aktararak seed'e bagli
    # baslangic agirliklarini aygitlar arasinda tutarli kuruyoruz.
    model = SimpleRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    history: list[float] = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        prediction = model(x_train)
        loss = criterion(prediction, y_train)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % 500 == 0:
            print(f"  Epoch {epoch:4d}/{EPOCHS} | MSE: {history[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        train_rmse = torch.sqrt(criterion(model(x_train), y_train)).item()
    return model, history, train_rmse


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def add_context(ax: plt.Axes, x_data, y_data, x_plot, y_true) -> None:
    ax.plot(x_plot, y_true, color="royalblue", linewidth=2.2,
            label="Ground truth: y = x^3")
    ax.scatter(x_data, y_data, color="crimson", s=35, zorder=5,
               label="20 noisy training samples")
    ax.axvline(TRAIN_MIN, color="gray", ls="--", alpha=0.55)
    ax.axvline(TRAIN_MAX, color="gray", ls="--", alpha=0.55)
    ax.set(xlabel="x", ylabel="y", xlim=(-6, 6))
    ax.grid(alpha=0.2)


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)
    x_cpu, y_cpu = read_training_data(DATA_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = x_cpu.to(device)
    y_train = y_cpu.to(device)
    x_plot = torch.linspace(-6.0, 6.0, 400, device=device).unsqueeze(1)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {len(x_cpu)} samples (same CSV as Steps 1-2)")
    print(f"Network: 1 -> {HIDDEN_UNITS} (ReLU) -> 1")
    print(f"Ensemble: {NUM_MODELS} independent models, seeds={MODEL_SEEDS}")
    print(f"Epochs: {EPOCHS}; LR: {LEARNING_RATE}")

    trained_models: list[SimpleRegressor] = []
    histories: list[list[float]] = []
    train_rmses: list[float] = []
    predictions: list[torch.Tensor] = []

    # Butun aglar ayni egitim verisini goruyor; yalnizca baslangic
    # agirliklari icin kullanilan random seed'ler farkli.
    for member_idx, seed in enumerate(MODEL_SEEDS, start=1):
        print(f"\n=== Model {member_idx}/{NUM_MODELS} | seed={seed} ===")
        model, history, train_rmse = train_member(seed, x_train, y_train, device)
        with torch.no_grad():
            predictions.append(model(x_plot).squeeze(-1).detach().cpu())

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_seed": seed,
                "model_number": member_idx,
                "hidden_units": HIDDEN_UNITS,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "training_rmse": train_rmse,
            },
            OUT / f"mse_ensemble_5_model_{member_idx:02d}.pt",
        )
        trained_models.append(model)
        histories.append(history)
        train_rmses.append(train_rmse)
        print(f"  Final training RMSE: {train_rmse:.4f}")

    # Tahminler: [5 model, 400 x noktasi]
    all_predictions = torch.stack(predictions, dim=0)
    mean_prediction = all_predictions.mean(dim=0)

    # N'e bolen ampirik varyans (population variance), 5 modelin esit
    # agirlikli karisimina uygun. Tek basina veri gurultusunu olcmez.
    variance = all_predictions.var(dim=0, correction=0)
    std_prediction = torch.sqrt(variance)
    lower = mean_prediction - 3.0 * std_prediction
    upper = mean_prediction + 3.0 * std_prediction

    x_np = x_plot.cpu().squeeze(-1).numpy()
    y_true = x_np**3
    train_x_np = x_cpu.squeeze(-1).numpy()
    train_y_np = y_cpu.squeeze(-1).numpy()
    mean_np = mean_prediction.numpy()
    std_np = std_prediction.numpy()

    # Grafik 1: Makaledeki Figure 1'in ilk paneline benzer.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    add_context(ax, train_x_np, train_y_np, x_np, y_true)
    ax.fill_between(x_np, lower.numpy(), upper.numpy(), color="gray", alpha=0.22,
                    label="Ensemble mean +/- 3 std")
    ax.plot(x_np, lower.numpy(), color="gray", lw=1.2, alpha=0.8)
    ax.plot(x_np, upper.numpy(), color="gray", lw=1.2, alpha=0.8)
    ax.plot(x_np, mean_np, color="darkorange", lw=2.2, label="MSE ensemble mean (5 models)")
    ax.set_title("Toy Regression | MSE Ensemble (5 networks) - Figure 1 baseline")
    ax.legend(loc="upper left", fontsize=9)
    save_figure(fig, "mse_ensemble_5_prediction.png")

    # Grafik 2: Aglarin farkli tahminlerini ayri ciziyoruz.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    add_context(ax, train_x_np, train_y_np, x_np, y_true)
    for member_idx, pred in enumerate(predictions, start=1):
        ax.plot(x_np, pred.numpy(), lw=1.5, alpha=0.75,
                label=f"Model {member_idx} (seed {MODEL_SEEDS[member_idx-1]})")
    ax.plot(x_np, mean_np, color="black", lw=2.4, label="Ensemble mean")
    ax.set_title("Toy Regression | Predictions of 5 independent MSE networks")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    save_figure(fig, "mse_ensemble_5_members.png")

    # Grafik 3: Egitim sirasinda kayiplar nasil degisti?
    fig, ax = plt.subplots(figsize=(9, 4.8))
    epochs = list(range(1, EPOCHS + 1))
    for member_idx, history in enumerate(histories, start=1):
        ax.plot(epochs, history, alpha=0.8, lw=1.2,
                label=f"Model {member_idx}")
    ax.set(xlabel="Epoch", ylabel="Training MSE",
           title="Training loss - 5 independent MSE networks")
    ax.grid(alpha=0.2)
    ax.legend()
    save_figure(fig, "mse_ensemble_5_loss.png")

    # Egitim kayiplarini csv olarak kaydet.
    with (OUT / "mse_ensemble_5_training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + [f"model_{m}_mse" for m in range(1, NUM_MODELS + 1)])
        for index, epoch in enumerate(epochs):
            writer.writerow([epoch] + [history[index] for history in histories])

    # Tahminleri ve standart sapmayi sayisal olarak kaydet.
    with (OUT / "mse_ensemble_5_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "ground_truth", "mean_prediction", "empirical_std",
                         "lower_3std", "upper_3std"] +
                        [f"model_{m}_prediction" for m in range(1, NUM_MODELS + 1)])
        for idx, x_val in enumerate(x_np):
            writer.writerow([float(x_val), float(y_true[idx]), float(mean_np[idx]),
                             float(std_np[idx]), float(lower[idx]), float(upper[idx])] +
                            [float(pred[idx]) for pred in predictions])

    # Karsilastirma icin ayri bir summary csv.
    with (OUT / "mse_ensemble_5_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "seed", "training_rmse"])
        for idx, (seed, rmse) in enumerate(zip(MODEL_SEEDS, train_rmses), start=1):
            writer.writerow([f"model_{idx}", seed, rmse])
        with torch.no_grad():
            train_member_predictions = torch.stack(
                [m(x_train).detach().cpu().squeeze(-1) for m in trained_models]
            )
        train_ensemble_rmse = torch.mean(
            (train_member_predictions.mean(dim=0) - y_cpu.squeeze(-1)) ** 2
        ).sqrt().item()
        writer.writerow(["ensemble_mean", "all", train_ensemble_rmse])

    # Onceki tek MSE modeli mevcutsa ayni eksenlerde karsilastirma uret.
    if SINGLE_CHECKPOINT.exists():
        single_model = SimpleRegressor()  # onceki script ile ayni mimari
        state = torch.load(SINGLE_CHECKPOINT, map_location="cpu", weights_only=True)
        single_model.load_state_dict(state["model_state_dict"])
        single_model.eval()
        with torch.no_grad():
            single_pred = single_model(x_plot.cpu()).squeeze(-1).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
        for ax in axes:
            add_context(ax, train_x_np, train_y_np, x_np, y_true)
        axes[0].plot(x_np, single_pred, color="darkorange", lw=2.1, label="Single MSE model")
        axes[0].set_title("Step 2: Single MSE network")
        axes[1].fill_between(x_np, lower.numpy(), upper.numpy(), color="gray", alpha=0.25,
                             label="Ensemble mean +/- 3 std")
        axes[1].plot(x_np, mean_np, color="darkorange", lw=2.1, label="MSE ensemble mean")
        axes[1].set_title("Step 3: MSE Ensemble (5 networks)")
        for ax in axes:
            ax.legend(loc="upper left", fontsize=8)
        save_figure(fig, "mse_single_vs_ensemble_5.png")
    else:
        print("\nNot: mse_single_model.pt bulunamadi; tek-vs-ensemble grafigi atlandi.")
        print("Dilersen once 02_train_mse.py dosyasini calistirabilirsin.")

    print("\n=== SONUC ===")
    print(f"Ensemble training RMSE (same 20 points): {train_ensemble_rmse:.4f}")
    print("x    ground truth     ensemble mean    std     abs error")
    x_check = torch.tensor([[-6.0], [-4.0], [0.0], [4.0], [6.0]], dtype=torch.float32)
    with torch.no_grad():
        check_preds = torch.stack(
            [m(x_check.to(device)).detach().cpu().squeeze(-1) for m in trained_models]
        )
    check_mean = check_preds.mean(dim=0)
    check_std = check_preds.std(dim=0, correction=0)
    for j, xval in enumerate(x_check.squeeze(-1).tolist()):
        truth = xval**3
        avg = check_mean[j].item()
        std = check_std[j].item()
        print(f"{xval:+.0f}   {truth:+9.2f}      {avg:+10.2f}    {std:6.2f}    {abs(truth-avg):9.2f}")
    print("\nDosyalar outputs/ klasorune kaydedildi.")
    print("Once mse_ensemble_5_members.png ve mse_ensemble_5_prediction.png grafigine bak.")
    print("Not: std model anlasmazligini olcer; bu, veri gurultusu veya garanti hata siniri degildir.")


if __name__ == "__main__":
    main()
