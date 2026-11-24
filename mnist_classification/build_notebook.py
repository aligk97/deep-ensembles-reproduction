"""Generate a self-contained Colab notebook from the actual project sources."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NAMES = ["data.py", "model.py", "metrics.py", "engine.py", "run_experiment.py", "plots.py",
         "test_correctness.py", "verify_artifacts.py", "requirements.txt", "README.md", "DIFFERENCES.md",
         "configs/smoke.json", "configs/local.json", "configs/full.json"]


def build():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in NAMES:
            # Fixed zip timestamps make notebook generation reproducible.
            info = zipfile.ZipInfo(name, (2026, 11, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (ROOT / name).read_bytes())
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode()
    sha = hashlib.sha256(payload).hexdigest()
    notebook = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell("""# CMP4509: MNIST Deep Ensembles - Figure 2(a)

Lakshminarayanan et al. (2017), [arXiv:1612.01474](https://arxiv.org/abs/1612.01474).
Bu notebook gerçek PyTorch deneylerini çalıştırır. Kodun tamamı aşağıdaki kaynak paketinde gömülüdür;
repo URL'si veya ek dosya yükleme gerekmez. Colab menüsünden **Runtime > Change runtime type > T4 GPU** seçin.

Sıra: ortam/kod -> matematiksel kontroller -> 3-seed smoke -> ana deney -> CSV/grafikler -> denetim.
Ana profil varsayılan `local` (M=1–5, 48 model, 10 epoch); `full` tüm M=1,2,3,4,5,10,15 (138 model).
Her iki profil tam MNIST train/test verisini kullanır; 60k training içinden 5k validation ayrılır.
Hiperparametreler test setinde ayarlanmaz; son epoch değerlendirilir. Smoke sayıları rapor sonucu değildir.
Ücretsiz GPU oturum süresi değişir; checkpoint'ler her epoch kaydedilir. Drive kullanılırsa aynı hücreleri
yeniden çalıştırarak eğitim devam eder. Gerçekte tamamlanmamış deneyler için sayı üretilmez.
"""), nbf.v4.new_code_cell("""import os
from pathlib import Path

# Kullanıcı ayarları. Test sonucuna bakarak hiperparametre seçmeyin.
PROFILE = os.environ.get("MNIST_NOTEBOOK_PROFILE", "local")  # "local" veya "full"
USE_DRIVE = os.environ.get("MNIST_NOTEBOOK_DRIVE", "1") == "1"
IN_COLAB = bool(__import__("importlib").util.find_spec("google.colab")) if __import__("importlib").util.find_spec("google") else False
WORK = Path(os.environ.get("MNIST_NOTEBOOK_ROOT", "/content/mnist_classification" if IN_COLAB else "notebook_workspace")).resolve()
WORK.mkdir(parents=True, exist_ok=True)
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/CMP4509_MNIST_outputs")
else:
    OUTPUT_ROOT = WORK / "outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
assert PROFILE in ("local", "full", "smoke")
print("Source:", WORK, "Results:", OUTPUT_ROOT, "Profile:", PROFILE)
"""), nbf.v4.new_code_cell(f'''# Bu paket build_notebook.py tarafından gerçek kaynaklardan oluşturuldu.
import base64, hashlib, io, zipfile
PAYLOAD = "{encoded}"
content = base64.b64decode(PAYLOAD)
assert hashlib.sha256(content).hexdigest() == "{sha}"
with zipfile.ZipFile(io.BytesIO(content)) as archive:
    archive.extractall(WORK)
print("Verified source archive SHA256:", "{sha}")
'''), nbf.v4.new_code_cell("""import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                "--no-input", "--no-cache-dir", "-r", str(WORK / "requirements.txt")], check=True)
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
elif PROFILE != "smoke":
    raise RuntimeError("Ana deney için Colab GPU runtime seçin; CPU için terminalde --device cpu kullanabilirsiniz.")
def run(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True)
run(WORK / "test_correctness.py")
"""), nbf.v4.new_markdown_cell("""## Önce smoke test

Üç seed, 1.000 training örneği, 200 test örneği, 1 epoch, M=1/2.
BatchNorm ve MC Dropout kontrolleri test betiğinde ayrıca yapılır.
"""), nbf.v4.new_code_cell("""run(WORK / "run_experiment.py", "--config", WORK / "configs/smoke.json",
    "--output", OUTPUT_ROOT / "smoke", "--data-dir", WORK / "data")
run(WORK / "verify_artifacts.py", OUTPUT_ROOT / "smoke", "--data-dir", WORK / "data")
"""), nbf.v4.new_markdown_cell("""## Ana deney

Bu hücre tamamlanmadan sonuç tablosu ana deney sonucu sayılmaz. Çalışma kesilirse aynı profille
tekrar çalıştırın. Kaynak veya konfigürasyon değiştirilirse yeni output dizini kullanın.
Clean/R/AT yöntemleri her seed için bağımsız üyeler; MC Dropout her seed için tek model eğitir.
Metrikler probability averaging ile hesaplanır. Accuracy/error oran, error grafikte yüzde, NLL nats,
Brier 1/K normalizasyonludur. Sample SD ddof=1, üç seed üzerindendir.
"""), nbf.v4.new_code_cell("""if PROFILE != "smoke":
    run(WORK / "run_experiment.py", "--config", WORK / f"configs/{PROFILE}.json",
        "--output", OUTPUT_ROOT / PROFILE, "--data-dir", WORK / "data")
RESULT = OUTPUT_ROOT / PROFILE
run(WORK / "verify_artifacts.py", RESULT, "--data-dir", WORK / "data")
"""), nbf.v4.new_code_cell("""import json
import pandas as pd
from IPython.display import display, Image
manifest = json.loads((RESULT / "manifest.json").read_text(encoding="utf-8"))
assert manifest["status"] == "complete"
display(pd.read_csv(RESULT / "summary.csv"))
display(Image(filename=str(RESULT / "figures/figure2a_comparison.png"), width=1200))
display(Image(filename=str(RESULT / "figures/training_curves.png"), width=1200))
display(Image(filename=str(RESULT / "figures/low_confidence.png"), width=900))
print("Training seconds:", manifest["total_training_epoch_seconds"])
print("Only completed results:", RESULT)
"""), nbf.v4.new_code_cell("""# Rapor ve sunum için küçük paylaşım paketi; ağırlık/veri/sanal ortam içermez.
import zipfile
archive_path = OUTPUT_ROOT / f"{PROFILE}_report_artifacts.zip"
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in RESULT.rglob("*"):
        if path.is_file() and path.suffix in (".csv", ".json", ".png", ".pdf") and "source_snapshot" not in path.parts:
            archive.write(path, path.relative_to(RESULT))
print("Shareable report artifacts:", archive_path)
print("Checkpoint klasörü Drive üzerinde ayrıca saklanır; GitHub'a yüklemeyin.")
"""), nbf.v4.new_markdown_cell("""## Makaleden farklar ve yorum

Makale §3.1'deki Adam lr=0.1, batch=100 ve piksel aralığının %1'i epsilon korundu.
10 epoch, validation holdout (55k/5k), BN yerleşimi, [0,1] clipping, PyTorch varsayılan optimizer/BN
ayrıntıları, üç seed ve SD error bar'ları bu uygulamanın tercihleridir. Kaynak paketindeki
`DIFFERENCES.md` tüm ayrıntıları içerir. Test seti hiperparametre seçiminde kullanılmaz.

M büyürken error/NLL/Brier her noktada monoton azalmak zorunda değildir. AT üstünlüğü elde edilmezse
bu başarısızlık saklanmaz. Sonuçları özgün Figure 2(a)'nın nitel eğilimleriyle karşılaştırın;
özgün ham sayıları elde etmeden kesin sayısal eşitlik iddia etmeyin. MC Dropout'ta M örnek sayısıdır;
ensemble'da M ağ sayısıdır, eğitim maliyetleri eşit değildir. Confidence görselleri MNIST içindir,
OOD başarısını kanıtlamaz. Ders raporu Teams'teki zorunlu LaTeX şablonuyla hazırlanmalıdır.
""")]
    notebook.cells = cells
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3"},
                         "colab": {"name": "MNIST_Deep_Ensembles_Colab.ipynb"}, "accelerator": "GPU"}
    nbf.validate(notebook)
    target = ROOT / "MNIST_Deep_Ensembles_Colab.ipynb"
    nbf.write(notebook, target)
    print(f"Created {target}; embedded source SHA256={sha}")


if __name__ == "__main__":
    build()
