"""Execute the actual notebook top-to-bottom in smoke mode in a local kernel."""
import os
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def main():
    import sys
    test_root = ROOT / "notebook_workspace"
    test_root.mkdir(exist_ok=True)
    # Reuse checksum-verified MNIST without any changes to training source/config.
    import shutil
    (test_root / "data").mkdir(exist_ok=True)
    for path in (ROOT / "data").glob("*.gz"):
        shutil.copyfile(path, test_root / "data" / path.name)
    runtime = ROOT / ".cache" / "jupyter"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.update(MNIST_NOTEBOOK_PROFILE="smoke", MNIST_NOTEBOOK_DRIVE="0",
                      MNIST_NOTEBOOK_ROOT=str(test_root), JUPYTER_RUNTIME_DIR=str(runtime),
                      PYTHONUTF8="1")
    notebook = nbformat.read(ROOT / "MNIST_Deep_Ensembles_Colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager()
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=600, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    target = ROOT / "outputs" / "notebook_smoke_executed.ipynb"
    nbformat.write(notebook, target)
    count = sum(cell.cell_type == "code" for cell in notebook.cells)
    assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code")
    print(f"PASS: all {count} code cells executed top-to-bottom: {target}")


if __name__ == "__main__":
    main()
