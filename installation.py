"""
Flexpert plugin installation and setup
"""

import os
import subprocess
import sys
import re
import urllib.request
import importlib.util
from pymol import cmd

from pymol.Qt import QtWidgets
from pymol.Qt import QtCore

from .predictions import weights_path, cache_path, flexpert_script, flexpert_path, predict_single_chain


# List of required Python modules/packages for Flexpert
# (fill this with the actual top-level module names from your requirements)
REQUIRED_PACKAGES = [
    "Bio",
    "biotite",
    "datasets",
    "matplotlib",
    "numpy",
    "pandas",
    "peft",
    "yaml",
    "requests",
    "sklearn",
    "scipy",
    "torch",
    "tqdm",
    "transformers==4.46.3",
    "sentencepiece",
    "prody",
    "evaluate",
    "Pillow==9.1.0"
]

IMPORT_MAP = {
    "Pillow": "PIaL"
}


class FlexpertInstallWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal()
    finished_error = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.success = False
        self.error_msg = ""

    def run(self):
        try:
            ok = install_flexpert()
            self.success = bool(ok)
            if self.success:
                self.finished_ok.emit()
            else:
                self.error_msg = "Flexpert installation did not complete successfully."
                self.finished_error.emit(self.error_msg)
        except Exception as e:
            self.success = False
            self.error_msg = str(e)
            self.finished_error.emit(self.error_msg)


def _check_python_package_installed(pkg_spec: str) -> bool:
    """Return True if a Python package/module is importable."""
    name = re.split(r'[><=!]', pkg_spec)[0].strip()
    import_name = IMPORT_MAP.get(name, name)
    return importlib.util.find_spec(import_name) is not None


def check_installation():
    missing = []

    # Check script and cache/model files
    if not os.path.exists(flexpert_script):
        missing.append('Flexpert predict.py')

    if not os.path.exists(cache_path):
        missing.append('cache directory')

    weight_3d = os.path.join(weights_path, 'flexpert_3d_weights.bin')
    weight_seq = os.path.join(weights_path, 'flexpert_seq_weights.bin')

    if not os.path.exists(weight_3d):
        missing.append('flexpert_3d_weights.bin')
    if not os.path.exists(weight_seq):
        missing.append('flexpert_seq_weights.bin')

    # Check HuggingFace model cache (hub dir populated = models downloaded)
    hub_path = os.path.join(cache_path, 'transformers', 'models--Rostlab--prot_t5_xl_uniref50')
    if not os.path.exists(hub_path) or not os.listdir(hub_path):
        missing.append('HuggingFace models (Rostlab/prot_t5_xl_half_uniref50)')

    # Check required Python packages
    missing_pkgs = []
    for pkg in REQUIRED_PACKAGES:
        if not _check_python_package_installed(pkg):
            missing_pkgs.append(pkg)

    if missing_pkgs:
        missing.append("Python packages: " + ", ".join(missing_pkgs))

    return len(missing) == 0, missing


def download_hf_models():
    """
    Pre-flight download of HuggingFace models (ProtT5) without running inference.
    This triggers the 60GB model download and caches it locally.
    """
    print("\n[4/4] Pre-downloading HuggingFace models (~60 GB)...")
    print("This may take a long time depending on your connection.")

    hf_home = cache_path
    env = os.environ.copy()
    env['HF_HOME'] = hf_home
    env['TRANSFORMERS_CACHE'] = os.path.join(hf_home, 'transformers')
    env['HF_HUB_CACHE'] = os.path.join(hf_home, 'hub')
    env['HF_DATASETS_CACHE'] = os.path.join(hf_home, 'datasets')
    env['XDG_CACHE_HOME'] = hf_home
    env['MODEL_HOME'] = os.path.join(hf_home, 'models')

    download_script = (
        "from transformers import T5EncoderModel, T5Tokenizer; "
        "print('Downloading ProtT5 tokenizer...'); "
        "T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50'); "
        "print('Downloading ProtT5 model...'); "
        "T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50'); "
        "print('HuggingFace models downloaded successfully.')"
    )
    print("Downloading ProtT5 model and tokenizer to " + hf_home + " ...")
    result = subprocess.run(
        [sys.executable, '-c', download_script],
        capture_output=False,  # stream output directly to console
        text=True,
        env=env,
        cwd=flexpert_path
    )

    if result.returncode != 0:
        print(f"HuggingFace model download failed (exit code {result.returncode})")
        raise Exception(f"HuggingFace model download failed (exit code {result.returncode})")

    print("HuggingFace models cached successfully")


def install_flexpert():
    print("=" * 60)
    print("Flexpert Plugin Installation")
    print("=" * 60)

    try:
        install_requirements(flexpert_path)
        create_cache_directory()
        download_model_weights(flexpert_path)
        download_hf_models()
        
        print("\n" + "=" * 60)
        print("Installation completed successfully!")
        # print("When running Flexpert for a first time, additional models (60 GB) will be downloaded which may require a lot of time.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nInstallation failed: {str(e)}")
        return False


def install_requirements(flexpert_dir):
    print("\n[1/4] Installing Python requirements...")

    requirements_file = os.path.join(flexpert_dir, 'requirements.txt')

    if not os.path.exists(requirements_file):
        raise Exception(f"requirements.txt not found at {requirements_file}")

    print(f"Installing from: {requirements_file}")

    # Always run pip, and fail if it returns non-zero
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-r', requirements_file],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        # Show stdout and stderr to help debugging
        msg = (
            "pip failed to install requirements.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        raise Exception(msg)
    else:
        print("Requirements installed successfully")


def create_cache_directory():
    print("\n[2/4] Creating cache directories...")

    cache_subdirs = [
        cache_path,
        os.path.join(cache_path, 'transformers'),
        os.path.join(cache_path, 'hub'),
        os.path.join(cache_path, 'datasets')
    ]

    for subdir in cache_subdirs:
        os.makedirs(subdir, exist_ok=True)
        print(f"Created: {subdir}")

    print("Cache directories created successfully")


def download_model_weights(flexpert_dir):
    print("\n[3/4] Downloading Flexpert model weights...")

    os.makedirs(weights_path, exist_ok=True)

    weights = {
        'flexpert_3d_weights.bin': 'https://data.ciirc.cvut.cz/public/projects/2025Flexpert/flexpert-weights/flexpert_3d_weights.bin',
        'flexpert_seq_weights.bin': 'https://data.ciirc.cvut.cz/public/projects/2025Flexpert/flexpert-weights/flexpert_seq_weights.bin'
    }

    for filename, url in weights.items():
        output_path = os.path.join(weights_path, filename)

        if os.path.exists(output_path):
            print(f"Already exists: {filename}")
            continue

        print(f"Downloading {filename}...")
        print(f"URL: {url}")

        try:
            download_file_with_progress(url, output_path)
            print(f"✓ Downloaded: {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {str(e)}")
            raise Exception(f"Failed to download {filename}: {str(e)}")

    print("Model weights downloaded successfully")


def download_file_with_progress(url, output_path):
    last_percent = {'value': -1}

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            if percent > last_percent['value']:
                last_percent['value'] = percent
                print(f"\rProgress: {percent}%", end="", flush=True)

    urllib.request.urlretrieve(url, output_path, reporthook)
    print()


def prompt_installation_gui():

    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Question)
    msg.setWindowTitle("Flexpert Setup Required")
    msg.setText("Flexpert plugin requires installation of dependencies and model weights.")
    msg.setInformativeText("This will:\n"
                           "1. Install Python requirements\n"
                           "2. Create cache directories\n"
                           "3. Download model weights\n\n"
                           "Would you like to install now?")
    msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

    return msg.exec_() == QtWidgets.QMessageBox.Yes


def check_and_install():
    is_installed, missing = check_installation()

    if is_installed:
        print("Flexpert is properly installed")
        return True

    print(f"Missing components: {', '.join(missing)}")

    if prompt_installation_gui():
        worker = FlexpertInstallWorker()
        worker.start()
        # Store reference so it doesn't get garbage collected
        if not hasattr(cmd, '_flexpert_workers'):
            cmd._flexpert_workers = []
        cmd._flexpert_workers.append(worker)
    else:
        print("Installation cancelled by user")
        return False
