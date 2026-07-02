# Flexpert – PyMOL Plugin

A PyMOL plugin for protein flexibility prediction using the Flexpert predictor.

## Overview

Flexpert predicts protein flexibility by combining sequence-based and structure-based neural network models. It integrates directly into PyMOL, enabling visualization of flexibility predictions within the molecular graphics environment.

The most recent implementation suitable for standalone installation of Flexpert can be found here: https://github.com/KoubaPetr/Flexpert

## Features

- Protein flexibility prediction
- Integration with the PyMOL GUI
- Automated model installation

## Requirements

- PyMOL (with Python and Qt support)
- Python 3.8+
- CUDA-capable GPU (optional)
- Plugin was tested mainly on systems running Linux-based OS. Windows remains untested.

A GPU is not required, but it can significantly speed up inference. In testing, CPU-based inference took approximately one minute per protein chain.

## Installation

### Option 1: Clone via Git

Clone this repository into your PyMOL plugins directory:

```bash
git clone <repository-url> ~/.pymol/startup/Flexpert
```

### Option 2: Install from Plugin Manager

Download the `.tar.gz` archive from **(TODO: link)** and install it via:

`Plugin → Plugin Manager → Install New Plugin → Choose File...`

### Post-Installation Setup

The plugin will automatically prompt for installation when loaded. If this doesn't happen, trigger it manually.

Installation is a **required** step — the plugin will not function without it.

You can trigger installation in either of these ways:

- Run the command `flexpert_check_install` in the PyMOL console
- Select `Plugin → Install/Update Flexpert` from the menu

This process downloads approximately **40 GB** of data, which may take a while depending on your internet connection. Please keep PyMOL open until installation finishes — you'll see `Installation completed successfully!` printed in the PyMOL console.

During installation, the plugin will:

- Install Python dependencies via `requirements.txt`
- Create cache directories
- Download model weights (~40 GB total)

## Usage

Once installed, Flexpert can be used in two ways:

**Console command:**

```
flexpert_predict <selection>,[colour_scheme]
```

Defaults: `selection = all`, `colour_scheme = blue_white_red`. Running `flexpert_predict` alone uses both default values.

**GUI:**

`Plugin → Flexpert Prediction`

Please note, GUI does not currently offer the option to specify colour palette.
## Model Weights

Flexpert requires two model weight files, downloaded automatically during installation from the project data server:

- `flexpert_3d_weights.bin` — structure-based prediction model
- `flexpert_seq_weights.bin` — sequence-based prediction model



## Model Weights

Flexpert requires two model weight files, downloaded automatically during installation from the project data server:

- `flexpert_3d_weights.bin` — structure-based prediction model
- `flexpert_seq_weights.bin` — sequence-based prediction model

In addition, the HuggingFace `Rostlab/prot_t5_xl_uniref50` model (tokenizer + encoder) is downloaded and cached for sequence-based inference.

## Requirements File

Python dependencies are installed from `requirements.txt`, which includes the following top-level packages:

`Bio, biotite, datasets, matplotlib, numpy, pandas, peft, pyyaml, requests, scikit-learn, scipy, torch, tqdm, transformers==4.46.3, sentencepiece, prody, evaluate, Pillow==9.1.0`

**Note on `pip install`:** Dependencies are installed using the same Python interpreter that runs PyMOL (`sys.executable -m pip install -r requirements.txt`), with no `--user` flag. Depending on your PyMOL/Python setup, this may attempt a **system-wide install**, which can:

- Fail due to insufficient permissions (requiring `sudo`, which is not recommended)
- Conflict with system-managed Python packages
- Affect other Python applications sharing the same interpreter

If you hit permission errors or want an isolated environment, consider running PyMOL inside a Python virtual environment or [Conda environment](https://pymol.org/conda/) where `pip install` targets a user-writable location, rather than installing directly into the system Python.

## Troubleshooting

**Installation fails partway through (weights or HuggingFace models):**

The installer downloads files in stages (requirements → cache dirs → Flexpert weights → HuggingFace models). If it fails or is interrupted partway, cached/partial files may be left in an inconsistent state.

To do a clean retry:

1. Close PyMOL.
2. Delete the entire Flexpert cache folder. (by default in `~/.pymol/startup/Flexpert/Flexpert/cache`) It contains `transformers`, `hub`, `datasets`, and the downloaded weight files.
3. Reopen PyMOL and re-trigger installation with `flexpert_check_install` or `Plugin → Install/Update Flexpert`.

Removing the whole cache folder forces all weights and models to re-download from scratch, avoiding issues from partially written or corrupted files.

**`pip install` fails or tries to install system-wide:**

- Check the pip error output printed to the console (return code, STDOUT/STDERR) for the exact failure reason.
- If it's a permissions error, avoid using `sudo pip install`; instead run PyMOL from a virtual environment or Conda environment so packages install into a user-writable location.
- If a specific package fails (e.g., `torch`, `transformers==4.46.3`), try installing that package manually first to see the underlying error, then re-run `flexpert_check_install`.

**Missing components after installation:**

Run `check_installation()` logic manually (or re-trigger install) — it reports exactly which pieces are missing: the predict script, cache directory, either weight file, the HuggingFace model cache, or specific Python packages. This can help pinpoint whether the issue is disk-related, network-related, or a dependency problem.

**HuggingFace model download is slow or stalls:**

The ProtT5 download is large and runs as a separate subprocess. If it stalls, cancel PyMOL, delete the cache folder as above, and retry with a stable connection — partial downloads are not resumed automatically.


## Project Structure

```
.
├── __init__.py          # Main plugin entry point
├── installation.py      # Installation and setup logic
├── predictions.py       # Prediction functionality
├── prediction_async.py  # Async prediction utilities
└── Flexpert/            
    ├── cache/           # Cached model weights and data
    └── Flexpert/        # Flexpert predictor package.
    
```

## License

*(TODO: License)*

## References

- [Flexpert Data Server](https://data.ciirc.cvut.cz/public/projects/2025Flexpert/)

***

Would you like me to also draft a short "Troubleshooting" or "FAQ" section — e.g., for cases where the 40 GB download stalls or GPU detection fails?