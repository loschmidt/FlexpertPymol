# Flexpert - PyMOL Plugin

A PyMOL plugin for protein flexibility prediction using machine learning models.

## Overview

Flexpert predicts protein flexibility by combining sequence-based and structure-based neural network models. It integrates with PyMOL to provide visualization capabilities directly within the molecular graphics environment.

## Features

- Protein flexibility prediction
- Sequence-based predictions using ProtT5 transformer model
- Structure-based 3D predictions
- Integration with PyMOL GUI
- Automatic installation and dependency management

## Requirements

- PyMOL (with Python and Qt support)
- Python 3.8+
- CUDA-capable GPU (recommended for faster inference)

## Installation

1. Clone this repository into your PyMOL plugins directory:
   ```bash
   git clone <repository-url> ~/.pymol/extensions/Flexpert
   ```

2. The plugin will automatically prompt for installation when loaded, or you can run:
   ```python
   from Flexpert import installation
   installation.check_and_install()
   ```

3. The installation process will:
   - Install Python dependencies via `requirements.txt`
   - Create cache directories
   - Download model weights (~60 GB total)

## Usage

After installation, Flexpert can be accessed from within PyMOL:

```python
from pymol import cmd
from Flexpert import predictions

# Predict flexibility for a loaded structure
predictions.predict_single_chain("your_object_name")
```

## Model Weights

The plugin requires downloading two model weight files:
- `flexpert_3d_weights.bin` - Structure-based prediction model
- `flexpert_seq_weights.bin` - Sequence-based prediction model

These are downloaded automatically during installation from the project data server.

## Project Structure

```
.
├── __init__.py          # Main plugin entry point
├── installation.py      # Installation and setup logic
├── predictions.py       # Prediction functionality
├── prediction_async.py  # Async prediction utilities
└── Flexpert/            # Plugin subdirectory
    └── cache/           # Cached model weights and data
```

## License

[Add your license here]

## References

- [Flexpert Data Server](https://data.ciirc.cvut.cz/public/projects/2025Flexpert/)
