from pymol import cmd
from pymol.plugins import addmenuitemqt
from pymol.Qt import QtWidgets
import os

from .prediction_async import PredictionWorker
from .installation import check_installation, check_and_install
from .predictions import PLUGIN_DIR, flexpert_script, cache_path, weights_path


def __init_plugin__(app=None):

    is_installed, missing = check_installation()

    if not is_installed:
        print(f"⚠ Flexpert plugin not fully installed. Missing: {', '.join(missing)}")
        print("Run 'flexpert_install' to complete installation")

    # Register menu items
    addmenuitemqt('Flexpert Prediction', flexpert_predict_gui)
    addmenuitemqt('Install/Update Flexpert', check_and_install)

    cmd.extend("flexpert_predict", flexpert_predict)
    cmd.extend("flexpert_info", get_plugin_info)
    cmd.extend("flexpert_check_install", check_and_install)

def flexpert_predict_gui():
    selection, ok = QtWidgets.QInputDialog.getText(
        None,
        'Prediction Tool',
        'Enter part of protein to predict (standard pymol selection syntax can be used, e.g. chain A):'
    )
    if not ok or not selection:
        return

    progress_dialog = QtWidgets.QProgressDialog(
        f"Running prediction for {selection}...",
        None,
        0, 0
    )
    progress_dialog.setWindowTitle("Flexpert Prediction")
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setWindowModality(2)  # Qt.WindowModal
    progress_dialog.show()

    # Create and start worker
    worker = PredictionWorker(selection, "blue_white_red")

    def on_finished():
        progress_dialog.close()
        if worker.success:
            QtWidgets.QMessageBox.information(None, 'Success', 'Prediction coloring applied successfully!')
        else:
            QtWidgets.QMessageBox.critical(None, 'Error', f'Failed to apply prediction: {worker.error_msg}')

    worker.finished.connect(on_finished)
    worker.start()


def flexpert_predict(selection="all", color_palette="blue_white_red"):
    """
    USAGE

    flexpert_predict [selection [, color_palette]]

    ARGUMENTS

    selection = string: part of protein to predict all/chain B
    color_palette = string: color scheme
                   Options: blue_white_red, red_white_blue, rainbow, etc.

    EXAMPLES

    flexpert_predict
    flexpert_predict chain A, rainbow
    flexpert_predict all, blue_white_red

    NOTE: Runs in background thread. Use flexpert_predict_sync for blocking version.
    """
    print(f"Starting prediction in background for: {selection}", flush=True)
    worker = PredictionWorker(selection, color_palette)
    worker.start()
    # Store reference so it doesn't get garbage collected
    if not hasattr(cmd, '_flexpert_workers'):
        cmd._flexpert_workers = []
    cmd._flexpert_workers.append(worker)


def get_plugin_info():
    print(f"=== Flexpert Plugin Info ===")
    print(f"Plugin directory: {PLUGIN_DIR}")
    print(f"Flexpert script: {flexpert_script}")
    print(f"Flexpert exists: {os.path.exists(flexpert_script)}")
    print(f"Cache directory: {cache_path}")
    print(f"Cache exists: {os.path.exists(cache_path)}")

    if os.path.exists(cache_path):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(cache_path)
            for filename in filenames
        )
        print(f"Cache size: {cache_size / (1024 ** 2):.2f} MB")

    is_installed, missing = check_installation()
    print(f"\nInstallation status: {'✓ Complete' if is_installed else '✗ Incomplete'}")
    if not is_installed:
        print(f"Missing: {', '.join(missing)}")

    # Cache info
    if os.path.exists(cache_path):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(cache_path)
            for filename in filenames
        )
        print(f"\nCache directory: {cache_path}")
        print(f"Cache size: {cache_size / (1024 ** 2):.2f} MB")

    # Weights info
    if os.path.exists(weights_path):
        weights = ['flexpert_3d_weights.bin', 'flexpert_seq_weights.bin']
        print(f"\nModel weights:")
        for weight in weights:
            weight_path = os.path.join(weights_path, weight)
            if os.path.exists(weight_path):
                size = os.path.getsize(weight_path) / (1024 ** 2)
                print(f"  ✓ {weight} ({size:.1f} MB)")
            else:
                print(f"  ✗ {weight} (missing)")

