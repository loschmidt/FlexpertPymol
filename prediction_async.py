from pymol import cmd

from pymol.Qt import QtCore

from .predictions import predict_single_chain


class PredictionWorker(QtCore.QThread):
    def __init__(self, selection, color_palette, parent=None):
        super().__init__(parent)
        self.selection = selection
        self.color_palette = color_palette
        self.success = False
        self.error_msg = ""

    def run(self):
        try:
            flexpert_predict_sync(self.selection, self.color_palette)
            self.success = True
        except Exception as e:
            self.success = False
            self.error_msg = str(e)


def flexpert_predict_sync(selection="all", color_palette="blue_white_red"):
    all_chains = []
    cmd.iterate(f"{selection} and name CA", "all_chains.append(chain)", space={'all_chains': all_chains})
    unique_chains = sorted(set(all_chains))

    if not unique_chains:
        raise Exception("No chains found in selection")

    print(f"Found {len(unique_chains)} chain(s): {', '.join(unique_chains)}")

    for chain_id in unique_chains:
        print(f"\n{'=' * 50}")
        print(f"Processing chain {chain_id}")
        print(f"{'=' * 50}")
        predict_single_chain(chain_id, color_palette)


