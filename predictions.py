from pymol import cmd
import subprocess
import tempfile
import os
import statistics

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
flexpert_path = os.path.join(PLUGIN_DIR, 'Flexpert', 'Flexpert')
flexpert_script = os.path.join(flexpert_path, 'predict.py')
cache_path = os.path.join(PLUGIN_DIR, 'Flexpert', 'cache')
models_path = os.path.join(cache_path, 'models')
weights_path = os.path.join(models_path, 'weights')

def predict_single_chain(chain_id, color_palette):

    # Create temporary PDB file
    tmp_dir = tempfile.gettempdir()
    tmp_pdb_path = os.path.join(tmp_dir, 'protein.pdb')
    selection = f"chain {chain_id}"
    cmd.save(tmp_pdb_path, selection)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_results:
        tmp_result_path = tmp_results.name

    try:
        env = os.environ.copy()
        env['INSTALL_PATH'] = PLUGIN_DIR

        # Set Hugging Face cache directories
        hf_home = cache_path
        env['HF_HOME'] = hf_home
        env['TRANSFORMERS_CACHE'] = os.path.join(hf_home, 'transformers')
        env['HF_HUB_CACHE'] = os.path.join(hf_home, 'hub')
        env['HF_DATASETS_CACHE'] = os.path.join(hf_home, 'datasets')
        env['XDG_CACHE_HOME'] = hf_home
        env['MODEL_HOME'] = os.path.join(hf_home, 'models')

        flexpert_command = [
            'python3', flexpert_script,
            '--input_file', tmp_pdb_path,
            '--modality', '3D',
            '--chain', chain_id,
            '--output_file', tmp_result_path]

        print(f"Plugin directory: {PLUGIN_DIR}")
        print(f"Running Flexpert: {flexpert_command}")
        print(f"HF_HOME: {hf_home}")

        # Run Flexpert prediction
        result = subprocess.run(flexpert_command,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=PLUGIN_DIR
        )

        print("Flexpert output:")
        print(result.stdout)

        if result.returncode != 0:
            print("Flexpert crashed with ret code: " + str(result.returncode))
            print(result.stderr)
            raise Exception(f"Flexpert failed with error:\n{result.stderr}")

        with open(tmp_result_path, 'r') as f:
            output_text = f.read()

        print(f"Read {len(output_text)} bytes from result file")

        predictions = parse_flexpert_output(output_text, selection, chain_id)

        if not predictions:
            raise Exception("No predictions parsed from Flexpert output")

        print(f"Applying {len(predictions)} predictions to chain {chain_id}...")
        for res_id, value in predictions.items():
            chain, res_num = res_id.split('/')
            cmd.alter(
                f"chain {chain} and resi {res_num}",
                f"b={value}"
            )

        values = list(predictions.values())
        predicted_min = min(values)
        predicted_max = max(values)
        quantiles = statistics.quantiles(values, n=10)
        scale_max = quantiles[9]
        scale_min = 0
        cmd.spectrum("b", color_palette, f"chain {chain_id}", minimum=scale_min, maximum=scale_max)

        print(f"\nPrediction statistics:")
        print(f"  Residues: {len(predictions)}")
        print(f"  Predicted Min: {predicted_min:.3f}")
        print(f"  Predicted Max: {predicted_max:.3f}")
        print(f"  Mean: {sum(values) / len(values):.3f}")
        print(f"  Max on colour scale (90th percentile): {scale_max:.3f}")
        print(f"  Min on colour scale: {scale_min:.3f}")

    finally:
        pass
        # if os.path.exists(tmp_pdb_path):
        #     os.remove(tmp_pdb_path)
        # if os.path.exists(tmp_result_path):
        #     os.remove(tmp_result_path)


def parse_flexpert_output(output_text, selection, current_chain):
    predictions = {}
    for line in output_text.strip().split('\n'):
        line = line.strip()

        if line.startswith('>'):
            parts = line[1:].split('_')
            if len(parts) >= 2:
                current_chain = parts[-1]
            continue

        if ',' in line:
            values_str = line.replace(' ', '')
            values = [float(v.strip()) for v in values_str.split(',') if v.strip()]

            if not values:
                continue

            stored_resi = []
            cmd.iterate(
                f"{selection} and name CA",
                "stored_resi.append(resi)",
                space={'stored_resi': stored_resi}
            )

            residue_numbers = sorted(set(stored_resi), key=lambda x: int(x))

            if len(values) != len(residue_numbers):
                print(f"Warning: Chain {current_chain} has {len(residue_numbers)} residues "
                      f"but got {len(values)} predictions")

            for i, value in enumerate(values):
                if i < len(residue_numbers):
                    res_num = residue_numbers[i]
                    predictions[f"{current_chain}/{res_num}"] = value
    return predictions


