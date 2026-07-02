import pickle
import numpy as np
import json
import pandas as pd
from data.scripts.extract_rmsf_labels import extract_rmsf_labels, extract_bfactor_labels, extract_plddt_labels
import yaml
from tqdm import tqdm
import os
def get_flucts_from_pickle(f):
    return pickle.load(f)

def get_flucts_from_jsonl(f):
    _flucts = f.readlines()
    pdb_code_to_fluct_dict = {}
    for line in _flucts:
        json_obj = json.loads(line.strip())
        pdb_code_to_fluct_dict[json_obj['pdb_name']] = np.array(json_obj['fluctuations'])
    return pdb_code_to_fluct_dict

def read_flexpert_predictions(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        pdb_code_to_fluct_dict = {}

        for name_line, fluct_line in zip(lines[::2], lines[1::2]):
            _name = name_line.strip().strip('>')
            if '.' in _name:
                _name = _name.replace('.', '_')
            pdb_code_to_fluct_dict[_name] = np.array(fluct_line.strip().split(','), dtype=np.float32)
    return pdb_code_to_fluct_dict

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate_flexpert', action='store_true', default=False)
    args = parser.parse_args()



    config = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)
    DATA_DIR = config['precomputed_flexibility_profiles_dir']


    if args.evaluate_flexpert:
        flexpert_3d_predictions_path = config['flexpert_3d_predictions_path']
        flexpert_seq_predictions_path = config['flexpert_seq_predictions_path']
        assert os.path.exists(flexpert_3d_predictions_path), f"Flexpert-3D predictions file does not exist: {flexpert_3d_predictions_path}"
        assert os.path.exists(flexpert_seq_predictions_path), f"Flexpert-Seq predictions file does not exist: {flexpert_seq_predictions_path}"
        flexpert_3d_predictions = read_flexpert_predictions(flexpert_3d_predictions_path)
        flexpert_seq_predictions = read_flexpert_predictions(flexpert_seq_predictions_path)

    with open(f'{DATA_DIR}/anm_square_fluctuations.pickle','rb') as f:
        anm_sqFlucts = get_flucts_from_pickle(f)

    with open(f'{DATA_DIR}/gnm_square_fluctuations.pickle','rb') as f:
        gnm_sqFlucts = get_flucts_from_pickle(f)

    with open(f'{DATA_DIR}/atlas_esm_plddt.jsonl','rb') as f:
        esm_plddt = get_flucts_from_jsonl(f)

    atlas_list_path = config['pdb_codes_path']
    atlas_analyses_dir = config['atlas_out_dir']

    atlas_bfactor_path = atlas_analyses_dir + "/{}_analysis/{}_Bfactor.tsv"
    atlas_plddt_path = atlas_analyses_dir + "/{}_analysis/{}_pLDDT.tsv"
    atlas_rmsf_path = atlas_analyses_dir + "/{}_analysis/{}_RMSF.tsv"

    with open(atlas_list_path,'r') as f:
        atlas_list = f.readlines()
        atlas_list = [a.strip() for a in atlas_list]

    fluctuations = {}

    if args.evaluate_flexpert:
        print("Filtering to testset only, to evaluate Flexpert-3D and Flexpert-Seq predictions")
        atlas_list = [a for a in atlas_list if a in flexpert_seq_predictions.keys()]

    for key in tqdm(atlas_list):
        fluctuations[key] = pd.DataFrame()
        fluctuations[key]['prody_ANM'] = np.sqrt(anm_sqFlucts.get(key, np.nan))
        fluctuations[key]['prody_GNM'] = np.sqrt(gnm_sqFlucts.get(key, np.nan))
        fluctuations[key]['esm_plddt'] = 1 - esm_plddt.get(key, np.nan)
        fluctuations[key]['rmsf'] = extract_rmsf_labels(atlas_rmsf_path.format(key, key))[1]
        fluctuations[key]['bfactor'] = extract_bfactor_labels(atlas_bfactor_path.format(key, key))[1]
        fluctuations[key]['af2_plddt'] = 1 - extract_plddt_labels(atlas_plddt_path.format(key, key))[1]
        if args.evaluate_flexpert and key in flexpert_seq_predictions.keys():
            fluctuations[key]['flexpert_3d'] = flexpert_3d_predictions.get(key)
            fluctuations[key]['flexpert_seq'] = flexpert_seq_predictions.get(key)

    pearson_correlations = []

    for pdb_code,df in fluctuations.items():
        cols = ['rmsf', 'bfactor', 'af2_plddt', 'esm_plddt', 'prody_GNM', 'prody_ANM']
        if args.evaluate_flexpert:
            cols.append('flexpert_3d')
            cols.append('flexpert_seq')

        pc = df[cols].corr(method='pearson')
        if  np.any(np.isnan(pc)):
            print(f'{pdb_code} has NaN values in Pearson correlation')
            continue
        pearson_correlations.append(pc)

    #compute average across all pdb codes
    columns = ['rmsf', 'bfactor', 'af2_plddt', 'esm_plddt', 'prody_GNM', 'prody_ANM']
    if args.evaluate_flexpert:
        columns.append('flexpert_3d')
        columns.append('flexpert_seq')
    print("Pearson correlations:")
    pearson_mean = np.mean(pearson_correlations, axis=0)
    pearson_mean_rounded = np.round(pearson_mean, 2)
    print(pd.DataFrame(pearson_mean_rounded, index=columns, columns=columns))
    print("\n")
    