import prody
from prody import ANM, GNM
from prody.dynamics.analysis import calcSqFlucts
import numpy as np
from biotite.structure.io.pdb import PDBFile
import pdb
import os
import json
from tqdm import tqdm
import yaml

def get_fluctuation_for_pdbfile(pdb_name, pdb_path, chain = None, enm_type = 'ANM'): 
    if enm_type not in ('ANM', 'GNM'):
        raise ValueError("enm_type must be 'ANM' or 'GNM'")
    
    if not chain:
        prodypdb = prody.parsePDB(pdb_path)
    else:
        prodypdb = prody.parsePDB(pdb_path, chain=chain)

    calphas = prodypdb.select("protein and name CA")
    sequence = calphas.getSequence()
    if enm_type == 'GNM':
        enm = GNM('gnm')
        enm.buildKirchhoff(calphas, cutoff=16., gamma=1.)
    elif enm_type == 'ANM':
        enm = ANM('anm')
        enm.buildHessian(calphas, cutoff=16.)
    enm.calcModes(3*len(calphas)-6)
    sq_flucts = calcSqFlucts(enm)
    flucts = np.sqrt(sq_flucts)
    return flucts, sequence

def get_fluctuation_for_json_dict(dict, chain = None, enm_type = 'ANM'):
    coords, elements = [], []
    nan_idcs = []
    for atom_name in ['CA']:#['N', 'CA', 'C', 'O']:
        for res_idx, atoms in enumerate(dict['coords'][atom_name]):
            if len([1 for a in atoms if np.isnan(a) or not np.isfinite(a)]) > 0:
                nan_idcs.append(res_idx)
                continue
            else:
                coords.append(atoms)  # array documentation to improve
                elements.append(atom_name)
    
    prody_molecule = prody.AtomGroup()
    prody_molecule.setCoords(coords)
    prody_molecule.setElements(elements)

    calphas = prody_molecule#prody_molecule.select("ca")
    if enm_type == 'GNM':
        enm = GNM('gnm')
        enm.buildKirchhoff(calphas, cutoff=16., gamma=1.)
    elif enm_type == 'ANM':
        enm = ANM('anm')
        enm.buildHessian(calphas, cutoff=16.)
    
    try:
        enm.calcModes(3*len(calphas)-6)
        sq_flucts = calcSqFlucts(enm)
        flucts = np.sqrt(sq_flucts)
    except:
        flucts = [np.nan for _ in range(len(calphas))]
    
    padded_flucts = []
    _padded_idcs_counter = 0
    for idx in range(len(flucts)+len(nan_idcs)):
        if idx in nan_idcs:
            padded_flucts.append(np.nan)
            _padded_idcs_counter += 1
        else:
            padded_flucts.append(flucts[idx-_padded_idcs_counter])
    flucts = np.array(padded_flucts)

    return flucts, dict['seq']


def write_jsonl(jsonl, output_filename):
    with open(output_filename, 'w') as f:
        for line in jsonl:
            f.write(json.dumps(line) + '\n')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Calculate fluctuations for a dataset of pdb files')
    parser.add_argument('--enm', type=str, default='ANM', help='Select ANM or GNM.')
    parser.add_argument('--subselect_chain', type=int, default=0, help='If set to 1, the name of the PDB is assumed to be in the format PDB_CHAINID and used to explicitely read just the chain.')
    #add store_true argument flag to distinguish chains, by default no flag means False, if flag added set it to true

    args = parser.parse_args()

    in_path = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)['backbones_dataset_path']
    out_path = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)['out_enm_dataset_path'].format(args.enm)

    if in_path.endswith('.csv'):
        with open(in_path, 'r') as f:
            pdb_paths = {line.strip().split(',')[0]: line.strip().split(',')[1] for line in f.readlines()}
        outputs = []
        for pdb_name, pdb_path in tqdm(pdb_paths.items()):
            if args.subselect_chain:
                chain = pdb_name.split('_')[1] #assuminh pdb_name is in the form PDB_CHAINID
            else:
                chain = None
            flucts, sequence = get_fluctuation_for_pdbfile(pdb_name=pdb_name, pdb_path=pdb_path, enm_type=args.enm, chain=chain)
            outputs.append({'pdb_name': pdb_name, 'fluctuations': flucts.tolist(), 'sequence': sequence})

    elif in_path.endswith('.jsonl'):
        with open(in_path, 'r') as f:
            lines = f.readlines()
            dicts = [json.loads(line.strip()) for line in lines]
        outputs = []
        for _dict in tqdm(dicts):
            #TODO: do I need to subselect the chain here?
            flucts, sequence = get_fluctuation_for_json_dict(_dict, enm_type=args.enm)
            outputs.append({'pdb_name': '_'.join(_dict['name'].split('.')), 'fluctuations': flucts.tolist(), 'sequence': sequence})
        #raise NotImplementedError('TODO: Implement the parsing of the .json') #TODO: implement this part to read the CATH4.3 dataset from the json(l)
    else:
        raise ValueError("input_structures are expected to be a csv file or a jsonl file")

    write_jsonl(outputs, out_path)