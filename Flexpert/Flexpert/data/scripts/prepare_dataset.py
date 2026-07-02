from data_utils import parse_PDB, align_pdb_dict_formats
import os
import re
import json
import yaml
from tqdm import tqdm

in_dir = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)['pdb_dir']
out_dir = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)['preprocessed_dir']

fold_list = []
fold_files = os.listdir(in_dir)
fold_files = [filename for filename in fold_files if re.match(".*\.pdb$", filename)]

for file in tqdm(fold_files):
    _name= file.split('_')[0]
    _chain = file.split('_')[1].split('.')[0]
    _path = f'{in_dir}/{file}'
    old_pdb = parse_PDB(_path,name=_name, input_chain_list=[_chain])[0]
    new_pdb = align_pdb_dict_formats(old_pdb,_chain)
    fold_list.append(new_pdb)

with open(f'{out_dir}/chain_set.jsonl','w') as f:
    for dict in fold_list:
        json.dump(dict,f)
        f.write('\n')

