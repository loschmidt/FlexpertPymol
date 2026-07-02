import os
import json
import numpy as np
import random
import pdb
import torch.utils.data as data
from .utils import cached_property
from transformers import AutoTokenizer

#Imports for the PDB parser utils
import glob
import json
import numpy as np
import gzip
import re
import multiprocessing
import tqdm
import shutil
SENTINEL = 1
import biotite.structure as struc
import biotite.application.dssp as dssp
import biotite.structure.io.pdb.file as file

class PDBInference(data.Dataset):
    def __init__(self, path='./',  max_length=500, *args, **kwargs):
        self.path = path
        self.max_length = max_length

        self.data = self.cache_data #TODO
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="./cache_dir/")
    
    @cached_property
    def cache_data(self):
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])
        print("path is: ", self.path)

        if not os.path.exists(self.path):
            raise "no such folder:{} !!!".format(self.path)
        else:

            #list all PDBs
            pdb_files = []
            _files = os.listdir(self.path)
            for _file in _files:
                if _file.endswith('.pdb'):
                    pdb_files.append(_file)
            print(f'pdb_files size = {len(pdb_files)}')
            #parse the PDBs into lines like if it was from the chain_set.json
            lines = []
            for _pdb in pdb_files:
                _input_chain = _pdb.split('_')[1].split('.')[0] #ASSUMING NAMING 'PDBCODE_CHAINCODE_XXX'
                _line = self.parse_PDB(self.path+'/'+_pdb, name=_pdb.split('.')[0], input_chain=_input_chain) #Input chain list can be parsed here as well
                #pdb.set_trace()
                lines.append(_line[0])
            
            print(f'lines size = {len(lines)}')
            data_list = []

            flex_instructions = {}
            flexibility_files = glob.glob(self.path + '/*instructions.csv')
            for file in flexibility_files:
                with open(file, 'r') as f:
                    flexibility_instructions_parsed= f.read().strip().split(',')
                    flexibility_instructions_parsed = [float(i) for i in flexibility_instructions_parsed] + [0.0] #add the padding here
                    flex_instructions[file.split('/')[-1].split('_instructions')[0]] = flexibility_instructions_parsed

            for line in tqdm.tqdm(lines):
                entry = line

                seq = entry['seq']

                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)
                
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                try:
                    _flex_instructions = flex_instructions[entry['name']]
                except KeyError:
                    _flex_instructions = [0.0] * len(seq)
                    print(f"No flexibility instructions found for {entry['name']}. Passing zeros.")

                if len(bad_chars) == 0:
                    if len(entry['seq']) <= self.max_length: 
                        chain_length = len(entry['seq'])
                        chain_mask = np.ones(chain_length)
                        data_list.append({
                            'title':entry['name'],
                            'seq':entry['seq'],
                            'CA':entry['coords']['CA'],
                            'C':entry['coords']['C'],
                            'O':entry['coords']['O'],
                            'N':entry['coords']['N'],
                            'chain_mask': chain_mask,
                            'chain_encoding': 1*chain_mask,
                            'gt_flex': _flex_instructions
                        })
                else:
                    print(f'Skipping PDBs with Bad chars, e.g. gaps in the sequence: {entry["name"]}')
            
            #data_dict = {'train':[],'valid':data_list,'test':data_list}
            print(f'data_list size = {len(data_list)}')
            return data_list#data_dict

    def change_mode(self, mode):
        self.data = self.cache_data[mode]
    
    def __len__(self):
        return len(self.data)
    
    def get_item(self, index):
        return self.data[index]
    
    def __getitem__(self, index):
        item = self.data[index]
        L = len(item['seq'])
        if L>self.max_length:
            # 计算截断的最大索引
            max_index = L - self.max_length
            # 生成随机的截断索引
            truncate_index = random.randint(0, max_index)
            # 进行截断
            item['seq'] = item['seq'][truncate_index:truncate_index+self.max_length]
            item['CA'] = item['CA'][truncate_index:truncate_index+self.max_length]
            item['C'] = item['C'][truncate_index:truncate_index+self.max_length]
            item['O'] = item['O'][truncate_index:truncate_index+self.max_length]
            item['N'] = item['N'][truncate_index:truncate_index+self.max_length]
            item['chain_mask'] = item['chain_mask'][truncate_index:truncate_index+self.max_length]
            item['chain_encoding'] = item['chain_encoding'][truncate_index:truncate_index+self.max_length]
            item['gt_flex'] = item['gt_flex'][truncate_index:truncate_index+self.max_length]
        return item

    #Code from data_utils on local PC, based on: https://github.com/JoreyYan/zetadesign/blob/master/data/data.py
    def parse_PDB_biounits(self, x, sse,ssedssp,atoms=['N', 'CA', 'C'], chain=None):
        '''
        input:  x = PDB filename
                atoms = atoms to extract (optional)
        output: (length, atoms, coords=(x,y,z)), sequence
        '''

        alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
        states = len(alpha_1)
        alpha_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP']

        aa_1_N = {a: n for n, a in enumerate(alpha_1)}
        aa_3_N = {a: n for n, a in enumerate(alpha_3)}
        aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
        aa_1_3 = {a: b for a, b in zip(alpha_1, alpha_3)}
        aa_3_1 = {b: a for a, b in zip(alpha_1, alpha_3)}

        def AA_to_N(x):
            x = np.array(x)
            if x.ndim == 0: x = x[None]
            return [[aa_1_N.get(a, states - 1) for a in y] for y in x]

        def N_to_AA(x):
            x = np.array(x)
            if x.ndim == 1: x = x[None]
            return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

        xyz, seq, plddts, min_resn, max_resn = {}, {}, [],  1e6, -1e6

        pdbcontents = x.split('\n')[0]
        with open(pdbcontents) as f:
            pdbcontents = f.readlines()
        for line in pdbcontents:

            if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
                line = line.replace("HETATM", "ATOM  ")
                line = line.replace("MSE", "MET")

            if line[:4] == "ATOM":
                ch = line[21:22]
                if ch == chain or chain is None or ch==' ':
                    atom = line[12:12 + 4].strip()
                    resi = line[17:17 + 3]
                    resn = line[22:22 + 5].strip()
                    plddt=line[60:60 + 6].strip()



                    x, y, z = [float(line[i:(i + 8)]) for i in [30, 38, 46]]

                    if resn[-1].isalpha():
                        resa, resn = resn[-1], int(resn[:-1]) - 1 # in same pos ,use last atoms
                    else:
                        resa, resn = "_", int(resn) - 1
                    #         resn = int(resn)
                    if resn < min_resn:
                        min_resn = resn
                    if resn > max_resn:
                        max_resn = resn



                    if resn not in xyz:
                        xyz[resn] = {}
                    if resa not in xyz[resn]:
                        xyz[resn][resa] = {}
                    if resn not in seq:
                        seq[resn] = {}

                    if resa not in seq[resn]:
                        seq[resn][resa] = resi

                    if atom not in xyz[resn][resa]:
                        xyz[resn][resa][atom] = np.array([x, y, z])

        # convert to numpy arrays, fill in missing values
        seq_, xyz_ ,sse_,ssedssp_= [], [], [], []
        dsspidx=0
        sseidx=0

        for resn in range(int(min_resn), int(max_resn + 1)):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(aa_3_N.get(seq[resn][k], 20))
                    try:
                        if 'CA' in xyz[resn][k]:
                            sse_.append(sse[sseidx])
                            sseidx = sseidx + 1
                        else:
                            sse_.append('-')
                    except:
                        print('error sse')


            else:
                seq_.append(20)
                sse_.append('-')

            misschianatom = False
            if resn in xyz:


                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])  #some will miss C and O ,but sse is normal,because sse just depend on CA
                        else:
                            xyz_.append(np.full(3, np.nan))
                            misschianatom=True
                    if misschianatom:
                        ssedssp_.append('-')
                        misschianatom = False
                    else:
                        try:
                            ssedssp_.append(ssedssp[dsspidx])         # if miss chain atom,xyz ,seq think is ok , but dssp miss this
                            dsspidx = dsspidx + 1
                        except:
                            pass
                            #print(dsspidx)


            else:
                for atom in atoms:
                    xyz_.append(np.full(3, np.nan))
                ssedssp_.append('-')


        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_)),np.array(sse_),np.array(ssedssp_)

    def parse_PDB(self, path_to_pdb, name, input_chain):
        """
        make sure every time just input 1 line
        """
        c = 0
        pdb_dict_list = []


        biounit_names = [path_to_pdb]
        for biounit in biounit_names:
            my_dict = {}
            s = 0
            concat_seq = ''


            letter = input_chain #Assuming single chain!!

            PDBFile = file.PDBFile.read(biounit)
            array_stack = PDBFile.get_structure(altloc="all")

            #In case the passed letter is unknown, select one chain from the PDB file based on the dominant protein chain
            if letter not in array_stack.chain_id:
                is_protein = struc.filter_amino_acids(array_stack)
                protein_atoms = array_stack[0][is_protein]
                chain_ids, chain_counts = np.unique(protein_atoms.chain_id, return_counts=True)
                dominant_chain_id = chain_ids[np.argmax(chain_counts)]
                letter = dominant_chain_id


            sse1 = struc.annotate_sse(array_stack[0], chain_id=letter).tolist()
            if len(sse1)==0:
                sse1 = struc.annotate_sse(array_stack[0], chain_id='').tolist()

            ssedssp1 = [] #not annotating dssp for now


            xyz, seq, sse, ssedssp = self.parse_PDB_biounits(biounit,sse1,ssedssp1,atoms=['N', 'CA', 'C','O'], chain=letter) #TODO: fix the float error
            ssedssp = sse  #faking it for now

            assert len(sse)==len(seq[0])
            assert len(ssedssp) == len(seq[0])

            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_' + letter] = seq[0]

                coords_dict_chain = {}
                coords_dict_chain['N'] = xyz[:, 0, :].tolist()
                coords_dict_chain['CA'] = xyz[:, 1, :].tolist()
                coords_dict_chain['C'] = xyz[:, 2, :].tolist()
                coords_dict_chain['O'] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_' + letter] = coords_dict_chain
                my_dict['coords'] = coords_dict_chain
                s += 1

            # if s>1:
            #     raise NotImplementedError('Inference so far implemented only for single chain proteins')

            my_dict['name'] = name
            my_dict['num_chains'] = s
            my_dict['seq'] = my_dict[f'seq_chain_{letter}'] #concat_seq
            # if s <= len(chain_alphabet):
            #     pdb_dict_list.append(my_dict)
            #     c += 1
            pdb_dict_list.append(my_dict)
        return pdb_dict_list