#From https://github.com/JoreyYan/zetadesign/blob/master/data/data.py
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


def parse_PDB_biounits(x, sse,ssedssp,atoms=['N', 'CA', 'C'], chain=None):
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
        # ["ARND"] -> [[0,1,2,3]]
        x = np.array(x);
        if x.ndim == 0: x = x[None]
        return [[aa_1_N.get(a, states - 1) for a in y] for y in x]

    def N_to_AA(x):
        # [[0,1,2,3]] -> ["ARND"]
        x = np.array(x);
        if x.ndim == 1: x = x[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

    xyz, seq, plddts, min_resn, max_resn = {}, {}, [],  1e6, -1e6

    pdbcontents = x.split('\n')[0]
    with open(pdbcontents) as f:
        pdbcontents = f.readlines()
    for line in pdbcontents:
        #line = line.decode("utf-8", "ignore").rstrip()

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
                # try:
                #     if 'CA' in xyz[resn][k]:
                #         sse_.append(sse[sseidx])
                #         sseidx = sseidx + 1
                #     else:
                #         sse_.append('-')
                # except:
                #     print('error sse')


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
                        # print(dsspidx)


        else:
            for atom in atoms:
                xyz_.append(np.full(3, np.nan))
            ssedssp_.append('-')


    return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_)),np.array(sse_),np.array(ssedssp_)


def parse_PDB(path_to_pdb,name, input_chain_list=None):
    """
    make sure every time just input 1 line
    """
    c = 0
    pdb_dict_list = []


    if input_chain_list:
        chain_alphabet = input_chain_list
    else:
        init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                         'T',
                         'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                         'n',
                         'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        extra_alphabet = [str(item) for item in list(np.arange(300))]
        chain_alphabet = init_alphabet + extra_alphabet

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''


        for letter in chain_alphabet:

            PDBFile = file.PDBFile.read(biounit)
            array_stack = PDBFile.get_structure(altloc="all")
            
            chain_atoms = array_stack[0][array_stack[0].chain_id == letter]
            sse1 = struc.annotate_sse(chain_atoms).tolist()
            if len(sse1)==0:
                sse1 = struc.annotate_sse(array_stack[0]).tolist()
            #ssedssp1 = dssp.DsspApp.annotate_sse(array_stack).tolist()
            ssedssp1 = [] #not annotating dssp for now


            xyz, seq, _, _= parse_PDB_biounits(biounit,sse1,ssedssp1,atoms=['N', 'CA', 'C','O'], chain=letter) #TODO: fix the float error
            #ssedssp = sse  #faking it for now
            # if len(sse)!=len(seq[0]):
            #     xxxx=len(seq[0])
            #     print(name)
            #assert len(sse)==len(seq[0])
            #assert len(ssedssp) == len(seq[0])

            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_' + letter] = seq[0]

                coords_dict_chain = {}
                coords_dict_chain['N'] = xyz[:, 0, :].tolist()
                coords_dict_chain['CA'] = xyz[:, 1, :].tolist()
                coords_dict_chain['C'] = xyz[:, 2, :].tolist()
                coords_dict_chain['O'] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_' + letter] = coords_dict_chain

                #sse=''.join(sse)
                #ssedssp=''.join(ssedssp)
                #my_dict['sse3' ] = sse
                #my_dict['sse8'] = ssedssp
                s += 1
        #fi = biounit.rfind("/")
        my_dict['name'] = name#biounit[(fi + 1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1
    return pdb_dict_list



def align_pdb_dict_formats(pdb_dict,chain):
    new_dict = {}
    new_dict['seq'] = pdb_dict[f'seq_chain_{chain}']
    new_dict['coords'] = pdb_dict[f'coords_chain_{chain}']
    new_dict['num_chains'] = pdb_dict['num_of_chains']
    new_dict['name'] = pdb_dict['name'] +"_"+chain
    new_dict['CATH'] = ["1.10.150", "3.30.160", "1.10.443"]
    return new_dict


def modify_bfactor_biotite(input_file, chain_id, output_file, flex_prediction):
    """
    Reads a PDB file, modifies the B-factor column, and writes the updated file using Biotite.

    :param input_file: Path to the input PDB file
    :param output_file: Path to save the modified PDB file
    :param flex_prediction: New B-factor value to set (should be a 2D array (1,n_residues))
    """
    # Read the PDB file into an AtomArray
    import biotite.structure as struc
    import biotite.structure.io as strucio
    structure = strucio.load_structure(input_file)
    structure = structure[structure.chain_id == chain_id]
    structure = structure[~structure.hetero]
    
    new_bfactor_column = []

    last_res_id = -1000
    pred_idx = -1
    
    flex_prediction = flex_prediction.cpu().numpy()
    
    for res_id in structure.res_id:
        if res_id != last_res_id:
            new_bfactor_column.append(flex_prediction[0,pred_idx+1])
            last_res_id = res_id
            pred_idx += 1
        else:
            new_bfactor_column.append(flex_prediction[0,pred_idx])
    
    new_bfactors = np.array(new_bfactor_column)

    if "b_factor" not in structure.get_annotation_categories():
        structure.set_annotation("b_factor", new_bfactors)
    else:  # Array of values
        structure.b_factor[:] = new_bfactors
    
    # Save the modified structure to a new PDB file
    strucio.save_structure(output_file, structure)
