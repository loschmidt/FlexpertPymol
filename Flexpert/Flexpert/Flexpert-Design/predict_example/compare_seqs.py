# read in the predictions.txt file
# take the sequence from there
import argparse
import os
import biotite.structure.io.pdb as pdb
from biotite.structure import get_residues

def compare_sequences(pdb_code):
    # Read the predicted sequence from predictions.txt
    with open('predict_example/predictions.txt', 'r') as f:
        predictions = f.readlines()
        # Extract the sequence (skip the header line that starts with '>')
        predicted_seqs = {}
        current_pdb = None
        
        for line in predictions:
            if line.startswith('>'):
                current_pdb = line.strip()[1:]  # Remove the '>' character
            elif current_pdb and line.strip():
                predicted_seqs[current_pdb] = line.strip()
        
        # Use the provided pdb_code to get the corresponding sequence
        predicted_seq = predicted_seqs.get(pdb_code, "")

    # Read the PDB file
    pdb_file = f'predict_example/{pdb_code}.pdb'
    with open(pdb_file, 'r') as f:
        structure = pdb.PDBFile.read(f)
        atoms = pdb.get_structure(structure)
        
        # Get residue names from the structure
        residues = get_residues(atoms)[1]
        # Convert three-letter codes to one-letter codes
        aa_dict = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        pdb_seq = ''.join([aa_dict.get(res, 'X') for res in residues])

    # Compare the two sequences
    match_count = sum(1 for a, b in zip(predicted_seq, pdb_seq) if a == b)
    total_length = max(len(predicted_seq), len(pdb_seq))
    percent_identity = (match_count / min(len(predicted_seq), len(pdb_seq))) * 100

    # Print the result
    print(f"Predicted sequence: {predicted_seq}")
    print(f"PDB sequence:       {pdb_seq}")
    print(f"Sequence length - Predicted: {len(predicted_seq)}, PDB: {len(pdb_seq)}")
    print(f"Matching residues: {match_count}/{min(len(predicted_seq), len(pdb_seq))}")
    print(f"Percent identity: {percent_identity:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare predicted sequence with PDB sequence')
    parser.add_argument('--pdb_code', type=str, help='PDB code (e.g., 1ah7_A)')
    args = parser.parse_args()
    
    compare_sequences(args.pdb_code)
