import os
import yaml
import pandas as pd
def extract_rmsf_labels(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        protein_id = file_path.split('/')[-2].split('_')[:-1]
        protein_id = '.'.join(protein_id)
        rmsf_values = []
        for line in lines[1:]:
            parts = line.strip().split('\t')
            rmsf_r1 = float(parts[1])
            rmsf_r2 = float(parts[2])
            rmsf_r3 = float(parts[3])
            avg_rmsf = (rmsf_r1 + rmsf_r2 + rmsf_r3) / 3
            rmsf_values.append(avg_rmsf)
    return protein_id, rmsf_values

def extract_bfactor_labels(file_path):
    bfactor = pd.read_csv(file_path, delimiter='\t')['Bfactor']
    protein_id = file_path.split('/')[-2].split('_')[:-1]
    protein_id = '.'.join(protein_id)
    return protein_id, bfactor

def extract_plddt_labels(file_path):
    plddt = pd.read_csv(file_path, delimiter='\t')['pLDDT']
    protein_id = file_path.split('/')[-2].split('_')[:-1]
    protein_id = '.'.join(protein_id)
    return protein_id, plddt

if __name__ == "__main__":
    config = yaml.load(open('configs/data_config.yaml'), Loader=yaml.FullLoader)
    in_path = config['atlas_out_dir']
    out_path = config['atlas_labels_path']
    rmsf_data = {}

    for folder in os.listdir(in_path):
        folder_path = os.path.join(in_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith("_RMSF.tsv"):
                    file_path = os.path.join(folder_path, file)
                    protein_id, rmsf_labels = extract_rmsf_labels(file_path)
                    rmsf_data[protein_id] = rmsf_labels
    
    with open(out_path, 'w') as out_file:
        for protein_id, values in rmsf_data.items():
            out_file.write(f"{protein_id}: {', '.join(map(str, values))}\n")
