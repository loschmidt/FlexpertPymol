
if __name__ == "__main__":
    from tqdm import tqdm
    import json
    import yaml
    
    in_path = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)['backbones_dataset_path']
    out_path = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)['fasta_dataset_path']

    with open(in_path, 'r') as f:
        lines = f.readlines()
        dicts = [json.loads(line.strip()) for line in lines]

    outputs = []
    with open(out_path, 'w') as f:
        for _dict in tqdm(dicts):
            pdb_name = ".".join(_dict['name'].split('_'))
            sequence = _dict['seq']
            f.write(f">{pdb_name}\n{sequence}\n")