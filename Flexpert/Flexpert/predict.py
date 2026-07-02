from data.scripts.data_utils import parse_PDB
from utils.utils import ClassConfig, DataCollatorForTokenRegression, process_in_batches_and_combine, get_dot_separated_name
from models.T5_encoder_per_token import PT5_classification_model
from data.scripts.get_enm_fluctuations_for_dataset import get_fluctuation_for_json_dict
import argparse
import os
import yaml
import torch
from pathlib import Path
from Bio import SeqIO, BiopythonDeprecationWarning
import json
import warnings
from datetime import datetime
import warnings
from data.scripts.data_utils import modify_bfactor_biotite

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file")
    parser.add_argument("--modality", type=str, required=True, help="Indicate 'Seq' or '3D' to use Flexpert-Seq or Flexpert-3D?")
    parser.add_argument("--splits_file", type=str, required=False, help="Path to the file defining the splits, in case that input_file is a dataset which should be subsampled.")
    parser.add_argument("--split", type=str, required=False, help="Specify test/train/val to subselect the respective split. If specified, the splits file needs to be provided as well.")
    parser.add_argument("--output_enm", action='store_true', help="If true, the ENM values will be outputted in separate file(s).")
    parser.add_argument("--output_fasta", action='store_true', help="If true, the sequences used for the prediction will be outputted in a fasta file (can be relevant when working with input list of PDB files).")
    parser.add_argument("--output_file", type=str, required=False, help="Name of the output file.")
    parser.add_argument("--chain", type=str, required=False, help="chain to predict.")
    args = parser.parse_args()

    args.modality = args.modality.upper()
    filename, suffix = os.path.splitext(args.input_file)
    
    if args.modality not in ["SEQ", "3D"]:
        raise ValueError("Modality must be either Seq or 3D")
    if args.splits_file is not None and args.split is None:
        raise ValueError("If splits_file is provided, split must be specified.")
    if args.split is not None and args.splits_file is None:
        raise ValueError("If split is specified, splits_file must be provided.")
    if args.split is not None and args.split not in ["test", "train", "val", "validation"]:
        raise ValueError("Split must be either 'test', 'train', 'val' or 'validation'")
    if args.output_enm and (args.modality not in ["3D"]):
        raise ValueError("Output ENM is only supported for 3D modality")
    if not args.output_file:
        default_name = 'untitled_{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
        args.output_file = default_name
        warnings.warn("Output name is not provided, using default name: {}".format(default_name))
        

    if args.splits_file is not None:
        with open(args.splits_file, 'r') as f:
            splits = json.load(f)
        if 'val' in splits.keys() and args.split == 'validation':
            args.split = 'val'
        elif 'validation' in splits.keys() and args.split == 'val':
            args.split = 'validation'
        
        datapoint_for_eval = splits[args.split]
    else:
        datapoint_for_eval = 'all'

    sequences = []
    names = []
    backbones = []
    pdb_files = []
    flucts_list = []

    def process_pdb_file(pdb_file, backbones, sequences, names):
        parsed_name = os.path.splitext(os.path.basename(pdb_file))[0].split('_')
        # if len(parsed_name[0]) != 4 or len(parsed_name[1]) != 1 or not parsed_name[1].isalpha():
        #     raise ValueError("PDB file name is expected to be in the format of 'name_chain.pdb', e.g.: 1BUI_C.pdb")
        _name = parsed_name[0]
        _chain = args.chain
        parsed_pdb = parse_PDB(pdb_file, name=_name, input_chain_list=[_chain])[0]
        backbone, sequence = parsed_pdb['coords_chain_{}'.format(_chain)], parsed_pdb['seq_chain_{}'.format(_chain)]
        if len(sequence) > 1023:
            print("Sequence length is greater than 1023, skipping {}".format(_name + "." + _chain))
        else:
            backbones.append(backbone)
            sequences.append(sequence)
            names.append(_name + "." + _chain)
        return backbones, sequences, names

    if suffix == ".fasta":
        if args.modality == "3D":
            raise ValueError("Flexpert-3D needs the structure, fasta is not enough")

        # Load FASTA file using Biopython
        for record in SeqIO.parse(args.input_file, "fasta"):
            if '_' in record.name:
                dot_separated_name = '.'.join(record.name.split('_'))
            elif '.' in record.name:
                dot_separated_name = record.name
            else:
                raise ValueError("Sequence name must contain either an underscore or a dot to separate the PDB code and the chain code.")
            if datapoint_for_eval == 'all' or dot_separated_name in datapoint_for_eval:
                names.append(dot_separated_name)
                sequences.append(str(record.seq))
                backbones.append(None)

    elif suffix == ".pdb":
        backbones, sequences, names = process_pdb_file(args.input_file, backbones, sequences, names)
        pdb_files.append(args.input_file)

    elif suffix == ".jsonl":
        for line in open(args.input_file, 'r'):
            _dict = json.loads(line)

            if 'fluctuations' in _dict.keys():
                print("fluctuations are precomputed, using them")
                dot_separated_name = get_dot_separated_name(key='pdb_name', _dict=_dict)
                if datapoint_for_eval == 'all' or dot_separated_name in datapoint_for_eval:
            
                    names.append(_dict['pdb_name'])
                    backbones.append(None)
                    sequences.append(_dict['sequence'])

                    flucts_list.append(_dict['fluctuations']+[0.0]) #padding for end cls token
                continue
            
            dot_separated_name = get_dot_separated_name(key='name', _dict=_dict)
            
            if datapoint_for_eval == 'all' or dot_separated_name in datapoint_for_eval:
                backbones.append(_dict['coords'])
                sequences.append(_dict['seq'])
                names.append(dot_separated_name)

    elif suffix == ".pdb_list":
        with open(args.input_file, 'r') as f:
            pdb_files = f.read().splitlines()
        for pdb_file in pdb_files:
            backbones, sequences, names = process_pdb_file(pdb_file, backbones, sequences, names)

    else:
        raise ValueError("Input file must be a fasta, pdb, jsonl file or a pdb list file")

    ### Set environment variables
    env_config = yaml.load(open(BASE_DIR + "/configs/env_config.yaml", 'r'), Loader=yaml.FullLoader)
    # Set folder for huggingface cache
    
    #os.environ['HF_HOME'] = env_config['huggingface']['HF_HOME']
    # Set gpu device
    os.environ["CUDA_VISIBLE_DEVICES"]= env_config['gpus']['cuda_visible_device']

    config = yaml.load(open(BASE_DIR + '/configs/train_config.yaml', 'r'), Loader=yaml.FullLoader)
    class_config=ClassConfig(config)
    class_config.adaptor_architecture = 'no-adaptor' if args.modality == 'SEQ' else 'conv'
    model, tokenizer = PT5_classification_model(half_precision=config['mixed_precision'], class_config=class_config)

    model.to(config['inference_args']['device'])
    if args.modality == 'SEQ':
        state_dict = torch.load(os.getenv('MODEL_HOME') + "/weights/flexpert_seq_weights.bin", map_location=config['inference_args']['device'])
        model.load_state_dict(state_dict, strict=False)
    elif args.modality == '3D':
        print("Loading 3D model from {}".format(config['inference_args']['3d_model_path']))
        state_dict = torch.load(os.getenv('MODEL_HOME') + "/weights/flexpert_3d_weights.bin", map_location=config['inference_args']['device'])
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    data_to_collate = []
    for idx, (backbone, sequence) in enumerate(zip(backbones, sequences)):
        
        if args.modality == '3D':
            if backbone is not None:
                _dict = {'coords': backbone, 'seq': sequence}
                flucts, _ = get_fluctuation_for_json_dict(_dict, enm_type = config['inference_args']['enm_type'])
                flucts = flucts.tolist()
                flucts.append(0.0) #To match the special token for the sequence
                flucts = torch.tensor(flucts).to(config['inference_args']['device'])
            else:
                flucts = flucts_list[idx]

        #Ensure that the missing residues in the sequence are not represented as '-' but as 'X'
        sequence = sequence.replace('-', 'X') #due to the tokenizer vocabulary

        tokenizer_out = tokenizer(' '.join(sequence), add_special_tokens=True, return_tensors='pt')
        tokenized_seq, attention_mask = tokenizer_out['input_ids'].to(config['inference_args']['device']), tokenizer_out['attention_mask'].to(config['inference_args']['device'])
        
        if args.modality == '3D':
            data_to_collate.append({'input_ids': tokenized_seq[0,:], 'attention_mask': attention_mask[0,:], 'enm_vals': flucts})
        elif args.modality == 'SEQ':
            data_to_collate.append({'input_ids': tokenized_seq[0,:], 'attention_mask': attention_mask[0,:]})

    # Use the data collator to process the input
    data_collator = DataCollatorForTokenRegression(tokenizer)

    batch = data_collator(data_to_collate)  # Wrap in list since collator expects batch
    batch.to(model.device)
    for key in batch.keys():
        print("___________-", key, "-___________")
        for b in batch[key]:
            if key == 'attention_mask':
                print(b.sum())
            else:
                print(b.shape)

    # Predict
    with torch.no_grad():
        output_logits = process_in_batches_and_combine(model, batch, config['inference_args']['batch_size'])
        predictions = output_logits[:,:,0] #includes the prediction for the added token
        # subselect the predictions using the attention mask
    
    #output_filename = Path(config['inference_args']['prediction_output_dir'].format(args.output_name, args.modality, 'all' if not args.split else args.split))
    #output_filename.parent.mkdir(parents=True, exist_ok=True)

    #Write the predictions to files
    with open(args.output_file, 'w') as f:
        print("Saving predictions to {}.".format(args.output_file))
        for prediction, mask, name, sequence in zip(predictions, batch['attention_mask'], names, sequences):
            prediction = prediction[mask.bool()]
            if len(prediction) != len(sequence)+1:
                print("Prediction length {} is not equal to sequence length + 1 {}".format(len(prediction), len(sequence)+1))

            assert len(prediction) == len(sequence)+1, "Prediction length {} is not equal to sequence length + 1 {}".format(len(prediction), len(sequence)+1)
            if '.' in name:
                name = name.replace('.', '_')
            f.write('>' + name + '\n')
            f.write(', '.join([str(p) for p in prediction.tolist()[:-1]]) + '\n')
    
    # if suffix == ".pdb" or suffix == ".pdb_list":
    #     for name, pdb_file, prediction in zip(names, pdb_files, predictions):
    #         chain_id = name.split('.')[1]
    #         _prediction = prediction[:-1].reshape(1,-1)
    #         _outname = args.output_file.with_name(output_filename.stem + '_{}.pdb'.format(name.replace('.', '_')))
    #         print("Saving prediction to {}.".format(_outname))
    #         modify_bfactor_biotite(pdb_file, chain_id, _outname, _prediction) #writing the prediction without the last token
    #
    # if args.output_enm:
    #     _outname = output_filename.with_name(output_filename.stem + '_enm.txt')
    #     with open(_outname, 'w') as f:
    #         print("Saving ENM predictions to {}.".format(_outname))
    #         for enm_prediction, name in zip(batch['enm_vals'], names):
    #             f.write('>' + name + '\n')
    #             f.write(', '.join([str(p) for p in enm_prediction.tolist()[:-1]]) + '\n')
    #
    #     if suffix == ".pdb" or suffix == ".pdb_list":
    #         for name, pdb_file, enm_vals_single in zip(names, pdb_files, batch['enm_vals']):
    #             _outname = output_filename.with_name(output_filename.stem + '_{}.pdb'.format(name.replace('.', '_')))
    #             print("Saving ENM prediction to {}.".format(_outname))
    #             chain_id = name.split('.')[1]
    #             _enm_vals = enm_vals_single[:-1].reshape(1,-1)
    #             modify_bfactor_biotite(pdb_file, chain_id, _outname, _enm_vals) #writing the prediction without the last token
    #
    # if args.output_fasta:
    #     _outname = output_filename.with_name(output_filename.stem + '_fasta.fasta')
    #     with open(_outname, 'w') as f:
    #         print("Saving fasta to {}.".format(_outname))
    #         for name, sequence in zip(names, sequences):
    #             f.write('>' + name + '\n')
    #             f.write(sequence + '\n')
