import os, sys, warnings, argparse, math, tqdm, datetime
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer import Trainer
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
from model_interface import MInterface
from data_interface import DInterface
from src.tools.logger import SetupCallback, BackupCodeCallback
from shutil import ignore_patterns
from transformers import AutoTokenizer
import numpy as np
import yaml
import wandb
warnings.filterwarnings("ignore")

def create_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument('--infer_path', type=str, help='Path where to read the data to be predicted and where to save the predictions.')

    # Set-up parameters
    parser.add_argument('--res_dir', default='./train/results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--stage', default='predict', type=str) #'fit', 'test' or 'predict'
    parser.add_argument('--val_check_interval', default=0.5, type=float, help='Validation check interval')
    
    parser.add_argument('--dataset', default='PDBInference') # AF2DB_dataset, CATH_dataset
    parser.add_argument('--model_name', default='ProteinMPNN', choices=['StructGNN', 'GraphTrans', 'GVP', 'GCA', 'AlphaDesign', 'ESMIF', 'PiFold', 'ProteinMPNN', 'KWDesign', 'E3PiFold'])
    # parser.add_argument('--lr', default=4e-4, type=float, help='Learning rate')
    # parser.add_argument('--lr_scheduler', default='onecycle')
    # parser.add_argument('--offline', default=1, type=int)
    parser.add_argument('--seed', default=111, type=int)
    
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pad', default=1024, type=int)
    parser.add_argument('--min_length', default=40, type=int)
    parser.add_argument('--data_root', default='./data/')
    
    # Training parameters
    # parser.add_argument('--epoch', default=10, type=int, help='end epoch')
    parser.add_argument('--augment_eps', default=0.0, type=float, help='noise level')
    # parser.add_argument('--gpus', default=1, type=int, help='how many GPUs to train on')
    # parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay for optimizer')

    # # Eval parameters
    # parser.add_argument('--eval_sequences_sampled', default=1, type=int, help='How many sequences to sample in evaluation.')
    # parser.add_argument('--eval_sequences_temperature', default=0, type=float, help='What temperature to use for the sampling in evaluation.')
    # parser.add_argument('--eval_output_dir', default=None, type=str, help='Where to save the evaluation output.')

    # Model parameters
    parser.add_argument('--use_dist', default=1, type=int)
    parser.add_argument('--use_product', default=0, type=int)
    parser.add_argument('--use_pmpnn_checkpoint', default=1, type=int, help='By 1 or 0 decide whether to start with pretrained ProteinMPNN.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the model checkpoint to load weights from')

    # Dynamics aware parameters
    parser.add_argument('--use_dynamics', default=0, type=int)
    # parser.add_argument('--flex_loss_coeff', default=0.5, type=float)
    # parser.add_argument('--get_gt_flex_onthefly', default=0, type=int, help='Flag to get ground truth flexibility on-the-fly (with subsequent caching)')
    parser.add_argument('--init_flex_features', default=1, type=int, help="Set to 0 if no flexibility information should be passed on input to the node features h_V")
    # parser.add_argument('--loss_fn', default='MSE', type=str, help= 'Define what loss to use. Choose MSE, L1 or DPO.')
    # parser.add_argument('--grad_normalization', default=1, type=int, help="Set to 0 if the gradients of the seq and flex losses should not be normalized.")
    # parser.add_argument('--test_engineering', default=0, type=int, help="In this main.py should be set to 0 to not overwrite the training dataset.")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = create_parser()
    args.batch_size = 1
    print('In the predict stage, defaulting batch size to 1.')

    assert args.use_dynamics == 0, "In the inference script this should be set to 0."

    if not os.path.exists(args.infer_path):
        os.makedirs(args.infer_path)
        
    if (len(args.infer_path) > 0 or args.dataset=='PDBInference') and (len(args.infer_path) == 0 or args.dataset!='PDBInference'):
        raise ValueError("You should only use --infer_path with --dataset 'PDBInference' and vice versa.")


    # Load model weights from checkpoint if provided
    if args.checkpoint_path is not None:
        trained_model_path = args.checkpoint_path
        print(f"Loading model weights from checkpoint passed by argument: {trained_model_path}")
    else:
        with open('configs/Flexpert-Design-inference.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        trained_model_path = config['pmpnn_model_path']
        print(f"Loading model weights from checkpoint specified in Flexpert-Design-inference.yaml: {trained_model_path}")

    if os.path.exists(trained_model_path):
        print(f"Rewriting the path to the Flexpert-Design trained ProteinMPNN weights in the model interface.")
        args.starting_checkpoint_path = trained_model_path
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {trained_model_path}")

    pl.seed_everything(args.seed)

    data_module = DInterface(**vars(args))

    data_module.setup(stage='predict')

    model = MInterface(**vars(args))


    trainer_config = {
        'devices': 1,
        'max_epochs': 1,
        'num_nodes': 1,
        "strategy": 'ddp',
        "precision": '32',
        'accelerator': 'gpu',
        'val_check_interval': args.val_check_interval,
        'check_val_every_n_epoch': args.check_val_every_n_epoch
    }

    trainer = Trainer(**trainer_config)

    predictions = trainer.predict(model, data_module)

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir='./cache_dir/') # mask token: 32


    serializable_predictions = []
    for pred_idx, pred in enumerate(predictions):
        logprobs = pred['log_probs'].cpu().numpy()[0]  # [L, 21]
        pmpnn_alphabet_tokens_argmax = logprobs.argmax(axis=-1)  # [L]
        
        aa_sequence = ''.join(tokenizer.decode(pmpnn_alphabet_tokens_argmax, skip_special_tokens=True).split())

        # Get probability of the predicted sequence
        seq_probs = np.exp(logprobs.max(axis=-1))  # [L]
        avg_prob = float(np.mean(seq_probs))
        
        serializable_predictions.append({
            'prediction_id': pred['batch']['title'][0],
            'amino_acid_sequence': aa_sequence
        })

    with open(f'{args.infer_path}/predictions.txt', 'w') as f:
        for pred in serializable_predictions:
            f.write(f'>{pred["prediction_id"]}\n{pred["amino_acid_sequence"]}\n')
