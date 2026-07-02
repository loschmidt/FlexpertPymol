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

import wandb
warnings.filterwarnings("ignore")

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--res_dir', default='./train/results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--stage', default='fit', type=str) #'fit', 'test' or 'predict'
    parser.add_argument('--val_check_interval', default=0.5, type=float, help='Validation check interval')
    
    parser.add_argument('--dataset', default='FLEX_CATH4.3') # AF2DB_dataset, CATH_dataset
    parser.add_argument('--model_name', default='ProteinMPNN', choices=['StructGNN', 'GraphTrans', 'GVP', 'GCA', 'AlphaDesign', 'ESMIF', 'PiFold', 'ProteinMPNN', 'KWDesign', 'E3PiFold'])
    parser.add_argument('--lr', default=4e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_scheduler', default='onecycle')
    parser.add_argument('--offline', default=1, type=int)
    parser.add_argument('--seed', default=111, type=int)
    
    # dataset parameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pad', default=1024, type=int)
    parser.add_argument('--min_length', default=40, type=int)
    parser.add_argument('--data_root', default='../data/')
    parser.add_argument('--infer_path', default='', type=str)
    
    # Training parameters
    parser.add_argument('--epoch', default=10, type=int, help='end epoch')
    parser.add_argument('--augment_eps', default=0.0, type=float, help='noise level')
    parser.add_argument('--gpus', default=1, type=int, help='how many GPUs to train on')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay for optimizer')

    # Eval parameters
    parser.add_argument('--eval_sequences_sampled', default=1, type=int, help='How many sequences to sample in evaluation.')
    parser.add_argument('--eval_sequences_temperature', default=0, type=float, help='What temperature to use for the sampling in evaluation.')
    parser.add_argument('--eval_output_dir', default=None, type=str, help='Where to save the evaluation output.')

    # Model parameters
    parser.add_argument('--use_dist', default=1, type=int)
    parser.add_argument('--use_product', default=0, type=int)
    parser.add_argument('--use_pmpnn_checkpoint', type=int, help='By 1 or 0 decide whether to start with pretrained ProteinMPNN.')

    # Dynamics aware parameters
    parser.add_argument('--use_dynamics', default=0, type=int)
    parser.add_argument('--flex_loss_coeff', default=0.5, type=float)
    parser.add_argument('--get_gt_flex_onthefly', default=0, type=int, help='Flag to get ground truth flexibility on-the-fly (with subsequent caching)')
    parser.add_argument('--init_flex_features', default=1, type=int, help="Set to 0 if no flexibility information should be passed on input to the node features h_V")
    parser.add_argument('--loss_fn', default='MSE', type=str, help= 'Define what loss to use. Choose MSE, L1 or DPO.')
    parser.add_argument('--grad_normalization', default=0, type=int, help="Set to 0 if the gradients of the seq and flex losses should not be normalized.")
    parser.add_argument('--test_engineering', default=0, type=int, help="In this main.py should be set to 0 to not overwrite the training dataset.")
    
    args = parser.parse_args()
    return args




def load_callbacks(args):
    callbacks = []
    
    logdir = str(os.path.join(args.res_dir, args.ex_name))
    
    ckptdir = os.path.join(logdir, "checkpoints")
    
    callbacks.append(BackupCodeCallback(os.path.dirname(args.res_dir),logdir, ignore_patterns=ignore_patterns('results*', 'pdb*', 'metadata*', 'vq_dataset*')))
    

    metric = "recovery"
    sv_filename = 'best-{epoch:02d}-{recovery:.3f}'
    callbacks.append(plc.ModelCheckpoint(
        monitor=metric,
        filename=sv_filename,
        save_top_k=15,
        mode='max',
        save_last=True,
        dirpath = ckptdir,
        verbose = True,
        every_n_epochs = args.check_val_every_n_epoch,
    ))

    
    now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
    cfgdir = os.path.join(logdir, "configs")
    callbacks.append(
        SetupCallback(
                now = now,
                logdir = logdir,
                ckptdir = ckptdir,
                cfgdir = cfgdir,
                config = args.__dict__,
                argv_content = sys.argv + ["gpus: {}".format(args.gpus)],)
    )
    
    
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval=None))
    return callbacks



if __name__ == "__main__":
    
    args = create_parser()
    if args.stage == 'predict':
        args.batch_size = 1
        print('In the predict stage, defaulting batch size to 1.')


    if (len(args.infer_path) > 0 or args.dataset=='PDBInference') and (len(args.infer_path) == 0 or args.dataset!='PDBInference'):
        raise ValueError("You should only use --infer_path with --dataset 'PDBInference' and vice versa.")

    pl.seed_everything(args.seed)

    data_module = DInterface(**vars(args))

    data_module.setup(stage=args.stage) #here is the cache_data called

    gpu_count = args.gpus #torch.cuda.device_count()
    if args.stage == 'fit':
        args.steps_per_epoch = math.ceil(len(data_module.trainset)/args.batch_size/gpu_count)
        print(f"steps_per_epoch {args.steps_per_epoch},  gpu_count {gpu_count}, batch_size{args.batch_size}")
        

    model = MInterface(**vars(args))

    trainer_config = {
        'devices': args.gpus,
        'max_epochs': args.epoch,
        'num_nodes': 1,
        "strategy": 'ddp',
        "precision": '32',
        'accelerator': 'gpu',
        'callbacks': load_callbacks(args),
        'logger': plog.WandbLogger(
                    project = 'ICLR2025',
                    name=args.ex_name,
                    save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                    offline = args.offline,
                    id = "_".join(args.ex_name.split("/")),
                    entity = "koubic"),
        'val_check_interval': args.val_check_interval,
        'check_val_every_n_epoch': args.check_val_every_n_epoch
    }

    trainer = Trainer(**trainer_config)

    if args.stage =='fit':
        trainer.fit(model, data_module)
    elif args.stage == 'test':
        test_out = trainer.test(model,data_module)
    elif args.stage == 'eval':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir='./cache_dir/')
        
        predictions = trainer.predict(model, data_module) #Just 1015 proteins loaded since they are required to be shorther than 500 residues
        out_dict = {}
        for pred in tqdm.tqdm(predictions):
            logprobs = pred['log_probs']
            for i in range(args.eval_sequences_sampled):

                if args.eval_sequences_temperature > 0 or args.eval_sequences_sampled > 1:
                    raise NotImplementedError('Sampling with temperature is not implemented yet.') #TODO!!!
                else:
                    AA_indices = logprobs.argmax(dim=-1, keepdim=False)
                    
                    decoded_seqs = tokenizer.batch_decode(AA_indices,skip_special_tokens=True)

                    for title, seq, mask in zip(pred['title'],decoded_seqs, pred['mask']):
                        _seq = [letter for letter,cond in zip(seq.split(' '),mask) if cond]
                        seq_cat = ''.join(_seq)
                        out_dict[title] = seq_cat
        with open(os.path.join(args.eval_output_dir,'inverse_folded_ATLAS.fasta'), 'w') as f:
            for pdb_name, seq in out_dict.items(): 
                f.write(f">{pdb_name}\n{seq}\n")

    elif args.stage == 'predict':
        predictions = trainer.predict(model, data_module)
        predictions_cuda = []
        for pred in predictions:
            _pred_cuda = {}
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    _pred_cuda[k] = v.to(torch.device('cuda'))
            _pred_cuda['batch'] = {k: pred['batch'][k].to(torch.device('cuda')) for k in ('X', 'S', 'mask', 'chain_M', 'chain_M_pos', 'residue_idx', 'chain_encoding_all')}
            predictions_cuda.append(_pred_cuda)

        seriazable_predictions = []
        for pred in predictions:
            ser_pred = {}
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    ser_pred[k] = v.cpu().numpy().tolist()
                else:
                    ser_pred[k] = v
            seriazable_predictions.append(ser_pred)

        import json
        with open(f'{args.infer_path}/predictions.json', 'w') as f:
            json.dump(seriazable_predictions, f)    
