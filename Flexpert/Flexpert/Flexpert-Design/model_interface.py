import sys; sys.path.append('/huyuqi/xmyu/DiffSDS')
import inspect
import torch
from src.tools.utils import cuda
import torch.nn as nn
import os
from torcheval.metrics.text import Perplexity
from src.interface.model_interface import MInterface_base
import math
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.tools.utils import load_yaml_config
import torchmetrics
 
class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.use_dynamics = kwargs.get('use_dynamics', 0)
        self.flex_loss_coeff = torch.Tensor([kwargs.get('flex_loss_coeff', 0)]).to('cuda:0').to(torch.float)
        self.flex_loss_coeff.requires_grad = False
        if self.use_dynamics:
            self.load_flex_predictor()
            self.flex_loss_type = kwargs.get('loss_fn', 0)
            if self.flex_loss_type == 'MSE':
                self.flex_loss_fn = nn.MSELoss(reduction='none')
            elif self.flex_loss_type == 'L1':
                self.flex_loss_fn = nn.L1Loss(reduction='none')
            elif self.flex_loss_type == 'DPO':
                self.flex_loss_fn = ...
            else:
                raise ValueError(f"Not recognized type of loss function {self.flex_loss_type}")
        self.cross_entropy = nn.NLLLoss(reduction='none')
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)

        self.control_sum_recovery = 0
        self.control_sum_batch_sizes = 0

        self.grad_normalization = kwargs.get('grad_normalization', 0)
        self.use_pmpnn_checkpoint = kwargs.get('use_pmpnn_checkpoint',0)
        
        if self.use_pmpnn_checkpoint:
            print('Loading pmpnn checkpoint from {}'.format(self.model.pmpnn_init_weights_path))
            state_dict = torch.load(self.model.pmpnn_init_weights_path)['state_dict'] #['module']
            state_dict = {key: value for key, value in state_dict.items() if 'model.' in key[:6]}
            state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
            self.model.load_state_dict(state_dict)

        self.MSE = nn.MSELoss(reduction='none')
        self.automatic_optimization = False

        if self.hparams.use_dynamics:
            self.pearson = torchmetrics.PearsonCorrCoef()
            self.spearman = torchmetrics.SpearmanCorrCoef()
            self.validation_step_outputs = []
            self.test_step_outputs = []

        #### setting forward hook

        # def forward_hook(module, input, output):
        #     def check_nan(tensor):
        #         if isinstance(tensor, torch.Tensor):
        #             if torch.isnan(tensor).any():
        #                 print(f"NaN detected in the output of {type(module).__name__}")
        #                 print(f"Tensor shape: {tensor.shape}")
        #                 print(f"Tensor stats: mean={tensor.mean()}, std={tensor.std()}, min={tensor.min()}, max={tensor.max()}, all={torch.isnan(tensor).all()}")
        #         elif isinstance(tensor, tuple):
        #             for i, t in enumerate(tensor):
        #                 if isinstance(t, torch.Tensor):
        #                     if torch.isnan(t).any():
        #                         print(f"NaN detected in the output[{i}] of {type(module).__name__}")
        #                         print(f"Tensor shape: {t.shape}")
        #                         print(f"Tensor stats: mean={t.mean()}, std={t.std()}, min={t.min()}, max={t.max()}, all={torch.isnan(tensor).all()}")

        #     if isinstance(output, tuple):
        #         for i, out in enumerate(output):
        #             check_nan(out)
        #     else:
        #         check_nan(output)

        # for name, module in self.model.named_modules():
        #     module.register_forward_hook(forward_hook)
        
        # for name, module in self.flex_model.named_modules():
        #     module.register_forward_hook(forward_hook)

        ####

    def forward(self, batch, mode='train', temperature=1.0):
        if self.hparams.augment_eps>0:
            batch['X'] = batch['X'] + self.hparams.augment_eps * torch.randn_like(batch['X'])

        batch = self.model._get_features(batch)
        results = self.model(batch)
        
        log_probs, mask = results['log_probs'], batch['mask']
        if len(log_probs.shape) == 3:
            if self.hparams.use_dynamics:
                loss = self.combined_flex_aware_loss(batch, pred_log_probs=log_probs)
                #loss = loss_dict['combined_loss']
            else:
                loss = self.cross_entropy(log_probs.permute(0,2,1), batch['S'])
                loss = (loss*mask).sum()/(mask.sum())
        elif len(log_probs.shape) == 2:
            if self.hparams.model_name == 'GVP':
                loss = self.cross_entropy(log_probs, batch.seq)
            else:
                loss = self.cross_entropy(log_probs, batch['S'])
            
            if self.hparams.model_name == 'AlphaDesign':
                loss += self.cross_entropy(results['log_probs0'], batch['S'])
            loss = (loss*mask).sum()/(mask.sum())
        
        cmp = log_probs.argmax(dim=-1)==batch['S']
        recovery = (cmp*mask).sum()/(mask.sum())

        if mode == 'predict':
            return {'original_sequence':batch['S'],'correct_positions': cmp, 'mask':mask,'loss':loss, 'recovery':recovery, 'title':batch['title'], 'log_probs': log_probs, 'batch':batch} #, 'gt_bfactors': batch['norm_bfactors'], 'batch':batch}
        elif mode == 'eval':
            return {'original_sequence':batch['S'],'correct_positions': cmp, 'mask':mask,'loss':loss, 'recovery':recovery, 'title':batch['title'], 'log_probs': log_probs, 'batch':batch}
        else:
            return loss, recovery

    def avgCorrelations(self, preds, gts, masks):
        pearson_R = 0
        spearman_R = 0
        valid_datapoints = 0
        for pred, gt, mask in zip(preds, gts, masks):
            dpR = self.pearson(pred[torch.where(mask)], gt[torch.where(mask)])
            if torch.isnan(dpR):
                continue
            else:
                pearson_R += dpR
                spearman_R += self.spearman(pred[torch.where(mask)], gt[torch.where(mask)])
                valid_datapoints += 1
        return pearson_R/valid_datapoints, spearman_R/valid_datapoints

    def temperature_schedular(self, batch_idx):
        total_steps = self.hparams.steps_per_epoch*self.hparams.epoch
        
        initial_lr = 1.0
        circle_steps = total_steps//100
        x = batch_idx / total_steps
        threshold = 0.48
        if x<threshold:
            linear_decay = 1 - 2*x
        else:
            K = 1 - 2*threshold
            linear_decay = K - K*(x-threshold)/(1-threshold)
        
        new_lr = (1+math.cos(batch_idx/circle_steps*math.pi))/2*linear_decay*initial_lr

        return new_lr
    
    # def get_grad_norm(self):
    #     total_norm = 0
    #     parameters = [p for p in self.parameters() if p.grad is not None and p.requires_grad]
    #     for p in parameters:
    #         param_norm = p.grad.detach().data.norm(2)
    #         total_norm += param_norm.item() ** 2
    #     total_norm = total_norm ** 0.5                
    #     return total_norm

    #https://lightning.ai/docs/pytorch/1.9.0/notebooks/lightning_examples/basic-gan.html
    def training_step(self, batch, batch_idx, **kwargs):
        if self.use_dynamics:
            raw_loss, recovery = self(batch)
            if type(raw_loss) == dict:
                flex_loss = raw_loss['flex_loss']
                seq_loss = raw_loss['seq_loss']
                opt = self.optimizers()
                opt.zero_grad()
                
                _params_for_optimization = [p for p in self.model.parameters() if p.requires_grad]
                _params_for_optimization += [p for p in self.flex_model.parameters() if p.requires_grad]
                
                grads_flex = torch.autograd.grad(flex_loss, _params_for_optimization, create_graph=True)
                grads_seq = torch.autograd.grad(seq_loss, _params_for_optimization, create_graph=True)
                if self.grad_normalization:
                    norm_grads_flex = [g / (g.norm() + 1e-10) for g in grads_flex]
                    norm_grads_seq = [g / (g.norm() + 1e-10) for g in grads_seq]
                else:
                    norm_grads_flex = grads_flex
                    norm_grads_seq = grads_seq

                combined_grads = [self.flex_loss_coeff * gflex + (1-self.flex_loss_coeff) * gseq for gflex, gseq in zip(norm_grads_flex, norm_grads_seq)]
            
                #maybe track the angle between the gradients?
                self.log_dict({'flex_grad_norm':torch.mean(torch.tensor([g.detach().norm() for g in norm_grads_flex])), 'seq_grad_norm': torch.mean(torch.tensor([g.detach().norm() for g in norm_grads_seq])), 'combined_grad_norm': torch.mean(torch.tensor([g.detach().norm() for g in combined_grads]))}, on_step=True, on_epoch=False, prog_bar=True)
                

                for param, grad in zip(_params_for_optimization, combined_grads):
                    if param.grad is None:
                        param.grad = grad.detach()
                    else:
                        param.grad += grad.detach()

                self.clip_gradients(opt, gradient_clip_val=1., gradient_clip_algorithm="norm")
                opt.step()
                
                # Update learning rate
                sch = self.lr_schedulers()
                if sch is not None:
                    sch.step()
                
                loss = flex_loss + seq_loss

                self.log_dict({'train_flex_loss':flex_loss, 'train_seq_loss':seq_loss}, on_step=True, on_epoch=False, prog_bar=True)
                
                # Log the current learning rate
                if sch is not None:
                    current_lr = sch.get_last_lr()[0]
                    self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=True)
            else:
                loss = raw_loss
                self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        else:
            raw_loss, recovery = self(batch)
            if type(raw_loss) == dict:
                loss = raw_loss['combined_loss']
                _ = raw_loss.pop('pred_flex')
                # _ = raw_loss.pop('gt_bfactors')
                _ = raw_loss.pop('gt_flex')
                _ = raw_loss.pop('flex_mask')

                self.log_dict(raw_loss, on_step=True, on_epoch=True, prog_bar=True)
            else:
                loss = raw_loss
                self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
    
    def validation_step(self, batch, batch_idx):
        raw_loss, recovery = self(batch)
        if type(raw_loss) == dict:
            loss = raw_loss['flex_loss']+raw_loss['seq_loss'] #raw_loss['combined_loss']
            raw_loss['recovery'] = recovery
            pred_flex = raw_loss.pop('pred_flex')
            gt_flex = batch['gt_flex']

            flex_mask = raw_loss.pop('flex_mask')
            #epoch_metric_ingredients = {'pred_bfactors':pred_bfactors, 'gt_bfactors':gt_bfactors, 'flex_mask':flex_mask}
            epoch_metric_ingredients = {'pred_flex': pred_flex,'gt_flex':gt_flex, 'flex_mask':flex_mask}
            self.validation_step_outputs.append(epoch_metric_ingredients)
            self.log_dict({ "val_combined_loss":loss,
                            "val_seq_loss":raw_loss['seq_loss'],
                            "val_flex_loss":raw_loss['flex_loss'],
                            "recovery": recovery})
        else:
            loss = raw_loss
            self.log_dict({"val_loss":loss,
                        "recovery": recovery})
        #if there is issue with validation metrics - see the test_step below
        return self.log_dict

    def on_validation_epoch_end(self):
        if self.hparams.use_dynamics:
            # all_preds = [b['pred_bfactors'] for b in self.validation_step_outputs]
            # all_gts = [b['gt_bfactors'] for b in self.validation_step_outputs]
            all_preds = [b['pred_flex'] for b in self.validation_step_outputs]
            all_gts = [b['gt_flex'] for b in self.validation_step_outputs]
            all_masks = [b['flex_mask'] for b in self.validation_step_outputs]
            
            max_seq_length = max([pred.size()[1] for pred in all_preds])

            for set_of_tensors in [all_preds, all_gts, all_masks]:
                for i in range(len(set_of_tensors)):
                    set_of_tensors[i] = F.pad(set_of_tensors[i], (0, max_seq_length - set_of_tensors[i].shape[1],0,0), value=float(0))
            all_preds = torch.cat(all_preds, dim=0)
            all_gts = torch.cat(all_gts, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            # print(all_preds.shape, all_gts.shape, all_masks.shape)
            # do something with all preds
            
            # pearson_R = self.pearson(all_preds[torch.where(all_masks)], all_gts[torch.where(all_masks)])
            pearson_R, spearman_R = self.avgCorrelations(all_preds, all_gts, all_masks)
            # try:
            #     spearman_R = self.spearman(all_preds[torch.where(all_masks)], all_gts[torch.where(all_masks)])
            # except IndexError:
            #     spearman_R = pearson_R
            self.log_dict({"val_pearson_R":pearson_R, "val_spearman_R":spearman_R})
            self.validation_step_outputs.clear()  # free memory
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        import pickle            #use pickle to save the self.test_step_outputs to a file
        with open(f'rebuttal_experiments/test_step_outputs_{self.hparams.starting_checkpoint_path.split("/")[-3]}_initFF{self.hparams.init_flex_features}_{self.hparams.test_eng_data_path.split("/")[-1][:-5]}.pkl', 'wb') as f:
            pickle.dump(self.test_step_outputs, f)
        if self.hparams.test_engineering and self.hparams.use_dynamics:
            all_preds = [b['pred_flex'] for b in self.test_step_outputs]
            all_eng_gts = [b['gt_flex'] for b in self.test_step_outputs]
            all_masks = [b['flex_mask'] for b in self.test_step_outputs]
            all_eng_masks = [b['eng_mask'] for b in self.test_step_outputs]
            all_original_gt_flex = [b['original_gt_flex'] for b in self.test_step_outputs]

            avg_sequence_recovery = sum([b['sequence_recovery'] for b in self.test_step_outputs]) / len(self.test_step_outputs)
            avg_sequence_recovery = avg_sequence_recovery.cpu().tolist()
            max_seq_length = max([pred.size()[1] for pred in all_preds])
            
            _pred_flex_pool = []
            _eng_gt_flex_pool = []
            _original_gt_flex_pool = []
            _original_gt_flex_ranks_pool = []
            _eng_gt_flex_ranks_pool = []
            _pred_flex_ranks_pool = []


            import numpy as np
            for eng_mask, flex_mask, original_gt_flex, eng_gt_flex, pred_flex in zip(all_eng_masks, all_masks, all_original_gt_flex, all_eng_gts, all_preds):
                #select only the values where the engineering mask is 1 and flex mask is 1
                _original_gt_flex = original_gt_flex[eng_mask == 1]
                _eng_gt_flex = eng_gt_flex[eng_mask == 1]
                _pred_flex = pred_flex[eng_mask == 1]
                _pred_flex_pool.append(_pred_flex.cpu().numpy())
                _eng_gt_flex_pool.append(_eng_gt_flex.cpu().numpy())
                _original_gt_flex_pool.append(_original_gt_flex.cpu().numpy())
                
                _original_gt_flex_ranks = torch.argsort(torch.argsort(torch.nan_to_num(original_gt_flex, nan=0)))[eng_mask == 1].cpu().numpy()
                _eng_gt_flex_ranks = torch.argsort(torch.argsort(torch.nan_to_num(eng_gt_flex, nan=0)))[eng_mask == 1].cpu().numpy()
                _pred_flex_ranks = torch.argsort(torch.argsort(torch.nan_to_num(pred_flex, nan=0)))[eng_mask == 1].cpu().numpy()
            
                _original_gt_flex_ranks_pool.append(_original_gt_flex_ranks)
                _eng_gt_flex_ranks_pool.append(_eng_gt_flex_ranks)
                _pred_flex_ranks_pool.append(_pred_flex_ranks)

            
            import matplotlib.pyplot as plt
            import os

            # # Create 'paper_figures' folder if it doesn't exist
            # if not os.path.exists('paper_figures'):
            #     os.makedirs('paper_figures')

            #pool the numpy arrays in the lists into one numpy array
            _pred_flex_pool = np.concatenate(_pred_flex_pool)
            _eng_gt_flex_pool = np.concatenate(_eng_gt_flex_pool)
            _original_gt_flex_pool = np.concatenate(_original_gt_flex_pool)
            
            ############################################################################
            all_gt_seqs = [b['gt_seq'] for b in self.test_step_outputs]
            all_pred_logprobs = [b['pred_logprobs'] for b in self.test_step_outputs]
            _gt_seq_pool = []
            _pred_seq_pool = []
            _outside_eng_region_pred_seq_pool = []
            _outside_eng_region_gt_seq_pool = []
            for eng_mask, gt_seq, pred_logprobs in zip(all_eng_masks, all_gt_seqs, all_pred_logprobs):
                #select only the values where the engineering mask is 1
                _outside_eng_region_pred_seq_pool.append(torch.argmax(pred_logprobs[(eng_mask == 0) & (flex_mask == 1)], dim=1).cpu().numpy())
                _outside_eng_region_gt_seq_pool.append(gt_seq[(eng_mask == 0) & (flex_mask == 1)].cpu().numpy())

                _pred_seq = torch.argmax(pred_logprobs[eng_mask == 1], dim=1)
                _gt_seq = gt_seq[eng_mask == 1]
                
                # create and add to the pools the numpy arrays
                _gt_seq_pool.append(_gt_seq.cpu().numpy())
                _pred_seq_pool.append(_pred_seq.cpu().numpy())
            _gt_seq_pool = np.concatenate(_gt_seq_pool)
            _pred_seq_pool = np.concatenate(_pred_seq_pool)
            _outside_eng_region_pred_seq_pool = np.concatenate(_outside_eng_region_pred_seq_pool)
            _outside_eng_region_gt_seq_pool = np.concatenate(_outside_eng_region_gt_seq_pool)
            #output these pools together with the other pools to a json_file
            import json
            with open(f'paper_figures/pools_{self.hparams.starting_checkpoint_path.split("/")[-3]}_initFF{self.hparams.init_flex_features}_{self.hparams.test_eng_data_path.split("/")[-1][:-5]}.json', 'w') as f:
                json.dump({
                    '_pred_flex_pool': _pred_flex_pool.tolist(),
                    '_eng_gt_flex_pool': _eng_gt_flex_pool.tolist(),
                    '_original_gt_flex_pool': _original_gt_flex_pool.tolist(),
                    '_pred_seq_pool': _pred_seq_pool.tolist(),
                    '_gt_seq_pool': _gt_seq_pool.tolist(),
                    '_sequence_recovery': avg_sequence_recovery,
                    '_outside_eng_region_pred_seq_pool': _outside_eng_region_pred_seq_pool.tolist(),
                    '_outside_eng_region_gt_seq_pool': _outside_eng_region_gt_seq_pool.tolist()
                }, f)



            ############################################################################


            self.test_step_outputs.clear()
        else:
            # all_preds = [b['pred_bfactors'] for b in self.test_step_outputs]
            # all_gts = [b['gt_bfactors'] for b in self.test_step_outputs]
            all_preds = [b['pred_flex'] for b in self.test_step_outputs]
            all_gts = [b['gt_flex'] for b in self.test_step_outputs]
            all_masks = [b['flex_mask'] for b in self.test_step_outputs]
            
            max_seq_length = max([pred.size()[1] for pred in all_preds])

            for set_of_tensors in [all_preds, all_gts, all_masks]:
                for i in range(len(set_of_tensors)):
                    set_of_tensors[i] = F.pad(set_of_tensors[i], (0, max_seq_length - set_of_tensors[i].shape[1],0,0), value=float(0))

            all_preds = torch.cat(all_preds, dim=0)
            all_gts = torch.cat(all_gts, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            # print(all_preds.shape, all_gts.shape, all_masks.shape)
            # do something with all preds
            # pearson_R = self.pearson(all_preds[torch.where(all_masks)], all_gts[torch.where(all_masks)])
            pearson_R, spearman_R = self.avgCorrelations(all_preds, all_gts, all_masks)
            try:
                spearman_R = self.spearman(all_preds[torch.where(all_masks)], all_gts[torch.where(all_masks)])
            except IndexError:
                spearman_R = pearson_R
            self.log_dict({"test_pearson_R":pearson_R, "test_spearman_R":spearman_R})
            self.test_step_outputs.clear()  # free memory
        return super().on_test_epoch_end()

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        #return self.validation_step(batch, batch_idx)
        
        raw_loss, recovery = self(batch)
        if type(raw_loss) == dict:
            #loss = raw_loss['combined_loss']
            loss = raw_loss['flex_loss']+raw_loss['seq_loss'] #raw_loss['combined_loss']
            raw_loss['recovery'] = recovery
            # pred_bfactors = raw_loss.pop('pred_bfactors')
            pred_flex = raw_loss.pop('pred_flex')
            # gt_bfactors = raw_loss.pop('gt_bfactors')
            gt_flex = raw_loss.pop('gt_flex')
            flex_mask = raw_loss.pop('flex_mask')
            epoch_metric_ingredients = {'pred_flex':pred_flex, 'gt_flex':gt_flex, 'flex_mask':flex_mask}

            if self.hparams.test_engineering and self.hparams.use_dynamics:
                eng_mask = raw_loss.pop('eng_mask')
                original_gt_flex = raw_loss.pop('original_gt_flex')
                epoch_metric_ingredients['eng_mask'] = eng_mask
                epoch_metric_ingredients['original_gt_flex'] = original_gt_flex
                epoch_metric_ingredients['gt_seq'] = raw_loss['gt_seq']
                epoch_metric_ingredients['pred_logprobs'] = raw_loss['pred_logprobs']
                epoch_metric_ingredients['sequence_recovery'] = raw_loss['recovery']
                epoch_metric_ingredients['id'] = batch['title']

            self.test_step_outputs.append(epoch_metric_ingredients)
            out_dict = {"val_combined_loss":loss,
                        "val_seq_loss":raw_loss['seq_loss'],
                        "val_flex_loss":raw_loss['flex_loss'],
                        "recovery": recovery}
        else:
            out_dict = {"val_loss":raw_loss, "recovery": recovery}
        self.log_dict(out_dict,on_step=True,on_epoch=True, sync_dist=True)
        #print(out_dict) #This print statement is fixing it - ultimately fixed by setting 'n_step=True' above
        #Below validation of the correctness of the above loging
        self.control_sum_batch_sizes += len(batch['X'])
        self.control_sum_recovery += len(batch['X'])*recovery
        return out_dict

    def predict_step(self, batch, batch_idx):
        predict_out = self(batch, mode=self.hparams.stage)
        return predict_out
    
    def combined_flex_aware_loss(self, batch, pred_log_probs):

        _mask = batch['mask']
    
        gt_seq = batch['S']
        gt_flex = batch['gt_flex']
        anm_input = batch['enm_vals'] #TODO: manage the loading of the anm input

        trail_idcs = torch.argmax((batch['S'] == 0).int(), dim=1)
        trail_idcs[trail_idcs == 0] = batch['S'].shape[1]  # For sequences without padding

        # # #TODO: test on one example - remove later
        # # trail_idcs = trail_idcs[0].unsqueeze(0)

        # # # ###########################################################################
        # # # #### TODO: change back to precomputed GT_FLEX once debugged ###############
        # dl_gtseq = batch['S']
        # dl_anm = batch['enm_vals']


        # attention_mask = torch.zeros_like(batch['mask'])
        # for i in range(attention_mask.size(0)):
        #     attention_mask[i, :trail_idcs[i]] = 1

        # dl_predflex_bs4 = self.flex_model(None, dl_anm, trail_idcs, attention_mask = attention_mask, sampled_pmpnn_sequence = dl_gtseq, alphabet='pmpnn') #['predicted_flex'][:,:-1,0]
        # dl_predflex_bs1 = self.flex_model(None, dl_anm[0].unsqueeze(0), trail_idcs[0].unsqueeze(0) , attention_mask = attention_mask[0].unsqueeze(0), sampled_pmpnn_sequence = dl_gtseq[0].unsqueeze(0), alphabet='pmpnn') #['predicted_flex'][:,:-1,0]
        
        # testseq = 'MKKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS'
        # tokenizer_predflex_bs4 = self.flex_model(None, dl_anm[0,:90].unsqueeze(0), trail_idcs[0].unsqueeze(0) , attention_mask = attention_mask[0,:90].unsqueeze(0), sampled_pmpnn_sequence = testseq, alphabet='aa') #['predicted_flex'][:,:-1,0] #['predicted_flex'][:,:-1,0]
        # import pdb; pdb.set_trace()
        # input_ids_predflex_bs4 = self.flex_model(dl_gtseq, dl_anm, trail_idcs, attention_mask = attention_mask, sampled_pmpnn_sequence = None, alphabet='aa') #['predicted_flex'][:,:-1,0]
        # gt_flex = batch['gt_flex']
        # # ####
        # import pdb; pdb.set_trace() #check the mask and the gt_flex vs. onthefly computed gt_flex
        # #TODO: here fix the mask for the prottrans and clean this,
        # #      the mask should have all 1s where there is sequence or eos token

        # attention_mask = ...
        # if self.hparams.get_gt_flex_onthefly:
            
        #     cache_keys = list(batch['title'])

        #     # Check if all cache_keys are in self.gt_flex_cache
        #     all_keys_in_cache = all(cache_key in self.model.gt_flex_cache for cache_key in cache_keys)
            
        #     if not all_keys_in_cache:
        #         gt_flex = self.flex_model(None, anm_input, trail_idcs, attention_mask=attention_mask, sampled_pmpnn_sequence=gt_seq, alphabet='pmpnn')['predicted_flex'][:,:-1,0]
        #         for key, val in zip(cache_keys, gt_flex):
        #             #TODO: iteruje to spravne?
        #             self.model.gt_flex_cache[key] = val
        #     else:
        #         retrieved_gt_flexs = []
        #         for key in cache_keys:
        #             _gt_flex = self.model.gt_flex_cache[key]
        #             retrieved_gt_flexs.append(_gt_flex)
        #         gt_flex = torch.cat(retrieved_gt_flexs, dim=0) #TODO: concat spravne?
        # else:
        #     raise NotImplementedError('The precomputed data were not realiable.')
        #     gt_flex = batch['gt_flex']
        # ###########################################################################


        attention_mask = torch.zeros_like(batch['mask'])
        for i in range(attention_mask.size(0)):
            attention_mask[i, :trail_idcs[i]] = 1

        #Original sequence loss
        seq_loss = self.cross_entropy(pred_log_probs.permute(0,2,1), gt_seq)
        seq_loss = (seq_loss*_mask).sum()/(_mask.sum())
        #New Dynamics-aware loss
        flex_model_input = pred_log_probs.permute(0,2,1)
        pred_flex = self.flex_model(flex_model_input, anm_input, trail_idcs, attention_mask=attention_mask)['predicted_flex'][:,:-1,0]
        #check here that the loss function is working properly (with the masking and all)
        # import pdb; pdb.set_trace()
        _filter_nans_mask = ~torch.isnan(pred_flex) & ~torch.isnan(gt_flex)
        flex_loss = self.flex_loss_fn(pred_flex[_filter_nans_mask]*_mask[_filter_nans_mask], gt_flex[_filter_nans_mask]*_mask[_filter_nans_mask])
        _flex_mask = _mask*_filter_nans_mask
        _flex_mask = _flex_mask.int()
        flex_loss = flex_loss.sum()/_flex_mask.sum()
        
        retVal ={'seq_loss':seq_loss, 'flex_loss':flex_loss, 'pred_flex':pred_flex, 'flex_mask':_flex_mask, 'gt_flex':gt_flex}
        if self.hparams.test_engineering and self.hparams.use_dynamics:
            retVal['eng_mask'] = batch['eng_mask']
            retVal['original_gt_flex'] = batch['original_gt_flex']
            retVal['gt_seq'] = batch['S']
            retVal['pred_logprobs'] = pred_log_probs
        return retVal


    def configure_loss(self):
        def loss_function(pred_angle, angles, pred_seq, seqs, seq_loss_mask, angle_loss_mask):
            angle_loss = self.MSE(torch.cat([angles[...,:1],torch.sin(angles[...,1:3]), torch.cos(angles[...,1:3])],dim=-1),
            torch.cat([pred_angle[...,:1],torch.sin(pred_angle[...,1:3]), torch.cos(pred_angle[...,1:3])],dim=-1))
            
            angle_loss = angle_loss[angle_loss_mask].sum(dim=-1).mean()
            logits = pred_seq.permute(0,2,1)
            seq_loss = self.cross_entropy(logits, seqs)
            seq_loss = seq_loss[seq_loss_mask].mean()

            metric=Perplexity()
            metric.update(pred_seq[seq_loss_mask][None,...].cpu(), seqs[seq_loss_mask][None,...].cpu())
            perp = metric.compute()
            
            return {"angle_loss": angle_loss, "seq_loss": seq_loss, "perp":perp}

        self.loss_function = loss_function
        
    def load_model(self):
        params = OmegaConf.load(f'configs/{self.hparams.model_name}.yaml')
        params.update(self.hparams)

        if self.hparams.model_name == 'GraphTrans':
            from src.models.graphtrans_model import GraphTrans_Model
            self.model = GraphTrans_Model(params)
        
        if self.hparams.model_name == 'StructGNN':
            from src.models.structgnn_model import StructGNN_Model
            self.model = StructGNN_Model(params)
            
        if self.hparams.model_name == 'GVP':
            from src.models.gvp_model import GVP_Model
            self.model = GVP_Model(params)

        if self.hparams.model_name == 'GCA':
            from src.models.gca_model import GCA_Model
            self.model = GCA_Model(params)

        if self.hparams.model_name == 'AlphaDesign':
            from src.models.alphadesign_model import AlphaDesign_Model
            self.model = AlphaDesign_Model(params)

        if self.hparams.model_name == 'ProteinMPNN':
            from src.models.proteinmpnn_model import ProteinMPNN_Model
            self.model = ProteinMPNN_Model(params)

        if self.hparams.model_name == 'ESMIF':
            pass

        if self.hparams.model_name == 'PiFold':
            from src.models.pifold_model import PiFold_Model
            self.model = PiFold_Model(params)

        if self.hparams.model_name == 'KWDesign':
            from src.models.kwdesign_model import KWDesign_model#Design_Model
            self.model = KWDesign_model(params) #Design_Model(params) - this required to significantly change the constructor of Design_Model
        
        if self.hparams.model_name == 'E3PiFold':
            from src.models.E3PiFold_model import E3PiFold
            self.model = E3PiFold(params)
    
    def load_flex_predictor(self):
        from src.models.anm_prottrans import ANMAwareFlexibilityProtTrans
        flex_params = load_yaml_config(f'configs/ANMAwareFlexibilityProtTrans.yaml')
        # flex_params_dict = OmegaConf.to_container(flex_params, resolve=True)
        self.flex_model = ANMAwareFlexibilityProtTrans(**flex_params)
        
        # consider turning on the gradients for debug purposes
        self.flex_model.eval()
        for params in self.flex_model.parameters():
            params.requires_grad = False
        
        #also pass it to proteinmpnn:
        # self.model.flex_model = self.flex_model


    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)