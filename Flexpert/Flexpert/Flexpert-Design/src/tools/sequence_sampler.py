import torch
import torch.nn.functional as F
import torchmetrics
from src.tools.utils import load_yaml_config
from tqdm import tqdm

class SequenceSampler():
    def __init__(self, num_sequences, sampling_temperature, sampling_type = 'primitive', bfactor_recovery_metric = 'pearson_R', pMPNN_model = None) -> None:
        self.num_sequences = num_sequences
        self.temperature = sampling_temperature
        self.primitively_sampled = None
        self.hard_sampled = None
        self.bfactor_predictor = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pMPNN_model = pMPNN_model.to(self.device) if pMPNN_model else None
        

        if sampling_type == 'primitive':
            self.chosen_sampling = self.primitive_sampling
        elif sampling_type == 'pMPNN':
            if not pMPNN_model:
                raise ValueError('pMPNN model must be provided for pMPNN sampling')
            self.chosen_sampling = self.proteinMPNN_sampling
        else:
            raise NotImplementedError
        
        if bfactor_recovery_metric == 'pearson_R':
            self.bfactor_recovery_metric = torchmetrics.PearsonCorrCoef().to(self.device)
        else:
            raise NotImplementedError

        #For now, sample from the logits / softmax

        #TODO: implement ProteinMPNN sampling
    
    def softmax_with_temperature(self, logits):
        # Scale logits by the temperature
        scaled_logits = logits / self.temperature
        # Compute softmax
        probabilities = torch.softmax(scaled_logits, dim=-1)
        return probabilities

    def sample_from_logits(self, logits):
        probabilities = self.softmax_with_temperature(logits)
        # Sample from the categorical distribution based on the computed probabilities
        categorical = torch.distributions.Categorical(probabilities)
        return categorical.sample()

    def proteinMPNN_sampling(self, logits, batch = None):
        if not batch:
            raise ValueError('Batch featurized for ProteinMPNN must be provided for pMPNN sampling')
        
        with torch.no_grad():
            retVal = []
            for i in range(self.num_sequences):
                X, S, mask, chain_M, chain_M_pos, residue_idx, chain_encoding_all = batch['X'], batch['S'], batch['mask'], batch['chain_M'], batch['chain_M_pos'], batch['residue_idx'], batch['chain_encoding_all']
                randn = torch.randn(chain_M.shape, device=X.device)
                sampled_seq = self.pMPNN_model.sample(X=X, randn = randn, S_true = S, chain_mask = chain_M, chain_M_pos = chain_M_pos , chain_encoding_all = chain_encoding_all, residue_idx = residue_idx, mask=mask, temperature=self.temperature)
                # X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0
                retVal.append(sampled_seq['S'])
            return retVal

    def primitive_sampling(self, logits, batch = None): #leads to poor sequences ignoring the context of already decoded AAs
        retVal = []
        for i in range(self.num_sequences):
            retVal.append(self.sample_from_logits(logits))
        return retVal

    def load_bfactor_predictor(self, config_path = './src/models/configs/FlexibilityProtTrans.yaml'):
        print('Loading model based on the configs:', config_path)
        # print('Setting precision to medium')
        # torch.set_float32_matmul_precision('medium')
        from src.models.prottrans import FlexibilityProtTrans
        flex_params = load_yaml_config(config_path)
        # flex_params_dict = OmegaConf.to_container(flex_params, resolve=True)
        self.bfactor_predictor = FlexibilityProtTrans(**flex_params)

    
    def hard_sampling(self, logits):
        retVal = torch.argmax(logits, dim=-1)
        return retVal
    
    def eval_oracle_recovery(self, gt_seq, logits, mask, batch = None):
        hard_recovery, oracle_recovery = None, 0
        sampled_seqs = self.chosen_sampling(logits, batch=batch)
        hard_seq = self.hard_sampling(logits)

        hard_cmp = hard_seq==gt_seq
        hard_recovery = (hard_cmp*mask).sum()/(mask.sum())

        sampled_recoveries = [hard_recovery]
        for seq in sampled_seqs:
            oracle_cmp = seq==gt_seq
            _sampled_recovery = (oracle_cmp*mask).sum()/(mask.sum())
            sampled_recoveries.append(_sampled_recovery)

        oracle_recovery = max(sampled_recoveries)

        return hard_recovery, oracle_recovery
    
    def eval_bfactor_profile_recovery(self, gt_bfactors, gt_seq, logits, mask, batch = None):
        if not self.bfactor_predictor:
            self.load_bfactor_predictor()
        

        seq_recovery_by_bfactor_profile = 0

        sampled_seqs = self.chosen_sampling(logits, batch=batch)
        hard_seq = self.hard_sampling(logits)
        sampled_seqs.append(hard_seq)

        bfactor_recoveries = []
        bfact_to_seq = {}
        for seq in tqdm(sampled_seqs):
            ### TODO adapt below
            #New Dynamics-aware loss
            one_hot_seq = F.one_hot(seq, num_classes=33)
            flex_model_input = one_hot_seq.permute(0, 2, 1).float().to(self.device)
            pred_bfactors = self.bfactor_predictor(flex_model_input)['predicted_normalized_bfactors'][:,:-1,0]

            _filter_nans_mask = ~torch.isnan(pred_bfactors) #torch.where(~torch.isnan(flex_loss))
            _flex_mask = mask*_filter_nans_mask
            _flex_mask = _flex_mask.int()
            bfactor_recovery = self.bfactor_recovery_metric(pred_bfactors[torch.where(_flex_mask)], gt_bfactors[torch.where(_flex_mask)])
            bfact_to_seq[bfactor_recovery] = seq
            bfactor_recoveries.append(bfactor_recovery)

        seq_selected_by_bfact = bfact_to_seq[max(bfactor_recoveries)]
        # import pdb; pdb.set_trace()
        _cmp = seq_selected_by_bfact==gt_seq
        seq_recovery_by_bfactor_profile = (_cmp*mask).sum()/(mask.sum())

        return seq_recovery_by_bfactor_profile

    def eval_multiple_predictions_oracle(self, predictions):
        hard_recovery, oracle_recovery = [], []
        print('Evaluating sequence recovery by oracle...')
        for pred in tqdm(predictions):
            _hr, _or = self.eval_oracle_recovery(logits = pred['log_probs'], mask = pred['mask'], gt_seq = pred['original_sequence'], batch=pred['batch'])
            hard_recovery.append(_hr)
            oracle_recovery.append(_or)
        return hard_recovery, oracle_recovery
    

class SamplerGrid():
    def __init__(self, sample_sizes: list, sampling_temperatures: list, sampling_type = 'primitive', pMPNN_model = None) -> None:
        self.samplers = [SequenceSampler(s,t, sampling_type, pMPNN_model=pMPNN_model) for t in sampling_temperatures for s in sample_sizes]
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def get_optimal_sampler(self, predictions):
        oracle_recoveries = []
        for sampler in self.samplers:
            hard_recovery, oracle_recovery = sampler.eval_multiple_predictions_oracle(predictions) #here
            avg_hard_recovery = sum(hard_recovery)/len(hard_recovery)
            avg_oracle_recovery = sum(oracle_recovery)/len(oracle_recovery)
            oracle_recoveries.append(avg_oracle_recovery)
            print(f'T = {sampler.temperature}, Sample_size = {sampler.num_sequences}, Average hard recovery: {avg_hard_recovery}, average oracle recovery: {avg_oracle_recovery}')
    
        best_sampler = self.samplers[oracle_recoveries.index(max(oracle_recoveries))]
        print(f'Best sampler: T = {best_sampler.temperature}, Sample_size = {best_sampler.num_sequences}, with oracle recovery: {max(oracle_recoveries)}')
        return best_sampler
    
    def eval_bfactor_selection(self, predictions, sampler = None):
        if not sampler:
            sampler = self.get_optimal_sampler(predictions)
        recoveries = []
        print('Evaluating sequence recovery by bfactor profile...')
        for pred in tqdm(predictions):
            seq_recovery_by_bfactor_profile = sampler.eval_bfactor_profile_recovery(logits = pred['log_probs'], mask = pred['mask'], gt_seq = pred['original_sequence'], gt_bfactors = pred['gt_bfactors'], batch = pred['batch']) #
            recoveries.append(seq_recovery_by_bfactor_profile)
        avg_recovery = sum(recoveries)/len(recoveries)
        print(f'Average sequence recovery by bfactor profile: {avg_recovery}')
        return avg_recovery