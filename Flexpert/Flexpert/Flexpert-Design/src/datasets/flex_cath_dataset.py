import os
import json
import numpy as np
from tqdm import tqdm
import random
import torch.utils.data as data
from .utils import cached_property
from transformers import AutoTokenizer
from src.tools.utils import load_yaml_config

class FlexCATHDataset(data.Dataset):
    def __init__(self, path='./',  split='train', max_length=500, test_name='All', data = None, removeTS=0, version=4.3, data_jsonl_name='/chain_set.jsonl', use_dynamics=True):
        self.version = version
        self.path = path
        self.mode = split
        self.max_length = max_length
        self.test_name = test_name
        self.removeTS = removeTS
        self.data_jsonl_name = data_jsonl_name

        self.using_dynamics = use_dynamics

        print(self.data_jsonl_name)
        if self.removeTS:
            self.remove = json.load(open(self.path+'/remove.json', 'r'))['remove']
        
        if data is None:
            if split == 'predict':
                _split = 'valid'
                print('In predict mode for CATH4.3 using VALIDATION split as the data. Consider switching to TEST set.')
            else:
                _split = split
            self.data = self.cache_data[_split]
        else:
            self.data = data
        
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="./cache_dir/")
    
    @cached_property
    def cache_data(self):
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])
        print("path is: ", self.path)
        if not os.path.exists(self.path):
            raise "no such file:{} !!!".format(self.path)
        else:
            with open(self.path+'/'+self.data_jsonl_name) as f:
                lines = f.readlines()
            data_list = []
            for line in tqdm(lines):
                entry = json.loads(line)
                if self.removeTS and entry['name'] in self.remove:
                    continue
                seq = entry['seq']

                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)
                
                bad_chars = set([s for s in seq]).difference(alphabet_set)

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
                            'chain_encoding': 1*chain_mask
                        })
                        if self.using_dynamics:
                            data_list[-1]['gt_flex'] = entry['gt_flex']
                            data_list[-1]['enm_vals'] = entry['enm_vals']
                            if 'original_gt_flex' in entry:
                                data_list[-1]['original_gt_flex'] = entry['original_gt_flex']
                            if 'eng_mask' in entry:
                                data_list[-1]['eng_mask'] = entry['eng_mask']
                # else:
                #     import pdb; pdb.set_trace()
                #     print("Bad chars found in sequence: ", bad_chars)
                        
            if self.version==4.2:
                with open(self.path+'/chain_set_splits.json') as f:
                    dataset_splits = json.load(f)
            
            if self.version==4.3:
                with open(self.path+'/chain_set_splits.json') as f:
                    dataset_splits = json.load(f)
                    # _dataset_splits = json.load(f)
                    # dataset_splits = {k: _dataset_splits['train'] for k,_ in _dataset_splits.items()}
                    # print("TODO: FIX THIS BACK!!!")
                    # import pdb; pdb.set_trace()
            
            if self.test_name == 'L100':
                with open(self.path+'/test_split_L100.json') as f:
                    test_splits = json.load(f)
                dataset_splits['test'] = test_splits['test']

            if self.test_name == 'sc':
                with open(self.path+'/test_split_sc.json') as f:
                    test_splits = json.load(f)
                dataset_splits['test'] = test_splits['test']
            
            name2set = {}
            name2set.update({name:'train' for name in dataset_splits['train']})
            name2set.update({name:'valid' for name in dataset_splits['validation']})
            name2set.update({name:'test' for name in dataset_splits['test']})

            data_dict = {'train':[],'valid':[],'test':[]}
            for data in data_list:
                if name2set.get(data['title']):
                    if name2set[data['title']] == 'train':
                        data_dict['train'].append(data)
                    
                    if name2set[data['title']] == 'valid':
                        data_dict['valid'].append(data)
                    
                    if name2set[data['title']] == 'test':
                        data['category'] = 'Unkown'
                        data['score'] = 100.0
                        data_dict['test'].append(data)
            return data_dict

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
            item['enm_vals'] = item['enm_vals'][truncate_index:truncate_index+self.max_length]
        return item