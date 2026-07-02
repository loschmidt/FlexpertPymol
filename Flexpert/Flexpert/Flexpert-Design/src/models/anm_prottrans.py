#import dependencies
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader

import re
import numpy as np
import pandas as pd
import copy
import pdb

import transformers, datasets
from transformers.modeling_outputs import TokenClassifierOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import T5EncoderModel, T5Tokenizer
from transformers import TrainingArguments, Trainer, set_seed
from safetensors import safe_open

#DataCollator
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from evaluate import load
from datasets import Dataset

from tqdm import tqdm
import random

from scipy import stats
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from Bio import SeqIO
from io import StringIO
import requests
import tempfile

from sklearn.model_selection import train_test_split
import csv


#### UTILS

class LoRAConfig:
    def __init__(self):
        self.lora_rank = 4
        self.lora_init_scale = 0.01
        self.lora_modules = ".*SelfAttention|.*EncDecAttention"
        self.lora_layers = "q|k|v|o"
        self.trainable_param_names = ".*layer_norm.*|.*lora_[ab].*"
        self.lora_scaling_rank = 1
        # lora_modules and lora_layers are speicified with regular expressions
        # see https://www.w3schools.com/python/python_regex.asp for reference
        
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, scaling_rank, init_scale):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling_rank = scaling_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        if self.rank > 0:
            self.lora_a = nn.Parameter(torch.randn(rank, linear_layer.in_features) * init_scale)
            if init_scale < 0:
                self.lora_b = nn.Parameter(torch.randn(linear_layer.out_features, rank) * init_scale)
            else:
                self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        if self.scaling_rank:
            self.multi_lora_a = nn.Parameter(
                torch.ones(self.scaling_rank, linear_layer.in_features)
                + torch.randn(self.scaling_rank, linear_layer.in_features) * init_scale
            )
            if init_scale < 0:
                self.multi_lora_b = nn.Parameter(
                    torch.ones(linear_layer.out_features, self.scaling_rank)
                    + torch.randn(linear_layer.out_features, self.scaling_rank) * init_scale
                )
            else:
                self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features, self.scaling_rank))

    def forward(self, input):
        if self.scaling_rank == 1 and self.rank == 0:
            # parsimonious implementation for ia3 and lora scaling
            if self.multi_lora_a.requires_grad:
                hidden = F.linear((input * self.multi_lora_a.flatten()), self.weight, self.bias)
            else:
                hidden = F.linear(input, self.weight, self.bias)
            if self.multi_lora_b.requires_grad:
                hidden = hidden * self.multi_lora_b.flatten()
            return hidden
        else:
            # general implementation for lora (adding and scaling)
            weight = self.weight
            if self.scaling_rank:
                weight = weight * torch.matmul(self.multi_lora_b, self.multi_lora_a) / self.scaling_rank
            if self.rank:
                weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
            return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, rank={}, scaling_rank={}".format(
            self.in_features, self.out_features, self.bias is not None, self.rank, self.scaling_rank
        )


def modify_with_lora(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        LoRALinear(layer, config.lora_rank, config.lora_scaling_rank, config.lora_init_scale),
                    )
    return transformer

class ClassConfig:
    def __init__(self, dropout=0.2, num_labels=1, add_pearson_loss=False, add_sse_loss=False, adaptor_architecture = None , enm_embed_dim = 512, enm_att_heads = 8, kernel_size = 3, num_layers = 2, **kwargs):
        self.dropout_rate = dropout
        self.num_labels = num_labels
        self.add_pearson_loss = add_pearson_loss
        self.add_sse_loss = add_sse_loss
        self.adaptor_architecture = adaptor_architecture
        self.enm_embed_dim = enm_embed_dim
        self.enm_att_heads = enm_att_heads
        self.kernel_size = kernel_size
        self.num_layers = num_layers

class ENMAdaptedAttentionClassifier(nn.Module):
    def __init__(self, seq_embedding_dim, out_dim, enm_embed_dim, num_att_heads):
        super(ENMAdaptedAttentionClassifier, self).__init__()
        self.embedding = nn.Linear(1, enm_embed_dim)
        self.enm_attention = nn.MultiheadAttention(enm_embed_dim, num_att_heads)
        self.layer_norm = nn.LayerNorm(enm_embed_dim)
        self.enm_adaptor = nn.Linear(enm_embed_dim, seq_embedding_dim)
        self.adapted_classifier = nn.Linear(2*seq_embedding_dim, out_dim)
    
    def forward(self, seq_embedding, enm_input):
        enm_input = enm_input.transpose(0, 1)  # Transpose to shape (N, B, E) for MultiheadAttention
        enm_input = enm_input.unsqueeze(-1)  # Add a dimension for the embedding
        enm_input_embedded = self.embedding(enm_input)
        enm_att, _ = self.enm_attention(enm_input_embedded, enm_input_embedded, enm_input_embedded)
        enm_att = enm_att.transpose(0, 1)  # Transpose back to shape (B, N, E)
        enm_att = self.layer_norm(enm_att + enm_input.transpose(0, 1))
        enm_embedding = self.enm_adaptor(enm_att)
        combined_embedding = torch.cat((seq_embedding, enm_embedding), dim=-1)
        logits = self.adapted_classifier(combined_embedding)
        return logits
    
class ENMAdaptedConvClassifier(nn.Module):
    def __init__(self, seq_embedding_dim, out_dim, kernel_size, enm_embedding_dim, num_layers):
        super(ENMAdaptedConvClassifier, self).__init__()
        layers = []
        self.conv1 = nn.Conv1d(1, enm_embedding_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        layers.append(self.conv1)
        layers.append(nn.ReLU())
        for i in range(num_layers-1):
            layers.append(nn.Conv1d(enm_embedding_dim, enm_embedding_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2))
            layers.append(nn.ReLU())
        self.conv_net = nn.Sequential(*layers)
        self.adapted_classifier = nn.Linear(seq_embedding_dim+1, out_dim)

    def forward(self, seq_embedding, enm_input, attention_mask=None):
        enm_input = torch.nan_to_num(enm_input, nan=0.0)
        enm_input = enm_input.unsqueeze(1)
        conv_out = self.conv_net(enm_input)
        enm_embedding = conv_out.transpose(1,2)
        
        if attention_mask is not None:
            # Use attention_mask to ignore padded elements
            mask = attention_mask.unsqueeze(-1).float()
            enm_embedding = enm_embedding * mask
            # Compute mean over non-padded elements
            
            enm_embedding = enm_embedding.mean(dim=-1).unsqueeze(-1)
            # enm_embedding = enm_embedding.sum(dim=2)/ mask.sum(dim=2).clamp(min=1e-9)
        else:
            raise ValueError('We actually want to provide the mask.')
            enm_embedding = torch.mean(enm_embedding, dim=1)
            
        # enm_embedding = enm_embedding.unsqueeze(1).expand(-1, seq_embedding.size(1), -1)
        combined_embedding = torch.cat((seq_embedding, enm_embedding), dim=-1)
        logits = self.adapted_classifier(combined_embedding)
        return logits
    

        
class ENMAdaptedDirectClassifier(nn.Module):
    def __init__(self, seq_embedding_dim, out_dim):
        super(ENMAdaptedDirectClassifier, self).__init__()
        self.adapted_classifier = nn.Linear(seq_embedding_dim+1, out_dim)

    def forward(self, seq_embedding, enm_input):
            enm_input = enm_input.unsqueeze(-1)
            combined_embedding = torch.cat((seq_embedding, enm_input), dim=-1)
            logits = self.adapted_classifier(combined_embedding)
            return logits

class ENMNoAdaptorClassifier(nn.Module):
    def __init__(self, seq_embedding_dim, out_dim):
        super(ENMNoAdaptorClassifier, self).__init__()
        self.adapted_classifier = nn.Linear(seq_embedding_dim, out_dim)

    def forward(self, seq_embedding, enm_input):
            _ = enm_input #ignoring enm_input
            logits = self.adapted_classifier(seq_embedding)
            return logits


class T5EncoderForTokenClassification(T5PreTrainedModel):

    def __init__(self, config: T5Config, class_config):
        super().__init__(config)
        self.num_labels = class_config.num_labels
        self.config = config
        self.add_pearson_loss = class_config.add_pearson_loss
        self.add_sse_loss = class_config.add_sse_loss
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        # self.encoder = CustomT5Stack(encoder_config, self.shared)

        # import pdb; pdb.set_trace()
        original_embedding = self.encoder.embed_tokens
        in_dim, out_dim = tuple(original_embedding.weight.shape)
        self.new_embedding = nn.Linear(in_dim, out_dim, bias=False).to('cuda:0') #TODO: pass the correct weights!!! And careful! the embedding layer and the linear layer are maybe mutually "transposed"
        print("Initialized new_embedding layer - without weights yet!")
        # self.new_embedding.weight = nn.Parameter(original_embedding.weight.T)

        # self.weight = original_embedding.weight
        # self.weight = nn.Parameter(self.new_embedding.weight.T)
        # self.encoder.forward = new_forward.__get__(self.encoder, self.encoder.__class__)

        self.dropout = nn.Dropout(class_config.dropout_rate)
        if class_config.adaptor_architecture == 'attention':
            self.classifier = ENMAdaptedAttentionClassifier(config.hidden_size, class_config.num_labels, class_config.enm_embed_dim, class_config.enm_att_heads) #nn.Linear(config.hidden_size, class_config.num_labels)
        elif class_config.adaptor_architecture == 'direct':
            self.classifier = ENMAdaptedDirectClassifier(config.hidden_size, class_config.num_labels)
        elif class_config.adaptor_architecture == 'conv':
            self.classifier = ENMAdaptedConvClassifier(config.hidden_size, class_config.num_labels, class_config.kernel_size, class_config.enm_embed_dim, class_config.num_layers)
        elif class_config.adaptor_architecture == 'no-adaptor':
            self.classifier = ENMNoAdaptorClassifier(config.hidden_size, class_config.num_labels)
        else:
            raise ValueError('Only attention, direct, conv and no-adaptor architectures are supported for the adaptor.')


        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        enm_vals = None,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is not None:
            outputs = self.encoder(input_ids=None,attention_mask=attention_mask,inputs_embeds=inputs_embeds,head_mask=head_mask,output_attentions=output_attentions,output_hidden_states=output_hidden_states,return_dict=return_dict,)
        elif input_ids is not None:
            outputs = self.encoder(input_ids=input_ids,attention_mask=attention_mask,inputs_embeds=None,head_mask=head_mask,output_attentions=output_attentions,output_hidden_states=output_hidden_states,return_dict=return_dict,)
        sequence_output = outputs[0]
        # import pdb; pdb.set_trace() #TODO: CHECK EVERYTHING IS IN EVAL MODE and the dropout below is OFF
        sequence_output = self.dropout(sequence_output)
        #TODO: check the enm_vals are padded properly and check that the sequence limit (in the transformer) is indeed 512
        # logits = self.classifier(sequence_output, enm_vals)
        
        logits = self.classifier(sequence_output, enm_vals, attention_mask)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            #loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ENMAdaptedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        #enm_vals = inputs.get("enm_vals")
        
        outputs = model(**inputs)
        logits = outputs.get('logits')
        mask = inputs.get('attention_mask')
        loss_fct = MSELoss()

        active_loss = mask.view(-1) == 1
        active_logits = logits.view(-1)
        active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(-100).type_as(labels))
        valid_logits=active_logits[active_labels!=-100]
        valid_labels=active_labels[active_labels!=-100]

        loss = loss_fct(valid_labels, valid_logits)
        return (loss, outputs) if return_outputs else loss
    


def PT5_classification_model(half_precision, class_config):
    # Load PT5 and tokenizer
    # possible to load the half preciion model (thanks to @pawel-rezo for pointing that out)
    if not half_precision:
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    elif half_precision and torch.cuda.is_available() : 
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16).to(torch.device('cuda'))
    else:
          raise ValueError('Half precision can be run on GPU only.')
    

    
    # Create new Classifier model with PT5 dimensions
    class_model=T5EncoderForTokenClassification(model.config,class_config)
    
    # Set encoder and embedding weights to checkpoint weights
    class_model.shared=model.shared
    class_model.encoder=model.encoder    
    
    # Delete the checkpoint model
    model=class_model
    del class_model
    
    # Print number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("ProtT5_Classfier\nTrainable Parameter: "+ str(params))    
 
    # Add model modification lora
    config = LoRAConfig()
    
    # Add LoRA layers
    model = modify_with_lora(model, config)
    
    # Freeze Embeddings and Encoder (except LoRA)
    for (param_name, param) in model.shared.named_parameters():
                param.requires_grad = False
    for (param_name, param) in model.encoder.named_parameters():
                param.requires_grad = False       

    for (param_name, param) in model.named_parameters():
            if re.fullmatch(config.trainable_param_names, param_name):
                param.requires_grad = True

    # Print trainable Parameter          
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("ProtT5_LoRA_Classfier\nTrainable Parameter: "+ str(params) + "\n")
    
    return model, tokenizer


@dataclass
class DataCollatorForTokenRegression(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name and k!= 'enm_vals'} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch['enm_vals'] = torch.nn.utils.rnn.pad_sequence([torch.tensor(feature['enm_vals'], dtype=torch.float) for feature in features], batch_first=True, padding_value=0.0)
#batch = self.tokenizer.pad(no_labels_features,padding=self.padding,max_length=self.max_length,pad_to_multiple_of=self.pad_to_multiple_of,return_tensors="pt")
        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels

            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.float)
        return batch

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()

#### END OF UTILS

def do_topology_split(df, split_path):
    import json
    with open(split_path, 'r') as f:
        splits = json.load(f)
    
    #split the dataframe according to the splits
    train_df = df[df['name'].isin(splits['train'])]
    valid_df = df[df['name'].isin(splits['validation'])]
    test_df = df[df['name'].isin(splits['test'])]
    return train_df, valid_df, test_df
    

class ANMAwareFlexibilityProtTrans(nn.Module):
    def __init__(self, gumbel_temperature, **kwargs):
        super(ANMAwareFlexibilityProtTrans, self).__init__()
        
        model, tokenizer = self.load_finetuned_model(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda')
        self.model.to(self.device)
        self.model.eval()
        self.gumbel_temperature = gumbel_temperature
        self.logit_transform = nn.functional.gumbel_softmax #Use the Straight Through Gumbel SoftMax - in forward process it does argmax, 

        # in the backward process it approximates the gradient of argmax by the gradient of the Gumbel Softmax
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html set hard=True to do the Straight-Through trick

        self.conversion_tensor = self.construct_pmpnn_t5_conversion_tensor()
        

    def construct_pmpnn_t5_conversion_tensor(self):
        """
        Creates tensor which takes the onehot encodings in the proteinmpnn vocabulary and maps them to ProtTrans vocabulary.
        """
        _one_hots = []
        for idx in [[0,1], 2, [3, 29, 30, 31, 32], 5, 4, 6, 7, 8, 10, 9, 13, 11, 12, 14, 15, 18, 16, 17, 19, 20, 21, 22, 23, 24, 25, 28, 26, 27]:
            if isinstance(idx, int):
                _oh = F.one_hot(torch.tensor([idx]), 33)
            else:
                _sohs = []
                for subidx in idx:
                    _soh = F.one_hot(torch.tensor([subidx]), 33)
                    _sohs.append(_soh)
                _oh = torch.sum(torch.stack(_sohs), dim=0)
            _one_hots.append(_oh)
        #_one_hots = [F.one_hot(torch.tensor([idx]), 33)[0] if isinstance(idx, int) else torch.sum(torch.stack([F.one_hot(torch.tensor([subidx]), 33)[0] for subidx in idx]), dim=0) for idx in [[0,1], 2, [3, 29, 30, 31, 32], 5, 4, 6, 7, 8, 10, 9, 13, 11, 12, 14, 15, 18, 16, 17, 19, 20, 21, 22, 23, 24, 25, 28, 26, 27]]
        _one_hots.extend([torch.zeros((1,33)) for _ in range(100)])
        return torch.cat(_one_hots, dim=0).to(torch.device('cuda')).float()

    def load_finetuned_model(self, checkpoint_path, half_precision, **kwargs):#num_labels, add_pearson_loss, add_sse_loss, adaptor_architecture, enm_embed_dim, enm_att_heads, num_layers, kernel_size):
        class_config=ClassConfig(**kwargs) #um_labels=num_labels, add_pearson_loss=add_pearson_loss, add_sse_loss=add_sse_loss, adaptor_architecture = adaptor_architecture, enm_embed_dim = enm_embed_dim, enm_att_heads = enm_att_heads, num_layers = num_layers, kernel_size = kernel_size)
        model, tokenizer = PT5_classification_model(half_precision=half_precision, class_config=class_config) #.from_pretrained(args.model_path)

        # model.load_state_dict(torch.load(args.model_path))
        # try:
        #     with safe_open(f"{checkpoint_path}/model.safetensors", framework="pt", device="cuda:0") as f:
        #         state_dict = {}
        #         for key in f.keys():
        #             state_dict[key] = f.get_tensor(key)
        #         model.load_state_dict(state_dict, strict=False)
        # except:
        #     state_dict = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location='cuda:0')
        #     model.load_state_dict(state_dict, strict=False)
        state_dict = torch.load(checkpoint_path, map_location='cuda:0')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        original_embedding = model.encoder.embed_tokens
        model.new_embedding.weight = nn.Parameter(original_embedding.weight.T)
        print('Set the weights for the new embedding layer!')
        return model, tokenizer
    
    def translate_to_model_vocab(self, batch_one_hot, trail_idcs):
            # Pad the batch_one_hot tensor with zeros along the last dimension
            batch_one_hot = F.pad(batch_one_hot, (0, 1, 0, 0, 0, 0), 'constant', 0)
            
            #TODO: VERIFY THAT THE GRADIENTS ARE OK AFTER THE MASKED_SCATTER OPERATION
            # Create a mask for the '2' token
            mask = torch.zeros_like(batch_one_hot, dtype=torch.bool)
            for i, trail_idx in enumerate(trail_idcs):
                if trail_idx < batch_one_hot.size(2):  # Ensure index is within bounds
                    mask[i, :, trail_idx] = True
            
            # Create a tensor with '2' in the one-hot encoding
            token_2 = torch.zeros_like(batch_one_hot)
            token_2[:, 2, :] = 1  # Assuming '2' corresponds to index 2 in the one-hot encoding
            
            # Use masked_scatter_ to modify the tensor in-place while preserving gradients
            batch_one_hot.masked_scatter_(mask, token_2[mask])

            T5_translation = torch.einsum('ej,ijk->iek', self.conversion_tensor, batch_one_hot)
            T5_translation = T5_translation.permute(0,2,1)
            return T5_translation
    
    def forward(self, pmpnn_logits, anm_input, trail_idcs, attention_mask, sampled_pmpnn_sequence = None, alphabet = None): #batch example 32x33x395 (batch_size x ProteinMPNN vocab size x seq length)
    
        anm_input = F.pad(anm_input, (0, 1, 0, 0), 'constant', 0)
        attention_mask = F.pad(attention_mask, (0, 1, 0, 0), 'constant', 1)
        
        if sampled_pmpnn_sequence is None:
            if alphabet is None:
                batch_one_hot = self.logit_transform(pmpnn_logits, tau=self.gumbel_temperature, hard=True, dim=1)
                batch_token_ids = self.translate_to_model_vocab(batch_one_hot, trail_idcs)
                inputs = batch_token_ids #.to(torch.int)
            # elif alphabet == 'aa':
            #     batch_one_hot = ... #TODO one hot encode the pmpnn tokens
            #     batch_token_ids = self.translate_to_model_vocab(batch_one_hot, trail_idcs)
            #     input_ids = ... #TODO: argmax to get the tokens from the one hot encodings
            #     outputs = self.model(input_ids = input_ids, enm_vals=anm_input, attention_mask = attention_mask) #TODO?: pass the mask as well (take it from the batch, pad it for the end of sequence, convert to Tensor)
            #     predicted_flex = outputs.logits
            #     return {'predicted_flex': predicted_flex, 'enm_vals': anm_input, 'input_ids': input_ids}
                
        # elif alphabet is None:
        #     raise ValueError('need to specify what alphabet is used to encode sampled_pmpnn_sequence!')
        # elif alphabet is 'pmpnn':
        #     # Convert sampled_pmpnn_sequence to one-hot encoding
        #     batch_one_hot = F.one_hot(sampled_pmpnn_sequence, num_classes=33).float().permute(0,2,1)
        #     batch_token_ids = self.translate_to_model_vocab(batch_one_hot, trail_idcs)
        #     inputs = batch_token_ids
        # elif alphabet is 'pt5':
        #     inputs = F.one_hot(sampled_pmpnn_sequence, num_classes=128).float() #.permute(0,2,1)
        # elif alphabet is 'aa':
        #     ... #TODO apply tokenizer
        #     #tokens = self.tokenizer(" ".join(sampled_pmpnn_sequence))
        #     tokens = self.tokenizer(" ".join(sampled_pmpnn_sequence))
        #     input_ids = torch.tensor(tokens['input_ids']).cuda().unsqueeze(0)
            
        #     outputs = self.model(input_ids = input_ids, enm_vals=anm_input, attention_mask = attention_mask) #TODO?: pass the mask as well (take it from the batch, pad it for the end of sequence, convert to Tensor)
        #     predicted_flex = outputs.logits
        #     return {'predicted_flex': predicted_flex, 'enm_vals': anm_input, 'input_ids': input_ids}

        inputs_embeds = self.model.new_embedding(inputs) #TODO pass through embedding
        outputs = self.model(enm_vals=anm_input, inputs_embeds = inputs_embeds, attention_mask = attention_mask) #TODO?: pass the mask as well (take it from the batch, pad it for the end of sequence, convert to Tensor)
        #TODO: above it throws RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: 
        # Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
        
        predicted_flex = outputs.logits
        return {'predicted_flex': predicted_flex, 'enm_vals': anm_input, 'input_ids': inputs}