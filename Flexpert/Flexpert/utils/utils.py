import torch
import os
import json
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers.data.data_collator import DataCollatorMixin
from evaluate import load
from transformers import Trainer, set_seed
from torch.nn import MSELoss
from dataclasses import dataclass
from typing import Union, Optional
from datasets import Dataset
import numpy as np
import random
from tqdm import tqdm

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
        if 'enm_vals' in features[0]:
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


class ClassConfig:
    def __init__(self, dropout=0.2, num_labels=1, add_pearson_loss=False, add_sse_loss=False, adaptor_architecture = None , enm_embed_dim = 512, enm_att_heads = 8, kernel_size = 3, num_layers = 2):
        self.dropout_rate = dropout
        self.num_labels = num_labels
        self.add_pearson_loss = add_pearson_loss
        self.add_sse_loss = add_sse_loss
        self.adaptor_architecture = adaptor_architecture
        self.enm_embed_dim = enm_embed_dim
        self.enm_att_heads = enm_att_heads
        self.kernel_size = kernel_size
        self.num_layers = num_layers

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

# Set random seeds for reproducibility of your trainings run
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)

# Dataset creation
def create_dataset(tokenizer,seqs,labels, enm_vals, names=None):
    tokenized = tokenizer(seqs, max_length=1024, padding=False, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    # we need to cut of labels after 1023 positions for the data collator to add the correct padding (1023 + 1 special tokens)
    labels = [l[:1023] for l in labels]
    enm_vals = [enm[:1023] for enm in enm_vals] #pad the enm values with 0.0 to account for the special token

    for enm in enm_vals:
        if len(enm) == 1023:
            enm.append(0.0)

    dataset = dataset.add_column("labels", labels)
    dataset = dataset.add_column("enm_vals", enm_vals)
    if names:
        dataset = dataset.add_column("name", names)
    return dataset


def do_topology_split(df, split_path):
    
    with open(split_path, 'r') as f:
        splits = json.load(f)
    #split the dataframe according to the splits
    train_df = df[df['name'].isin(splits['train'])]
    valid_df = df[df['name'].isin(splits['validation'])]
    test_df = df[df['name'].isin(splits['test'])]
    return train_df, valid_df, test_df

def save_finetuned_model(model, target_folder):
    # Saves all parameters that were changed during finetuning
    filepath = os.path.join(target_folder, "final_model")
    model.save_pretrained(filepath, safe_serialization=False)
    print(f"Final model saved to {filepath}")


def update_config(config, args):
    # Update config with any non-None command-line arguments
    for arg in vars(args):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)
    return config

class ClassConfig:
    def __init__(self, config):
        # Set class attributes based on the loaded YAML config
        for key, value in config.items():
            setattr(self, key, value)

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation - used by the HuggingFace Trainer
    """
    predictions, labels = eval_pred
    predictions=predictions.flatten()
    labels=labels.flatten()

    valid_labels=labels[np.where((labels != -100 ) & (labels < 900 ))]
    valid_predictions=predictions[np.where((labels != -100 ) & (labels < 900 ))]
    #assuming the ENM vals are subtracted from the labels for correct evaluation
    spearman = load("spearmanr")
    pearson = load("pearsonr")
    mse = load("mse")
    return {"spearmanr": spearman.compute(predictions=valid_predictions, references=valid_labels)['spearmanr'],
            "pearsonr": pearson.compute(predictions=valid_predictions, references=valid_labels)['pearsonr'],
            "mse": mse.compute(predictions=valid_predictions, references=valid_labels)['mse']}

### ESM-2

# based on transformers DataCollatorForTokenClassification
@dataclass
class DataCollatorForTokenRegression_esm(DataCollatorMixin):
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
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name and k != 'enm_vals'} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Handle ENM values padding
        if "enm_vals" in features[0]:
            max_length = batch["input_ids"].shape[1]
            padded_enm_vals = []
            for feature in features:
                enm = feature['enm_vals']
                if len(enm) < max_length:
                    enm = enm + [0.0] * (max_length - len(enm))  # pad with zeros
                elif len(enm) > max_length:
                    enm = enm[:max_length]  # truncate
                padded_enm_vals.append(enm)
            batch["enm_vals"] = torch.tensor(padded_enm_vals, dtype=torch.float)

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right": 
            # changed to pad the special tokens at the beginning and end of the sequence
            batch[label_name] = [
                [self.label_pad_token_id] + to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label) - 1) 
                for label in labels
            ]                
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) 
                for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.float)
        return batch

#load ESM2 models
def load_esm_model_regression(checkpoint, half_precision, full = False, deepspeed=True):
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    if half_precision and deepspeed:
        model = EsmForTokenRegression.from_pretrained(checkpoint, num_labels = 1, torch_dtype = torch.float16)
    else:
        model = EsmForTokenRegression.from_pretrained(checkpoint, num_labels = 1)
        
    if full == True:
        return model, tokenizer 
        
    peft_config = LoraConfig(
        r=4, lora_alpha=1, bias="all", target_modules=["query","key","value","dense"]
    )
    
    model = inject_adapter_in_model(peft_config, model) 
    
    # Unfreeze the prediction head
    for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True 
    
    return model, tokenizer


def split_batch(batch, sub_batch_size):
    """Splits a batch dictionary into smaller sub-batches."""
    batch_size = batch["input_ids"].size(0)
    for i in range(0, batch_size, sub_batch_size):
        yield {key: tensor[i:i + sub_batch_size] for key, tensor in batch.items()}

def process_batch_in_chunks(model, batch, sub_batch_size):
    outputs = []
    for sub_batch in tqdm(split_batch(batch, sub_batch_size)):
        with torch.no_grad():  # To save memory
            output = model(**sub_batch)
            outputs.append(output)
    return outputs

def process_in_batches_and_combine(model, batch, sub_batch_size):
    outputs = process_batch_in_chunks(model, batch, sub_batch_size)
    # Example: combine logits if that's what you need
    logits = torch.cat([out.logits for out in outputs], dim=0)
    return logits

def get_dot_separated_name(key, _dict):
    if '_' in _dict[key]:
        dot_separated_name = '.'.join(_dict[key].split('_'))
    elif '.' in _dict[key]:
        dot_separated_name = _dict[key]
    else:
        raise ValueError("Sequence name must contain either an underscore or a dot to separate the PDB code and the chain code.")
    return dot_separated_name