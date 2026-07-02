from transformers.models.esm.modeling_esm import EsmPreTrainedModel, EsmModel
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from torch.nn import MSELoss
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
import re
from utils.lora_utils import LoRAConfig, modify_with_lora
from models.enm_adaptor_heads import (
    ENMAdaptedAttentionClassifier, ENMAdaptedDirectClassifier, 
    ENMAdaptedConvClassifier, ENMNoAdaptorClassifier
)
from peft import LoraConfig, inject_adapter_in_model

class EsmForTokenRegression(EsmPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, class_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.add_pearson_loss = class_config.add_pearson_loss
        self.add_sse_loss = class_config.add_sse_loss

        self.esm = EsmModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if class_config.adaptor_architecture == 'attention':
            self.classifier = ENMAdaptedAttentionClassifier(
                config.hidden_size, 
                class_config.num_labels, 
                class_config.enm_embed_dim, 
                class_config.enm_att_heads
            )
        elif class_config.adaptor_architecture == 'direct':
            self.classifier = ENMAdaptedDirectClassifier(
                config.hidden_size, 
                class_config.num_labels
            )
        elif class_config.adaptor_architecture == 'conv':
            self.classifier = ENMAdaptedConvClassifier(
                config.hidden_size, 
                class_config.num_labels, 
                class_config.kernel_size, 
                class_config.enm_embed_dim, 
                class_config.num_layers
            )
        elif class_config.adaptor_architecture == 'no-adaptor':
            self.classifier = ENMNoAdaptorClassifier(
                config.hidden_size, 
                class_config.num_labels
            )
        else:
            raise ValueError('Only attention, direct, conv and no-adaptor architectures are supported.')

        self.init_weights()

    def forward(
        self,
        enm_vals=None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        logits = self.classifier(sequence_output, enm_vals, attention_mask)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def ESM_classification_model(half_precision, class_config, lora_config):
    # Load ESM and tokenizer
    if not half_precision:
        model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
    elif half_precision and torch.cuda.is_available():
        model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D", torch_dtype=torch.float16).to(torch.device('cuda'))
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
    else:
        raise ValueError('Half precision can be run on GPU only.')
    
    # Create new Classifier model with ESM dimensions
    class_model = EsmForTokenRegression(model.config, class_config)
    
    # Set encoder weights to checkpoint weights
    class_model.esm = model
    
    # Delete the checkpoint model
    del model
    
    # Print number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, class_model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("ESM_Classifier\nTrainable Parameter: " + str(params))
    
    # Add model modification lora
    esm_lora_peft_config = LoraConfig(
        r=4, lora_alpha=1, bias="all", target_modules=["query","key","value","dense"]
    )
    
    # Add LoRA layers
    class_model.esm = inject_adapter_in_model(esm_lora_peft_config, class_model.esm) 
    
    # Freeze Encoder (except LoRA)
    for (param_name, param) in class_model.esm.named_parameters():
        param.requires_grad = False
    
    for (param_name, param) in class_model.esm.named_parameters():
        if re.fullmatch(".*lora.*", param_name): #".*layer_norm.*|.*lora_[ab].*"
            param.requires_grad = True
        if re.fullmatch(".*layer_norm.*", param_name): #".*layer_norm.*|.*lora_[ab].*"
            param.requires_grad = True
    # Print trainable Parameter
    model_parameters = filter(lambda p: p.requires_grad, class_model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("ESM_LoRA_Classifier\nTrainable Parameter: " + str(params) + "\n")
    
    return class_model, tokenizer