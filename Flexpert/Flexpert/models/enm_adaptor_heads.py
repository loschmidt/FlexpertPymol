import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # import pdb; pdb.set_trace()
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
        enm_input = enm_input.to(seq_embedding.device)
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

    def forward(self, seq_embedding, enm_input, attention_mask=None):
            _ = enm_input #ignoring enm_input
            logits = self.adapted_classifier(seq_embedding)
            return logits