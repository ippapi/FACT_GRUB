import torch
import torch.nn as nn

from transformers import AutoModel

class MeanMaxPooling(nn.Module):
    def __init__(self, hidden_dim, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim
        self.proj = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, hidden_states, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(-1) 
            hidden_states = hidden_states * mask

            sum_hidden = hidden_states.sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            mean_pooled = sum_hidden / lengths
        else:
            mean_pooled = hidden_states.mean(dim=1)

        if mask is not None:
            hidden_states = hidden_states + (mask.eq(0) * -1e9)
        max_pooled, _ = hidden_states.max(dim=1)

        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
        return self.proj(pooled)

class MeanMaxPoolingModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.pooling = MeanMaxPooling(hidden_size, hidden_size)  
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        pooled = self.pooling(hidden_states, mask=attention_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


