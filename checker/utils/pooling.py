import torch
import torch.nn as nn

from transformers import AutoModel

class MeanMaxPoolingModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.pooling = MeanMaxPooling(hidden_size, hidden_size)  # class pooling hôm trước
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        pooled = self.pooling(hidden_states, mask=attention_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


