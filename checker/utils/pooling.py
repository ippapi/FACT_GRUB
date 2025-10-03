import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5EncoderModel

class MeanMaxAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, out_dim=None, attn_dropout=0.1):
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, 1)
        self.proj = nn.Linear(3 * hidden_dim, out_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, hidden_states, mask=None):
        if mask is not None:
            mask_unsq = mask.unsqueeze(-1)
            hidden_states_masked = hidden_states * mask_unsq

            sum_hidden = hidden_states_masked.sum(dim=1)
            lengths = mask_unsq.sum(dim=1).clamp(min=1)
            mean_pooled = sum_hidden / lengths
        else:
            mean_pooled = hidden_states.mean(dim=1)

        if mask is not None:
            hidden_states_masked = hidden_states + (mask.unsqueeze(-1).eq(0) * -1e9)
        max_pooled, _ = hidden_states_masked.max(dim=1)

        scores = self.attn(hidden_states).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)

        pooled = torch.cat([mean_pooled, max_pooled, attn_pooled], dim=-1)
        return self.proj(pooled)

class MeanMaxPoolingModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.d_model
        self.pooling = MeanMaxPooling(hidden_size, hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        pooled = self.pooling(hidden_states, mask=attention_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

    def save(self, output_dir, tokenizer=None):
        self.encoder.save_pretrained(f"{output_dir}/encoder")
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        torch.save({
            "classifier": self.classifier.state_dict(),
            "pooling": self.pooling.state_dict()
        }, f"{output_dir}/head_pooling.bin")

    @classmethod
    def load(cls, output_dir, model_name, num_labels, device="cpu"):
        encoder = T5EncoderModel.from_pretrained(f"{output_dir}/encoder")
        model = cls.__new__(cls)
        super(cls, model).__init__()
        model.encoder = encoder
        hidden_size = encoder.config.d_model
        model.pooling = MeanMaxPooling(hidden_size, hidden_size)
        model.classifier = nn.Linear(hidden_size, num_labels)

        checkpoint = torch.load(f"{output_dir}/head_pooling.bin", map_location=device)
        model.classifier.load_state_dict(checkpoint["classifier"])
        model.pooling.load_state_dict(checkpoint["pooling"])
        model.to(device)
        return model

