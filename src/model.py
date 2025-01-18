import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from torchcrf import CRF
import yaml

class TokenClassificationModel(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        self.num_labels = cfg['model']['num_labels']
        self.use_crf = cfg['advanced'].get('use_crf', False)
        base_model = cfg['model']['base_model']
        
        # Load base transformer
        model_config = AutoConfig.from_pretrained(base_model)
        model_config.num_labels = self.num_labels
        self.backbone = AutoModel.from_pretrained(base_model, config=model_config)
        freeze_backbone = cfg['model'].get('freeze_backbone', False)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        # Classification head
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(cfg['advanced'].get('crf_dropout', 0.1))
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        
        # Optional CRF
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)
        else:
            self.crf = None
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Ensure the first timestep in the attention mask is all True
        attention_mask[:, 0] = 1
        # Optional: check label device
        if labels is not None:
            # labels = labels.to(device)
            # Safety checks if you suspect out-of-bound labels
            assert labels.max() < self.num_labels or labels.max() == -100, \
                f"Label index out of range: max={labels.max()} >= num_labels={self.num_labels}"
        
        # 1. Pass through base model
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        
        # 2. Apply dropout and get logits
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch_size, seq_length, num_labels)
        
        # 3. CRF branch
        if self.crf is not None:
            if labels is not None:
                # Create a mask that excludes positions where labels == -100 or attention_mask == 0
                valid_positions = (labels != -100) & attention_mask.bool()

                # Replace -100 labels with a safe index (e.g., 0 for "O")
                labels_for_crf = labels.clone()
                labels_for_crf[labels_for_crf == -100] = 0
                # Ensure the first timestep is always unmasked for CRF
                valid_positions[:, 0] = True

                # CRF loss is negative log-likelihood, so we take the negative
                loss = -self.crf(logits, labels_for_crf, mask=valid_positions)
                
                # We also decode using the same mask
                # => returns a list of list of predicted labels per sequence
                decoded_tags = self.crf.decode(logits, mask=valid_positions)
                
                # Return the "logits" as the decoded tags from the CRF
                return {"loss": loss, "logits": decoded_tags}
            else:
                # Inference mode
                # We consider the entire attention_mask
                decoded_tags = self.crf.decode(logits, mask=attention_mask.bool())
                return {"logits": decoded_tags}
        
        # 4. Standard cross-entropy path
        else:
            loss = None
            if labels is not None:
                # Standard token-level cross-entropy with ignore_index = -100
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

                # Flatten the tokens for cross-entropy
                # We'll use attention_mask to ignore padding, plus ignore_index=-100
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                
                loss = loss_fct(active_logits, active_labels)
            
            return {"loss": loss, "logits": logits}
