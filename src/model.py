import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from torchcrf import CRF
import yaml
from torch.nn import CrossEntropyLoss
from src.losses.focal_loss import FocalLoss
from src.losses.label_smoothing import LabelSmoothingCrossEntropy

def configure_loss_function(cfg, crf=None):
    """
    Configure the loss function based on the configuration.
    
    Args:
        cfg: Configuration dictionary.
            - Expected to contain loss function configuration under 'training.losses'.
        crf: Optional CRF layer for calculating CRF loss.
    
    Returns:
        A loss function instance or a combination of loss functions.

    Dimensionality:
        For label_smoothing:
            inputs: Predicted logits (batch_size * sequence_length, num_classes)
            targets: Ground truth labels (batch_size * sequence_length)
    """
    loss_function = cfg['training']['losses'].get('loss_function', 'cross_entropy')
    if loss_function == 'compound_loss':
        # Compound loss configuration
        cross_entropy_weight = cfg['training']['losses']['compound_loss'].get('cross_entropy_weight', 0.5)
        crf_weight = cfg['training']['losses']['compound_loss'].get('crf_weight', 0.3)
        label_smoothing_weight = cfg['training']['losses']['compound_loss'].get('label_smoothing_weight', 0.2)

        cross_entropy_loss = CrossEntropyLoss(ignore_index=-100)
        label_smoothing_loss = LabelSmoothingCrossEntropy(smoothing=cfg['training']['losses']['label_smoothing']['smoothing'])

        def compound_loss_fn(logits, labels, valid_positions):
            # Cross-Entropy Loss
            active_loss = valid_positions.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            ce_loss = cross_entropy_loss(active_logits, active_labels)

            # CRF Loss
            crf_loss = -crf(logits, labels, mask=valid_positions) if crf else 0

            # Label Smoothing Loss
            ls_loss = label_smoothing_loss(active_logits, active_labels)

            # Weighted sum of losses
            total_loss = (cross_entropy_weight * ce_loss) + (crf_weight * crf_loss) + (label_smoothing_weight * ls_loss)
            return total_loss

        return compound_loss_fn
    elif loss_function == 'focal_loss':
        return FocalLoss(alpha=cfg['training']['losses']['focal_loss']['alpha'],
                         gamma=cfg['training']['losses']['focal_loss']['gamma'])
    elif loss_function == 'label_smoothing':
        return LabelSmoothingCrossEntropy(smoothing=cfg['training']['losses']['label_smoothing']['smoothing'])
    else:
        return CrossEntropyLoss(ignore_index=-100)


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

        # Configure loss function using the modularized function
        self.loss_fct = configure_loss_function(cfg, self.crf)

    def forward(self, input_ids, attention_mask, labels=None):
        # Ensure the first timestep in the attention mask is all True
        attention_mask[:, 0] = 1
        # Optional: check label device
        if labels is not None:
            # Safety checks if you suspect out-of-bound labels
            assert labels.max() < self.num_labels or labels.max() == -100, \
                f"Label index out of range: max={labels.max()} >= num_labels={self.num_labels}"
        
        # 1. Pass through base model
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        
        # 2. Apply dropout and get logits
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch_size, seq_length, num_labels)
        
        # Calculate loss and decode tags if labels are provided
        loss = None
        decoded_tags = None
        if labels is not None:
            # Create a mask that excludes positions where labels == -100 or attention_mask == 0
            valid_positions = (labels != -100) & attention_mask.bool()

            # Replace -100 labels with a safe index (e.g., 0 for "O")
            labels_for_crf = labels.clone()
            labels_for_crf[labels_for_crf == -100] = 0
            # Ensure the first timestep is always unmasked for CRF
            valid_positions[:, 0] = True

            # Compute loss using the configured loss function
            loss = self.loss_fct(logits, labels_for_crf, valid_positions)

            # Decode using the same mask if CRF is used
            if self.crf is not None:
                decoded_tags = self.crf.decode(logits, mask=valid_positions)
        else:
            # Inference mode
            if self.crf is not None:
                decoded_tags = self.crf.decode(logits, mask=attention_mask.bool())

        return {"loss": loss, "logits": decoded_tags if decoded_tags is not None else logits}
