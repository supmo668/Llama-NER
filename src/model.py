import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from torchcrf import CRF
import yaml
from torch.nn import CrossEntropyLoss
from src.losses.focal_loss import FocalLoss
from src.losses.label_smoothing import LabelSmoothingCrossEntropy
from typing import Dict, Optional
from datasets import load_dataset


def configure_loss_function(cfg, crf=None):
    """
    Configure the loss function based on the configuration.
    
    Args:
        cfg: Configuration dictionary.
        crf: Optional CRF layer for calculating CRF loss.
    
    Returns:
        A loss function instance or a combination of loss functions.
    """
    loss_function = cfg['training']['losses'].get('loss_function', 'cross_entropy')

    if loss_function == 'compound_loss':
        cross_entropy_weight = cfg['training']['losses']['compound_loss'].get('cross_entropy_weight', 0.5)
        crf_weight = cfg['training']['losses']['compound_loss'].get('crf_weight', 0.3)
        label_smoothing_weight = cfg['training']['losses']['compound_loss'].get('label_smoothing_weight', 0.2)

        cross_entropy_loss = CrossEntropyLoss(ignore_index=-100)
        label_smoothing_loss = LabelSmoothingCrossEntropy(
            smoothing=cfg['training']['losses']['label_smoothing']['smoothing'])

        def compound_loss_fn(logits, labels, valid_positions):
            # logits: (batch_size, seq_length, num_labels)
            # labels: (batch_size, seq_length)
            # valid_positions: (batch_size, seq_length)

            active_loss = valid_positions.view(-1)  
            # bool shape: (batch_size*seq_length,)
            logits_flat = logits.view(-1, logits.size(-1))  
            # (batch_size*seq_length, num_labels)
            labels_flat = labels.view(-1)                  
            # (batch_size*seq_length,)

            active_logits, active_labels = logits_flat[active_loss], labels_flat[active_loss]
            ce_loss = cross_entropy_loss(active_logits, active_labels)

            # Compute CRF Loss if CRF is enabled
            crf_loss = -crf(logits, labels, mask=valid_positions) if crf else 0

            # Compute Label Smoothing Loss
            ls_loss = label_smoothing_loss(active_logits, active_labels)

            # Weighted Sum of Losses
            total_loss = (cross_entropy_weight * ce_loss) + (crf_weight * crf_loss) + (label_smoothing_weight * ls_loss)
            return total_loss

        return compound_loss_fn

    elif loss_function == 'focal_loss':
        return FocalLoss(alpha=cfg['training']['losses']['focal_loss']['alpha'],
                         gamma=cfg['training']['losses']['focal_loss']['gamma'])
    elif loss_function == 'label_smoothing':
        return LabelSmoothingCrossEntropy(
            smoothing=cfg['training']['losses']['label_smoothing']['smoothing'])
    else:
        return CrossEntropyLoss(ignore_index=-100)


class TokenClassificationModel(nn.Module):
    """
    Token classification model for Named Entity Recognition (NER).
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the token classification model.
        
        Args:
            config_path (str, optional): Path to the configuration file.
        """
        super().__init__()

        # Load config
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        # Load dataset metadata to get number of labels
        with open(cfg['data']['metadata_path'], 'r') as f:
            metadata = yaml.safe_load(f)
            self.num_labels = metadata['label_info']['ner_tags']['num_classes']

        self.use_crf = cfg['advanced'].get('use_crf', False)
        base_model = cfg['model']['base_model']

        # Load transformer backbone
        model_config = AutoConfig.from_pretrained(base_model)
        model_config.num_labels = self.num_labels
        self.backbone = AutoModel.from_pretrained(base_model, config=model_config)

        # Optionally freeze backbone
        if cfg['model'].get('freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classification head
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(cfg['advanced'].get('crf_dropout', 0.1))
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # Conditional CRF
        self.crf = CRF(self.num_labels, batch_first=True) if self.use_crf else None

        # Configure loss function
        self.loss_fn = configure_loss_function(cfg, self.crf)

    def forward(
        self,
        input_ids: torch.Tensor,  # Shape: (batch_size, seq_length)
        attention_mask: torch.Tensor,  # Shape: (batch_size, seq_length)
        labels: Optional[torch.Tensor] = None,  # Shape: (batch_size, seq_length)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Shape (batch_size, seq_length)
            attention_mask (torch.Tensor): Shape (batch_size, seq_length)
            labels (torch.Tensor, optional): Shape (batch_size, seq_length)
            
        Returns:
            Dict containing:
                - logits: Tensor of shape (batch_size, seq_length, num_labels)
                - loss: Scalar tensor if labels provided
                - decoded_tags: List[List[int]] if using CRF
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        sequence_output = outputs[0]  # Shape: (batch_size, seq_length, hidden_size)
        sequence_output = self.dropout(sequence_output)  # Apply dropout before classifier
        logits = self.classifier(sequence_output)  # Shape: (batch_size, seq_length, num_labels)

        loss, decoded_tags = None, None

        if labels is not None:
            # Create mask for valid positions
            # valid_positions: Boolean tensor of shape (batch_size, seq_length)
            valid_positions = (labels != -100) & attention_mask.bool()
            
            # Ensure the first timestep in the mask is always unmasked for CRF
            valid_positions[:, 0] = True

            # labels_for_crf: Tensor of shape (batch_size, seq_length)
            labels_for_crf = labels.clone()
            labels_for_crf[labels_for_crf == -100] = 0  # Safe default for ignored tokens
            
            if self.use_crf:
                # CRF loss computation
                loss = -self.crf(
                    logits, labels_for_crf, mask=valid_positions)
            else:
                # custom and CE loss
                loss = self.loss_fn(logits, labels, valid_positions)

        # Always decode during validation/testing when using CRF
        if self.use_crf and (not self.training or labels is None):
            # Use attention_mask if labels is None (during inference)
            mask_for_decode = valid_positions if labels is not None else attention_mask.bool()
            decoded_tags = self.crf.decode(logits, mask=mask_for_decode)
            # decoded_tags: List of lists, each inner list has length seq_length
        # Batch size, seq length, num labels
        return {
            "loss": loss,
            "logits": logits,  
            # Shape: (batch_size, seq_length, num_labels)
            "decoded_tags": decoded_tags if self.use_crf else None
        }