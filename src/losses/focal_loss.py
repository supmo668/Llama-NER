import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in NER tasks.
    
    Implementation follows the paper "Focal Loss for Dense Object Detection"
    with adaptations for sequence labeling.
    """
    
    def __init__(self, alpha: float = 1, gamma: float = 2, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for the rare class
            gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (batch_size, sequence_length, num_classes)
            targets: Ground truth labels (batch_size, sequence_length)
            mask: Optional mask for padding (batch_size, sequence_length)
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Get the log probability of the target classes
        target_probs = torch.gather(probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Calculate focal loss
        loss = -self.alpha * (1 - target_probs) ** self.gamma * log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        if mask is not None:
            loss = loss * mask
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
