import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing for NER tasks.
    
    Helps prevent the model from becoming overconfident and improves generalization.
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean', ignore_index: int = -100):
        """
        Args:
            smoothing: Label smoothing factor (0 means no smoothing)
            reduction: 'none', 'mean', or 'sum'
            ignore_index: Index to ignore (e.g., for padding)
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (batch_size, sequence_length, num_classes)
            targets: Ground truth labels (batch_size, sequence_length)
        """
        batch_size, seq_length, num_classes = inputs.size()
        
        # Create a mask for ignored indices
        mask = (targets != self.ignore_index).float()
        
        # KL div expects log-probabilities for input
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Get the log probability of the target classes
        target_probs = torch.zeros_like(log_probs).scatter_(-1, targets.unsqueeze(-1), 1)
        
        # Apply label smoothing
        smoothed_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
        smoothed_targets.scatter_(-1, targets.unsqueeze(-1), 1 - self.smoothing)
        
        # Calculate the loss
        loss = -(smoothed_targets * log_probs).sum(dim=-1) * mask
        
        if self.reduction == 'mean':
            return loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
