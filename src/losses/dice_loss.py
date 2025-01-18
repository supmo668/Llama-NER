import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss for NER tasks, useful for handling class imbalance.
    
    Implements a smooth version of the Dice coefficient as a loss function.
    """
    
    def __init__(
        self, smooth: float = 1.0, square_denominator: bool = True, reduction: str = 'mean'
        ):
        """
        Args:
            smooth: Smoothing factor to prevent division by zero
            square_denominator: Whether to square the denominator
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (batch_size, sequence_length, num_classes)
            targets: Ground truth labels (batch_size, sequence_length)
            mask: Optional mask for padding (batch_size, sequence_length)
        """
        # Convert targets to one-hot
        targets = F.one_hot(targets, inputs.size(-1)).float()
        
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=-1)
        
        # Calculate intersection and union
        intersection = (probs * targets).sum(dim=(1, 2))
        if self.square_denominator:
            denominator = (probs * probs).sum(dim=(1, 2)) + (targets * targets).sum(dim=(1, 2))
        else:
            denominator = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        
        # Calculate Dice coefficient
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        loss = 1 - dice
        
        if mask is not None:
            loss = loss * mask.any(dim=1).float()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
