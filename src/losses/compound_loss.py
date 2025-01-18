import torch
import torch.nn as nn

class CompoundLoss(nn.Module):
    """Combines multiple loss functions with weighted contributions.
    
    Useful for combining different loss functions to capture different aspects
    of the NER task.
    """
    
    def __init__(self, losses: dict, weights: dict):
        """
        Args:
            losses: Dictionary of loss functions
            weights: Dictionary of weights for each loss
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights
        
        # Validate weights
        assert set(losses.keys()) == set(weights.keys()), "Losses and weights must have the same keys"
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (batch_size, sequence_length, num_classes)
            targets: Ground truth labels (batch_size, sequence_length)
            mask: Optional mask for padding (batch_size, sequence_length)
        """
        total_loss = 0
        losses_dict = {}
        
        for name, loss_fn in self.losses.items():
            if mask is not None and hasattr(loss_fn, 'forward') and 'mask' in loss_fn.forward.__code__.co_varnames:
                curr_loss = loss_fn(inputs, targets, mask)
            else:
                curr_loss = loss_fn(inputs, targets)
            
            weighted_loss = self.weights[name] * curr_loss
            total_loss += weighted_loss
            losses_dict[name] = curr_loss.item()
        
        return total_loss, losses_dict
