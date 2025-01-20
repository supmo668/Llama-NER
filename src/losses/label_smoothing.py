import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing for token classification with an ignore_index.
    This version excludes tokens where label == -100.
    """
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, logits, target):
        """
        logits: (N, C) float tensor, where N = number of valid tokens
        target: (N,) long tensor of gold labels in [0..C-1]
        """
        # 1. Filter out ignore_index tokens
        valid_mask = (target != self.ignore_index)
        logits = logits[valid_mask]     # (N_valid, C)
        target = target[valid_mask]     # (N_valid,)

        if target.numel() == 0:
            # If there are no valid tokens, return 0.0 loss
            return logits.sum() * 0.0

        # 2. Softmax probabilities
        log_probs = F.log_softmax(logits, dim=-1)   # shape: (N_valid, C)
        n_classes = logits.size(-1)

        # 3. Construct smoothed label distribution
        #    For correct label: (1 - epsilon); for others: epsilon / (n_classes - 1)
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            # place (1 - epsilon) at the correct label index
            confidence = 1.0 - self.smoothing
            smooth_dist.scatter_(1, target.unsqueeze(1), confidence)

        # 4. Compute cross entropy with the smoothed distribution
        #    CE = - \sum_y p_smooth(y) * log_probs(y)
        loss = -torch.sum(smooth_dist * log_probs, dim=-1).mean()
        return loss
