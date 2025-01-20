import torch
import torch.nn.functional as F

def iou_loss_fn(logits, labels, valid_positions):
    """
    Compute a token-level soft IoU loss for multi-class NER.
    
    Args:
        logits: (batch_size, seq_len, num_labels) raw model outputs
        labels: (batch_size, seq_len) integer labels in [0, num_labels-1], or -100 for ignore
        valid_positions: (batch_size, seq_len) boolean mask for valid tokens
    
    Returns:
        A scalar IoU loss.
    """
    # 1. Filter out invalid tokens
    #    shape => (batch_size, seq_len)
    mask_flat = valid_positions.view(-1)  # shape => (batch_size * seq_len)

    # 2. Flatten logits => (batch_size*seq_len, num_labels)
    batch_size, seq_len, num_labels = logits.size()
    logits_flat = logits.view(-1, num_labels)

    # 3. Flatten labels => (batch_size*seq_len,)
    labels_flat = labels.view(-1)

    # 4. Extract only valid tokens
    logits_valid = logits_flat[mask_flat]      # shape => (N_valid, num_labels)
    labels_valid = labels_flat[mask_flat]      # shape => (N_valid,)

    if labels_valid.numel() == 0:
        # If there's no valid tokens in the batch, return 0
        return logits_valid.sum() * 0.0

    # 5. Convert logits to probabilities => softmax
    probs = F.softmax(logits_valid, dim=-1)  # shape => (N_valid, num_labels)

    # 6. One-hot encode the gold labels
    #    shape => (N_valid, num_labels)
    one_hot_labels = F.one_hot(labels_valid, num_classes=num_labels).float()

    # 7. Intersection: sum_{valid tokens} (probs * one_hot)
    #    intersection = Σ (p * y)
    intersection = torch.sum(probs * one_hot_labels, dim=-1)  # shape => (N_valid,)

    # 8. Union: sum_{valid tokens} (p + y - p*y)
    #    but since y is {0,1}, p + y - p*y = p + y - p if y=1 => y + p(1-y) => etc.
    #    simpler to do element-wise on the vectors then sum
    union_elementwise = probs + one_hot_labels - (probs * one_hot_labels)
    union = torch.sum(union_elementwise, dim=-1)  # shape => (N_valid,)

    # 9. IoU per token => intersection / union
    #    Some tokens might have union=0 if label is all zeros (shouldn't happen with valid positions)
    iou_per_token = intersection / (union + 1e-8)

    # 10. IoU Loss = 1 - average IoU
    iou = torch.mean(iou_per_token)
    iou_loss = 1.0 - iou
    return iou_loss

return iou_loss_fn