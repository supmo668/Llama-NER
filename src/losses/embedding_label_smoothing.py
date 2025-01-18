import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLabelSmoothing(nn.Module):
    """Enhanced Label Smoothing using label embeddings for NER tasks.
    
    This implementation improves upon traditional label smoothing by:
    1. Learning label embeddings that capture semantic relationships
    2. Computing smoothing weights based on label similarities
    3. Using cosine similarity for more robust label comparisons
    """
    
    def __init__(
        self,
        num_labels: int,
        hidden_size: int,
        label_emb_dim: int = 32,
        smoothing_alpha: float = 10.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            num_labels: Number of NER labels
            hidden_size: Size of input hidden states
            label_emb_dim: Dimension of label embeddings
            smoothing_alpha: Temperature parameter for similarity scaling
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.num_labels = num_labels
        self.label_emb_dim = label_emb_dim
        self.smoothing_alpha = smoothing_alpha
        self.reduction = reduction
        
        # Project hidden states to label embedding space
        self.projection = nn.Linear(hidden_size, label_emb_dim)
        
        # Learnable label embeddings
        self.label_embeddings = nn.Parameter(torch.randn(num_labels, label_emb_dim))
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize label embeddings using Xavier initialization."""
        nn.init.xavier_uniform_(self.label_embeddings)
    
    def _compute_label_distribution(self, gold_label_idx: torch.Tensor) -> torch.Tensor:
        """Compute smooth label distribution based on embedding similarities.
        
        Args:
            gold_label_idx: Label indices of shape [batch_size, seq_len]
            
        Returns:
            Tensor of shape [batch_size, seq_len, num_labels] containing
            smoothed label distributions
        """
        # Get embeddings for gold labels
        gold_label_emb = self.label_embeddings[gold_label_idx]  # [batch, seq, emb_dim]
        
        # Compute similarities with all labels
        # [batch, seq, num_labels]
        sims = F.cosine_similarity(
            self.label_embeddings.unsqueeze(0).unsqueeze(0),  # [1, 1, num_labels, emb_dim]
            gold_label_emb.unsqueeze(2),  # [batch, seq, 1, emb_dim]
            dim=-1
        )
        
        # Scale and convert to probabilities
        logits = self.smoothing_alpha * sims
        return F.softmax(logits, dim=-1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_size]
            labels: Label indices [batch, seq_len]
            attention_mask: Optional mask [batch, seq_len]
        """
        # Project hidden states to label embedding space
        predicted_embs = self.projection(hidden_states)  # [batch, seq_len, label_emb_dim]
        
        # Compute similarities with all label embeddings
        # [batch, seq_len, num_labels]
        sim_matrix = F.cosine_similarity(
            self.label_embeddings.unsqueeze(0).unsqueeze(0),  # [1, 1, num_labels, emb_dim]
            predicted_embs.unsqueeze(2),  # [batch, seq_len, 1, emb_dim]
            dim=-1
        )
        
        # Get predicted distribution
        pred_logits = self.smoothing_alpha * sim_matrix
        pred_dist = F.softmax(pred_logits, dim=-1)  # [batch, seq_len, num_labels]
        
        # Get smoothed gold distribution
        gold_dist = self._compute_label_distribution(labels)  # [batch, seq_len, num_labels]
        
        # Compute KL divergence loss
        loss = -torch.sum(gold_dist * torch.log(pred_dist + 1e-9), dim=-1)  # [batch, seq_len]
        
        if attention_mask is not None:
            loss = loss * attention_mask
        
        # Apply reduction
        if self.reduction == 'mean':
            if attention_mask is not None:
                return loss.sum() / attention_mask.sum()
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def get_predictions(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get label predictions for inference.
        
        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_size]
            
        Returns:
            Label predictions [batch, seq_len]
        """
        predicted_embs = self.projection(hidden_states)
        sim_matrix = F.cosine_similarity(
            self.label_embeddings.unsqueeze(0).unsqueeze(0),
            predicted_embs.unsqueeze(2),
            dim=-1
        )
        return sim_matrix.argmax(dim=-1)
