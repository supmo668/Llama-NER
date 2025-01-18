import torch
import torch.nn as nn

class CRF(nn.Module):
    """Conditional Random Field (CRF) layer for sequence labeling.
    
    Implements the forward-backward algorithm for training and Viterbi decoding
    for inference.
    """
    
    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition scores from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Score for transitioning to STOP
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        # Score for transitioning from START
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters following the Xavier initialization."""
        nn.init.xavier_normal_(self.transitions)
        nn.init.xavier_normal_(self.end_transitions.unsqueeze(0))
        nn.init.xavier_normal_(self.start_transitions.unsqueeze(0))
    
    def forward(self, emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: torch.ByteTensor = None) -> torch.Tensor:
        """Compute the negative log-likelihood for the given sequence of emissions and tags.
        
        Args:
            emissions: (seq_length, batch_size, num_tags) or (batch_size, seq_length, num_tags)
            tags: (seq_length, batch_size) or (batch_size, seq_length)
            mask: (seq_length, batch_size) or (batch_size, seq_length)
        
        Returns:
            neg_log_likelihood: The negative log-likelihood for the sequence
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        neg_log_likelihood = denominator - numerator
        
        return neg_log_likelihood
    
    def decode(self, emissions: torch.Tensor,
               mask: torch.ByteTensor = None) -> torch.Tensor:
        """Find the most likely tag sequence using Viterbi algorithm.
        
        Args:
            emissions: (seq_length, batch_size, num_tags) or (batch_size, seq_length, num_tags)
            mask: (seq_length, batch_size) or (batch_size, seq_length)
        
        Returns:
            tags: The most likely tag sequence
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        
        return self._viterbi_decode(emissions, mask)
