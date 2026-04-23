import torch


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    '''
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
                containing unnormalized logits.
        Returns:
                torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
                prediction.
    '''
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    return -torch.sum(log_probs.exp() * log_probs, dim=-1)