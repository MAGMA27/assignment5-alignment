import torch


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Args:
    advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
        probs from the policy being trained.
    old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
        from the old policy.
    cliprange: float Clip parameter ε (e.g. 0.2).

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
            loss.
        metadata dict containing whatever you want to log. We suggest logging whether each
            token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
            the min was lower than the LHS
    '''
    importance_sampling = torch.exp(policy_log_probs - old_log_probs)
    clip_loss = -torch.minimum(advantages*importance_sampling, advantages*torch.clip(importance_sampling, 1-cliprange, 1+cliprange))
    return clip_loss, {"metadata": -1}