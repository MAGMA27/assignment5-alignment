import torch
from typing import Literal
from cs336_alignment.compute_naive_policy_gradient_loss import compute_naive_policy_gradient_loss
from cs336_alignment.compute_grpo_clip_loss import compute_grpo_clip_loss

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Select and compute the desired policy-gradient loss.

    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
    loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
    advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape
        (batch_size, 1).
    old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
    cliprange Required for "grpo_clip"; scalar ε used for clipping.

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss (batch_size, sequence_length), per-token loss.
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip)
    '''
    meta_dict = {"metadate": -1}

    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), meta_dict
    
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), meta_dict

    else:
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
