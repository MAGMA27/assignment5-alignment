import torch
from typing import Literal
from cs336_alignment.compute_policy_gradient_loss import compute_policy_gradient_loss
from cs336_alignment.masked_mean import masked_mean


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Execute a forward-and-backward pass on a microbatch.

    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
    response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.
    gradient_accumulation_steps Number of microbatches per optimizer step.
    loss_type One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
    raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).
    advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
    old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
    cliprange Clip parameter ε for GRPO-Clip.

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
            this so we can log it.
        metadata Dict with metadata from the underlying loss call, and any other statistics you
            might want to log.
    '''
    pg_loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    pg_loss = masked_mean(pg_loss, response_mask, -1)
    pg_loss = torch.mean(pg_loss) / gradient_accumulation_steps
    pg_loss.backward()

    return pg_loss, metadata