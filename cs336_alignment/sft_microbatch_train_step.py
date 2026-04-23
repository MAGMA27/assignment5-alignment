import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
            SFT policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
            prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
    
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
            this so we can log it.
        metadata Dict with metadata from the underlying loss call, and any other statistics you
            might want to log.
    '''
    # forward pass
    batch_size = policy_log_probs.shape[0]
    loss = -torch.sum(policy_log_probs.masked_fill(~response_mask, 0)) / batch_size / gradient_accumulation_steps / normalize_constant

    # backward pass
    loss.backward()

    # metadata
    metadata = {"gradient_accumulation_steps": gradient_accumulation_steps,
                "normalize_constant": normalize_constant}
    
    return loss, metadata