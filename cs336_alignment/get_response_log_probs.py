import torch
from cs336_alignment.compute_entropy import compute_entropy

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    '''
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
            response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling
            compute_entropy.
    Returns:
        dict[str, torch.Tensor].
            "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
            log pθ(xt |x<t).
            "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
            for each position (present only if return_token_entropy=True)
    '''
    logits = model(input_ids).logits
    results = {"log_probs": torch.nn.functional.log_softmax(logits, -1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)}

    if return_token_entropy:
        results['token_entropy'] = compute_entropy(logits)

    return results