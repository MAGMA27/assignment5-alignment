import torch

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    '''
    Sum over a dimension and normalize by a constant, considering only those elements where mask
    == 1.
    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all
            dimensions.
    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to
        the sum.
    '''
    tensor_masked = tensor.masked_fill(~mask, 0)
    return torch.sum(tensor_masked, dim) / normalize_constant
