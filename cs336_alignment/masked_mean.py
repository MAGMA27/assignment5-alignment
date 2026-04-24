import torch


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    '''
    Compute the mean of tensor along a given dimension, considering only those elements where
    mask == 1.

    Args:
        tensor: torch.Tensor The data to be averaged.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        dim: int | None Dimension over which to average. If None, compute the mean over all
        masked elements.

    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    '''
    masked_tensor = tensor.masked_fill(~mask, 0)
    return torch.sum(masked_tensor, dim=dim) / torch.sum(mask, dim=dim)