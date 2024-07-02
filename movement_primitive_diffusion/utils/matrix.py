import torch
from typing import Optional


def unsqueeze_and_expand(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Unsqueeze and expand tensor to given size along given dimension

    Example:
        >>> a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> a.shape
        torch.Size([2, 4])
        >>> b = unsqueeze_and_expand(a, dim=1, size=3)
        >>> b.shape
        torch.Size([2, 3, 4])
        >>> b
        tensor([[[1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4]],

            [[5, 6, 7, 8],
            [5, 6, 7, 8],
            [5, 6, 7, 8]]])

    Args:
        tensor (torch.Tensor): Tensor to unsqueeze and expand
        dim (int): Dimension to unsqueeze and expand
        size (int): Size to expand to

    Returns:
        torch.Tensor: Unsqueezed and expanded tensor
    """

    unsqueezed_tensor = tensor.unsqueeze(dim)
    target_shape = list(unsqueezed_tensor.shape)
    target_shape[dim] = size
    return unsqueezed_tensor.expand(target_shape)


def build_lower_matrix(diagonal_values: torch.Tensor, off_diagonal_values: Optional[torch.Tensor]) -> torch.Tensor:
    """Compose the lower triangular matrix L from diagonal and off-diagonal elements.

    It seems like faster than using the cholesky transformation from PyTorch.

    Args:
        diagonal_values (torch.Tensor): diagonal parameters
        off_diagonal_values (torch.Tensor): off-diagonal parameters

    Returns:
        L (torch.Tensor): Lower triangular matrix L
    """

    # Determine the height/width of the square matrix
    matrix_size = diagonal_values.shape[-1]
    # Fill diagonal terms
    L = diagonal_values.diag_embed()
    if off_diagonal_values is not None:
        # Fill off-diagonal terms
        [row, col] = torch.tril_indices(matrix_size, matrix_size, -1)
        L[..., row, col] = off_diagonal_values[..., :]

    return L
