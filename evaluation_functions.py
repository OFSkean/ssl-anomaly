import torch

def rankme(Z: torch.Tensor, eps: float = 1e-6) -> float:  # pylint: disable=invalid-name
    if Z.dtype != torch.float32:
        # svdvals cannot be computed over half precision
        Z = Z.to(torch.float32)

    # Handle the case where the input matrix is not square and the SVD cannot be computed
    try:
        # Compute Singluar Values of the feature matrix
        S = torch.linalg.svdvals(Z)  # pylint: disable=invalid-name, not-callable

        # Compute the norm-1 of the singular values vector
        S_norm = torch.linalg.norm(  # pylint: disable=invalid-name, not-callable
            S, ord=1
        )

        # Compute p_k
        p_k = (S / S_norm) + eps

        # Compute Shannon's entropy
        entropy = -torch.sum(p_k * torch.log(p_k))  # pylint: disable=no-member

        rank_me_score = torch.exp(entropy).item()  # pylint: disable=no-member
    except:  # pylint: disable=bare-except
        rank_me_score = -1

    return rank_me_score


def renyi_rankme(Z: torch.Tensor, alpha: float = 2.0, eps: float = 1e-6) -> float:  # pylint: disable=invalid-name
    """
    The Renyi generalization of the RankMe score.

    Is equivalent to the rankme score in the limit case of alpha = 1.
    Is equivalent to the squared Frobenius norm in the case of alpha = 2.

    """
    if Z.dtype != torch.float32:
        # svdvals cannot be computed over half precision
        Z = Z.to(torch.float32)

    # Handle the case where the input matrix is not square and the SVD cannot be computed
    try:
        # Compute Singluar Values of the feature matrix
        S = torch.linalg.svdvals(Z)  # pylint: disable=invalid-name, not-callable

        # Compute the norm-1 of the singular values vector
        S_norm = torch.linalg.norm(  # pylint: disable=invalid-name, not-callable
            S, ord=1
        )

        # Compute p_k
        p_k = (S / S_norm) + eps

        # Compute Shannon's entropy
        entropy = 1/(1-alpha) * torch.log(torch.sum(p_k**alpha))

        rank_me_score = torch.exp(entropy).item()  # pylint: disable=no-member
    except:  # pylint: disable=bare-except
        rank_me_score = -1

    return rank_me_score