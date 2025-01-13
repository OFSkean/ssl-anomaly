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

def compute_LDA_matrix(Z, return_within_class_scatter=False):
    """
    Compute the LDA matrix as defined in the LIDAR paper.

    Args:
        Z (torch.Tensor): Tensor of augmented prompts.
        return_within_class_scatter (bool): Whether to return the within-class scatter matrix.

    Returns:
        torch.Tensor: The computed LDA matrix or within-class scatter matrix.
    """
    # Z is tensor that is NUM_SAMPLES x NUM_AUGMENTATIONS x D
    NUM_SAMPLES, NUM_AUGMENTATIONS, D = Z.shape

    delta = 1e-4

    dataset_mean = torch.mean(Z, dim=(0, 1)).squeeze() # D
    class_means = torch.mean(Z, dim=1) # NUM_SAMPLES x D

    # Equation 1 in LIDAR paper
    between_class_scatter = torch.zeros((D, D)).to(Z.device)
    for i in range(NUM_SAMPLES):
        between_class_scatter += torch.outer(class_means[i] - dataset_mean, class_means[i] - dataset_mean)
    between_class_scatter /= NUM_SAMPLES

    # Equation 2 in LIDAR paper
    within_class_scatter = torch.zeros((D, D)).to(Z.device)
    for i in range(NUM_SAMPLES):
        for j in range(NUM_AUGMENTATIONS):
            within_class_scatter += torch.outer(Z[i, j] - class_means[i], Z[i, j] - class_means[i])
    within_class_scatter /= (NUM_SAMPLES * NUM_AUGMENTATIONS)
    within_class_scatter += delta * torch.eye(D).to(Z.device)

    if return_within_class_scatter:
        return within_class_scatter 
    
    # Equation 3 in LIDAR paper
    eigs, eigvecs = torch.linalg.eigh(within_class_scatter)
    within_sqrt = torch.diag(eigs**(-0.5))
    fractional_inverse = eigvecs @ within_sqrt @ eigvecs.T
    LDA_matrix = fractional_inverse @ between_class_scatter @ fractional_inverse

    return LDA_matrix

def compute_lidar(Z, alpha=1, return_within_scatter=False):
    """
    Compute the LIDAR metric for hidden states.

    Args:
        hidden_states (torch.Tensor): The hidden states to compute LIDAR for. Must be of shape NUM_SAMPLES x NUM_AUGMENTATIONS x D, 
                                             where NUM_AUGMENTATIONS is preferably high (50ish).
        alpha (float): The alpha parameter for entropy calculation.
        return_within_scatter (bool): Whether to return the within-class scatter matrix.

    Returns:
        float: The computed LIDAR metric.
    """
    lda_matrix = compute_LDA_matrix(Z.double(), return_within_class_scatter=return_within_scatter)
    lda_rankme = rankme(lda_matrix, alpha=alpha)
    return lda_rankme