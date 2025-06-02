import torch
import torch.nn as nn


class MaskedL1Loss(nn.Module):
    """
    Custom L1 loss for regression that masks (ignores) dimensions where the target is zero.
    """

    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'.
        """
        super(MaskedL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, prediction, target):
        """
        Args:
            prediction (Tensor): Predicted output of shape (batch_size, output_dim).
            target (Tensor): Ground truth target of same shape.

        Returns:
            loss (Tensor): Computed masked L1 loss.
        """
        # Compute absolute error
        abs_error = torch.abs(prediction - target)

        # Create a mask where target is not zero
        mask = (target != 0).float()

        # Apply mask to the error
        masked_error = abs_error * mask

        if self.reduction == 'mean':
            # Avoid division by zero
            nonzero_elements = mask.sum()
            return masked_error.sum() / (nonzero_elements + 1e-8)
        elif self.reduction == 'sum':
            return masked_error.sum()
        else:
            return masked_error


class MaskedLpLoss(nn.Module):
    """
    Generalized masked Lp loss for regression. Ignores dimensions where the target is zero.

    Args:
        degree (float): The exponent `p` in the Lp loss (e.g., 1 for L1, 2 for L2).
        reduction (str): 'mean', 'sum', or 'none' to control reduction over batch.
    """

    def __init__(self, degree=2, reduction='mean'):
        super(MaskedLpLoss, self).__init__()
        assert degree > 0, "Degree must be positive"
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'"
        self.degree = degree
        self.reduction = reduction

    def forward(self, prediction, target):
        # Compute Lp error: |pred - target|^p
        error = torch.abs(prediction - target) ** self.degree

        # Mask where target is not zero
        mask = (target != 0).float()

        # Apply mask
        masked_error = error * mask

        if self.reduction == 'mean':
            nonzero_elements = mask.sum()
            return masked_error.sum() / (nonzero_elements + 1e-8)
        elif self.reduction == 'sum':
            return masked_error.sum()
        else:  # 'none'
            return masked_error
