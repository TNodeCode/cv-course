from typing import Optional

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Multi class dice loss for output logits.

    The dice loss is defined as 1 - dice_score, where dice_score is defined as
    (2 * intersection) / (cardinality).

    """

    def __init__(self, reduce: Optional[str] = None, eps: float = 1e-7):
        """
        Args:
            reduce: None, "mean", or "sum"
            eps: epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.reduce = reduce

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """

        Args:
            inputs: logits of shape (N, C, H, W)
            targets: ground truths  (N, C, H, W)

        Returns:
            torch.Tensor: Dice loss of shape (C,) if reduce is None, else scalar
        """
        assert inputs.shape == targets.shape, f"Inputs and targets must have the same shape, got {inputs.shape} and {targets.shape}"
        N = inputs.shape[0]  # batch size
        C = inputs.shape[1]  # number of classes

        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################
        
        # Calculate log softmax probabilities of classes
        log_sm = torch.exp(torch.nn.functional.log_softmax(inputs, dim=1))
        
        # Create empty tensor that contains loss values for each sample
        loss = torch.zeros(c)
        
        # Iterate over classes to compute the class loss
        for c in range(C):
            nominator = torch.sum(log_sm[:, c] * targets[:, c]) + self.eps
            denominator = torch.sum(log_sm[:, c] + targets[:, c]) + self.eps
            loss[c] = 1 - (2/C) * nominator / denominator
            
        # convert loss to scalar by computing the mean loss if the reduce flag is set to true
        if self.reduce:
            loss = torch.mean(loss)        

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################

        return loss
