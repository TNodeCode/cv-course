from typing import Tuple

import torch


def stat_scores(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Compute the total number of true positives, false positives, false negatives and true negatives
    for a given input and target tensor on pixel level.

    Args:
        inputs: Int tensor (predictions of the model) of shape (N, H, W)
        targets: Int tensor of the same shape as input
        num_classes: Number of classes

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
            True positives, false positives, false negatives and true negatives
    """
    assert inputs.shape == targets.shape, f"Inputs and targets must have the same shape, got {inputs.shape} and {targets.shape}"
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################

    tp = ((inputs == targets) & (targets > 0)).sum() # True positives
    tn = ((inputs == targets) & (targets == 0)).sum() # True negatives
    fp = ((inputs != targets) & (targets > 0)).sum() # False positives
    fn = ((inputs != targets) & (targets == 0)).sum() # False negatives

    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################
    return tp, fp, fn, tn


def iou_score(tp, fp, fn, tn):
    """
    Compute the Intersection over Union score or Jaccard index for a given number of true positives,
    false positives, false negatives and true negatives.
    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################
    
    # Compute IoU score
    score = tp / (tp + fp + fn)

    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################
    return score


def accuracy_score(tp, fp, fn, tn):
    """
    Compute the accuracy for a given number of true positives,
    false positives, false negatives and true negatives.
    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################
    
    # Compute accuracy score
    score = (tp + tn) / (tp + tn + fp + fn)

    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################
    return score
