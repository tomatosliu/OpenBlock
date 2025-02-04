# test a cross entropy loss for a target in [0, 1, 2, 3]
# assume we get input logits in shape [batch_size, 4]
# and target in shape [batch_size]

import torch
import torch.nn as nn
import torch.nn.functional as F


def CrossEntropyLoss(logits, target, weights=None, reduction='mean'):
    logits = F.log_softmax(logits, dim=1)
    target = F.one_hot(target, num_classes=logits.shape[1])
    if weights is not None:
        loss = -torch.sum(target * logits * weights, dim=1)
    else:
        loss = -torch.sum(target * logits, dim=1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


if __name__ == "__main__":
    # means 4 classes and batch size is 4
    logits = torch.randn(4, 4, dtype=torch.float)
    target = torch.randint(0, 4, (4,), dtype=torch.long)
    ce_loss = nn.CrossEntropyLoss()
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0],
                           dtype=torch.float).softmax(dim=0)
    wce_loss = nn.CrossEntropyLoss(
        weight=weights,
        reduction='mean')

    print("logits: ", logits)
    print("target: ", target)
    print("torch CrossEntropy: ", ce_loss(logits, target))
    print("my CrossEntropy: ", CrossEntropyLoss(
        logits, target, reduction='mean'))
    print("torch WeightedCrossEntropy: ", wce_loss(logits, target))
    print("my WeightedCrossEntropy: ", CrossEntropyLoss(
        logits, target, weights, reduction='mean'))
