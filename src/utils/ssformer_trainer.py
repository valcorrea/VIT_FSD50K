from src.models.ssformer import SSTransformer
import torch
from torch import optim, nn
import math

def compute_var(y: torch.Tensor):
    """
    Function for computing standard deviation of target
    :param y:
    :return: standard deviation of y
    """
    y = y.view(-1, y.size(-1))
    return torch.sqrt(y.var(dim=0) + 1e-6).mean()

def train_single_batch(net: SSTransformer, data: torch.Tensor, mask: torch.Tensor, optimizer: optim.Optimizer,
                       criterion, device: torch.device):
    """
    Performs single training step for Data2Vec model
    :param net: Data2Vec model
    :param data: input data
    :param mask: mask for student input
    :param optimizer: torch optimizer
    :param criterion: torch criterion
    :param device: device for model and optimizer
    :return: loss, target variance and prediction variance
    """

    # Send data to GPU if available
    data = data.to(device)

    # Reset gradients
    optimizer.zero_grad()

    # Put the same input for student and teacher. Get outputs (final hidden states)
    predictions, targets = net(data, data, mask)
    scale = math.sqrt(predictions.size(dim=-1))
    loss = criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        target_var = compute_var(targets.float())
        prediction_var = compute_var(predictions.float())
    return loss.item(), target_var.item(), prediction_var.item()