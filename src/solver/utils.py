import torch
import torch.nn as nn
def linear_normalize(weights):
    weights = torch.max(weights, torch.zeros_like(weights))
    if torch.sum(weights) > 1e-8:
        return weights / torch.sum(weights)
    return torch.zeros_like(weights)


def softmax_normalize(weights, temperature):
    return nn.functional.softmax(weights / temperature, dim=0)