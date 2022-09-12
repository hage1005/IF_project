import torch
import torch.nn as nn


class Normalizer:
    def __init__(self, norm_type, temperature = None):
        self.norm_type = norm_type
        self.temperature = temperature

    def __call__(self, weights):
        if self.norm_type == 'softmax':
            return self.softmax_normalize(weights, self.temperature)
        elif self.norm_type == 'linear':
            return self.linear_normalize(weights)
        elif self.norm_type == 'linear_clip_min':
            return self.linear_normalize_clip_min(weights)
        else:
            raise NotImplementedError
    
    def linear_normalize_clip_min(self, weights):
        weights = torch.max(weights, torch.zeros_like(weights))
        if torch.sum(weights) > 1e-8:
            return weights / torch.sum(weights)
        return weights

    def linear_normalize(self, weights):
        if torch.sum(weights) > 1e-8:
            return weights / torch.sum(weights)
        return weights


    def softmax_normalize(self, weights, temperature):
        return nn.functional.softmax(weights / temperature, dim=0)