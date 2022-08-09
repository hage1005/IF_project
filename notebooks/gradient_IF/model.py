#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pdb
from tqdm import tqdm
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from functools import reduce
from operator import __or__
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision import models
from magic_module import MagicModule
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

class NeuralNet(nn.Module):   
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
  
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

EPSILON = 1e-5

class ImageClassifier:
    def __init__(self, norm_fn='linear', softmax_temp=1.): # pretrained=True, 

        self._model = NeuralNet(784, 32, 10)
        self._optimizer = None

        self._dataset = {}
        self._data_loader = {}

        self._weights = None
        self._w_decay = None
        self.weight_grad_list = []

        self._norm_fn = norm_fn
        self._softmax_temp = softmax_temp
        
    def test_image(self, test_inputs, test_labels):
        self._test_inputs = test_inputs.reshape(-1, 28*28)
        self._test_labels = test_labels

    def init_weights(self, n_examples, w_init, w_decay):
        self._weights = torch.tensor([w_init] * n_examples, requires_grad=True)
        self._w_decay = w_decay
    
    def get_sampler(self, labels, num_classes, num_per_class):
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(num_classes)]))
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:num_per_class] for i in range(num_classes)]) 
        indices = torch.from_numpy(indices)
        return indices

    def load_data(self, num_classes, num_per_class, batch_size):
        mnist_train = MNIST(root = "./data", train = True, transform = transforms.ToTensor(), download = True)
        mnist_test = MNIST(root = "./data", train = False, transform = transforms.ToTensor(), download = True)
        
        indices = self.get_sampler(mnist_train.targets, num_classes, num_per_class)
        mnist_train_inputs = mnist_train.data.float()[indices]
        mnist_train_targets = mnist_train.targets[indices]
        mnist_test_inputs = mnist_test.data.float()
        mnist_test_targets = mnist_test.targets
        
        self._data_loader_train = DataLoader(TensorDataset(mnist_train_inputs, mnist_train_targets, 
                                             torch.linspace(0,len(indices)-1,len(indices)).long()),
                                             batch_size = batch_size, shuffle = True)
        
        self._data_loader_test = DataLoader(TensorDataset(mnist_test_inputs, mnist_test_targets,
                                            torch.linspace(0,len(mnist_test)-1,len(mnist_test)).long()), 
                                            batch_size = batch_size, shuffle = False)
        self.mnist_train = mnist_train
        self.mnist_test = mnist_test
        self.indices = indices
        
    def get_optimizer(self, learning_rate): 
        self._optimizer = optim.Adam(self._model.parameters(), lr = learning_rate) 

    def pretrain_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self._model.train()
        for batch in tqdm(self._data_loader_train, desc = 'Training Epoch'):
            inputs, labels, ids = tuple(t for t in batch)
            x = inputs.reshape(-1, 28*28)     
            logits = self._model(x)
            loss = criterion(logits, labels)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
                     
    def train_epoch(self):
        criterion = nn.CrossEntropyLoss()
        for batch in tqdm(self._data_loader_train, desc = 'Training Epoch'):
            if self._norm_fn == 'softmax':
                weight_grad = softmax_normalize(self._get_weights(batch), temperature = self._softmax_temp)
                self.weight_grad_list.append(weight_grad)
            else:
                weights = linear_normalize(self._get_weights(batch))
                self.weight_grad_list.append(weight_grad)

    def _get_weights(self, batch):
        self._model.eval()
        inputs, labels, ids = tuple(t for t in batch)
        batch_size = inputs.shape[0]
        x = inputs.reshape(-1, 28*28)
        weights = self._weights[ids]
        magic_model = MagicModule(self._model)
        criterion = nn.CrossEntropyLoss()
        model_tmp = copy.deepcopy(self._model)
        optimizer_hparams = self._optimizer.state_dict()['param_groups'][0]
        optimizer_tmp = optim.Adam(model_tmp.parameters(),lr = optimizer_hparams['lr'])

        for i in range(batch_size):
            model_tmp.load_state_dict(self._model.state_dict())
            optimizer_tmp.load_state_dict(self._optimizer.state_dict())
            model_tmp.zero_grad()

            if i > 0:
                l, r, t = i - 1, i + 1, 1
            else:
                l, r, t = i, i + 2, 0

            logits = model_tmp(x[l:r])[t:t+1]
            loss = criterion(logits, labels[i:i+1])
            loss.backward()
            optimizer_tmp.step()

            deltas = {}
            for (name, param), (name_tmp, param_tmp) in zip(
                    self._model.named_parameters(),
                    model_tmp.named_parameters()):
                assert name == name_tmp
                deltas[name] = weights[i] * (param_tmp.data - param.data)
            magic_model.update_params(deltas)
        
        if weights.grad is not None:
            weights.grad.zero_()

        test_logits = magic_model(self._test_inputs)
        test_logits = torch.squeeze(test_logits)
        test_loss = criterion(test_logits, self._test_labels)
        weights_grad = torch.autograd.grad(test_loss, weights, retain_graph=True)[0]
        
        with torch.no_grad(): 
            self._weights[ids] = weights.data / self._w_decay - weights_grad 
            self._weights[ids] = torch.max(self._weights[ids], torch.ones_like(self._weights[ids]).fill_(EPSILON))
        return weights_grad

    def evaluate(self):
        self._model.eval()
        preds_all, labels_all = [], []
        for batch in tqdm(self._data_loader_test, desc='Test Epoch'):
            x, y, _ = tuple(t for t in batch)
            x = x.reshape(-1, 28*28)
            with torch.no_grad():
                test_logits = self._model(x)
            preds = torch.argmax(test_logits, dim=1)
            preds_all.append(preds)
            labels_all.append(y)

        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
                
        return torch.sum(preds_all == labels_all).item() / labels_all.shape[0]


def linear_normalize(weights):
    weights = torch.max(weights, torch.zeros_like(weights))
    if torch.sum(weights) > 1e-8:
        return weights / torch.sum(weights)
    return torch.zeros_like(weights)


def softmax_normalize(weights, temperature):
    return nn.functional.softmax(weights / temperature, dim=0)

