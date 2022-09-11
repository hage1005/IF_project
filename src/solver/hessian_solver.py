import argparse
import os
import random
from tkinter import Image
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
import copy

import numpy as np
import torch

from torch import optim
from torchvision import models
from src.data_utils.MnistDataset import MnistDataset
from src.utils.utils import save_json
from src.data_utils.Cifar10Dataset import Cifar10Dataset
from src.solver.fenchel_solver import FenchelSolver
from src.modeling.classification_models import CnnCifar, MNIST_1
from src.modeling.influence_models import Net_IF, MNIST_IF_1
from torch.autograd.functional import hessian
from torch.nn.utils import _stateless
from torch.nn import CrossEntropyLoss 
import torch.nn.functional as F
from torch.nn.utils import _stateless
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler


class hessianSolver:
    def __init__(self, classification_model, inv_hessian_path, pretrained_classification_path):
        self._classification_model = classification_model
        self.shapes = [p.shape for p in classification_model.parameters()]
        self.inv_hessian_path = inv_hessian_path
        self.pretrained_classification_path = pretrained_classification_path
        self.global_iter = 0
        self.global_epoch = 0
        self.criterion = CrossEntropyLoss()
        self.names = list(n for n, _ in classification_model.named_parameters())
        self._dataset = {}
        self._data_loader = {}
        self._optimizer_classification = None



    def get_optimizer_classification(self, lr, momentum, weight_decay, optimizer_classification, lr_scheduler_step_size, lr_scheduler_gamma):
        if optimizer_classification == "SGD":
            self._optimizer_classification = optim.SGD(
                self._classification_model.parameters(),
                lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_classification == "Adam":
            self._optimizer_classification = optim.Adam(
                self._classification_model.parameters(),
                lr=lr, weight_decay=weight_decay)
        self.scheduler_classification = lr_scheduler.StepLR(self._optimizer_classification, step_size = lr_scheduler_step_size, gamma = lr_scheduler_gamma)

    def load_data(self, set_type, examples, batch_size, shuffle):
        self._dataset[set_type] = examples

        all_inputs = torch.tensor([t[0].tolist() for t in examples])
        all_labels = torch.tensor([t[1] for t in examples])
        all_ids = torch.arange(len(examples))

        self._data_loader[set_type] = DataLoader(
            TensorDataset(all_inputs, all_labels, all_ids),
            batch_size=batch_size, shuffle=shuffle)

    def pretrain_epoch(self):
        pbar = tqdm(self._data_loader['train'], desc='Training Epoch')
        self.global_epoch += 1
        for batch_idx, batch in enumerate(pbar):
            self._optimizer_classification.zero_grad()
            self.global_iter += 1
            inputs, labels, ids = tuple(t.to('cuda') for t in batch)
            logits = self._classification_model(inputs)
            loss_classification = self.criterion(logits, labels)
            loss_classification.backward()
            self._optimizer_classification.step()
            if self.global_iter % 20 == 0:
                pbar.write('[{}] loss: {:.3F} '.format(
                    self.global_iter, loss_classification))
        self.scheduler_classification.step()
    
    def loss(self, params):
        print(params.shape)
        params_splitted_reshaped = []
        start = 0
        end = 0
        for shape in self.shapes:
            end +=  np.prod(shape)
            print(start, end)
            params_splitted_reshaped.append(params[start:end].reshape(shape))
            start = end
        inputs = self._dataset['train'][:][0].to('cuda')
        labels = self._dataset['train'][:][1].to('cuda')
        out: torch.Tensor = _stateless.functional_call(self._classification_model, \
            {n: p for n, p in zip(self.names, params_splitted_reshaped)}, inputs)
        loss = self.criterion(out, labels) * len(self._dataset['train'])
        return loss


    def calculate_inv_hessian(self):
        print("Calculating Hessian")
        Hessian = hessian(self.loss, torch.cat(tuple([_.view(-1) for _ in self._classification_model.parameters()])))
        np_Hessian = Hessian.to("cpu").numpy()/len(self._dataset['train'])
        damping_matrix = np.diag(np.full(Hessian.shape[0],0.01),0)
        damping_hessian = np_Hessian + damping_matrix
        print("calculating_inv_hessian")
        inv_hessian = np.linalg.inv(damping_hessian)
        # print("rank of np_Hessian: ", np.linalg.matrix_rank(np_Hessian))
        # inv_hessian = np.linalg.inv(np_Hessian)
        print("finished calculating_inv_hessian")
        return inv_hessian



    def save_checkpoint_classification(self, file_path, silent=True):
        model_states = {'classification': self._classification_model.state_dict(), }
        optim_states = {'classification': self._optimizer_classification.state_dict(), }
        states = {'epoch': self.global_epoch,
                  'model_states': model_states,
                  'optim_states': optim_states}
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print(
                "=> saved checkpoint '{}' (iter {})".format(
                    file_path, self.global_iter))
    
    def evaluate(self, set_type):
        self._classification_model.eval()

        preds_all, labels_all = [], []
        for batch in tqdm(self._data_loader[set_type],
                          desc="Evaluating {} set".format(set_type)):
            batch = tuple(t.to('cuda') for t in batch)
            inputs, labels, _ = batch

            with torch.no_grad():
                logits = self._classification_model(inputs)

            preds = torch.argmax(logits, dim=1)
            preds_all.append(preds)
            labels_all.append(labels)

        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        return torch.sum(preds_all == labels_all).item() / labels_all.shape[0]
    
    def load_checkpoint_classification(self, file_path):
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path, map_location='cuda')
            self.global_epoch = checkpoint['epoch']
            self._classification_model.load_state_dict(checkpoint['model_states']['classification'])
            print(
                "=> loaded checkpoint '{} (epoch {})'".format(
                    file_path, self.global_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
        return self._classification_model
 
