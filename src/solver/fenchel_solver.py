
import os
from tqdm import tqdm
import copy

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from torchvision import models

from .magic_module import MagicModule
from torch.optim import lr_scheduler

import wandb

EPSILON = 1e-5


class FenchelSolver:
    def __init__(
            self,
            x_dev,
            y_dev,
            classification_model,
            influence_model,
            is_influence_model_hashmap,
            normalize_fn_classification,
            normalize_fn_influence,
            softmax_temp=1.,
            train_classification_till_converge=False,
            clip_min_weight=True,
            ):

        self._influence_model = influence_model
        self._classification_model = classification_model

        self._optimizer_classification = None

        self._dataset = {}
        self._data_loader = {}

        self._weights = None
        self._w_decay = None

        self._softmax_temp = softmax_temp

        self.global_iter = 0
        self.global_epoch = 0

        self.x_dev = x_dev.to('cuda')
        self.y_dev = y_dev.to('cuda')

        self.train_classification_till_converge = train_classification_till_converge
        self.clip_min_weight = clip_min_weight
        self.is_influence_model_hashmap = is_influence_model_hashmap

        self.normalize_fn_classification = normalize_fn_classification
        self.normalize_fn_influence = normalize_fn_influence

        self.first_iteration_grad = None

    def init_weights(self, n_examples, w_init, w_decay):
        self._weights = torch.tensor(
            [w_init] * n_examples, requires_grad=True).to('cuda')
        self.first_iteration_grad = torch.zeros(n_examples)
        self._w_decay = w_decay

    def load_data(self, set_type, examples, batch_size, shuffle):
        self._dataset[set_type] = examples

        all_inputs = torch.tensor([t[0].tolist() for t in examples])
        all_labels = torch.tensor([t[1] for t in examples])
        all_ids = torch.arange(len(examples))

        self._data_loader[set_type] = DataLoader(
            TensorDataset(all_inputs, all_labels, all_ids),
            batch_size=batch_size, shuffle=shuffle)

    def get_optimizer_influence(self, lr, momentum, weight_decay, optimizer_influence):
        if optimizer_influence == "SGD":
            self._optimizer_influence = optim.SGD(
                self._influence_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_influence == "Adam":
            self._optimizer_influence = optim.Adam(
                self._influence_model.parameters(), lr=lr, weight_decay=weight_decay)

    def get_optimizer_classification_and_scheduler(self, lr, momentum, weight_decay, optimizer_classification, lr_scheduler_step_size, lr_scheduler_gamma):
        if optimizer_classification == "SGD":
            self._optimizer_classification = optim.SGD(
                self._classification_model.parameters(),
                lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_classification == "Adam":
            self._optimizer_classification = optim.Adam(
                self._classification_model.parameters(),
                lr=lr, weight_decay=weight_decay)
        self.scheduler_classification = lr_scheduler.StepLR(self._optimizer_classification, step_size = lr_scheduler_step_size, gamma = lr_scheduler_gamma)

    def pretrain_epoch(self):
        criterion = nn.CrossEntropyLoss(reduction='none')
        pbar = tqdm(self._data_loader['train'], desc='Training Epoch')
        for batch_idx, batch in enumerate(pbar):
            inputs, labels, ids = tuple(t.to('cuda') for t in batch) # t[0] is input, t[1] is label, t[2] is id
            self._optimizer_classification.zero_grad()
            logits = self._classification_model(inputs)
            loss_classification = criterion(logits, labels)
            loss_classification = torch.mean(loss_classification)
            loss_classification.backward()  # theta 3 update
            self._optimizer_classification.step()
        self.scheduler_classification.step()

    def train_epoch(self):
        self.global_epoch += 1
        criterion = nn.CrossEntropyLoss(reduction='none')
        pbar = tqdm(self._data_loader['train'], desc='Training Epoch')

        for batch_idx, batch in enumerate(pbar):
            inputs, labels, ids = tuple(t.to('cuda') for t in batch) # t[0] is input, t[1] is label, t[2] is id
            self.global_iter += 1

            weights = self.normalize_fn_influence(self._weights[ids].detach() )

            wandb.log({'weights_std': torch.std(weights), 'batch_idx': batch_idx, 'epoch': self.global_epoch})
            self._optimizer_influence.zero_grad()
            if self.is_influence_model_hashmap:
                IF_score = self._influence_model(ids).flatten()
            else:
                IF_score = self._influence_model(inputs, labels).flatten()
            loss_influence = torch.sum(IF_score * weights) - torch.mean(IF_score)
            # loss_influence = torch.log(loss_influence+10)
            loss_influence.backward()  # theta 1 update, todo: does weights change?
            self._optimizer_influence.step()

            weights = self.normalize_fn_classification(
                self._get_weights(batch),
                )  # theta 2 update

            self._optimizer_classification.zero_grad()
            # while True:
            logits = self._classification_model(inputs)
            loss_classification = criterion(logits, labels)
            loss_classification = torch.sum(
                loss_classification * weights.data)
            loss_classification.backward()  # theta 3 update
            self._optimizer_classification.step()
            if self.train_classification_till_converge:
                lossHistory = []
                while True:
                    loss_classification_total = 0
                    for batch_idx, batch in enumerate(self._data_loader['train']):
                        inputs, labels, ids = tuple(t.to('cuda') for t in batch) # t[0] is input, t[1] is label, t[2] is id
                        weights = self.normalize_fn_classification(self._weights[ids].detach() )
                        self._optimizer_classification.zero_grad()
                        logits = self._classification_model(inputs)
                        loss_classification = criterion(logits, labels)
                        loss_classification = torch.sum(loss_classification * weights.data)
                        loss_classification.backward()  # theta 3 update
                        loss_classification_total += loss_classification
                        self._optimizer_classification.step()
                    lossHistory.append(loss_classification_total)
                    # compare the variance of last 5 loss
                    if len(lossHistory) > 5 and torch.std(torch.stack(lossHistory[-5:])).item() < 0.05:
                        break


            wandb.log({'uniform_minus_weighted_influence': - \
                      loss_influence, 'classification_loss': loss_classification})
            if self.global_iter % 20 == 0:
                pbar.write('[{}] loss_influence: {:.3f}, loss: {:.3F} '.format(
                    self.global_iter, loss_influence, loss_classification))
            self.scheduler_classification.step()

    def _get_weights(self, batch, no_update=False):  # batch is from train set
        self._classification_model.eval()

        inputs, labels, ids = tuple(t.to('cuda') for t in batch)
        batch_size = inputs.shape[0]

        weights = self._weights[ids]
        if no_update:
            return weights
        magic_model = MagicModule(self._classification_model)
        criterion = nn.CrossEntropyLoss()

        model_tmp = copy.deepcopy(self._classification_model)
        optimizer_hparams = self._optimizer_classification.state_dict()[
            'param_groups'][0]
        if issubclass(type(self._optimizer_classification), torch.optim.Adam):
            optimizer_tmp = optim.Adam(
                model_tmp.parameters(), lr=optimizer_hparams['lr'], weight_decay=optimizer_hparams['weight_decay']
            )
        elif issubclass(type(self._optimizer_classification), torch.optim.SGD):
            optimizer_tmp = optim.SGD(
                model_tmp.parameters(),
                lr=optimizer_hparams['lr'],
                momentum=optimizer_hparams['momentum'],
                weight_decay=optimizer_hparams['weight_decay'])

        for i in range(batch_size):
            model_tmp.load_state_dict(self._classification_model.state_dict())
            optimizer_tmp.load_state_dict(self._optimizer_classification.state_dict())

            model_tmp.zero_grad()

            if i > 0:
                l, r, t = i - 1, i + 1, 1
            else:
                l, r, t = i, i + 2, 0

            # same as logits = model_tmp(inputs[i:i+1]) ?
            logits = model_tmp(inputs[l:r])[t:t + 1]
            loss = criterion(logits, labels[i:i + 1])  # Not magic_module
            loss.backward()
            optimizer_tmp.step()

            deltas = {}
            for (name, param), (name_tmp, param_tmp) in zip(
                    self._classification_model.named_parameters(),
                    model_tmp.named_parameters()):
                assert name == name_tmp
                deltas[name] = weights[i] * (param_tmp.data - param.data)
            # theta prime, make it the same as model_tmp
            magic_model.update_params(deltas)

        # TODO remove this
        weights_grad_list = []
        batch_size = labels.shape[0]

        if weights.grad is not None:
            weights.grad.zero_()
        magic_model.eval()  # batchnorm will give error if we don't do this
        dev_logits = magic_model(self.x_dev)  # this part is magic_module
        dev_loss = criterion(dev_logits,
                              self.y_dev.long())  # the third term
    
        weights_tmp = self.normalize_fn_classification(weights)

        if self.is_influence_model_hashmap:
            weighted_influence = torch.sum(self._influence_model(ids).squeeze().detach() * weights_tmp)
        else:
            weighted_influence = torch.sum(self._influence_model(inputs, labels).squeeze().detach() * weights_tmp)
        infTerm = dev_loss - weighted_influence
            
        weights_grad = torch.autograd.grad(
            infTerm, weights, retain_graph=True)[0]
        weights_grad_list.append(weights_grad)

        weights_grad = sum(weights_grad_list)
        wandb.log({'dev_loss': dev_loss, 'weighted_influence': weighted_influence, 'weights_grad_std': torch.std(weights_grad), 'weights_grad_mean_abs': torch.mean(torch.abs(weights_grad))})

        self._weights[ids] = weights.data / self._w_decay - weights_grad

        if self.clip_min_weight:
            self._weights[ids] = torch.max(self._weights[ids], torch.ones_like(
                self._weights[ids]).fill_(EPSILON))

        if self.global_epoch == 1:
            self.first_iteration_grad[ids.cpu()] = weights_grad.cpu()

        if self._weights[ids].data[0] == torch.inf or self._weights[ids].data[0].isnan(
        ):
            self._weights[ids] = torch.zeros_like(self._weights[ids])
        return self._weights[ids].data

    # TODO Get rid of evaulate and test dataloader
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

    def load_checkpoint_classification(self, file_path):
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path, map_location='cuda')
            self._classification_model.load_state_dict(checkpoint['model_states']['classification'])
            print(
                "=> loaded checkpoint '{} (epoch {})'".format(
                    file_path, self.global_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))


