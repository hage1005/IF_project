
import os
from tqdm import tqdm
import copy

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from torchvision import models

from .magic_module import MagicModule
import wandb

EPSILON = 1e-5


class FenchelSolver:
    def __init__(
            self,
            x_test,
            y_test,
            classification_model,
            influence_model,
            pretrained=True,
            softmax_temp=1.,
            train_classification_till_converge=False):

        self._influence_model = influence_model
        self._classification_model = classification_model

        self._optimizer_theta3 = None

        self._dataset = {}
        self._data_loader = {}

        self._weights = None
        self._w_decay = None

        self._softmax_temp = softmax_temp

        self.global_iter = 0
        self.global_epoch = 0

        self.x_test = x_test.to('cuda')
        self.y_test = y_test.to('cuda')

        self.train_classification_till_converge = train_classification_till_converge

    def init_weights(self, n_examples, w_init, w_decay):
        self._weights = torch.tensor(
            [w_init] * n_examples, requires_grad=True).to('cuda')
        self._w_decay = w_decay

    def load_data(self, set_type, examples, batch_size, shuffle):
        self._dataset[set_type] = examples

        all_inputs = torch.tensor([t[0].tolist() for t in examples])
        all_labels = torch.tensor([t[1] for t in examples])
        all_ids = torch.arange(len(examples))

        self._data_loader[set_type] = DataLoader(
            TensorDataset(all_inputs, all_labels, all_ids),
            batch_size=batch_size, shuffle=shuffle)

    def get_optimizer(self, classification_lr, influence_lr, momentum, weight_decay):
        self._optimizer_theta3 = optim.SGD(
            self._classification_model.parameters(),
            lr=classification_lr, momentum=momentum, weight_decay=weight_decay)
        self._optimizer_theta1 = optim.SGD(
            self._influence_model.parameters(),
            lr=influence_lr, momentum=momentum, weight_decay=weight_decay)

    def pretrain_epoch(self):
        self.train_epoch(is_pretrain=True)

    def train_epoch(self, is_pretrain=False):
        self.global_epoch += 1
        criterion = nn.CrossEntropyLoss(reduction='none')
        pbar = tqdm(self._data_loader['train'], desc='Training Epoch')

        for batch in pbar:
            inputs, labels, ids = tuple(t.to('cuda') for t in batch)
            self.global_iter += 1

            weights = softmax_normalize(
                self._weights[ids].detach(),
                temperature=self._softmax_temp)

            weights_variance = torch.var(weights)
            wandb.log({'weights_variance': weights_variance})
            self._optimizer_theta1.zero_grad()

            loss_influence = torch.sum(self._influence_model(inputs, labels).flatten(
            ) * weights) - torch.mean(self._influence_model(inputs, labels).flatten())
            # loss_influence = torch.log(loss_influence+10)
            loss_influence.backward()  # theta 1 update, todo: does weights change?
            self._optimizer_theta1.step()

            if is_pretrain:
                weights = linear_normalize(
                    torch.ones(inputs.shape[0]).to('cuda'))
            else:
                weights = softmax_normalize(
                    self._get_weights(batch),
                    temperature=self._softmax_temp)  # theta 2 update

            self._optimizer_theta3.zero_grad()
            lossHistory = []
            while True:
                logits = self._classification_model(inputs)
                loss_classification = criterion(logits, labels)
                loss_classification = torch.sum(
                    loss_classification * weights.data)
                loss_classification.backward()  # theta 3 update
                self._optimizer_theta3.step()
                if loss_classification.isnan() or not self.train_classification_till_converge:
                    break
                lossHistory.append(loss_classification)

                # compare the variance of last 5 loss
                if len(lossHistory) > 5:
                    if torch.var(torch.stack(lossHistory[-5:])).item() < 0.1:
                        break

            wandb.log({'uniform_minus_weighted_influence': - \
                      loss_influence, 'classification_loss': loss_classification})
            if self.global_iter % 200 == 0:
                pbar.write('[{}] loss_influence: {:.3f}, loss: {:.3F} '.format(
                    self.global_iter, loss_influence, loss_classification))

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
        optimizer_hparams = self._optimizer_theta3.state_dict()[
            'param_groups'][0]
        optimizer_tmp = optim.SGD(
            model_tmp.parameters(),
            lr=optimizer_hparams['lr'],
            momentum=optimizer_hparams['momentum'],
            weight_decay=optimizer_hparams['weight_decay'])

        for i in range(batch_size):
            model_tmp.load_state_dict(self._classification_model.state_dict())
            optimizer_tmp.load_state_dict(self._optimizer_theta3.state_dict())

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
        # for step, batch in enumerate(self._data_loader['train']):
        batch = (t.to('cuda') for t in batch)
        train_inputs, train_labels, _ = batch
        batch_size = train_labels.shape[0]

        if weights.grad is not None:
            weights.grad.zero_()
        magic_model.eval()  # batchnorm will give error if we don't do this
        test_logits = magic_model(self.x_test)  # this part is magic_module
        test_loss = criterion(test_logits,
                              self.y_test.long())  # the third term

        weights_tmp = softmax_normalize(
            weights,
            temperature=self._softmax_temp)
        infTerm = test_loss - \
            torch.sum(self._influence_model(inputs, labels).squeeze().detach() * weights_tmp)

        weights_grad = torch.autograd.grad(
            infTerm, weights, retain_graph=True)[0]
        weights_grad_list.append(weights_grad)

        weights_grad = sum(weights_grad_list)

        self._weights[ids] = weights.data / self._w_decay - weights_grad
        self._weights[ids] = torch.max(self._weights[ids], torch.ones_like(
            self._weights[ids]).fill_(EPSILON))

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

    def save_checkpoint(self, file_path, silent=True):
        model_states = {'net': self._influence_model.state_dict(), }
        optim_states = {'optim': self._optimizer_theta1.state_dict(), }
        states = {'epoch': self.global_epoch,
                  'model_states': model_states,
                  'optim_states': optim_states}
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print(
                "=> saved checkpoint '{}' (iter {})".format(
                    file_path, self.global_iter))

    def load_checkpoint(self, file_path):
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path,
                                    map_location='cuda:{}'.format(self.gpu))
            self.global_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print(
                "=> loaded checkpoint '{} (epoch {})'".format(
                    file_path, self.global_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))


def linear_normalize(weights):
    weights = torch.max(weights, torch.zeros_like(weights))
    if torch.sum(weights) > 1e-8:
        return weights / torch.sum(weights)
    return torch.zeros_like(weights)


def softmax_normalize(weights, temperature):
    return nn.functional.softmax(weights / temperature, dim=0)
