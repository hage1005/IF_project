
import os
from tqdm import tqdm
import copy

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from torchvision import models

from .magic_module import MagicModule

from .model import Net
EPSILON = 1e-5


class ImageClassifier:
    def __init__(self, x_test, y_test, model, influence_model, pretrained=True,
                 baseline=False, softmax_temp=1.):
                 
        self._influence_model = influence_model
        # self._model = models.resnet34(pretrained=pretrained).to('cuda')
        self._model = model

        self._optimizer_theta3 = None

        self._dataset = {}
        self._data_loader = {}

        self._weights = None
        self._w_decay = None

        self._baseline = baseline

        self._softmax_temp = softmax_temp

        self.global_iter = 0
        self.global_epoch = 0

        self.x_test = x_test.to('cuda')
        self.y_test = y_test.to('cuda')

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
    
    def get_optimizer(self, learning_rate, momentum, weight_decay):
        self._optimizer_theta3 = optim.SGD(
            self._model.parameters(),
            lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self._optimizer_theta1 = optim.SGD(
            self._influence_model.parameters(),
            lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    def pretrain_epoch(self):
        self.train_epoch(is_pretrain=True)

    def train_epoch(self, is_pretrain=False):
        self.global_epoch += 1
        criterion = nn.CrossEntropyLoss(reduction='none')
        pbar = tqdm(self._data_loader['train'], desc='Training Epoch')

        for batch in pbar:
            inputs, labels, ids = tuple(t.to('cuda') for t in batch)
            self.global_iter += 1
            
            weights= softmax_normalize(
                self._weights[ids].detach(),
                temperature=self._softmax_temp)
            self._optimizer_theta1.zero_grad()
            # concatenate inputs and labels
            # inputs_labels = torch.cat([inputs, labels.unsqueeze(1)], dim=1)
            loss_F = - torch.sum(torch.mean( self._influence_model(inputs, labels) - self._influence_model(inputs, labels).flatten() * weights))
            loss_F = torch.log(loss_F+10) 
            loss_F.backward() #theta 1 update, todo: does weights change?
            self._optimizer_theta1.step()

            if is_pretrain:
                weights = linear_normalize(
                    torch.ones(inputs.shape[0]).to('cuda'))
            else:
                weights = softmax_normalize(
                    self._get_weights(batch),
                    temperature=self._softmax_temp) #theta 2 update

            self._optimizer_theta3.zero_grad()
            lossHistory = []
            while 1:
                logits = self._model(inputs)
                loss = criterion(logits, labels)
                loss = torch.sum(loss * weights.data)
                loss.backward() #theta 3 update
                self._optimizer_theta3.step()
                lossHistory.append(loss)
                if loss.isnan():
                    break
                # compare the variance of last 5 loss
                if len(lossHistory) > 5:
                    if torch.var(torch.stack(lossHistory[-5:])).item() < 0.1:
                        break
                    

            if self.global_iter % 200 == 0:
                pbar.write('[{}] loss_F: {:.3f}, loss: {:.3F} '.format(
                            self.global_iter, loss_F, loss))

    def _get_weights(self, batch, no_update = False): # batch is from train set
        self._model.eval()

        inputs, labels, ids = tuple(t.to('cuda') for t in batch)
        batch_size = inputs.shape[0]

        weights = self._weights[ids]
        if no_update:
            return weights
        magic_model = MagicModule(self._model)
        criterion = nn.CrossEntropyLoss()

        model_tmp = copy.deepcopy(self._model)
        optimizer_hparams = self._optimizer_theta3.state_dict()['param_groups'][0]
        optimizer_tmp = optim.SGD(
            model_tmp.parameters(),
            lr=optimizer_hparams['lr'],
            momentum=optimizer_hparams['momentum'],
            weight_decay=optimizer_hparams['weight_decay'])

        for i in range(batch_size):
            model_tmp.load_state_dict(self._model.state_dict())
            optimizer_tmp.load_state_dict(self._optimizer_theta3.state_dict())

            model_tmp.zero_grad()

            if i > 0:
                l, r, t = i - 1, i + 1, 1
            else:
                l, r, t = i, i + 2, 0

            logits = model_tmp(inputs[l:r])[t:t+1] # same as logits = model_tmp(inputs[i:i+1]) ?
            loss = criterion(logits, labels[i:i+1]) #Not magic_module
            loss.backward()
            optimizer_tmp.step()

            deltas = {}
            for (name, param), (name_tmp, param_tmp) in zip(
                    self._model.named_parameters(),
                    model_tmp.named_parameters()):
                assert name == name_tmp
                deltas[name] = weights[i] * (param_tmp.data - param.data)
            magic_model.update_params(deltas) #theta prime, make it the same as model_tmp

        # TODO remove this
        weights_grad_list = []
        # for step, batch in enumerate(self._data_loader['train']):
        batch = (t.to('cuda') for t in batch)
        train_inputs, train_labels, _ = batch
        batch_size = train_labels.shape[0]

        if weights.grad is not None:
            weights.grad.zero_()
        test_logits = magic_model(self.x_test) #this part is magic_module
        test_loss = criterion(test_logits, self.y_test.long()) #the third term
        # inputs_labels = torch.cat([train_inputs, train_labels.unsqueeze(1)], dim=1)
        weights = softmax_normalize(
                    weights,
                    temperature=self._softmax_temp)
        infTerm = test_loss - torch.sum(self._influence_model(inputs,labels).squeeze().detach() * weights)  # the inf term, sum? todo: check this
        print("infTerm", infTerm)
        # test_loss = test_loss * float(batch_size) / float(
        #     len(self._dataset['dev'])) #not using dev but train batch

        weights_grad = torch.autograd.grad(
            infTerm, weights, retain_graph=True)[0]
        weights_grad_list.append(weights_grad)
#       weight_grad += self._influence_model(train_inputs)
        weights_grad = sum(weights_grad_list)

        print("weights_grad average before clip", sum(weights_grad) / len(weights_grad))
        # weights_grad = torch.min(weights_grad, torch.ones_like(weights_grad))
        # weights_grad = torch.max(weights_grad, -torch.ones_like(weights_grad))


        self._weights[ids] = weights.data / self._w_decay - weights_grad
        self._weights[ids] = torch.max(self._weights[ids], torch.ones_like(
            self._weights[ids]).fill_(EPSILON))
        
        if self._weights[ids].data[0] == torch.inf or self._weights[ids].data[0].isnan():
            self._weights[ids] = torch.zeros_like(self._weights[ids])
        return self._weights[ids].data

    def evaluate(self, set_type):
        self._model.eval()

        preds_all, labels_all = [], []
        for batch in tqdm(self._data_loader[set_type],
                          desc="Evaluating {} set".format(set_type)):
            batch = tuple(t.to('cuda') for t in batch)
            inputs, labels, _ = batch

            with torch.no_grad():
                logits = self._model(inputs)

            preds = torch.argmax(logits, dim=1)
            preds_all.append(preds)
            labels_all.append(labels)

        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        return torch.sum(preds_all == labels_all).item() / labels_all.shape[0]
        
    def save_checkpoint(self, file_path, silent=True):
        model_states = {'net':self._influence_model.state_dict(),}
        optim_states = {'optim':self._optimizer_theta1.state_dict(),}
        states = {'epoch':self.global_epoch,
                'model_states':model_states,
                'optim_states':optim_states}
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, file_path):
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path, map_location='cuda:{}'.format(self.gpu))
            self.global_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (epoch {})'".format(file_path, self.global_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

def linear_normalize(weights):
    weights = torch.max(weights, torch.zeros_like(weights))
    if torch.sum(weights) > 1e-8:
        return weights / torch.sum(weights)
    return torch.zeros_like(weights)


def softmax_normalize(weights, temperature):
    return nn.functional.softmax(weights / temperature, dim=0)
