import os
import numpy as np

import torch
from torch.autograd import grad
import torch.nn as nn
from torch.nn import functional as F
from .utils import display_progress, grad_z, load_json

import warnings

from .model import Net
warnings.simplefilter("ignore")


def get_ordered_checkpoint_list(path):
    all_files = os.listdir(os.path.join(path))
    all_checkpoints = [f for f in all_files if f.endswith('0')]
    all_checkpoints.sort(key=int)
    print('{} checkpoints found'.format(len(all_checkpoints)))
    return all_checkpoints


def tracin_cp(ckpt_dir, x_test, y_test, train_loader, device):
    """
    dic.keys(): path, network_config, suffix, lr, batchsize, use_last_layer
    """

    train_dataset_size = len(train_loader.dataset)
    dic = load_json(os.path.join(ckpt_dir, 'config_dic'))

    ckpt_iter = 0
    influences = [0.0 for _ in range(train_dataset_size)]
    ordered_checkpoint_list = get_ordered_checkpoint_list(ckpt_dir)

    # add \nabla loss(z_i, beta) * \nabla loss(z_test, beta) for every z_i
    # over all steps
    for checkpoint_name in ordered_checkpoint_list:
        # load model at this checkpoint
        model = Net()
        checkpoint = torch.load(os.path.join(ckpt_dir, checkpoint_name),
                                map_location='cpu')
        model.load_state_dict(checkpoint['model_states']['net'])
        model = model.to(device)

        # print checkpoint iteration
        ckpt_interval = int(checkpoint_name) - ckpt_iter
        ckpt_iter = int(checkpoint_name)
        print('checkpoint: {}'.format(ckpt_iter))

        # compute gradient at z_test
        grad_test = grad_z(x_test, y_test, model, device)

        # compute gradient at z_i for each i
        for i in range(train_dataset_size):
            input, label = train_loader.dataset[i][0], train_loader.dataset[i][1]
            input, label = train_loader.collate_fn(
                [input]), train_loader.collate_fn([label])

            grad_train = grad_z(input, label, model, device)

            # add gradient dot product to influences
            grad_dot_product = sum(
                [torch.sum(k * j).data for k, j in zip(grad_train, grad_test)]).cpu().numpy()
            influences[i] += grad_dot_product * dic['lr'] * \
                dic['batch_size'] * ckpt_interval / train_dataset_size

            display_progress("Calc. grad dot product: ", i, train_dataset_size)

    influences = np.array(influences)
    harmful = np.argsort(influences)
    helpful = harmful[::-1]
    return influences.tolist(), harmful.tolist(), helpful.tolist()
