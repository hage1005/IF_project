
import argparse
import os
import random
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
import copy

import numpy as np
import torch

from src.utils.Path_IF import Path_IF
from src.utils.utils import get_single_from_dataset
from torchvision import models, transforms
from src.data_utils.MnistDataset import MnistDataset
from src.utils.utils import save_json
from src.data_utils.Cifar10Dataset import Cifar10Dataset
from src.solver.hessian_solver import hessianSolver
from src.modeling.classification_models import get_classification_model
from src.modeling.influence_models import Net_IF, MNIST_IF_1
from torch.autograd.functional import hessian
from torch.nn.utils import _stateless
from torch.nn import CrossEntropyLoss 
from src.data_utils.index import get_dataset

import yaml
# YAMLPath = 'src/config/MNIST/default.yaml'
# YAMLPath = 'src/config/MNIST/single_test/exp/MNIST_2_oneAndSevenAll.yaml'
YAMLPath = 'src/config/MNIST/single_test/exp/MNIST_1_100each/test_id_1/fenchel.yaml'

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main(args):
    Path = Path_IF(args)
    # set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

    
    Dataset, _ = get_dataset(args.dataset_name)

    class_label_dict = Dataset.get_class_label_dict()
    train_classes = [class_label_dict[c] for c in args.train_classes]
    ImageDataset = Dataset(args.dev_original_folder, args.num_per_class)

    classification_model = get_classification_model(args.classification_model, args._num_class)
    
    train_dataset = ImageDataset.get_data("train")
    dev_dataset = ImageDataset.get_data("dev")

    x_dev, y_dev = get_single_from_dataset(
        dev_dataset, args.dev_id_num)
    train_dataset_size = len(train_dataset)

    hessian_solver = hessianSolver(classification_model, Path.pretrain_ckpt_path, Path.inv_hessian_path)
    hessian_solver.load_data('train', train_dataset, 32, shuffle= True)
    hessian_solver.load_data('dev', dev_dataset, 32, shuffle= False)

    if not os.path.exists(Path.pretrain_ckpt_path):
        print("Pretrain ckpt not found, training from scratch")
        hessian_solver.get_optimizer_classification(
        args.classification_lr,
        args.classification_momentum,
        args.classification_weight_decay,
        args.optimizer_classification,
        args.max_checkpoint_epoch // 5,
        0.2)

        for epoch in range(args.max_checkpoint_epoch):
            hessian_solver.pretrain_epoch()
            dev_acc = hessian_solver.evaluate('dev')
            print('Pre-train Epoch {}, dev Acc: {:.4f}'.format(
                epoch, 100. * dev_acc))
        hessian_solver.save_checkpoint_classification(Path.pretrain_ckpt_path)
    
    classification_model_pretrained = hessian_solver.load_checkpoint_classification(Path.pretrain_ckpt_path)
    if not os.path.exists(Path.inv_hessian_path + '.npy') and args.compare_with_inv_hessian:
        print("Inv Hessian not found, calculating")
        inv_hessian = hessian_solver.calculate_inv_hessian()
        np.save(Path.inv_hessian_path, inv_hessian)
    else:
        inv_hessian = np.load(Path.inv_hessian_path + '.npy')
        
    
    def loss_grad_at_point(model, x, y):
        # w should be torch.cat(tuple([_.view(-1) for _ in model.parameters()]))
        # train_point should be alpha_train_dataset[0], a tuple of image and label
        criterion = CrossEntropyLoss()
        out = model(x)
        loss = criterion(out, torch.tensor([y]))
        loss.backward()
        grad = torch.cat(tuple([_.grad.view(-1) for _ in model.parameters()]))
        for p in model.parameters():
            p.grad = None
        return grad

    def calculate_identity(x_train, y_train, x_test, y_test, inv_Hessian):
        # test point should be alpha_test_dataset[10]

        test_loss = loss_grad_at_point(classification_model_pretrained, x_test, y_test).to("cpu").numpy()

        train_loss = loss_grad_at_point(classification_model_pretrained, x_train, y_train).to("cpu").numpy()

        if_score_identity = -np.matmul(test_loss.T, train_loss)

        return if_score_identity

    def calculate_percy(x_train, y_train, x_test, y_test, inv_Hessian):
        # test point should be alpha_test_dataset[10]

        test_loss = loss_grad_at_point(classification_model_pretrained, x_test, y_test).to("cpu").numpy()

        train_loss = loss_grad_at_point(classification_model_pretrained, x_train, y_train).to("cpu").numpy()

        if_score_percy = -np.matmul(np.matmul(test_loss.T, inv_Hessian), train_loss)

        return if_score_percy
    
    classification_model_pretrained.to('cpu')
    if args.compare_with_identity:
        influences_identity = []
        for i in tqdm(range(train_dataset_size)):
            if_score_identity, if_score_percy = calculate_identity(train_dataset[i][0], train_dataset[i][1], x_dev, y_dev, inv_hessian)
            influences_identity.append(if_score_identity)

    if args.compare_with_inv_hessian:
        influences_percy = []
        for i in tqdm(range(train_dataset_size)):
            if_score_identity, if_score_percy = calculate_percy(train_dataset[i][0], train_dataset[i][1], x_dev, y_dev, inv_hessian)
            influences_percy.append(if_score_identity)

    x = np.array(influences_identity)
    y = np.array(influences_percy)
    corr = round(np.corrcoef(x,y)[0,1],3)

    Path.save_percy_influence(influences_percy)

    Path.save_identitiy_influence(influences_identity)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--YAMLPath", type=str)
    args, unknown = parser.parse_known_args()
    if args.YAMLPath:
        YAMLPath = args.YAMLPath
    with open(YAMLPath) as file:
        config = yaml.safe_load(file)
    main(Struct(**config))