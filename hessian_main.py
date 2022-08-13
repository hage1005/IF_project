
import argparse
import os
import random
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
import copy

import numpy as np
import torch

from torchvision import models, transforms
from src.data_utils.MnistDataset import MnistDataset
from src.utils.utils import save_json
from src.data_utils.Cifar10Dataset import Cifar10Dataset
from src.solver.hessian_solver import hessianSolver
from src.modeling.classification_models import CnnCifar, MNIST_1, MNIST_LogisticRegression
from src.modeling.influence_models import Net_IF, MNIST_IF_1
from torch.autograd.functional import hessian
from torch.nn.utils import _stateless
from torch.nn import CrossEntropyLoss 
import yaml
# YAMLPath = 'src/config/MNIST/default.yaml'
YAMLPath = 'src/config/MNIST/single_test/exp/MNIST_1_100each.yaml'
# method = "Identity"
method = "Percy"
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_single_from_dataset(dataset, idx):
        x, y = dataset[idx]
        x = x.unsqueeze(0)
        y = torch.LongTensor([y])
        return x, y

def main(args):
    # set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

    

    if args.dataset_name == 'cifar10':
        Dataset = Cifar10Dataset
        trans = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    elif args.dataset_name == 'mnist':
        Dataset = MnistDataset
        trans = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Normalize(
            (0.1307,),(0.3081,))
            ])
    else:
        raise NotImplementedError()

    class_label_dict = Dataset.get_class_label_dict()
    train_classes = [class_label_dict[c] for c in args.train_classes]
    ImageDataset = Dataset(args.dev_original_folder, args.dev_transformed_folder, args.test_original_folder, args.test_transformed_folder, train_classes, trans, args.num_per_class)

    if args.classification_model == 'MNIST_1':
        classification_model = MNIST_1(args._hidden_size_classification, args._num_class).to('cuda')
    elif args.classification_model == 'MNIST_LogisticRegression':
        classification_model = MNIST_LogisticRegression(args._num_class).to('cuda')
    else:
        raise NotImplementedError()
    
    train_dataset, train_dataset_no_transform = ImageDataset.get_train()
    dev_dataset, dev_dataset_no_transform = ImageDataset.get_dev()

    x_dev, y_dev = get_single_from_dataset(
        dev_dataset, args.dev_id_num)
    train_dataset_size = len(train_dataset)

    pretrain_ckpt_path = os.path.join(args._ckpt_dir, args.classification_model, args._pretrain_ckpt_name)
    inv_hessian_path = os.path.join(args._ckpt_dir, args.classification_model, "numpy_inv_hessian_" + args._pretrain_ckpt_name)
    hessian_solver = hessianSolver(classification_model, pretrain_ckpt_path, inv_hessian_path)
    hessian_solver.load_data('train', train_dataset, 32, shuffle= True)
    hessian_solver.load_data('dev', dev_dataset, 32, shuffle= False)

    if not os.path.exists(pretrain_ckpt_path):
        print("Pretrain ckpt not found, training from scratch")
        hessian_solver.get_optimizer_classification(
        args.classification_lr,
        args.classification_momentum,
        args.classification_weight_decay)

        for epoch in range(20):
            hessian_solver.pretrain_epoch()
            dev_acc = hessian_solver.evaluate('dev')
            print('Pre-train Epoch {}, dev Acc: {:.4f}'.format(
                epoch, 100. * dev_acc))
            hessian_solver.save_checkpoint_classification(pretrain_ckpt_path)
    
    classification_model_pretrained = hessian_solver.load_checkpoint_classification(pretrain_ckpt_path)
    if not os.path.exists(inv_hessian_path + '.npy'):
        print("Inv Hessian not found, calculating")
        inv_hessian = hessian_solver.calculate_inv_hessian()
        np.save(inv_hessian_path, inv_hessian)
    else:
        inv_hessian = np.load(inv_hessian_path + '.npy')
        
    
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

    def calculate_if(x_train, y_train, x_test, y_test, inv_Hessian):
        # test point should be alpha_test_dataset[10]

        test_loss = loss_grad_at_point(classification_model_pretrained, x_test, y_test).to("cpu").numpy()

        train_loss = loss_grad_at_point(classification_model_pretrained, x_train, y_train).to("cpu").numpy()

        if method == "Identity":
            if_score = -np.matmul(test_loss.T, train_loss)
        elif method == "Percy":
            if_score = -np.matmul(np.matmul(test_loss.T, inv_Hessian), train_loss)

        return if_score
    influences = []

    classification_model_pretrained.to('cpu')
    for i in tqdm(range(train_dataset_size)):
        if_score = calculate_if(train_dataset[i][0], train_dataset[i][1], x_dev, y_dev, inv_hessian)
        influences.append(if_score)

    result = {}
    influences = np.array(influences)
    helpful = np.argsort(influences)
    harmful = helpful[::-1]
    result["helpful"] = helpful[:500].tolist()
    result["harmful"] = harmful[:500].tolist()
    result["influence"] = influences.tolist()
    json_path = os.path.join(
        "outputs",
        args.dataset_name,
        f"{method}_{args.dataset_name}_devId_{args.dev_id_num}.json")
    save_json(result, json_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--YAMLPath", type=str)
    args, unknown = parser.parse_known_args()
    if args.YAMLPath:
        YAMLPath = args.YAMLPath
    with open(YAMLPath) as file:
        config = yaml.safe_load(file)
    main(Struct(**config))