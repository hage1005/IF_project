import argparse
from sklearn.utils import shuffle
from tqdm import tqdm
import copy

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from torchvision import models
from src.dataset import return_data
from src.fenchel_classifier import ImageClassifier

from src.magic_module import MagicModule

EPSILON = 1e-5

def main(args):
    train_loader, test_loader = return_data(args.batch_size)
    x_test, y_test = test_loader.dataset[args.test_id_num]
    x_test, y_test = test_loader.collate_fn([x_test]), test_loader.collate_fn([y_test])

    fenchen_classifier = ImageClassifier(x_test = x_test, y_test = y_test)

    fenchen_classifier.load_data("train", train_loader.dataset, args.batch_size, shuffle=False)
    fenchen_classifier.load_data("test", test_loader.dataset, args.batch_size, shuffle=False)
    

    fenchen_classifier.init_weights(
        n_examples=len(train_loader.dataset),
        w_init=args.w_init,
        w_decay=args.w_decay)

    fenchen_classifier.get_optimizer(args.image_lr, args.image_momentum, args.image_weight_decay)

    for epoch in range(args.max_epoch):
        fenchen_classifier.train_epoch()
        fenchen_classifier.save_checkpoint(args.ckpt_dir + args.ckpt_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--gpu', default=1, type=int, help='gpu index')
    parser.add_argument('--max_epoch', default=10, type=float, help='maximum training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--ckpt_dir', default='../checkpoints/fenchel/', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')
    parser.add_argument('--display_step', default=2000, type=int, help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    parser.add_argument('--image_lr', default=1e-3, type=float)
    parser.add_argument('--image_momentum', default=0.9, type=float)
    parser.add_argument('--image_weight_decay', default=0.01, type=float)

    # weighting
    parser.add_argument("--w_decay", default=10., type=float)
    parser.add_argument("--w_init", default=0., type=float)

    parser.add_argument('--test_id_num', type=int, default=0, help="id of test example in testloader")
    args = parser.parse_args()

    # Target

    main(args)