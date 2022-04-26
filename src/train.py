
import argparse
import os
import numpy as np
import torch
from solver import Solver
from dataset import return_data
from utils import save_json

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    '''
    define training data
    '''
    train_loader, test_loader = return_data(args.batch_size)

    '''
    define model
    '''
    solver = Solver(args)

    '''
    train the model
    '''
    config_dic = {
            'lr':args.lr,
            'batch_size':args.batch_size
        }
    file_path = os.path.join(args.ckpt_dir, 'config_dic')
    save_json(config_dic, file_path)
    
    solver.train(train_loader)

    '''test the model'''
    solver.test(test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--gpu', default=1, type=int, help='gpu index')
    parser.add_argument('--max_iter', default=1e5, type=float, help='maximum training iteration')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--ckpt_dir', default='../checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')
    parser.add_argument('--display_step', default=2000, type=int, help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    #TracIn
    parser.add_argument('--save_step', default=10000, type=int, help='number of iterations after which a checkpoint is saved')

    args = parser.parse_args()

    main(args)