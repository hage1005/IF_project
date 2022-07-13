import os
import torch
import json
import argparse
# from kewen.IF_project.fenchel import fenchel_calc

from utils.utils import save_json
from data.dataset import return_data
from src.baseline.tracIn import tracin_cp
from src.baseline.hvp_grad import calc_influence_single

def main(args):
    device = torch.device("cuda:{}".format(args.gpu))
    print('-'*50)

    train_loader, test_loader = return_data(batch_size=1)
    x_test, y_test = test_loader.dataset[args.test_id_num]
    x_test, y_test = test_loader.collate_fn([x_test]), test_loader.collate_fn([y_test])

    if args.method == 'tracIn':
        influences, harmful, helpful = tracin_cp(args.ckpt_dir, x_test, y_test, train_loader, device)
    elif args.method == 'hvp':
         influences, harmful, helpful = calc_influence_single(args.ckpt_dir, x_test, y_test, train_loader, device, args.recursion_depth, args.r_averaging)

    else:
        raise NotImplementedError()
    
    result = {}
    result["helpful"] = helpful[:500]
    result["harmful"] = harmful[:500]
    result["influence"] = influences
    json_path = os.path.join(args.output_path, args.method+f"_test_id_{args.test_id_num}.json")
    save_json(result, json_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE-TracIn self influences')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')

    # model
    parser.add_argument('--ckpt_dir', default='../checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_iter', type=int, default=1500000, help='checkpoint iteration')

    # dataset
    parser.add_argument('--method', type=str, default="fenchel", help="use tracIn or hvp or fenchel")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    
    # Target
    parser.add_argument('--test_id_num', type=int, default=0, help="id of test example in testloader")

    #TracIn param
    parser.add_argument('--r_averaging', type=int, default=1, help="1 is the default value from repo")
    parser.add_argument('--recursion_depth', type=int, default=50000, help="1 is the default value from repo")

    # Output
    parser.add_argument('--output_path', type=str, default="../outputs", help="where to put images")
    args = parser.parse_args()
    main(args)