import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import yaml

from src.utils.utils import save_json
from src.data_utils.GMM1D import get_GMM1D_data
from src.data_utils.GMM2D import get_GMM2D_data
from src.solver.fenchel_solver import FenchelSolver

from src.modeling.influence_models import Linear_IF
from src.modeling.classification_models import LogisticRegression

import wandb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from types import SimpleNamespace

os.chdir('/home/xiaochen/kewen/IF_project')
YAMLPath = 'src/config/GMM2D/exp02.yaml'
# YAMLPath = 'src/config/GMM1D/exp01.yaml'


def main(args):
     # set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True
    
    if args.dataset_name == 'GMM1D':
        train_dataset, test_dataset = get_GMM1D_data()
    elif args.dataset_name == 'GMM2D':
        train_dataset, test_dataset = get_GMM2D_data()
    else:
        raise NotImplementedError()

    x_test, y_test = test_dataset[args.test_id_num:args.test_id_num + 1]
    # x_test, y_test = torch.FloatTensor([[x_test]]), torch.IntTensor([y_test])
    wandb.run.summary['x_test'] = str(x_test.tolist())
    wandb.run.summary['y_test'] = str(y_test.tolist())
    if args.classification_model == 'LogisticRegression':
        classification_model = LogisticRegression(
            args._input_dim, args._num_class).to('cuda')
    else:
        raise NotImplementedError()

    if args.influence_model == 'Linear_IF':
        influence_model = Linear_IF(
            args._input_dim, args._num_class).to('cuda')
    else:
        raise NotImplementedError()

    fenchen_classifier = FenchelSolver(
        x_test,
        y_test,
        classification_model=classification_model,
        influence_model=influence_model,
        softmax_temp=args.softmax_temp)

    fenchen_classifier.load_data(
        "train",
        train_dataset,
        args.batch_size,
        shuffle=True)
    fenchen_classifier.load_data(
        "test",
        test_dataset,
        args.batch_size,
        shuffle=False)

    fenchen_classifier.init_weights(
        n_examples=len(train_dataset),
        w_init=args.influence_weight_init,
        w_decay=args.influence_weight_decay)

    fenchen_classifier.get_optimizer(
        args.classification_lr,
        args.influence_lr,
        args.classification_momentum,
        args.classification_weight_decay)

    for epoch in range(args.max_epoch):

        fenchen_classifier.train_epoch()
        fenchen_classifier.save_checkpoint(args._ckpt_dir + args._ckpt_name)

        result = {}
        train_dataset_size = len(train_dataset)
        influences = [0.0 for _ in range(train_dataset_size)]
        # TODO compute by batch
        for i in tqdm(range(train_dataset_size)):
            x, y = train_dataset[i:i + 1][0], train_dataset[i:i + 1][1]
            influences[i] = influence_model(x.cuda(), y.cuda()).cpu().item()
        influences = np.array(influences)
        helpful = np.argsort(influences)
        harmful = helpful[::-1]

        result["helpful"] = helpful[:500].tolist()
        result["harmful"] = harmful[:500].tolist()
        result["influence"] = influences.tolist()
        json_path = os.path.join(
            "outputs",
            args.dataset_name,
            f"IF_{args.dataset_name}_testid_{args.test_id_num}_epoch_{epoch}.json")
        save_json(result, json_path)

        # wandb doesn't support legend somehow
        # helpful_data = [ [train_dataset[x][0],train_dataset[x][1].item()] for x in helpful[:num_to_plot]]
        # table = wandb.Table(data=helpful_data, columns = ["x", "class"])
        # wandb.log({f"helpful_data_epoch{epoch}" : wandb.plot.scatter(table,
        #                             "x", "class")})
        wandb.log({'total_weight_std': torch.std(
            fenchen_classifier._weights).item()})

        """Start Ploting"""
        if epoch % 2:
            continue
        num_to_plot = 100
        helpful_data_x = [train_dataset[x][0] for x in helpful[:num_to_plot]]
        helpful_data_y = [train_dataset[x][1].item()
                          for x in helpful[:num_to_plot]]

        if args.dataset_name == 'GMM1D':
            x_axis = [x[0].item() for x in helpful_data_x]
            y_axis = [0] * num_to_plot
            classes = ['mean -1', 'mean 1']
        elif args.dataset_name == 'GMM2D':
            x_axis = [x[0].item() for x in helpful_data_x]
            y_axis = [x[1].item() for x in helpful_data_x]
            classes = ['mean(-1,-1)', 'mean(1,1)']

        colours = ListedColormap(['r', 'b'])
        scatter = plt.scatter(x_axis, y_axis, c=helpful_data_y, cmap=colours)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        wandb.log({f"helpful_data_epoch": plt})
        plt.clf()

        harmful_data_x = [train_dataset[x][0] for x in harmful[:num_to_plot]]
        harmful_data_y = [train_dataset[x][1].item()
                          for x in harmful[:num_to_plot]]

        if args.dataset_name == 'GMM1D':
            x_axis = [x[0].item() for x in harmful_data_x]
            y_axis = [0] * num_to_plot
            classes = ['mean -1', 'mean 1']
        elif args.dataset_name == 'GMM2D':
            x_axis = [x[0].item() for x in harmful_data_x]
            y_axis = [x[1].item() for x in harmful_data_x]
            classes = ['mean(-1,-1)', 'mean(1,1)']

        colours = ListedColormap(['r', 'b'])
        scatter = plt.scatter(x_axis, y_axis, c=harmful_data_y, cmap=colours)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        wandb.log({f"harmful_data_epoch": plt})


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--YAMLPath", type=str)
    # args = parser.parse_args()
    # if args.YAMLPath:
    #     YAMLPath = args.YAMLPath

    with open(YAMLPath) as file:
        config = yaml.safe_load(file)
        wandb.init(
            project="IF_PROJECT",
            name=f"{config['dataset_name']}_testId{config['test_id_num']}_IFlr{config['influence_lr']}_IFlr{config['classification_lr']}_IFwd{config['classification_weight_decay']}_IFmomentum{config['classification_momentum']}_IFdecay{config['influence_weight_decay']}_softmaxTemp{config['softmax_temp']}",
            config=config
        )
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(config["_gpu_id"])

    main(wandb.config)
