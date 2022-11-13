import argparse
import json
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import yaml

from src.utils.utils import save_json
from src.data_utils.GMM1D import get_GMM1D_data
from src.data_utils.GMM2D import get_GMM2D_data
from src.solver.utils import Normalizer

from src.solver.fenchel_solver import FenchelSolver

from src.modeling.influence_models import Linear_IF, hashmap_IF
from src.modeling.classification_models import LogisticRegression

import wandb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from types import SimpleNamespace

os.chdir('/home/xiaochen/kewen/IF_project')
# YAMLPath = 'src/config/GMM2D/exp02.yaml'
YAMLPath = 'src/config/GMM1D/exp01.yaml'


def main(args, truth_path, Identity_path, base_path):
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

    x_dev, y_dev = test_dataset[args.dev_id_num:args.dev_id_num + 1]
    # x_dev, y_dev = torch.FloatTensor([[x_dev]]), torch.IntTensor([y_dev])
    wandb.run.summary['x_dev'] = str(x_dev.tolist())
    wandb.run.summary['y_dev'] = str(y_dev.tolist())
    if args.classification_model == 'LogisticRegression':
        classification_model = LogisticRegression(
            args._input_dim, args._num_class).to('cuda')
    else:
        raise NotImplementedError()

    is_influence_model_hashmap = False
    if args.influence_model == 'Linear_IF':
        influence_model = Linear_IF(
            args._input_dim, args._num_class).to('cuda')
    elif args.influence_model == 'hashmap_IF':
        influence_model = hashmap_IF(train_dataset_size).to('cuda')
        is_influence_model_hashmap = True
    else:
        raise NotImplementedError()

    normalize_fn_classification = Normalizer(args.normalize_fn_classification, args.softmax_temp)
    normalize_fn_influence = Normalizer(args.normalize_fn_influence, args.softmax_temp)

    fenchel_classifier = FenchelSolver(
        x_dev,
        y_dev,
        classification_model=classification_model,
        influence_model=influence_model,
        is_influence_model_hashmap=is_influence_model_hashmap,
        normalize_fn_classification = normalize_fn_classification,
        normalize_fn_influence = normalize_fn_influence,
        softmax_temp=args.softmax_temp,
        train_classification_till_converge=args.train_classification_till_converge,
        clip_min_weight=args.clip_min_weight)

    fenchel_classifier.load_data(
        "train",
        train_dataset,
        args.batch_size,
        shuffle=True)
    fenchel_classifier.load_data(
        "test",
        test_dataset,
        args.batch_size,
        shuffle=False)

    fenchel_classifier.init_weights(
        n_examples=len(train_dataset),
        w_init=args.influence_weight_init,
        w_decay=args.influence_weight_decay)

    fenchel_classifier.get_optimizer(
        args.classification_lr,
        args.influence_lr,
        args.classification_momentum,
        args.classification_weight_decay)

    ckpt_dir = os.path.join("checkpoints/fenchel", args.dataset_name, args.classification_model)
    os.makedirs(ckpt_dir, exist_ok=True)
    pretrain_ckpt_path = os.path.join(ckpt_dir,
    f"epoch{args.max_pretrain_epoch}_lr{args.pretrain_classification_lr}_" + args._pretrain_ckpt_name)
    if not os.path.exists(pretrain_ckpt_path):
        fenchel_classifier.get_optimizer_classification_and_scheduler(
            args.pretrain_classification_lr,
            args.classification_momentum,
            args.classification_weight_decay,
            args.optimizer_classification,
            args.max_checkpoint_epoch // 5,
            0.8)
        for epoch in range(args.max_pretrain_epoch):
            fenchel_classifier.pretrain_epoch()
            dev_acc = fenchel_classifier.evaluate('dev')
            print('Pre-train Epoch {}, dev Acc: {:.4f}'.format(
                epoch, 100. * dev_acc))
        fenchel_classifier.save_checkpoint_classification(pretrain_ckpt_path)
        fenchel_classifier.global_epoch = 0

    with open (truth_path, "r") as f:
        result_true = json.loads(f.read())
    with open (Identity_path, "r") as f:
        result_identity = json.loads(f.read())
    
    def draw_scatter(x, y, x_label, y_label, epoch = 0):
        corr = round(np.corrcoef(x,y)[0,1],3)
        data = [[x, y] for (x, y) in zip(x, y)]
        table = wandb.Table(data=data, columns = [x_label, y_label])
        wandb.log({ f"{y_label} {x_label} epoch{epoch}" : wandb.plot.scatter(table, x_label, y_label, title=f"{y_label} {x_label} corr:{corr} epoch{epoch}")})
        wandb.run.summary[f"{y_label} {x_label} {epoch} corr"] = corr
        return corr

    for epoch in range(args.max_epoch):
        fenchel_classifier.train_epoch()
        result = {}
        
        train_dataset_size = len(train_dataset)
        influences = [0.0 for _ in range(train_dataset_size)]

        if is_influence_model_hashmap:
            influences = influence_model.get_all_influence().cpu().detach().numpy()
        # TODO compute by batch
        else:
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
            f"IF_{args.dataset_name}_testid_{args.dev_id_num}_epoch_{epoch}.json")
        save_json(result, json_path)

        # wandb doesn't support legend somehow
        # helpful_data = [ [train_dataset[x][0],train_dataset[x][1].item()] for x in helpful[:num_to_plot]]
        # table = wandb.Table(data=helpful_data, columns = ["x", "class"])
        # wandb.log({f"helpful_data_epoch{epoch}" : wandb.plot.scatter(table,
        #                             "x", "class")})
        wandb.log({'total_weight_std': torch.std(
            fenchel_classifier._weights).item()})

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
            project="IF_PROJECT_GMM",
            name=f"{config['dataset_name']}_testId{config['dev_id_num']}_IFlr{config['influence_lr']}_IFlr{config['classification_lr']}_IFwd{config['classification_weight_decay']}_IFmomentum{config['classification_momentum']}_IFdecay{config['influence_weight_decay']}_softmaxTemp{config['softmax_temp']}",
            config=config
        )
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config["_gpu_id"])

    main(wandb.config)
