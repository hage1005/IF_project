import argparse
import os
from sklearn.utils import shuffle
from tqdm import tqdm
import copy

import numpy as np
import torch

from torchvision import models
from src.utils.utils import save_json
from src.utils.dataset import return_data
from src.solver.fenchel_solver import FenchelSolver
from src.modeling.classification_models import CnnCifar
from src.modeling.influence_models import Net_IF

import wandb
import yaml

os.chdir('/home/xiaochen/kewen/IF_project')
EPSILON = 1e-5

CLASS_MAP = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']									

YAMLPath = 'src/config/CIFAR/exp01.yaml'

def main(args):
    def get_single_image(dataset, idx):
        x, y = dataset[idx]
        x = x.unsqueeze(0)
        y = torch.LongTensor([y])
        return x, y

    train_dataset, test_dataset = return_data(args.batch_size, "cifar10")
    x_test, y_test = get_single_image(test_dataset, args.test_id_num)
    wandb.run.summary['test_image'] = wandb.Image(x_test, caption={'label': CLASS_MAP[y_test.item()]})


    if args.classification_model == 'Resnet34':
        classification_model = models.resnet34(pretrained=True).to('cuda')
        classification_model.fc = torch.nn.Linear(classification_model.fc.in_features, args._num_class).to('cuda')
    elif args.classification_model == 'CnnCifar':
        classification_model = CnnCifar().to('cuda')
    else:
        raise NotImplementedError()

    if args.influence_model == 'Net_IF':
        influence_model = Net_IF(args._num_class).to('cuda')
    else:
        raise NotImplementedError()

    fenchen_classifier = FenchelSolver(x_test, y_test, classification_model = classification_model, influence_model = influence_model
        , softmax_temp=args.softmax_temp, train_classification_till_converge = args.train_classification_till_converge)

    fenchen_classifier.load_data("train", train_dataset, args.batch_size, shuffle=True)
    fenchen_classifier.load_data("test", test_dataset, args.batch_size, shuffle=False)
    

    fenchen_classifier.init_weights(
        n_examples=len(train_dataset),
        w_init=args.influence_weight_init,
        w_decay=args.influence_weight_decay)

    fenchen_classifier.get_optimizer(args.classification_lr, args.classification_momentum, args.classification_weight_decay)

    for epoch in range(args.max_epoch):
        fenchen_classifier.train_epoch()
        fenchen_classifier.save_checkpoint(args._ckpt_dir + args._ckpt_name)
        result = {}
        
        train_dataset_size = len(train_dataset)
        influences = [0.0 for _ in range(train_dataset_size)]
        # TODO compute by batch
        for i in tqdm(range(train_dataset_size)):
            x, y = train_dataset[i:i+1]
            influences[i] = influence_model(x.cuda(), y.cuda()).cpu().item()
        influences = np.array(influences)
        harmful = np.argsort(influences)
        helpful = harmful[::-1]
        result["helpful"] = helpful[:500].tolist()
        result["harmful"] = harmful[:500].tolist()
        result["influence"] = influences.tolist()
        json_path = os.path.join("outputs", args.dataset_name, f"IF_{args.dataset_name}_testid_{args.test_id_num}_epoch_{epoch}.json")
        save_json(result, json_path)

        for i in range(min(9, len(helpful))):
            x, y = train_dataset[helpful[i]]
            wandb.run.summary[f"helpful_{i}"] = wandb.Image(train_dataset[helpful[i]])
            Image = wandb.Image(train_dataset[helpful[i]][0], caption=f"{CLASS_MAP[y]}, id:{i}, influence={influences[helpful[i]]}")
            wandb.log({"helpful_{i}": Image})
        for i in range(min(9, len(harmful))):
            x, y = train_dataset[harmful[i]]
            wandb.run.summary[f"harmful_{i}"] = wandb.Image(train_dataset[harmful[i]])
            Image = wandb.Image(train_dataset[harmful[i]][0], caption=f"{CLASS_MAP[y]}, id:{i}, influence={influences[harmful[i]]}")
            wandb.log({"harmful_{i}": Image})
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--YAMLPath", type=str)
    args = parser.parse_args()
    if args.YAMLPath:
        YAMLPath = args.YAMLPath

    with open(YAMLPath) as file:
        config = yaml.safe_load(file)   
        wandb.init(
            project="IF_PROJECT",
            name = f"{config['dataset_name']}_testId{config['test_id_num']}_IFlr{config['influence_lr']}_IFlr{config['classification_lr']}_IFwd{config['classification_weight_decay']}_IFmomentum{config['classification_momentum']}_IFdecay{config['influence_weight_decay']}_softmaxTemp{config['softmax_temp']}",
            config=config
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["_gpu_id"])

    main(wandb.config)