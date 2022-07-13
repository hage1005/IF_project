import argparse
import os
import numpy as np
from tqdm import tqdm

import torch

from src.utils.utils import save_json
from src.utils.dataset import return_data
from src.solver.fenchel_solver import FenchelSolver

from src.modeling.influence_models import GMM1D_IF
from src.modeling.classification_models import LogisticRegression

import wandb
import yaml
from types import SimpleNamespace

os.chdir('/home/xiaochen/kewen/IF_project')
EPSILON = 1e-5


YAMLPath = 'src/config/GMM1D/GMM1D_exp09.yaml'

def main(args):
    train_dataset, test_dataset = return_data(args.batch_size, "GMM1D")
    input_dim = 1
    num_class = 2
    x_test, y_test = test_dataset[args.test_id_num]
    x_test, y_test = torch.FloatTensor([[x_test]]), torch.IntTensor([y_test])

    if args.classification_model == 'LogisticRegression':
        classification_model = LogisticRegression(input_dim,num_class).to('cuda') # 2class
    else:
        raise NotImplementedError()
    
    if args.influence_model == 'GMM1D_IF':
        influence_model = GMM1D_IF(input_dim, 1, num_class).to('cuda') # influence score
    else:
        raise NotImplementedError()

    fenchen_classifier = FenchelSolver(x_test, y_test, classification_model = classification_model, influence_model = influence_model, softmax_temp=args.softmax_temp)

    fenchen_classifier.load_data("train", train_dataset, args.batch_size, shuffle=True)
    fenchen_classifier.load_data("test", test_dataset, args.batch_size, shuffle=False)
    

    fenchen_classifier.init_weights(
        n_examples=len(train_dataset),
        w_init=args.influence_weight_init,
        w_decay=args.influence_weight_decay)

    fenchen_classifier.get_optimizer(args.classification_lr, args.classification_momentum, args.classification_weight_decay)

    for epoch in range(args.max_epoch):
        
        fenchen_classifier.train_epoch()
        fenchen_classifier.save_checkpoint(args._ckpt_dir + args.ckpt_name)
        
        result = {}
        train_dataset_size = len(train_dataset)
        influences = [0.0 for _ in range(train_dataset_size)]
        # TODO compute by batch
        for i in tqdm(range(train_dataset_size)):
            x, y = train_dataset[i]
            influences[i] = influence_model(torch.cuda.FloatTensor([x.item()]).unsqueeze(0),torch.cuda.LongTensor([y.item()])).cpu().item()
        influences = np.array(influences)
        helpful = np.argsort(influences)
        harmful = helpful[::-1]
        result["helpful"] = helpful[:500].tolist()
        result["harmful"] = harmful[:500].tolist()
        result["influence"] = influences.tolist()
        json_path = os.path.join("outputs", args.dataset_name, f"IF_GMM1D_testid_{args.test_id_num}_epoch_{epoch}.json")
        save_json(result, json_path)
        helpful_data = [ [train_dataset[x][0].item(), train_dataset[x][1].item()] for x in helpful[:100]]
        table = wandb.Table(data=helpful_data, columns = ["x", "class"])
        wandb.log({f"helpful_data_epoch{epoch}" : wandb.plot.scatter(table,
                                    "x", "class")})



if __name__ == "__main__":
    with open(YAMLPath) as file:
        config = yaml.safe_load(file)   
        wandb.init(
            project="IF_PROJECT",
            name = f"{config['dataset_name']}_test_id_{config['test_id_num']}",
            config=config
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["_gpu_id"])
        config = SimpleNamespace(**config)

    main(config)