import argparse
import os
import random
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
import copy

import numpy as np
import torch

from torchvision import models
from src.utils.utils import save_json
from src.data_utils.cifar10 import get_cifar10_train, get_cifar10_test
from src.solver.fenchel_solver import FenchelSolver
from src.modeling.classification_models import CnnCifar
from src.modeling.influence_models import Net_IF
from src.data_utils.cifar10_label_class_map import CLASS_MAP, cifar_class_label_dict

import wandb
import yaml

os.chdir('/home/xiaochen/kewen/IF_project')
YAMLPath = 'src/config/cifar10/default.yaml'
# YAMLPath = 'src/config/cifar10/good_config/8.yaml'

def main(args):
    # set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

    def get_single_image_from_dataset(dataset, idx):
        x, y = dataset[idx]
        x = x.unsqueeze(0)
        y = torch.LongTensor([y])
        return x, y

    train_classes = [cifar_class_label_dict[c] for c in args.train_classes]

    train_dataset, train_dataset_no_transform = get_cifar10_train(
        classes=train_classes, num_per_class=args.num_per_class)
    test_dataset, test_dataset_no_transform = get_cifar10_test()

    x_test, y_test = get_single_image_from_dataset(
        test_dataset, args.test_id_num)

    wandb.run.summary['test_image'] = wandb.Image(
        x_test, caption={'label': CLASS_MAP[y_test.item()]})

    if args.classification_model == 'Resnet34':
        classification_model = models.resnet34(pretrained=True).to('cuda')
        classification_model.fc = torch.nn.Linear(
            classification_model.fc.in_features,
            args._num_class).to('cuda')
    elif args.classification_model == 'CnnCifar':
        classification_model = CnnCifar().to('cuda')
    else:
        raise NotImplementedError()

    if args.influence_model == 'Net_IF':
        influence_model = Net_IF(args._num_class).to('cuda')
    else:
        raise NotImplementedError()

    fenchel_classifier = FenchelSolver(
        x_test,
        y_test,
        classification_model=classification_model,
        influence_model=influence_model,
        softmax_temp=args.softmax_temp,
        train_classification_till_converge=args.train_classification_till_converge)

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
        w_init=args.dataWeight_weight_init,
        w_decay=args.dataWeight_weight_decay)

    fenchel_classifier.get_optimizer_classification(
        args.classification_lr,
        args.classification_momentum,
        args.classification_weight_decay)
    
    fenchel_classifier.get_optimizer_influence(
        args.influence_lr,
        args.influence_momentum,
        args.influence_weight_decay)

    if not os.path.exists(args._ckpt_dir + args._pretrain_ckpt_name):
        for epoch in range(20):
            fenchel_classifier.pretrain_epoch()
            test_acc = fenchel_classifier.evaluate('test')
            print('Pre-train Epoch {}, Test Acc: {:.4f}'.format(
                epoch, 100. * test_acc))
            fenchel_classifier.save_checkpoint_classification(args._ckpt_dir + args._pretrain_ckpt_name)

    if args.use_pretrain_classification:
        fenchel_classifier.load_checkpoint_classification(args._ckpt_dir + args._pretrain_ckpt_name)

    for epoch in range(args.max_epoch):
        if args.reset_pretrain_classification_every_epoch and epoch > 0:
            fenchel_classifier.load_checkpoint_classification(args._ckpt_dir + args._pretrain_ckpt_name)
        fenchel_classifier.train_epoch()
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

        wandb.log({'total_weight_std': torch.std(
            fenchel_classifier._weights).item()})

        fig = plt.figure(figsize=(6, 7))
        for i in range(1, 10):
            x, y = train_dataset_no_transform[helpful[i]]
            fig.add_subplot(3, 3, i)
            plt.title(f"{CLASS_MAP[y]}_{influences[helpful[i]]:.2f}")
            plt.imshow(x.permute(1, 2, 0))
        wandb.log({f"helpful_image_for_{CLASS_MAP[y_test.item()]}": fig})

        plt.clf()
        fig = plt.figure(figsize=(6, 7))
        for i in range(1, 10):
            x, y = train_dataset_no_transform[harmful[i]]
            fig.add_subplot(3, 3, i)
            plt.title(f"{CLASS_MAP[y]}_{influences[harmful[i]]:.2f}")
            plt.imshow(x.permute(1, 2, 0))
        wandb.log({f"harmful_image_for_{CLASS_MAP[y_test.item()]}": fig})
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--YAMLPath", type=str)
    args, unknown = parser.parse_known_args()
    if args.YAMLPath:
        YAMLPath = args.YAMLPath

    with open(YAMLPath) as file:
        config = yaml.safe_load(file)
        wandb.init(
            project="IF_PROJECT",
            name=f"{config['dataset_name']}_testId{config['test_id_num']}",
            config=config
        )
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config["_gpu_id"])

    main(wandb.config)
