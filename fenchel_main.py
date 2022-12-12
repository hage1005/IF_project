import argparse
import json
import os
import random
from tkinter import Image
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
import copy

import numpy as np
import torch

from src.data_utils.index import get_dataset
from src.utils.utils import get_single_from_dataset
from src.utils.Plotter_IF import Plotter_IF
from src.utils.Path_IF import Path_IF
from src.solver.fenchel_solver import FenchelSolver
from src.modeling.classification_models import get_classification_model
from src.modeling.influence_models import get_influence_model
from src.solver.utils import Normalizer

from hessian_main import main as hessian_main
import wandb
import yaml

os.chdir('/home/xiaochen/kewen/IF_project')
# YAMLPath = 'src/config/MNIST/single_test/exp/MNIST_1_100each.yaml'
YAMLPath = 'src/config/MNIST/single_test/exp/MNIST_1_100each/test_id_1/fenchel.yaml'
YAMLPath = 'src/config/GMM2D/exp01.yaml'
# YAMLPath = 'src/config/MNIST/single_test/exp/Cnn.yaml'

# YAMLPath = 'src/config/cifar10/single_test/default.yaml'

def main(args):
    # set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True
    Path = Path_IF(args)
    Dataset, dataset_type = get_dataset(args.dataset_name)

    class_label_dict = Dataset.get_class_label_dict()
    CLASS_MAP = Dataset.get_class_map()
    train_classes = [class_label_dict[c] for c in args.train_classes]
    ImageDataset = Dataset(train_classes, args.num_per_class)


    train_dataset, train_dataset_no_transform = ImageDataset.get_data("train"), ImageDataset.get_data("train", transform=False)
    dev_dataset, dev_dataset_no_transform = ImageDataset.get_data("dev"), ImageDataset.get_data("dev", transform=False)
    # test_dataset, test_dataset_no_transform = ImageDataset.get_test()

    train_dataset_size = len(train_dataset)

    x_dev, y_dev = get_single_from_dataset(
        dev_dataset, args.dev_id_num)

    wandb.run.summary['test_image'] = wandb.Image(
        x_dev, caption={'label': CLASS_MAP[y_dev.item()]})

    classification_model = get_classification_model(args.classification_model, args._num_class)

    influence_model, is_influence_model_hashmap = get_influence_model(args.influence_model, args._num_class, train_dataset_size, args._hidden_size_influence)

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
        "dev",
        dev_dataset,
        args.batch_size,
        shuffle=False)

    fenchel_classifier.init_weights(
        n_examples=len(train_dataset),
        w_init=args.dataWeight_weight_init,
        w_decay=args.dataWeight_weight_decay)
    
    fenchel_classifier.get_optimizer_influence(
        args.influence_lr,
        args.influence_momentum,
        args.influence_weight_decay,
        args.optimizer_influence)

    if args.use_pretrain_classification:
        fenchel_classifier.load_checkpoint_classification(Path.pretrain_ckpt_path)

    fenchel_classifier.get_optimizer_classification_and_scheduler(
        args.classification_lr,
        args.classification_momentum,
        args.classification_weight_decay,
        args.optimizer_classification,
        args.max_checkpoint_epoch // 5,
        0.8)

    result_true = Path.get_inv_hessian_influences()
    result_identity = Path.get_identity_influences()

    Plotter = Plotter_IF(train_dataset_no_transform, CLASS_MAP, CLASS_MAP[y_dev.item()], dataset_type, np.array(result_true['influence']), np.array(result_identity['influence']))
    Plotter.scatter_corr_identity_invHessian()

    for epoch in range(args.max_epoch):
        if epoch == 100:
            print("stop")
        if args.reset_pretrain_classification_every_epoch and epoch > 0:
            fenchel_classifier.load_checkpoint_classification(Path.pretrain_ckpt_path)
        fenchel_classifier.train_epoch()

        influences = [0.0 for _ in range(train_dataset_size)]
        # return influence for all trainig point if using hashmap
        if is_influence_model_hashmap:
            influences = influence_model.get_all_influence().cpu().detach().numpy()
        else:
            for i in tqdm(range(train_dataset_size)):
                x, y = train_dataset[i:i + 1][0], train_dataset[i:i + 1][1]
                influences[i] = influence_model(x.cuda(), y.cuda()).cpu().item()

        Path.save_influence(influences, epoch)

        if epoch == 0:
            Plotter.first_iter_influences = fenchel_classifier.first_iteration_grad.cpu().detach().numpy()
            Plotter.scatter_corr_ours_first_iter_grad(influences)
            Path.save_first_iter_grad(fenchel_classifier.first_iteration_grad.cpu().detach().numpy())

        Plotter.log_weight_stat(fenchel_classifier._weights.cpu().detach().numpy())

        Plotter.helpful_and_harmful_top_nine(influences)

        Plotter.plot_influence(influences)

        Plotter.plot_weight(fenchel_classifier._weights.cpu().detach().numpy())

        Plotter.log_correlation_ours_invHessian(influences)

        Plotter.log_correlation_ours_invHessian_top_k(influences, k=10)

        if epoch % 50 == 0:
            Plotter.scatter_corr_ours_invHessian(influences, epoch)
            Plotter.scatter_corr_ours_invHessian_top_k(influences, epoch, k = 10)
            Plotter.scatter_corr_ours_identity(influences, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--YAMLPath", type=str)
    args, unknown = parser.parse_known_args()
    if args.YAMLPath:
        YAMLPath = args.YAMLPath
    
    with open(YAMLPath) as file:
        config = yaml.safe_load(file)
        wandb.init(
            project="IF_PROJECT_single_test",
            name=f"{config['dataset_name']}_devId{config['dev_id_num']}",
            config=config
        )
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config["_gpu_id"])

    args = wandb.config
    Path = Path_IF(args)
    pretrain_needed_not_found = not os.path.exists(Path.pretrain_ckpt_path) and args.use_pretrain_classification
    inv_hessian_needed_not_found = not os.path.exists(Path.inv_hessian_json_path) and args.compare_with_inv_hessian
    identity_needed_not_found = not os.path.exists(Path.identity_json_path) and args.compare_with_identity

    if pretrain_needed_not_found or inv_hessian_needed_not_found or identity_needed_not_found:
        hessian_main(args)

    main(args)
