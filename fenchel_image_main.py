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

from torchvision import models, transforms
from src.data_utils.MnistDataset import MnistDataset
from src.utils.utils import save_json
from src.data_utils.Cifar10Dataset import Cifar10Dataset
from src.solver.fenchel_solver import FenchelSolver
from src.modeling.classification_models import CnnCifar, MNIST_1, MNIST_2, CnnMnist
from src.modeling.influence_models import Net_IF, MNIST_IF_1, hashmap_IF
from src.solver.utils import Normalizer

from hessian_main import main as hessian_main
import wandb
import yaml

os.chdir('/home/xiaochen/kewen/IF_project')
# YAMLPath = 'src/config/MNIST/single_test/exp/MNIST_1_100each.yaml'
YAMLPath = 'src/config/MNIST/single_test/exp/MNIST_1_100each/test_id_1/fenchel.yaml'
# YAMLPath = 'src/config/MNIST/single_test/exp/Cnn.yaml'

# YAMLPath = 'src/config/cifar10/single_test/default.yaml'

def main(args, truth_path, Identity_path, base_path):
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

    if args.dataset_name == 'cifar10':
        Dataset = Cifar10Dataset
    elif args.dataset_name == 'mnist':
        Dataset = MnistDataset
    else:
        raise NotImplementedError()

    class_label_dict = Dataset.get_class_label_dict()
    CLASS_MAP = Dataset.get_class_map()
    train_classes = [class_label_dict[c] for c in args.train_classes]
    ImageDataset = Dataset(args.dev_original_folder, train_classes, args.num_per_class)


    train_dataset, train_dataset_no_transform = ImageDataset.get_data("train"), ImageDataset.get_data("train", transform=False)
    dev_dataset, dev_dataset_no_transform = ImageDataset.get_data("dev"), ImageDataset.get_data("dev", transform=False)
    # test_dataset, test_dataset_no_transform = ImageDataset.get_test()
    train_dataset_size = len(train_dataset)

    x_dev, y_dev = get_single_image_from_dataset(
        dev_dataset, args.dev_id_num)

    wandb.run.summary['test_image'] = wandb.Image(
        x_dev, caption={'label': CLASS_MAP[y_dev.item()]})

    if args.classification_model == 'Resnet34':
        classification_model = models.resnet34(pretrained=True).to('cuda')
        classification_model.fc = torch.nn.Linear(
            classification_model.fc.in_features,
            args._num_class).to('cuda')
    elif args.classification_model == 'CnnCifar':
        classification_model = CnnCifar(args._num_class).to('cuda')
    elif args.classification_model == 'MNIST_1':
        classification_model = MNIST_1(args._hidden_size_classification, args._num_class).to('cuda')
    elif args.classification_model == 'MNIST_2':
        classification_model = MNIST_2(args._num_class).to('cuda')
    elif args.classification_model == 'CnnMnist':
        classification_model = CnnMnist(args._num_class).to('cuda')
    else:
        raise NotImplementedError()

    is_influence_model_hashmap = False
    if args.influence_model == 'Net_IF':
        influence_model = Net_IF(args._num_class).to('cuda')
    elif args.influence_model == 'MNIST_IF_1':
        influence_model = MNIST_IF_1(args._hidden_size_influence, args._num_class).to('cuda')
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
        "dev",
        dev_dataset,
        args.batch_size,
        shuffle=False)
    # fenchel_classifier.load_data(
    #     "test",
    #     test_dataset
    #     args.batch_size,
    #     shuffle=False)

    fenchel_classifier.init_weights(
        n_examples=len(train_dataset),
        w_init=args.dataWeight_weight_init,
        w_decay=args.dataWeight_weight_decay)
    
    fenchel_classifier.get_optimizer_influence(
        args.influence_lr,
        args.influence_momentum,
        args.influence_weight_decay,
        args.optimizer_influence)

    ckpt_dir = os.path.join("checkpoints/fenchel", args.dataset_name, args.classification_model)
    os.makedirs(ckpt_dir, exist_ok=True)
    pretrain_ckpt_path = os.path.join(ckpt_dir,
    f"epoch{args.max_pretrain_epoch}_lr{args.pretrain_classification_lr}_{args._pretrain_ckpt_name}")
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


    if args.use_pretrain_classification:
        fenchel_classifier.load_checkpoint_classification(pretrain_ckpt_path)

    fenchel_classifier.get_optimizer_classification_and_scheduler(
        args.classification_lr,
        args.classification_momentum,
        args.classification_weight_decay,
        args.optimizer_classification,
        args.max_checkpoint_epoch // 5,
        0.8)

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
    
    x = np.array(result_identity['influence'])
    influence_true = np.array(result_true['influence'])
    draw_scatter(x, influence_true, 'identity', 'invHessian')

    y_idx_helpful_and_harmful = result_true['helpful'][:10] + result_true['harmful'][:10]
    draw_scatter(x[y_idx_helpful_and_harmful], influence_true[y_idx_helpful_and_harmful], 'identity', 'invHessian top10', epoch = 0)

    for epoch in range(args.max_epoch):
        if epoch == 100:
            print("stop")
        if args.reset_pretrain_classification_every_epoch and epoch > 0:
            fenchel_classifier.load_checkpoint_classification(pretrain_ckpt_path)
        fenchel_classifier.train_epoch()
        result = {}

        influences = [0.0 for _ in range(train_dataset_size)]
        # return influence for all trainig point if using hashmap
        if is_influence_model_hashmap:
            influences = influence_model.get_all_influence().cpu().detach().numpy()
        # TODO compute by batch
        else:
            for i in tqdm(range(train_dataset_size)):
                x, y = train_dataset[i:i + 1][0], train_dataset[i:i + 1][1]
                influences[i] = influence_model(x.cuda(), y.cuda()).cpu().item()
        
        def save_result(influences, path):
            influences = np.array(influences)
            helpful = np.argsort(influences)
            harmful = helpful[::-1]
            result["helpful"] = helpful[:500].tolist()
            result["harmful"] = harmful[:500].tolist()
            result["influence"] = influences.tolist()

            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_json(result, path)

        save_result(influences, os.path.join(base_path, args.influence_model, f"epoch{epoch}.json"))
        if epoch == 0:
            x = fenchel_classifier.first_iteration_grad.cpu().detach().numpy()
            corr = draw_scatter(x, influence_true, "first_iter_grad", "invHessian")
            draw_scatter(x[y_idx_helpful_and_harmful], influence_true[y_idx_helpful_and_harmful], "first_iter_grad", "invHessian top10")
            save_result(fenchel_classifier.first_iteration_grad.cpu().detach(), 
                os.path.join(base_path, f"first_iteration_grad.json"))
            save_json({},os.path.join(base_path, f"first_iteration_grad_lr{args.classification_lr}_epoch{args.max_pretrain_epoch}_corr{corr}.json"))

            wandb.run.summary['y_true x_first_iter'] = corr

        wandb.log({
                f'all_{train_dataset_size}_weight_std': torch.std(fenchel_classifier._weights).item(),
                f'all_{train_dataset_size}_weight_mean': torch.mean(fenchel_classifier._weights).item(),
                f'all_{train_dataset_size}_weight_mean_abs': torch.mean(torch.abs(fenchel_classifier._weights)).item(),
            })

        """plot helpful"""
        helpful = np.argsort(influences)
        harmful = helpful[::-1]
        fig = plt.figure(figsize=(6, 7))
        for i in range(1, 10):
            x, y = train_dataset_no_transform[helpful[i]]
            fig.add_subplot(3, 3, i)
            plt.title(f"{CLASS_MAP[y]}_{influences[helpful[i]]:.2f}")
            plt.imshow(x.permute(1, 2, 0))
        wandb.log({f"helpful_image_for_{CLASS_MAP[y_dev.item()]}": fig})

        """plot harmful"""
        plt.clf()
        fig = plt.figure(figsize=(6, 7))
        for i in range(1, 10):
            x, y = train_dataset_no_transform[harmful[i]]
            fig.add_subplot(3, 3, i)
            plt.title(f"{CLASS_MAP[y]}_{influences[harmful[i]]:.2f}")
            plt.imshow(x.permute(1, 2, 0))
        wandb.log({f"harmful_image_for_{CLASS_MAP[y_dev.item()]}": fig})
        plt.clf()

        """plot scatter plot, y is influence and x is id"""
        x = list(range(train_dataset_size))
        y = influences
        plt.scatter(x, y)
        plt.xlabel("id")
        plt.ylabel("influence")
        wandb.log({f"scatter_plot_influence": plt})
        plt.clf()

        """plot scatter plot, y is weight and x is id"""
        x = list(range(train_dataset_size))
        y = fenchel_classifier._weights.cpu().detach().numpy()
        plt.scatter(x, y)
        plt.xlabel("id")
        plt.ylabel("weight")
        wandb.log({f"scatter_plot_weight": plt})
        plt.clf()


        """plot influence"""
        x = influences
        y = result_true['influence']
        wandb.log({'correlation_ours': round(np.corrcoef(x, y)[0, 1], 3)})
        wandb.log({'correlation_ours_top10': round(np.corrcoef(x[y_idx_helpful_and_harmful], influence_true[y_idx_helpful_and_harmful])[0, 1], 3)})
        if epoch % 40 == 0:
            x = influences
            draw_scatter(x, influence_true, 'ours', 'invHessian', epoch)
            draw_scatter(x[y_idx_helpful_and_harmful], influence_true[y_idx_helpful_and_harmful], 'ours', 'invHessian top10', epoch)

            x = influences
            y = result_identity['influence']
            draw_scatter(x, y, 'ours', 'identity', epoch)

            y = fenchel_classifier._weights.cpu().detach().numpy()
            draw_scatter(x, y, 'ours', 'weight', epoch)
    


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
    base_path_truth = os.path.join(
        "outputs",
        args.dataset_name,
        args.classification_model,
        "dev_id_" + str(args.dev_id_num),
        f"pretrain{args.max_checkpoint_epoch}epoch"
    )
    truth_path = os.path.join(base_path_truth, "Percy.json")
    Identity_path = os.path.join(base_path_truth, "Identity.json")
    base_path = os.path.join(
        "outputs",
        args.dataset_name,
        args.classification_model,
        "dev_id_" + str(args.dev_id_num),
        f"pretrain{args.max_checkpoint_epoch}epoch"
    )
    if not os.path.exists(truth_path): #first get the truth and identity hessian
        hessian_main(args)
    main(args, truth_path, Identity_path, base_path)
