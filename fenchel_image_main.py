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
        trans = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    elif args.dataset_name == 'mnist':
        Dataset = MnistDataset
        trans = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Normalize(
            (0.1307,),(0.3081,))
            ])
    else:
        raise NotImplementedError()

    class_label_dict = Dataset.get_class_label_dict()
    CLASS_MAP = Dataset.get_class_map()
    train_classes = [class_label_dict[c] for c in args.train_classes]
    ImageDataset = Dataset(args.dev_original_folder, args.dev_transformed_folder, args.test_original_folder, args.test_transformed_folder, train_classes, trans, args.num_per_class)


    train_dataset, train_dataset_no_transform = ImageDataset.get_train()
    dev_dataset, dev_dataset_no_transform = ImageDataset.get_dev()
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

    fenchel_classifier = FenchelSolver(
        x_dev,
        y_dev,
        classification_model=classification_model,
        influence_model=influence_model,
        is_influence_model_hashmap=is_influence_model_hashmap,
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
    f"epoch{args.max_pretrain_epoch}_lr{args.pretrain_classification_lr}_" + args._pretrain_ckpt_name)
    if not os.path.exists(pretrain_ckpt_path):
        fenchel_classifier.get_optimizer_classification(
            args.pretrain_classification_lr,
            args.classification_momentum,
            args.classification_weight_decay,
            args.optimizer_classification
        )
        for epoch in range(args.max_pretrain_epoch):
            fenchel_classifier.pretrain_epoch()
            dev_acc = fenchel_classifier.evaluate('dev')
            print('Pre-train Epoch {}, dev Acc: {:.4f}'.format(
                epoch, 100. * dev_acc))
        fenchel_classifier.save_checkpoint_classification(pretrain_ckpt_path)
        fenchel_classifier.global_epoch = 0


    if args.use_pretrain_classification:
        fenchel_classifier.load_checkpoint_classification(pretrain_ckpt_path)

    fenchel_classifier.get_optimizer_classification(
        args.classification_lr,
        args.classification_momentum,
        args.classification_weight_decay,
        args.optimizer_classification
    )
    
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
    
    x = result_identity['influence']
    y = result_true['influence']
    draw_scatter(x, y, 'identity', 'invHessian')
    for epoch in range(args.max_epoch):
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
            y = result_true['influence']
            corr = draw_scatter(x, y, "first_iter_grad", "invHessian")
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

        """plot histogram"""
        cmap = plt.cm.get_cmap('hsv', len(args.train_classes))
        colors = [cmap(i) for i in range(len(args.train_classes))]
        n_bins = 10
        all_labels = train_dataset[:][1].numpy()
        indset= sorted(list(set(all_labels)))
        indmap = {indset[i]: i for i in range(len(indset))}
        x = [[] for _ in  range(len(indset))]
        for i in range(train_dataset_size):
            x[indmap[all_labels[i]]].append(influences[i])
        plt.hist(x, n_bins, density=True, histtype='bar', color=colors, label = args.train_classes, stacked = True)
        plt.legend(prop={'size': 10})
        try:
            wandb.log({'histogram_influence': plt})
        except:
            pass
        plt.clf()

        """plot influence"""
        x = influences
        y = result_true['influence']
        wandb.log({'correlation_ours': round(np.corrcoef(x, y)[0, 1], 3)})
        if epoch % 20 == 0:
            x = influences
            y = result_true['influence']
            draw_scatter(x, y, 'ours', 'invHessian', epoch)

            x = influences
            y = result_identity['influence']
            draw_scatter(x, y, 'ours', 'identity', epoch)
            

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
