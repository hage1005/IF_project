
from model import Net
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")


class Solver():
    def __init__(self, args):
        self.device = torch.device("cuda:{}".format(args.gpu))
        self.max_iter = args.max_iter
        self.global_iter = 0
        self.batch_size = args.batch_size
        self.net = Net().to(self.device)
        self.optim = optim.SGD(self.net.parameters(), lr=args.influence_lr)
        self.display_step = args.display_step
        self.save_step = args.save_step
        self.ckpt_dir = args.ckpt_dir

    def train(self, train_loader):
        criterion = nn.CrossEntropyLoss()
        self.net.train()
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            for data in train_loader:
                self.global_iter += 1
                pbar.update(1)
                inputs, labels = data[0].to(
                    self.device), data[1].to(
                    self.device)

                self.optim.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optim.step()

                if self.global_iter % self.display_step == 0 or self.global_iter == self.max_iter + 1:
                    pbar.write('[{}] loss: {:.3f} '.format(
                        self.global_iter, loss))

                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint(str(self.global_iter))
                    pbar.write(
                        'Saved checkpoint(iter:{})'.format(
                            self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        self.save_checkpoint('last')
        pbar.write("[Training Finished]")
        pbar.close()

    def test(self, testloader):
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(
                    self.device), data[1].to(
                    self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                _, pred = torch.max(outputs, 1)
                c = (pred == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net': self.net.state_dict(), }
        optim_states = {'optim': self.optim.state_dict(), }
        states = {'iter': self.global_iter,
                  'model_states': model_states,
                  'optim_states': optim_states}
        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print(
                "=> saved checkpoint '{}' (iter {})".format(
                    file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path,
                                    map_location='cuda:{}'.format(self.gpu))
            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print(
                "=> loaded checkpoint '{} (iter {})'".format(
                    file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
