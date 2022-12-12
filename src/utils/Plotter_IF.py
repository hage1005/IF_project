
from matplotlib.colors import ListedColormap
import wandb
import numpy as np
import matplotlib.pyplot as plt

def draw_scatter_with_corr(x, y, x_label, y_label, epoch = 0):
    corr = round(np.corrcoef(x,y)[0,1],3)
    data = [[x, y] for (x, y) in zip(x, y)]
    table = wandb.Table(data=data, columns = [x_label, y_label])
    wandb.log({ f"{y_label} {x_label} epoch{epoch}" : wandb.plot.scatter(table, x_label, y_label, title=f"{y_label} {x_label} corr:{corr} epoch{epoch}")})
    wandb.run.summary[f"{y_label} {x_label} {epoch} corr"] = corr
    return corr

def plot_nine_images(images, titles, wandb_msg):
    fig = plt.figure(figsize=(6, 7))
    for i in range(9):
        x, y = images[i], titles[i]
        fig.add_subplot(3, 3, i + 1)
        plt.title(y)
        plt.imshow(x.permute(1, 2, 0))
    wandb.log({wandb_msg: fig})
    plt.clf()

def plot_scatter(x, y, x_label, y_label, wandb_msg):
    x = x
    y = y
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    wandb.log({wandb_msg: plt})
    plt.clf()


class Plotter_IF:
    def __init__(self, train_dataset, numeric_label_name_map, dev_label, data_type,
            true_influences=None, 
            identity_influences=None, 
            first_iter_influences=None
        ):
        self.train_dataset = train_dataset
        self.numeric_label_name_map = numeric_label_name_map
        self.dev_label = dev_label
        self.data_type = data_type
        self.true_influences = true_influences
        self.identity_influences = identity_influences
        self.first_iter_influences = first_iter_influences
        
        self.true_helpful = None
        self.true_harmful = None
        if true_influences is not None:
            self.true_helpful = np.argsort(true_influences)
            self.true_harmful = self.true_helpful[::-1]

    def helpful_and_harmful_top_nine(self, influences):
        if self.data_type == 'image':
            self.image_helpful_and_harmful_top_nine(influences)
        elif self.data_type == 'GMM2D':
            self.GMM2D_helpful_and_harmful_top_nine(influences)
        else:
            raise NotImplementedError()

    def image_helpful_and_harmful_top_nine(self, influences):
        helpful = np.argsort(influences)

        """plot helpful"""
        top_9_ind = helpful[:9]
        images, labels = self.train_dataset[top_9_ind]

        ys = [f"{self.numeric_label_name_map[c]}_{influences[idx]:.2f}" for c, idx in zip(labels, top_9_ind)]
        plot_nine_images(images, ys, f"helpful_image_for_{self.dev_label}")

        """plot harmful"""
        top_9_ind = helpful[-9:]
        images, labels = self.train_dataset[top_9_ind]

        ys = [f"{self.numeric_label_name_map[c]}_{influences[idx]:.2f}" for c, idx in zip(labels, top_9_ind)]
        plot_nine_images(images, ys, f"harmful_image_for_{self.dev_label}")

    def plot_influence(self, influences):
        """plot scatter plot, y is influence and x is id"""

        plot_scatter(list(range(len(influences))), influences, "id", "influence", f"scatter_plot_influence")

    def plot_weight(self, weights):
        """plot scatter plot, y is weight and x is id"""

        plot_scatter(list(range(len(weights))), weights, "id", "weight", f"scatter_plot_weight")

    def log_correlation_ours_invHessian(self, influences):
        wandb.log({'correlation_ours': round(np.corrcoef(influences, self.true_influences)[0, 1], 3)})

    def log_correlation_ours_invHessian_top_k(self, influences, k):

        top_k_ind = np.append(self.true_helpful[:k], self.true_harmful[:k])
        wandb.log({f"correlation_ours_top{k}": round(np.corrcoef(influences[top_k_ind], self.true_influences[top_k_ind])[0, 1], 3)})
    
    def scatter_corr_ours_invHessian(self, influences, epoch):
        draw_scatter_with_corr(influences, self.true_influences, 'ours', 'invHessian', epoch)

    def scatter_corr_ours_invHessian_top_k(self, influences, epoch, k):
        top_k_ind = np.append(self.true_helpful[:k], self.true_harmful[:k])
        draw_scatter_with_corr(influences[top_k_ind], self.true_influences[top_k_ind], 'ours', f"invHessian top{k}", epoch)
    
    def scatter_corr_ours_identity(self, influences, epoch):
        draw_scatter_with_corr(influences, self.identity_influences, 'ours', 'identity', epoch)
    
    def scatter_corr_ours_first_iter_grad(self, influences):
        draw_scatter_with_corr(influences, self.first_iter_influences, 'ours', 'first_iter_grad')

    def log_weight_stat(self, weights):
        wandb.log({
                f'overall_weight_std': np.std(weights).item(),
                f'overall_weight_mean': np.mean(weights).item(),
                f'overall_weight_mean_abs': np.mean(np.abs(weights)).item(),
            })
        
    def scatter_corr_first_iter_wrt_invHessian(self):
        draw_scatter_with_corr(self.first_iter_influences, self.true_influences, 'first_iter_grad', 'invHessian')

    def scatter_corr_first_iter_wrt_invHessian_top_k(self, k):
        top_k_ind = self.true_helpful[:k] + self.true_harmful[:k]
        draw_scatter_with_corr(self.first_iter_influences[top_k_ind], self.true_influences[top_k_ind], 'first_iter_grad', f'invHessian top{k}')

    def scatter_corr_identity_invHessian(self):
        draw_scatter_with_corr(self.identity_influences, self.true_influences, 'identity', 'invHessian')

    def GMM2D_helpful_and_harmful_top_nine(self, influences):
        helpful = np.argsort(influences)

        """plot helpful"""
        top_9_ind = helpful[:9]
        data, labels = self.train_dataset[top_9_ind]

        x_axis = [x[0].item() for x in data]
        y_axis = [x[1].item() for x in data]
        classes_to_name = ['mean(-1,-1)', 'mean(1,1)']
    
        colours = ListedColormap(['r', 'b'])
        scatter = plt.scatter(x_axis, y_axis, c=labels, cmap=colours)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes_to_name)
        wandb.log({f"helpful_data_epoch": plt})

        """plot harmful"""
        top_9_ind = helpful[-9:]
        data, labels = self.train_dataset[top_9_ind]

        x_axis = [x[0].item() for x in data]
        y_axis = [x[1].item() for x in data]
        classes_to_name = ['mean(-1,-1)', 'mean(1,1)']

        colours = ListedColormap(['r', 'b'])
        scatter = plt.scatter(x_axis, y_axis, c=labels, cmap=colours)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes_to_name)
        wandb.log({f"harmful_data_epoch": plt})

        

    