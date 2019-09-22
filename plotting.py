import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

FONTSIZE = 15
ALPHA = 0.5
BINS, RANGE = 100, (0, 0.20)


def format_plot(x_label, y_label, title, fontsize=None):
    if not fontsize:
        fontsize = FONTSIZE
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()


def plot_line(x, train_y, val_y, x_label, y_label, cfg):
    if train_y:
        plt.plot(x, train_y, label='Training')
    if val_y:
        plt.plot(x, val_y, label='Validation')
    format_plot(x_label, y_label, title='{} vs {}'.format(y_label, x_label))
    plt.savefig('{}/{}-vs-{}.png'.format(os.path.join(cfg.plot_dir, cfg.model_name), y_label.lower(), x_label.lower()))
    plt.close()


def plot_hist(xs, colors, prune_type, layer, k, x_label, y_label, cfg):
    for i in range(len(xs)):
        plt.hist(xs[i], label='layer: {} k: {:.2f}'.format(layer, k), bins=BINS, range=RANGE, color=colors[i], alpha=ALPHA)
    format_plot(x_label, y_label, title='Histogram of {} for {} Pruning'.format(x_label, prune_type))
    plt.savefig('{}/{}_{}-prune_layer{}_k{:.2f}.png'.format(os.path.join(cfg.plot_dir, cfg.model_name), x_label.replace(' ', '-').lower(), prune_type.lower(), layer, k))
    plt.close()
