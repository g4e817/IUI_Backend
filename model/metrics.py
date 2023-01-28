import matplotlib.pyplot as plt
import numpy as np
import torch


# Source: from internet
def F_score(output, label, threshold=0.5, beta=1):
    prob = output > threshold
    label = label > threshold

    # true-positive
    TP = (prob & label).sum(1).float()

    # true-negative
    TN = ((~prob) & (~label)).sum(1).float()

    # false-postive
    FP = (prob & (~label)).sum(1).float()

    # false negative
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-12)

    return F2.mean(0)


def plot_scores(history):
    scores = [x['val_score'] for x in history]
    plt.plot(scores, '-x')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('F1 score vs. No. of epochs')


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')


def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')
