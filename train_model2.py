import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import CustomDataSet
from model.metrics import plot_scores, plot_losses, plot_lrs
from model.network2 import RecipeModelV2
from model.util import load_classes, get_default_device, DeviceDataLoader, to_device, save_checkpoint

batch_size = 32


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(epochs, max_lr, model, train_loader, test_loader, weight_decay=0.0, grad_clip=None,
          opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []

    # setup optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    # enable LR scheduler (1cycle policy)
    # The 1cycle policy anneals the learning rate from an initial learning rate to some maximum learning rate and then
    # from that maximum learning rate to some minimum learning rate much lower than the initial learning rate.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []

        # tqdm shows progress
        for batch in tqdm(train_loader):
            # compute the loss based on model output and real labels
            loss = model.training_step(batch)
            train_losses.append(loss)
            # backpropagate the loss
            loss.backward()

            # Gradient clipping
            if grad_clip:
                # fixes exploding gradients during training
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            # adjust parameters based on the calculated gradients
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()

            # Record and update learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

        # Validation phase
        result = evaluate(model, test_loader)
        # get mean
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        # print losses
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def main():
    device = get_default_device()
    print("running on device", device)

    classes = load_classes()
    print("number of classes", len(classes))

    train_dataset = CustomDataSet(classes, 'data/images/', 'data/cleaned_train.jsonl', v2=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    train_dl = DeviceDataLoader(train_loader, device)

    test_dataset = CustomDataSet(classes, 'data/images/', 'data/cleaned_test.jsonl', test=True, v2=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_dl = DeviceDataLoader(test_loader, device)

    print("number of images in test", len(train_loader) * batch_size)
    print("number of images in train", len(test_loader) * batch_size)

    # Debug
    # show_example(*test_dataset[100], classes=classes)
    # for img, label in train_loader:
    #     print(img.shape)
    #     print(label.shape)
    #     exit(0)
    # show_batch(train_loader)

    # Input 3 channels (=3 colors)
    # Output 100 channels (=all labels)
    model = to_device(RecipeModelV2(3, len(classes)), device)

    # Show structure
    # print(model)
    # for images, labels in train_dl:
    #     print(images.shape)
    #     outputs = model(images)
    #     break
    #
    # print('output shape', outputs.shape)
    # print('sample output', outputs[:2].data)

    epochs = 1
    max_lr = 0.001
    grad_clip = 0.1
    weight_decay = 0.0001

    history = [evaluate(model, train_dl)]
    start = time.perf_counter()

    history += train(epochs, max_lr, model, train_dl, test_dl,
                     grad_clip=grad_clip,
                     weight_decay=weight_decay,
                     opt_func=torch.optim.Adam)

    duration = time.perf_counter() - start
    print(f"finished training in {duration:.4f}s")

    save_checkpoint(model, v2=True)

    # Plotting
    plot_scores(history)
    plot_losses(history)
    plot_lrs(history)


if __name__ == '__main__':
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main()
