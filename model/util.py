import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

# mean and std deviation for imagenet to support pre-trained
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_classes():
    classes = []
    with open('data/unique_cats.txt') as f:
        for line in f:
            classes.append(line.strip())
    return classes


def save_checkpoint(model, v2=False):
    if v2:
        path = 'data/model2_checkpoint.pth'
    else:
        path = 'data/model_checkpoint.pth'
    torch.save(model.state_dict(), path)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def to_device(data, device):
    # Helper to move data to device
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Similar to fastai, bind a dataloader to a torch.device
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def encode_label(label, classes):
    target = torch.zeros(len(classes))
    for item in label:
        idx = classes.index(item)
        # Set 1s in tensor
        target[idx] = 1
    return target


def decode_target(target, classes, threshold=0.5):
    result = []
    for idx, item in enumerate(target):
        # Only return label if threshold >= 0.5
        if item >= threshold:
            result.append(classes[idx])
    return ', '.join(result)


def denorm(img_tensors, normalization=imagenet_stats):
    return img_tensors * normalization[1][0] + normalization[0][0]


def show_example(img, label, classes):
    plt.imshow(denorm(img).permute(1, 2, 0))
    print("Label:", decode_target(label, classes))
    print(label)
    plt.show()


def show_batch(dl, nmax=16):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(denorm(images[:nmax]), nrow=4).permute(1, 2, 0))
        break
