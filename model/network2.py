import torch
import torch.nn as nn
import torch.nn.functional as nnf

from model.metrics import F_score


# Helper for conv kernels
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        # Predict
        out = self(images)
        # Calc loss
        loss = nnf.binary_cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        # Predict
        out = self(images)
        # Calc loss
        loss = nnf.binary_cross_entropy(out, labels)
        # f measure accuracy
        score = F_score(out, labels)
        return {'test_loss': loss.detach(), 'test_score': score.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['test_loss'] for x in outputs]
        # get mean loss
        epoch_loss = torch.stack(batch_losses).mean()
        batch_scores = [x['test_score'] for x in outputs]
        # get mean accuracy
        epoch_score = torch.stack(batch_scores).mean()
        return {'test_loss': epoch_loss.item(), 'test_score': epoch_score.item()}

    def epoch_end(self, epoch, result):
        print(f"epoch {epoch}: "
              f"lr {result['lrs'][-1]:.5f}, "
              f"train_loss {result['train_loss']:.4f}, "
              f"test_loss {result['test_loss']:.4f}, "
              f"test_score {result['test_score']:.4f}")


# Resnet15
class RecipeModelV2(ClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # input 3 x 128 x 128
        self.conv1 = conv_block(in_channels, 64)
        # output 64 x 128 x 128
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        # output 64 x 128 x 128

        self.conv2 = conv_block(64, 128, pool=True)
        # output 128 x 32 x 32
        self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128),
                                  conv_block(128, 128))
        # output 128 x 32 x 32

        self.conv3 = conv_block(128, 512, pool=True)
        # output 512 x 8 x 8
        self.res3 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        # output 512 x 8 x 8

        self.conv4 = conv_block(512, 1024, pool=True)
        # output 1024 x 2 x 2
        self.res4 = nn.Sequential(conv_block(1024, 1024), conv_block(1024, 1024))
        # output 1024 x 2 x 2

        self.classifier = nn.Sequential(nn.MaxPool2d(2),  # output 1024 x 1 x 1
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(1024 * 1 * 1, 512),  # output 512
                                        nn.ReLU(),
                                        nn.Linear(512, num_classes))  # output 100

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.conv4(out)
        out = self.res4(out) + out
        out = self.classifier(out)
        out = torch.sigmoid(out)
        return out
