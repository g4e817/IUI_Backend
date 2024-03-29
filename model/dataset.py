import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as transforms

from model.util import encode_label, imagenet_stats


class CustomDataSet(Dataset):
    def __init__(self, classes, images_path, json_path, test=False, v2=False):
        self.classes = classes
        self.images_path = images_path
        self.json_path = json_path
        self.test = test
        self.v2 = v2
        self.recipes = self.get_recipes(json_path)

    def get_recipes(self, json_path):
        recipes = []
        with open(json_path) as f:
            for line in f:
                item = json.loads(line)
                recipes.append(item)
        return recipes

    def get_image_tensor(self, filename):
        filepath = os.path.join(self.images_path, os.path.basename(filename))
        pipeline = [
            transforms.Resize(36),
            transforms.CenterCrop(32),
        ]

        if self.v2:
            pipeline = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]

        if not self.test:
            pipeline += [
                # transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(2),
            ]

        pipeline += [
            transforms.ToTensor(),
            transforms.Normalize(*imagenet_stats)
        ]

        trans = transforms.Compose(pipeline)
        image = Image.open(filepath)
        tensor = trans(image)
        return tensor

    def get_category_tensor(self, cats):
        return encode_label(cats, self.classes)

    def __getitem__(self, idx):
        recipe = self.recipes[idx]
        image_tensor = self.get_image_tensor(recipe['image'])

        if self.v2:
            category_tensor = self.get_category_tensor(recipe['categories'])
        else:
            category_tensor = torch.as_tensor(self.classes.index(recipe['categories'][0]))

        return image_tensor, category_tensor

    def __len__(self):
        return len(self.recipes)

class FoodDataSet(Dataset):
    def __init__(self, classes, images_path, txt_path, test=False):
        self.classes = classes
        self.images_path = images_path
        self.txt_path = txt_path
        self.test = test
        self.recipes = self.get_recipes(txt_path)

    def get_recipes(self, txt_path):
        recipes = []
        with open(txt_path) as f:
            for line in f:
                recipes.append(line.strip())
        return recipes

    def get_image_tensor(self, label, filename):
        filepath = os.path.join(self.images_path, label, filename)
        pipeline = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]

        if not self.test:
            pipeline += [
                # transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(2),
            ]

        pipeline += [
            transforms.ToTensor(),
            transforms.Normalize(*imagenet_stats)
        ]

        trans = transforms.Compose(pipeline)
        image = Image.open(filepath)
        return trans(image)

    def get_category_tensor(self, cats):
        return encode_label(cats, self.classes)

    def __getitem__(self, idx):
        recipe = self.recipes[idx]

        label, img = recipe.split("/")
        image_tensor = self.get_image_tensor(label, img + ".jpg")
        category_tensor = self.get_category_tensor([label])

        return image_tensor, category_tensor

    def __len__(self):
        return len(self.recipes)
