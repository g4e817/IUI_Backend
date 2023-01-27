import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as transforms


class CustomDataSet(Dataset):
    def __init__(self, classes, images_path, json_path):
        self.classes = classes
        self.images_path = images_path
        self.json_path = json_path
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
        trans = transforms.Compose([
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image = Image.open(filepath)
        tensor = trans(image)
        return tensor

    def __getitem__(self, idx):
        recipe = self.recipes[idx]
        cats = recipe['categories']

        # Get random category for training
        # cat = random.choice(cats)
        # Get first cat
        cat = cats[0]

        tensor = self.get_image_tensor(recipe['image'])
        # indices = torch.as_tensor(le.transform(cats))
        index = torch.as_tensor(self.classes.index(cat))

        return tensor, index

    def __len__(self):
        return len(self.recipes)
