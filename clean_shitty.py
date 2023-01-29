import os

from PIL import Image
import imghdr

with open('data/food-101/meta/train_cleaned.txt', 'w') as w:
    with open('data/food-101/meta/train.txt') as f:
        for line in f:
            line = line.strip()
            label, img = line.split("/")
            path = os.path.join('data/food-101/images', label, img + ".jpg")
            image = Image.open(path)
            # if image.verify() and image.mode == "RGB":
            #     w.write(line + "\n")
            if imghdr.what(path) == 'jpeg' and image.mode == "RGB":
                w.write(line + "\n")

with open('data/food-101/meta/test_cleaned.txt', 'w') as w:
    with open('data/food-101/meta/test.txt') as f:
        for line in f:
            line = line.strip()
            label, img = line.split("/")
            path = os.path.join('data/food-101/images', label, img + ".jpg")
            image = Image.open(path)
            # if image.verify() and image.mode == "RGB":
            #     w.write(line + "\n")
            if imghdr.what(path) == 'jpeg' and image.mode == "RGB":
                w.write(line + "\n")
