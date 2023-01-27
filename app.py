import json

import torch
import torch.nn.functional as nnf
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

from train_model import Network

app = Flask(__name__)
model = Network()
model.load_state_dict(torch.load('data/model_checkpoint.pth'))
model.eval()

classes = []
with open('data/unique_cats.txt') as f:
    for line in f:
        classes.append(line.strip())

def transform_image(file):
    trans = transforms.Compose([
        transforms.Resize(36),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = Image.open(file)
    tensor = trans(image)
    tensor.unsqueeze_(0)
    return tensor


def get_pred(tensor, n=5):
    outputs = model.forward(tensor)
    prob = nnf.softmax(outputs, dim=1)
    top_p, top_class = prob.topk(n, dim=1)
    return zip(top_p.tolist()[0], top_class.tolist()[0])


def render_pred(pred):
    probs = []
    for (prob, id) in pred:
        name = classes[id]
        probs.append((name, prob))

    return probs


@app.route("/", methods=['GET'])
def hello_world():
    res = jsonify({'hi': 2})
    res.headers.add('Access-Control-Allow-Origin', '*')
    return res


@app.route('/pred', methods=['POST'])
def predict():
    file = request.files['file']
    if file is not None:
        tensor = transform_image(file)
        pred = get_pred(tensor)
        probs = render_pred(pred)
        return jsonify({'prediction': probs})
    return jsonify({'msg': 'No input file given.'})
