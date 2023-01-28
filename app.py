import torch
import torch.nn.functional as nnf
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys

from model.network import Network
from model.network2 import RecipeModelV2
from model.util import get_default_device, to_device, imagenet_stats, decode_target

app = Flask(__name__)
CORS(app)

classes = []
with open('data/unique_cats.txt') as f:
    for line in f:
        classes.append(line.strip())

device = get_default_device()

model = Network(classes)
model.load_state_dict(torch.load('data/model_checkpoint.pth'))
model.eval()

model2 = to_device(RecipeModelV2(3, len(classes)), device)
model2.load_state_dict(torch.load('data/model2_checkpoint.pth'))
model2.eval()


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


def transform_image_v2(file):
    trans = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats)
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


def get_pred_v2(tensor, threshold=0.5):
    outputs = model2(tensor)
    print(outputs)
    prediction = outputs[0]
    return decode_target(prediction, classes, threshold=threshold)


def render_pred(pred):
    probs = []
    for (prob, id) in pred:
        name = classes[id]
        probs.append((name, prob))

    return probs


@app.route("/", methods=['GET'])
def hello_world():
    res = jsonify({'hi': 2})
    return res

@app.route("/categories", methods=['GET'])
def get_categories():
    with open("data/unique_cats.txt", "r") as f:
        data = f.read()

    res = jsonify({'categories': data.splitlines()})
    return res

#TODO MARKE HOW DO YOU WANT TO SAVE THAT SHIT?
@app.route('/user-validation', methods=['POST'])
def userValidation():
    file = request.files['file']
    category = request.form.get('category')
    print(category, file=sys.stdout)
    print(file, file=sys.stdout)
    if file is not None:
        return jsonify({'msg': 'success'})
    res = jsonify({'msg': 'No input file given.'})
    return res


@app.route('/pred', methods=['POST'])
def predict():
    file = request.files['file']
    if file is not None:
        tensor = transform_image(file)
        pred = get_pred(tensor)
        probs = render_pred(pred)
        return jsonify({'prediction': probs})
    res = jsonify({'msg': 'No input file given.'})
    res.headers.add('Access-Control-Allow-Origin', '*')
    return res


@app.route('/v2/pred', methods=['POST'])
def predict2():
    file = request.files['file']
    if file is not None:
        tensor = to_device(transform_image_v2(file), device)
        pred = get_pred_v2(tensor, threshold=0.1)
        return jsonify({'prediction': pred})
    res = jsonify({'msg': 'No input file given.'})
    res.headers.add('Access-Control-Allow-Origin', '*')
    return res
