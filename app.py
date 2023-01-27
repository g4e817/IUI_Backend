from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello_world():
    res = jsonify({'hi': 2})
    res.headers.add('Access-Control-Allow-Origin', '*')
    return res
