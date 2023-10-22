#!/usr/bin/env python3.6
from flask import Flask
from flask import request
import numpy as np
import cv2
from models import cnn_ctc
from predict import junka_predict_one
from captcha_utils.generator import TextImageGenerator
import json
import local_settings as settings

app = Flask(__name__)

# init_model
weights = settings.weight_path
model, test_func = cnn_ctc.junka_model()
model.load_weights(weights)


@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'hello'


@app.route('/predict', methods=['POST'])
def predict():
    """预测的结果

    """
    file = request.files['files']
    filestr = file.read()
    # convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # TODO 预测验证码
    ret = junka_predict_one(img, test_func)[0]

    return json.dumps({"data": ret})


if __name__ == '__main__':
    app.run(debug=False, port=2000, host="0.0.0.0")
