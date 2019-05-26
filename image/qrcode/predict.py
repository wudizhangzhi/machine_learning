import numpy as np

from models import cnn_ctc
from captcha_utils.generator import TextImageGenerator


def junka_predict_one(captcha, weights=None):
    model, test_func = cnn_ctc.junka_model()
    model.load_weights(weights)
    captcha = TextImageGenerator.formatCaptcha(captcha)
    captcha = np.expand_dims(captcha, axis=0)
    config = cnn_ctc.config
    x = np.zeros((1, config['img_h'], config['img_w'], 1))
    x[0, :, :captcha.shape[2], :] = captcha
    ret = cnn_ctc.decode_batch(test_func, TextImageGenerator.labels_to_text, x)
    return ret


if __name__ == '__main__':
    import cv2

    weight_path = 'E:\\Workplace\\bdzh\\MachineLearning\\SmallCaptcha\\image_ocr\\2019_05_24_21_49_38\\weights70_acc_87.00000.h5'
    test_path = 'E:\\test.png'
    captcha = cv2.imread(test_path)
    print(junka_predict_one(captcha, weight_path)[0])
