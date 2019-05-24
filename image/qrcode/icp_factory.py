# coding: utf-8

"""
target captcha url: http://www.miitbeian.gov.cn/getVerifyCode?4
"""
import json

from model.capthafactory import CaptchaFactory
import numpy as np


def custom_fn(single_char):
    # do something
    # return single_char.filter(ImageFilter.GaussianBlur)
    return single_char


def bg_custom_fn(bg):
    # do something
    # return bg.filter(ImageFilter.GaussianBlur)
    return bg


def main():
    project_name = "icp"
    with open("configs/icp.json", encoding="utf-8") as fp:
        demo_config = json.load(fp)

    # with open("configs/char/specific_chars.json", encoding="utf-8") as fp:
    #     specific = json.load(fp)

    demo_factory = CaptchaFactory(char_custom_fns=[custom_fn], bg_custom_fns=[bg_custom_fn], **demo_config)
    number = 10000 * 50
    while number:
        # captcha = demo_factory.generate_captcha(specific_chars=specific)
        captcha = demo_factory.generate_captcha()
        captcha.save("output/%s/%s.jpg" % (project_name, captcha.text))
        # print(captcha.text, captcha.num)
        print(number)
        number -= 1


class GenCaptcha(object):
    def __init__(self):
        with open("configs/icp.json", encoding="utf-8") as fp:
            demo_config = json.load(fp)
        # self.config = demo_config
        self.factory = CaptchaFactory(**demo_config)

    def gen_one(self):
        captcha = self.factory.generate_captcha()
        return np.array(captcha.captcha), ''.join([i.char_text.lower() for i in captcha.chars])


def test():
    gen = GenCaptcha()
    captcha, text = gen.gen_one()
    print(captcha.shape, text)
    captcha, text = gen.gen_one()
    print(captcha.shape, text)


if __name__ == "__main__":
    # main()
    test()
