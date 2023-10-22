import os
from uuid import uuid4

import requests
# from user_agent import generate_user_agent
from random import randint
from captcha_utils.yundama import recognize_by_http
import time
import datetime


class CaptchaRequest(object):
    TIMEOUT = 10
    DIRPATH = 'images'
    HOST = 'beian.miit.gov.cn'

    def __init__(self, username, password):
        self.sess = requests.Session()
        self.username = username
        self.password = password

        self.init_sess()

    def init_sess(self):
        self.sess.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:46.0) Gecko/20100101 Firefox/46.0'}

    def get_captcha(self):
        url = 'http://%s/getVerifyCode?%d' % (self.HOST, randint(0, 100))
        return self.sess.get(url, timeout=self.TIMEOUT)

    def verifyCode(self, code):
        url = 'http://%s/common/validate/validCode.action' % self.HOST
        data = {
            'validateValue': code
        }
        return self.sess.post(url, data=data, timeout=self.TIMEOUT).json()['result']

    def recognize_captcha(self, filename):
        path = os.path.join(self.DIRPATH, filename)
        result, balance = recognize_by_http(path, self.username, self.password)
        return result

    def save_captcha(self, response):

        filename = '%s.png' % uuid4().hex
        with open(os.path.join(self.DIRPATH, filename), 'wb') as f:
            f.write(response.content)
        return filename

    def rename_captcha(self, origin_name, new_name):
        os.rename(os.path.join(self.DIRPATH, origin_name), os.path.join(self.DIRPATH, new_name))

    def remove_captcha(self, filename):
        os.remove(os.path.join(self.DIRPATH, filename))

    def run(self, num=5000):
        count = 0
        while count < num:
            try:
                filename = ''
                response = self.get_captcha()
                filename = self.save_captcha(response)
                code = self.recognize_captcha(filename)
                if self.verifyCode(code):
                    self.rename_captcha(filename, '%s_%s.png' % (uuid4().hex, code))
                    count += 1
                    print('%s, 完成: %s, ' % (datetime.datetime.now(), count))
                else:
                    self.remove_captcha(filename)
            except Exception as e:
                print(e)
                if filename:
                    self.remove_captcha(filename)
