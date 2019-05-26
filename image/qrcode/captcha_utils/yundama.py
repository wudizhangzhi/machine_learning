from sys import argv
from os.path import dirname, join
from ctypes import *





import requests

appId = 1
appKey = '22cc5376925e9387a23cf797cb9ba745'



def recognize_common(filename, username, password):
    print('\r\n>>>??????...')
    lib_path = join(dirname(argv[0]), 'yundamaAPI-x64')
    YDMApi = windll.LoadLibrary(lib_path)
    YDMApi.YDM_SetAppInfo(appId, appKey)

    uid = YDMApi.YDM_Login(username, password)

    if uid > 0:

        print('>>>?????????...')

        balance = YDMApi.YDM_GetBalance(username, password)


        codetype = 1000

        result = c_char_p("                              ")

        timeout = 60
        YDMApi.YDM_SetTimeOut(timeout)

        captchaId = YDMApi.YDM_DecodeByPath(filename, codetype, result)
        print('captchaId', captchaId, type(captchaId))
        if captchaId < 0:
            return '', ''

        return str(result)[10: -2], balance
    else:
        return False, False


def recognize_by_byte(filename, username, password):
    lib_path = join(dirname(argv[0]), 'yundamaAPI-x64')
    YDMApi = windll.LoadLibrary(lib_path)
    result = c_char_p("                              ")
    codetype = 1000
    timeout = 60
    lpBuffer = ''
    nNumberOfBytesToRead = ''
    captchaId = YDMApi.YDM_EasyDecodeByBytes(username, password, appId, appKey, filename, codetype, timeout, result)
    if captchaId < 0:
        return '', ''
    return str(result)[10: -2], None


################################################################################
# import httplib, mimetypes, urlparse, json, time
import time

class YDMHttp:
    # apiurl = 'http://api.yundama.com/api.php'
    apiurl = 'http://api.yundama.net:5678/api.php'

    username = ''
    password = ''
    appid = ''
    appkey = ''

    def __init__(self, username, password, appid, appkey):
        self.username = username
        self.password = password
        self.appid = str(appid)
        self.appkey = appkey

    def request(self, fields, files=[]):
        try:
            # response = post_url(self.apiurl, fields, files)
            # print('response', response)
            # response = json.loads(response)
            if files:
                f = open(files['file'], 'rb')
                files['file'] = f
            # _files = files if files else None
            response = requests.post(self.apiurl, data=fields, files=files if files else None, timeout=10)
            response = response.json()
        except Exception as e:
            print(e)
            response = None
        return response

    def balance(self):
        data = {'method': 'balance', 'username': self.username, 'password': self.password, 'appid': self.appid,
                'appkey': self.appkey}
        response = self.request(data)
        if (response):
            if (response['ret'] and response['ret'] < 0):
                return response['ret']
            else:
                return response['balance']
        else:
            return -9001

    def login(self):
        data = {'method': 'login', 'username': self.username, 'password': self.password, 'appid': self.appid,
                'appkey': self.appkey}
        response = self.request(data)
        if (response):
            if (response['ret'] and response['ret'] < 0):
                return response['ret']
            else:
                return response['uid']
        else:
            return -9001

    def upload(self, filename, codetype, timeout):
        data = {'method': 'upload', 'username': self.username, 'password': self.password, 'appid': self.appid,
                'appkey': self.appkey, 'codetype': str(codetype), 'timeout': str(timeout)}
        file = {'file': filename}
        response = self.request(data, file)
        if (response):
            if (response['ret'] and response['ret'] < 0):
                return response['ret']
            else:
                return response['cid']
        else:
            return -9001

    def result(self, cid):
        data = {'method': 'result', 'username': self.username, 'password': self.password, 'appid': self.appid,
                'appkey': self.appkey, 'cid': str(cid)}
        response = self.request(data)
        return response and response['text'] or ''

    def decode(self, filename, codetype, timeout):
        cid = self.upload(filename, codetype, timeout)
        if (cid > 0):
            for i in range(0, timeout):
                result = self.result(cid)
                if (result != ''):
                    return cid, result
                else:
                    time.sleep(1)
            return -3003, ''
        else:
            return cid, ''


######################################################################
appid = 1

appkey = '22cc5376925e9387a23cf797cb9ba745'


# username = ''
# password = ''

# print 'uid: %s' % uid


def recognize_by_http(filename, username, password):
    yundama = YDMHttp(username, password, appid, appkey)
    uid = yundama.login()
    balance = yundama.balance()
    codetype = 1000
    timeout = 60
    cid, result = yundama.decode(filename, codetype, timeout)
    return result, balance


if __name__ == '__main__':
    from base import cf
    print(recognize_by_http('tmp/captcha_bd@sjyx8.cn.png', cf.get('captcha', 'username'), cf.get('captcha', 'password')))
