import os
import numpy as np
import keras.callbacks
import cv2
from captcha_utils.icp_factory import GenCaptcha
from scipy import ndimage
import keras.preprocessing.image


def image_blur(image):
    # row, col, ch = image.shape
    # mean = 0
    # var = 0.1
    # sigma = var ** 0.5
    # gauss = np.random.normal(mean, sigma, (row, col, ch))
    # gauss = gauss.reshape(row, col, ch)
    # noisy = image + gauss
    # return noisy
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*image.shape) * severity, 1)
    img_speck = (image + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


class TextImageGenerator(keras.callbacks.Callback):
    # 所有可能字符
    LABELS = '0123456789abcdefghijklmnopqrstuvwxyz '

    def __init__(self, train_path, validate_path, img_w, img_h, channel, downsample_factor, absolute_max_string_len=6):
        """
        Args:
            train_path: 训练数据路径
            validate_path: 验证图片路径
            img_w:
            img_h:
            channel:
            downsample_factor: TODO 未知
            absolute_max_string_len: 最大字符串长度
        """
        self.img_w = img_w
        self.img_h = img_h
        self.channel = channel
        self.train_path = train_path
        self.validate_path = validate_path
        self.downsample_factor = downsample_factor
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len
        # 数据
        self.train_imgs = self.get_all_imgs(self.train_path)
        self.validate_imgs = self.get_all_imgs(self.validate_path)
        self.cur_idx = 0
        self.cur_train_idx = 0
        self.cur_vald_idx = 0
        # 打乱
        np.random.shuffle(self.train_imgs)
        np.random.shuffle(self.validate_imgs)

    def get_all_imgs(self, path):
        return [os.path.join(path, i) for i in os.listdir(path)]

    def get_output_size(self):
        return len(self.LABELS) + 1

    def char2idx(self, char):
        idx = self.LABELS.find(char.lower())
        return idx if idx != -1 else self.blank_label

    @staticmethod
    def labels_to_text(labels):
        ret = []
        for c in labels:
            if c == len(TextImageGenerator.LABELS):  # CTC Blank
                ret.append("")
            else:
                ret.append(TextImageGenerator.LABELS[c])
        return "".join(ret)

    def path2matrix(self, path):
        """
        input shape: (batch_size, w, h, channel)
        """
        img = cv2.imread(path)
        img = self.formatCaptcha(img)
        return img

    @classmethod
    def formatCaptcha(cls, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.
        #         img_transpose = np.einsum('hw->wh', img)
        img = np.expand_dims(img, axis=-1)
        # rotation
        img = keras.preprocessing.image.random_rotation(img, np.random.randint(0, 8))
        # noise
        img = image_blur(img)
        return img

    def get_next_batch(self, paths, cur_idx, batch_size=32):
        def get_label(img_path):
            """
            获取验证码对应的字符串
            """
            return os.path.basename(img_path).split('_')[-1].split('.')[0].lower()

        i = 0
        X_data = np.zeros((batch_size, self.img_h, self.img_w, self.channel))
        labels = np.zeros((batch_size, self.absolute_max_string_len))
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        source_str = []
        while i < batch_size:
            if cur_idx >= len(paths):
                # 归零，洗牌
                cur_idx = 0
                np.random.shuffle(paths)
            img_path = paths[self.cur_train_idx]
            label_text = get_label(img_path)
            captcha = self.path2matrix(img_path)
            X_data[i, :, :captcha.shape[1], :] = captcha
            input_length[i] = self.img_w // self.downsample_factor - 2
            label_length[i] = len(label_text)
            labels[i] = [self.char2idx(char) for char in label_text]
            source_str.append(label_text)

            cur_idx += 1
            i += 1

        inputs = {
            'the_input': X_data,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
            'source_str': source_str  # used for visualization only
        }
        outputs = {'ctc': np.zeros([batch_size])}
        return inputs, outputs

    def get_next_train(self, batch_size=32):
        while True:
            yield self.get_next_batch(self.train_imgs, batch_size)

    def get_next_val(self, batch_size=100):
        while True:
            yield self.get_next_batch(self.validate_imgs, batch_size, is_train=False)


class RandomTextImageGenerator(TextImageGenerator):
    """随机使用生成的验证码或者保存的"""

    def __init__(self, train_path, validate_path, img_w, img_h, channel, downsample_factor, absolute_max_string_len=6):
        self.diy_gen = GenCaptcha()
        super(RandomTextImageGenerator, self).__init__(train_path, validate_path, img_w, img_h, channel,
                                                       downsample_factor, absolute_max_string_len)

    def get_next_batch(self, paths, batch_size=32, is_random=True):
        def get_label(img_path):
            """
            获取验证码对应的字符串
            """
            return os.path.basename(img_path).split('_')[-1].split('.')[0].lower()

        i = 0
        X_data = np.zeros((batch_size, self.img_h, self.img_w, self.channel))
        labels = np.zeros((batch_size, self.absolute_max_string_len))
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        source_str = []
        while i < batch_size:
            if self.cur_train_idx >= len(paths):
                # 归零，洗牌
                self.cur_train_idx = 0
                np.random.shuffle(paths)
            is_use_diy = np.random.random() > 0.2
            if is_random and is_use_diy:
                captcha, label_text = self.diy_gen.gen_one()
                captcha = self.formatCaptcha(captcha)
            else:
                img_path = paths[self.cur_train_idx]
                label_text = get_label(img_path)
                captcha = self.path2matrix(img_path)

            X_data[i, :, :captcha.shape[1], :] = captcha
            input_length[i] = self.img_w // self.downsample_factor - 2
            label_length[i] = len(label_text)
            labels[i] = [self.char2idx(char) for char in label_text]
            source_str.append(label_text)
            i += 1
            self.cur_train_idx += 1

        inputs = {
            'the_input': X_data,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
            'source_str': source_str  # used for visualization only
        }
        outputs = {'ctc': np.zeros([batch_size])}

        return inputs, outputs

    def get_next_val_batch(self, paths, batch_size=32, is_random=False):
        def get_label(img_path):
            """
            获取验证码对应的字符串
            """
            return os.path.basename(img_path).split('_')[-1].split('.')[0].lower()

        i = 0
        X_data = np.zeros((batch_size, self.img_h, self.img_w, self.channel))
        labels = np.zeros((batch_size, self.absolute_max_string_len))
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        source_str = []
        while i < batch_size:
            if self.cur_vald_idx >= len(paths):
                # 归零，洗牌
                self.cur_vald_idx = 0
                np.random.shuffle(paths)
            # is_use_diy = np.random.random() > 0.5
            # if is_random and is_use_diy:
            #     captcha, label_text = self.diy_gen.gen_one()
            #     captcha = self.formatCaptcha(captcha)
            # else:
            img_path = paths[self.cur_vald_idx]
            label_text = get_label(img_path)
            captcha = self.path2matrix(img_path)

            X_data[i, :, :captcha.shape[1], :] = captcha
            input_length[i] = self.img_w // self.downsample_factor - 2
            label_length[i] = len(label_text)
            labels[i] = [self.char2idx(char) for char in label_text]
            source_str.append(label_text)
            i += 1
            self.cur_vald_idx += 1

        inputs = {
            'the_input': X_data,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
            'source_str': source_str  # used for visualization only
        }
        outputs = {'ctc': np.zeros([batch_size])}

        return inputs, outputs

    def get_next_val(self, batch_size=100):
        while True:
            yield self.get_next_val_batch(self.validate_imgs, batch_size, is_random=False)
