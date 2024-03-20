import os
import warnings
import numpy as np
import pandas as pd
import random
from osgeo import gdal
from PIL import Image
import config

config = config.config
import pickle


def normalize(img):
    # 计算均值和标准差
    mean = img.mean()
    std = img.std()
    # 进行标准化
    normalized_img = (img - mean) / std
    return normalized_img


def normalize_to_0_1(img):
    # 计算最小值和最大值
    min_val = img.min()
    max_val = img.max()
    # 进行归一化
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img


def resample_tif(img_re):
    """
    :param img: original factors data
    :return: resampled factors data
    """
    warnings.filterwarnings("ignore")
    img_re = np.array(Image.fromarray(img_re).resize((config["height"], config["width"])))
    return img_re


def read_data_from_tif(tif_path):
    """
    读取影响因子数据并转换为nparray
    """
    tif = gdal.Open(tif_path)
    w, h = tif.RasterXSize, tif.RasterYSize
    img = np.array(tif.ReadAsArray(0, 0, w, h).astype(np.float32))
    if w != config["width"] and h != config["height"]:
        img = resample_tif(img)
    return img


def get_feature_data():
    """"
    Read features and normalize
    """
    tif_paths = config["data_path"]
    data = np.zeros((config["feature"], config["height"], config["width"])).astype(np.float32)
    for i, tif_path in enumerate(tif_paths):
        img = read_data_from_tif(tif_path)
        img[img == -np.finfo(np.float32).max] = 0
        if config["normalize"] and not config["normalize_to_0_1"]:
            data[i, :, :] = normalize(img)
        elif not config["normalize"] and config["normalize_to_0_1"]:
            data[i, :, :] = normalize_to_0_1(img)
        # Choose to use standardization or normalization
        elif config["normalize"] and config["normalize_to_0_1"]:
            raise ValueError("config['normalize'] 和 config['normalize_to_0_1'] 不能同时为 True")
        else:
            data[i, :, :] = img
    return data

class creat_dataset():

    def __init__(self, tensor_data, n):
        self.data = tensor_data
        self.n = int(n)
        self.p = int((n - 1) / 2)
        self.F = tensor_data.shape[0]
        self.w = tensor_data.shape[1]
        self.h = tensor_data.shape[2]
        self.all_results = None

    def creat_new_tensor(self):
        # Expand image edges
        new_tensor = np.zeros((self.F, self.w + self.n - 1, self.h + self.n - 1))
        # Assign data values to the center of the image
        new_tensor[:, self.p:self.w + self.p, self.p:self.h + self.p] = self.data
        return new_tensor

    def get_images_labels(self, data, labels, mode='train'):
        train_images, train_labels = [], []
        valid_images, valid_labels = [], []
        count_0, count_1, count_2, count_3 = 0, 0, 0, 0
        if self.all_results is not None:
            train_images, train_labels, valid_images, valid_labels = self.all_results
            if mode == "train":
                print('training set： ' + str(len(train_images)), str(len(train_labels)))
                return train_images, train_labels
            else:
                print('test set： ' + str(len(valid_images)), str(len(valid_labels)))
                return valid_images, valid_labels
        # 遍历label
        for i in range(config["height"]):
            for j in range(config["width"]):
                # training set
                if labels[i, j] == 0 or labels[i, j] == 2:
                    # reading factors size*size*factor
                    train_images.append(data[:, i - self.p:i + self.p + 1, j - self.p:j + self.p + 1].astype(np.float32))
                    # landslides
                    if labels[i, j] == 0:
                        count_0 += 1
                        train_labels.append(1)
                    # non-landslides
                    if labels[i, j] == 2:
                        count_2 += 1
                        train_labels.append(0)
                # validation set
                if labels[i, j] == 1 or labels[i, j] == 3:
                    valid_images.append(data[:, i - self.p:i + self.p + 1, j - self.p:j + self.p + 1].astype(np.float32))
                    # landslides
                    if labels[i, j] == 1:
                        count_1 += 1
                        valid_labels.append(1)
                    # non-landslide
                    if labels[i, j] == 3:
                        count_3 += 1
                        valid_labels.append(0)
        print("No. of label as 0，1，2，3 are {},{},{},{}".format(count_0, count_1, count_2, count_3))
        if self.all_results is None:
            self.all_results = train_images, train_labels, valid_images, valid_labels
        if mode == "train":
            print('training set： ' + str(len(train_images)), str(len(train_labels)))
            return train_images, train_labels
        else:
            print('test set： ' + str(len(valid_images)), str(len(valid_labels)))
            return valid_images, valid_labels


def get_train_data(config, creat):
    data = creat.creat_new_tensor()  
    labels = read_data_from_tif(config["label_path"])
    train_images, train_labels = creat.get_images_labels(data, labels, mode='train')
    return train_images, train_labels


def get_test_data(config, creat):
    data = creat.creat_new_tensor()
    labels = read_data_from_tif(config["label_path"])
    valid_images, valid_labels = creat.get_images_labels(data, labels, mode='valid')
    return valid_images, valid_labels


def shuffle_image_label_0(images, labels):
    """
    Randomly disrupt two list with the same shuffle
    """
    # Randomly shuffle the order of two lists (images and labels) while maintaining the correspondence between them
    # This is usually used in the data enhancement or data set preparation stage to improve the model's generalization ability and training effect.
    randnum = random.randint(0, len(images) - 1)
    random.seed(randnum)
    random.shuffle(images)
    random.seed(randnum)
    random.shuffle(labels)
    return images, labels


def train_data(creat):
    data_file = 'train_data.pkl'
    if os.path.exists(data_file):
        saved_config_dict = {}
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        saved_config = data.get('config', {})
        for item in saved_config:
            key = item[0]
            value = item[1]
            saved_config_dict[key] = value
        keys = ['label_path', 'size', 'feature', 'width', 'height', 'normalize', 'normalize_to_0_1']
        values = [config.get(key) for key in keys]
        saved_values = [saved_config_dict.get(key) for key in keys]
        if np.all(values == saved_values):
            train_images = data['train_images']
            train_labels = data['train_labels']
        else:
            train_images, train_labels = get_train_data(config, creat)
            config_list = [[k, v] for k, v in config.items()]
            data = {'train_images': train_images, 'train_labels': train_labels, 'config': config_list}
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
    else:
        train_images, train_labels = get_train_data(config, creat)
        config_list = [[k, v] for k, v in config.items()]
        data = {'train_images': train_images, 'train_labels': train_labels, 'config': config_list}
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
    train_images, train_labels = shuffle_image_label_0(train_images, train_labels)
    return np.array(train_images).reshape((-1, config["feature"], config["size"], config["size"])), np.array(
            train_labels).reshape((-1, 1))


def test_data(creat):
    data_file = 'valid_data.pkl'
    if os.path.exists(data_file):
        saved_config_dict = {}
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        saved_config = data.get('config', {})
        for item in saved_config:
            key = item[0]
            value = item[1]
            saved_config_dict[key] = value
        keys = ['label_path', 'size', 'feature', 'width', 'height', 'normalize', 'normalize_to_0_1']
        values = [config.get(key) for key in keys]
        saved_values = [saved_config_dict.get(key) for key in keys]
        if np.all(values == saved_values):
            valid_images = data['valid_images']
            valid_labels = data['valid_labels']
        else:
            valid_images, valid_labels = get_test_data(config, creat)
            config_list = [[k, v] for k, v in config.items()]
            data = {'valid_images': valid_images, 'valid_labels': valid_labels, 'config': config_list}
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
    else:
        valid_images, valid_labels = get_test_data(config, creat)
        config_list = [[k, v] for k, v in config.items()]
        data = {'valid_images': valid_images, 'valid_labels': valid_labels, 'config': config_list}
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
    return np.array(valid_images).reshape((-1, config["feature"], config["size"], config["size"])), np.array(
            valid_labels).reshape((-1, 1))


def save_to_tif(pred_result, save_path):
    """
    :save LSM
    """
    img = pred_result.reshape((config["height"], config["width"]))
    im_geotrans, im_prof = [], []
    for tif_path in config["data_path"]:  # Get the affine matrix and projection coordinates
        tif = gdal.Open(tif_path)
        im_geotrans.append(tif.GetGeoTransform())
        im_prof.append(tif.GetProjection())

    if 'int8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    im_height, im_width = img.shape

    # Create a file
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(save_path, im_width, im_height, 1, datatype)
    dataset.GetRasterBand(1).WriteArray(img)
    dataset.SetGeoTransform(im_geotrans[0])
    dataset.SetProjection(im_prof[0])
    print('ok')
