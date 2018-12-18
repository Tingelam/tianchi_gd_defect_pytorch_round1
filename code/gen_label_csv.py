# encoding: utf-8

import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

def gen_data():
    label_map = {'正常': 0,
                  '不导电': 1,
                  '擦花': 2,
                  '横条压凹': 3,
                  '桔皮': 4,
                  '漏底': 5,
                  '碰伤': 6,
                  '起坑': 7,
                  '凸粉': 8,
                  '涂层开裂': 9,
                  '脏点': 10,
                  '其他': 11,
                  }
    # train_1 data
    data_path = '../data/guangdong_round1_train1_20180903'
    img_list, label = [], []

    for img_name in os.listdir(data_path):
        label_name = img_name.split('2018')[0]
        if label_name == '碰凹':
            label.append('其他')
        else:
            label.append(label_name)
        img_path = data_path + '/' + img_name
        img_list.append(img_path)
        
    # data = pd.DataFrame({'img_path': img_list, 'label': label})
    # data['label'] = data['label'].map(label_map)

    # data.to_csv('../data/train1.csv', index=False)

    # train_2 data
    data_path = '../data/guangdong_round1_train2_20180916'

    for id_name in os.listdir(data_path):
        id_path = data_path + '/' + id_name
        if '无瑕疵样本' in id_path:
            for img_name in os.listdir(id_path):
                img_path = id_path + '/' + img_name
                img_list.append(img_path)
                label.append('正常')
        else:
            for next_id_name in os.listdir(id_path):
                next_id_path = id_path + '/' + next_id_name
                if next_id_name != '其他':
                    for img_name in os.listdir(next_id_path):
                        img_path = next_id_path + '/' + img_name
                        img_list.append(img_path)
                        label.append(next_id_name)
                else:
                    for third_id_name in os.listdir(next_id_path):
                        third_id_path = next_id_path + '/' + third_id_name
                        if os.path.isdir(third_id_path):
                            for img_name in os.listdir(third_id_path):
                                if '.DS_Store' in img_name:
                                    continue
                                img_path = third_id_path + '/' + img_name
                                img_list.append(img_path)
                                label.append(next_id_name)

    data = pd.DataFrame({'img_path': img_list, 'label': label})
    data['label'] = data['label'].map(label_map)

    data.to_csv('../data/train.csv', index=False)

    # test_a data
    test_data_path = '../data/guangdong_round1_test_a_20180916'
    all_test_img = os.listdir(test_data_path)
    all_test_img.sort(key=lambda x: int(os.path.splitext(x)[0]))

    test_img_path = []

    for img_name in all_test_img:
        if os.path.splitext(img_name)[1] == '.jpg':
            test_img_path.append(os.path.join(test_data_path, img_name))

    data = pd.DataFrame({'img_path': test_img_path})
    data.to_csv('../data/test_a.csv', index=False)

    # test_b data
    test_data_path = '../data/guangdong_round1_test_b_20181009'
    all_test_img = os.listdir(test_data_path)
    all_test_img.sort(key=lambda x: int(os.path.splitext(x)[0]))

    test_img_path = []

    for img_name in all_test_img:
        if os.path.splitext(img_name)[1] == '.jpg':
            test_img_path.append(os.path.join(test_data_path, img_name))

    data = pd.DataFrame({'img_path': test_img_path})
    data.to_csv('../data/test_b.csv', index=False)