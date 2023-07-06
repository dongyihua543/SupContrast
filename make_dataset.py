# -*- coding: utf-8 -*-
# ModuleName: make_dataset
# Author: dongyihua543
# Time: 2023/7/6 13:43


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras.losses import sparse_categorical_crossentropy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# load data
user_order_data = "data/dataset_user_order_xiaoka.csv"
base_data = "data/dataset_base_xiaoka.csv"


def data_processing(file_name=None):
    total_df = pd.read_csv(file_name).head(100)
    print("total_df len: {}".format(len(total_df)))

    dict_df = total_df[["origin_id"]].drop_duplicates().copy()
    dict_df["cate_no"] = dict_df.reset_index().index
    total_df = pd.merge(total_df, dict_df, how="inner", on="origin_id")

    classes = len(total_df.origin_id.drop_duplicates())
    print("classes len: {}".format(classes))

    total_df["data"] = total_df["data"].apply(lambda x: eval(x))

    expand_df = total_df["data"].apply(lambda x: pd.Series(x))
    expand_df.rename(columns=lambda x: "fea_" + str(x), inplace=True)

    total_df.drop(columns=["data"], inplace=True)
    total_df = pd.concat([total_df, expand_df], axis=1)
    return total_df


# 用户挂单数据
df_user_order = data_processing(user_order_data)

# 基础数据
df_base = data_processing(base_data)

# 两种数据合并
df_merge = pd.concat([df_user_order, df_base], axis=0)

classes = len(df_merge.origin_id.drop_duplicates())
print("classes len: {}".format(classes))

df_train, df_test = train_test_split(df_merge, test_size=0.1, random_state=4)

features = [i for i in df_merge.columns if i.startswith("fea_")]

train_X = df_train[features]
train_Y = df_train["cate_no"]
test_X = df_test[features]
test_Y = df_test["cate_no"]

# 训练数据
x_train, y_train, x_test, y_test = train_X.values, train_Y.values, test_X.values, test_Y.values

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

x_train = x_train.float()
y_train = y_train.long()
x_test = x_test.float()
y_test = y_test.long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)



# class NumpyDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         # 在这里对数据进行预处理或转换
#         # 返回的数据应该是一个样本和其对应的标签（如果有的话）
#         return item, None  # 返回数据样本和标签
#
#
# # 假设有一个名为 `numpy_data` 的 NumPy 数组
# numpy_data = ...  # 假设为 NumPy 数组
#
# # 创建自定义数据集实例
# dataset = NumpyDataset(numpy_data)
#
# # 使用 DataLoader 进行数据加载
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
#
# # 遍历数据集
# for batch in dataloader:
#     # 在这里进行训练或推理操作
#     # 每个 batch 是一个包含 64 个样本的数据批次
#     input_data, labels = batch
#     # 进行模型的前向计算、损失计算等操作
