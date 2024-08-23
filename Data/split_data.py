import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
def split_data(df_path,out_path,train=0.05,val=0.1):
    df = pd.read_csv(df_path)
    df = df[['SMILES', 'Protein','Y']]
    df = df.dropna(subset=['SMILES'])
    df = df.dropna(subset=['Protein'])
    data_len = len(df)
    train = int(data_len*train)
    val = int(data_len*val)
    test = data_len-train-val
    df_train,df_test = train_test_split(df, train_size=train)
    df_val,df_test = train_test_split(df_test,train_size=val)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    df_train.to_csv(os.path.join(out_path, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(out_path, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(out_path, 'test.csv'), index=False)

def split_unseen_data(df_path,out_path,unseen_name,unseen_ratio=0.2,val=0.7):
    df = pd.read_csv(df_path)
    df = df[['SMILES', 'Protein','Y']]
    df = df.dropna(subset=['SMILES'])
    df = df.dropna(subset=['Protein'])
    # 获取'SMILES'列的唯一值
    unique_smiles = df[unseen_name].unique()

    # 随机打乱这些唯一值
    np.random.shuffle(unique_smiles)

    # 计算要选择的元素数量
    num_val_test = int(len(unique_smiles) *unseen_ratio)

    # 选择20%的'SMILES'作为验证和测试集
    val_test_smiles = unique_smiles[:num_val_test]

    # 筛选出包含这些'SMILES'的行作为验证和测试集
    df_val_test = df[df['SMILES'].isin(val_test_smiles)].reset_index(drop=True)

    # 从原始数据集中移除这些行，得到训练集
    df_train = df[~df['SMILES'].isin(val_test_smiles)].reset_index(drop=True)
    # 如果您需要进一步将验证集和测试集分开，可以再次随机打乱并分割
    #
    df_val = df_val_test[:int(len(df_val_test) *val)]
    df_test = df_val_test[int(len(df_val_test) *val):]
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    df_train.to_csv(os.path.join(out_path, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(out_path, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(out_path, 'test.csv'), index=False)