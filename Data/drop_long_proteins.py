import os.path

import torch
from sklearn.model_selection import train_test_split
import esm
import pandas as pd
from collections import OrderedDict
savename = 'random_5'

data1 = pd.read_csv('../Data/biosnap/full.csv')
# proteins = list(OrderedDict.fromkeys(data1['Protein']))
# num_counts = data1['Protein'].value_counts()
# num_counts = num_counts[proteins]
# proteins_list = proteins
# esm2, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# data = []
# for idx, i in enumerate(proteins):
#     data.append((f'protein{idx}', i))
# _, _, proteins_tokens = batch_converter(data)
# #
# proteins_tokens[proteins_tokens!=1] = 2
# proteins_tokens[proteins_tokens==1] = 0
# proteins_tokens[proteins_tokens==2] = 1
# proteins_len = torch.sum(proteins_tokens,dim=1).tolist()
# dict_data = {'proteins':num_counts.index,'counts':num_counts.values,'len':proteins_len}
# a = pd.DataFrame(dict_data)
#
# b = a.loc[a['len']<2000,'proteins']
#
# drop_long_data = data1.loc[data1['Protein'].isin(b.values),:]
#
# drop_long_data.to_csv('../Data/biosnap/drop_full.csv',index=False)

# 先将数据集进行拼接，要不然我们只针对样本进行采样的话，会找不到对应的标签的
train_set, test_set = train_test_split(data1, test_size=0.3)
test_set,val_set = train_test_split(test_set,test_size=0.333)
if not os.path.exists(f'../Data/biosnap/{savename}/'):
    os.mkdir(f'../Data/biosnap/{savename}/')
train_set.to_csv(f'../Data/biosnap/{savename}/train.csv',index=False)
test_set.to_csv(f'../Data/biosnap/{savename}/test.csv',index=False)
val_set.to_csv(f'../Data/biosnap/{savename}/val.csv',index=False)
print(len(train_set))
