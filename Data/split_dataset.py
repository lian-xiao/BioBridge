import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

from scaffold_split import generate_scaffolds


def filter_both_label(df, min=6, cluster_name='target_cluster'):
    grouped = df.groupby(cluster_name)

    # 步骤2: 对每个分组检查'Y'列中是否同时存在0和1
    def check_conditions(group):
        count_0 = (group['Y'] == 0).sum()
        count_1 = (group['Y'] == 1).sum()
        return count_0 >= min and count_1 >= min

    mixed_clusters = grouped.apply(check_conditions).dropna()
    # 步骤3: 记录下满足条件的'cluster'的名称
    clusters_without_both_labels = mixed_clusters[~mixed_clusters].index
    clusters_with_both_labels = mixed_clusters[mixed_clusters].index
    data_with_both_labels = df[df[cluster_name].isin(clusters_with_both_labels)]
    data_without_both_labels = df[df[cluster_name].isin(clusters_without_both_labels)]
    return data_with_both_labels.reset_index(drop=True), data_without_both_labels.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='settings')
    parser.add_argument('--dataset', type=str, default="biosnap",
                        choices=["human", "celegans", "bindingdb", "biosnap"], help='select dataset for training')
    parser.add_argument('--split_settings', type=str, default="meta_unseen_drug",
                        choices=["random", "cold", "cluster", "meta_unseen_protein", "meta_unseen_drug",
                                 "unseen_protein", "unseen_smiles"], help='select split settings')
    args = parser.parse_args()

    data_path = os.path.join('../Data', args.dataset)
    dir_path = os.path.join(data_path, args.split_settings)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data_path = os.path.join(data_path, "fulldata.csv")
    full = pd.read_csv(data_path)
    if args.split_settings == 'cluster':
        source = []
        target = []
        trg_cluster = np.array(list(set(full['target_cluster'].values)))
        drug_cluster = np.array(list(set(full['drug_cluster'].values)))
        print(trg_cluster.shape)
        print(drug_cluster.shape)
        print(max(full['target_cluster'].values))
        print(max(full['drug_cluster'].values))
        np.random.shuffle(trg_cluster)
        np.random.shuffle(drug_cluster)
        trg_src, trg_trg = np.split(trg_cluster, [int(0.6 * trg_cluster.shape[0])])
        drug_src, drug_trg = np.split(drug_cluster, [int(0.6 * drug_cluster.shape[0])])
        print(full.values.shape)

        smiledictnum2str = {};
        smiledictstr2num = {}
        sqddictnum2str = {};
        sqdictstr2num = {}
        trainsamples = []
        valtest_example = []
        smilelist = []
        sequencelist = []
        for no, data in enumerate(full.values):
            smiles, sequence, interaction, d_cluster, t_cluster = data
            if smiledictstr2num.get(smiles) == None:
                smiledictstr2num[smiles] = len(smiledictstr2num)
                smiledictnum2str[str(len(smiledictnum2str))] = smiles
                smilelist.append(smiles)
            if sqdictstr2num.get(sequence) == None:
                sqdictstr2num[sequence] = len(sqdictstr2num)
                sqddictnum2str[str(len(sqddictnum2str))] = sequence
                sequencelist.append(sequence)
            if d_cluster in drug_src and t_cluster in trg_src:
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])
                trainsamples.append([smilesidx, sequenceidx, int(interaction)])
                source.append(data)
            if d_cluster in drug_trg and t_cluster in trg_trg:
                target.append(data)
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])
                valtest_example.append([smilesidx, sequenceidx, int(interaction)])
        print(len(source), len(target))

        valsamples = valtest_example[:int(0.8 * len(valtest_example))]
        testsamples = valtest_example[int(0.8 * len(valtest_example)):]
        target_train = target[0:int(0.8 * len(target))]
        target_test = target[int(0.8 * len(target)):]
        column = ['SMILES', 'Protein', 'Y', 'drug_cluster', 'target_cluster']
        source_train_set = pd.DataFrame(columns=column, data=source)
        source_train_set.to_csv(dir_path + '/source_train.csv', index=False)
        target_train_set = pd.DataFrame(columns=column, data=target_train)
        target_train_set.to_csv(dir_path + '/target_train.csv', index=False)
        target_test_set = pd.DataFrame(columns=column, data=target_test)
        target_test_set.to_csv(dir_path + '/target_test.csv', index=False)
        print(len(target_train), len(target_test))
    elif args.split_settings == 'meta_unseen_protein':
        source = []
        target_train = []
        target_test = []
        data_with_both_labels, data_without_both_labels = filter_both_label(full)

        source_trg_cluster = np.array(list(set(data_without_both_labels['target_cluster'].values)))
        source_drug_cluster = np.array(list(set(data_without_both_labels['drug_cluster'].values)))

        trg_cluster = np.array(list(set(data_with_both_labels['target_cluster'].values)))
        drug_cluster = np.array(list(set(data_with_both_labels['drug_cluster'].values)))
        # print(source_trg_cluster.shape)
        # print(source_drug_cluster.shape)

        value_counts = data_with_both_labels['target_cluster'].value_counts(normalize=True) * 100  # 转换为百分比
        # 步骤2: 计算前40%的阈值
        cumulative_sum = value_counts.cumsum()  # 计算累积和
        threshold_index = (cumulative_sum <= 40).sum()  # 找到累积和小于或等于40%的索引数量

        # 步骤3: 获取前40%的'target_cluster'值
        top_clusters = value_counts.iloc[:threshold_index]
        bottom_clusters = value_counts.iloc[threshold_index:]
        # 步骤4: 从原始DataFrame中筛选出包含这些'target_cluster'的行
        top_with_both_labels = data_with_both_labels[data_with_both_labels['target_cluster'].isin(top_clusters.index)]
        bottom_with_both_labels = data_with_both_labels[
            data_with_both_labels['target_cluster'].isin(bottom_clusters.index)]

        # 将source中划分比较多的cluster，target划分比较少的cluster
        # trg_src, trg_trg = np.split(trg_cluster, [int(0.4 * trg_cluster.shape[0])])
        trg_src = np.array(list(set(top_with_both_labels['target_cluster'].values)))
        trg_cluster = np.array(list(set(bottom_with_both_labels['target_cluster'].values)))
        np.random.shuffle(trg_cluster)
        trg_trg_train, trg_trg_test = np.split(trg_cluster, [int(0.7 * trg_cluster.shape[0])])
        print(full.values.shape)

        smiledictnum2str = {}
        smiledictstr2num = {}
        sqddictnum2str = {}
        sqdictstr2num = {}
        trainsamples = []
        valtest_example = []
        smilelist = []
        sequencelist = []
        for no, data in enumerate(data_with_both_labels.values):
            smiles, sequence, interaction, d_cluster, t_cluster = data
            if smiledictstr2num.get(smiles) == None:
                smiledictstr2num[smiles] = len(smiledictstr2num)
                smiledictnum2str[str(len(smiledictnum2str))] = smiles
                smilelist.append(smiles)
            if sqdictstr2num.get(sequence) == None:
                sqdictstr2num[sequence] = len(sqdictstr2num)
                sqddictnum2str[str(len(sqddictnum2str))] = sequence
                sequencelist.append(sequence)
            if t_cluster in trg_src:
                # and d_cluster in drug_src
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])
                # trainsamples.append([smilesidx, sequenceidx, int(interaction)])
                source.append(data)
            if t_cluster in trg_trg_train:
                # and t_cluster in trg_trg d_cluster in drug_trg
                target_train.append(data)
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])
                # valtest_example.append([smilesidx, sequenceidx, int(interaction)])
            if t_cluster in trg_trg_test:
                # and t_cluster in trg_trg d_cluster in drug_trg
                target_test.append(data)
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])

        for no, data in enumerate(data_without_both_labels.values):
            source.append(data)
        print(len(source), len(target_train), len(target_test))
        column = ['SMILES', 'Protein', 'Y', 'drug_cluster', 'target_cluster']
        source_train_set = pd.DataFrame(columns=column, data=source)
        source_train_set.to_csv(dir_path + '/source_train.csv', index=False)
        target_train_set = pd.DataFrame(columns=column, data=target_train)
        target_train_set.to_csv(dir_path + '/target_train.csv', index=False)
        target_test_set = pd.DataFrame(columns=column, data=target_test)
        target_test_set.to_csv(dir_path + '/target_test.csv', index=False)

    elif args.split_settings == 'meta_unseen_drug':
        source = []
        target_train = []
        target_test = []
        scaffold = generate_scaffolds(list(full['SMILES']))
        new_dict = {i: key for i, key in enumerate(scaffold)}
        full.drop('drug_cluster', axis=1)
        full['drug_cluster'] = pd.NA
        for i, key in enumerate(new_dict.keys()):
            full.loc[new_dict[key], 'drug_cluster'] = key

        full = full.reindex(columns=['SMILES', 'Protein', 'Y', 'drug_cluster', 'target_cluster'])
        cluster_name = 'drug_cluster'
        data_with_both_labels, data_without_both_labels = filter_both_label(full, cluster_name=cluster_name)

        source_trg_cluster = np.array(list(set(data_without_both_labels[cluster_name].values)))
        source_drug_cluster = np.array(list(set(data_without_both_labels[cluster_name].values)))

        trg_cluster = np.array(list(set(data_with_both_labels[cluster_name].values)))
        drug_cluster = np.array(list(set(data_with_both_labels[cluster_name].values)))
        # print(source_trg_cluster.shape)
        # print(source_drug_cluster.shape)

        value_counts = data_with_both_labels[cluster_name].value_counts(normalize=True) * 100  # 转换为百分比
        # 步骤2: 计算前40%的阈值
        cumulative_sum = value_counts.cumsum()  # 计算累积和
        threshold_index = (cumulative_sum <= 40).sum()  # 找到累积和小于或等于40%的索引数量

        # 步骤3: 获取前40%的'target_cluster'值
        top_clusters = value_counts.iloc[:threshold_index]
        bottom_clusters = value_counts.iloc[threshold_index:]
        # 步骤4: 从原始DataFrame中筛选出包含这些'target_cluster'的行
        top_with_both_labels = data_with_both_labels[data_with_both_labels[cluster_name].isin(top_clusters.index)]
        bottom_with_both_labels = data_with_both_labels[
            data_with_both_labels[cluster_name].isin(bottom_clusters.index)]

        # 将source中划分比较多的cluster，target划分比较少的cluster
        # trg_src, trg_trg = np.split(trg_cluster, [int(0.4 * trg_cluster.shape[0])])
        trg_src = np.array(list(set(top_with_both_labels[cluster_name].values)))
        trg_cluster = np.array(list(set(bottom_with_both_labels[cluster_name].values)))
        np.random.shuffle(trg_cluster)
        trg_trg_train, trg_trg_test = np.split(trg_cluster, [int(0.7 * trg_cluster.shape[0])])
        print(full.values.shape)
        smiledictnum2str = {}
        smiledictstr2num = {}
        sqddictnum2str = {}
        sqdictstr2num = {}
        trainsamples = []
        valtest_example = []
        smilelist = []
        sequencelist = []
        for no, data in enumerate(data_with_both_labels.values):
            smiles, sequence, interaction, d_cluster, t_cluster  = data
            if smiledictstr2num.get(smiles) == None:
                smiledictstr2num[smiles] = len(smiledictstr2num)
                smiledictnum2str[str(len(smiledictnum2str))] = smiles
                smilelist.append(smiles)
            if sqdictstr2num.get(sequence) == None:
                sqdictstr2num[sequence] = len(sqdictstr2num)
                sqddictnum2str[str(len(sqddictnum2str))] = sequence
                sequencelist.append(sequence)
            if d_cluster in trg_src:
                # and d_cluster in drug_src
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])
                # trainsamples.append([smilesidx, sequenceidx, int(interaction)])
                source.append(data)
            if d_cluster in trg_trg_train:
                # and t_cluster in trg_trg d_cluster in drug_trg
                target_train.append(data)
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])
                # valtest_example.append([smilesidx, sequenceidx, int(interaction)])
            if d_cluster in trg_trg_test:
                # and t_cluster in trg_trg d_cluster in drug_trg
                target_test.append(data)
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])

        for no, data in enumerate(data_without_both_labels.values):
            source.append(data)
        print(len(source), len(target_train), len(target_test))

        # valsamples = valtest_example[:int(0.8 * len(valtest_example))]
        # testsamples = valtest_example[int(0.8 * len(valtest_example)):]
        # target_train = target[0:int(0.6 * len(target))]
        # target_test = target[int(0.6 * len(target)):]
        column = ['SMILES', 'Protein', 'Y', 'drug_cluster', 'target_cluster']
        source_train_set = pd.DataFrame(columns=column, data=source)
        source_train_set.to_csv(dir_path + '/source_train.csv', index=False)
        target_train_set = pd.DataFrame(columns=column, data=target_train)
        target_train_set.to_csv(dir_path + '/target_train.csv', index=False)
        target_test_set = pd.DataFrame(columns=column, data=target_test)
        target_test_set.to_csv(dir_path + '/target_test.csv', index=False)

    elif args.split_settings == 'cold':
        train = []
        valtest = []

        smiledictnum2str = {};
        smiledictstr2num = {}
        sqddictnum2str = {};
        sqdictstr2num = {}
        trainsamples = []
        valtest_example = []
        smilelist = []
        sequencelist = []

        for no, data in enumerate(full.values):
            smiles, sequence, interaction, _, _ = data
            if smiledictstr2num.get(smiles) == None:
                smiledictstr2num[smiles] = len(smiledictstr2num)
                smiledictnum2str[str(len(smiledictnum2str))] = smiles
                smilelist.append(smiles)
            if sqdictstr2num.get(sequence) == None:
                sqdictstr2num[sequence] = len(sqdictstr2num)
                sqddictnum2str[str(len(sqddictnum2str))] = sequence
                sequencelist.append(sequence)
        trg_cluster = np.array(list(sqddictnum2str.keys()), dtype=int).squeeze()
        drug_cluster = np.array(list(smiledictnum2str.keys()), dtype=int).squeeze()
        print(trg_cluster.shape)
        print(drug_cluster.shape)
        np.random.shuffle(trg_cluster)
        np.random.shuffle(drug_cluster)
        trg_src, trg_trg = np.split(trg_cluster, [int(0.7 * trg_cluster.shape[0])])
        drug_src, drug_trg = np.split(drug_cluster, [int(0.7 * drug_cluster.shape[0])])
        for no, data in enumerate(full.values):
            smiles, sequence, interaction, _, _ = data
            # smiles, sequence, interaction, d_cluster, t_cluster = data
            if smiledictstr2num.get(smiles) in drug_src and sqdictstr2num.get(sequence) in trg_src:
                train.append(data)
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])
                trainsamples.append([smilesidx, sequenceidx, int(interaction)])
            if smiledictstr2num.get(smiles) in drug_trg and sqdictstr2num.get(sequence) in trg_trg:
                valtest.append(data)
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])
                valtest_example.append([smilesidx, sequenceidx, int(interaction)])
        print(len(train), len(valtest))
        val = valtest[0:int(0.3 * len(valtest))]
        test = valtest[int(0.3 * len(valtest)):]
        valsamples = valtest_example[0:int(0.3 * len(valtest_example))]
        testsamples = valtest_example[int(0.3 * len(valtest_example)):]
        print(len(val), len(valsamples), len(test), len(testsamples))
        column = ['SMILES', 'Protein', 'Y', 'drug_cluster', 'target_cluster']
        train_set = pd.DataFrame(columns=column, data=train)
        train_set.to_csv(dir_path + '/train.csv', index=False)
        val_set = pd.DataFrame(columns=column, data=val)
        val_set.to_csv(dir_path + '/val.csv', index=False)
        test_set = pd.DataFrame(columns=column, data=test)
        test_set.to_csv(dir_path + '/test.csv', index=False)
        # column=['smiles','sequence','interactions']
        # Tr=pd.DataFrame(columns=column, data=trainsamples)
        # Tr.to_csv(dir_path+'/train/samples.csv')
        # val=pd.DataFrame(columns=column,data=valsamples)
        # val.to_csv(dir_path+'/valid/samples.csv')
        # Ts=pd.DataFrame(columns=column,data=testsamples)
        # Ts.to_csv(dir_path+'/test/samples.csv')
    elif args.split_settings == 'random':
        smiledictnum2str = {}
        smiledictstr2num = {}
        sqddictnum2str = {}
        sqdictstr2num = {}
        trainsamples = []
        valtest_example = []
        smilelist = []
        sequencelist = []
        samples = []
        data_list = []
        for no, data in enumerate(full.values):
            smiles, sequence, interaction, _, _ = data
            data_list.append(data)
            smilesidx = 0;
            sequenceidx = 0
            if smiledictstr2num.get(smiles) == None:
                smiledictstr2num[smiles] = len(smiledictstr2num)
                smiledictnum2str[str(len(smiledictnum2str))] = smiles
                smilesidx = int(smiledictstr2num[smiles])
                smilelist.append(smiles)
            else:
                smilesidx = int(smiledictstr2num.get(smiles))
            if sqdictstr2num.get(sequence) == None:
                sqdictstr2num[sequence] = len(sqdictstr2num)
                sqddictnum2str[str(len(sqddictnum2str))] = sequence
                sequenceidx = int(sqdictstr2num[sequence])
                sequencelist.append(sequence)
            else:
                sequenceidx = int(sqdictstr2num[sequence])
            samples.append([smilesidx, sequenceidx, int(interaction)])
        dataset = list(zip(samples, data_list))
        np.random.shuffle(dataset)
        samples, data_list = [s[0] for s in dataset], [s[1] for s in dataset]
        samples = np.array(samples)
        N = samples.shape[0]
        if args.dataset == 'bindingdb' or args.dataset == 'biosnap':
            trainsamples, valsamples, testsamples = np.split(samples, [int(0.7 * N), int(0.8 * N)])
            train, val, test = np.split(data_list, [int(0.7 * N), int(0.8 * N)])
        elif args.dataset == 'human' or args.dataset == 'celegans':
            trainsamples, valsamples, testsamples = np.split(samples, [int(0.8 * N), int(0.9 * N)])
            train, val, test = np.split(data_list, [int(0.8 * N), int(0.9 * N)])
        column = ['SMILES', 'Protein', 'Y', 'drug_cluster', 'target_cluster']
        train_set = pd.DataFrame(columns=column, data=train)
        train_set.to_csv(dir_path + '/train.csv', index=False)
        val_set = pd.DataFrame(columns=column, data=val)
        val_set.to_csv(dir_path + '/val.csv', index=False)
        test_set = pd.DataFrame(columns=column, data=test)
        test_set.to_csv(dir_path + '/test.csv', index=False)

    elif args.split_settings == 'unseen_smiles':
        # 获取'SMILES'列的唯一值
        unseen_name = 'SMILES'
        unseen_ratio = 0.2
        val = 0.7
        unique_smiles = full[unseen_name].unique()

        # 随机打乱这些唯一值
        np.random.shuffle(unique_smiles)

        # 计算要选择的元素数量
        num_val_test = int(len(unique_smiles) * unseen_ratio)

        # 选择20%的'SMILES'作为验证和测试集
        val_test_smiles = unique_smiles[:num_val_test]

        # 筛选出包含这些'SMILES'的行作为验证和测试集
        df_val_test = full[full['SMILES'].isin(val_test_smiles)].reset_index(drop=True)

        # 从原始数据集中移除这些行，得到训练集
        df_train = full[~full['SMILES'].isin(val_test_smiles)].reset_index(drop=True)
        # 如果您需要进一步将验证集和测试集分开，可以再次随机打乱并分割
        #
        df_val = df_val_test[:int(len(df_val_test) * val)]
        df_test = df_val_test[int(len(df_val_test) * val):]

        column = ['SMILES', 'Protein', 'Y', 'drug_cluster', 'target_cluster']
        train_set = pd.DataFrame(columns=column, data=df_train.values)
        train_set.to_csv(dir_path + '/train.csv', index=False)
        val_set = pd.DataFrame(columns=column, data=df_val.values)
        val_set.to_csv(dir_path + '/val.csv', index=False)
        test_set = pd.DataFrame(columns=column, data=df_test.values)
        test_set.to_csv(dir_path + '/test.csv', index=False)
    elif args.split_settings == 'unseen_protein':
        # 获取'SMILES'列的唯一值
        unseen_name = 'Protein'
        unseen_ratio = 0.2
        val = 0.7
        unique_smiles = full[unseen_name].unique()

        # 随机打乱这些唯一值
        np.random.shuffle(unique_smiles)

        # 计算要选择的元素数量
        num_val_test = int(len(unique_smiles) * unseen_ratio)

        # 选择20%的'SMILES'作为验证和测试集
        val_test_smiles = unique_smiles[:num_val_test]

        # 筛选出包含这些'SMILES'的行作为验证和测试集
        df_val_test = full[full['SMILES'].isin(val_test_smiles)].reset_index(drop=True)

        # 从原始数据集中移除这些行，得到训练集
        df_train = full[~full['SMILES'].isin(val_test_smiles)].reset_index(drop=True)
        # 如果您需要进一步将验证集和测试集分开，可以再次随机打乱并分割
        #
        df_val = df_val_test[:int(len(df_val_test) * val)]
        df_test = df_val_test[int(len(df_val_test) * val):]

        column = ['SMILES', 'Protein', 'Y', 'drug_cluster', 'target_cluster']
        train_set = pd.DataFrame(columns=column, data=df_train.values)
        train_set.to_csv(dir_path + '/train.csv', index=False)
        val_set = pd.DataFrame(columns=column, data=df_val.values)
        val_set.to_csv(dir_path + '/val.csv', index=False)
        test_set = pd.DataFrame(columns=column, data=df_test.values)
        test_set.to_csv(dir_path + '/test.csv', index=False)

    elif args.split_settings == 'missing':
        train_radio = [0.05, 0.1, 0.2, 0.3]
        val = 0.7
        data_len = len(full)
        for train in train_radio:
            train = int(data_len * train)
            val = int(data_len * val)
            test = data_len - train - val
            df_train, df_test = train_test_split(full, train_size=train)
            df_val, df_test = train_test_split(df_test, train_size=val)
            out_path = os.path.join(dir_path, str((1 - train) * 100))
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            df_train.to_csv(os.path.join(out_path, 'train.csv'), index=False)
            df_val.to_csv(os.path.join(out_path, 'val.csv'), index=False)
            df_test.to_csv(os.path.join(out_path, 'test.csv'), index=False)










