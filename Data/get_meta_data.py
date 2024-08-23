
def filter_both_label(df):
    grouped = df.groupby('target_cluster')
    # 步骤2: 对每个分组检查'Y'列中是否同时存在0和1
    mixed_clusters = grouped['Y'].apply(lambda x: (x == 0).any() and (x == 1).any()).dropna()
    # 步骤3: 记录下满足条件的'cluster'的名称
    clusters_without_both_labels = mixed_clusters[~mixed_clusters].index
    clusters_with_both_labels = mixed_clusters[mixed_clusters].index
    data_with_both_labels = df[df['target_cluster'].isin(clusters_with_both_labels)]
    data_without_both_labels = df[df['target_cluster'].isin(clusters_without_both_labels)]
    return data_with_both_labels.reset_index(drop=True),data_without_both_labels.reset_index(drop=True)

