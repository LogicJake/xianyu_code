import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import (SelectKBest, VarianceThreshold, chi2,
                                       f_classif)

df_feature = pd.read_csv('data1.csv')

feature_names = [
    f for f in df_feature.columns if f not in ['Default', 'is_train']
]
print('原始特征数量', len(feature_names))


# VarianceThreshold可以删除所有低方差的特征，方差越低，特征值离散程度越低，参考价值也不大
def select_variance(data, feature_names, threshold=0.7):
    # 调用VarianceThreshold获取所有特征的方差
    selector = VarianceThreshold()
    # fit 函数传入特征数据
    selector.fit(data[feature_names])

    # 获取所有待筛选特征的得分
    scores = selector.variances_
    # dict(zip(feature_names, scores)) 获取特征名称和对应的方差的字典
    # 然后将字典转为dataframe格式，总共两列，第一列是特征名称，第二列是对应的方差
    scores = pd.DataFrame(dict(zip(feature_names, scores)), index=[0]).T
    # 将dataframe的索引进行重置
    scores.reset_index(inplace=True)
    # 赋予列名
    scores.columns = ['feature', 'score']
    # 根据score进行降序排序
    scores.sort_values('score', inplace=True, ascending=False)

    preserved_df = scores[scores['score'] > threshold]

    return preserved_df['feature'].values.tolist()


# 卡方检验
def select_chi2(data, num, feature_names, ycol):
    # 调用SelectKBest，使用chi2指标，’all‘表明保留所有特征
    # fit 函数传入两个参数：特征数据和标签数据
    selector = SelectKBest(chi2, k='all').fit(data[feature_names], data[ycol])

    # 获取所有待筛选特征的得分
    scores = selector.scores_
    # dict(zip(feature_names, scores)) 获取特征名称和对应的chi2分数的字典
    # 然后将字典转为dataframe个数，总共两列，第一列是特征名称，第二列是对应的chi2分数
    scores = pd.DataFrame(dict(zip(feature_names, scores)), index=[0]).T
    # 将dataframe的索引进行重置
    scores.reset_index(inplace=True)
    # 赋予列名
    scores.columns = ['feature', 'score']
    # 根据score进行降序排序
    scores.sort_values('score', inplace=True, ascending=False)

    # 获取筛选后保留的特征名称，head函数保存前num数据，然后获取对应的特征名
    preserved_features = scores.head(num)['feature'].values.tolist()
    return preserved_features


# F-检验
def select_f_classif(data, num, feature_names, ycol):
    # 调用SelectKBest，使用f_classif指标，’all‘表明保留所有特征
    # fit 函数传入两个参数：特征数据和标签数据
    selector = SelectKBest(f_classif, k='all').fit(data[feature_names],
                                                   data[ycol])

    # 获取所有待筛选特征的得分
    scores = selector.scores_
    # dict(zip(feature_names, scores)) 获取特征名称和对应的chi2分数的字典
    # 然后将字典转为dataframe个数，总共两列，第一列是特征名称，第二列是对应的chi2分数
    scores = pd.DataFrame(dict(zip(feature_names, scores)), index=[0]).T
    # 将dataframe的索引进行重置
    scores.reset_index(inplace=True)
    # 赋予列名
    scores.columns = ['feature', 'score']
    # 根据score进行降序排序
    scores.sort_values('score', inplace=True, ascending=False)

    # 获取筛选后保留的特征名称，head函数保存前num数据，然后获取对应的特征名
    preserved_features = scores.head(num)['feature'].values.tolist()
    return preserved_features


# 互信息检验
def select_mis(data, num, feature_names, ycol):
    scores = []
    for f in feature_names:
        score = metrics.mutual_info_score(data[ycol], data[f])
        scores.append(score)

    scores = pd.DataFrame(dict(zip(feature_names, scores)), index=[0]).T
    # 将dataframe的索引进行重置
    scores.reset_index(inplace=True)
    # 赋予列名
    scores.columns = ['feature', 'score']
    # 根据score进行降序排序
    scores.sort_values('score', inplace=True, ascending=False)

    # 获取筛选后保留的特征名称，head函数保存前num数据，然后获取对应的特征名
    preserved_features = scores.head(num)['feature'].values.tolist()
    return preserved_features


# 先将方差特别低的特征删除
variance_feat = select_variance(df_feature, feature_names, threshold=0.02)
print('方差过滤后特征数量', len(variance_feat))

# 卡方检验筛选 TOP 20 特征
select_feat1 = select_chi2(df_feature,
                           num=20,
                           feature_names=variance_feat,
                           ycol='Default')
# F-检验筛选 TOP 20 特征
select_feat2 = select_f_classif(df_feature,
                                num=20,
                                feature_names=variance_feat,
                                ycol='Default')
# 互信息筛选 TOP 20 特征
select_feat3 = select_mis(df_feature,
                          num=20,
                          feature_names=variance_feat,
                          ycol='Default')

print(select_feat1)
print(select_feat2)
print(select_feat3)

preserve_feat = set(select_feat1) & set(select_feat2) & set(select_feat3)
preserve_feat = list(preserve_feat)
print('筛选过滤后特征数量', len(preserve_feat))

print(preserve_feat)

df_feature = df_feature[['Default', 'is_train'] + preserve_feat]
df_feature.to_csv('data.csv', index=False)
