import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from model import DeepFM

seed = 2022

# 如果存在cuda使用gpu，否则cpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


class dataset(Dataset):
    def __init__(self, df):
        self.data = df.values
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index, :-1], self.data[index, -1]

    def __len__(self):
        return self.len


data = pd.read_excel('1101特征排序-edit.xlsx')

# 删除userid
del data['USER_ID']
del data['商品ID']

# 类别特征编码成连续变量
for col in ['BASE_CD', '用户等级', '性别', '标签']:
    lbe = LabelEncoder()
    data[col] = lbe.fit_transform(data[col])
    # +1 是为了空出来位置0，用来代表缺失值
    data[col] = data[col] + 1
    # 缺失值设置为0
    data[col].fillna(0, inplace=True)

# 连续特征归一化到0-1
for col in ['年龄', '是否vip', '装备数量']:
    data[col] = (data[col] - data[col].min()) / (data[col].max() -
                                                 data[col].min())
    data[col].fillna(0, inplace=True)

# 数据集打乱顺序
data = data.sample(frac=1, random_state=seed)
# 80%训练集
train_data = data[:int(len(data) * 0.8)]
# 20%测试集
test_data = data[int(len(data) * 0.8):]

# 训练集和测试集包装为dataloader，可以在后面一个batch一个batch取
train_loader = DataLoader(dataset(train_data), num_workers=2, batch_size=1024)
test_loader = DataLoader(dataset(test_data), num_workers=2, batch_size=1024)

# 模型训练
# 初始化模型 deepfm
sparse_features = {
    'BASE_CD': data['BASE_CD'].nunique() + 1,
    '用户等级': data['用户等级'].nunique() + 1,
    '性别': data['性别'].nunique() + 1,
    '标签': data['标签'].nunique() + 1,
}

dense_features = ['年龄', '是否vip', '装备数量']

model = DeepFM(sparse_features=sparse_features,
               dense_features=dense_features,
               embedding_dim=8)
model = model.to(device)

# 定义优化器
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
# 进入训练状态
model.train()

for epoch in range(3):
    loss_total = []
    for X, y in tqdm(train_loader):
        # 离散特征
        base_cd = X[:, 0].to(device)
        level = X[:, 1].to(device)
        sex = X[:, 2].to(device)
        tag = X[:, 3].to(device)

        # 连续特征
        dense_features = X[:, 4:].float().to(device)

        y = y.float().to(device)

        _, loss = model(base_cd, level, sex, tag, dense_features, y)

        # 根据loss计算梯度
        loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        # 梯度重置为0
        model.zero_grad()

        loss_total.append(loss.item())

    # 计算这轮平均loss
    loss = np.mean(loss_total)
    print('Epoch-{}: loss: {}'.format(epoch, loss))

# 测试集预测
model.eval()
y_list = []
predict_list = []

for X, y in tqdm(test_loader):
    # 离散特征
    base_cd = X[:, 0].to(device)
    level = X[:, 1].to(device)
    sex = X[:, 2].to(device)
    tag = X[:, 3].to(device)

    # 连续特征
    dense_features = X[:, 4:].float().to(device)

    y = y.float().to(device)

    predict, loss = model(base_cd, level, sex, tag, dense_features, y)

    y_list.extend(list(y.cpu().detach().numpy()))
    predict_list.extend(list(predict.cpu().detach().numpy()))

# print(y_list)
# print(predict_list)
test_data['predict'] = predict_list
test_data.to_csv('预测.csv', index=False)

auc = roc_auc_score(y_list, predict_list)
print('AUC: ', auc)
