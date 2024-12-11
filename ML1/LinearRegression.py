import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 使用 Microsoft YaHei 或 SimSun 字体，这两种字体在大部分 Windows 系统上都可以找到
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mean_y = np.mean(y_true)
    # 计算残差平方和
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    # 计算总平方和
    total_sum_of_squares = np.sum((y_true - mean_y) ** 2)
    # 计算R^2
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2


def train_test_split(X, y, test_size=0.25, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = X.shape[0]
    test_size = int(num_samples * test_size)

    indices = np.random.permutation(num_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

class MyStandardScaler:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def load_data(file_path, test_size=0.2, random_state=42):
    data = pd.read_csv(file_path, header=1)
    # 提取特征名和标签名
    feature_names = data.columns[:-1]
    target_name = data.columns[-1]
    num_samples = data.shape[0]
    # 输出基本信息
    print("特征名称：", list(feature_names))
    print("标签名称：", target_name)
    print("样本量：", num_samples)
    print("数据集前几行：\n", data.head())
    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    # 划分训练集和测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def data_processing(X_train, X_test):
    scaler = MyStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


class LinearRegression(object):
    def __init__(self, num_of_weights):
        self.w = np.random.randn(num_of_weights)

    def forward(self, X):
        intercept = np.ones((X.shape[0], 1))
        X_with_intercept = np.concatenate([intercept, X], axis=1)
        y_pred = np.dot(X_with_intercept, self.w)
        return y_pred

    def loss(self, X, y):
        y_pred = self.forward(X)
        cost = np.mean((y_pred - y) ** 2)
        return cost

    def gradient(self, X, y):
        # (2/m) * X_with_intercept(T) * residual
        y_pred = self.forward(X)
        residual = y_pred - y
        intercept = np.ones((X.shape[0], 1))
        X_with_intercept = np.concatenate([intercept, X], axis=1)
        grad = (2 / X.shape[0]) * np.dot(X_with_intercept.T, residual)
        return grad

    def update(self, grad, eta):
        self.w -= eta * grad

    def train_BGD(self, X, y, num_epochs, eta):
        loss_list = []
        for epoch_id in range(num_epochs):
            cost = self.loss(X, y)
            grad = self.gradient(X, y)
            self.update(grad, eta)
            loss_list.append(cost)
            if (epoch_id + 1) % 100 == 0:
                print(f'第{epoch_id + 1}次迭代，损失函数值：{cost}')
        return loss_list

    def train_MBGD(self, X, y, num_epochs, eta, batch_size):
        loss_list = []
        n_samples = X.shape[0]
        for epoch_id in range(num_epochs):
            # 生成一个随机排列的索引数组，用来打乱样本顺序
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                grad = self.gradient(X_batch, y_batch)
                self.update(grad, eta)
            cost = self.loss(X, y)
            loss_list.append(cost)
            if (epoch_id + 1) % 100 == 0:
                print(f'第{epoch_id + 1}次迭代，损失函数值：{cost}')
        return loss_list


