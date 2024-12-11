import numpy as np


class DecisionStump:
    def __init__(self):
        self.feature_index = None  # 特征索引
        self.threshold = None  # 阈值
        self.polarity = 1  # 极性，默认为1

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = float('inf')  # 初始化最小误差为正无穷

        # 遍历每个特征
        for feature in range(n_features):
            feature_values = X[:, feature]
            unique_values = np.unique(feature_values)

            # 可能的阈值是相邻特征值的中点
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                for polarity in [1, -1]:
                    # 预测：如果 (polarity * feature) < (polarity * threshold)，预测为1，否则预测为-1
                    predictions = np.ones(n_samples)
                    predictions[polarity * feature_values < polarity * threshold] = -1

                    # 计算加权误差
                    misclassified = predictions != y
                    error = np.sum(sample_weights[misclassified])

                    # 选择误差最小的阈值
                    if error < min_error:
                        min_error = error
                        self.polarity = polarity
                        self.threshold = threshold
                        self.feature_index = feature

    def predict(self, X):
        n_samples = X.shape[0]
        feature_values = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        predictions[self.polarity * feature_values < self.polarity * self.threshold] = -1
        return predictions


class AdaBoost:
    def __init__(self, n_clf=50):
        self.n_clf = n_clf  # 基分类器的数量，默认50
        self.clfs = []  # 存储分类器的列表
        self.alpha = []  # 存储每个分类器的权重（alpha）

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # 初始化样本权重
        w = np.full(n_samples, (1 / n_samples))

        for clf_idx in range(self.n_clf):
            clf = DecisionStump()  # 创建一个决策桩分类器
            clf.fit(X, y, w)  # 训练决策桩
            predictions = clf.predict(X)  # 获取预测结果

            # 计算误差和alpha
            error = np.sum(w[y != predictions])
            if error == 0:
                error = 1e-10  # 避免除以零的情况
            alpha = 0.5 * np.log((1 - error) / error)  # 计算分类器权重

            # 更新样本权重
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)  # 权重归一化

            # 保存分类器和它的alpha
            self.clfs.append(clf)
            self.alpha.append(alpha)

    def predict(self, X):
        # 获取每个基分类器的预测结果
        clf_preds = np.array([clf.predict(X) for clf in self.clfs])
        # 计算加权预测结果
        weighted_preds = np.dot(self.alpha, clf_preds)
        return np.sign(weighted_preds)  # 返回加权预测的符号

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)  # 返回准确率
