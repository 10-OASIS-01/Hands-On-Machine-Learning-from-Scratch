import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None, task='classification'):
        """
        初始化决策树。

        参数:
        - max_depth: 树的最大深度
        - min_samples_split: 节点分裂所需的最小样本数
        - n_features: 每次分裂时考虑的特征数
        - task: 'classification' 或 'regression'，决定树是用于分类任务还是回归任务
        """
        self.max_depth = max_depth  # 最大深度
        self.min_samples_split = min_samples_split  # 最小样本分裂数
        self.n_features = n_features  # 每次分裂时考虑的特征数
        self.task = task  # 任务类型：分类或回归
        self.tree = None  # 决策树的根节点，初始化为None

    def fit(self, X, y):
        """
        训练决策树模型。

        参数:
        - X: 特征数据，形状为 (样本数, 特征数)
        - y: 标签数据
        """
        if self.task == 'classification':
            self.n_classes_ = len(set(y))  # 分类任务中，类别数
        else:
            self.n_classes_ = None  # 对于回归任务，无需类别数
        self.n_features_ = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])  # 每次分裂时选择的特征数
        self.tree = self._grow_tree(X, y)  # 构建树

    def _best_split_classification(self, X, y):
        """
        分类任务中，计算最佳分裂点。

        参数:
        - X: 特征数据
        - y: 标签数据

        返回：
        - best_idx: 最佳特征的索引
        - best_thr: 最佳分裂阈值
        """
        m, n = X.shape
        if m <= 1:  # 如果样本数小于等于1，无法继续分裂
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]  # 计算每个类别的样本数

        # 计算初始的基尼指数
        best_gini = 1.0 - sum((count / m) ** 2 for count in num_parent)
        best_idx, best_thr = None, None

        feature_idxs = np.random.choice(n, self.n_features_, replace=False)  # 随机选择特征进行分裂

        for idx in feature_idxs:  # 遍历选择的特征
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))  # 按特征值排序
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()  # 右子树的类别分布

            for i in range(1, m):  # 遍历每个样本作为分裂点
                c = classes[i - 1]
                num_left[c] += 1  # 左子树的样本数更新
                num_right[c] -= 1  # 右子树的样本数更新

                # 计算左右子树的基尼指数
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))

                gini = (i * gini_left + (m - i) * gini_right) / m  # 加权平均基尼指数

                if thresholds[i] == thresholds[i - 1]:  # 如果特征值相等，跳过
                    continue

                if gini < best_gini:  # 找到最小的基尼指数
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # 选择最优阈值

        return best_idx, best_thr

    def _best_split_regression(self, X, y):
        """
        回归任务中，计算最佳分裂点。

        参数:
        - X: 特征数据
        - y: 标签数据

        返回：
        - best_idx: 最佳特征的索引
        - best_thr: 最佳分裂阈值
        """
        m, n = X.shape
        if m <= 1:  # 如果样本数小于等于1，无法继续分裂
            return None, None

        best_mse = float('inf')  # 初始化最小均方误差
        best_idx, best_thr = None, None

        feature_idxs = np.random.choice(n, self.n_features_, replace=False)  # 随机选择特征进行分裂

        for idx in feature_idxs:  # 遍历选择的特征
            thresholds, values = zip(*sorted(zip(X[:, idx], y)))  # 按特征值排序
            left_sum = 0.0
            left_sum_sq = 0.0
            left_count = 0
            right_sum = sum(values)
            right_sum_sq = sum(v ** 2 for v in values)

            for i in range(1, m):  # 遍历每个样本作为分裂点
                xi, yi = thresholds[i - 1], values[i - 1]
                left_sum += yi  # 左子树的样本和
                left_sum_sq += yi ** 2  # 左子树的样本平方和
                left_count += 1  # 左子树样本数
                right_sum -= yi  # 右子树的样本和
                right_sum_sq -= yi ** 2  # 右子树的样本平方和

                if thresholds[i] == thresholds[i - 1]:  # 如果特征值相等，跳过
                    continue

                left_mean = left_sum / left_count  # 左子树均值
                right_count = m - i
                right_mean = right_sum / right_count  # 右子树均值

                mse_left = left_sum_sq - left_sum ** 2 / left_count  # 左子树均方误差
                mse_right = right_sum_sq - right_sum ** 2 / right_count  # 右子树均方误差

                mse = (mse_left + mse_right) / m  # 总的均方误差

                if mse < best_mse:  # 找到最小的均方误差
                    best_mse = mse
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # 选择最优阈值

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        """
        递归地构建决策树。

        参数:
        - X: 特征数据
        - y: 标签数据
        - depth: 当前树的深度

        返回：
        - node: 当前树节点
        """
        if self.task == 'classification':  # 分类任务
            if self.n_classes_ is None:
                self.n_classes_ = len(set(y))  # 获取类别数
            num_samples = len(y)
            num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]  # 计算每个类别的样本数
            predicted_class = np.argmax(num_samples_per_class)  # 预测的类别为样本数最多的类别
            node = {"type": "leaf", "value": predicted_class}  # 创建叶节点

            if depth < self.max_depth and num_samples >= self.min_samples_split:
                idx, thr = self._best_split_classification(X, y)  # 查找最佳分裂
                if idx is not None:
                    indices_left = X[:, idx] < thr  # 根据阈值进行分裂
                    X_left, y_left = X[indices_left], y[indices_left]
                    X_right, y_right = X[~indices_left], y[~indices_left]
                    if len(y_left) > 0 and len(y_right) > 0:  # 保证左右子树非空
                        node = {
                            "type": "node",
                            "feature_index": idx,
                            "threshold": thr,
                            "left": self._grow_tree(X_left, y_left, depth + 1),  # 递归构建左子树
                            "right": self._grow_tree(X_right, y_right, depth + 1),  # 递归构建右子树
                        }
        else:  # 回归任务
            num_samples = len(y)
            predicted_value = np.mean(y)  # 预测值为样本的均值
            node = {"type": "leaf", "value": predicted_value}

            if depth < self.max_depth and num_samples >= self.min_samples_split:
                idx, thr = self._best_split_regression(X, y)  # 查找最佳分裂
                if idx is not None:
                    indices_left = X[:, idx] < thr  # 根据阈值进行分裂
                    X_left, y_left = X[indices_left], y[indices_left]
                    X_right, y_right = X[~indices_left], y[~indices_left]
                    if len(y_left) > 0 and len(y_right) > 0:  # 保证左右子树非空
                        node = {
                            "type": "node",
                            "feature_index": idx,
                            "threshold": thr,
                            "left": self._grow_tree(X_left, y_left, depth + 1),
                            "right": self._grow_tree(X_right, y_right, depth + 1),
                        }
        return node

    def predict(self, X):
        """
        对输入数据进行预测。

        参数:
        - X: 特征数据

        返回：
        - predictions: 预测结果
        """
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        """
        对单个样本进行预测。

        参数:
        - inputs: 样本数据

        返回：
        - node["value"]: 预测结果（分类任务是类别，回归任务是数值）
        """
        node = self.tree
        while node["type"] == "node":  # 遍历树直到叶节点
            if inputs[node["feature_index"]] < node["threshold"]:  # 根据阈值选择分支
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]  # 返回叶节点的值

class RandomForest:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        task='classification',
    ):
        """
        初始化随机森林。

        参数:
        - n_estimators: 树的数量
        - max_depth: 每棵树的最大深度
        - min_samples_split: 节点分裂所需的最小样本数
        - max_features: 每次分裂时考虑的特征数
        - task: 'classification' 或 'regression'，决定任务类型
        """
        self.n_estimators = n_estimators  # 树的数量
        self.max_depth = max_depth  # 每棵树的最大深度
        self.min_samples_split = min_samples_split  # 最小样本分裂数
        self.max_features = max_features  # 每次分裂时考虑的特征数
        self.task = task  # 任务类型
        self.trees = []  # 存储决策树

    def fit(self, X, y):
        """
        训练随机森林模型。

        参数:
        - X: 特征数据
        - y: 标签数据
        """
        self.trees = []  # 清空已训练的树
        n_samples, n_features = X.shape  # 获取样本数和特征数

        # 根据 max_features 参数选择每棵树使用的特征数
        if self.max_features == "sqrt":
            self.n_features_ = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            self.n_features_ = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            self.n_features_ = self.max_features
        else:
            self.n_features_ = n_features

        # 训练多棵决策树
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)  # 有放回采样
            X_sample = X[indices]
            y_sample = y[indices]

            # 训练一棵决策树
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features_,
                task=self.task,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)  # 将树加入随机森林

    def predict(self, X):
        """
        对输入数据进行预测。

        参数:
        - X: 特征数据

        返回：
        - predictions: 预测结果
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])  # 获取每棵树的预测结果

        if self.task == 'classification':  # 分类任务，采用投票机制
            return np.array([Counter(col).most_common(1)[0][0] for col in tree_preds.T])
        else:  # 回归任务，采用均值
            return np.mean(tree_preds, axis=0)
