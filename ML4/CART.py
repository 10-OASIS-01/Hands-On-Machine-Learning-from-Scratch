import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        """
        初始化一个决策树节点。

        参数：
        - feature_index: 当前节点进行分裂的特征索引
        - threshold: 当前节点的分裂阈值
        - left: 左子树
        - right: 右子树
        - value: 叶节点的值（对于分类是类标签，对于回归是数值）
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, task='classification', min_samples_split=2, min_impurity_decrease=1e-7):
        """
        初始化决策树对象。

        参数：
        - task: 'classification' 或 'regression'，定义任务是分类还是回归
        - min_samples_split: 最小拆分样本数，若样本数小于该值则不再拆分
        - min_impurity_decrease: 最小纯度减少，若拆分后的增益小于该值则不进行拆分
        """
        self.task = task
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None  # 根节点初始化为空

    def fit(self, X, y):
        """
        拟合决策树模型，构建树结构。

        参数：
        - X: 特征数据
        - y: 标签数据
        """
        self.root = self._build_tree(X, y)  # 从数据开始构建决策树

    def _build_tree(self, X, y):
        """
        递归地构建决策树。

        参数：
        - X: 特征数据
        - y: 标签数据

        返回：
        - Node对象: 树的根节点或叶子节点
        """
        num_samples, num_features = X.shape  # 获取样本数和特征数

        # 如果样本数小于最小拆分数，则不再拆分，创建叶节点
        if num_samples < self.min_samples_split:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # 选择最佳特征和阈值进行分裂
        best_feature, best_threshold, best_gain = self._best_split(X, y, num_features)

        # 如果增益小于最小纯度减少阈值，创建叶节点
        if best_gain < self.min_impurity_decrease:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # 根据最佳分裂进行样本划分
        left_indices = X[:, best_feature] <= best_threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]

        # 递归构建左子树和右子树
        left_child = self._build_tree(X_left, y_left)
        right_child = self._build_tree(X_right, y_right)

        # 返回当前节点
        return Node(feature_index=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _best_split(self, X, y, num_features):
        """
        通过遍历所有特征和阈值找到最佳的拆分方式。

        参数：
        - X: 特征数据
        - y: 标签数据
        - num_features: 特征的总数

        返回：
        - best_feature: 最佳分裂特征的索引
        - best_threshold: 最佳分裂阈值
        - best_gain: 最佳分裂的增益值
        """
        best_gain = -np.inf  # 初始化增益为负无穷
        best_feature, best_threshold = None, None

        # 遍历每个特征
        for feature in range(num_features):
            X_column = X[:, feature]  # 获取当前特征的所有样本值
            thresholds = np.unique(X_column)  # 获取该特征的唯一值作为候选阈值

            # 遍历每个阈值
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)  # 计算当前阈值的增益
                if gain > best_gain:
                    best_gain = gain  # 更新最佳增益
                    best_feature = feature  # 更新最佳特征
                    best_threshold = threshold  # 更新最佳阈值

        return best_feature, best_threshold, best_gain

    def _information_gain(self, y, X_column, threshold):
        """
        计算潜在拆分的增益。

        参数：
        - y: 标签数据
        - X_column: 当前特征列
        - threshold: 当前的分裂阈值

        返回：
        - 增益值
        """
        if self.task == 'classification':
            return self._gini_gain(y, X_column, threshold)  # 分类任务使用基尼增益
        else:
            return self._variance_gain(y, X_column, threshold)  # 回归任务使用方差增益

    def _gini_gain(self, y, X_column, threshold):
        """
        计算分类任务中的基尼增益。

        参数：
        - y: 标签数据
        - X_column: 当前特征列
        - threshold: 当前的分裂阈值

        返回：
        - 基尼增益
        """
        parent_gini = self._gini(y)  # 计算父节点的基尼不纯度
        left_indices = X_column <= threshold  # 小于等于阈值的样本索引
        right_indices = X_column > threshold  # 大于阈值的样本索引
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0  # 如果某一边为空，则增益为0

        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])  # 左右子集的样本数
        # 计算子节点的加权基尼不纯度
        child_gini = (n_left / n) * self._gini(y[left_indices]) + (n_right / n) * self._gini(y[right_indices])

        return parent_gini - child_gini  # 返回基尼增益

    def _variance_gain(self, y, X_column, threshold):
        """
        计算回归任务中的方差增益。

        参数：
        - y: 标签数据
        - X_column: 当前特征列
        - threshold: 当前的分裂阈值

        返回：
        - 方差增益
        """
        parent_variance = self._variance(y)  # 计算父节点的方差
        left_indices = X_column <= threshold
        right_indices = X_column > threshold
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0  # 如果某一边为空，则增益为0

        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])
        # 计算子节点的加权方差
        child_variance = (n_left / n) * self._variance(y[left_indices]) + (n_right / n) * self._variance(
            y[right_indices])

        return parent_variance - child_variance  # 返回方差增益

    def _gini(self, y):
        """
        计算标签集的基尼不纯度。

        参数：
        - y: 标签数据

        返回：
        - 基尼不纯度值
        """
        unique_classes, counts = np.unique(y, return_counts=True)  # 获取标签的唯一值和出现次数
        probabilities = counts / counts.sum()  # 计算每个标签的概率
        return 1 - np.sum(probabilities ** 2)  # 计算基尼不纯度

    def _variance(self, y):
        """
        计算一组连续值的方差。

        参数：
        - y: 数值标签

        返回：
        - 方差值
        """
        return np.var(y)

    def _calculate_leaf_value(self, y):
        """
        计算叶子节点的值。

        参数：
        - y: 标签数据

        返回：
        - 分类任务返回最频繁的类，回归任务返回平均值
        """
        if self.task == 'classification':
            unique_classes, counts = np.unique(y, return_counts=True)  # 分类任务：返回最多出现的类别
            return unique_classes[np.argmax(counts)]
        else:
            return np.mean(y)  # 回归任务：返回平均值

    def predict(self, X):
        """
        对给定样本X进行预测。

        参数：
        - X: 特征数据

        返回：
        - 预测结果（分类标签或回归值）
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        递归遍历决策树，为单个样本做出预测。

        参数：
        - x: 单个样本的特征
        - node: 当前节点

        返回：
        - 预测值
        """
        # 如果到达叶节点，返回叶节点的值
        if node.value is not None:
            return node.value

        # 否则，根据特征值选择走左子树还是右子树
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def print_tree(self, node=None, depth=0, prefix=""):
        """
        打印决策树的结构（树形结构）。

        参数：
        - node: 当前节点，默认为根节点
        - depth: 当前节点的深度，用于控制树的层级
        - prefix: 当前节点前缀，表示当前树的分支位置
        """
        if node is None:
            node = self.root

        if node.value is not None:
            # 如果是叶节点，打印叶节点的值
            print(f"{prefix}Leaf: {node.value}")
        else:
            # 打印当前节点的分裂条件
            print(f"{prefix}Feature {node.feature_index} <= {node.threshold}")

            # 递归打印左子树
            if node.left:
                self.print_tree(node.left, depth + 1, prefix + "|-- Left: ")

            # 递归打印右子树
            if node.right:
                self.print_tree(node.right, depth + 1, prefix + "|-- Right: ")


def train_test_split_data(X, y, test_size=0.2, random_state=None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_classification(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def evaluate_regression(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
