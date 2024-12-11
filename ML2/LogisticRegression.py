import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

# 定义 Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义 Softmax 函数
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 防止溢出
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='binary', num_classes=None, learning_rate=0.01, iterations=1000, lambda_=0.01, batch_size=32):
        """
        参数：
        - model_type: 选择模型类型，'binary'、'ovr' 或 'softmax'
        - num_classes: 类别数，仅在 'ovr' 和 'softmax' 模型中需要
        - learning_rate: 学习率
        - iterations: 迭代次数
        - lambda_: 正则化参数
        - batch_size: 批量大小
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_ = lambda_
        self.batch_size = batch_size
        # 初始化模型参数
        self.w = None
        self.b = None
        self.cost_history = []

    # 训练模型
    def fit(self, X, y):
        n_features = X.shape[1]
        m = X.shape[0]

        if self.model_type == 'binary':
            # 检查 y 是否为二分类
            if len(np.unique(y)) > 2:
                raise ValueError("For 'binary' model_type, y should contain only two classes.")

            self.w = np.zeros(n_features)
            self.b = 0
            self._gradient_descent_binary(X, y)

        elif self.model_type == 'ovr':
            # 检查 num_classes 是否指定
            if self.num_classes is None:
                self.num_classes = len(np.unique(y))

            self.w = np.zeros((self.num_classes, n_features))
            self.b = np.zeros(self.num_classes)
            self._gradient_descent_ovr(X, y)

        elif self.model_type == 'softmax':
            # 检查 num_classes 是否指定
            if self.num_classes is None:
                self.num_classes = len(np.unique(y))

            self.w = np.zeros((self.num_classes, n_features))
            self.b = np.zeros(self.num_classes)
            self._gradient_descent_softmax(X, y)

        else:
            raise ValueError("model_type must be 'binary', 'ovr', or 'softmax'.")

        return self

    # 预测
    def predict(self, X):
        if self.model_type == 'binary':
            z = X @ self.w + self.b
            h = sigmoid(z)
            return (h >= 0.5).astype(int)

        elif self.model_type == 'ovr':
            z = X @ self.w.T + self.b
            h = sigmoid(z)
            return np.argmax(h, axis=1)

        elif self.model_type == 'softmax':
            z = X @ self.w.T + self.b
            h = softmax(z)
            return np.argmax(h, axis=1)

    # 预测概率
    def predict_proba(self, X):
        if self.model_type == 'binary':
            z = X @ self.w + self.b
            h = sigmoid(z)
            return np.vstack([1 - h, h]).T  # 返回两列，分别为类0和类1的概率

        elif self.model_type == 'ovr':
            z = X @ self.w.T + self.b
            h = sigmoid(z)
            # 对每个样本归一化概率，使得每行的概率和为1
            return h / np.sum(h, axis=1, keepdims=True)

        elif self.model_type == 'softmax':
            z = X @ self.w.T + self.b
            h = softmax(z)
            return h

    # 私有方法：二分类梯度下降
    def _gradient_descent_binary(self, X, y):
        m = X.shape[0]
        for i in tqdm(range(self.iterations), desc="Training Binary Logistic Regression"):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                dw, db = self._compute_gradients_binary(X_batch, y_batch)
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            # 每100次记录一次成本
            if i % 100 == 0:
                cost = self._compute_cost_binary(X, y)
                self.cost_history.append(cost)

    # 私有方法：计算二分类损失
    def _compute_cost_binary(self, X, y):
        epsilon = 1e-8
        m = X.shape[0]
        z = X @ self.w + self.b
        h = sigmoid(z)
        cost = -(1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        reg_cost = (self.lambda_ / (2 * m)) * np.sum(np.square(self.w))
        return cost + reg_cost

    # 私有方法：计算二分类梯度
    def _compute_gradients_binary(self, X, y):
        m = X.shape[0]
        z = X @ self.w + self.b
        h = sigmoid(z)
        error = h - y
        dw = (1 / m) * (X.T @ error) + (self.lambda_ / m) * self.w
        db = (1 / m) * np.sum(error)
        return dw, db

    # 私有方法：OvR梯度下降
    def _gradient_descent_ovr(self, X, y):
        m = X.shape[0]
        for c in range(self.num_classes):
            y_binary = (y == c).astype(int)
            w_c = np.zeros(X.shape[1])
            b_c = 0
            cost_history_c = []

            for i in tqdm(range(self.iterations), desc=f"Training OvR Class {c}"):
                indices = np.random.permutation(m)
                X_shuffled = X[indices]
                y_shuffled = y_binary[indices]

                for start in range(0, m, self.batch_size):
                    end = start + self.batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    dw, db = self._compute_gradients_binary(X_batch, y_batch)
                    w_c -= self.learning_rate * dw
                    b_c -= self.learning_rate * db

                # 每100次记录一次成本
                if i % 100 == 0:
                    cost = self._compute_cost_binary(X, y_binary)
                    cost_history_c.append(cost)

            self.w[c] = w_c
            self.b[c] = b_c
            self.cost_history.append(cost_history_c)

    # 私有方法：Softmax梯度下降
    def _gradient_descent_softmax(self, X, y):
        m = X.shape[0]
        for i in tqdm(range(self.iterations), desc="Training Softmax Regression"):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                dw, db = self._compute_gradients_softmax(X_batch, y_batch)
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            if i % 100 == 0:
                cost = self._compute_cost_softmax(X, y)
                self.cost_history.append(cost)

    # 私有方法：计算Softmax损失
    def _compute_cost_softmax(self, X, y):
        epsilon = 1e-8
        m = X.shape[0]
        z = X @ self.w.T + self.b
        h = softmax(z)
        y_one_hot = np.zeros_like(h)
        y_one_hot[np.arange(m), y] = 1
        cost = -(1 / m) * np.sum(y_one_hot * np.log(h + epsilon))
        reg_cost = (self.lambda_ / (2 * m)) * np.sum(np.square(self.w))
        return cost + reg_cost

    # 私有方法：计算Softmax梯度
    def _compute_gradients_softmax(self, X, y):
        m = X.shape[0]
        z = X @ self.w.T + self.b
        h = softmax(z)
        y_one_hot = np.zeros_like(h)
        y_one_hot[np.arange(m), y] = 1
        error = h - y_one_hot
        dw = (1 / m) * (error.T @ X) + (self.lambda_ / m) * self.w
        db = (1 / m) * np.sum(error, axis=0)
        return dw, db
