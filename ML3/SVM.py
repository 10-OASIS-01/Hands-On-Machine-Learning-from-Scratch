import numpy as np

class SVM:
    def __init__(self, C=1.0, tol=1e-3, max_passes=5):
        """
        初始化SVM模型的超参数。

        参数:
        - C: 惩罚参数，控制分类误差的惩罚程度。较大的C值会更严格地惩罚误分类样本。
        - tol: 容忍度，确定停止训练的条件，主要用于检测误差是否小于容忍度。
        - max_passes: 最大遍历次数，在所有样本都没有更新时，遍历次数达到最大值则停止训练。
        """
        self.C = C  # 惩罚参数，控制误分类的惩罚程度
        self.tol = tol  # 容忍度，决定误差的容忍范围
        self.max_passes = max_passes  # 最大遍历次数
        self.alpha = None  # Lagrange乘子
        self.b = None  # 偏置
        self.w = None  # 权重向量
        self.support_vectors_ = None  # 支持向量
        self.support_vectors_y = None  # 支持向量的标签
        self.alpha_sv = None  # 支持向量的alpha值

    def fit(self, X, y):
        """
        训练SVM模型。

        参数:
        - X: 特征矩阵，形状为(m, n)，m为样本数，n为特征数
        - y: 标签向量，形状为(m,)，其中y[i]为样本i的标签（+1或-1）
        """
        m, n = X.shape  # 获取样本数m和特征数n
        self.alpha = np.zeros(m)  # 初始化Lagrange乘子为0
        self.b = 0  # 初始化偏置b为0
        passes = 0  # 遍历次数初始化为0

        # 预计算内积矩阵K，K[i, j]表示样本i和样本j之间的内积
        K = X @ X.T

        # 训练过程，直到达到最大遍历次数
        while passes < self.max_passes:
            num_changed_alphas = 0  # 记录更新的alpha值个数

            # 遍历所有样本
            for i in range(m):
                Ei = self._E(i, X, y, K)  # 计算样本i的误差Ei

                # 如果样本i违反KKT条件，则更新对应的alpha值
                if (y[i] * Ei < -self.tol and self.alpha[i] < self.C) or (y[i] * Ei > self.tol and self.alpha[i] > 0):
                    j = self._select_j(i, m)  # 随机选择另一个样本j
                    Ej = self._E(j, X, y, K)  # 计算样本j的误差Ej

                    alpha_i_old = self.alpha[i]  # 保存当前的alpha[i]值
                    alpha_j_old = self.alpha[j]  # 保存当前的alpha[j]值

                    # 计算L和H，L和H是alpha[j]更新时的范围
                    if y[i] != y[j]:
                        L = max(0.0, float(self.alpha[j] - self.alpha[i]))  # alpha[j]更新的下界
                        H = min(self.C, float(self.C + self.alpha[j] - self.alpha[i]))  # alpha[j]更新的上界
                    else:
                        L = max(0.0, float(self.alpha[i] + self.alpha[j] - self.C))  # alpha[j]更新的下界
                        H = min(self.C, float(self.alpha[i] + self.alpha[j]))  # alpha[j]更新的上界

                    if L == H:  # 如果L == H，则跳过当前的更新
                        continue

                    # 计算eta（用于更新alpha[j]）
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:  # 如果eta大于等于0，跳过更新
                        continue

                    # 更新alpha[j]的值
                    self.alpha[j] -= y[j] * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)  # 将alpha[j]限制在L和H之间

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:  # 如果alpha[j]变化不大，跳过
                        continue

                    # 更新alpha[i]的值
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # 更新偏置b
                    b1 = self.b - Ei - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - Ej - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]

                    # 更新b的规则，根据alpha的值确定
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_changed_alphas += 1  # 记录更新的alpha数目

            if num_changed_alphas == 0:  # 如果没有alpha值更新，增加遍历次数
                passes += 1
            else:
                passes = 0  # 有alpha值更新，重置遍历次数

        # 计算权重向量w
        self.w = ((self.alpha * y)[:, np.newaxis] * X).sum(axis=0)

        # 找出支持向量
        self.support_vectors_ = X[self.alpha > 1e-5]  # alpha值大于0的样本为支持向量
        self.support_vectors_y = y[self.alpha > 1e-5]  # 支持向量的标签
        self.alpha_sv = self.alpha[self.alpha > 1e-5]  # 支持向量的alpha值

    def _E(self, i, X, y, K):
        """
        计算误差E_i。

        参数:
        - i: 样本索引
        - X: 特征矩阵
        - y: 标签向量
        - K: 样本内积矩阵

        返回:
        - E_i: 样本i的误差
        """
        f_xi = (self.alpha * y) @ K[:, i] + self.b  # 计算预测值
        E_i = f_xi - y[i]  # 计算误差
        return E_i

    def _select_j(self, i, m):
        """
        随机选择另一个不等于i的样本索引j。

        参数:
        - i: 当前选择的样本索引
        - m: 样本总数

        返回:
        - j: 随机选择的另一个样本的索引
        """
        j = i
        while j == i:  # 保证选择不同于i的样本
            j = np.random.randint(0, m)
        return j

    def predict(self, X):
        """
        使用训练好的SVM模型对新数据进行预测。

        参数:
        - X: 新的特征数据，形状为(m, n)

        返回:
        - predictions: 预测结果，+1或-1
        """
        return np.sign(X @ self.w + self.b)  # 根据w和b计算预测结果并返回类别（+1或-1）
