
## 四、决策树

### 题目一：采用 scikit-learn 中的 DecisionTreeClassifier 决策树对葡萄酒数据集进行预测

#### 具体内容：

1. **导入数据集**  
   - 葡萄酒数据集是 sklearn 中自带的数据集，API 使用参考网址：[sklearn.datasets.load_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine)。  
   通过查看数据量和维度、特征类型（离散或连续）、特征名、标签名、标签分布情况、数据集的描述等信息了解数据集。

2. **模型建立**  
   - 使用全部特征建立决策树多分类模型（树模型参数可按默认设置）。

3. **输出**  
   - 特征重要程度、分类准确率、绘制树形图。

#### 讨论：

- **讨论一：模型参数对模型性能有何影响？**  
   1. 不同特征选择标准（`criterion='gini'` 或 `'entropy'`）对模型性能是否有影响？  
   2. 不同特征划分标准（`splitter='best'` 或 `'random'`）对模型性能是否有影响？  
   3. 尝试修改 `max_depth`、`min_samples_leaf`、`min_samples_split` 等参数，通过树形图分析参数的作用。

- **讨论二：如何确定最优的剪枝参数？**  
   找到合适的超参数，展示调整后的模型效果。可使用学习曲线进行超参数选取，其优点是可以看到超参数对模型性能影响的趋势。如果需要确定的超参数比较多，且超参数之间相互影响时，可尝试使用 `GridSearchCV` 选择模型最优的超参数。

### 题目二：使用 scikit-learn 中的 DecisionTreeClassifier 决策树对 kddcup99 数据集进行预测

#### 具体内容：

1. **数据集介绍**  
   - `kddcup99` 数据集是 KDD 竞赛在 1999 年举行时采用的数据集，是网络入侵检测领域的真实数据，具有 41 个特征，类别数量有 23 种（1 种正常网络、22 种网络入侵类型），约五百万数据量。由于数据量很大，可以选择获取总数据量的 10%。

   数据集获取方法：  
   ```python
   sklearn.datasets.fetch_kddcup99(*, subset=None, data_home=None, shuffle=False, random_state=None, percent10=True, download_if_missing=True, return_X_y=False, as_frame=False)
   ```

   - [Kaggle 官网数据源](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

2. **数据预处理**  
   - 由于 `kddcup99` 数据集中的第 2、3、4 列特征为文本信息，需要重新编码。类别标签也为文本数据，需要编码操作。选择适合的编码方式，对数据进行编码之后，再进行建模。

3. **选择评价指标**  
   - 选择适合的评价指标，尝试调参，建立效果最好的一个模型。

### 题目三：使用 numpy 编写的 CART 分类/回归树（选择一种即可）算法，并对 iris 数据集/california 数据集进行预测

#### 具体内容：

1. **导入数据集**  
   - 导入数据集。

2. **划分数据**  
   - 将数据划分为训练集和测试集。

3. **训练模型**  
   - 参考程序模板 `cart_numpy_template.py` 进行训练。

4. **输出树模型**  
   - 输出训练后的决策

树模型。

5. **模型预测**  
   - 在测试集上进行预测，评估模型性能。

#### 拓展内容：

- **尝试加入 TN 样本数量阈值和 TG 基尼指数阈值作为终止条件。**
- **尝试对离散特征进行分枝。**

#### 讨论：

- **讨论四：树递归分枝的终止条件是什么？展示对应的代码。**  
  请结合代码简述树的递归分枝的过程。
