# 动手学机器学习

## 一、线性回归

### 题目一：采用 scikit-learn 中的 LinearRegression(最小二乘)线性回归模型对波士顿房价数据集进行预测，分别使用正则方程和随机梯度下降方法建模。

#### 具体内容：

1. **导入数据**  
   a) 查看数据集的描述、特征名、标签名、数据样本量等信息。  
   b) 获取样本的特征数据和标签数据。

2. **划分数据**  
   - 将数据划分为训练集和测试集。

3. **数据归一化**  
   - 对数据进行归一化处理。

4. **训练模型**  
   a) 使用 sklearn 中线性回归的正规方程（LinearRegression）优化方法建模。  
   b) 使用 sklearn 中线性回归的随机梯度下降（SGDRegressor）优化方法建模。

5. **模型评估（2 个模型）**  
   - 评价指标：MSE 和 R²值。

#### 讨论：

- **讨论一：梯度下降和正规方程两种算法有何不同？**  
   分析梯度下降和正规方程两种算法的差异（计算时间、评价指标对比）与优劣点。提示：Python 中计时器 `timeit.default_timer()` 方法。

- **讨论二：数据归一化对算法有什么影响？**  
   对比使用数据归一化和不使用数据归一化，正规方程和梯度下降算法性能是否有差异？分析原因。

- **讨论三：梯度下降算法中的学习率如何影响其工作？**  
   尝试修改随机梯度下降算法(SGDRegressor)的学习率（eta0），观察参数对模型性能的影响。试分析学习率与模型性能之间的关系。

- **讨论四：模型的泛化能力如何？**  
   1. 分别计算模型在训练样本上性能和在测试样本上的性能，判断模型是过拟合还是欠拟合？  
   2. 数据集划分不同对模型性能是否有影响？可尝试修改方法 `train_test_split` 中的 `test_size` 参数，观察数据集划分对模型性能的影响。  
   3. 尝试使用其他线性回归模型。线性回归模型中除了 `LinearRegression`，还有 `Ridge`（岭回归）、`Lasso`、`Polynomial regression`（多项式回归）等模型，使用不同模型进行建模，观察不同模型训练后的模型权重差异，试分析模型的使用场合。

### 题目二：采用梯度下降法（BGD）优化线性回归模型，对波士顿房价进行预测。

#### 具体内容：

1. **导入数据**  
   - 从 `.csv` 文件中导入数据。

2. **划分数据**  
   - 将数据划分为训练集和测试集。

3. **数据归一化**  
   - 对数据进行归一化处理。

4. **训练模型**  
   a) 初始化参数 `w`，可使用 `np.concatenate` 数组拼接函数，将截距与权重参数合并在一起（也可以不拼接合并）。  
   b) 求 `f(x)`。  
   c) 求 `J(w)`。  
   d) 求梯度。  
   e) 更新参数 `w`。  
   (b-e) 的过程经过 `epochs` 次迭代。

5. **画出损失函数随迭代次数的变化曲线**  
   - 通过损失函数变化曲线来观察梯度下降执行情况。

6. **测试集数据进行预测，模型评估**  
   - 评估模型在测试集上的性能。

7. **可视化**  
   - 展示数据拟合的效果。

8. **小批量梯度下降算法（MBGD）的编程实现**  
   - 实现小批量梯度下降（MBGD）。

---

## 二、逻辑回归

### 题目一：采用 scikit-learn 中的 LogisticRegression 逻辑回归模型对 iris 数据集进行二分类。

#### 具体内容：

1. **特征可视化**  
   - 任选两个特征和两种类别进行散点图可视化，观察是否线性可分。

2. **模型建立**  
   - 使用选取的特征和两种类别建立二分类模型。

3. **输出**  
   - 输出决策函数的参数、预测值、分类准确率等。

4. **决策边界可视化**  
   - 将二分类问题的边界可视化。

### 题目二：采用 scikit-learn 中的 LogisticRegression 逻辑回归模型对 iris 数据集进行多分类。

#### 具体内容：

1. **模型建立**  
   - 任选两个特征和全部类别进行散点图可视化，并建立多分类模型。

2. **输出**  
   - 输出决策函数的参数、预测值、分类准确率等。

3. **决策边界可视化**  
   - 将多分类问题的边界可视化。  
   提示：可以使用 `numpy` 中的 `meshgrid` 生成绘图网格数据，使用 `matplotlib` 中的 `contourf` 将等高线之间颜色进行填充。

#### 讨论：

- **讨论一：不同多分类策略的效果如何？有何差异？**  
   1. 尝试对比 `LogisticRegression` 中的 `multi_class='ovr'` 或 `'multinomial'` 两种多分类的差异。  
   2. 尝试使用 Multiclass classification 中提供的 3 种多分类策略，并对比效果。  
   提示：进行对比时，要保证数据集划分一致且分析的特征一致。可从训练集、测试集准确率和边界可视化角度进行对比。

### 题目三：采用 scikit-learn 中的 LogisticRegression 逻辑回归模型对非线性数据集进行分类。

#### 具体内容：

1. **数据集**  
   - 使用 sklearn 自带数据生成器 `make_moons` 产生两类数据样本，示例程序如下，参数可自行修改。

2. **特征衍生（数据增强）**  
   - 使用 sklearn 自带的 `sklearn.preprocessing.PolynomialFeatures` 生成指定阶次的多项式特征，从而得到所有多项式组合成的新特征矩阵，`degree` 参数任选。

3. **模型建立**  
   - 在新特征基础上建立逻辑回归二分类模型。

4. **决策边界可视化**  
   - 绘制决策边界，观察非线性边界的变化。

#### 讨论：

- **讨论二：在不加正则项的情况下，改变特征衍生的特征数量（即 `degree` 参数），观察决策边界的变化情况，以及训练集和测试集分数，体会模型从欠拟合 -> 拟合 -> 过拟合的过程。**  
   提示：可使用 `for` 循环对不同 `degree` 进行遍历，观察模型的建模结果。可通过绘制训练集和测试集分数曲线帮助观察（如示例图）。

- **讨论三：在讨论二的基础上选择一种模型过拟合的 `degree`，在模型中分别加入 'l1' 和 'l2' 正则项，观察决策边界的变化情况，以及训练集和测试集分数，体会两种正则项对模型的作用。**

- **讨论四：可尝试手动调整 `degree`、正则项系数 `C` 和正则项种类，寻找使模型泛化性能最好的一组参数。**  
   提示：手动调参采用“单一变量”原则。可先设定正则项种类（如 'l1'）和正则项系数 `C`（如默认），再人为设定特征最高阶次 `degree` 的范围进行 `degree` 寻优，在选定的 `degree` 和 'l1' 正则化后，设定正则项系数 `C` 的范围进行寻优。

### 题目四：使用 numpy 编写逻辑回归算法，对 iris 数据进行二分类。

#### 具体内容：

1. **任选两个特征和两个类别进行二分类。**

2. **输出**  
   - 输出决策函数的参数、预测值、分类准确率等。

3. **可视化**  
   - 选取两个特征进行散点图可视化，并可视化决策边界。

### 题目五：使用 numpy 编写逻辑回归算法，对 iris 数据进行多分类。

#### 具体内容：

- 输出决策函数的参数、预测值、分类准确率等。

#### 提示：

1. 可采用 OVR、OVO、ECOC 策略。
2. 可采用 CrossEntropy Loss +

 softmax 策略：
   a) 需将三个类别（如 0, 1, 2）进行 one-hot 编码。  
   b) 每个线性分类器对应一组模型参数，3 个线性分类器对应 3 组模型参数。  
   c) 可通过 softmax 回归计算多种类别的概率（K 种类别概率和为 1）。  
   d) 通过最小化 CrossEntropy Loss 的梯度下降算法进行分类器参数寻优。

--- 

## 三、支持向量机

### 题目一：采用 scikit-learn 中的线性 SVM 对 iris 数据集进行二分类

#### 具体内容：

1. **选取两个特征和两类数据使用 scikit-learn 中的 SVM 进行二分类**  
   - 使用线性 SVM 进行二分类建模。

2. **输出**  
   - 输出决策边界的参数和截距、支持向量等信息。

3. **可视化**  
   - 通过散点图可视化数据样本（之前选择的两个特征），并画出决策边界和 2 个最大间隔边界，标出支持向量。

#### 讨论：

- **讨论一：选取的两个特征能否线性可分？若线性可分，可选择 scikit-learn 中何种 SVM 进行建模？若线性不可分，可选择 scikit-learn 中何种 SVM 进行建模？**

- **讨论二：SVM 中的惩罚系数 C 对模型有何影响？**  
   1. 尝试改变惩罚系数 `C`，分析其变化对应间隔宽度、支持向量数量的变化趋势，并解释原因。  
   2. 尝试改变惩罚系数 `C`，分析其对 iris 分类模型性能的影响，并解释原因。

### 题目二：采用不同的 SVM 核函数对多种类型数据集进行二分类

#### 具体内容：

1. **生成数据集**  
   - 使用 scikit-learn 中提供的样本生成器 `make_blobs`、`make_classification`、`make_moons`、`make_circles` 生成一系列线性或非线性可分的二类别数据（数据量任取）。

2. **建模**  
   - 分别将 SVM 中四种核函数（线性核、多项式核、高斯核、S 形核）用于上述四种数据集。提示：对于每一种核函数，选择最适合的核参数（如 RBF 核中 `gamma`，多项式核中 `degree` 等）。可通过超参数曲线帮助选择超参数。

3. **可视化**  
   - 通过散点图可视化数据样本，并画出 SVM 模型的决策边界。

4. **模型评价**  
   - 分类准确率。

#### 讨论：

- **讨论三：如何选择最优超参数？**  
   为每种模型选择适合的核函数及核参数，参数寻优方式自选。

- **讨论四：不同核函数在不同数据集上表现如何？**  
   通过观察不同核函数在不同数据集上的决策边界和分类准确率，分析不同核函数的适用场合。

### 题目三：使用 scikit-learn 中的 SVM 分类器对乳腺癌威斯康星州数据集进行分类

#### 具体内容：

1. **导入数据集**  
   - 乳腺癌威斯康星州数据集是 sklearn 中自带的数据集（`load_breast_cancer`）。通过查看数据量和维度、特征类型（离散或连续）、特征名、标签名、标签分布情况、数据集的描述等信息了解数据集。

2. **建模**  
   - 分别使用四种核函数对数据集进行分类。

3. **模型评价**  
   - 每种核函数下的分类准确率、计算时间等。

#### 讨论：

- **讨论五：四种核函数在这个数据集上表现如何？**  
   提示：不要求可视化，从准确率上判断即可。

- **讨论六：SVM 是否需要进行数据归一化处理？数据归一化对核函数有何影响？**  
   提示：尝试分析数据归一化对四种核函数的工作有何影响，从分类准确率、计算时间等角度对比。

### 题目四：编写 SMO 算法实现线性 SVM 分类器，对 iris 数据集进行二分类

#### 具体内容：

1. **选取两个特征和两类数据进行二分类**  
   - 注意：二分类标签为 1 和 -1。

2. **划分数据**  
   - 将数据划分为训练集和测试集。

3. **数据归一化**  
   - 对数据进行归一化处理。

4. **训练模型**  
   - 参考程序模板 `SVM_numpy_template.py`。

5. **输出**  
   - 输出 SVM 对偶问题目标函数的最优解 `𝛼`，决策函数的参数和截距，支持向量等。

6. **可视化**  
   - 通过散点图可视化训练数据样本，并画出决策面和 2 个最大间隔面，标出支持向量（包括间隔上和间隔内的样本），帮助检验算法正确性。

7. **测试集数据进行预测，评估模型性能**  
   - 在测试集上进行预测并评估模型性能。

#### 讨论：

- **讨论七：请根据实验结果描述软间隔 SVM 中的 C 参数、拉格朗日乘子 `α`、支持向量与最优决策面和间隔区域之间的关系。**

---

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

---

## 五、随机森林

### 题目一：采用 scikit-learn 中的 RandomForestRegressor 对加利福尼亚房价数据集进行预测

#### 具体要求：

1. **导入数据集**  
   加利福尼亚房价数据集是 sklearn 中自带的数据集。通过查看数据量、维度、特征类型（离散或连续）、特征名、标签名、标签分布情况等信息了解数据集。

2. **模型建立**  
   - 使用 `DecisionTreeRegressor` 和 `RandomForestRegressor` 分别建立预测模型，参数使用默认设置。
   
3. **模型评估**  
   - 输出训练集和测试集评分，以根均方误差（RMSE）为评估指标。
   - 使用 `cross_validate` 来评估模型，可以通过 `scoring` 参数设置多种评分指标，如 RMSE。
   - 如果需要训练集和测试集的评分，确保使用 `return_train_score=True`。

#### 提示：
- `cross_val_score` 只能得到测试集的分数，且只能使用一种评分指标。  
- `cross_validate` 允许使用多个评分指标，并返回训练时间、测试时间、测试集分数和训练集分数。

#### 讨论：

- **讨论一：比较随机森林和决策树在数据集上的表现**  
   - 比较两种模型的 RMSE，分析交叉验证的评分，尝试通过可视化分析两者在加利福尼亚房价数据集上的表现差异。
   
- **讨论二：随机森林中的 `n_estimators` 超参数如何选择？**  
   - 可采用学习曲线来选择 `n_estimators` 的超参数范围，观察不同 `n_estimators` 下模型性能变化，确定最佳值。

- **讨论三：选择合适的超参数**  
   - 使用不同的超参数搜索方法（如网格搜索、随机搜索）来寻找最优超参数，最终在交叉验证集上建模并计算 RMSE 评分。  
   - 介绍调参过程，并比较调参前后的效果。

### 题目二：编写随机森林算法，并对葡萄酒数据/加利福尼亚房价数据（任选其一）进行预测，并展示模型评分，与 sklearn 自带的评估器建模结果进行对比。

#### 具体要求：
1. 编写一个自定义的随机森林算法。
2. 使用葡萄酒数据或加利福尼亚房价数据集进行建模。
3. 比较自定义算法与 sklearn 的 `RandomForestRegressor` 的模型评分。

---

## 六、Adaboost

### 题目一：采用 scikit-learn 中的 AdaBoostClassifier 对葡萄酒数据集进行预测

#### 具体要求：

1. **导入数据集**  
   使用之前决策树中提到的葡萄酒数据集。

2. **模型建立**  
   使用 `AdaBoostClassifier` 建立分类模型，使用默认参数。

3. **输出**  
   输出模型评分，使用交叉验证获得模型的综合评分。

#### 讨论：

- **讨论一：模型在葡萄酒数据集上的表现如何？是过拟合还是欠拟合？**
   - 通过交叉验证结果和学习曲线，判断模型是否有过拟合或欠拟合问题。

- **讨论二：如何提升模型性能？模型超参数对性能有何影响？**  
   - 观察不同求解算法（"SAMME" 和 "SAMME.R"）的性能差异。  
   - 参考：[官网示例](https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html#sphx-glr-auto-examples-ensemble-plot-adaboost-hastie-10-2-py)。

- **讨论三：通过学习曲线分析 AdaBoost 算法中的 `n_estimators` 参数对模型性能的影响**  
   - 观察不同 `n_estimators` 下模型的学习曲线，分析模型性能的变化趋势。

- **讨论四：通过学习曲线分析 AdaBoost 算法中的 `learning_rate` 参数对模型性能的影响**  
   - 展现不同学习率下，模型性能随 `n_estimators` 变化的曲线。

- **讨论五：综合进行超参数选择，找到最好的一组超参数**  
   - 在交叉验证集上验证模型效果，并与其他算法（如决策树、随机森林等）进行对比。

### 题目二：编写 AdaBoost-SAMME 算法，并对乳腺癌数据集进行预测，展示模型评分。

#### 具体要求：

1. 编写 AdaBoost-SAMME 算法。
2. 对乳腺癌数据集进行预测，并展示模型评分。

#### 讨论：

- **讨论六：与 sklearn 自带的评估器建模结果进行对比**  
   比较自定义的 AdaBoost-SAMME 实现与 sklearn 的内置 AdaBoost 模型的评分。

### 题目三：采用 scikit-learn 中的 GradientBoostingRegressor 对加利福尼亚房价数据集进行预测

#### 具体要求：

1. **导入数据集**  
   使用加利福尼亚房价数据集。

2. **模型建立**  
   使用 `GradientBoostingRegressor` 建立回归模型。

3. **输出**  
   输出模型评分，使用交叉验证得到模型的综合评分。

#### 讨论：

- **讨论七：自行选择超参数寻优的方法，确定模型最优超参数，并将建模结果与之前学过的调参后的模型（如随机森林、Adaboost）进行比较**  
   - 通过学习曲线、袋外数据、提前停止等手段，帮助判断超参数范围，并确定最优参数空间。  
   - 比较 `GradientBoostingRegressor` 与随机森林、AdaBoost 等模型的建模时间和预测效果。


---


## 七、BP神经网络

## 1. 题目一：采用 scikit-learn 中的 MLPClassifier 对红酒数据集进行分类，并通过特征和边界的可视化，直观体会多层感知机网络中的隐层上神经元数量、隐层层数、激活函数、正则化项系数等超参数对模型复杂程度的影响。

### 具体内容：

1. **选取前两个特征，建立多层感知机网络进行多分类。**

2. **可视化**  
   - 通过散点图可视化数据样本（之前选择的两个特征），并画出模型训练后得到的决策边界。

#### 讨论：

- **讨论一：改变单隐层中神经元个数（如 10 个，100 个），其他参数不变，观察其对决策边界的影响。**
  
- **讨论二：改变神经网络深度（如深度为 2，每层 10 个神经元），其他参数不变，与讨论一进行对比，观察神经网络深度对决策边界的影响。**

- **讨论三：在讨论一（或讨论二）的基础上，改变激活函数（如 `tanh`、`relu`），与讨论一（或讨论二）进行对比，观察不同激活函数对决策边界的影响。**

- **讨论四：在讨论三的基础上，增大正则化系数，观察正则化对决策边界的影响。**

#### 总结：

- 综合上述讨论，隐层上神经元数量、隐层层数、激活函数、正则化项系数对模型复杂程度有何影响。

---

## 2. 题目二：采用 scikit-learn 中的 MLPClassifier 对自带手写数字数据集进行分类。

### 具体要求：

1. **导入数据集**  
   - 手写数字集是 sklearn 中自带的数据集，它是一个三维数组 `(1797, 8, 8)`，即有 1797 个手写数字，每个数字由 8×8 的像素矩阵组成。矩阵中每个元素都是 0-16 范围内的整数。分类标签为 0-9 的数字。

2. **模型建立**  
   - 使用 `MLPClassifier` 建立分类模型。

3. **输出**  
   - 输出分类结果的准确率。

#### 讨论：

- **讨论五：结合模型复杂度与模型泛化误差之间的关系，调节模型超参数，提升模型泛化性能。**  
   可尝试调节隐层神经元个数和隐层数、激活函数、学习率、正则项系数等超参数。

---

## 3. 题目三：编写 BPNN 算法，对 iris 数据集/手写数字集进行二分类或多分类。

### 具体要求：

1. **数据样本标签处理**  
   - 二分类任务：正类为 1，负类为 0。  
   - 多分类任务：将样本标签变为 one-hot 向量。

2. **搭建浅层神经网络**  
   - 隐层数 1-2 个即可。每层神经元的个数自选。

3. **激活函数**  
   - 自选（`relu`、`sigmoid`、`tanh`）。  
   - 代价函数：自选（交叉熵损失、均方误差）。

4. **输出**  
   - 输出分类准确率。

5. **可视化**  
   - 迭代的代价函数曲线。

6. **尝试手写 BP 网络链式法则的反向传播计算过程**。

#### 注意事项：

- **提高运算效率**  
   - 算法编写尽量使用向量化技术，避免使用 `for` 循环遍历样本和神经元。

- **权重和偏置计算**  
   - 一般将权重 `w` 和偏置 `b` 分开进行计算。

- **初始化权重**  
   - 为避免梯度消失或梯度爆炸，通常采用随机初始化权重 `w`，而不是将权重全部初始化为 0 或 1。  
     示例：若要生成服从 `𝒩(0, √(2/(𝑛𝑖𝑛+𝑛𝑜𝑢𝑡)))` 分布的随机数，可使用如下程序：  
     ```python
     np.random.randn(m, n) * np.sqrt(2 / (nin + nout))
     ```
     其中 `nin` 为神经元的输入连接数量，`nout` 为神经元的输出连接数量，`m` 和 `n` 分别为返回数组的行数和列数。

- **注意数组维度**  
   - 向量计算过程中，要注意数组维度，避免出现异常 bug。避免使用一维数组（例如 `np.random.randn(5)`），这不是列向量也不是行向量！  
     可通过下面示例 `np.random.randn(5, 1)` 或 `np.random.randn(1, 5)` 创建列向量或行向量，或使用 `reshape` 方法变换数组的维度。

--- 