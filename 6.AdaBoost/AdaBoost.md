## 六、Adaboost

### [1. 采用 scikit-learn 中的 AdaBoostClassifier 对葡萄酒数据集进行预测](ML6_1.ipynb)

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

### [2. 编写 AdaBoost-SAMME 算法，并对乳腺癌数据集进行预测，展示模型评分。](ML6_2.ipynb)

`AdaBoost.py`中为 AdaBoost-SAMME 算法的具体实现代码 

#### 具体要求：

1. 编写 AdaBoost-SAMME 算法。
2. 对乳腺癌数据集进行预测，并展示模型评分。

#### 讨论：

- **讨论六：与 sklearn 自带的评估器建模结果进行对比**  
   比较自定义的 AdaBoost-SAMME 实现与 sklearn 的内置 AdaBoost 模型的评分。

### [3. 采用 scikit-learn 中的 GradientBoostingRegressor 对加利福尼亚房价数据集进行预测](ML6_3.ipynb)

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
