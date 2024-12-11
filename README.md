# Hands-on Learning of Machine Learning from Scratch

This repository is designed for those who want to understand **machine learning from the ground up**. It provides practical, hands-on examples and clear explanations of essential machine learning concepts, algorithms, and techniques. Each section includes both **theoretical insights** and **Python code implementations** (often from scratch) to reinforce your understanding of the topics.

The guide is structured as a series of progressively more complex topics, starting with foundational algorithms and advancing to more sophisticated models, ensuring that you build a solid understanding of machine learning principles.

---

### Contents

1. **[Linear Regression](1.LinearRegression/LinearRegression.md)**
2. **[Logistic Regression](2.LogisticRegression/LogisticRegression.md)**
3. **[Support Vector Machines](3.SupportVectorMachine/SupportVectorMachine.md)**
4. **[Decision Trees](4.DecisionTree/DecisionTree.md)**
5. **[Random Forests](5.RandomForest/RandomForest.md)**
6. **[AdaBoost](6.AdaBoost/AdaBoost.md)**
7. **[Backpropagation Neural Networks (BP Neural Networks)](7.BackpropagationNeuralNetwork/BackpropagationNeuralNetwork.md)**

---

### Section Breakdown

#### **[I. Linear Regression](1.LinearRegression/LinearRegression.md)**

- **Theory**: Introduction to linear regression, the least squares method, and gradient descent.
- **Practice**:
  - Use scikit-learn's **LinearRegression** model to predict housing prices using the **Boston Housing dataset**.
  - Implement linear regression from scratch using **numpy**, comparing the **normal equation** with **stochastic gradient descent** (SGD).

#### **[II. Logistic Regression](2.LogisticRegression/LogisticRegression.md)**

- **Theory**: Learn how logistic regression applies to classification tasks, both binary and multi-class.
- **Practice**:
  - Use scikit-learn's **LogisticRegression** model for binary classification on the **Iris dataset**.
  - Extend logistic regression to multi-class classification and apply it to the **Iris dataset**.
  - Implement logistic regression from scratch using **numpy** for both binary and multi-class classification tasks.

#### **[III. Support Vector Machines (SVM)](3.SupportVectorMachine/SupportVectorMachine.md)**

- **Theory**: Understand the concept of hyperplanes, margins, and kernels in SVM.
- **Practice**:
  - Train a linear SVM on the **Iris dataset** for binary classification.
  - Experiment with different SVM kernel functions for various datasets.
  - Use the **Breast Cancer Wisconsin dataset** with scikit-learn’s SVM classifier.
  - Implement the **SMO algorithm** (Sequential Minimal Optimization) from scratch to build a linear SVM classifier.

#### **[IV. Decision Trees](4.DecisionTree/DecisionTree.md)**

- **Theory**: Understand decision trees' construction, splits, entropy, and Gini impurity.
- **Practice**:
  - Use the **DecisionTreeClassifier** from scikit-learn to predict the **Wine dataset** and the **KDD Cup 99 dataset**.
  - Implement the **CART algorithm** (Classification and Regression Trees) from scratch using **numpy**, and perform classification/regression on the **Iris dataset** or the **California Housing dataset**.

#### **[V. Random Forests](5.RandomForest/RandomForest.md)**

- **Theory**: Learn the fundamentals of ensemble learning and random forests.
- **Practice**:
  - Use the **RandomForestRegressor** from scikit-learn to predict the **California Housing dataset**.
  - Implement the **Random Forest** algorithm from scratch using **numpy**, and compare it with scikit-learn's built-in implementation on the **Wine dataset** or **California Housing dataset**.

#### **[VI. AdaBoost](6.AdaBoost/AdaBoost.md)**

- **Theory**: Understand the principles of boosting and how AdaBoost improves weak learners.
- **Practice**:
  - Use scikit-learn’s **AdaBoostClassifier** to predict the **Wine dataset**.
  - Implement the **AdaBoost-SAMME** algorithm from scratch to predict the **Breast Cancer dataset**.
  - Use the **GradientBoostingRegressor** to predict the **California Housing dataset**.

#### **[VII. Backpropagation Neural Networks (BPNN)](7.BackpropagationNeuralNetwork/BackpropagationNeuralNetwork.md)**

- **Theory**: Learn about neural networks, backpropagation, and gradient descent for training.
- **Practice**:
  - Use **MLPClassifier** from scikit-learn to classify the **Red Wine dataset**, and experiment with model complexity via hyperparameters (number of neurons, activation functions, etc.).
  - Apply **MLPClassifier** to classify the **handwritten digit dataset** (MNIST).
  - Implement the **Backpropagation Neural Network (BPNN)** from scratch using **numpy**, and apply it for binary or multi-class classification on the **Iris dataset** or the **handwritten digits dataset**.

---

### Key Features

- **Step-by-step code implementations**: Each section includes a combination of both theory and code so you can build a deep understanding of the algorithms.
- **From-scratch implementations**: Many algorithms are implemented from scratch using **numpy** to help reinforce core concepts.
- **Practical applications**: We use real-world datasets like **Iris**, **Wine**, **California Housing**, and **Breast Cancer** to test and evaluate our models.
- **Hands-on learning**: This repository provides an opportunity for interactive learning. You can experiment with different parameters, tweak models, and observe how they affect performance.

---

### How to Use This Repository

1. **Clone the repository**:
   ```bash
   git clone https://github.com/10-OASIS-01/Hands-On-Machine-Learning-from-Scratch.git
   cd machine-learning-from-scratch
   ```

2. **Set up the environment**: Make sure you have Python 3.x installed, then create and activate a virtual environment. Install required dependencies via `requirements.txt`.
   ```bash
   python -m venv env
   source env/bin/activate   # For Linux/Mac
   .\env\Scripts\activate    # For Windows
   ```

3. **Navigate to a topic**: Choose a directory (e.g., `1.LinearRegression`) and follow the notebook or markdown instructions to run the code and explore the topic.

4. **Experiment**: Feel free to modify the code and datasets to explore different machine learning concepts, experiment with hyperparameters, and analyze model performance.

---

### Contributing

If you find an issue, typo, or want to improve the repository, feel free to open an issue or a pull request. Contributions are welcome!

---

### License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
