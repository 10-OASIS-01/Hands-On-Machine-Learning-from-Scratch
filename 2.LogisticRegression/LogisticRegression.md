## II. Logistic Regression

### [1. Use the LogisticRegression model from scikit-learn for binary classification on the Iris dataset.](ML2_1.ipynb)

#### Specific Content:

1. **Feature Visualization**  
   - Select two features and two classes, and visualize them using a scatter plot to observe if they are linearly separable.

2. **Model Building**  
   - Build a binary classification model using the selected features and classes.

3. **Output**  
   - Output the decision function parameters, predicted values, and classification accuracy.

4. **Decision Boundary Visualization**  
   - Visualize the decision boundary for the binary classification problem.

### [2. Use the LogisticRegression model from scikit-learn for multi-class classification on the Iris dataset.](ML2_2.ipynb)

#### Specific Content:

1. **Model Building**  
   - Select two features and all classes, visualize the data using a scatter plot, and build a multi-class classification model.

2. **Output**  
   - Output the decision function parameters, predicted values, and classification accuracy.

3. **Decision Boundary Visualization**  
   - Visualize the decision boundary for the multi-class classification problem.  
   Hint: You can use `numpy`'s `meshgrid` to generate a grid for plotting, and `matplotlib`'s `contourf` to fill the color between contour lines.

#### Discussion:

- **Discussion 1: How do different multi-class strategies perform? What are the differences?**  
   1. Compare the multi-class strategies in `LogisticRegression`, specifically `multi_class='ovr'` and `multi_class='multinomial'`.  
   2. Compare the performance of the three multi-class strategies available in Multiclass classification.  
   Hint: When comparing, ensure the dataset split is consistent, and the features being analyzed are the same. You can compare the results based on training/test accuracy and decision boundary visualizations.

### [3. Use the LogisticRegression model from scikit-learn for classification on a non-linear dataset.](ML2_3.ipynb)

#### Specific Content:

1. **Dataset**  
   - Use the built-in `make_moons` dataset generator from sklearn to create two-class data samples. Modify the parameters as needed.

2. **Feature Engineering (Data Augmentation)**  
   - Use `sklearn.preprocessing.PolynomialFeatures` to generate polynomial features of a specified degree, creating a new feature matrix consisting of all polynomial combinations. The `degree` parameter is adjustable.

3. **Model Building**  
   - Build a logistic regression binary classification model using the new features.

4. **Decision Boundary Visualization**  
   - Plot the decision boundary and observe how the non-linear boundary changes.

#### Discussion:

- **Discussion 2: Without regularization, observe the effect of changing the number of derived features (i.e., the `degree` parameter) on the decision boundary and training/test scores. Experience the model's transition from underfitting -> fitting -> overfitting.**  
   Hint: You can loop through different values of `degree` to observe the model results. Plot the training and test scores over iterations to help visualize the process (e.g., score curves).

- **Discussion 3: Based on Discussion 2, choose a model that is overfitting and add 'l1' and 'l2' regularization to observe changes in the decision boundary and training/test scores. Analyze the effect of the two regularization terms on the model.**

- **Discussion 4: Try manually adjusting the `degree`, regularization coefficient `C`, and regularization type to find the optimal set of parameters for the best generalization performance.**  
   Hint: Use the "single-variable" tuning approach. Set the regularization type (e.g., 'l1') and coefficient `C` (e.g., default value), then manually adjust the maximum feature degree (`degree`) and optimize it. After choosing the best `degree` with 'l1' regularization, tune the regularization coefficient `C` to find the optimal model.

### [4. Implement Logistic Regression from scratch using numpy for binary classification on the Iris dataset.](ML2_4.ipynb)

The specific implementation code for the logistic regression algorithm can be found in `LogisticRegression.py`.

#### Specific Content:

1. **Choose two features and two classes for binary classification.**

2. **Output**  
   - Output the decision function parameters, predicted values, and classification accuracy.

3. **Visualization**  
   - Visualize the data with a scatter plot for the selected features and visualize the decision boundary.

### [5. Implement Logistic Regression from scratch using numpy for multi-class classification on the Iris dataset.](ML2_5.ipynb)

#### Specific Content:

- Output the decision function parameters, predicted values, and classification accuracy.

#### Tips:

1. You can use OVR, OVO, or ECOC strategies for multi-class classification.
2. Use the Cross-Entropy Loss with the Softmax strategy:

   a) Perform one-hot encoding on the three classes (e.g., 0, 1, 2).  
   b) Each linear classifier corresponds to a set of model parameters, and 3 classifiers will result in 3 sets of parameters.  
   c) Compute the probability of each class using softmax regression (the sum of the probabilities of all K classes should be 1).  
   d) Use gradient descent to minimize the Cross-Entropy Loss and optimize the classifier parameters.
