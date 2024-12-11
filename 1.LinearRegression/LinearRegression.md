## I. Linear Regression

### [1. Use the LinearRegression (Least Squares) model from scikit-learn to predict the Boston Housing dataset, and model using both the Normal Equation and Stochastic Gradient Descent methods.](ML1_1.ipynb)

#### Specific Content:

1. **Importing Data**  
   a) Explore the dataset’s description, feature names, label names, number of samples, and other relevant information.  
   b) Retrieve the feature data and label data from the samples.

2. **Data Splitting**  
   - Split the data into training and testing sets.

3. **Data Normalization**  
   - Perform normalization on the data.

4. **Training the Model**  
   a) Use the LinearRegression model from scikit-learn to optimize the model using the Normal Equation method.  
   b) Use the SGDRegressor model from scikit-learn to optimize the model using the Stochastic Gradient Descent (SGD) method.

5. **Model Evaluation (for both models)**  
   - Evaluation Metrics: MSE (Mean Squared Error) and R² score.

#### Discussion:

- **Discussion 1: What are the differences between the Gradient Descent and Normal Equation algorithms?**  
   Analyze the differences between Gradient Descent and the Normal Equation algorithms (computation time, comparison of evaluation metrics) and their pros and cons. Hint: Use Python’s timer `timeit.default_timer()` method.

- **Discussion 2: How does data normalization affect the algorithm?**  
   Compare the performance of both the Normal Equation and Gradient Descent algorithms with and without data normalization. Analyze the reason for any performance differences.

- **Discussion 3: How does the learning rate in Gradient Descent affect its performance?**  
   Modify the learning rate (eta0) of the SGDRegressor and observe its impact on model performance. Try to analyze the relationship between the learning rate and model performance.

- **Discussion 4: What is the model's generalization ability?**  
   1. Calculate the model’s performance on both the training and testing samples, and determine if the model is overfitting or underfitting.  
   2. Does the data split ratio affect model performance? Try adjusting the `test_size` parameter in the `train_test_split` function and observe how the dataset split impacts performance.  
   3. Try using other linear regression models. In addition to `LinearRegression`, there are models like `Ridge` (Ridge Regression), `Lasso`, and `Polynomial Regression`. Use different models to observe the variations in the model weights after training, and analyze the use cases of each model.

### [2. Implement the Linear Regression algorithm from scratch using numpy, optimizing it with Gradient Descent (BGD) to predict the Boston Housing prices.](ML1_2.ipynb)

The algorithm implementation code is provided in `LinearRegression.py`.

#### Specific Content:

1. **Importing Data**  
   - Import the data from a `.csv` file.

2. **Data Splitting**  
   - Split the data into training and testing sets.

3. **Data Normalization**  
   - Normalize the data.

4. **Training the Model**  
   a) Initialize the parameters `w`. You can use the `np.concatenate` function to combine the intercept and weights into one array (or leave them separate).  
   b) Compute `f(x)`.  
   c) Compute `J(w)` (the cost function).  
   d) Compute the gradient.  
   e) Update the parameters `w`.  
   Steps (b)-(e) are iterated for a specified number of `epochs`.

5. **Plotting the Loss Function Curve**  
   - Plot the loss function curve over iterations to observe the progress of gradient descent.

6. **Prediction on the Test Set and Model Evaluation**  
   - Evaluate the model’s performance on the test set.

7. **Visualization**  
   - Visualize the data and the model’s fitting results.

8. **Mini-batch Gradient Descent (MBGD) Implementation**  
   - Implement the Mini-batch Gradient Descent (MBGD) algorithm.
