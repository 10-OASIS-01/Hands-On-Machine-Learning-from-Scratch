## VI. AdaBoost

### [1. Predict the Wine Dataset Using AdaBoostClassifier from scikit-learn](ML6_1.ipynb)

#### Requirements:

1. **Load the Dataset**  
   Use the Wine dataset mentioned previously in the Decision Tree section.

2. **Model Building**  
   Use `AdaBoostClassifier` to build a classification model with default parameters.

3. **Output**  
   Output the model score, and obtain the overall model score using cross-validation.

#### Discussion:

- **Discussion 1: How does the model perform on the Wine dataset? Is it overfitting or underfitting?**  
   - Use cross-validation results and learning curves to determine if the model is overfitting or underfitting.

- **Discussion 2: How can model performance be improved? What impact do hyperparameters have on performance?**  
   - Observe the performance differences between different solving algorithms (`"SAMME"` and `"SAMME.R"`).  
   - Reference: [Official Example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html#sphx-glr-auto-examples-ensemble-plot-adaboost-hastie-10-2-py).

- **Discussion 3: Analyze the impact of the `n_estimators` parameter on model performance through learning curves in AdaBoost.**  
   - Observe how the model performance changes with different values of `n_estimators` and analyze the trend in the learning curves.

- **Discussion 4: Analyze the impact of the `learning_rate` parameter on model performance in AdaBoost using learning curves.**  
   - Show how the model performance changes with `n_estimators` under different learning rates.

- **Discussion 5: Perform hyperparameter tuning and find the best set of hyperparameters.**  
   - Validate the model on the cross-validation set and compare the results with other algorithms (e.g., Decision Tree, Random Forest).

### [2. Implement AdaBoost-SAMME Algorithm and Predict on the Breast Cancer Dataset, Display the Model Score](ML6_2.ipynb)

The `AdaBoost.py` file contains the implementation of the AdaBoost-SAMME algorithm.

#### Requirements:

1. Implement the AdaBoost-SAMME algorithm.
2. Predict on the Breast Cancer dataset and display the model score.

#### Discussion:

- **Discussion 6: Compare the custom AdaBoost-SAMME implementation with the built-in AdaBoost model from scikit-learn.**  
   - Compare the scores of the custom AdaBoost-SAMME implementation with the results from scikit-learn's built-in AdaBoost model.

### [3. Predict the California Housing Dataset Using GradientBoostingRegressor from scikit-learn](ML6_3.ipynb)

#### Requirements:

1. **Load the Dataset**  
   Use the California housing dataset.

2. **Model Building**  
   Use `GradientBoostingRegressor` to build a regression model.

3. **Output**  
   Output the model score, and obtain the overall model score using cross-validation.

#### Discussion:

- **Discussion 7: Choose a hyperparameter optimization method, determine the optimal model hyperparameters, and compare the results with previously learned tuned models (e.g., Random Forest, AdaBoost).**  
   - Use learning curves, out-of-bag data, early stopping, etc., to help identify the hyperparameter range and find the optimal parameter space.  
   - Compare the training time and prediction results of `GradientBoostingRegressor` with Random Forest, AdaBoost, and other models.
