## V. Random Forests

### [1. Predict the California Housing Dataset Using RandomForestRegressor from scikit-learn](ML5_1.ipynb)

#### Requirements:

1. **Load the Dataset**  
   The California housing dataset is a built-in dataset in scikit-learn. Explore the dataset by reviewing its size, dimensions, feature types (discrete or continuous), feature names, label names, label distribution, and other descriptive information.

2. **Model Building**  
   - Build prediction models using `DecisionTreeRegressor` and `RandomForestRegressor`, with default parameter settings.
   
3. **Model Evaluation**  
   - Output the training and testing scores using Root Mean Squared Error (RMSE) as the evaluation metric.
   - Use `cross_validate` to assess the model, and specify the `scoring` parameter to set multiple evaluation metrics like RMSE.
   - To obtain both training and testing scores, ensure `return_train_score=True` is set.

#### Tips:
- `cross_val_score` only provides the test score and can only use one evaluation metric.  
- `cross_validate` allows multiple evaluation metrics and returns training time, testing time, test scores, and training scores.

#### Discussion:

- **Discussion 1: Compare the performance of Random Forest and Decision Tree on the dataset**  
   - Compare the RMSE of both models, analyze cross-validation scores, and attempt to visualize the differences in their performance on the California housing dataset.

- **Discussion 2: How to choose the `n_estimators` hyperparameter in Random Forest?**  
   - Use learning curves to select a range for the `n_estimators` hyperparameter, observe how model performance changes with different values of `n_estimators`, and determine the optimal value.

- **Discussion 3: Selecting the Appropriate Hyperparameters**  
   - Use different hyperparameter search methods (such as grid search or random search) to find the optimal hyperparameters. Finally, build a model on the cross-validation set and calculate the RMSE score.  
   - Explain the hyperparameter tuning process and compare the model performance before and after tuning.

### [2. Implement a Random Forest Algorithm and Predict on the Wine or California Housing Dataset (Choose One), Comparing the Model Scores with the Results from scikit-learn's Built-in Evaluator](ML5_2.ipynb)

The `RandomForest.py` file contains the specific implementation of the Random Forest algorithm.

#### Requirements:

1. Implement a custom Random Forest algorithm.
2. Use either the Wine dataset or the California housing dataset for modeling.
3. Compare the model scores of your custom algorithm with those of the scikit-learn `RandomForestRegressor`.

