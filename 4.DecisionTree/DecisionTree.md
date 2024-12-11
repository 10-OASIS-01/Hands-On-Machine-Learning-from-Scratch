## IV. Decision Trees

### [1. Predict the Wine Dataset Using DecisionTreeClassifier from scikit-learn](ML4_1.ipynb)

#### Details:

1. **Load the Dataset**  
   - The Wine dataset is a built-in dataset in scikit-learn. You can refer to the API documentation here: [sklearn.datasets.load_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine).  
   Explore the dataset by reviewing its size, dimensions, feature types (discrete or continuous), feature names, label names, label distribution, and other descriptive information.

2. **Model Building**  
   - Build a decision tree classifier for multi-class classification using all features (you can use the default settings for tree model parameters).

3. **Output**  
   - Feature importance, classification accuracy, and a tree diagram.

#### Discussion:

- **Discussion 1: How do the model parameters affect model performance?**  
   1. Does changing the feature selection criterion (`criterion='gini'` or `'entropy'`) affect model performance?  
   2. Does changing the feature splitting criterion (`splitter='best'` or `'random'`) affect model performance?  
   3. Try modifying parameters like `max_depth`, `min_samples_leaf`, and `min_samples_split`, and analyze their effects through the tree diagram.

- **Discussion 2: How to determine the optimal pruning parameters?**  
   Find suitable hyperparameters and show the model's performance after tuning. You can use learning curves to help choose hyperparameters, as they allow you to observe trends in how hyperparameters affect model performance. If many hyperparameters need to be determined, and they are interdependent, consider using `GridSearchCV` to select the best hyperparameters for the model.

### [2. Predict the KDD Cup 99 Dataset Using DecisionTreeClassifier from scikit-learn](ML4_2.ipynb)

#### Details:

1. **Dataset Overview**  
   - The `kddcup99` dataset is the dataset used in the KDD Cup 1999 competition, which is a real-world dataset for network intrusion detection. It contains 41 features, with 23 classes (1 normal network, 22 types of network intrusions), and approximately 5 million data points. Due to the large size, you may choose to work with only 10% of the total dataset.

   Dataset can be accessed using:  
   ```python
   sklearn.datasets.fetch_kddcup99(*, subset=None, data_home=None, shuffle=False, random_state=None, percent10=True, download_if_missing=True, return_X_y=False, as_frame=False)
   ```

   - [Kaggle dataset source](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

2. **Data Preprocessing**  
   - Columns 2, 3, and 4 in the `kddcup99` dataset contain textual features, which need to be re-encoded. The class labels are also textual, requiring encoding. Choose appropriate encoding methods, preprocess the data, and then proceed with modeling.

3. **Choose Evaluation Metrics**  
   - Select suitable evaluation metrics and try tuning parameters to build the best-performing model.

### [3. Implement CART Classification/Regression Tree Algorithm (Choose One) Using Numpy and Predict on the Iris/California Dataset](ML4_3.ipynb)

`CART.py` contains the specific implementation of the CART classification/regression tree algorithm.

#### Details:

1. **Load the Dataset**  
   - Import the dataset.

2. **Split the Data**  
   - Divide the data into training and testing sets.

3. **Train the Model**  
   - Train the model following the program template `cart_numpy_template.py`.

4. **Output the Tree Model**  
   - Output the trained decision tree model.

5. **Model Prediction**  
   - Make predictions on the test set and evaluate model performance.

#### Extension:

- **Try adding a threshold for the number of TN (True Negative) samples and the Gini index threshold for TG as stopping conditions.**
- **Try branching on discrete features.**

#### Discussion:

- **Discussion 4: What are the stopping conditions for tree recursion branching? Please display the corresponding code.**  
   Briefly describe the recursive branching process of the tree using the code.
