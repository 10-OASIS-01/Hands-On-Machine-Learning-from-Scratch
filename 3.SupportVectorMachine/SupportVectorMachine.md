## III. Support Vector Machine (SVM)

### [1. Perform Binary Classification on the Iris Dataset Using Linear SVM in scikit-learn](ML3_1.ipynb)

#### Details:

1. **Select two features and two classes of data for binary classification using SVM in scikit-learn**  
   - Use linear SVM to build a binary classification model.

2. **Output**  
   - Output parameters and intercept of the decision boundary, support vectors, and other related information.

3. **Visualization**  
   - Visualize data samples (the two selected features) using a scatter plot, draw the decision boundary and two maximum margin boundaries, and mark the support vectors.

#### Discussion:

- **Discussion 1: Can the two selected features be linearly separable? If they are linearly separable, which type of SVM in scikit-learn would be appropriate for modeling? If they are not linearly separable, which type of SVM in scikit-learn would be appropriate for modeling?**

- **Discussion 2: What impact does the penalty parameter C have on the SVM model?**  
   1. Try changing the penalty parameter C, analyze its effect on the margin width, the number of support vectors, and explain the reasons.  
   2. Try changing the penalty parameter C, analyze its effect on the performance of the Iris classification model, and explain the reasons.

### [2. Perform Binary Classification on Various Data Sets Using Different SVM Kernels](ML3_2.ipynb)

#### Details:

1. **Generate Data Sets**  
   - Use sample generators from scikit-learn such as `make_blobs`, `make_classification`, `make_moons`, and `make_circles` to create a series of linearly or non-linearly separable binary datasets (choose any size).

2. **Modeling**  
   - Apply four types of SVM kernels (linear kernel, polynomial kernel, Gaussian kernel, and sigmoid kernel) to the four datasets. Tip: For each kernel, choose the most appropriate kernel parameters (such as gamma for the RBF kernel, degree for the polynomial kernel, etc.). Hyperparameter tuning can be done using a hyperparameter search.

3. **Visualization**  
   - Visualize the data samples using a scatter plot and draw the decision boundary of the SVM model.

4. **Model Evaluation**  
   - Evaluate the classification accuracy.

#### Discussion:

- **Discussion 3: How to select the optimal hyperparameters?**  
   Choose the most suitable kernel and kernel parameters for each model and select the parameter tuning method.

- **Discussion 4: How do different kernels perform on different datasets?**  
   By observing the decision boundaries and classification accuracy for different kernels on different datasets, analyze the suitability of each kernel for specific scenarios.

### [3. Perform Classification on the Breast Cancer Wisconsin Dataset Using the SVM Classifier in scikit-learn](ML3_3.ipynb)

#### Details:

1. **Load the Dataset**  
   - The Breast Cancer Wisconsin dataset is built into scikit-learn (`load_breast_cancer`). Explore the dataset by reviewing its size, dimensions, feature types (discrete or continuous), feature names, label names, label distribution, and other descriptive information.

2. **Modeling**  
   - Use four kernel functions to classify the dataset.

3. **Model Evaluation**  
   - Evaluate classification accuracy, computation time, and other metrics for each kernel function.

#### Discussion:

- **Discussion 5: How do the four kernels perform on this dataset?**  
   Tip: Visualization is not required, focus on evaluating based on accuracy.

- **Discussion 6: Does SVM require data normalization? What is the impact of normalization on the kernels?**  
   Tip: Try analyzing the effect of data normalization on the four kernel functions, comparing classification accuracy, computation time, and other relevant aspects.

### [4. Implement an SMO-Based Linear SVM Classifier to Perform Binary Classification on the Iris Dataset](ML3_4.ipynb)

The SVM.py file contains the implementation of an SVM classifier based on the SMO algorithm.

#### Details:

1. **Select two features and two classes of data for binary classification**  
   - Note: The binary labels are 1 and -1.

2. **Split the Data**  
   - Divide the data into training and testing sets.

3. **Data Normalization**  
   - Perform data normalization.

4. **Train the Model**  
   - Refer to the program template `SVM_numpy_template.py`.

5. **Output**  
   - Output the optimal solution for the SVM dual problem (α), parameters and intercept of the decision function, support vectors, and other relevant details.

6. **Visualization**  
   - Visualize the training data samples using a scatter plot, draw the decision boundary and two maximum margin boundaries, and mark the support vectors (including those on the margin and inside the margin) to help verify the correctness of the algorithm.

7. **Prediction on the Test Set and Model Evaluation**  
   - Make predictions on the test set and evaluate the model performance.

#### Discussion:

- **Discussion 7: Based on the experimental results, describe the relationship between the C parameter in soft margin SVM, the Lagrange multipliers (α), support vectors, and the optimal decision boundary and margin regions.**
