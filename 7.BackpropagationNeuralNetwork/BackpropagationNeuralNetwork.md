## VII. BP Neural Networks

### [1. Classify the Wine Dataset Using MLPClassifier from scikit-learn and Visualize the Effects of Hyperparameters on Model Complexity Through Features and Decision Boundaries](ML7_1.ipynb)

#### Specific Content:

1. **Select the first two features and build a multilayer perceptron (MLP) for multi-class classification.**

2. **Visualization**  
   - Use a scatter plot to visualize the data samples (the two selected features), and plot the decision boundary obtained after training the model.

#### Discussion:

- **Discussion 1: Change the number of neurons in the single hidden layer (e.g., 10, 100 neurons), keeping other parameters constant, and observe its impact on the decision boundary.**

- **Discussion 2: Change the depth of the neural network (e.g., depth of 2 layers, 10 neurons per layer), keeping other parameters constant, and compare with Discussion 1 to observe the effect of network depth on the decision boundary.**

- **Discussion 3: Based on Discussion 1 (or 2), change the activation function (e.g., `tanh`, `relu`), and compare with Discussion 1 (or 2) to observe the impact of different activation functions on the decision boundary.**

- **Discussion 4: Based on Discussion 3, increase the regularization coefficient and observe its effect on the decision boundary.**

#### Summary:

- Summarize the effects of the number of neurons in the hidden layer, the number of hidden layers, the activation function, and the regularization coefficient on the model complexity.

---

### [2. Classify the Built-in Handwritten Digit Dataset Using MLPClassifier from scikit-learn](ML7_2.ipynb)

#### Specific Requirements:

1. **Load the Dataset**  
   - The Handwritten Digit dataset is a built-in dataset from scikit-learn. It is a 3D array of shape `(1797, 8, 8)`, containing 1797 handwritten digits, each represented by an 8×8 pixel matrix. Each matrix element is an integer between 0 and 16. The classification labels are digits from 0 to 9.

2. **Model Building**  
   - Use `MLPClassifier` to build a classification model.

3. **Output**  
   - Output the classification accuracy.

#### Discussion:

- **Discussion 5: Relate model complexity to the model's generalization error, tune the hyperparameters, and improve model generalization performance.**  
   - Try adjusting hyperparameters such as the number of neurons in the hidden layers, the number of hidden layers, activation functions, learning rate, and regularization coefficients.

---

### [3. Implement a BPNN Algorithm and Perform Classification on the Iris Dataset or Handwritten Digit Dataset (Choose One)](ML7_3.ipynb)

The following files contain the implementation of the BPNN algorithm:
- `Autograd.py`: Implements an automatic differentiation system. It uses the `Value` class to store values and gradients, and supports common mathematical operations (e.g., addition, multiplication, exponentiation, logarithms) as well as backward propagation to compute gradients.
- `Microtorch.py`: Implements the basic framework for a neural network module, including the `Module` class and its subclasses `Neuron` and `Layer`. This framework is used to build and train neural networks, and includes functions for gradient resetting, parameter retrieval, and structure description.
- `MLP.py`: Implements a multilayer perceptron (MLP) neural network model, including an input layer, multiple hidden layers, and an output layer. It supports forward propagation to compute outputs, gradient updates, and structure descriptions.

#### Specific Requirements:

1. **Label Processing**  
   - For binary classification: Positive class as 1, negative class as 0.  
   - For multi-class classification: Convert the sample labels to one-hot vectors.

2. **Build a Shallow Neural Network**  
   - 1-2 hidden layers. The number of neurons in each layer is to be selected freely.

3. **Activation Function**  
   - Choose from `relu`, `sigmoid`, or `tanh`.  
   - Loss function: Choose from cross-entropy loss or mean squared error.

4. **Output**  
   - Output the classification accuracy.

5. **Visualization**  
   - Plot the cost function curve over iterations.

6. **Try Implementing the Chain Rule for Backpropagation in BP Networks.**

#### Notes:

- **Improve Computational Efficiency**  
   - Use vectorization techniques in the algorithm to avoid using `for` loops to iterate through samples and neurons.

- **Weight and Bias Calculations**  
   - Typically, weights `w` and biases `b` are computed separately.

- **Weight Initialization**  
   - To avoid gradient vanishing or exploding, it is common to initialize weights randomly rather than setting them all to 0 or 1.  
     Example: To generate random numbers from the distribution `𝒩(0, √(2/(𝑛𝑖𝑛+𝑛𝑜𝑢𝑡)))`, you can use:  
     ```python
     np.random.randn(m, n) * np.sqrt(2 / (nin + nout))
     ```
     where `nin` is the number of input connections to a neuron, `nout` is the number of output connections, and `m` and `n` are the dimensions of the resulting array.

- **Watch Out for Array Dimensions**  
   - Be careful with array dimensions during vectorized operations to avoid bugs. Do not use one-dimensional arrays (e.g., `np.random.randn(5)`) as these are neither column vectors nor row vectors.  
     To create column or row vectors, you can use examples like `np.random.randn(5, 1)` or `np.random.randn(1, 5)`, or reshape arrays as needed.
