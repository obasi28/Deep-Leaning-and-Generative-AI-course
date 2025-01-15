# Week 1: Introduction to Deep Learning and Neural Networks

## Day-by-Day Breakdown

---

### Day 1: Introduction to Deep Learning

#### Topics Covered
- Overview of deep learning and its importance
- Key concepts:
  - Neurons
  - Weights, biases
  - Activation functions
- Forward propagation:
  - Loss functions
  - Optimization

#### Hands-On
1. Setting up your environment:
   - Anaconda
   - TensorFlow/Keras
   - PyTorch
2. Implementing a **single-layer perceptron**

#### Resources
- [Deep Learning Book – Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org)
- [TensorFlow Setup: Official Guide](https://www.tensorflow.org/install)

#### Assignment
- Write a Python function for forward propagation in a simple neural network.

---

### Day 2: Activation Functions and Backpropagation

#### Topics Covered
- Common activation functions:
  - ReLU
  - Sigmoid
  - Tanh
  - Softmax
- Derivation and implementation of backpropagation
- Gradient descent and learning rate selection

#### Hands-On
1. Implementing activation functions from scratch
2. Training a single-layer perceptron using backpropagation

#### Resources
- Paper: *[Backpropagation Applied to Handwritten Zip Code Recognition](https://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)* by LeCun
- [Activation Functions Tutorial](https://www.tensorflow.org/guide/keras/activation_functions)

#### Assignment
- Compare the performance of sigmoid and ReLU activation on a toy dataset.

---

### Day 3: Multi-Layer Perceptrons (MLPs)

#### Topics Covered
- Introduction to multi-layer architectures
- Vanishing gradient problem and its mitigation
- Batch processing in neural networks

#### Hands-On
1. Building an MLP with Keras to classify handwritten digits (MNIST dataset)
2. Visualizing weights and activations

#### Resources
- Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Keras MLP Guide](https://keras.io/getting_started/)

#### Assignment
- Add a hidden layer to your MLP and observe performance changes.

---

### Day 4: Optimization Techniques

#### Topics Covered
- Optimization algorithms:
  - SGD
  - Adam
  - RMSProp
- Loss functions:
  - Mean Squared Error (MSE)
  - Cross-Entropy
- Regularization techniques:
  - Dropout
  - Weight Decay

#### Hands-On
1. Experimenting with Adam and SGD on the MNIST dataset
2. Applying dropout to reduce overfitting

#### Resources
- Paper: *[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)* by Kingma and Ba
- [TensorFlow Regularization Guide](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)

#### Assignment
- Tune the learning rate and dropout percentage to optimize model performance.

---

### Day 5: Neural Network Performance Evaluation

#### Topics Covered
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Confusion matrix and ROC-AUC curves
- Saving and loading trained models

#### Hands-On
1. Evaluate your MNIST model using various metrics
2. Save your trained model and reload it for inference

#### Resources
- [Scikit-Learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Saving and Loading Models in TensorFlow](https://www.tensorflow.org/tutorials/keras/save_and_load)

#### Assignment
- Create a Python script that evaluates and saves your model, then reloads it for prediction.

---

## Research Papers for Week 1

1. **"Deep Learning"** by LeCun, Bengio, and Hinton (Nature, 2015)  
   Overview of deep learning's evolution and its potential.  
   [Link: Deep Learning Paper](https://www.nature.com/articles/nature14539)

2. **"Adam: A Method for Stochastic Optimization"** by Kingma and Ba  
   Explanation of the Adam optimizer.  
   [Link: Adam Paper](https://arxiv.org/abs/1412.6980)

3. **"Gradient-Based Learning Applied to Document Recognition"** by LeCun et al.  
   An early application of neural networks for document recognition.  
   [Link: Document Recognition Paper](https://ieeexplore.ieee.org/document/726791)

---

## Mini-Project for Week 1

### Title
Handwritten Digit Classifier

### Objective
Build and train an MLP to classify the MNIST dataset with at least **90% accuracy**.

### Deliverables
- Jupyter Notebook with code and explanations
- Saved model file
- Report including evaluation metrics and insights

### Bonus Challenge
Use different optimization algorithms and compare their performance.
