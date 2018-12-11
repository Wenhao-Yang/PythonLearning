# TensorflowML
All examples of python are from 《**TensorFlow Machine Learning Cookbook**》.
## Chapter 1: Getting Started with TensorFlow

* Declaring Tensors
* Using Placeholders and Variables
* Working with Matrices
* Implementing Activation Functions

## Chapter 2: The TensorFlow Way 

* Operating in a Computational Graph
* Layering Nested Operations
* Working with Multiple Layers
* Implementing **Loss Functions**
* Implementing **Back Propagation**
* Working with **Batch and Stochastic Training**  
Extend the prior regression example using stochastic training to batch training. Stocahstic training is only putting through one randomly sampled data-target pair at a time. Batch training is putting a large protion of the training examples in at a time and average the loss for the gradient calculation.In stochastic training, randomness may help move out of local minimums. But it needs more iterations to converge. However, batch training finds minimums quicker and takes more resources to compute.  

Here's the loss vaules during the two training process.  
![Stochastic&Batch](image/Stochastic&Batch.png)   
* Combining Everything Together  

We create a classifier on the iris dataset by combining everything together. It's a binary classifier to predict whether a flower is the species Iris setosa or not. Here we define the linear model. The model will take the form ***x2=x1\*A+b***. And if we want to find points above or below that line, we see whether they are above or below zore when plugged into the equation ***x2-x1\*A-b***.  

Here is the figure for the problem.  
![Setosa&Non-setosa](image/Setosa&Non-setosa.png)
* **Evaluating Models**  
Here is that we visualize the model and data with two separate histograms using matplotlib:
![BinaryClassificationEM](image/BinaryClassificationEM.png) 

## Chapter 3: Linear Regression
* Using the **matrix inverse method**  

When ***Ax=b***,the solution to solving ***x*** can be expressed as ***x=(A<sup>T</sup>A)<sup>-1</sup>A<sup>T</sup>b***.  

The plot image that the linear regression produces will be:
![MatrixInverseMethod](image/MatrixInverseMethod.png)   

* Imlementing a **Decomposition Method**  

We implement a matrix decomposition method for linear method. Implementing inverse methods in the previous recipe can be numerically inefficient in most cases, especially when the matrices get cery large. Another approach is to use the **Cholesky decomposition method**. The Cholesky decomposition decomposes a mtrix into a lower and upper triangular matrix, say ***L*** and ***L'*** , such that these matrices are transposition of each other. Here we solve the system, ***Ax=b***, by writing it as ***LL'x=b***. We will first solve ***Ly=b*** and then solve ***L'x=y*** to arrive at our coefficient matrix ,***x***.  

And the following image shows the result.  

![CholeskyDecompositionMethod](image/CholeskyDecompositionMethod.png)



