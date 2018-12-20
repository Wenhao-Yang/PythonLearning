## Chapter 3: Linear Regression
* Using the **matrix inverse method**  

When ***Ax=b***,the solution to solving ***x*** can be expressed as ***x=(A<sup>T</sup>A)<sup>-1</sup>A<sup>T</sup>b***.  

The plot image that the linear regression produces will be:
![MatrixInverseMethod](../image/MatrixInverseMethod.png)   

* Imlementing a **Decomposition Method**  

We implement a matrix decomposition method for linear method. Implementing inverse methods in the previous recipe can be numerically inefficient in most cases, especially when the matrices get cery large. Another approach is to use the **Cholesky decomposition method**. The Cholesky decomposition decomposes a mtrix into a lower and upper triangular matrix, say ***L*** and ***L'*** , such that these matrices are transposition of each other. Here we solve the system, ***Ax=b***, by writing it as ***LL'x=b***. We will first solve ***Ly=b*** and then solve ***L'x=y*** to arrive at our coefficient matrix ,***x***.  

And the following image shows the result.  

![CholeskyDecompositionMethod](../image/CholeskyDecompositionMethod.png)  

* The TensorFlow Way of Linear Regression  

Loop through batches of data points and let TensorFlow update the slopes and y-intercept. We will find an optional line through data points where the x-value is the petal width and the y-value is the sepal length. In the end, we plot the best fit line and data points image. And we plot the L2 loss vs generation image.  

Here are the images.  

![SepalLength&PedalWidth](../image/SepalLength&PedalWidth.png)  

![L2Loss&Generation](../image/L2Loss&Generation.png)  

* Understanding Loss Function in Linear Regression  

We use the same irirs dataset as in the prior recipe, but we will change our loss functions and learning rates to see how convergence changes.

When the learning rate is 0.1, the loss values will change with following trend:  

![0.1rate&LossFunctionL1&L2](../image/0.1rate&LossFunctionL1&L2.png)  

However, when the learning rate is 0.4, the model will not converge:   

![0.4rate&LossFunctionL1&L2](../image/0.4rate&LossFunctionL1&L2.png)   

* Implement Deming regression

In the recipe, we will implement Deming regression, which means we will need a different way to measure the distance between the model line and data points.  

Here are the best fit line and data points:  

![DemingRegression](../image/DemingRegression.png)  

And the following is the loss values during the training:  

![DemingLossValue](../image/DemingLossValue.png)  

* Implement Lasso and Ridge Regression  

There are also ways to limit the influence of coefficients on the regression output. These methods are called regularization methods and two of the most common regularization mathods are lasso and ridge regression. We add regularization terms to limit the slopes in the formula.

For lasso regression, we must add a term that greatly increases our loss function if the slope, A, gets above a certain value. We use a continuous approximation to a step function, called the continuous heavy step function, that is scaled up and over to the regularization cut off we choose.  

Here are the results of fit line and Loss value during the training:  

![LassoRegressionDataPoints](../image/LassoRegressionDataPoints.png)  

![LassoLossValue](../image/LassoLossValue.png)  

For ridge regression, we just ass a term to the L2 norm, which is the scaled L2 norm of the slope coefficient.
