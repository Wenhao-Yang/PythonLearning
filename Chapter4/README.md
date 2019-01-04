## Chapter 4: Support Vector Machines
* Working with a Linear SVM  

The basic idea is to find a linear separating line(or hyperplane) between the two classes. Here we create a linear separable binary data set for predicting if a flower is I.setosa or not.  

Here are the results of the classification.  

![IrisLinearSVM](Image/IrisLinearSVM.png)  

![IrisLinearSVMAccuracy](Image/IrisLinearSVMAccuracy.png)   
 
![](Image/IrisLinearSVMLoss.png)

* Reduction to Linear Regression

We use the concept that maximizing the margin that contains the most (x,y) points to fit a line between sepal lenght and petal width in Iris dataset. The corresponding loss function will be similar to max(0, |yi - (Axi + b)|-ε). Here, ε is half of the width of the margin, which makes the loss equal to zero if a point lies in this region.

The best fit line and loss value during the iteration are showing in the following figures.  

![CReductionLinearRegression](Image/ReductionLinearRegression.png)  

![Image/ReductionLinearRegressionLoss](Image/ReductionLinearRegressionLoss.png)

* Working With Kernels 

The prior SVMs worked with linear separable data. If we would like to separate non-linear data, we can change how we project the linear separator onto the data. This is done by changing the kernel in the SVM loss function. And we implement the Gaussian kernel here.  

For the parameter set in the recipe, I find out that the iteration times might be a little bit small. So I increase the iteration from 500 to 1500. And here are the changes with different parameters.  

![WorkingWithKernel500Iteration](Image/WorkingWithKernel500Iteration.png)  

![WorkingWithKernel500Iteration](Image/WorkingWithKernel1000Iteration.png)  

![WorkingWithKernel500Iteration](Image/WorkingWithKernel1500Iteration.png)

