## Chapter 8 Convolutional Neural Network

* Implement A Simpler CNN

Here a four-layer convolutionql neural network will be developed to improve upon the accuracy in predicting the MNIST digits.

This is the softmax loss values during the training:

![SimpleCNNmnistSoftmaxLoss](Image/SimpleCNNmnistSoftmaxLoss.png)

This is the Test Accuracy on MNIST per Generation:

![SimpleCNNmnistTrainTestAcc](Image/SimpleCNNmnistTrainTestAcc.png) 

This is the last batch for model to predict:

![SimpleCNNmnistTrainPre](Image/SimpleCNNmnistTrainPre.png)

* Implement an Advanced CNN

In the recipe, a more advanced method of reading image data and a larger CNN to do image recognition on the CIFAR10 dataset will be implemented. The dataset has 60000 32*32 images that fall into exactly one of tem possible classes. The potential classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

Most of images datasets will be too large to fit into memory, so we can do with Tensorflow and set up an image pipeline to read in a batch at a time from a file. We set up an image reader and then create a batch queue taht operates on the image reader.

This is the loss values during the training:

![ImplementAdvancedCNNLoss](Image/ImplementAdvancedCNNLoss.png)

This is the Accuracy of test set:

![ImplementAdvancedCNNTestAcc](Image/ImplementAdvancedCNNTestAcc.png)

* Retraining Exist CNNs models

In this recipe, we will show how to use a pre-trained TensorFlow image recognition model and fine-tune it to work on a different set of images.

* Applying Stylenet /Neural-Style

Stylenet is a procedure that attempts to learn an image style from one picture and apply it to a second picture while keeping the second image structure.

However, the output for the Stylenet is not ideal as the example mentioned in the book. As I change the optimizer or the weight of different layers, it didn't change a lot. Here is a output image during the procedure.

![temp_output_74](8.4_ApplyingStylenet/temp_output_74.jpg)

* Implementing DeepDream

As some of the intermediate nodes of trained CNN detect features of labels, we can find ways to transform any image to reflect those node features of any nodes we choose. In this recipe, we will go through the DeepDream tutorial on TensorFlow's website, which includes preparing reader to use the DeepDream algorithm for exploration of CNNs and features created in such CNNs.

Here are the results for applying DeepDream algorithm on the book_cover.jpg under the different feature numbers.

![DeepDream30](Image/DeepDream30.png)
 Feature number 30
 
 ![DeepDream30](Image/DeepDream50.png)
 Feature number 50
 
 ![DeepDream30](Image/DeepDream100.png)
 Feature number 100
 
 ![DeepDream30](Image/DeepDream110.png)
 Feature number 110
 
 ![DeepDream30](Image/DeepDream139.png)
 Feature number 139
 
 