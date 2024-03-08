# TensorFlow Template

An artificial neural network is trained on the MNIST dataset in order to classify images of digits.
TensorFlow is applied for this purpose.
For each epoch, the train/test account/loss is written to the TensorFlow for each epoch.
At the end of training, these values are read from the TensorFlow and visualized by a plotting script.

<img src="./plots/AccuracyLoss.png" width="400" height="250">

Personally, I consider the code in this repository as a template for my own upcoming projects in TensorFlow because the train and test steps, including the logging (writing train/test accuracy/loss to the TensorBoard) can be slightly adapted or reused. 
The same applies to the plotting script.
