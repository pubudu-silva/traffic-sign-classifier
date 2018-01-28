# Traffic Sign Classifier
In this project I am going to classify traffic signs found in the ['German Traffic Sign Dataset'](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) using a CNN. The ipython notebook included in the repository contains the step-by-step approach followed in this project. I am using Tensorflow as the deep learning framework to implement the CNN. Numpy and Pandas libraries are also utilized. 

## Exploring the data set
The first step, like in any machine learning project, is to explore the dataset. Training, validation and testing data are found as seperate pickle files named, 'train.p', 'valid.p', and 'test.p' respectively. You can simply load the dataset using the pickle library and explore it using basic python and numpy methods to make following observation.
*	The size of training set: 34799 samples
*	The size of the validation set: 4410 samples
*	The size of test set is: 12630 samples
*	The shape of a traffic sign image: 32x32x3
*	The number of unique classes/labels in the data set: 43


