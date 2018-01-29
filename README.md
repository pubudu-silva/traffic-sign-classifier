# Traffic Sign Classifier
In this project I am going to classify traffic signs found in the ['German Traffic Sign Dataset'](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) using a CNN. The ipython notebook included in the repository contains the step-by-step approach followed in this project. I am using Tensorflow as the deep learning framework to implement the CNN. Numpy and Pandas libraries are also utilized. 

## Exploring the data set
The first step, like in any machine learning project, is to explore the dataset. Training, validation and testing data are found as seperate pickle files named, 'train.p', 'valid.p', and 'test.p' respectively. You can simply load the dataset using the pickle library and explore it using basic python and numpy methods to make following observation.
*	The size of training set: 34799 samples
*	The size of the validation set: 4410 samples
*	The size of test set is: 12630 samples
*	The shape of a traffic sign image: 32x32x3
*	The number of unique classes/labels in the data set: 43

It is very important to visualize the dataset, at least in a highlevel, before we start to solve the classification problem. The insights we get from even basic visualisation help us drive our approach to the solution. One important visualization, especially in a classification task, is to explore the distribution of samples across all possible classes. As clearly seen in the drown histograms (found in the notebook) for training, validation and testing sets, the samples are unevenly distributed across 43 classes - with majority of classes only having couple of hundred samples while others having couple of thousands. This is a clear warning that we have to augment the training set to fill those sparse classes so every class has at least a minimum number of samples, in order to get a significant accuracy in our classification task. However to prove the utility of such augmentation I first trained a network with the training data as is and measure the accuracy, and then augment it and show the improvement. I will be using a simple LeNet inspired CNN with 2 convolutional layers and 3 fully connected layers for both cases. 

## Classifying without Pre-processing and Data Augmentation
In the first attempt the only pre-processing I did was is to normalize the image using the basic arithmatics, i.e. (image-128)/128. Which consistently gave me a training accuracy in 90s, validation accuracy in 80s and a testing accuracy in 70s.

## Pre-processing
While there can be many effective ways of pre-processing images to enhance the classification accuracy, I have found in my experience Contrast Limited Adaptive Histogram Eqalization (CLAHE) to be very effective in image classification tasks. Since the contrast is meaningful only in the intensity channel I preprocess images as follows. Convert the color space from RGB to YUV. Apply CLAHE only to the Y channel and convert the image back to the RGB space.

## Data Augmentation
As noted earlier we have to augment the dataset such that each class has at least a minimum number of samples. The method of determining what that minimum number is, is partly a trial and error process and partly influenced by experience. Other than roughly equalizing the class distribution, augmentation generally plays a very important role of adding variations (once that is hard to capture from a small dataset) to the training set. As observed in the data exploration stage the least frequent classes had only 180 samples while the  most frequent one had 2010 samples. The total training set is around 35K. Since we are going to be training a CNN from scratch, as opposed to transfer learning, we are going to need more than 35K samples train the CNN successfully. I determined that minimum number to be 5000,to make the final training set 43x5000 ~= 215,000 samples. I am using typical image augmentation techniques such as shifting the image along width and height, zooming, rotating, and sheering to generate new images from each original image. Keras image processing library offers an easy way to implement such commonly used augmentations via its ImageDataGenerator class.   

**Note**: Since we are dealing with traffic signs, these images may be sensitve to horizontal and vertical flips. For example a left turn signal become a right turn signal if it is flipped horizontally. Hence it is important not to use flips in augmentation, although they are commonly use in other classification tasks.

**Note**: While augmentation adds necessary variation to the dataset, it is important to keep the variation limits in generating imagation in augmentation to moderate limits. Again how much pricisely is learned with trial and error and experience. If you go overboard with variations, the final accuracy will suffer.

**Note**: This whole augmentation process will take 15-30 minutes to finish depending on the power of your system. As a practical measure, it is always adviced to save the final augmented dataset in a pickle before moving forward

## Training, Validation and Testing with the CNN
Now that we have pre-processed and augmented the dataset, let's go ahead and train our model. Following is the overview of the simple CNN model I am using for this project.

Layer    | Description
-------- | -----------
Input | 32x32x3 RGB image
5x5 Convolution | 1x1 stride, no padding; 6 filters; output 28x28x6
RELU |       
Max Pooling | 2x2 kernal and stride, no padding; output 14x14x6
Drop-out |    
5x5 Convolution | 1x1 stride, no padding; 16 filters; output 10x10x16
RELU |     
Max Pooling | 2x2 kernal and stride, no padding; output 5x5x16
Drop-out |     
Fully Connected | input 400; output 120
RELU |     
Drop-out |    
Fully Connected | input 120; output 84
RELU |    
Drop-out |    
Fully Connected | input 84; output 43
Output | 43x1 vector


