# Traffic Sign Classifier
In this project I am going to classify traffic signs found in the ['German Traffic Sign Dataset'](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) using a CNN. The ipython notebook included in the repository contains the step-by-step approach followed in this project. I am using Tensorflow as the deep learning framework to implement the CNN. Numpy and Pandas libraries are also utilized. 

## Exploring the data set
The first step, like in any machine learning project, is to explore the dataset. Training, validation and testing data are loaded as seperate pickle files named, 'train.p', 'valid.p', and 'test.p' respectively (I am uploading these 3 pickle files to this project as a set of compressed tar.gz files split in to 25MB size blocks to workaround github file size limitations) . You can simply load the dataset using the pickle library and explore it using basic python and numpy methods to make following observation.
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

With the above model I was able to acheive following accuracies for three datasets with the pre-processed and augmented training data.
* Training accuracy - 97.9%
* Validation accuracy - 96.7%
* Test accuracy - 94.7%

## Iterative Approach Followed in Getting to the Final Solution
As it is the case with many problems, once you know the solution it is easy to implement. The same is true for this project, but my journey getting to this final solution was iterative as usually it is the case with machine learning in general.

I started with the LeNet lab solution, hence the LeNet architecture. Just using the architecture as is and only modifying the first convolution layer to convolve color images and the last layer to have 43 logits, I could get a validation accuracy about 89%. However the model was seen to be over fitting as the training accuracy was over 99%. 

I tried a list of normalization techniques described above. I experimented with different weights and bias initializing techniques in order to get out of any local optimums the model could be trapped in. Several optimizers were tried, momentum was introduced to the simple stochastic gradient decent. I played with different learning rate adapting techniques, as the model displayed oscillatory behavior half way down the epochs. That got me over 92% of validation accuracy but not above 93%.

Then I tried increasing number of convolution filters in both convolution layers and number of neurons in fully connected layers. I added several convolution and fully connected layers. I increased both depth and breadth of the model in several ways to upgrade the original LeNet, which was designed for 10 handwritten digits classification, to a sufficient complexity to perform relatively complex 43 traffic sign classification. However additional features obviously didn’t help with the over-fitting issues, in fact it got worsened (it makes sense now why it did).

In order to address the over-fitting issue I introduced drop-out layers both between convolution layers and fully connected layers. Tweaking drop-out probability helped a little in getting the validation accuracy closer to 93% but couldn’t get it beyond that.
A typical way to address over fitting, after regularization (drop-out in this case) fails is to get more training data. As I explained in reasons for augmentation above, training/validation/testing sets were unbalanced with different distributions. Therefore in order to make the training set balance and increase the number of training samples I tried simply filling each class up to 2000 samples by simply copying original samples in that class in iterations until it reach 2000 samples. This didn’t provide any significant accuracy gains, as the training set just got filled by redundant copies of same instanced under this approach. Then I tried to balance the data set, but in slightly sophisticated approach: by making different versions of the original instances by randomly tweaking in offsets, rotations etc. (explained in the augmentation section). I tried making each class 1000 samples, 2000 samples, 3000 samples, 4000 samples and finally 5000 samples. The best results were reached with 5000 samples and going further than that didn’t add significant gains. I tried with lot of variations with higher intensities. When I try to make drastically different versions of original instances, I got rid of over-fitting but started getting slight under-fitting effects, as the training accuracy felt significantly. After experimenting with various settings I settled down to the current set of operations with current intensities (I probably could get even better accuracies if I played with this for some more time). As you can see in the epochs with data augmentation I have almost completely addressed the over-fitting and under-fitting issues. Therefore I didn’t need to use drop-out layers so I trained with drop-out probability of 1.0
I could have converged to a better validation and training accuracy if included early stop conditions as I continue to train the model over 
