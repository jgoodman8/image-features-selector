# Images Features Selection Study

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
  <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png"/>
</a>

## Table of contents

1. [Introduction](#intro)
2. [What this project contains](#content)
3. [Technologies](#tech)

## <a name="intro"/>1. Introduction</a>

&nbsp;&nbsp;&nbsp;&nbsp;Digital images have completely changed the way people acquire, store and/or share images, 
resulting in large amounts of visual information. In order to take advantage of this information, computer vision tasks 
include methods to process and analyze digital images.

&nbsp;&nbsp;&nbsp;&nbsp;For the training and improvement of these artificial vision methods, a data set Reference is 
ImageNet, which includes more than 10 million images that belong to more than 20 thousand classes. Datasets of images 
like this have grown not just in the number of samples, but so in the number of features that describe them. At this 
point, it could be reasonable to expect that having more features would provide more information and better results. 
However, this does not happen, due to the so-called "curse of dimensionality".

&nbsp;&nbsp;&nbsp;&nbsp;In this context, the feature selection (FS) contributes to the scalability of the machine 
learning algorithms, by reducing the number of features, finding the most relevant properties of the images and 
decreasing their execution times. For this tasks, there is a wide variety of methods that, in their origin, were 
designed to process small data sets and their efficiency is drastically diminished in such dimensions. In such a way, 
its application on Big Data technologies, may allow to employ larger datasets. This increase in the size of the 
training sets is desirable to increase the accuracy of the algorithms, but their pre-processing and training time must 
be manageable; therefore, the inclusion of such technologies is necessary.

&nbsp;&nbsp;&nbsp;&nbsp;In the other hand, convolutional neural networks (_CNNs_) apply convolutions, among other 
operations, at different scales in order to extract image features, both global and local and of different complexity. 
Although this is not usual in all classical vision techniques, other examples can be found, such as wavelet 
transformations. Therefore, it may be interesting to observe the results obtained when FS techniques are applied to 
the same sets of images at different scales.

&nbsp;&nbsp;&nbsp;&nbsp;Additionally, it is worth pay attention to the deep learning algorithms, which are normally used 
for the extraction of relevant features. By removing the last layer of a CNN, it is possible to network the output as a 
vector of features (_deep features_). Based on this idea, there are several FS hybrid techniques that try to get the 
salient features. This type of approach is at the head of the state of the art, when dealing with artificial vision 
problems. Therefore, it can be very interesting to make a comparison between the results obtained with the methods of 
feature selection traditional and hybrid, based on neural networks.

## <a name="content"/>2. What does this project contain</a>

This project will try to solve the problems described at the top. For it, it will focus in:

- Find out the relevance of the different images features and the relevance of the scale at which they are extracted.
- Apply such feature selection methods, by using large sets of images, on Big Data platforms.
- Compare the results when applying deep learning algorithms at the feature selection.

## <a name="tech"/>3. Technologies</a>

- Dataset: [Tiny Imagenet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) 
- Scala 2.11.8
- Spark 2.3.1
- Additional Libraries: 
    - [Spark Deep Learning](https://github.com/databricks/spark-deep-learning)

