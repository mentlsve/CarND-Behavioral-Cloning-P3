
# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[udacity-data-set]: ./writeup_images/udacity_data_set.png "Udacity data set"
[generated-data-distribution]: ./writeup_images/data_distribution.png "Data distribution from generator"
[center]: ./writeup_images/center_2016_12_01_13_30_48_287.jpg "Center"
[right]: ./writeup_images/right_2016_12_01_13_39_31_979.jpg "Cropped"
[cropped]: ./writeup_images/center_2016_12_01_13_30_48_287_cropped.jpg "Cropped"
[flipped]: ./writeup_images/right_2016_12_01_13_39_31_979_flipped.jpg "Flipped"
[yuv]: ./writeup_images/yuv.jpg "YUV"
[rgb]: ./writeup_images/rgb.jpg "RGB"
[cropped_portion]: ./writeup_images/cropped_portion.jpg "cropped_portion"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The `model.py` file contains two methods:
* `get_model()` which returns a Keras Sequential model. The model is described in detail in the section about the [appropriate model architecture](#1-an-appropriate-model-architecture-has-been-employed)
* `main()` which 
  * loads the metadata. That is the `./data/driving_log.csv` which contains information about the image names and the corresponding steering values. Note that I removed the header row from `./data/driving_log.csv` to make iterating straight forward.
    ```
    metadata = dg.read_dataset_metadata( "./data/driving_log.csv")
    ```
  * gets the model using `get_model()`
  * sets up the generators. I have one generator for training data and one generator for validation data. They operate on disjunct subsets of the whole dataset.
    ```
    train_lines, val_lines = train_test_split(metadata, test_size=0.2)
    train_gen = dg.generator(train_lines, batch_size=128)
    valid_gen = dg.generator(val_lines, batch_size=128)
    ```
  * configures the learning process 
    ```
    model.compile(optimizer=Adam(1e-3), loss="mse")
    ```
  * fits the model on data generated batch-by-batch 
    ```  
    history = model.fit_generator(train_gen,
                                samples_per_epoch=20480,
                                nb_epoch=6,
                                validation_data=valid_gen,
                                nb_val_samples=4096,
                                verbose=1)
    ```
  * saves the model, so it can be downloaded and used for generating steering angles in the autonomous mode of the simulator 
    ```  
    model.save('model.h5')
    ```
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is heavily inspired by the paper ["End to End Learning for Self-Driving Cars" by Nvidia engineers](https://arxiv.org/pdf/1604.07316.pdf).

The network consists of:
* a `Cropping2D layer` to remove 70 pixels from the top, 25 pixels from the bottom, and 1 pixel from the left and right. After this step 
the images have a shape of 65 x 318 (height x width)
* a `Lambda layer` to normalize the input so that they have zero mean and unit variance
* 5 `convolutional layers`, with `relu` for non-linearities `batch normalization` after every `convolutional layer` and the following characteristics shape of the convolutional layers:

  | |Filter size|Stride| Output shape|
  |-|-----------|------|-------------|
  |Conv1|5x5|x=2, y=2|height=31, width=157, depth=24|
  |Conv2|5x5|x=2, y=2|height=14, width=77, depth=36|
  |Conv3|5x5|x=2, y=2|height=5, width=37, depth=48|
  |Conv4|3x3|x=1, y=1|height=3, width=35, depth=64|
  |Conv5|3x3|x=1, y=1|height=1, width=33, depth=64|

* 4 `fully connected` layers with
  * output shape 1164
  * output shape 100
  * output shape 50
  * output shape 10
* the `output layer`

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after the first 3 fully connected layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

The model uses an `Adam` optimizer with the default parameters and `mean_squared_error` as a loss function (model.py line 84).

```
model.compile(optimizer=Adam(1e-3), loss="mse")
```

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

Architecture and Training Documentation


history = model.fit_generator(train_gen,
                                samples_per_epoch=20480,
                                nb_epoch=6,
                                validation_data=valid_gen,
                                nb_val_samples=4096,
                                verbose=1)

#### Training data

_Answering rubric points:_ 
* _Is the training data chosen appropriately?_
* _Is the creation of the training dataset documented?_

I started with the Udacity training data set and had a slightly different model (for example using `tanh` instead of `relu` activations). Since my car was going off the road I recorded two additional rounds and a lot of recovery data (i.e. going from the side back to the center of the road). However I must have been not as accurate as required when recording the recovery data, so I must have partially recorded going from the center to the side of the road and in the end model did not get better.

So I dumped my training data and focussed on getting it working with only the Udacity training data set. This data set is however very unbalanced and when using it as is, it is likely that the model will have a bias to go straight.

![alt text][udacity-data-set]

Therfore the generator which operates on the Udacity data set does the following things:

* 70% of the data where the absolute value of the steering angle is less than 0.1 is dropped
* with a probability of 50% the image is flipped in left/right direction (and so the sign of the steering angle)

  ![alt text][right]
  ![alt text][flipped]
* randomly one of the three camera images is taking when processing a line of the driving_log.csv and the steering angle is adjusted with a value of +0.25 for an image from the left camera and -0.25 for an image from the right camera.


Therefore the data set generated by `data_generator.py` yields a more balanced data set:
![generated-data-distribution][generated-data-distribution]

In the generator the color space is also changed from RGB to YUV. This has been reported to be very helpful by other students and is also mentioned in the original paper for the Nvidia architecture. 

![alt text][rgb]
![alt text][yuv]

In the above visualisation the two chrominance components U and V seem to separate on-track and off-track quite good, so I would guess this is the reason why this is a more suitable color space to train on.

This above steps happen in the generator and outside the model. In the model itself the image additionally gets normalized and cropped. The cropped portion is then fed to the first convolutional layer (after normalization)
![cropped_portion][cropped_portion]

#### Training process

_Answering rubric points:_ 
* _Is the training process documented?_

```
20480/20480 [==============================] - 56s - loss: 0.4012 - val_loss: 0.0758
Epoch 2/6
20480/20480 [==============================] - 54s - loss: 0.0759 - val_loss: 0.0670
Epoch 3/6
20480/20480 [==============================] - 54s - loss: 0.0607 - val_loss: 0.0461
Epoch 4/6
20480/20480 [==============================] - 54s - loss: 0.0499 - val_loss: 0.0373
Epoch 5/6
20480/20480 [==============================] - 54s - loss: 0.0448 - val_loss: 0.0318
Epoch 6/6
20480/20480 [==============================] - 53s - loss: 0.0442 - val_loss: 0.0297
```

### Simulation

_Answering rubric points:_ 
* _Is the car able to navigate correctly on test data?_

The video below shows how the car is driving in autonomous mode based on steering angles predicted by the model.

[![](http://img.youtube.com/vi/4zrLci3FdoQ/0.jpg)](http://www.youtube.com/watch?v=4zrLci3FdoQ "SDC Nanodegree - Project 3 Behavorial Cloning Sumbission")

In the upper left corner the `video.mp4` is embedded which has been generated as part of this run using

```
python drive.py model.h5 video
python video.py video --fps 48
```
