
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
### Submitted files

_Answering rubric points:_ 
* _Are all required files submitted?_

My project includes the following files:
* `model.py` containing the script to create and train the model
* `data_generator.py` containing the gemerator and performing data augmentation
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `video.mp4` video recording of autonomous mode 
* `https://youtu.be/4zrLci3FdoQ` screen recording of autonomous mode 
* `writeup_report.md` (this file) summarizing the results

### Simulation

_Answering rubric points:_ 
* _Is the code functional?_
* _Is the car able to navigate correctly on test data?_

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The video below shows how the car is driving in autonomous mode based on steering angles predicted by the model.

[![](http://img.youtube.com/vi/4zrLci3FdoQ/0.jpg)](http://www.youtube.com/watch?v=4zrLci3FdoQ "SDC Nanodegree - Project 3 Behavorial Cloning Sumbission")

In the upper left corner the `video.mp4` is embedded which has been generated as part of this run using

```
python drive.py model.h5 video
python video.py video --fps 48
```

### Code Organization

_Answering rubric points:_ 
* _Is the code usable and readable?_

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

_Answering rubric points:_ 
* _Has an appropriate model architecture been employed for the task?_
* _Is the model architecture documented?_
* _Is the solution design documented?_

My model is heavily inspired by the paper ["End to End Learning for Self-Driving Cars" by Nvidia engineers](https://arxiv.org/pdf/1604.07316.pdf).

The network consists of:
* a `Cropping2D layer` to remove 70 pixels from the top, 25 pixels from the bottom, and 1 pixel from the left and right. After this step 
the images have a shape of 65 x 318 (height x width)
* a `Lambda layer` to normalize the input so that they have zero mean and unit variance
* 5 `convolutional layers` followd by `relu` for non-linearity and `batch normalization`. The convolutional layers have the following properties.

  | |Filter size|Stride| Output shape|
  |-|-----------|------|-------------|
  |Conv1|5x5|x=2, y=2|height=31, width=157, depth=24|
  |Conv2|5x5|x=2, y=2|height=14, width=77, depth=36|
  |Conv3|5x5|x=2, y=2|height=5, width=37, depth=48|
  |Conv4|3x3|x=1, y=1|height=3, width=35, depth=64|
  |Conv5|3x3|x=1, y=1|height=1, width=33, depth=64|
* 4 `fully connected` layers with
  * output shape 1164 (followed by dropout to reduce overfitting)
  * output shape 100 (followed by dropout to reduce overfitting)
  * output shape 50 (followed by dropout to reduce overfitting)
  * output shape 10
* the `output layer`

#### Attempts to reduce overfitting in the model

_Answering rubric points:_ 
* _Has an attempt been made to reduce overfitting of the model?_

The model contains dropout layers after the first 3 fully connected layers. The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Batch normalization also seems to be helpful to reduce overfitting as stated in [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf), although it is not its primary benefit (which is speedup of training):
_"When training with Batch Normalization, a training example is seen in conjunction with other examples in the mini-batch, and the training network no longer producing deterministic values for a given training example. In our experiments, we found this effect to be advantageous to the generalization of the network."_

#### Hyperparamter tuning

_Answering rubric points:_ 
* _Have the model parameters been tuned appropriately?_

The model uses an `Adam` optimizer with the default parameters and `mean_squared_error` as a loss function (model.py line 84). 

```
model.compile(optimizer=Adam(1e-3), loss="mse")
```

The `Adam` optimizer adaptes the learning rate automatically, so there is no need to tune this parameter. 

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

The model is trained on data generated batch-by-batch by a Python generator. In every epoch the model is fitted on 20480 samples and validated on 4096 samples (1/5 of the training set).

``` 
history = model.fit_generator(train_gen,
                                samples_per_epoch=20480,
                                nb_epoch=6,
                                validation_data=valid_gen,
                                nb_val_samples=4096,
                                verbose=1)
``` 

I started with 4 epochs and then tried my model in the simulator. Because it did not work good enough I increased the number of epochs and with 6 epochs no tire is leaving the drivable portion of the track surface anymore.

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


