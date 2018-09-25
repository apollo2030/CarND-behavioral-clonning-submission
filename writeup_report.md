# **Behavioral Cloning** 

---

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train_the_model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The train_the_model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 36 (train_the_model.py lines 52-76) 

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

Adding BatchNormalization layers helped reduce the wobbling of the driving model.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving in opposite direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a model that is well know to perform well on this kind of tasks.

My first step was to use a convolution neural network model similar to the NVIDIA model I thought this model might be appropriate because it solves a similar problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

To combat the over-fitting, I modified the model so that it includes some dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I recorded some segments where I recove from comming too close to the edge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

|layer (type)                | output shape          |    param #   |
|:--------------------------:|:---------------------:|:------------:|
|cropping2d_1 (Cropping2D)   | (None, 130, 320, 3)   |    0         |
|lambda_1 (Lambda)           | (None, 130, 320, 3)   |    0         |
|conv2d_1 (Conv2D)           | (None, 63, 158, 24)   |    1824      |
|dropout_1 (Dropout)         | (None, 63, 158, 24)   |    0         |
|elu_1 (ELU)                 | (None, 63, 158, 24)   |    0         |
|conv2d_2 (Conv2D)           | (None, 30, 77, 36)    |    21636     |
|batch_normalization_1 (Batch| (None, 30, 77, 36)    |    144       |
|dropout_2 (Dropout)         | (None, 30, 77, 36)    |    0         |
|elu_2 (ELU)                 | (None, 30, 77, 36)    |    0         |
|conv2d_3 (Conv2D)           | (None, 13, 37, 48)    |    43248     |
|batch_normalization_2 (Batch| (None, 13, 37, 48)    |    192       |
|dropout_3 (Dropout)         | (None, 13, 37, 48)    |    0         |
|elu_3 (ELU)                 | (None, 13, 37, 48)    |    0         |
|conv2d_4 (Conv2D)           | (None, 11, 35, 64)    |    27712     |
|elu_4 (ELU)                 | (None, 11, 35, 64)    |    0         |
|flatten_1 (Flatten)         | (None, 24640)         |    0         |
|dense_1 (Dense)             | (None, 200)           |    4928200   |
|elu_5 (ELU)                 | (None, 200)           |    0         |
|dense_2 (Dense)             | (None, 50)            |    10050     |
|elu_6 (ELU)                 | (None, 50)            |    0         |
|dense_3 (Dense)             | (None, 10)            |    510       |
|elu_7 (ELU)                 | (None, 10)            |    0         |
|dense_4 (Dense)             | (None, 1)             |    11        |

Total params: 5,033,527
Trainable params: 5,033,359
Non-trainable params: 168

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Then I recorded two more laps of driving in opposite direction.  

Then I repeated this process on track two in order to get more data points.

After the collection process, I had X number of data points. I then pre-processed this data by cropping the top and bottom part of the image. I used the values (10,20) in order to not loose valuable information in case of the second track.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the accuracy not improving after 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

----------------------------
i used the training data for the second track mainly, i added the left images and right images with a large (1.72) correction angle. i trained the model for 5 epochs and got it running the entire track, To be able to iterate faster, i selected randomly 1000 samples and trained preliminary on those.

One of the biggest mistake I made was to use a numpy array of ints for the steering angle tensor 🤦 - I was at the point to remove all my training data. 

Another thing that helped a lot is the shuffling of the data even before getting a sample for training.

I took the training data in low resolution and the fastest settings, and in autonomous mode its the setting I needed to use to be able to see the car driving the track.

I ended up having a model that drove the entire second  track 2 (model7.h5), the trouble is that it fails at the middle of the first track. At this point I have already 30k image which makes retraining a daunting task with a 50% chance of succeeding. I decided to fine tune the model so that it works for the first track too.

To do that, I need to cut the top layer of the model and pass the new training images through it so that I have the weights that I will use to initialize my top part (because for the most part it is doing the right thing).

Unfortunately fine tuning didn't work so I ended up training the model again after adding batch normalization layers.

