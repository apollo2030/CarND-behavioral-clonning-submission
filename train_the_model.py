import pandas as pd
from data_generator import DataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout, ELU, BatchNormalization
from keras.callbacks import TensorBoard
import numpy as np
import sklearn

if __name__ == '__main__': 
    driving = pd.read_csv('C:\CarND\CarND-Behavioral-Cloning-P3\data\driving_log.csv')

    # Parameters
    params = {'dim': (160, 320),
              'batch_size': 128,
              'n_classes': 1,
              'n_channels': 3,
              'shuffle': True}

    # Generators
    #max_samples = 5000
    max_samples = len(driving.center) * 3
    cutting_index = int(max_samples * 0.8)
    steering_angle_correction = 0.22

    steering_angle = np.array(driving.steering_angle.rolling(1, min_periods=1).mean())
    #steering_angle = np.array(driving.steering_angle)

    X_center = np.array(driving.center)
    y_center = np.array(steering_angle)

    X_left = np.array(driving.left)
    y_left = np.array(steering_angle) + steering_angle_correction

    X_right = np.array(driving.right)
    y_right = np.array(steering_angle) - steering_angle_correction

    X = np.concatenate((X_center, X_left, X_right))
    y = np.concatenate((y_center, y_left, y_right))

    X, y = sklearn.utils.shuffle(X, y)

    X_training = X[:cutting_index:] # start at 0 until cutting_index with the default step of 1
    y_training = y[:cutting_index:]

    X_validation = X[cutting_index:max_samples:] # start at cutting_index until end with the default step of 1
    y_validation = y[cutting_index:max_samples:]

    training_generator = DataGenerator(X_training, y_training, **params)
    validation_generator = DataGenerator(X_validation, y_validation, **params)

    # Design model
    model = Sequential()
    model.add(Cropping2D(cropping=((10,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 127.5 - 1.0))
    model.add(Convolution2D(24, 5, strides=(2,2)))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Convolution2D(36, 5, strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(ELU())
    model.add(Convolution2D(48, 5, strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(ELU())
    model.add(Convolution2D(64, 3))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Dropout(0.2))
    model.add(ELU())
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    tesonrboard_callback = TensorBoard(write_images=True)
    # Train model on dataset
  
    model.fit_generator(generator = training_generator,
                        validation_data = validation_generator,
                        use_multiprocessing = True,
                        workers = 4,
                        epochs = 5, callbacks=[tesonrboard_callback])
    model.save('model2.h5')