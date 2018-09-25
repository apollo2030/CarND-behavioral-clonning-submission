import numpy as np
from scipy import ndimage
import random

import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 # unidimensional vector of image absolute paths
                 # arranged in the same order as the labels 
                 list_images, labels, batch_size=32, dim=(160,320), n_channels=3, n_classes=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_images = list_images 
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int((len(self.list_images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Get the image with the corresponding index in the big set,
        # as the indexes are shuffled we can't count on the order 
        try:
            #list_images_temp = [zip((self.list_images[indexes], indexes) for k in indexes]
            list_images_temp = zip(self.list_images[indexes], indexes)

        # Generate data
            X, y = self.__data_generation(list_images_temp)
            return X, y
        except Exception as e:
            print(e)
            print(self.list_images)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_images))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_images_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        try:
            for i, data_point in enumerate(list_images_temp):
                ## Store sample
                #if(bool(random.getrandbits(1)) == True):
                #    X[i,] = np.flip(ndimage.imread(data_point[0]), 1)
                #    y[i] = -self.labels[data_point[1]]
                #else:
                X[i,] = ndimage.imread(data_point[0])
                y[i] = self.labels[data_point[1]]
        except Exception as e:
            print(e)

        return X, y